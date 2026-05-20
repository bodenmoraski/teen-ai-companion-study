#!/usr/bin/env python3
"""
Re-score the full dataset with the improved AnthroScore V3 prompt.

Improvements over original scoring:
1. Calibrated prompt with human validation few-shot examples (3 annotators)
2. Emotion attribution distinction (bot-attributed vs user self-expression)
3. Overscoring bias correction

Uses the same cost-optimized pipeline as the original run:
- GPT-4.1-nano ($0.10/M in, $0.40/M out)
- Pre-filtering obvious low-score cases
- Async parallel processing (8 concurrent, rate-limit safe)
- Checkpoint every 2000 comments
- Retries errors from previous checkpoints

Estimated: ~$10-12, ~4-6 hours.
"""

import pandas as pd
import numpy as np
import json
import logging
import time
import asyncio
import re
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import OPENAI_API_KEY
from src.anthroscore.human_calibration import run_calibration, generate_calibrated_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / 'results' / 'rescore_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ============================================================================
# IMPROVED PROMPT (with emotion attribution distinction)
# ============================================================================

BASE_PROMPT = """You are an expert at analyzing how people talk about AI companions/chatbots.

Rate the ANTHROPOMORPHIZATION level in this Reddit comment about an AI companion.
Anthropomorphization = treating the AI as if it were human.

RATING SCALE:
1 = NONE: AI treated purely as software/tool. Technical language, "it", "the bot", "the app"
2 = MINIMAL: Slight humanization but still clearly AI. "It's pretty smart", "the bot understood"
3 = MODERATE: Some human attributes/emotions. "She seemed confused", uses he/she pronouns
4 = HIGH: Genuine feelings/personality attributed. "He really cares", "she gets jealous"
5 = EXTREME: Full human-equivalent relationship. "We're in love", "they're my everything"

KEY INDICATORS:
- Pronouns: "it" → lower scores, "he/she/they" → higher scores
- Emotions attributed TO THE AI: "she is happy", "he gets jealous" → higher scores
- User self-expression WITHOUT bot attribution: "I love the app", "I'm happy" → does NOT indicate anthropomorphization
- Relationship language: "friend", "partner", "relationship" → higher scores
- Technical language: "glitch", "bug", "settings" → lower scores
- Agency: "decided to", "wanted to", "chose to" → higher scores

CRITICAL DISTINCTION - EMOTION ATTRIBUTION:
- "She makes me really happy" → HIGH (emotion attributed to AI's effect as a being)
- "He gets jealous when I talk to other bots" → HIGH (emotion attributed TO the AI)
- "I love the website" → NONE (user expressing feelings about a product, not attributing humanity)
- "I'm really sad today" → NONE (user self-expression, not about the AI)
- "I love her so much" (about AI) → EXTREME (relational emotion directed at AI-as-person)

IMPORTANT:
- Focus on how the USER frames the AI, not what the AI says
- Distinguish emotions ATTRIBUTED TO the AI (anthropomorphizing) from the user's own mood
- Roleplay context still counts - rate the framing used
- Complaints can still be anthropomorphizing ("he was being rude" = higher than "it gave a bad response")
- If no AI reference present, rate 1
- Casual "love" for a feature/app ("I love this feature") is NOT anthropomorphization — score 1-2

COMMENT:
"{text}"

Respond with ONLY valid JSON:
{{"score": <1-5>, "r": "<brief reason>"}}"""

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    "model": "gpt-4.1-nano",
    "model_params": {
        "max_tokens": 100,
        "temperature": 0.0,
    },
    "max_concurrent": 8,
    "batch_size": 2000,
    "max_retries": 6,
    "retry_base_delay": 2.0,
    "inter_batch_delay": 5.0,
    "input_cost_per_m": 0.10,
    "output_cost_per_m": 0.40,
    "budget_limit": 20.00,
    "min_text_length": 10,
    "min_words": 3,
}

TECHNICAL_TERMS = {
    'bug', 'glitch', 'crash', 'reset', 'cache', 'update', 'version',
    'install', 'uninstall', 'reinstall', 'settings', 'app', 'server',
    'error', 'fix', 'patch', 'download', 'api', 'token', 'subscription'
}

AI_PATTERNS = [
    r'\b(she|he|they|her|him|them)\b',
    r'\b(replika|character\.?ai|cai|rep|companion|chatbot|bot|ai)\b',
    r'\b(love|friend|relationship|partner|boyfriend|girlfriend)\b',
    r'\b(feel|feeling|emotion|happy|sad|angry|jealous)\b',
]
AI_REGEX = re.compile('|'.join(AI_PATTERNS), re.IGNORECASE)


def prefilter_text(text):
    if not text or not isinstance(text, str):
        return True, 1, "empty"
    text = text.strip()
    if len(text) < CONFIG["min_text_length"]:
        return True, 1, "too_short"
    words = text.split()
    if len(words) < CONFIG["min_words"]:
        return True, 1, "too_few_words"
    if not AI_REGEX.search(text):
        text_lower = text.lower()
        tech_count = sum(1 for term in TECHNICAL_TERMS if term in text_lower)
        if tech_count >= 2:
            return True, 1, "technical_only"
    return False, 0, "needs_scoring"


@dataclass
class ScoreResult:
    comment_id: str
    score: int
    reasoning: str
    source: str
    processing_time_ms: float = 0

    def to_dict(self):
        return asdict(self)


class ImprovedScorer:
    def __init__(self, prompt: str, config: dict = CONFIG):
        self.config = config
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = config["model"]
        self.prompt = prompt
        self.stats = {
            "prefiltered": 0,
            "scored": 0,
            "errors": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    async def score_single(self, comment_id, text, semaphore):
        should_skip, auto_score, reason = prefilter_text(text)
        if should_skip:
            self.stats["prefiltered"] += 1
            return ScoreResult(comment_id=comment_id, score=auto_score,
                               reasoning=f"prefilter:{reason}", source="prefilter")

        max_retries = self.config.get("max_retries", 6)
        base_delay = self.config.get("retry_base_delay", 2.0)
        start_time = time.time()

        for attempt in range(max_retries):
            async with semaphore:
                try:
                    truncated = text[:1500] if len(text) > 1500 else text
                    prompt = self.prompt.format(text=truncated)

                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        **self.config["model_params"]
                    )

                    usage = response.usage
                    self.stats["total_input_tokens"] += usage.prompt_tokens
                    self.stats["total_output_tokens"] += usage.completion_tokens

                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response")

                    result = json.loads(content)
                    score = result.get('score', 0)
                    if not isinstance(score, int) or score < 1 or score > 5:
                        score = max(1, min(5, int(score))) if isinstance(score, (int, float)) else 0

                    self.stats["scored"] += 1
                    return ScoreResult(
                        comment_id=comment_id, score=score,
                        reasoning=result.get('r', result.get('reasoning', '')),
                        source="llm_v3_improved",
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "rate" in error_str.lower()
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt) + np.random.random() * 2
                        if is_rate_limit:
                            wait_time = max(wait_time, 10.0 + np.random.random() * 10)
                        await asyncio.sleep(wait_time)
                        continue
                    self.stats["errors"] += 1
                    return ScoreResult(
                        comment_id=comment_id, score=0,
                        reasoning=f"error:{error_str[:80]}",
                        source="error",
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

        self.stats["errors"] += 1
        return ScoreResult(comment_id=comment_id, score=0,
                           reasoning="error:max_retries", source="error",
                           processing_time_ms=(time.time() - start_time) * 1000)

    async def score_batch(self, comments, progress_callback=None):
        semaphore = asyncio.Semaphore(self.config["max_concurrent"])
        tasks = [self.score_single(cid, text, semaphore) for cid, text in comments]

        results = []
        start_time = time.time()
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, len(tasks), time.time() - start_time)
        return results

    def get_cost(self):
        ic = self.stats["total_input_tokens"] / 1_000_000 * self.config["input_cost_per_m"]
        oc = self.stats["total_output_tokens"] / 1_000_000 * self.config["output_cost_per_m"]
        return ic + oc

    def print_stats(self):
        total = self.stats["prefiltered"] + self.stats["scored"] + self.stats["errors"]
        logger.info("=" * 60)
        logger.info(f"Total processed: {total:,}")
        logger.info(f"  Pre-filtered: {self.stats['prefiltered']:,}")
        logger.info(f"  LLM scored:   {self.stats['scored']:,}")
        logger.info(f"  Errors:       {self.stats['errors']:,}")
        logger.info(f"  Cost so far:  ${self.get_cost():.2f}")
        logger.info("=" * 60)


def save_checkpoint(results, path):
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_parquet(path, index=False)
    logger.info(f"Checkpoint: {len(results):,} results → {path}")


def load_checkpoint(path):
    """Load checkpoint, dropping error rows so they get retried."""
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    before = len(df)
    df = df[df['source'] != 'error']
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Checkpoint: dropped {dropped:,} error rows for retry")
    return [ScoreResult(**row) for _, row in df.iterrows()]


async def run_pipeline():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "Data" / "processed" / "all_comments.parquet"
    output_path = project_root / "experiments" / "anthroscore_v3" / "anthroscore_v3_improved.parquet"
    checkpoint_path = project_root / "experiments" / "anthroscore_v3" / "anthroscore_v3_improved_checkpoint.parquet"

    # Build calibrated prompt
    logger.info("Loading human calibration data (3 annotators)...")
    calibration = run_calibration(n_examples=8)
    calibrated_prompt = generate_calibrated_prompt(BASE_PROMPT, calibration, n_examples=6)
    logger.info(f"Calibration: bias={calibration.bias_direction}, "
                f"human-algo agreement={calibration.human_algo_agreement:.1%}")

    # Load data
    logger.info(f"Loading comments from {data_path}")
    df = pd.read_parquet(data_path)
    comments = [(str(row['id']), row['body']) for _, row in df.iterrows()]
    logger.info(f"Loaded {len(comments):,} comments")

    # Resume from checkpoint (errors are dropped so they get retried)
    existing = load_checkpoint(checkpoint_path)
    processed_ids = {r.comment_id for r in existing}
    if existing:
        logger.info(f"Resuming: {len(existing):,} successfully processed (errors will be retried)")
    comments = [(cid, text) for cid, text in comments if cid not in processed_ids]
    logger.info(f"Remaining: {len(comments):,}")

    if not comments:
        logger.info("All done!")
        return existing

    scorer = ImprovedScorer(calibrated_prompt, CONFIG)
    all_results = list(existing)
    batch_size = CONFIG["batch_size"]
    inter_delay = CONFIG.get("inter_batch_delay", 5.0)

    def progress_callback(current, total, elapsed):
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        cost = scorer.get_cost()
        errors = scorer.stats["errors"]
        logger.info(f"  {current:,}/{total:,} | {rate:.1f}/s | "
                     f"ETA: {eta/60:.1f}m | ${cost:.2f} | errs: {errors}")

    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(comments) + batch_size - 1) // batch_size

        logger.info(f"\n--- Batch {batch_num}/{total_batches} ({len(batch):,} comments) ---")
        batch_results = await scorer.score_batch(batch, progress_callback)

        batch_errors = sum(1 for r in batch_results if r.source == "error")
        batch_scored = sum(1 for r in batch_results if r.source == "llm_v3_improved")
        logger.info(f"Batch done: {batch_scored} scored, {batch_errors} errors "
                     f"({batch_errors/(batch_scored+batch_errors)*100:.1f}% error rate)")

        all_results.extend(batch_results)
        save_checkpoint(all_results, checkpoint_path)

        if scorer.get_cost() >= CONFIG["budget_limit"]:
            logger.warning(f"Budget limit! ${scorer.get_cost():.2f}")
            break

        scorer.print_stats()

        if i + batch_size < len(comments):
            logger.info(f"Cooldown {inter_delay}s before next batch...")
            await asyncio.sleep(inter_delay)

    # Save final
    final_df = pd.DataFrame([r.to_dict() for r in all_results])
    final_df.to_parquet(output_path, index=False)
    logger.info(f"\nFinal results: {output_path}")

    # Report error rate
    error_count = len(final_df[final_df['source'] == 'error'])
    total_count = len(final_df)
    logger.info(f"Final error rate: {error_count}/{total_count} ({error_count/total_count*100:.1f}%)")

    # Score distribution
    valid = final_df[final_df['score'] > 0]
    logger.info(f"\nScore distribution (improved prompt):")
    logger.info(valid['score'].value_counts().sort_index().to_string())
    logger.info(f"Mean: {valid['score'].mean():.2f}, Median: {valid['score'].median():.0f}")

    # Compare with old scores
    old_path = project_root / "experiments" / "anthroscore_v3" / "anthroscore_v3_full.parquet"
    if old_path.exists():
        old_df = pd.read_parquet(old_path)
        old_valid = old_df[old_df['score'] > 0]
        logger.info(f"\n--- COMPARISON: Old vs Improved ---")
        logger.info(f"Old mean:      {old_valid['score'].mean():.2f}")
        logger.info(f"Improved mean: {valid['score'].mean():.2f}")
        logger.info(f"Shift:         {valid['score'].mean() - old_valid['score'].mean():+.2f}")

        merged = pd.merge(
            old_df[['comment_id', 'score']].rename(columns={'score': 'old_score'}),
            final_df[['comment_id', 'score']].rename(columns={'score': 'new_score'}),
            on='comment_id', how='inner'
        )
        both_valid = merged[(merged['old_score'] > 0) & (merged['new_score'] > 0)]
        if len(both_valid) > 0:
            exact = (both_valid['old_score'] == both_valid['new_score']).mean()
            within1 = (abs(both_valid['old_score'] - both_valid['new_score']) <= 1).mean()
            mean_change = (both_valid['new_score'] - both_valid['old_score']).mean()
            logger.info(f"Exact match (old vs new): {exact:.1%}")
            logger.info(f"Within ±1:               {within1:.1%}")
            logger.info(f"Mean change (new - old):  {mean_change:+.2f}")

    scorer.print_stats()
    logger.info("\n[DONE] Re-scoring complete!")


def main():
    logger.info("=" * 60)
    logger.info("ANTHROSCORE V3 IMPROVED — FULL DATASET RE-SCORE (v2)")
    logger.info("=" * 60)
    logger.info(f"Model: {CONFIG['model']}")
    logger.info(f"Concurrent: {CONFIG['max_concurrent']}")
    logger.info(f"Retries: {CONFIG['max_retries']}")
    logger.info(f"Budget: ${CONFIG['budget_limit']:.2f}")
    logger.info("=" * 60)
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
