#!/usr/bin/env python3
"""
Score the confirmatory dataset (Jan 2024) with the improved AnthroScore V3.
Same prompt/config as the main dataset scoring.
"""

import pandas as pd
import numpy as np
import json
import logging
import time
import asyncio
import re
from pathlib import Path
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
        logging.FileHandler(Path(__file__).parent.parent / 'results' / 'confirmatory_scoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

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

CONFIG = {
    "model": "gpt-4.1-nano",
    "model_params": {"max_tokens": 100, "temperature": 0.0},
    "max_concurrent": 8,
    "batch_size": 2000,
    "max_retries": 6,
    "retry_base_delay": 2.0,
    "inter_batch_delay": 5.0,
    "input_cost_per_m": 0.10,
    "output_cost_per_m": 0.40,
    "budget_limit": 15.00,
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
    if len(text.split()) < CONFIG["min_words"]:
        return True, 1, "too_few_words"
    if not AI_REGEX.search(text):
        text_lower = text.lower()
        if sum(1 for t in TECHNICAL_TERMS if t in text_lower) >= 2:
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


class Scorer:
    def __init__(self, prompt, config=CONFIG):
        self.config = config
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = config["model"]
        self.prompt = prompt
        self.stats = {"prefiltered": 0, "scored": 0, "errors": 0,
                      "total_input_tokens": 0, "total_output_tokens": 0}

    async def score_single(self, cid, text, sem):
        skip, auto, reason = prefilter_text(text)
        if skip:
            self.stats["prefiltered"] += 1
            return ScoreResult(cid, auto, f"prefilter:{reason}", "prefilter")
        start = time.time()
        for attempt in range(self.config["max_retries"]):
            async with sem:
                try:
                    trunc = text[:1500]
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": self.prompt.format(text=trunc)}],
                        response_format={"type": "json_object"},
                        **self.config["model_params"])
                    self.stats["total_input_tokens"] += resp.usage.prompt_tokens
                    self.stats["total_output_tokens"] += resp.usage.completion_tokens
                    result = json.loads(resp.choices[0].message.content)
                    score = result.get('score', 0)
                    if not isinstance(score, int) or score < 1 or score > 5:
                        score = max(1, min(5, int(score))) if isinstance(score, (int, float)) else 0
                    self.stats["scored"] += 1
                    return ScoreResult(cid, score, result.get('r', ''),
                                       "llm_v3_improved", (time.time()-start)*1000)
                except Exception as e:
                    es = str(e)
                    if attempt < self.config["max_retries"] - 1:
                        wait = self.config["retry_base_delay"] * (2**attempt) + np.random.random()*2
                        if "429" in es or "rate" in es.lower():
                            wait = max(wait, 10 + np.random.random()*10)
                        await asyncio.sleep(wait)
                        continue
                    self.stats["errors"] += 1
                    return ScoreResult(cid, 0, f"error:{es[:80]}", "error", (time.time()-start)*1000)
        self.stats["errors"] += 1
        return ScoreResult(cid, 0, "error:max_retries", "error", (time.time()-start)*1000)

    async def score_batch(self, comments, progress_cb=None):
        sem = asyncio.Semaphore(self.config["max_concurrent"])
        tasks = [self.score_single(c, t, sem) for c, t in comments]
        results = []
        start = time.time()
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            results.append(await coro)
            if progress_cb and (i+1) % 500 == 0:
                progress_cb(i+1, len(tasks), time.time()-start)
        return results

    def get_cost(self):
        return (self.stats["total_input_tokens"]/1e6*self.config["input_cost_per_m"] +
                self.stats["total_output_tokens"]/1e6*self.config["output_cost_per_m"])


async def run():
    project = Path(__file__).parent.parent
    data_path = project / "Data" / "confirmatory" / "confirmatory_comments.parquet"
    out_path = project / "Data" / "confirmatory" / "confirmatory_scored.parquet"
    ckpt_path = project / "Data" / "confirmatory" / "confirmatory_checkpoint.parquet"

    logger.info("Loading calibration...")
    cal = run_calibration(n_examples=8)
    prompt = generate_calibrated_prompt(BASE_PROMPT, cal, n_examples=6)

    df = pd.read_parquet(data_path)
    comments = [(str(r['id']), r['body']) for _, r in df.iterrows()]
    logger.info(f"Loaded {len(comments):,} confirmatory comments")

    # Resume
    existing = []
    if ckpt_path.exists():
        ckpt = pd.read_parquet(ckpt_path)
        ckpt = ckpt[ckpt['source'] != 'error']
        existing = [ScoreResult(**row) for _, row in ckpt.iterrows()]
        logger.info(f"Resuming: {len(existing):,} done")
    done_ids = {r.comment_id for r in existing}
    comments = [(c, t) for c, t in comments if c not in done_ids]
    logger.info(f"Remaining: {len(comments):,}")

    if not comments:
        logger.info("All done!")
        return

    scorer = Scorer(prompt, CONFIG)
    all_results = list(existing)
    bs = CONFIG["batch_size"]

    def progress(cur, tot, elapsed):
        rate = cur/elapsed if elapsed > 0 else 0
        eta = (tot-cur)/rate/60 if rate > 0 else 0
        logger.info(f"  {cur:,}/{tot:,} | {rate:.1f}/s | ETA:{eta:.1f}m | ${scorer.get_cost():.2f} | errs:{scorer.stats['errors']}")

    for i in range(0, len(comments), bs):
        batch = comments[i:i+bs]
        bn = i//bs+1
        tb = (len(comments)+bs-1)//bs
        logger.info(f"\n--- Batch {bn}/{tb} ({len(batch):,}) ---")
        results = await scorer.score_batch(batch, progress)
        all_results.extend(results)
        pd.DataFrame([r.to_dict() for r in all_results]).to_parquet(ckpt_path, index=False)
        logger.info(f"Checkpoint: {len(all_results):,}")
        if scorer.get_cost() >= CONFIG["budget_limit"]:
            logger.warning(f"Budget! ${scorer.get_cost():.2f}")
            break
        if i+bs < len(comments):
            await asyncio.sleep(CONFIG["inter_batch_delay"])

    final = pd.DataFrame([r.to_dict() for r in all_results])
    final.to_parquet(out_path, index=False)
    valid = final[final['score'] > 0]
    logger.info(f"\n=== CONFIRMATORY RESULTS ===")
    logger.info(f"Total: {len(final):,}, Errors: {len(final[final['source']=='error']):,}")
    logger.info(f"Score distribution:\n{valid['score'].value_counts().sort_index()}")
    logger.info(f"Mean: {valid['score'].mean():.2f}, Median: {valid['score'].median():.0f}")
    logger.info(f"Cost: ${scorer.get_cost():.2f}")
    logger.info(f"Saved: {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("SCORING CONFIRMATORY DATASET (Jan 2024)")
    logger.info("=" * 60)
    asyncio.run(run())

if __name__ == "__main__":
    main()
