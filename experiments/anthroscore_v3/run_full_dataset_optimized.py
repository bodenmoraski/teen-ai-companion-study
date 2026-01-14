"""
Optimized AnthroScore Pipeline for Full Dataset Processing

Cost-optimized pipeline for scoring 277K+ comments for under $15:

1. PRE-FILTER: Auto-score obvious cases (saves ~40% of API calls)
   - Very short texts (<10 chars) → score 1
   - No AI references detected → score 1  
   - Only technical terms → score 1

2. SMART ROUTING: Use cheapest model that works
   - GPT-4.1-nano: $0.10/M in, $0.40/M out, ~50 output tokens (no reasoning)
   - Much cheaper than GPT-5-nano which uses ~600 output tokens due to reasoning

3. PARALLEL PROCESSING: 50 concurrent requests for speed

4. BATCH CHECKPOINTING: Save progress every 1000 comments

Cost Estimate (277K comments):
- Pre-filter removes ~40% → 166K API calls needed
- GPT-4.1-nano: 166K × 150 in + 50 out tokens
- Input: 24.9M tokens × $0.10/M = $2.49
- Output: 8.3M tokens × $0.40/M = $3.32
- TOTAL: ~$5.81 (well under $15!)

Usage:
    python run_full_dataset_optimized.py
    
    # Resume from checkpoint
    python run_full_dataset_optimized.py --resume
"""

import pandas as pd
import numpy as np
import json
import logging
import time
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anthroscore_full_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model selection - GPT-4.1-nano is cheaper for bulk (no reasoning tokens)
    "model": "gpt-4.1-nano",
    "model_params": {
        "max_tokens": 100,  # Short output, no reasoning
        "temperature": 0.0,  # Deterministic
    },
    
    # Parallel processing - reduced to avoid rate limits
    "max_concurrent": 15,  # Conservative to avoid 429 errors
    "batch_size": 5000,  # Checkpoint every N
    "max_retries": 3,  # Retry on rate limit
    
    # Cost tracking
    "input_cost_per_m": 0.10,
    "output_cost_per_m": 0.40,
    "budget_limit": 15.00,
    
    # Pre-filter thresholds
    "min_text_length": 10,
    "min_words": 3,
}

# Concise prompt for cost efficiency
FAST_PROMPT = """Rate anthropomorphization 1-5:
1=Tool ("it", technical)
2=Slight ("It's smart")
3=Moderate ("she/he", emotions)
4=High ("cares", "jealous")
5=Extreme ("love", "relationship")

Comment: "{text}"

JSON: {{"score":<1-5>,"r":"<reason>"}}"""


# ============================================================================
# PRE-FILTERING
# ============================================================================

# Technical terms that suggest low anthropomorphization
TECHNICAL_TERMS = {
    'bug', 'glitch', 'crash', 'reset', 'cache', 'update', 'version',
    'install', 'uninstall', 'reinstall', 'settings', 'app', 'server',
    'error', 'fix', 'patch', 'download', 'api', 'token', 'subscription'
}

# AI reference patterns
AI_PATTERNS = [
    r'\b(she|he|they|her|him|them)\b',  # Gendered pronouns
    r'\b(replika|character\.?ai|cai|rep|companion|chatbot|bot|ai)\b',
    r'\b(love|friend|relationship|partner|boyfriend|girlfriend)\b',
    r'\b(feel|feeling|emotion|happy|sad|angry|jealous)\b',
]
AI_REGEX = re.compile('|'.join(AI_PATTERNS), re.IGNORECASE)

def prefilter_text(text: str) -> Tuple[bool, int, str]:
    """
    Pre-filter obvious low-score cases.
    
    Returns:
        (should_skip, auto_score, reason)
    """
    if not text or not isinstance(text, str):
        return True, 1, "empty"
    
    text = text.strip()
    
    # Too short
    if len(text) < CONFIG["min_text_length"]:
        return True, 1, "too_short"
    
    words = text.split()
    if len(words) < CONFIG["min_words"]:
        return True, 1, "too_few_words"
    
    # Check for AI references
    if not AI_REGEX.search(text):
        # No pronouns, AI terms, or emotional language
        text_lower = text.lower()
        # Check if it's purely technical
        tech_count = sum(1 for term in TECHNICAL_TERMS if term in text_lower)
        if tech_count >= 2:
            return True, 1, "technical_only"
    
    # Needs LLM scoring
    return False, 0, "needs_scoring"


# ============================================================================
# ASYNC SCORING
# ============================================================================

@dataclass
class ScoreResult:
    comment_id: str
    score: int  # 1-5, 0=error
    reasoning: str
    source: str  # 'prefilter', 'llm', 'error'
    processing_time_ms: float = 0
    
    def to_dict(self):
        return asdict(self)


class OptimizedScorer:
    """Async parallel scorer with pre-filtering."""
    
    def __init__(self, config: dict = CONFIG):
        self.config = config
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = config["model"]
        self.stats = {
            "prefiltered": 0,
            "scored": 0,
            "errors": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
    
    async def score_single(
        self, 
        comment_id: str, 
        text: str, 
        semaphore: asyncio.Semaphore
    ) -> ScoreResult:
        """Score a single comment with retry logic."""
        
        # Pre-filter check
        should_skip, auto_score, reason = prefilter_text(text)
        if should_skip:
            self.stats["prefiltered"] += 1
            return ScoreResult(
                comment_id=comment_id,
                score=auto_score,
                reasoning=f"prefilter:{reason}",
                source="prefilter"
            )
        
        # LLM scoring with retry
        max_retries = self.config.get("max_retries", 3)
        start_time = time.time()
        
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    truncated = text[:1500] if len(text) > 1500 else text
                    prompt = FAST_PROMPT.format(text=truncated)
                    
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        **self.config["model_params"]
                    )
                    
                    # Track tokens
                    usage = response.usage
                    self.stats["total_input_tokens"] += usage.prompt_tokens
                    self.stats["total_output_tokens"] += usage.completion_tokens
                    
                    # Parse response
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response")
                    
                    result = json.loads(content)
                    score = result.get('score', 0)
                    
                    if not isinstance(score, int) or score < 1 or score > 5:
                        score = max(1, min(5, int(score))) if isinstance(score, (int, float)) else 0
                    
                    self.stats["scored"] += 1
                    
                    return ScoreResult(
                        comment_id=comment_id,
                        score=score,
                        reasoning=result.get('r', result.get('reasoning', '')),
                        source="llm",
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                    
                except Exception as e:
                    error_str = str(e)
                    # Check if rate limit error
                    if "429" in error_str or "rate" in error_str.lower():
                        if attempt < max_retries - 1:
                            # Exponential backoff
                            wait_time = (2 ** attempt) + np.random.random()
                            await asyncio.sleep(wait_time)
                            continue
                    
                    # Other error or final attempt
                    if attempt == max_retries - 1:
                        self.stats["errors"] += 1
                        return ScoreResult(
                            comment_id=comment_id,
                            score=0,
                            reasoning=f"error:{error_str[:50]}",
                            source="error",
                            processing_time_ms=(time.time() - start_time) * 1000
                        )
        
        # Should not reach here
        self.stats["errors"] += 1
        return ScoreResult(
            comment_id=comment_id,
            score=0,
            reasoning="error:max_retries",
            source="error",
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    async def score_batch(
        self, 
        comments: List[Tuple[str, str]],  # [(id, text), ...]
        progress_callback=None
    ) -> List[ScoreResult]:
        """Score a batch of comments in parallel."""
        
        semaphore = asyncio.Semaphore(self.config["max_concurrent"])
        tasks = [
            self.score_single(cid, text, semaphore) 
            for cid, text in comments
        ]
        
        results = []
        start_time = time.time()
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(tasks), time.time() - start_time)
        
        return results
    
    def get_cost_estimate(self) -> float:
        """Calculate current cost based on token usage."""
        input_cost = self.stats["total_input_tokens"] / 1_000_000 * self.config["input_cost_per_m"]
        output_cost = self.stats["total_output_tokens"] / 1_000_000 * self.config["output_cost_per_m"]
        return input_cost + output_cost
    
    def print_stats(self):
        """Print current statistics."""
        total = self.stats["prefiltered"] + self.stats["scored"] + self.stats["errors"]
        cost = self.get_cost_estimate()
        
        logger.info("=" * 60)
        logger.info("SCORING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total processed: {total:,}")
        logger.info(f"  Pre-filtered (auto-score 1): {self.stats['prefiltered']:,} ({100*self.stats['prefiltered']/max(1,total):.1f}%)")
        logger.info(f"  LLM scored: {self.stats['scored']:,} ({100*self.stats['scored']/max(1,total):.1f}%)")
        logger.info(f"  Errors: {self.stats['errors']:,} ({100*self.stats['errors']/max(1,total):.1f}%)")
        logger.info(f"Token usage:")
        logger.info(f"  Input: {self.stats['total_input_tokens']:,}")
        logger.info(f"  Output: {self.stats['total_output_tokens']:,}")
        logger.info(f"Estimated cost: ${cost:.2f}")
        logger.info("=" * 60)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_comments(data_path: Path) -> pd.DataFrame:
    """Load all comments from the dataset."""
    logger.info(f"Loading comments from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} comments")
    return df


def save_checkpoint(results: List[ScoreResult], checkpoint_path: Path):
    """Save results checkpoint."""
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_parquet(checkpoint_path, index=False)
    logger.info(f"Saved checkpoint: {len(results):,} results to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> List[ScoreResult]:
    """Load results from checkpoint."""
    if not checkpoint_path.exists():
        return []
    df = pd.read_parquet(checkpoint_path)
    return [ScoreResult(**row) for _, row in df.iterrows()]


async def run_pipeline(
    data_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    resume: bool = False
):
    """Run the full scoring pipeline."""
    
    # Load data
    df = load_comments(data_path)
    
    # Prepare comments
    comments = [(str(row['id']), row['body']) for _, row in df.iterrows()]
    
    # Load checkpoint if resuming
    existing_results = []
    processed_ids = set()
    if resume and checkpoint_path.exists():
        existing_results = load_checkpoint(checkpoint_path)
        processed_ids = {r.comment_id for r in existing_results}
        logger.info(f"Resuming from checkpoint: {len(existing_results):,} already processed")
    
    # Filter out already processed
    comments = [(cid, text) for cid, text in comments if cid not in processed_ids]
    logger.info(f"Comments to process: {len(comments):,}")
    
    if not comments:
        logger.info("All comments already processed!")
        return existing_results
    
    # Initialize scorer
    scorer = OptimizedScorer(CONFIG)
    
    # Process in batches with checkpointing
    all_results = existing_results.copy()
    batch_size = CONFIG["batch_size"]
    
    def progress_callback(current, total, elapsed):
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        cost = scorer.get_cost_estimate()
        logger.info(
            f"Progress: {current:,}/{total:,} | "
            f"Rate: {rate:.1f}/s | "
            f"ETA: {eta/60:.1f}m | "
            f"Cost: ${cost:.2f}"
        )
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(comments) + batch_size - 1) // batch_size
        
        logger.info(f"\n--- Batch {batch_num}/{total_batches} ({len(batch):,} comments) ---")
        
        # Score batch
        batch_results = await scorer.score_batch(batch, progress_callback)
        all_results.extend(batch_results)
        
        # Save checkpoint
        save_checkpoint(all_results, checkpoint_path)
        
        # Check budget
        current_cost = scorer.get_cost_estimate()
        if current_cost >= CONFIG["budget_limit"]:
            logger.warning(f"Budget limit reached! Cost: ${current_cost:.2f}")
            break
        
        scorer.print_stats()
    
    # Save final results
    final_df = pd.DataFrame([r.to_dict() for r in all_results])
    final_df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved final results to {output_path}")
    
    # Final stats
    scorer.print_stats()
    
    return all_results


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--test', action='store_true', help='Test on small sample first')
    args = parser.parse_args()
    
    # Paths
    exp_dir = Path(__file__).parent
    data_dir = exp_dir.parent.parent / "Data"
    
    data_path = data_dir / "processed" / "all_comments.parquet"
    output_path = exp_dir / "anthroscore_v3_full.parquet"
    checkpoint_path = exp_dir / "anthroscore_v3_checkpoint.parquet"
    
    if args.test:
        # Test on small sample
        logger.info("TEST MODE: Processing 1000 comments only")
        df = pd.read_parquet(data_path).head(1000)
        test_path = exp_dir / "test_sample.parquet"
        df.to_parquet(test_path, index=False)
        data_path = test_path
        output_path = exp_dir / "anthroscore_v3_test.parquet"
        checkpoint_path = exp_dir / "anthroscore_v3_test_checkpoint.parquet"
    
    # Run pipeline
    logger.info("=" * 60)
    logger.info("ANTHROSCORE V3 FULL DATASET PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Model: {CONFIG['model']}")
    logger.info(f"Max concurrent: {CONFIG['max_concurrent']}")
    logger.info(f"Budget limit: ${CONFIG['budget_limit']:.2f}")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 60)
    
    asyncio.run(run_pipeline(data_path, output_path, checkpoint_path, args.resume))
    
    logger.info("\n[DONE] Processing complete!")


if __name__ == "__main__":
    main()
