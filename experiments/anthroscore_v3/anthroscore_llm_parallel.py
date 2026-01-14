"""
Parallel/Async version of AnthroScore LLM scorer.

Uses asyncio and concurrent API calls to dramatically speed up processing.
Can process 10-50x faster than sequential version.
"""

import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import OPENAI_API_KEY
from anthroscore_llm import CLASSIFICATION_PROMPT, AnthroScoreResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsyncAnthroScoreLLM:
    """
    Async/parallel version of LLM-based anthropomorphization scorer.
    
    Uses asyncio to make concurrent API calls, dramatically speeding up
    batch processing.
    """
    
    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_concurrent: int = 20,  # Number of concurrent requests
        max_retries: int = 3
    ):
        """
        Initialize async LLM scorer.
        
        Args:
            model: Model to use for classification
            api_key: OpenAI API key (defaults to env var)
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent API calls
            max_retries: Number of retry attempts on failure
        """
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env")
        
        self.client = AsyncOpenAI(api_key=api_key)
        
        logger.info(f"Initialized AsyncAnthroScoreLLM with model: {model}")
        logger.info(f"  Max concurrent requests: {max_concurrent}")
    
    async def score_text_async(self, text: str, semaphore: asyncio.Semaphore) -> AnthroScoreResult:
        """
        Score a single text asynchronously.
        
        Args:
            text: Comment text to analyze
            semaphore: Semaphore to limit concurrent requests
            
        Returns:
            AnthroScoreResult
        """
        if not text or not text.strip():
            return AnthroScoreResult(
                score=1,
                reasoning="Empty or invalid text",
                confidence=1.0,
                raw_text=text,
                model=self.model,
                processing_time_ms=0
            )
        
        # Truncate very long texts
        truncated_text = text[:2000] if len(text) > 2000 else text
        prompt = CLASSIFICATION_PROMPT.format(text=truncated_text)
        
        start_time = time.time()
        
        async with semaphore:  # Limit concurrent requests
            for attempt in range(self.max_retries):
                try:
                    # GPT-5 models: use max_completion_tokens (~800 for reasoning + output)
                    # GPT-5 only supports default temperature (1), don't set it
                    # GPT-4.x models: use max_tokens, can set custom temperature
                    is_gpt5 = "gpt-5" in self.model.lower()
                    
                    params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"}
                    }
                    
                    if is_gpt5:
                        params["max_completion_tokens"] = 800  # ~500 reasoning + ~300 output
                        # Don't set temperature - GPT-5 only supports default (1)
                    else:
                        params["max_tokens"] = 200
                        params["temperature"] = self.temperature
                    
                    response = await self.client.chat.completions.create(**params)
                    
                    result_json = json.loads(response.choices[0].message.content)
                    
                    # Validate and extract
                    score = result_json.get('score', 1)
                    if not isinstance(score, int) or score < 1 or score > 5:
                        score = max(1, min(5, int(score))) if isinstance(score, (int, float)) else 3
                    
                    reasoning = result_json.get('reasoning', 'No reasoning provided')
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    return AnthroScoreResult(
                        score=score,
                        reasoning=reasoning,
                        confidence=1.0,
                        raw_text=text,
                        model=self.model,
                        processing_time_ms=elapsed_ms
                    )
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        return self._error_result(text, f"JSON parse error: {e}", start_time)
                    await asyncio.sleep(0.5)  # Brief delay
                    
                except Exception as e:
                    logger.warning(f"API error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        return self._error_result(text, f"API error: {e}", start_time)
                    await asyncio.sleep(1)  # Longer delay for API errors
        
        return self._error_result(text, "Max retries exceeded", start_time)
    
    def _error_result(self, text: str, error: str, start_time: float) -> AnthroScoreResult:
        """Create result for error cases."""
        return AnthroScoreResult(
            score=0,
            reasoning=f"ERROR: {error}",
            confidence=0.0,
            raw_text=text,
            model=self.model,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    async def score_batch_async(
        self,
        texts: List[str],
        progress_interval: int = 100
    ) -> List[AnthroScoreResult]:
        """
        Score multiple texts concurrently.
        
        Args:
            texts: List of comment texts
            progress_interval: Log progress every N items
            
        Returns:
            List of AnthroScoreResult objects
        """
        total = len(texts)
        logger.info(f"Scoring {total} texts with {self.model} (max {self.max_concurrent} concurrent)...")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create all tasks
        tasks = [self.score_text_async(text, semaphore) for text in texts]
        
        # Process with progress tracking
        results = []
        completed = 0
        start_time = time.time()
        
        # Use asyncio.as_completed to get results as they finish
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            
            if completed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {completed}/{total} ({100*completed/total:.1f}%) | "
                    f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f}m"
                )
        
        # Sort results to match input order (as_completed doesn't preserve order)
        # We'll match by text content
        text_to_result = {r.raw_text: r for r in results}
        ordered_results = [text_to_result.get(text, results[0]) for text in texts]
        
        elapsed = time.time() - start_time
        logger.info(f"Completed scoring {total} texts in {elapsed:.1f}s ({total/elapsed:.1f} texts/sec)")
        
        # Log statistics
        valid_scores = [r.score for r in ordered_results if r.score > 0]
        if valid_scores:
            logger.info(
                f"  Score distribution: min={min(valid_scores)}, max={max(valid_scores)}, "
                f"mean={sum(valid_scores)/len(valid_scores):.2f}"
            )
            logger.info(f"  Errors: {len(ordered_results) - len(valid_scores)}")
        
        return ordered_results


def score_batch_parallel(
    texts: List[str],
    model: str = "gpt-5-nano",
    max_concurrent: int = 20,
    progress_interval: int = 100
) -> List[AnthroScoreResult]:
    """
    Convenience function to score a batch of texts in parallel.
    
    Args:
        texts: List of comment texts
        model: Model to use
        max_concurrent: Maximum concurrent API calls
        progress_interval: Log progress every N items
        
    Returns:
        List of AnthroScoreResult objects
    """
    scorer = AsyncAnthroScoreLLM(model=model, max_concurrent=max_concurrent)
    return asyncio.run(scorer.score_batch_async(texts, progress_interval))


async def main():
    """Test the async scorer."""
    scorer = AsyncAnthroScoreLLM(model="gpt-4.1-nano", max_concurrent=10)
    
    test_texts = [
        "I cleared the cache and the app works fine now",
        "The AI gave a pretty good response about cooking tips",
        "She seemed to understand what I was going through",
        "He really gets me. Like he actually cares about my problems",
        "I know it sounds crazy but I'm genuinely in love with her. She's my everything",
    ] * 4  # 20 total for testing
    
    print("\n" + "="*80)
    print("ASYNC ANTHROSCORE LLM TEST")
    print("="*80)
    
    start = time.time()
    results = await scorer.score_batch_async(test_texts, progress_interval=5)
    elapsed = time.time() - start
    
    print(f"\nProcessed {len(results)} texts in {elapsed:.2f}s ({len(results)/elapsed:.1f} texts/sec)")
    print("\nSample results:")
    for i, (text, result) in enumerate(zip(test_texts[:5], results[:5])):
        print(f"\n{i+1}. Text: {text[:50]}...")
        print(f"   Score: {result.score}/5")
        print(f"   Time: {result.processing_time_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
