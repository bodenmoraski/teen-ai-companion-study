"""
AnthroScore V3: LLM-based Anthropomorphization Scoring

Uses GPT-4.1-nano (or configurable model) to directly classify 
anthropomorphization levels on a 1-5 scale.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from openai import OpenAI

# Add parent directories to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Classification rubric aligned with annotation guidelines
CLASSIFICATION_PROMPT = """You are an expert at analyzing how people talk about AI companions/chatbots.

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
- Emotions: Attributing feelings (happy, sad, jealous, caring) → higher scores
- Relationship language: "friend", "partner", "relationship" → higher scores  
- Technical language: "glitch", "bug", "settings" → lower scores
- Agency: "decided to", "wanted to", "chose to" → higher scores

IMPORTANT:
- Focus on how the USER frames the AI, not what the AI says
- Roleplay context still counts - rate the framing used
- Complaints can still be anthropomorphizing ("he was being rude" = higher than "it gave a bad response")
- If no AI reference present, rate 1

COMMENT:
"{text}"

Respond with ONLY valid JSON:
{{"score": <1-5>, "reasoning": "<brief 1-2 sentence explanation>"}}"""


@dataclass
class AnthroScoreResult:
    """Result from LLM-based AnthroScore classification."""
    score: int
    reasoning: str
    confidence: float
    raw_text: str
    model: str
    processing_time_ms: float


class AnthroScoreLLM:
    """
    LLM-based anthropomorphization scorer.
    
    Uses GPT-4.1-nano by default for cost-effective classification.
    Can be configured to use more powerful models for expert labeling.
    """
    
    # Model options with estimated costs (per 1M tokens, Jan 2026)
    # Prices vary - check openai.com/pricing for current rates
    MODELS = {
        'gpt-4.1-nano': {'input': 0.10, 'output': 0.40, 'speed': 'fast'},      # Cheapest
        'gpt-4.1-mini': {'input': 0.40, 'output': 1.60, 'speed': 'fast'},      # Good balance
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60, 'speed': 'fast'},       # OpenAI standard mini
        'gpt-5-nano': {'input': 0.05, 'output': 0.20, 'speed': 'fast'},        # GPT-5 cheapest (NEW!)
        'gpt-5-mini': {'input': 0.25, 'output': 1.00, 'speed': 'fast'},        # GPT-5 for expert labels (NEW! Smarter + cheaper than GPT-4!)
        'gpt-4o': {'input': 2.50, 'output': 10.00, 'speed': 'medium'},         # Previous gen
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00, 'speed': 'medium'},   # Previous gen flagship
    }
    
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        api_key: Optional[str] = None,
        temperature: float = 0.1,  # Low temp for consistency
        max_retries: int = 3
    ):
        """
        Initialize LLM-based scorer.
        
        Args:
            model: Model to use for classification
            api_key: OpenAI API key (defaults to env var)
            temperature: Sampling temperature (lower = more consistent)
            max_retries: Number of retry attempts on failure
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env")
        
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"Initialized AnthroScoreLLM with model: {model}")
        if model in self.MODELS:
            costs = self.MODELS[model]
            logger.info(f"  Estimated cost: ${costs['input']}/M input, ${costs['output']}/M output")
    
    def score_text(self, text: str) -> AnthroScoreResult:
        """
        Score a single text for anthropomorphization.
        
        Args:
            text: Comment text to analyze
            
        Returns:
            AnthroScoreResult with score, reasoning, and metadata
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
        
        for attempt in range(self.max_retries):
            try:
                # Determine token limit parameter based on model
                # GPT-5 models: use max_completion_tokens, don't set temperature (only supports default)
                # GPT-4.x models: use max_tokens, can set custom temperature
                is_gpt5 = "gpt-5" in self.model.lower()
                
                params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
                
                if is_gpt5:
                    # GPT-5 uses reasoning tokens BEFORE output (~256-512 tokens)
                    # Need ~800 total to ensure output isn't cut off
                    params["max_completion_tokens"] = 800
                    # GPT-5 only supports default temperature (1), don't set it
                else:
                    params["max_tokens"] = 200
                    params["temperature"] = self.temperature
                
                response = self.client.chat.completions.create(**params)
                
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
                    confidence=1.0,  # LLM gives categorical, no probability
                    raw_text=text,
                    model=self.model,
                    processing_time_ms=elapsed_ms
                )
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return self._error_result(text, f"JSON parse error: {e}", start_time)
                    
            except Exception as e:
                logger.warning(f"API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return self._error_result(text, f"API error: {e}", start_time)
                time.sleep(1)  # Brief delay before retry
        
        return self._error_result(text, "Max retries exceeded", start_time)
    
    def _error_result(self, text: str, error: str, start_time: float) -> AnthroScoreResult:
        """Create result for error cases."""
        return AnthroScoreResult(
            score=0,  # 0 = error/invalid
            reasoning=f"ERROR: {error}",
            confidence=0.0,
            raw_text=text,
            model=self.model,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def score_batch(
        self,
        texts: List[str],
        rate_limit_delay: float = 0.1,
        progress_interval: int = 10
    ) -> List[AnthroScoreResult]:
        """
        Score multiple texts with rate limiting.
        
        Args:
            texts: List of comment texts
            rate_limit_delay: Seconds between API calls
            progress_interval: Log progress every N items
            
        Returns:
            List of AnthroScoreResult objects
        """
        results = []
        total = len(texts)
        
        logger.info(f"Scoring {total} texts with {self.model}...")
        
        for i, text in enumerate(texts):
            if i > 0 and i % progress_interval == 0:
                logger.info(f"  Progress: {i}/{total} ({100*i/total:.1f}%)")
            
            result = self.score_text(text)
            results.append(result)
            
            if rate_limit_delay > 0 and i < total - 1:
                time.sleep(rate_limit_delay)
        
        logger.info(f"Completed scoring {total} texts")
        
        # Log statistics
        valid_scores = [r.score for r in results if r.score > 0]
        if valid_scores:
            logger.info(f"  Score distribution: min={min(valid_scores)}, max={max(valid_scores)}, mean={sum(valid_scores)/len(valid_scores):.2f}")
            logger.info(f"  Errors: {len(results) - len(valid_scores)}")
        
        return results
    
    def results_to_dataframe(self, results: List[AnthroScoreResult]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([
            {
                'anthroscore_llm': r.score,
                'llm_reasoning': r.reasoning,
                'llm_confidence': r.confidence,
                'llm_model': r.model,
                'llm_time_ms': r.processing_time_ms
            }
            for r in results
        ])


def main():
    """Quick test of the LLM scorer."""
    scorer = AnthroScoreLLM(model="gpt-4.1-nano")
    
    test_texts = [
        "I cleared the cache and the app works fine now",  # Expected: 1
        "The AI gave a pretty good response about cooking tips",  # Expected: 2
        "She seemed to understand what I was going through",  # Expected: 3
        "He really gets me. Like he actually cares about my problems",  # Expected: 4
        "I know it sounds crazy but I'm genuinely in love with her. She's my everything",  # Expected: 5
    ]
    
    print("\n" + "="*80)
    print("ANTHROSCORE LLM TEST")
    print("="*80)
    
    for text in test_texts:
        result = scorer.score_text(text)
        print(f"\nText: {text[:60]}...")
        print(f"Score: {result.score}/5")
        print(f"Reasoning: {result.reasoning}")
        print(f"Time: {result.processing_time_ms:.0f}ms")
        print("-"*40)


if __name__ == "__main__":
    main()
