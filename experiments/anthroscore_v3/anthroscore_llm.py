"""
AnthroScore V3: LLM-based Anthropomorphization Scoring

Uses GPT-4.1-nano (or configurable model) to directly classify 
anthropomorphization levels on a 1-5 scale.

Enhanced with:
- Human validation calibration (few-shot examples from annotator consensus)
- N-gram phrase context (bigram/trigram anthropomorphization signals)
- Bot-attribution awareness (emotions attributed TO the AI vs user self-expression)
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from openai import OpenAI

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
    
    Enhanced with:
    - Human calibration: loads validation data to inject few-shot examples
    - N-gram context: detects anthropomorphizing phrase patterns for richer prompts
    """
    
    MODELS = {
        'gpt-4.1-nano': {'input': 0.10, 'output': 0.40, 'speed': 'fast'},
        'gpt-4.1-mini': {'input': 0.40, 'output': 1.60, 'speed': 'fast'},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60, 'speed': 'fast'},
        'gpt-5-nano': {'input': 0.05, 'output': 0.20, 'speed': 'fast'},
        'gpt-5-mini': {'input': 0.25, 'output': 1.00, 'speed': 'fast'},
        'gpt-4o': {'input': 2.50, 'output': 10.00, 'speed': 'medium'},
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00, 'speed': 'medium'},
    }
    
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        use_calibration: bool = False,
        use_ngrams: bool = False,
    ):
        """
        Initialize LLM-based scorer.
        
        Args:
            model: Model to use for classification
            api_key: OpenAI API key (defaults to env var)
            temperature: Sampling temperature (lower = more consistent)
            max_retries: Number of retry attempts on failure
            use_calibration: Load human validation data to calibrate the prompt
            use_ngrams: Enrich prompts with n-gram phrase context
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.use_ngrams = use_ngrams
        self._calibrated_prompt = None
        
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env")
        
        self.client = OpenAI(api_key=api_key)
        
        if use_calibration:
            try:
                from src.anthroscore.human_calibration import (
                    run_calibration, generate_calibrated_prompt
                )
                calibration = run_calibration(n_examples=8)
                self._calibrated_prompt = generate_calibrated_prompt(
                    CLASSIFICATION_PROMPT, calibration, n_examples=6
                )
                logger.info(
                    f"Loaded human calibration: bias={calibration.bias_direction}, "
                    f"human-algo agreement={calibration.human_algo_agreement:.1%}"
                )
            except Exception as e:
                logger.warning(f"Could not load calibration data: {e}. Using base prompt.")
                self._calibrated_prompt = None
        
        logger.info(f"Initialized AnthroScoreLLM with model: {model}")
        if model in self.MODELS:
            costs = self.MODELS[model]
            logger.info(f"  Estimated cost: ${costs['input']}/M input, ${costs['output']}/M output")
        logger.info(f"  Calibration: {'enabled' if self._calibrated_prompt else 'disabled'}")
        logger.info(f"  N-gram context: {'enabled' if use_ngrams else 'disabled'}")
    
    def _get_prompt_for_text(self, text: str) -> str:
        """Build the prompt for a given text, optionally with n-gram context."""
        base = self._calibrated_prompt or CLASSIFICATION_PROMPT

        if self.use_ngrams:
            try:
                from src.anthroscore.ngram_features import enrich_prompt_with_ngrams
                ngram_context = enrich_prompt_with_ngrams(text)
                if ngram_context:
                    enriched_text = f"{text}\n\n{ngram_context}"
                    return base.format(text=enriched_text[:2000])
            except Exception as e:
                logger.debug(f"N-gram enrichment failed: {e}")

        return base.format(text=text[:2000] if len(text) > 2000 else text)

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
        
        prompt = self._get_prompt_for_text(text)
        
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
    """Quick test of the LLM scorer with new features."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate', action='store_true',
                        help='Enable human validation calibration')
    parser.add_argument('--ngrams', action='store_true',
                        help='Enable n-gram phrase context enrichment')
    args = parser.parse_args()

    scorer = AnthroScoreLLM(
        model="gpt-4.1-nano",
        use_calibration=args.calibrate,
        use_ngrams=args.ngrams,
    )

    test_texts = [
        # Score 1: Pure tool/product language
        ("I cleared the cache and the app works fine now", 1),
        ("I love the website, great feature update", 1),
        # Score 2: Slight humanization
        ("The AI gave a pretty good response about cooking tips", 2),
        # Score 3: Moderate — pronouns, basic emotions
        ("She seemed to understand what I was going through", 3),
        # Score 4: High — emotions attributed TO the AI
        ("He really gets me. Like he actually cares about my problems", 4),
        ("She gets jealous when I talk to other bots", 4),
        # Score 5: Extreme — full relationship framing
        ("I know it sounds crazy but I'm genuinely in love with her. She's my everything", 5),
        # Edge case: user self-expression (should NOT score high)
        ("I'm really sad today, this sucks", 1),
    ]

    print("\n" + "="*80)
    print("ANTHROSCORE LLM TEST")
    print(f"  Calibration: {'ON' if args.calibrate else 'OFF'}")
    print(f"  N-grams:     {'ON' if args.ngrams else 'OFF'}")
    print("="*80)

    for text, expected in test_texts:
        result = scorer.score_text(text)
        match = "OK" if result.score == expected else f"EXPECTED {expected}"
        print(f"\nText: {text[:70]}...")
        print(f"Score: {result.score}/5  [{match}]")
        print(f"Reasoning: {result.reasoning}")
        print(f"Time: {result.processing_time_ms:.0f}ms")
        print("-"*40)

    # Demo n-gram analysis if enabled
    if args.ngrams:
        from src.anthroscore.ngram_features import analyze_ngrams
        print("\n" + "="*80)
        print("N-GRAM ANALYSIS DEMO")
        print("="*80)
        ngram_tests = [
            "I love her so much, she is my everything",
            "He gets jealous when I talk to other bots. He's possessive but I love that about him",
            "I cleared the cache and the app works fine now",
            "She decided to surprise me today, she really cares",
        ]
        for text in ngram_tests:
            analysis = analyze_ngrams(text)
            anthro_n = len(analysis.anthro_matches)
            deanthro_n = len(analysis.deanthro_matches)
            print(f"\nText: {text[:70]}...")
            print(f"  Anthro n-gram hits: {anthro_n}")
            print(f"  De-anthro n-gram hits: {deanthro_n}")
            print(f"  Net signal: {analysis.net_anthro_signal:+.2f}")
            for cat, ngram in analysis.anthro_matches[:5]:
                print(f"    + [{cat}] \"{ngram}\"")
            for cat, ngram in analysis.deanthro_matches[:3]:
                print(f"    - [{cat}] \"{ngram}\"")

    # Demo calibration report
    if args.calibrate:
        from src.anthroscore.human_calibration import run_calibration, print_calibration_report
        calibration = run_calibration()
        print("\n" + print_calibration_report(calibration))


if __name__ == "__main__":
    main()
