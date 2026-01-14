"""
Label test set with expert-level LLM (GPT-5.0-instant).

Creates gold standard labels for validating cheaper models.
Uses a more powerful model with detailed reasoning to establish
ground truth for the test set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Detailed expert prompt with more examples
EXPERT_PROMPT = """You are an expert researcher studying how people anthropomorphize AI companions.

Analyze this Reddit comment and rate the level of ANTHROPOMORPHIZATION (treating AI as human).

RATING SCALE:
1 = NONE: Pure software/tool framing
   Examples: "I reset the app", "the chatbot glitched", "cleared the cache"
   
2 = MINIMAL: Slight humanization but AI nature clear
   Examples: "It's pretty smart", "the bot understood", "it works well"
   
3 = MODERATE: Some human attributes/emotions ascribed
   Examples: "She seemed confused", "he was being stubborn", uses he/she pronouns casually
   
4 = HIGH: Genuine feelings, personality, or agency attributed
   Examples: "He really cares about me", "she gets jealous", "he remembered what I said"
   
5 = EXTREME: Full human-equivalent relationship framing
   Examples: "We're in a relationship", "I love them", "they're my best friend/partner"

EVALUATION CRITERIA:
- Pronouns: "it" = lower, "he/she/they" = higher
- Emotional attribution: describing AI feelings = higher
- Relationship terms: friend, partner, relationship = higher
- Technical terms: bug, glitch, settings, response = lower
- Agency language: "decided to", "wanted to" = higher
- Consciousness: "thinks", "feels", "knows" = higher

IMPORTANT NOTES:
- Focus on the USER's framing, not AI speech being quoted
- Roleplay context still counts - rate the actual framing used
- If comment doesn't reference AI at all, rate 1
- Consider overall dominant tone if mixed signals

COMMENT:
"{text}"

RESPOND WITH JSON:
{{
    "score": <1-5>,
    "reasoning": "<detailed 2-3 sentence explanation>",
    "key_indicators": ["<list of specific phrases that influenced rating>"],
    "confidence": <0.0-1.0 how confident you are>
}}"""


def label_with_expert(
    test_set_path: Path,
    output_path: Path,
    model: str = "gpt-5.0-instant",
    rate_limit: float = 0.2
) -> pd.DataFrame:
    """
    Label test set comments using an expert-level LLM.
    
    Args:
        test_set_path: Path to unlabeled test set
        output_path: Path to save labeled test set
        model: Model to use for expert labeling
        rate_limit: Delay between API calls
        
    Returns:
        DataFrame with expert labels
    """
    logger.info(f"Loading test set from: {test_set_path}")
    df = pd.read_parquet(test_set_path)
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    logger.info(f"Labeling {len(df)} comments with {model}...")
    
    expert_scores = []
    expert_reasoning = []
    expert_indicators = []
    expert_confidence = []
    
    import time
    
    for i, row in df.iterrows():
        text = row['body']
        
        # Retry logic for empty responses
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                prompt = EXPERT_PROMPT.format(text=text[:2000])
                
                # GPT-5 models require max_completion_tokens and only support default temperature (1)
                is_gpt5 = "gpt-5" in model.lower()
                
                params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
                
                # GPT-5 models MUST use max_completion_tokens (not max_tokens)
                # GPT-5 models only support default temperature (1), so don't set it
                # GPT-5 uses reasoning tokens BEFORE output - need enough for reasoning + output
                if is_gpt5:
                    params["max_completion_tokens"] = 2500  # ~1500 reasoning + ~1000 output
                    # Don't set temperature - GPT-5 only supports default (1)
                else:
                    params["max_tokens"] = 500
                    params["temperature"] = 0.1  # Lower temperature for more consistent outputs
                
                response = client.chat.completions.create(**params)
                
                # Check finish reason
                finish_reason = response.choices[0].finish_reason
                content = response.choices[0].message.content
                
                if finish_reason == 'length':
                    logger.warning(f"Comment {i}: Hit token limit (attempt {attempt+1})")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                
                if not content or not content.strip():
                    logger.warning(f"Comment {i}: Empty response (attempt {attempt+1})")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise ValueError("Empty response from model after retries")
                
                result = json.loads(content)
                
                score = result.get('score', 0)
                if not isinstance(score, int) or score < 1 or score > 5:
                    score = max(1, min(5, int(score))) if isinstance(score, (int, float)) else 3
                
                expert_scores.append(score)
                expert_reasoning.append(result.get('reasoning', ''))
                expert_indicators.append(json.dumps(result.get('key_indicators', [])))
                expert_confidence.append(result.get('confidence', 0.5))
                success = True
                break
                
            except json.JSONDecodeError as e:
                logger.warning(f"Comment {i}: JSON parse error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error on comment {i} (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        if not success:
            expert_scores.append(0)
            expert_reasoning.append("ERROR: Failed after max retries")
            expert_indicators.append("[]")
            expert_confidence.append(0.0)
        
        if (i + 1) % 10 == 0:
            valid_so_far = sum(1 for s in expert_scores if s > 0)
            logger.info(f"  Progress: {i+1}/{len(df)} ({valid_so_far} valid)")
        
        time.sleep(rate_limit)
    
    df['expert_score'] = expert_scores
    df['expert_reasoning'] = expert_reasoning
    df['expert_indicators'] = expert_indicators
    df['expert_confidence'] = expert_confidence
    
    # Save
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved labeled test set to: {output_path}")
    
    # Statistics
    valid_scores = [s for s in expert_scores if s > 0]
    logger.info(f"\nExpert labeling complete:")
    logger.info(f"  Valid labels: {len(valid_scores)}/{len(df)}")
    logger.info(f"  Score distribution: {pd.Series(valid_scores).value_counts().sort_index().to_dict()}")
    logger.info(f"  Mean score: {np.mean(valid_scores):.2f}")
    
    return df


def main():
    """Run expert labeling on test set."""
    exp_dir = Path(__file__).parent
    
    test_set_path = exp_dir / "test_set_unlabeled.parquet"
    output_path = exp_dir / "test_set_expert_labeled.parquet"
    
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        logger.info("Run create_test_set.py first!")
        return
    
    # Try different models based on availability
    # Use GPT-5 as the "expert" model - smarter and better!
    # Fall back to alternatives if needed
    models_to_try = ["gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini", "gpt-4o"]
    
    for model in models_to_try:
        try:
            logger.info(f"Attempting to use model: {model}")
            df = label_with_expert(test_set_path, output_path, model=model)
            break
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            if model == models_to_try[-1]:
                raise
    
    # Show samples
    print("\n" + "="*80)
    print("EXPERT LABELED SAMPLES")
    print("="*80)
    
    for i, row in df.head(5).iterrows():
        print(f"\n[{i+1}] Score: {row['expert_score']}/5")
        print(f"    Text: {row['body'][:100]}...")
        print(f"    Reasoning: {row['expert_reasoning']}")
        print("-"*40)


if __name__ == "__main__":
    main()
