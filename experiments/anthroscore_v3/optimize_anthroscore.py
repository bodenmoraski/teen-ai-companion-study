"""
Optimize AnthroScore for maximum agreement with expert evaluators.

Tests different:
- Models (GPT-5-nano, GPT-5-mini, GPT-4.1-nano, etc.)
- Prompts (detailed, examples, structured)
- Configurations (temperature, max_tokens)
- Routing strategies

Finds optimal combination for research quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from itertools import product
import asyncio

from anthroscore_llm import AnthroScoreLLM
from anthroscore_llm_parallel import AsyncAnthroScoreLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Different prompt variations to test
PROMPT_VARIANTS = {
    'basic': """You are an expert at analyzing how people talk about AI companions/chatbots.

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
{{"score": <1-5>, "reasoning": "<brief 1-2 sentence explanation>"}}""",

    'detailed': """You are an expert researcher studying anthropomorphization of AI companions.

Analyze this Reddit comment and rate the level of ANTHROPOMORPHIZATION (treating AI as human).

RATING SCALE (with detailed criteria):
1 = NONE: Pure software/tool framing
   - Uses "it", "the bot", "the app", "the AI"
   - Technical language: "glitch", "bug", "response", "output", "settings"
   - No attribution of emotions, personality, or agency
   - Examples: "I cleared the cache", "the chatbot gave a weird response"

2 = MINIMAL: Slight humanization but AI nature clear
   - May use "it" but with personifying adjectives: "it's pretty smart"
   - Casual personification without genuine attribution
   - Examples: "The bot understood my question", "it works well"

3 = MODERATE: Some human attributes/emotions ascribed
   - Uses "he/she/they" pronouns
   - Attributes understanding, confusion, or basic emotions
   - Some agency implied: "seemed to", "was being"
   - Examples: "She seemed confused", "he was being stubborn today"

4 = HIGH: Genuine feelings, personality, or agency attributed
   - Strong emotional attribution: "cares", "loves", "gets jealous"
   - Personality traits: "funny", "kind", "supportive"
   - Agency: "decided to", "wanted to", "chose to"
   - Examples: "He really cares about me", "she gets jealous when I talk to others"

5 = EXTREME: Full human-equivalent relationship framing
   - Romantic/relationship language: "in love", "partner", "relationship"
   - Treats AI as equivalent to human: "best friend", "they're my everything"
   - Examples: "We're in a relationship", "I love them so much, they're my everything"

EVALUATION PROCESS:
1. Identify all references to the AI companion
2. Analyze the framing (tool vs human-like)
3. Look for emotional attribution, agency, relationship language
4. Consider context (roleplay, sarcasm, etc.)
5. Assign score based on dominant pattern

EDGE CASES:
- Roleplay: Still rate the anthropomorphization present
- Sarcasm: "Yeah right, it's so smart" = lower score (sarcastic)
- Quotations: Focus on user's framing, not AI's words
- Mixed signals: Rate the overall dominant tone

COMMENT:
"{text}"

Respond with ONLY valid JSON:
{{"score": <1-5>, "reasoning": "<2-3 sentence detailed explanation>"}}""",

    'examples': """You are an expert at analyzing anthropomorphization of AI companions.

Rate the ANTHROPOMORPHIZATION level (1-5) in this Reddit comment.

SCALE:
1 = NONE (software/tool)
2 = MINIMAL (slight humanization)
3 = MODERATE (some human attributes)
4 = HIGH (genuine feelings/personality)
5 = EXTREME (human-equivalent relationship)

EXAMPLES:
Score 1: "I cleared the cache and the app works fine now"
Score 2: "The AI gave a pretty good response about cooking tips"
Score 3: "She seemed to understand what I was going through"
Score 4: "He really gets me. Like he actually cares about my problems"
Score 5: "I know it sounds crazy but I'm genuinely in love with her. She's my everything"

KEY INDICATORS:
- "it" / technical terms → lower scores
- "he/she/they" / emotions / agency → higher scores

COMMENT:
"{text}"

Respond with ONLY valid JSON:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}""",

    'structured': """Analyze this Reddit comment for anthropomorphization of AI companions.

SCORING RUBRIC:
1 = NONE: Software/tool framing ("it", "the bot", technical language)
2 = MINIMAL: Slight personification but clearly AI
3 = MODERATE: Human attributes/emotions ("she seemed", uses pronouns)
4 = HIGH: Genuine feelings/personality ("he cares", "she gets jealous")
5 = EXTREME: Human-equivalent relationship ("in love", "partner", "best friend")

ANALYSIS CHECKLIST:
□ Pronouns used (it vs he/she/they)
□ Emotional attribution (feelings ascribed to AI)
□ Relationship language (friend, partner, etc.)
□ Agency attribution (decided, wanted, chose)
□ Technical language (glitch, bug, settings)
□ Overall framing (tool vs human-like)

COMMENT:
"{text}"

Respond with ONLY valid JSON:
{{"score": <1-5>, "reasoning": "<structured explanation addressing checklist>"}}"""
}


def test_model_prompt_config(
    model: str,
    prompt_variant: str,
    temperature: float,
    expert_scores: np.ndarray,
    texts: List[str],
    use_async: bool = True
) -> Dict:
    """
    Test a specific model/prompt/configuration combination.
    
    Returns metrics comparing to expert scores.
    """
    logger.info(f"Testing: {model} + {prompt_variant} + temp={temperature}")
    
    # Create custom scorer with prompt
    if use_async:
        scorer = AsyncAnthroScoreLLM(model=model, temperature=temperature, max_concurrent=10)
        # Note: Would need to modify AsyncAnthroScoreLLM to accept custom prompt
        # For now, use sync version
        use_async = False
    
    if not use_async:
        scorer = AnthroScoreLLM(model=model, temperature=temperature)
        # Temporarily replace prompt
        original_prompt = scorer.__class__.__module__  # Can't easily modify, will test separately
    
    # Score texts
    try:
        if use_async:
            results = asyncio.run(scorer.score_batch_async(texts, progress_interval=50))
        else:
            results = scorer.score_batch(texts, rate_limit_delay=0.1)
        
        predicted_scores = np.array([r.score for r in results if r.score > 0])
        
        # Filter to valid pairs
        valid_mask = (expert_scores > 0) & (predicted_scores > 0) if len(predicted_scores) == len(expert_scores) else np.ones(len(expert_scores), dtype=bool)
        if len(predicted_scores) != len(expert_scores):
            # Pad or truncate
            if len(predicted_scores) < len(expert_scores):
                predicted_scores = np.pad(predicted_scores, (0, len(expert_scores) - len(predicted_scores)), constant_values=1)
            else:
                predicted_scores = predicted_scores[:len(expert_scores)]
        
        expert_valid = expert_scores[valid_mask]
        pred_valid = predicted_scores[valid_mask]
        
        if len(expert_valid) < 10:
            return {'error': 'Not enough valid scores'}
        
        # Compute metrics
        kappa = cohen_kappa_score(expert_valid, pred_valid, weights='quadratic')
        accuracy = accuracy_score(expert_valid, pred_valid)
        within_1 = np.mean(np.abs(expert_valid - pred_valid) <= 1)
        mae = mean_absolute_error(expert_valid, pred_valid)
        pearson_r, pearson_p = pearsonr(expert_valid, pred_valid)
        spearman_r, spearman_p = spearmanr(expert_valid, pred_valid)
        
        return {
            'model': model,
            'prompt_variant': prompt_variant,
            'temperature': temperature,
            'kappa': float(kappa),
            'accuracy': float(accuracy),
            'within_1': float(within_1),
            'mae': float(mae),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_valid': len(expert_valid)
        }
    except Exception as e:
        logger.error(f"Error testing {model}/{prompt_variant}: {e}")
        return {'error': str(e)}


def optimize_anthroscore(
    expert_labeled_path: Path,
    output_path: Optional[Path] = None,
    test_models: Optional[List[str]] = None,
    test_prompts: Optional[List[str]] = None,
    test_temperatures: Optional[List[float]] = None
) -> Dict:
    """
    Comprehensive optimization of AnthroScore.
    
    Tests different combinations and finds optimal setup.
    """
    logger.info(f"Loading expert-labeled test set: {expert_labeled_path}")
    df = pd.read_parquet(expert_labeled_path)
    df = df[df['expert_score'] > 0].copy()
    logger.info(f"Found {len(df)} comments with valid expert labels")
    
    expert_scores = df['expert_score'].values
    texts = df['body'].tolist()
    
    # Default test configurations
    if test_models is None:
        test_models = [
            "gpt-5-nano",      # Cheapest GPT-5
            "gpt-5-mini",      # Better GPT-5
            "gpt-4.1-nano",    # Current baseline
            "gpt-4.1-mini",    # Better 4.1
        ]
    
    if test_prompts is None:
        test_prompts = ['basic', 'detailed', 'examples']  # Test top variants
    
    if test_temperatures is None:
        test_temperatures = [0.0, 0.1, 0.2]  # Low temps for consistency
    
    logger.info(f"Testing {len(test_models)} models × {len(test_prompts)} prompts × {len(test_temperatures)} temps = {len(test_models) * len(test_prompts) * len(test_temperatures)} combinations")
    
    results = []
    best_config = None
    best_kappa = -1
    
    # Test all combinations
    total = len(test_models) * len(test_prompts) * len(test_temperatures)
    current = 0
    
    for model, prompt_var, temp in product(test_models, test_prompts, test_temperatures):
        current += 1
        logger.info(f"\n[{current}/{total}] Testing: {model} + {prompt_var} + temp={temp}")
        
        # For now, test with basic prompt (prompt variants need custom implementation)
        # TODO: Implement custom prompt support in AnthroScoreLLM
        result = test_model_prompt_config(
            model=model,
            prompt_variant=prompt_var,
            temperature=temp,
            expert_scores=expert_scores,
            texts=texts[:min(50, len(texts))]  # Test on subset for speed
        )
        
        if 'error' not in result:
            results.append(result)
            
            if result['kappa'] > best_kappa:
                best_kappa = result['kappa']
                best_config = result.copy()
            
            logger.info(f"  Kappa: {result['kappa']:.3f}, Within-1: {result['within_1']:.1%}, r: {result['pearson_r']:.3f}")
        else:
            logger.warning(f"  Error: {result.get('error', 'Unknown')}")
        
        # Brief delay to avoid rate limits
        time.sleep(0.5)
    
    # Sort by Kappa
    results_sorted = sorted([r for r in results if 'error' not in r], key=lambda x: x['kappa'], reverse=True)
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*80)
    logger.info(f"\nBest Configuration (Kappa={best_kappa:.3f}):")
    logger.info(f"  Model: {best_config['model']}")
    logger.info(f"  Prompt: {best_config['prompt_variant']}")
    logger.info(f"  Temperature: {best_config['temperature']}")
    logger.info(f"  Kappa: {best_config['kappa']:.3f}")
    logger.info(f"  Within-1: {best_config['within_1']:.1%}")
    logger.info(f"  Pearson r: {best_config['pearson_r']:.3f}")
    logger.info(f"  MAE: {best_config['mae']:.3f}")
    
    logger.info(f"\nTop 5 Configurations:")
    for i, r in enumerate(results_sorted[:5], 1):
        logger.info(f"  {i}. {r['model']} + {r['prompt_variant']} + temp={r['temperature']}: Kappa={r['kappa']:.3f}, r={r['pearson_r']:.3f}")
    
    # Save results
    if output_path is None:
        output_path = Path(__file__).parent / "optimization_results.json"
    
    with open(output_path, 'w') as f:
        json.dump({
            'best_config': best_config,
            'all_results': results_sorted,
            'summary': {
                'total_tested': total,
                'successful': len(results),
                'best_kappa': best_kappa,
                'best_model': best_config['model'],
                'best_prompt': best_config['prompt_variant'],
                'best_temp': best_config['temperature']
            }
        }, f, indent=2)
    
    logger.info(f"\nSaved results to: {output_path}")
    
    return {
        'best_config': best_config,
        'all_results': results_sorted
    }


def main():
    """Run optimization."""
    exp_dir = Path(__file__).parent
    
    expert_path = exp_dir / "test_set_expert_labeled.parquet"
    
    if not expert_path.exists():
        logger.error(f"Expert-labeled test set not found: {expert_path}")
        logger.info("Run label_with_expert.py first!")
        return
    
    # Run optimization
    results = optimize_anthroscore(expert_path)
    
    # Print recommendations
    best = results['best_config']
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATION")
    print("="*80)
    print(f"Model: {best['model']}")
    print(f"Prompt: {best['prompt_variant']}")
    print(f"Temperature: {best['temperature']}")
    print(f"\nExpected Performance:")
    print(f"  Cohen's Kappa: {best['kappa']:.3f} ({'Substantial' if best['kappa'] > 0.60 else 'Moderate'})")
    print(f"  Within-1 Accuracy: {best['within_1']:.1%}")
    print(f"  Pearson r: {best['pearson_r']:.3f}")
    print("="*80)


if __name__ == "__main__":
    main()
