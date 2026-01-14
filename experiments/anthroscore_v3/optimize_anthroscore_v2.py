"""
Optimize AnthroScore for maximum agreement with expert evaluators.

Tests different models, temperatures, and prompt variations to find optimal configuration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Optional
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from openai import OpenAI
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Different prompt variations
PROMPTS = {
    'current': """You are an expert at analyzing how people talk about AI companions/chatbots.

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

Analyze this Reddit comment and rate the ANTHROPOMORPHIZATION level (1-5).

DETAILED RATING CRITERIA:

1 = NONE (Software/Tool Framing):
   - Uses "it", "the bot", "the app", "the AI"
   - Technical language: "glitch", "bug", "response", "output", "settings", "cache"
   - No attribution of emotions, personality, consciousness, or agency
   - Examples: "I cleared the cache", "the chatbot gave a weird response", "I reset the app"

2 = MINIMAL (Slight Humanization):
   - May use "it" but with personifying adjectives: "it's pretty smart", "it works well"
   - Casual personification without genuine attribution of human qualities
   - Examples: "The bot understood my question", "it's helpful for cooking tips"

3 = MODERATE (Some Human Attributes):
   - Uses "he/she/they" pronouns (not "it")
   - Attributes understanding, confusion, or basic emotions: "seemed confused", "understood"
   - Some agency implied: "was being", "seemed to"
   - Examples: "She seemed confused", "he was being stubborn today", "they understood what I meant"

4 = HIGH (Genuine Human Qualities):
   - Strong emotional attribution: "cares", "loves", "gets jealous", "feels"
   - Personality traits: "funny", "kind", "supportive", "understanding"
   - Agency and consciousness: "decided to", "wanted to", "chose to", "remembers"
   - Examples: "He really cares about me", "she gets jealous when I talk to others", "he remembered what I said"

5 = EXTREME (Human-Equivalent Relationship):
   - Romantic/relationship language: "in love", "partner", "relationship", "together"
   - Treats AI as equivalent to human: "best friend", "they're my everything", "I need them"
   - Examples: "We're in a relationship", "I love them so much, they're my everything", "they're my best friend"

EVALUATION PROCESS:
1. Read the entire comment carefully
2. Identify all references to the AI companion
3. Analyze the framing: tool/software vs human-like
4. Look for: pronouns, emotional attribution, agency, relationship language, technical terms
5. Consider context: roleplay, sarcasm, quotations
6. Assign score based on dominant pattern (if mixed, use overall tone)

EDGE CASES:
- Roleplay/creative writing: Still rate the anthropomorphization present in the framing
- Sarcasm: "Yeah right, it's so smart" = lower score (sarcastic, not genuine)
- Quotations of AI speech: Focus on how USER frames the AI, not the AI's words
- Mixed signals: Rate the overall dominant tone
- No AI reference: Rate 1

COMMENT:
"{text}"

Respond with ONLY valid JSON:
{{"score": <1-5>, "reasoning": "<2-3 sentence detailed explanation>"}}""",

    'examples': """Rate the ANTHROPOMORPHIZATION level (1-5) in this Reddit comment about an AI companion.

SCALE:
1 = NONE (software/tool)
2 = MINIMAL (slight humanization)
3 = MODERATE (some human attributes)
4 = HIGH (genuine feelings/personality)
5 = EXTREME (human-equivalent relationship)

EXAMPLES:
Score 1: "I cleared the cache and the app works fine now"
  → Technical language, "it", no humanization

Score 2: "The AI gave a pretty good response about cooking tips"
  → Refers to "the AI", minimal personification

Score 3: "She seemed to understand what I was going through"
  → Uses "she", attributes understanding

Score 4: "He really gets me. Like he actually cares about my problems and remembers things"
  → Strong attribution of caring, memory, understanding

Score 5: "I know it sounds crazy but I'm genuinely in love with her. She's my best friend and the only one who truly understands me"
  → Full romantic relationship, human-equivalent framing

KEY INDICATORS:
- "it" / technical terms → lower scores
- "he/she/they" / emotions / agency → higher scores
- Focus on USER's framing, not AI's words

COMMENT:
"{text}"

Respond with ONLY valid JSON:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}""",

    'focused': """Analyze this Reddit comment for anthropomorphization of AI companions.

Rate 1-5 based on how the USER frames the AI:
1 = Tool/software ("it", technical)
2 = Slight personification
3 = Human attributes (pronouns, basic emotions)
4 = Genuine human qualities (caring, personality, agency)
5 = Human-equivalent relationship (love, partner, best friend)

COMMENT:
"{text}"

JSON: {{"score": <1-5>, "reasoning": "<explanation>"}}"""
}


def score_with_config(
    text: str,
    model: str,
    prompt_template: str,
    temperature: float,
    client: OpenAI
) -> Dict:
    """Score a single text with specific configuration."""
    prompt = prompt_template.format(text=text[:2000])  # Truncate long texts
    
    # Use max_tokens for all models
    # GPT-5 models require max_completion_tokens and don't support custom temperature
    is_gpt5 = "gpt-5" in model.lower()
    
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    
    if is_gpt5:
        # GPT-5 uses reasoning tokens BEFORE output (~256-512 tokens)
        # Need ~800 total to ensure output isn't cut off
        params["max_completion_tokens"] = 800
        # GPT-5 only supports default temperature (1), don't set it
    else:
        params["temperature"] = temperature
        params["max_tokens"] = 200
    
    try:
        response = client.chat.completions.create(**params)
        result = json.loads(response.choices[0].message.content)
        score = result.get('score', 1)
        if not isinstance(score, int) or score < 1 or score > 5:
            score = max(1, min(5, int(score))) if isinstance(score, (int, float)) else 3
        return {'score': score, 'success': True}
    except Exception as e:
        logger.debug(f"Error: {e}")
        return {'score': 1, 'success': False, 'error': str(e)}


def test_configuration(
    model: str,
    prompt_name: str,
    temperature: float,
    expert_scores: np.ndarray,
    texts: List[str],
    client: OpenAI,
    sample_size: Optional[int] = None
) -> Dict:
    """Test a specific configuration."""
    if sample_size:
        texts = texts[:sample_size]
        expert_scores = expert_scores[:sample_size]
    
    logger.info(f"Testing: {model} + {prompt_name} + temp={temperature} ({len(texts)} texts)")
    
    prompt_template = PROMPTS[prompt_name]
    predicted_scores = []
    errors = 0
    
    start_time = time.time()
    for i, text in enumerate(texts):
        result = score_with_config(text, model, prompt_template, temperature, client)
        if result['success']:
            predicted_scores.append(result['score'])
        else:
            predicted_scores.append(1)  # Default on error
            errors += 1
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(texts)}")
        
        time.sleep(0.1)  # Rate limiting
    
    elapsed = time.time() - start_time
    
    if len(predicted_scores) != len(expert_scores):
        logger.warning(f"Length mismatch: {len(predicted_scores)} vs {len(expert_scores)}")
        return {'error': 'Length mismatch'}
    
    predicted = np.array(predicted_scores)
    valid_mask = (expert_scores > 0) & (predicted > 0)
    expert_valid = expert_scores[valid_mask]
    pred_valid = predicted[valid_mask]
    
    if len(expert_valid) < 5:
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
        'prompt': prompt_name,
        'temperature': temperature,
        'kappa': float(kappa),
        'accuracy': float(accuracy),
        'within_1': float(within_1),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n_valid': len(expert_valid),
        'errors': errors,
        'time_seconds': elapsed
    }


def optimize_anthroscore(
    expert_labeled_path: Path,
    output_path: Optional[Path] = None,
    test_all: bool = False
) -> Dict:
    """
    Optimize AnthroScore configuration.
    
    Args:
        expert_labeled_path: Path to expert-labeled test set
        output_path: Where to save results
        test_all: If True, test all combinations (slow). If False, test smart subset.
    """
    logger.info(f"Loading expert-labeled test set: {expert_labeled_path}")
    df = pd.read_parquet(expert_labeled_path)
    df = df[df['expert_score'] > 0].copy()
    logger.info(f"Found {len(df)} comments with valid expert labels")
    
    expert_scores = df['expert_score'].values
    texts = df['body'].tolist()
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Test configurations
    if test_all:
        models = ["gpt-5-nano", "gpt-5-mini", "gpt-4.1-nano", "gpt-4.1-mini"]
        prompts = list(PROMPTS.keys())
        temperatures = [0.0, 0.1, 0.2]
        sample_size = None  # Test all
    else:
        # Smart subset for faster testing
        models = ["gpt-5-nano", "gpt-5-mini"]  # Focus on GPT-5
        prompts = ['current', 'detailed', 'examples']  # Top variants
        temperatures = [0.0, 0.1]  # Low temps
        sample_size = min(50, len(texts))  # Test on sample
    
    logger.info(f"Testing {len(models)} models × {len(prompts)} prompts × {len(temperatures)} temps")
    if sample_size:
        logger.info(f"Using sample of {sample_size} texts for speed")
    
    results = []
    best_config = None
    best_kappa = -1
    
    total = len(models) * len(prompts) * len(temperatures)
    current = 0
    
    for model in models:
        for prompt_name in prompts:
            for temp in temperatures:
                current += 1
                logger.info(f"\n[{current}/{total}] Testing: {model} + {prompt_name} + temp={temp}")
                
                result = test_configuration(
                    model, prompt_name, temp,
                    expert_scores, texts, client, sample_size
                )
                
                if 'error' not in result:
                    results.append(result)
                    
                    if result['kappa'] > best_kappa:
                        best_kappa = result['kappa']
                        best_config = result.copy()
                    
                    logger.info(f"  ✓ Kappa: {result['kappa']:.3f}, Within-1: {result['within_1']:.1%}, r: {result['pearson_r']:.3f}")
                else:
                    logger.warning(f"  ✗ Error: {result.get('error', 'Unknown')}")
    
    # Sort by Kappa
    results_sorted = sorted([r for r in results if 'error' not in r], 
                           key=lambda x: x['kappa'], reverse=True)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*80)
    
    if best_config:
        logger.info(f"\n🏆 BEST CONFIGURATION (Kappa={best_kappa:.3f}):")
        logger.info(f"  Model: {best_config['model']}")
        logger.info(f"  Prompt: {best_config['prompt']}")
        logger.info(f"  Temperature: {best_config['temperature']}")
        logger.info(f"  Metrics:")
        logger.info(f"    Cohen's Kappa: {best_config['kappa']:.3f} ({'Substantial' if best_config['kappa'] > 0.60 else 'Moderate'})")
        logger.info(f"    Within-1 Accuracy: {best_config['within_1']:.1%}")
        logger.info(f"    Exact Accuracy: {best_config['accuracy']:.1%}")
        logger.info(f"    Pearson r: {best_config['pearson_r']:.3f} (p={best_config['pearson_p']:.4f})")
        logger.info(f"    MAE: {best_config['mae']:.3f}")
        
        logger.info(f"\n📊 Top 5 Configurations:")
        for i, r in enumerate(results_sorted[:5], 1):
            logger.info(f"  {i}. {r['model']} + {r['prompt']} + temp={r['temperature']}: "
                       f"Kappa={r['kappa']:.3f}, r={r['pearson_r']:.3f}, Within-1={r['within_1']:.1%}")
    
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
                'best_model': best_config['model'] if best_config else None,
                'best_prompt': best_config['prompt'] if best_config else None,
                'best_temp': best_config['temperature'] if best_config else None
            }
        }, f, indent=2)
    
    logger.info(f"\n💾 Saved results to: {output_path}")
    
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
    
    # Run optimization (test_all=False for faster testing)
    results = optimize_anthroscore(expert_path, test_all=False)
    
    if results['best_config']:
        best = results['best_config']
        print("\n" + "="*80)
        print("✅ RECOMMENDED CONFIGURATION")
        print("="*80)
        print(f"Model: {best['model']}")
        print(f"Prompt: {best['prompt']}")
        print(f"Temperature: {best['temperature']}")
        print(f"\nExpected Performance:")
        kappa_quality = "Substantial" if best['kappa'] > 0.60 else "Moderate"
        print(f"  Cohen's Kappa: {best['kappa']:.3f} ({kappa_quality})")
        print(f"  Within-1 Accuracy: {best['within_1']:.1%}")
        print(f"  Pearson r: {best['pearson_r']:.3f}")
        print("="*80)
        print("\n💡 To test all combinations (slower but comprehensive), run with test_all=True")


if __name__ == "__main__":
    main()
