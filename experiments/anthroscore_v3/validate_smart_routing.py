"""
Validate and optimize smart routing system.

Tests routing decisions against expert labels to ensure routing improves quality.
Calibrates thresholds based on actual performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr
import json
from typing import Dict, List, Tuple

from smart_routing_scorer import SmartRoutingScorer
from anthroscore_llm import AnthroScoreLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_routing_quality(
    expert_labeled_path: Path,
    test_routing: bool = True
) -> Dict:
    """
    Validate that smart routing improves quality vs cheap-only.
    
    Args:
        expert_labeled_path: Path to expert-labeled test set
        test_routing: If True, test actual routing decisions
        
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Loading expert-labeled test set: {expert_labeled_path}")
    df = pd.read_parquet(expert_labeled_path)
    df = df[df['expert_score'] > 0].copy()
    logger.info(f"Found {len(df)} comments with valid expert labels")
    
    results = {}
    
    # 1. Baseline: Cheap model only (Tier 1)
    logger.info("\n" + "="*60)
    logger.info("BASELINE: Tier 1 (Cheap) Only")
    logger.info("="*60)
    tier1_scorer = AnthroScoreLLM(model="gpt-4.1-nano")
    tier1_results = tier1_scorer.score_batch(df['body'].tolist(), rate_limit_delay=0.1)
    tier1_scores = np.array([r.score for r in tier1_results])
    
    tier1_metrics = {
        'cohen_kappa': cohen_kappa_score(df['expert_score'], tier1_scores, weights='quadratic'),
        'accuracy': accuracy_score(df['expert_score'], tier1_scores),
        'within_1': np.mean(np.abs(df['expert_score'] - tier1_scores) <= 1),
        'mae': mean_absolute_error(df['expert_score'], tier1_scores),
        'pearson_r': pearsonr(df['expert_score'], tier1_scores)[0]
    }
    results['tier1_only'] = tier1_metrics
    logger.info(f"Cohen's Kappa: {tier1_metrics['cohen_kappa']:.3f}")
    logger.info(f"Within-1 Accuracy: {tier1_metrics['within_1']:.1%}")
    logger.info(f"Pearson r: {tier1_metrics['pearson_r']:.3f}")
    
    # 2. Expert model only (Tier 3)
    logger.info("\n" + "="*60)
    logger.info("EXPERT: Tier 3 (Expert) Only")
    logger.info("="*60)
    tier3_scorer = AnthroScoreLLM(model="gpt-5-mini")  # GPT-5 expert model
    tier3_results = tier3_scorer.score_batch(df['body'].tolist(), rate_limit_delay=0.2)
    tier3_scores = np.array([r.score for r in tier3_results])
    
    tier3_metrics = {
        'cohen_kappa': cohen_kappa_score(df['expert_score'], tier3_scores, weights='quadratic'),
        'accuracy': accuracy_score(df['expert_score'], tier3_scores),
        'within_1': np.mean(np.abs(df['expert_score'] - tier3_scores) <= 1),
        'mae': mean_absolute_error(df['expert_score'], tier3_scores),
        'pearson_r': pearsonr(df['expert_score'], tier3_scores)[0]
    }
    results['tier3_only'] = tier3_metrics
    logger.info(f"Cohen's Kappa: {tier3_metrics['cohen_kappa']:.3f}")
    logger.info(f"Within-1 Accuracy: {tier3_metrics['within_1']:.1%}")
    logger.info(f"Pearson r: {tier3_metrics['pearson_r']:.3f}")
    
    # 3. Smart routing (if test_routing)
    if test_routing:
        logger.info("\n" + "="*60)
        logger.info("SMART ROUTING: Adaptive")
        logger.info("="*60)
        
        router = SmartRoutingScorer()
        routing_scores = []
        routing_tiers = []
        
        for i, row in df.iterrows():
            result = router.score_comment(
                row['body'],
                username="test_user",
                comment_count=1
            )
            routing_scores.append(result.score)
            routing_tiers.append(result.tier_used)
        
        routing_scores = np.array(routing_scores)
        
        routing_metrics = {
            'cohen_kappa': cohen_kappa_score(df['expert_score'], routing_scores, weights='quadratic'),
            'accuracy': accuracy_score(df['expert_score'], routing_scores),
            'within_1': np.mean(np.abs(df['expert_score'] - routing_scores) <= 1),
            'mae': mean_absolute_error(df['expert_score'], routing_scores),
            'pearson_r': pearsonr(df['expert_score'], routing_scores)[0],
            'tier_distribution': {
                'tier1': int(np.sum(np.array(routing_tiers) == 1)),
                'tier2': int(np.sum(np.array(routing_tiers) == 2)),
                'tier3': int(np.sum(np.array(routing_tiers) == 3))
            }
        }
        results['smart_routing'] = routing_metrics
        logger.info(f"Cohen's Kappa: {routing_metrics['cohen_kappa']:.3f}")
        logger.info(f"Within-1 Accuracy: {routing_metrics['within_1']:.1%}")
        logger.info(f"Pearson r: {routing_metrics['pearson_r']:.3f}")
        logger.info(f"Tier distribution: {routing_metrics['tier_distribution']}")
    
    # 4. Compare improvements
    logger.info("\n" + "="*60)
    logger.info("COMPARISON")
    logger.info("="*60)
    
    improvement_vs_tier1 = {
        'kappa_improvement': routing_metrics['cohen_kappa'] - tier1_metrics['cohen_kappa'],
        'within1_improvement': routing_metrics['within_1'] - tier1_metrics['within_1'],
        'mae_improvement': tier1_metrics['mae'] - routing_metrics['mae'],
        'r_improvement': routing_metrics['pearson_r'] - tier1_metrics['pearson_r']
    }
    results['improvement_vs_tier1'] = improvement_vs_tier1
    
    logger.info(f"Kappa improvement: {improvement_vs_tier1['kappa_improvement']:+.3f}")
    logger.info(f"Within-1 improvement: {improvement_vs_tier1['within1_improvement']:+.1%}")
    logger.info(f"MAE improvement: {improvement_vs_tier1['mae_improvement']:+.3f}")
    
    # 5. Analyze which cases benefit from escalation
    logger.info("\n" + "="*60)
    logger.info("ROUTING ANALYSIS")
    logger.info("="*60)
    
    df['tier1_score'] = tier1_scores
    df['tier3_score'] = tier3_scores
    df['routing_score'] = routing_scores
    df['routing_tier'] = routing_tiers
    
    # Cases where Tier 3 was better than Tier 1
    tier1_errors = np.abs(df['expert_score'] - df['tier1_score'])
    tier3_errors = np.abs(df['expert_score'] - df['tier3_score'])
    better_with_tier3 = tier3_errors < tier1_errors
    
    logger.info(f"Cases where Tier 3 was better: {np.sum(better_with_tier3)}/{len(df)} ({100*np.sum(better_with_tier3)/len(df):.1f}%)")
    
    # Cases that were escalated
    escalated = np.array(routing_tiers) > 1
    logger.info(f"Cases escalated: {np.sum(escalated)}/{len(df)} ({100*np.sum(escalated)/len(df):.1f}%)")
    
    # Of escalated cases, how many improved?
    if np.sum(escalated) > 0:
        escalated_improved = np.sum(
            (escalated) & 
            (np.abs(df['routing_score'] - df['expert_score']) < np.abs(df['tier1_score'] - df['expert_score']))
        )
        logger.info(f"Escalated cases that improved: {escalated_improved}/{np.sum(escalated)} ({100*escalated_improved/np.sum(escalated):.1f}%)")
    
    results['routing_analysis'] = {
        'cases_better_with_tier3': int(np.sum(better_with_tier3)),
        'cases_escalated': int(np.sum(escalated)),
        'escalated_improved': int(escalated_improved) if np.sum(escalated) > 0 else 0
    }
    
    return results


def calibrate_thresholds(
    expert_labeled_path: Path,
    threshold_range: Tuple[float, float] = (0.3, 0.9),
    step: float = 0.05
) -> Dict:
    """
    Calibrate routing thresholds to optimize quality.
    
    Tests different confidence thresholds and finds optimal values.
    """
    logger.info("Calibrating routing thresholds...")
    df = pd.read_parquet(expert_labeled_path)
    df = df[df['expert_score'] > 0].copy()
    
    # Score with Tier 1
    tier1_scorer = AnthroScoreLLM(model="gpt-4.1-nano")
    tier1_results = tier1_scorer.score_batch(df['body'].tolist(), rate_limit_delay=0.1)
    tier1_scores = [r.score for r in tier1_results]
    tier1_reasonings = [r.reasoning.lower() for r in tier1_results]
    
    # Score with Tier 3 (expert)
    tier3_scorer = AnthroScoreLLM(model="gpt-5-mini")
    tier3_results = tier3_scorer.score_batch(df['body'].tolist(), rate_limit_delay=0.2)
    tier3_scores = [r.score for r in tier3_results]
    
    best_config = None
    best_kappa = -1
    
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    
    for tier2_thresh in thresholds:
        for tier3_thresh in thresholds:
            if tier3_thresh >= tier2_thresh:
                continue  # Tier 3 should be lower threshold
            
            # Simulate routing decisions
            routing_scores = []
            for i, (t1_score, t1_reason, t3_score) in enumerate(zip(tier1_scores, tier1_reasonings, tier3_scores)):
                # Estimate confidence
                low_conf_phrases = ['unclear', 'ambiguous', 'mixed', 'could be', 'might be']
                has_low_conf = any(phrase in t1_reason for phrase in low_conf_phrases)
                estimated_conf = 0.8 if not has_low_conf else 0.5
                
                # Route
                if estimated_conf < tier3_thresh:
                    routing_scores.append(t3_score)  # Tier 3
                elif estimated_conf < tier2_thresh:
                    routing_scores.append(t3_score)  # Tier 2 (use Tier 3 for now)
                else:
                    routing_scores.append(t1_score)  # Tier 1
            
            # Evaluate
            kappa = cohen_kappa_score(df['expert_score'], routing_scores, weights='quadratic')
            
            if kappa > best_kappa:
                best_kappa = kappa
                best_config = {
                    'tier2_threshold': tier2_thresh,
                    'tier3_threshold': tier3_thresh,
                    'kappa': kappa
                }
    
    logger.info(f"Best configuration: Tier2={best_config['tier2_threshold']:.2f}, Tier3={best_config['tier3_threshold']:.2f}, Kappa={best_config['kappa']:.3f}")
    
    return best_config


def main():
    """Run full validation pipeline."""
    exp_dir = Path(__file__).parent
    
    expert_path = exp_dir / "test_set_expert_labeled.parquet"
    
    if not expert_path.exists():
        logger.error(f"Expert-labeled test set not found: {expert_path}")
        logger.info("Run label_with_expert.py first!")
        return
    
    # Validate routing quality
    results = validate_routing_quality(expert_path, test_routing=True)
    
    # Save results
    output_path = exp_dir / "routing_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved validation results to: {output_path}")
    
    # Calibrate thresholds
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD CALIBRATION")
    logger.info("="*60)
    best_config = calibrate_thresholds(expert_path)
    
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    
    if results['smart_routing']['cohen_kappa'] > results['tier1_only']['cohen_kappa']:
        logger.info("✅ Smart routing improves quality!")
        logger.info(f"   Improvement: {results['improvement_vs_tier1']['kappa_improvement']:+.3f} kappa")
    else:
        logger.warning("⚠️  Smart routing does not improve quality - may need tuning")
    
    logger.info(f"\nRecommended thresholds:")
    logger.info(f"  CONFIDENCE_THRESHOLD_TIER2 = {best_config['tier2_threshold']:.2f}")
    logger.info(f"  CONFIDENCE_THRESHOLD_TIER3 = {best_config['tier3_threshold']:.2f}")


if __name__ == "__main__":
    main()
