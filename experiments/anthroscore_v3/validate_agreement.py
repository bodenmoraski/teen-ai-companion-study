"""
Validate agreement between cheap LLM (GPT-4.1-nano) and expert labels.

Computes various agreement metrics and determines if the cheap model
is reliable enough for full dataset processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import json

from anthroscore_llm import AnthroScoreLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_agreement_metrics(expert_scores: np.ndarray, llm_scores: np.ndarray) -> dict:
    """
    Compute various agreement metrics between expert and LLM scores.
    
    Args:
        expert_scores: Array of expert (gold standard) scores
        llm_scores: Array of LLM predictions
        
    Returns:
        Dictionary of metrics
    """
    # Filter out invalid scores
    valid_mask = (expert_scores > 0) & (llm_scores > 0)
    expert = expert_scores[valid_mask]
    llm = llm_scores[valid_mask]
    
    if len(expert) < 10:
        return {"error": "Not enough valid pairs"}
    
    metrics = {}
    
    # Cohen's Kappa (chance-corrected agreement)
    # Weighted kappa for ordinal data
    metrics['cohen_kappa'] = cohen_kappa_score(expert, llm, weights='quadratic')
    metrics['cohen_kappa_linear'] = cohen_kappa_score(expert, llm, weights='linear')
    
    # Exact accuracy
    metrics['accuracy'] = accuracy_score(expert, llm)
    
    # Within-1 accuracy (predictions within 1 point)
    within_1 = np.mean(np.abs(expert - llm) <= 1)
    metrics['accuracy_within_1'] = within_1
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(expert, llm)
    
    # Correlation coefficients
    metrics['pearson_r'], metrics['pearson_p'] = pearsonr(expert, llm)
    metrics['spearman_r'], metrics['spearman_p'] = spearmanr(expert, llm)
    
    # Score distribution comparison
    metrics['expert_mean'] = np.mean(expert)
    metrics['llm_mean'] = np.mean(llm)
    metrics['expert_std'] = np.std(expert)
    metrics['llm_std'] = np.std(llm)
    
    # Confusion matrix (simplified: low/mid/high)
    def to_category(s):
        if s <= 2: return 'low'
        elif s <= 3: return 'mid'
        else: return 'high'
    
    expert_cat = np.array([to_category(s) for s in expert])
    llm_cat = np.array([to_category(s) for s in llm])
    metrics['category_accuracy'] = np.mean(expert_cat == llm_cat)
    
    return metrics


def validate_cheap_model(
    expert_labeled_path: Path,
    model: str = "gpt-4.1-nano",
    output_path: Path = None
) -> dict:
    """
    Run cheap model on test set and compare to expert labels.
    
    Args:
        expert_labeled_path: Path to expert-labeled test set
        model: Cheap model to validate
        output_path: Optional path to save results
        
    Returns:
        Dictionary with metrics and pass/fail decision
    """
    logger.info(f"Loading expert-labeled test set: {expert_labeled_path}")
    df = pd.read_parquet(expert_labeled_path)
    
    # Filter to valid expert labels
    df = df[df['expert_score'] > 0].copy()
    logger.info(f"Found {len(df)} comments with valid expert labels")
    
    # Score with cheap model
    scorer = AnthroScoreLLM(model=model)
    results = scorer.score_batch(df['body'].tolist(), rate_limit_delay=0.1)
    
    # Add to dataframe
    df['llm_score'] = [r.score for r in results]
    df['llm_reasoning'] = [r.reasoning for r in results]
    df['llm_time_ms'] = [r.processing_time_ms for r in results]
    
    # Compute metrics
    metrics = compute_agreement_metrics(
        df['expert_score'].values,
        df['llm_score'].values
    )
    
    # Decision thresholds
    KAPPA_THRESHOLD = 0.6  # Substantial agreement
    WITHIN_1_THRESHOLD = 0.85  # 85% within 1 point
    
    metrics['passes_kappa'] = metrics.get('cohen_kappa', 0) >= KAPPA_THRESHOLD
    metrics['passes_within_1'] = metrics.get('accuracy_within_1', 0) >= WITHIN_1_THRESHOLD
    metrics['recommendation'] = "APPROVED" if (metrics['passes_kappa'] and metrics['passes_within_1']) else "NEEDS REVIEW"
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Model: {model}")
    logger.info(f"Test set size: {len(df)}")
    logger.info("-"*60)
    logger.info(f"Cohen's Kappa (quadratic): {metrics.get('cohen_kappa', 0):.3f}")
    logger.info(f"  → {'PASS' if metrics['passes_kappa'] else 'FAIL'} (threshold: {KAPPA_THRESHOLD})")
    logger.info(f"Exact Accuracy: {metrics.get('accuracy', 0):.1%}")
    logger.info(f"Within-1 Accuracy: {metrics.get('accuracy_within_1', 0):.1%}")
    logger.info(f"  → {'PASS' if metrics['passes_within_1'] else 'FAIL'} (threshold: {WITHIN_1_THRESHOLD:.0%})")
    logger.info(f"Mean Absolute Error: {metrics.get('mae', 0):.2f}")
    logger.info(f"Pearson r: {metrics.get('pearson_r', 0):.3f} (p={metrics.get('pearson_p', 1):.4f})")
    logger.info(f"Spearman r: {metrics.get('spearman_r', 0):.3f} (p={metrics.get('spearman_p', 1):.4f})")
    logger.info("-"*60)
    logger.info(f"RECOMMENDATION: {metrics['recommendation']}")
    logger.info("="*60)
    
    # Save validation results
    if output_path is None:
        output_path = Path(__file__).parent / "validation_results.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        return obj
    
    metrics_serializable = convert_to_python(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info(f"\nSaved metrics to: {output_path}")
    
    # Save labeled dataset
    df_output_path = Path(__file__).parent / "test_set_fully_labeled.parquet"
    df.to_parquet(df_output_path, index=False)
    logger.info(f"Saved fully labeled test set to: {df_output_path}")
    
    # Show disagreements
    logger.info("\n" + "="*60)
    logger.info("NOTABLE DISAGREEMENTS (diff >= 2)")
    logger.info("="*60)
    
    df['diff'] = np.abs(df['expert_score'] - df['llm_score'])
    disagreements = df[df['diff'] >= 2].sort_values('diff', ascending=False)
    
    for i, row in disagreements.head(5).iterrows():
        print(f"\nText: {row['body'][:100]}...")
        print(f"  Expert: {row['expert_score']}/5 | LLM: {row['llm_score']}/5")
        print(f"  Expert reasoning: {row['expert_reasoning'][:100]}...")
        print(f"  LLM reasoning: {row['llm_reasoning'][:100]}...")
    
    return metrics


def compare_to_mlm_scores(fully_labeled_path: Path) -> dict:
    """
    Compare LLM scores to original MLM-based AnthroScore V2.
    
    Args:
        fully_labeled_path: Path to fully labeled test set
        
    Returns:
        Dictionary with comparison metrics
    """
    df = pd.read_parquet(fully_labeled_path)
    
    if 'anthroscore_mean' not in df.columns:
        logger.warning("No MLM anthroscores in test set for comparison")
        return {}
    
    # Filter valid
    df = df[(df['expert_score'] > 0) & (df['llm_score'] > 0) & df['anthroscore_mean'].notna()].copy()
    
    if len(df) < 10:
        return {"error": "Not enough data for comparison"}
    
    # Correlate LLM scores with expert
    llm_expert_r, _ = pearsonr(df['llm_score'], df['expert_score'])
    
    # Correlate MLM scores with expert
    # First normalize MLM to 1-5 scale
    mlm_min, mlm_max = df['anthroscore_mean'].min(), df['anthroscore_mean'].max()
    df['anthroscore_normalized'] = 1 + 4 * (df['anthroscore_mean'] - mlm_min) / (mlm_max - mlm_min + 1e-10)
    
    mlm_expert_r, _ = pearsonr(df['anthroscore_normalized'], df['expert_score'])
    
    comparison = {
        'llm_expert_correlation': llm_expert_r,
        'mlm_expert_correlation': mlm_expert_r,
        'llm_improvement': llm_expert_r - mlm_expert_r,
        'winner': 'LLM' if llm_expert_r > mlm_expert_r else 'MLM'
    }
    
    logger.info("\n" + "="*60)
    logger.info("LLM vs MLM COMPARISON")
    logger.info("="*60)
    logger.info(f"LLM-Expert correlation: {llm_expert_r:.3f}")
    logger.info(f"MLM-Expert correlation: {mlm_expert_r:.3f}")
    logger.info(f"Improvement: {comparison['llm_improvement']:+.3f}")
    logger.info(f"Winner: {comparison['winner']}")
    
    return comparison


def main():
    """Run full validation pipeline."""
    exp_dir = Path(__file__).parent
    
    expert_path = exp_dir / "test_set_expert_labeled.parquet"
    
    if not expert_path.exists():
        logger.error(f"Expert-labeled test set not found: {expert_path}")
        logger.info("Run label_with_expert.py first!")
        return
    
    # Validate cheap model
    metrics = validate_cheap_model(expert_path, model="gpt-4.1-nano")
    
    # Compare to MLM
    fully_labeled_path = exp_dir / "test_set_fully_labeled.parquet"
    if fully_labeled_path.exists():
        comparison = compare_to_mlm_scores(fully_labeled_path)
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if metrics.get('recommendation') == "APPROVED":
        print("""
✅ GPT-4.1-nano APPROVED for production use!

The cheap model shows substantial agreement with expert labels:
- Cohen's Kappa is above threshold
- Within-1 accuracy is above threshold

NEXT STEPS:
1. Run run_full_analysis.py to process the entire dataset
2. Compare results to existing MLM-based scores
3. Use LLM scores for research analyses
        """)
    else:
        print("""
⚠️  GPT-4.1-nano NEEDS REVIEW

The model did not meet all validation thresholds.

OPTIONS:
1. Try a more capable model (gpt-4.1-mini, gpt-4o-mini)
2. Improve the prompt with more examples
3. Use hybrid approach: LLM for ambiguous cases only
4. Accept lower threshold if results are interpretable
        """)


if __name__ == "__main__":
    main()
