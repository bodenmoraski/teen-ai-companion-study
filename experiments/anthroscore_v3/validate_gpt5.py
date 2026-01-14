"""
Validate GPT-5-nano against GPT-5-mini expert labels.

This script:
1. Loads the expert-labeled test set (labeled by GPT-5-mini)
2. Scores the same texts with GPT-5-nano
3. Computes agreement metrics
4. Compares to the old MLM baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anthroscore_gpt5 import AnthroScoreGPT5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_scorer(
    expert_labeled_path: Path,
    model: str = "gpt-5-nano",
    output_path: Path = None
) -> dict:
    """
    Validate a scorer against expert labels.
    
    Args:
        expert_labeled_path: Path to expert-labeled parquet file
        model: Model to validate
        output_path: Where to save results
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Loading expert labels from: {expert_labeled_path}")
    df = pd.read_parquet(expert_labeled_path)
    
    # Filter to valid expert labels
    df_valid = df[df['expert_score'] > 0].copy()
    logger.info(f"Found {len(df_valid)} valid expert labels (out of {len(df)})")
    
    if len(df_valid) < 20:
        logger.error("Not enough valid labels for validation")
        return {"error": "Insufficient labels"}
    
    # Score with the model being validated
    logger.info(f"Scoring {len(df_valid)} texts with {model}...")
    scorer = AnthroScoreGPT5(model=model)
    
    results = scorer.score_batch(df_valid['body'].tolist(), progress_interval=25)
    
    # Extract scores
    predicted_scores = [r.score for r in results]
    expert_scores = df_valid['expert_score'].values
    
    # Filter out failed predictions
    valid_mask = np.array([r.success for r in results])
    pred_valid = np.array(predicted_scores)[valid_mask]
    expert_valid = expert_scores[valid_mask]
    
    logger.info(f"Valid predictions: {len(pred_valid)}/{len(results)}")
    
    if len(pred_valid) < 20:
        logger.error("Too many prediction failures")
        return {"error": "Too many failures"}
    
    # Compute metrics
    metrics = {
        "model": model,
        "n_total": len(df),
        "n_expert_valid": len(df_valid),
        "n_predictions_valid": len(pred_valid),
        
        # Agreement metrics
        "cohen_kappa_quadratic": float(cohen_kappa_score(expert_valid, pred_valid, weights='quadratic')),
        "cohen_kappa_linear": float(cohen_kappa_score(expert_valid, pred_valid, weights='linear')),
        "exact_accuracy": float(accuracy_score(expert_valid, pred_valid)),
        "within_1_accuracy": float(np.mean(np.abs(expert_valid - pred_valid) <= 1)),
        "mae": float(mean_absolute_error(expert_valid, pred_valid)),
        
        # Correlation
        "pearson_r": float(pearsonr(expert_valid, pred_valid)[0]),
        "pearson_p": float(pearsonr(expert_valid, pred_valid)[1]),
        "spearman_r": float(spearmanr(expert_valid, pred_valid)[0]),
        "spearman_p": float(spearmanr(expert_valid, pred_valid)[1]),
        
        # Score distributions
        "expert_mean": float(np.mean(expert_valid)),
        "expert_std": float(np.std(expert_valid)),
        "predicted_mean": float(np.mean(pred_valid)),
        "predicted_std": float(np.std(pred_valid)),
    }
    
    # Quality thresholds
    metrics["passes_kappa_threshold"] = metrics["cohen_kappa_quadratic"] >= 0.60
    metrics["passes_within_1_threshold"] = metrics["within_1_accuracy"] >= 0.95
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info(f"VALIDATION RESULTS: {model} vs GPT-5-mini Expert")
    logger.info("="*70)
    
    kappa_quality = "Substantial" if metrics["cohen_kappa_quadratic"] >= 0.60 else "Moderate"
    logger.info(f"\n  Cohen's Kappa (quadratic): {metrics['cohen_kappa_quadratic']:.3f} ({kappa_quality})")
    logger.info(f"  Within-1 Accuracy: {metrics['within_1_accuracy']:.1%}")
    logger.info(f"  Exact Accuracy: {metrics['exact_accuracy']:.1%}")
    logger.info(f"  Pearson r: {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.4f})")
    logger.info(f"  MAE: {metrics['mae']:.2f}")
    
    logger.info(f"\n  Expert mean: {metrics['expert_mean']:.2f} (SD={metrics['expert_std']:.2f})")
    logger.info(f"  Predicted mean: {metrics['predicted_mean']:.2f} (SD={metrics['predicted_std']:.2f})")
    
    if metrics["passes_kappa_threshold"] and metrics["passes_within_1_threshold"]:
        logger.info("\n  ✅ PASSES ALL THRESHOLDS - Ready for production!")
    else:
        logger.info("\n  ⚠️ Some thresholds not met - consider prompt improvements")
    
    # Save results
    if output_path is None:
        output_path = Path(__file__).parent / f"validation_{model.replace('-', '_')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\n  Saved to: {output_path}")
    
    # Also save predictions to the dataframe
    df_valid['predicted_score'] = predicted_scores
    df_valid['predicted_reasoning'] = [r.reasoning for r in results]
    df_valid['prediction_success'] = [r.success for r in results]
    
    labeled_output = expert_labeled_path.parent / f"test_set_with_{model.replace('-', '_')}.parquet"
    df_valid.to_parquet(labeled_output, index=False)
    logger.info(f"  Saved predictions to: {labeled_output}")
    
    return metrics


def compare_to_mlm(df: pd.DataFrame) -> dict:
    """
    Compare LLM predictions to MLM baseline.
    
    Requires df to have 'expert_score', 'predicted_score', and 'anthroscore_mean' columns.
    """
    valid = df[
        (df['expert_score'] > 0) & 
        (df['predicted_score'] > 0) &
        (df['anthroscore_mean'].notna())
    ].copy()
    
    if len(valid) < 20:
        return {"error": "Insufficient data"}
    
    expert = valid['expert_score'].values
    llm = valid['predicted_score'].values
    mlm = valid['anthroscore_mean'].values
    
    # Correlations with expert
    llm_r = pearsonr(expert, llm)[0]
    mlm_r = pearsonr(expert, mlm)[0]
    
    # Head-to-head comparison
    llm_errors = np.abs(expert - llm)
    mlm_errors = np.abs(expert - mlm)  # Need to convert MLM to 1-5 scale first
    
    # Convert MLM to 1-5 scale for fair comparison
    # MLM is typically in range -5 to 5, map to 1-5
    mlm_scaled = np.clip((mlm + 5) / 2, 1, 5)
    mlm_errors_scaled = np.abs(expert - mlm_scaled)
    
    llm_wins = np.sum(llm_errors < mlm_errors_scaled)
    mlm_wins = np.sum(mlm_errors_scaled < llm_errors)
    ties = np.sum(llm_errors == mlm_errors_scaled)
    
    return {
        "llm_expert_r": float(llm_r),
        "mlm_expert_r": float(mlm_r),
        "llm_wins": int(llm_wins),
        "mlm_wins": int(mlm_wins),
        "ties": int(ties),
        "n_samples": len(valid)
    }


def main():
    """Run validation."""
    exp_dir = Path(__file__).parent
    expert_path = exp_dir / "test_set_expert_labeled.parquet"
    
    if not expert_path.exists():
        logger.error(f"Expert labels not found: {expert_path}")
        logger.info("Run label_with_expert.py first!")
        return
    
    # Validate GPT-5-nano against GPT-5-mini expert labels
    metrics = validate_scorer(expert_path, model="gpt-5-nano")
    
    if "error" not in metrics:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\nGPT-5-nano vs GPT-5-mini Expert:")
        print(f"  Kappa: {metrics['cohen_kappa_quadratic']:.3f}")
        print(f"  Within-1: {metrics['within_1_accuracy']:.1%}")
        print(f"  Pearson r: {metrics['pearson_r']:.3f}")
        
        if metrics["passes_kappa_threshold"]:
            print("\n✅ GPT-5-nano is validated for production use!")
        else:
            print("\n⚠️ Consider using GPT-5-mini for better accuracy")


if __name__ == "__main__":
    main()
