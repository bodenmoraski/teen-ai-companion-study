"""
Validation of demographic classification methods using self-declarations as ground truth.

This validates:
1. Community embeddings accuracy (vs. self-declarations)
2. LLM accuracy (vs. self-declarations)
3. Inter-method agreement
4. Confusion matrices
5. Precision, recall, F1 for each method
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score
)
from scipy.stats import chi2_contingency

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validate_demographics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_classification_metrics(y_true, y_pred, labels=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label order for metrics
        
    Returns:
        Dictionary with metrics
    """
    # Filter out NaN values
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    metrics = {
        'n_samples': len(y_true_clean),
        'accuracy': accuracy_score(y_true_clean, y_pred_clean),
        'cohen_kappa': cohen_kappa_score(y_true_clean, y_pred_clean),
    }
    
    # Per-class metrics (if labels provided)
    if labels is not None:
        try:
            metrics['precision_macro'] = precision_score(
                y_true_clean, y_pred_clean, labels=labels, average='macro', zero_division=0
            )
            metrics['recall_macro'] = recall_score(
                y_true_clean, y_pred_clean, labels=labels, average='macro', zero_division=0
            )
            metrics['f1_macro'] = f1_score(
                y_true_clean, y_pred_clean, labels=labels, average='macro', zero_division=0
            )
            metrics['precision_weighted'] = precision_score(
                y_true_clean, y_pred_clean, labels=labels, average='weighted', zero_division=0
            )
            metrics['recall_weighted'] = recall_score(
                y_true_clean, y_pred_clean, labels=labels, average='weighted', zero_division=0
            )
            metrics['f1_weighted'] = f1_score(
                y_true_clean, y_pred_clean, labels=labels, average='weighted', zero_division=0
            )
        except:
            pass
    
    return metrics


def validate_age_classification(demo_df: pd.DataFrame, output_path: Path):
    """
    Validate age classification methods using self-declarations as ground truth.
    
    Args:
        demo_df: Demographics dataframe
        output_path: Path to save validation results
    """
    logger.info("Validating age classification methods")
    
    # Filter to users with self-declarations (ground truth)
    ground_truth = demo_df[demo_df['age_bucket_self_declared'].notna()].copy()
    logger.info(f"Users with self-declared age (ground truth): {len(ground_truth):,}")
    
    if len(ground_truth) == 0:
        logger.warning("No self-declarations found - cannot validate")
        return
    
    age_buckets = ['13-18', '19-25', '26-40', '41-60', '61-80']
    
    results = []
    
    # Validate community embeddings
    if 'age_bucket_community' in ground_truth.columns:
        comm_metrics = calculate_classification_metrics(
            ground_truth['age_bucket_self_declared'],
            ground_truth['age_bucket_community'],
            labels=age_buckets
        )
        if comm_metrics:
            comm_metrics['method'] = 'Community Embeddings'
            results.append(comm_metrics)
            logger.info(f"Community Embeddings: Accuracy={comm_metrics['accuracy']:.3f}, Kappa={comm_metrics['cohen_kappa']:.3f}, N={comm_metrics['n_samples']}")
    
    # Validate LLM
    if 'age_bucket_llm' in ground_truth.columns:
        llm_metrics = calculate_classification_metrics(
            ground_truth['age_bucket_self_declared'],
            ground_truth['age_bucket_llm'],
            labels=age_buckets
        )
        if llm_metrics:
            llm_metrics['method'] = 'LLM'
            results.append(llm_metrics)
            logger.info(f"LLM: Accuracy={llm_metrics['accuracy']:.3f}, Kappa={llm_metrics['cohen_kappa']:.3f}, N={llm_metrics['n_samples']}")
    
    # Inter-method agreement (community vs LLM)
    if 'age_bucket_community' in ground_truth.columns and 'age_bucket_llm' in ground_truth.columns:
        comm_llm_mask = ground_truth['age_bucket_community'].notna() & ground_truth['age_bucket_llm'].notna()
        if comm_llm_mask.sum() > 0:
            comm_llm_kappa = cohen_kappa_score(
                ground_truth.loc[comm_llm_mask, 'age_bucket_community'],
                ground_truth.loc[comm_llm_mask, 'age_bucket_llm']
            )
            logger.info(f"Community vs LLM agreement: Kappa={comm_llm_kappa:.3f}, N={comm_llm_mask.sum()}")
            results.append({
                'method': 'Community vs LLM Agreement',
                'cohen_kappa': comm_llm_kappa,
                'n_samples': comm_llm_mask.sum()
            })
    
    # Generate confusion matrices
    confusion_matrices = {}
    
    if 'age_bucket_community' in ground_truth.columns:
        comm_mask = ground_truth['age_bucket_community'].notna()
        if comm_mask.sum() > 0:
            cm_comm = confusion_matrix(
                ground_truth.loc[comm_mask, 'age_bucket_self_declared'],
                ground_truth.loc[comm_mask, 'age_bucket_community'],
                labels=age_buckets
            )
            confusion_matrices['Community Embeddings'] = cm_comm
    
    if 'age_bucket_llm' in ground_truth.columns:
        llm_mask = ground_truth['age_bucket_llm'].notna()
        if llm_mask.sum() > 0:
            cm_llm = confusion_matrix(
                ground_truth.loc[llm_mask, 'age_bucket_self_declared'],
                ground_truth.loc[llm_mask, 'age_bucket_llm'],
                labels=age_buckets
            )
            confusion_matrices['LLM'] = cm_llm
    
    # Write results
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Age Classification Validation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Ground Truth: Self-declared age\n")
        f.write(f"Total users with ground truth: {len(ground_truth):,}\n\n")
        
        f.write("Method Performance:\n")
        f.write("-" * 70 + "\n")
        for result in results:
            f.write(f"\n{result.get('method', 'Unknown')}:\n")
            for key, value in result.items():
                if key != 'method':
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        f.write("\n\nConfusion Matrices:\n")
        f.write("-" * 70 + "\n")
        for method_name, cm in confusion_matrices.items():
            f.write(f"\n{method_name}:\n")
            f.write("Rows: True (self-declared), Columns: Predicted\n")
            f.write("Labels: " + ", ".join(age_buckets) + "\n\n")
            cm_df = pd.DataFrame(cm, index=age_buckets, columns=age_buckets)
            f.write(cm_df.to_string())
            f.write("\n\n")
            
            # Per-class accuracy
            f.write("Per-class accuracy:\n")
            for i, bucket in enumerate(age_buckets):
                if cm[i, :].sum() > 0:
                    acc = cm[i, i] / cm[i, :].sum()
                    f.write(f"  {bucket}: {acc:.3f} ({cm[i, i]}/{cm[i, :].sum()})\n")
            f.write("\n")
    
    logger.info(f"Validation results saved to {output_path}")


def validate_gender_classification(demo_df: pd.DataFrame, output_path: Path):
    """
    Validate gender classification methods using self-declarations as ground truth.
    
    Args:
        demo_df: Demographics dataframe
        output_path: Path to save validation results
    """
    logger.info("Validating gender classification methods")
    
    # Filter to users with self-declarations (ground truth)
    ground_truth = demo_df[demo_df['gender_self_declared'].notna()].copy()
    logger.info(f"Users with self-declared gender (ground truth): {len(ground_truth):,}")
    
    if len(ground_truth) == 0:
        logger.warning("No self-declarations found - cannot validate")
        return
    
    gender_labels = ['male', 'female', 'nonbinary']
    
    results = []
    
    # Validate community embeddings
    if 'gender_community' in ground_truth.columns:
        comm_mask = ground_truth['gender_community'].notna()
        if comm_mask.sum() > 0:
            comm_metrics = calculate_classification_metrics(
                ground_truth.loc[comm_mask, 'gender_self_declared'],
                ground_truth.loc[comm_mask, 'gender_community'],
                labels=gender_labels
            )
            if comm_metrics:
                comm_metrics['method'] = 'Community Embeddings'
                results.append(comm_metrics)
                logger.info(f"Community Embeddings: Accuracy={comm_metrics['accuracy']:.3f}, Kappa={comm_metrics['cohen_kappa']:.3f}, N={comm_metrics['n_samples']}")
    
    # Generate confusion matrix
    confusion_matrices = {}
    
    if 'gender_community' in ground_truth.columns:
        comm_mask = ground_truth['gender_community'].notna()
        if comm_mask.sum() > 0:
            cm_comm = confusion_matrix(
                ground_truth.loc[comm_mask, 'gender_self_declared'],
                ground_truth.loc[comm_mask, 'gender_community'],
                labels=gender_labels
            )
            confusion_matrices['Community Embeddings'] = cm_comm
    
    # Write results
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Gender Classification Validation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Ground Truth: Self-declared gender\n")
        f.write(f"Total users with ground truth: {len(ground_truth):,}\n\n")
        
        f.write("Method Performance:\n")
        f.write("-" * 70 + "\n")
        for result in results:
            f.write(f"\n{result.get('method', 'Unknown')}:\n")
            for key, value in result.items():
                if key != 'method':
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        f.write("\n\nConfusion Matrices:\n")
        f.write("-" * 70 + "\n")
        for method_name, cm in confusion_matrices.items():
            f.write(f"\n{method_name}:\n")
            f.write("Rows: True (self-declared), Columns: Predicted\n")
            f.write("Labels: " + ", ".join(gender_labels) + "\n\n")
            cm_df = pd.DataFrame(cm, index=gender_labels, columns=gender_labels)
            f.write(cm_df.to_string())
            f.write("\n\n")
    
    logger.info(f"Validation results saved to {output_path}")


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("Demographic Classification Validation")
    logger.info("Using Self-Declarations as Ground Truth")
    logger.info("=" * 70)
    
    # Load demographics
    demo_path = Path("data/features/demographics.parquet")
    if not demo_path.exists():
        logger.error(f"Demographics not found: {demo_path}")
        return
    
    demo_df = pd.read_parquet(demo_path)
    logger.info(f"Loaded demographics: {len(demo_df):,} users")
    
    # Create output directory
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate age classification
    logger.info("\n" + "-" * 70)
    logger.info("Validating Age Classification")
    logger.info("-" * 70)
    validate_age_classification(
        demo_df,
        output_dir / "age_classification_validation.txt"
    )
    
    # Validate gender classification
    logger.info("\n" + "-" * 70)
    logger.info("Validating Gender Classification")
    logger.info("-" * 70)
    validate_gender_classification(
        demo_df,
        output_dir / "gender_classification_validation.txt"
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("Validation Complete!")
    logger.info("=" * 70)
    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()

