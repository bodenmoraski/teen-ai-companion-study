#!/usr/bin/env python3
"""
Inter-Rater Reliability Calculator for AnthroScore Validation
==============================================================

Calculates:
1. Krippendorff's Alpha (primary metric)
2. Cohen's Kappa (pairwise)
3. Intraclass Correlation Coefficient (ICC)
4. Pearson/Spearman correlations with computed AnthroScore

Usage:
    python scripts/calculate_irr.py --annotations Data/annotations/annotation_sheet_completed.csv

Output: Data/annotations/irr_report.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
ANNOTATIONS_DIR = Path("Data/annotations")


def load_annotations(filepath: Path) -> pd.DataFrame:
    """Load completed annotations from CSV."""
    logger.info(f"Loading annotations from {filepath}")
    df = pd.read_csv(filepath)
    
    # Identify annotator columns
    annotator_cols = [col for col in df.columns if 'score' in col.lower() and 'annotator' in col.lower()]
    logger.info(f"Found {len(annotator_cols)} annotator columns: {annotator_cols}")
    
    return df, annotator_cols


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth AnthroScores."""
    gt_path = ANNOTATIONS_DIR / "ground_truth_DO_NOT_SHARE.csv"
    if gt_path.exists():
        return pd.read_csv(gt_path)
    else:
        logger.warning("Ground truth file not found")
        return None


def calculate_krippendorff_alpha(ratings: np.ndarray, level: str = 'ordinal') -> float:
    """
    Calculate Krippendorff's Alpha for inter-rater reliability.
    
    Args:
        ratings: 2D array where rows are items and columns are raters
        level: Type of data ('nominal', 'ordinal', 'interval', 'ratio')
    
    Returns:
        Krippendorff's alpha coefficient
    """
    # Remove items where all raters gave the same value or all missing
    n_items, n_raters = ratings.shape
    
    # Convert to float and replace empty/invalid with NaN
    ratings = ratings.astype(float)
    
    # Get all unique values
    values = np.unique(ratings[~np.isnan(ratings)])
    
    if len(values) <= 1:
        logger.warning("All ratings are identical - cannot compute alpha")
        return 1.0
    
    # Count coincidence matrix
    # For ordinal data, use ordinal metric
    def ordinal_metric(v1, v2, values):
        """Ordinal difference metric."""
        i1 = np.where(values == v1)[0][0]
        i2 = np.where(values == v2)[0][0]
        # Sum of counts between the two values
        return sum(1 for k in range(min(i1, i2), max(i1, i2) + 1)) ** 2 - 1
    
    def interval_metric(v1, v2, values):
        """Interval/ratio difference metric."""
        return (v1 - v2) ** 2
    
    if level == 'ordinal':
        metric_func = ordinal_metric
    else:
        metric_func = interval_metric
    
    # Calculate observed disagreement
    Do = 0
    n_pairs = 0
    
    for item in range(n_items):
        item_ratings = ratings[item, :]
        valid_ratings = item_ratings[~np.isnan(item_ratings)]
        
        if len(valid_ratings) < 2:
            continue
        
        # All pairs within this item
        for i in range(len(valid_ratings)):
            for j in range(i + 1, len(valid_ratings)):
                Do += metric_func(valid_ratings[i], valid_ratings[j], values)
                n_pairs += 1
    
    if n_pairs == 0:
        logger.warning("No valid pairs for comparison")
        return np.nan
    
    Do = Do / n_pairs
    
    # Calculate expected disagreement (from marginal distribution)
    all_ratings = ratings[~np.isnan(ratings)]
    n_total = len(all_ratings)
    
    De = 0
    for v1 in values:
        for v2 in values:
            if v1 != v2:
                p1 = np.sum(all_ratings == v1) / n_total
                p2 = np.sum(all_ratings == v2) / n_total
                De += p1 * p2 * metric_func(v1, v2, values)
    
    if De == 0:
        logger.warning("Expected disagreement is zero")
        return 1.0
    
    alpha = 1 - (Do / De)
    
    return alpha


def calculate_cohens_kappa(rater1: np.ndarray, rater2: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa for two raters.
    
    Args:
        rater1, rater2: Arrays of ratings (same length)
    
    Returns:
        Cohen's kappa coefficient
    """
    # Remove pairs with missing values
    valid_mask = ~(np.isnan(rater1) | np.isnan(rater2))
    r1 = rater1[valid_mask].astype(int)
    r2 = rater2[valid_mask].astype(int)
    
    if len(r1) == 0:
        return np.nan
    
    # Get all possible categories
    categories = np.unique(np.concatenate([r1, r2]))
    n = len(r1)
    
    # Calculate observed agreement
    po = np.sum(r1 == r2) / n
    
    # Calculate expected agreement
    pe = 0
    for cat in categories:
        pe += (np.sum(r1 == cat) / n) * (np.sum(r2 == cat) / n)
    
    if pe == 1:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)
    
    return kappa


def calculate_weighted_kappa(rater1: np.ndarray, rater2: np.ndarray, weights: str = 'quadratic') -> float:
    """
    Calculate weighted Cohen's Kappa for ordinal data.
    
    Args:
        rater1, rater2: Arrays of ratings
        weights: 'linear' or 'quadratic'
    
    Returns:
        Weighted kappa coefficient
    """
    # Remove pairs with missing values
    valid_mask = ~(np.isnan(rater1) | np.isnan(rater2))
    r1 = rater1[valid_mask].astype(int)
    r2 = rater2[valid_mask].astype(int)
    
    if len(r1) == 0:
        return np.nan
    
    categories = np.sort(np.unique(np.concatenate([r1, r2])))
    n_cat = len(categories)
    n = len(r1)
    
    # Create weight matrix
    W = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            if weights == 'linear':
                W[i, j] = abs(i - j) / (n_cat - 1)
            else:  # quadratic
                W[i, j] = (i - j) ** 2 / (n_cat - 1) ** 2
    
    # Create confusion matrix
    conf = np.zeros((n_cat, n_cat))
    for i, j in zip(r1, r2):
        cat_i = np.where(categories == i)[0][0]
        cat_j = np.where(categories == j)[0][0]
        conf[cat_i, cat_j] += 1
    
    conf = conf / n
    
    # Marginal distributions
    row_marginals = conf.sum(axis=1)
    col_marginals = conf.sum(axis=0)
    
    # Observed and expected weighted disagreement
    observed = np.sum(W * conf)
    expected = np.sum(W * np.outer(row_marginals, col_marginals))
    
    if expected == 0:
        return 1.0
    
    kappa = 1 - (observed / expected)
    
    return kappa


def calculate_icc(ratings: np.ndarray, icc_type: str = 'ICC(2,1)') -> Tuple[float, float, float]:
    """
    Calculate Intraclass Correlation Coefficient.
    
    Args:
        ratings: 2D array (items × raters)
        icc_type: Type of ICC ('ICC(2,1)' for single measures, 'ICC(2,k)' for average)
    
    Returns:
        Tuple of (ICC value, lower CI, upper CI)
    """
    # Remove items with any missing values
    valid_items = ~np.any(np.isnan(ratings), axis=1)
    ratings = ratings[valid_items, :]
    
    n, k = ratings.shape  # n items, k raters
    
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan
    
    # Grand mean
    grand_mean = np.mean(ratings)
    
    # Sum of squares
    SS_total = np.sum((ratings - grand_mean) ** 2)
    SS_rows = k * np.sum((np.mean(ratings, axis=1) - grand_mean) ** 2)
    SS_columns = n * np.sum((np.mean(ratings, axis=0) - grand_mean) ** 2)
    SS_error = SS_total - SS_rows - SS_columns
    
    # Mean squares
    MS_rows = SS_rows / (n - 1)
    MS_columns = SS_columns / (k - 1)
    MS_error = SS_error / ((n - 1) * (k - 1))
    
    # ICC(2,1) - Two-way random, single measures
    icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error + k * (MS_columns - MS_error) / n)
    
    # Confidence intervals (F-test based)
    F = MS_rows / MS_error
    df1, df2 = n - 1, (n - 1) * (k - 1)
    
    # Lower and upper F values
    F_lower = F / stats.f.ppf(0.975, df1, df2)
    F_upper = F / stats.f.ppf(0.025, df1, df2)
    
    # Convert to ICC bounds (simplified)
    icc_lower = (F_lower - 1) / (F_lower + k - 1)
    icc_upper = (F_upper - 1) / (F_upper + k - 1)
    
    return icc, icc_lower, icc_upper


def validate_against_anthroscore(annotations: pd.DataFrame, 
                                   annotator_cols: List[str],
                                   ground_truth: pd.DataFrame) -> Dict:
    """
    Validate human annotations against computed AnthroScore.
    
    Args:
        annotations: DataFrame with human annotations
        annotator_cols: List of annotator score columns
        ground_truth: DataFrame with true AnthroScores
    
    Returns:
        Dictionary with validation metrics
    """
    logger.info("Validating against computed AnthroScore...")
    
    # Merge annotations with ground truth
    merged = annotations.merge(ground_truth, on='comment_id', how='inner')
    
    results = {}
    
    # Calculate mean human rating
    ratings = merged[annotator_cols].values.astype(float)
    mean_human = np.nanmean(ratings, axis=1)
    true_anthro = merged['true_anthroscore'].values
    
    # Pearson correlation
    valid_mask = ~(np.isnan(mean_human) | np.isnan(true_anthro))
    if np.sum(valid_mask) > 2:
        r, p = stats.pearsonr(mean_human[valid_mask], true_anthro[valid_mask])
        results['pearson_r'] = r
        results['pearson_p'] = p
        
        # Spearman correlation
        rho, p_rho = stats.spearmanr(mean_human[valid_mask], true_anthro[valid_mask])
        results['spearman_rho'] = rho
        results['spearman_p'] = p_rho
        
        # MAE and RMSE (after scaling human ratings to AnthroScore range)
        # Scale 1-5 to 0-1 range
        human_scaled = (mean_human - 1) / 4
        mae = np.nanmean(np.abs(human_scaled[valid_mask] - true_anthro[valid_mask]))
        rmse = np.sqrt(np.nanmean((human_scaled[valid_mask] - true_anthro[valid_mask]) ** 2))
        results['mae'] = mae
        results['rmse'] = rmse
        
        logger.info(f"Validation: r={r:.3f}, rho={rho:.3f}, MAE={mae:.3f}")
    else:
        logger.warning("Insufficient data for validation")
    
    return results


def generate_irr_report(annotations: pd.DataFrame, 
                         annotator_cols: List[str],
                         ground_truth: Optional[pd.DataFrame] = None) -> str:
    """Generate comprehensive IRR report."""
    logger.info("Generating IRR report...")
    
    report = []
    report.append("=" * 80)
    report.append("INTER-RATER RELIABILITY REPORT")
    report.append("AnthroScore Human Validation")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Number of items: {len(annotations)}")
    report.append(f"Number of raters: {len(annotator_cols)}")
    
    # Extract ratings matrix
    ratings = annotations[annotator_cols].values.astype(float)
    
    # Replace empty strings with NaN
    ratings = np.where(ratings == '', np.nan, ratings).astype(float)
    
    # Count valid ratings
    valid_per_rater = np.sum(~np.isnan(ratings), axis=0)
    valid_per_item = np.sum(~np.isnan(ratings), axis=1)
    
    report.append(f"\nRatings per rater: {valid_per_rater}")
    report.append(f"Items with all raters: {np.sum(valid_per_item == len(annotator_cols))}")
    report.append(f"Items with at least 2 raters: {np.sum(valid_per_item >= 2)}")
    
    # --- Krippendorff's Alpha ---
    report.append("\n" + "-" * 80)
    report.append("KRIPPENDORFF'S ALPHA (Primary IRR Metric)")
    report.append("-" * 80)
    
    alpha_ordinal = calculate_krippendorff_alpha(ratings, level='ordinal')
    alpha_interval = calculate_krippendorff_alpha(ratings, level='interval')
    
    report.append(f"\nKrippendorff's Alpha (ordinal):  {alpha_ordinal:.4f}")
    report.append(f"Krippendorff's Alpha (interval): {alpha_interval:.4f}")
    
    # Interpretation
    if alpha_ordinal >= 0.80:
        interp = "EXCELLENT - Highly reliable"
    elif alpha_ordinal >= 0.67:
        interp = "ACCEPTABLE - Allows tentative conclusions"
    elif alpha_ordinal >= 0.50:
        interp = "MODERATE - Caution advised"
    else:
        interp = "POOR - Do not use for analysis"
    
    report.append(f"Interpretation: {interp}")
    report.append("\nThresholds (Krippendorff, 2004):")
    report.append("  α ≥ 0.80: Reliable for most purposes")
    report.append("  α ≥ 0.67: Allows tentative conclusions")
    report.append("  α < 0.67: Insufficient reliability")
    
    # --- Cohen's Kappa (pairwise) ---
    report.append("\n" + "-" * 80)
    report.append("COHEN'S KAPPA (Pairwise Agreement)")
    report.append("-" * 80)
    
    n_raters = len(annotator_cols)
    if n_raters >= 2:
        kappa_matrix = np.full((n_raters, n_raters), np.nan)
        weighted_kappa_matrix = np.full((n_raters, n_raters), np.nan)
        
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                kappa = calculate_cohens_kappa(ratings[:, i], ratings[:, j])
                w_kappa = calculate_weighted_kappa(ratings[:, i], ratings[:, j])
                kappa_matrix[i, j] = kappa
                kappa_matrix[j, i] = kappa
                weighted_kappa_matrix[i, j] = w_kappa
                weighted_kappa_matrix[j, i] = w_kappa
        
        report.append("\nCohen's Kappa (unweighted):")
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                report.append(f"  Rater {i+1} vs Rater {j+1}: κ = {kappa_matrix[i, j]:.4f}")
        
        report.append("\nWeighted Kappa (quadratic weights - for ordinal data):")
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                report.append(f"  Rater {i+1} vs Rater {j+1}: κ_w = {weighted_kappa_matrix[i, j]:.4f}")
        
        avg_kappa = np.nanmean(kappa_matrix[~np.isnan(kappa_matrix) & (kappa_matrix != 0)])
        avg_weighted = np.nanmean(weighted_kappa_matrix[~np.isnan(weighted_kappa_matrix)])
        report.append(f"\nAverage κ: {avg_kappa:.4f}")
        report.append(f"Average κ_w: {avg_weighted:.4f}")
    
    # --- ICC ---
    report.append("\n" + "-" * 80)
    report.append("INTRACLASS CORRELATION COEFFICIENT (ICC)")
    report.append("-" * 80)
    
    icc, icc_lower, icc_upper = calculate_icc(ratings)
    report.append(f"\nICC(2,1) - Two-way random, single measures:")
    report.append(f"  ICC = {icc:.4f}")
    report.append(f"  95% CI: [{icc_lower:.4f}, {icc_upper:.4f}]")
    
    # ICC interpretation
    if icc >= 0.90:
        icc_interp = "EXCELLENT"
    elif icc >= 0.75:
        icc_interp = "GOOD"
    elif icc >= 0.50:
        icc_interp = "MODERATE"
    else:
        icc_interp = "POOR"
    
    report.append(f"Interpretation: {icc_interp}")
    
    # --- Validation against AnthroScore ---
    if ground_truth is not None:
        report.append("\n" + "-" * 80)
        report.append("VALIDATION AGAINST COMPUTED ANTHROSCORE")
        report.append("-" * 80)
        
        validation = validate_against_anthroscore(annotations, annotator_cols, ground_truth)
        
        if validation:
            report.append(f"\nCorrelation (Mean Human Rating vs AnthroScore):")
            report.append(f"  Pearson r:  {validation.get('pearson_r', np.nan):.4f} (p = {validation.get('pearson_p', np.nan):.4f})")
            report.append(f"  Spearman ρ: {validation.get('spearman_rho', np.nan):.4f} (p = {validation.get('spearman_p', np.nan):.4f})")
            report.append(f"\nError Metrics (human ratings scaled to 0-1):")
            report.append(f"  MAE:  {validation.get('mae', np.nan):.4f}")
            report.append(f"  RMSE: {validation.get('rmse', np.nan):.4f}")
            
            # Interpretation
            r = validation.get('pearson_r', 0)
            if r >= 0.7:
                val_interp = "STRONG validity - AnthroScore aligns well with human judgment"
            elif r >= 0.5:
                val_interp = "MODERATE validity - AnthroScore reasonably captures human perception"
            elif r >= 0.3:
                val_interp = "WEAK validity - Some alignment but may need refinement"
            else:
                val_interp = "POOR validity - AnthroScore may not capture human perception"
            
            report.append(f"\nValidity interpretation: {val_interp}")
    
    # --- Summary ---
    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append(f"\nPrimary metric (Krippendorff's α): {alpha_ordinal:.4f}")
    report.append(f"ICC(2,1):                          {icc:.4f}")
    
    if alpha_ordinal >= 0.67 and icc >= 0.50:
        overall = "✓ ACCEPTABLE RELIABILITY - Proceed with analysis"
    else:
        overall = "✗ INSUFFICIENT RELIABILITY - Review annotations"
    
    report.append(f"\nOverall: {overall}")
    
    return '\n'.join(report)


def main():
    """Main function to calculate IRR."""
    parser = argparse.ArgumentParser(description='Calculate Inter-Rater Reliability')
    parser.add_argument('--annotations', type=str, 
                        default='Data/annotations/annotation_sheet_completed.csv',
                        help='Path to completed annotations CSV')
    parser.add_argument('--output', type=str,
                        default='Data/annotations/irr_report.txt',
                        help='Path for output report')
    args = parser.parse_args()
    
    # Check if annotations exist
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        logger.error(f"Annotations file not found: {annotations_path}")
        logger.info("Please complete annotations first, then run this script.")
        logger.info(f"Expected file: {annotations_path}")
        return
    
    # Load data
    annotations, annotator_cols = load_annotations(annotations_path)
    ground_truth = load_ground_truth()
    
    # Generate report
    report = generate_irr_report(annotations, annotator_cols, ground_truth)
    
    # Save report
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")
    
    # Print report
    print(report)


if __name__ == "__main__":
    main()

