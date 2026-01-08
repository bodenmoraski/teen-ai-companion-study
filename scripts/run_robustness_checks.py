#!/usr/bin/env python3
"""
Robustness Checks for RQ2 Findings
==================================

This script performs comprehensive robustness checks:
1. Sensitivity analysis at multiple confidence thresholds (0.5, 0.6, 0.7, 0.8)
2. Analysis with self-declared demographics only
3. Bootstrap confidence intervals for key effect sizes

Output: results/robustness/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("Data/features")
RESULTS_DIR = Path("results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets."""
    logger.info("Loading data...")
    
    # Load anthroscores
    anthro = pd.read_parquet(DATA_DIR / "user_anthroscores.parquet")
    logger.info(f"AnthroScores: {len(anthro):,} users")
    
    # Load age predictions
    age_pred = pd.read_parquet(DATA_DIR / "ultimate_predictor" / "ultimate_predictions.parquet")
    logger.info(f"Age predictions: {len(age_pred):,} users")
    
    # Load gender predictions
    gender_pred = pd.read_parquet(DATA_DIR / "ultimate_predictor" / "gender_predictions.parquet")
    logger.info(f"Gender predictions: {len(gender_pred):,} users")
    
    # Load self-declarations
    self_decl = pd.read_parquet(DATA_DIR / "self_declarations.parquet")
    logger.info(f"Self-declarations: {len(self_decl):,} users")
    
    return anthro, age_pred, gender_pred, self_decl


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (group1.mean() - group2.mean()) / pooled_std


def bootstrap_ci(group1: np.ndarray, group2: np.ndarray, 
                 statistic: str = 'cohens_d',
                 n_bootstrap: int = 10000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        group1, group2: Data arrays for two groups
        statistic: 'cohens_d' or 'mean_diff'
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
    
    Returns:
        Tuple of (point estimate, lower CI, upper CI)
    """
    rng = np.random.default_rng(42)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot1 = rng.choice(group1, size=len(group1), replace=True)
        boot2 = rng.choice(group2, size=len(group2), replace=True)
        
        if statistic == 'cohens_d':
            boot_stat = calculate_cohens_d(boot1, boot2)
        else:  # mean_diff
            boot_stat = boot1.mean() - boot2.mean()
        
        bootstrap_stats.append(boot_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Point estimate
    if statistic == 'cohens_d':
        point_est = calculate_cohens_d(group1, group2)
    else:
        point_est = group1.mean() - group2.mean()
    
    # Percentile method for CI
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_est, lower, upper


def analyze_at_threshold(anthro: pd.DataFrame, 
                         age_pred: pd.DataFrame,
                         threshold: float) -> Dict:
    """
    Run key RQ2 analyses at a specific confidence threshold.
    
    Args:
        anthro: User anthroscores
        age_pred: Age predictions with confidence
        threshold: Minimum confidence threshold
    
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing at confidence threshold {threshold}...")
    
    # Filter by confidence
    confident_users = age_pred[age_pred['confidence'] >= threshold].copy()
    
    # Handle different column names
    age_col = 'age_bucket' if 'age_bucket' in confident_users.columns else 'age_bucket_predicted'
    
    # Merge with anthroscores
    df = anthro.merge(confident_users[['author', age_col, 'confidence']], 
                      on='author', how='inner')
    
    # Standardize column name
    if age_col != 'age_bucket':
        df = df.rename(columns={age_col: 'age_bucket'})
    
    results = {
        'threshold': threshold,
        'n_users': len(df),
        'pct_of_total': len(df) / len(anthro) * 100,
    }
    
    # Age distribution at this threshold
    age_dist = df['age_bucket'].value_counts(normalize=True) * 100
    results['age_distribution'] = age_dist.to_dict()
    
    # IMPORTANT: Filter to users with non-zero anthropomorphization
    # This matches the original analysis methodology
    df_nonzero = df[df['anthroscore_mean'] != 0].copy()
    
    results['n_users_raw'] = len(df)
    results['n_users_nonzero'] = len(df_nonzero)
    results['pct_nonzero'] = len(df_nonzero) / len(df) * 100 if len(df) > 0 else 0
    
    # Create teen vs adult comparison (handle different formats)
    df_nonzero['is_teen'] = df_nonzero['age_bucket'].str.lower().isin(['13-18', 'teen', '13-17'])
    teens = df_nonzero[df_nonzero['is_teen']]['anthroscore_max'].dropna()
    adults = df_nonzero[~df_nonzero['is_teen']]['anthroscore_max'].dropna()
    
    results['n_teens'] = len(teens)
    results['n_adults'] = len(adults)
    
    if len(teens) > 10 and len(adults) > 10:
        # T-test
        t_stat, p_value = ttest_ind(teens, adults)
        results['t_statistic'] = t_stat
        results['p_value'] = p_value
        
        # Effect size
        d = calculate_cohens_d(teens.values, adults.values)
        results['cohens_d'] = d
        
        # Means
        results['teen_mean'] = teens.mean()
        results['adult_mean'] = adults.mean()
        
        # Mann-Whitney U (non-parametric)
        u_stat, p_mw = mannwhitneyu(teens, adults, alternative='two-sided')
        results['mann_whitney_u'] = u_stat
        results['mann_whitney_p'] = p_mw
        
        # Significance
        results['significant_at_05'] = p_value < 0.05
        results['significant_at_01'] = p_value < 0.01
        results['significant_at_001'] = p_value < 0.001
        
        logger.info(f"  n={len(df):,}, d={d:.3f}, p={p_value:.4f}")
    else:
        logger.warning(f"  Insufficient data at threshold {threshold}")
        results['error'] = "Insufficient data"
    
    return results


def run_threshold_sensitivity(anthro: pd.DataFrame, 
                               age_pred: pd.DataFrame) -> List[Dict]:
    """Run sensitivity analysis across multiple thresholds."""
    logger.info("=" * 70)
    logger.info("THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("=" * 70)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for threshold in thresholds:
        result = analyze_at_threshold(anthro, age_pred, threshold)
        results.append(result)
    
    return results


def run_self_declared_analysis(anthro: pd.DataFrame,
                                self_decl: pd.DataFrame) -> Dict:
    """
    Run analysis using only self-declared demographics.
    
    This is the gold-standard validation.
    """
    logger.info("=" * 70)
    logger.info("SELF-DECLARED DEMOGRAPHICS ANALYSIS")
    logger.info("=" * 70)
    
    # Filter to users with self-declared age (not None/NaN)
    self_decl_valid = self_decl.copy()
    
    # Check for age columns
    age_col = None
    for col in ['age_self_declared', 'age', 'declared_age', 'self_declared_age']:
        if col in self_decl_valid.columns:
            age_col = col
            break
    
    if age_col is None:
        logger.warning("No age column found in self-declarations")
        return {'error': "No age column", 'n_users': 0}
    
    # Filter to users with actual self-declared age
    self_decl_valid = self_decl_valid[self_decl_valid[age_col].notna()].copy()
    logger.info(f"Users with self-declared age: {len(self_decl_valid):,}")
    
    if len(self_decl_valid) == 0:
        logger.warning("No users with self-declared age")
        return {'error': "No users with self-declared age", 'n_users': 0}
    
    # Merge with anthroscores
    df = anthro.merge(self_decl_valid, on='author', how='inner')
    logger.info(f"After merge with anthroscores: {len(df):,}")
    
    results = {'n_users': len(df)}
    
    # Create teen indicator (age < 19)
    if df[age_col].dtype in ['int64', 'float64']:
        df['is_teen'] = df[age_col] < 19
    else:
        # Assume categorical - check for bucket column too
        bucket_col = 'age_bucket_self_declared' if 'age_bucket_self_declared' in df.columns else None
        if bucket_col:
            df['is_teen'] = df[bucket_col].str.lower().isin(['13-18', 'teen', '13-17'])
        else:
            df['is_teen'] = df[age_col].astype(str).str.lower().isin(['13-18', 'teen', '13-17', '13-18 years'])
    
    # IMPORTANT: Filter to non-zero anthropomorphizers (matches original methodology)
    df = df[df['anthroscore_max'] > 0].copy()
    
    teens = df[df['is_teen']]['anthroscore_max'].dropna()
    adults = df[~df['is_teen']]['anthroscore_max'].dropna()
    
    results['n_teens'] = len(teens)
    results['n_adults'] = len(adults)
    
    logger.info(f"Teens: {len(teens)}, Adults: {len(adults)} (filtered to nonzero anthroscore)")
    
    if len(teens) > 5 and len(adults) > 5:
        # T-test
        t_stat, p_value = ttest_ind(teens, adults)
        results['t_statistic'] = t_stat
        results['p_value'] = p_value
        
        # Effect size
        d = calculate_cohens_d(teens.values, adults.values)
        results['cohens_d'] = d
        
        # Means
        results['teen_mean'] = teens.mean()
        results['adult_mean'] = adults.mean()
        results['mean_diff'] = teens.mean() - adults.mean()
        
        # Bootstrap CI for Cohen's d
        d_point, d_lower, d_upper = bootstrap_ci(teens.values, adults.values, 
                                                  statistic='cohens_d', n_bootstrap=10000)
        results['cohens_d_ci_lower'] = d_lower
        results['cohens_d_ci_upper'] = d_upper
        
        logger.info(f"Cohen's d = {d:.3f} [{d_lower:.3f}, {d_upper:.3f}]")
        logger.info(f"p = {p_value:.4f}")
    else:
        logger.warning("Insufficient self-declared data for analysis")
        results['error'] = "Insufficient data"
    
    return results


def run_bootstrap_cis(anthro: pd.DataFrame, 
                       age_pred: pd.DataFrame,
                       threshold: float = 0.6,
                       n_bootstrap: int = 10000) -> Dict:
    """
    Calculate bootstrap confidence intervals for key effect sizes.
    """
    logger.info("=" * 70)
    logger.info("BOOTSTRAP CONFIDENCE INTERVALS")
    logger.info(f"Using {n_bootstrap:,} bootstrap samples")
    logger.info("=" * 70)
    
    # Prepare data
    confident_users = age_pred[age_pred['confidence'] >= threshold].copy()
    
    # Handle different column names
    age_col = 'age_bucket' if 'age_bucket' in confident_users.columns else 'age_bucket_predicted'
    df = anthro.merge(confident_users[['author', age_col]], on='author', how='inner')
    
    # Standardize column name
    if age_col != 'age_bucket':
        df = df.rename(columns={age_col: 'age_bucket'})
    
    # IMPORTANT: Filter to non-zero anthropomorphizers (matches original methodology)
    df = df[df['anthroscore_mean'] != 0].copy()
    
    df['is_teen'] = df['age_bucket'].str.lower().isin(['13-18', 'teen', '13-17'])
    teens = df[df['is_teen']]['anthroscore_max'].dropna().values
    adults = df[~df['is_teen']]['anthroscore_max'].dropna().values
    
    results = {
        'threshold': threshold,
        'n_bootstrap': n_bootstrap,
        'n_teens': len(teens),
        'n_adults': len(adults),
    }
    
    # Bootstrap for Cohen's d (main effect)
    logger.info("Computing bootstrap CI for Cohen's d (Teen vs Adult)...")
    d_point, d_lower, d_upper = bootstrap_ci(teens, adults, 'cohens_d', n_bootstrap)
    results['cohens_d'] = d_point
    results['cohens_d_ci_lower'] = d_lower
    results['cohens_d_ci_upper'] = d_upper
    results['cohens_d_ci_width'] = d_upper - d_lower
    logger.info(f"  Cohen's d = {d_point:.4f} [{d_lower:.4f}, {d_upper:.4f}]")
    
    # Bootstrap for mean difference
    logger.info("Computing bootstrap CI for mean difference...")
    md_point, md_lower, md_upper = bootstrap_ci(teens, adults, 'mean_diff', n_bootstrap)
    results['mean_diff'] = md_point
    results['mean_diff_ci_lower'] = md_lower
    results['mean_diff_ci_upper'] = md_upper
    logger.info(f"  Mean diff = {md_point:.4f} [{md_lower:.4f}, {md_upper:.4f}]")
    
    # Check if CI excludes zero
    results['ci_excludes_zero'] = (d_lower > 0) or (d_upper < 0)
    
    # Bootstrap for high anthropomorphization rate difference
    logger.info("Computing bootstrap CI for high anthro rate difference...")
    
    teen_high_rate = (teens >= 1.5).mean()
    adult_high_rate = (adults >= 1.5).mean()
    
    rng = np.random.default_rng(42)
    rate_diffs = []
    
    for _ in range(n_bootstrap):
        boot_teens = rng.choice(teens, size=len(teens), replace=True)
        boot_adults = rng.choice(adults, size=len(adults), replace=True)
        
        boot_teen_rate = (boot_teens >= 1.5).mean()
        boot_adult_rate = (boot_adults >= 1.5).mean()
        rate_diffs.append(boot_teen_rate - boot_adult_rate)
    
    rate_diffs = np.array(rate_diffs)
    rate_diff_point = teen_high_rate - adult_high_rate
    rate_diff_lower = np.percentile(rate_diffs, 2.5)
    rate_diff_upper = np.percentile(rate_diffs, 97.5)
    
    results['high_anthro_rate_teen'] = teen_high_rate
    results['high_anthro_rate_adult'] = adult_high_rate
    results['high_anthro_rate_diff'] = rate_diff_point
    results['high_anthro_rate_diff_ci_lower'] = rate_diff_lower
    results['high_anthro_rate_diff_ci_upper'] = rate_diff_upper
    
    logger.info(f"  Rate diff = {rate_diff_point:.4f} [{rate_diff_lower:.4f}, {rate_diff_upper:.4f}]")
    
    return results


def generate_robustness_report(threshold_results: List[Dict],
                                self_declared_results: Dict,
                                bootstrap_results: Dict) -> str:
    """Generate comprehensive robustness report."""
    logger.info("Generating robustness report...")
    
    report = []
    report.append("=" * 80)
    report.append("ROBUSTNESS CHECKS REPORT")
    report.append("RQ2: Demographics and Anthropomorphization")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Section 1: Threshold Sensitivity
    report.append("\n" + "=" * 80)
    report.append("SECTION 1: CONFIDENCE THRESHOLD SENSITIVITY ANALYSIS")
    report.append("=" * 80)
    report.append("\nQuestion: Does the teen effect hold at different prediction confidence levels?")
    
    report.append("\n" + "-" * 80)
    report.append(f"{'Threshold':>10} {'N Users':>10} {'N Teens':>10} {'N Adults':>10} "
                  f"{'Cohen d':>10} {'p-value':>12} {'Sig?':>8}")
    report.append("-" * 80)
    
    for r in threshold_results:
        if 'error' not in r:
            sig = "***" if r['significant_at_001'] else "**" if r['significant_at_01'] else "*" if r['significant_at_05'] else ""
            report.append(f"{r['threshold']:>10.1f} {r['n_users']:>10,} {r['n_teens']:>10,} "
                         f"{r['n_adults']:>10,} {r['cohens_d']:>10.4f} {r['p_value']:>12.4e} {sig:>8}")
        else:
            report.append(f"{r['threshold']:>10.1f} {r.get('n_users', 'N/A'):>10} - Insufficient data -")
    
    report.append("-" * 80)
    
    # Summary of threshold sensitivity
    valid_results = [r for r in threshold_results if 'error' not in r and r.get('significant_at_05')]
    if len(valid_results) == len([r for r in threshold_results if 'error' not in r]):
        report.append("\n✓ ROBUST: Effect is significant at ALL tested thresholds")
    elif len(valid_results) > 0:
        report.append(f"\n⚠ PARTIALLY ROBUST: Effect significant at {len(valid_results)}/{len(threshold_results)} thresholds")
    else:
        report.append("\n✗ NOT ROBUST: Effect not significant at any threshold")
    
    # Effect size consistency
    ds = [r['cohens_d'] for r in threshold_results if 'error' not in r]
    if ds:
        report.append(f"\nEffect size range: d = {min(ds):.3f} to {max(ds):.3f}")
        report.append(f"Effect size variability: SD = {np.std(ds):.4f}")
    
    # Section 2: Self-Declared Analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 2: SELF-DECLARED DEMOGRAPHICS ONLY")
    report.append("=" * 80)
    report.append("\nQuestion: Does the effect hold when using ground-truth demographics?")
    
    if 'error' not in self_declared_results:
        report.append(f"\nSample size: {self_declared_results['n_users']:,} users")
        report.append(f"  Teens: {self_declared_results['n_teens']:,}")
        report.append(f"  Adults: {self_declared_results['n_adults']:,}")
        report.append(f"\nTeen mean AnthroScore: {self_declared_results['teen_mean']:.4f}")
        report.append(f"Adult mean AnthroScore: {self_declared_results['adult_mean']:.4f}")
        report.append(f"Difference: {self_declared_results['mean_diff']:.4f}")
        report.append(f"\nCohen's d: {self_declared_results['cohens_d']:.4f}")
        if 'cohens_d_ci_lower' in self_declared_results:
            report.append(f"  95% CI: [{self_declared_results['cohens_d_ci_lower']:.4f}, "
                         f"{self_declared_results['cohens_d_ci_upper']:.4f}]")
        report.append(f"p-value: {self_declared_results['p_value']:.4e}")
        
        if self_declared_results['p_value'] < 0.05:
            report.append("\n✓ VALIDATED: Effect replicates with self-declared demographics")
        else:
            report.append("\n⚠ NOT VALIDATED: Effect not significant with self-declared demographics")
            report.append("  (Note: Sample size may be insufficient for detection)")
    else:
        report.append(f"\n⚠ Could not analyze: {self_declared_results.get('error', 'Unknown error')}")
    
    # Section 3: Bootstrap CIs
    report.append("\n" + "=" * 80)
    report.append("SECTION 3: BOOTSTRAP CONFIDENCE INTERVALS")
    report.append("=" * 80)
    report.append(f"\nBootstrap samples: {bootstrap_results['n_bootstrap']:,}")
    report.append(f"Confidence threshold used: {bootstrap_results['threshold']}")
    report.append(f"Sample: {bootstrap_results['n_teens']:,} teens, {bootstrap_results['n_adults']:,} adults")
    
    report.append("\n" + "-" * 60)
    report.append("Cohen's d (Teen vs Adult on Max AnthroScore):")
    report.append("-" * 60)
    report.append(f"  Point estimate: {bootstrap_results['cohens_d']:.4f}")
    report.append(f"  95% CI: [{bootstrap_results['cohens_d_ci_lower']:.4f}, "
                 f"{bootstrap_results['cohens_d_ci_upper']:.4f}]")
    report.append(f"  CI width: {bootstrap_results['cohens_d_ci_width']:.4f}")
    
    if bootstrap_results['ci_excludes_zero']:
        report.append("  ✓ CI excludes zero - effect is statistically robust")
    else:
        report.append("  ⚠ CI includes zero - effect may not be reliable")
    
    report.append("\n" + "-" * 60)
    report.append("Mean Difference:")
    report.append("-" * 60)
    report.append(f"  Point estimate: {bootstrap_results['mean_diff']:.4f}")
    report.append(f"  95% CI: [{bootstrap_results['mean_diff_ci_lower']:.4f}, "
                 f"{bootstrap_results['mean_diff_ci_upper']:.4f}]")
    
    report.append("\n" + "-" * 60)
    report.append("High Anthropomorphization Rate (AnthroScore ≥ 1.5):")
    report.append("-" * 60)
    report.append(f"  Teen rate: {bootstrap_results['high_anthro_rate_teen']:.1%}")
    report.append(f"  Adult rate: {bootstrap_results['high_anthro_rate_adult']:.1%}")
    report.append(f"  Difference: {bootstrap_results['high_anthro_rate_diff']:.1%}")
    report.append(f"  95% CI: [{bootstrap_results['high_anthro_rate_diff_ci_lower']:.1%}, "
                 f"{bootstrap_results['high_anthro_rate_diff_ci_upper']:.1%}]")
    
    # Overall Summary
    report.append("\n" + "=" * 80)
    report.append("OVERALL ROBUSTNESS SUMMARY")
    report.append("=" * 80)
    
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Threshold sensitivity (pass if significant at 3+ thresholds)
    if valid_results and len(valid_results) >= 3:
        checks_passed += 1
        report.append("\n✓ Threshold Sensitivity: PASSED ({}/{} thresholds significant)".format(
            len(valid_results), len([r for r in threshold_results if 'error' not in r])))
    else:
        report.append("\n⚠ Threshold Sensitivity: PARTIAL ({}/{} thresholds significant)".format(
            len(valid_results), len([r for r in threshold_results if 'error' not in r])))
    
    # Check 2: Self-declared validation (note: may show different direction)
    if 'error' not in self_declared_results:
        self_d = self_declared_results.get('cohens_d', 0)
        if self_d > 0:
            checks_passed += 1
            report.append("✓ Self-Declared Validation: PASSED (d={:.3f})".format(self_d))
        else:
            report.append("⚠ Self-Declared Validation: OPPOSITE DIRECTION (d={:.3f})".format(self_d))
            report.append("   Note: Self-declared may differ from predicted demographics")
    else:
        report.append("⚠ Self-Declared Validation: INCONCLUSIVE")
    
    # Check 3: Bootstrap CI
    if bootstrap_results['ci_excludes_zero']:
        checks_passed += 1
        report.append("✓ Bootstrap CI: PASSED (excludes zero)")
    else:
        report.append("✗ Bootstrap CI: FAILED (includes zero)")
    
    report.append(f"\nOverall: {checks_passed}/{total_checks} robustness checks passed")
    
    if checks_passed == total_checks:
        report.append("\n★★★ HIGHLY ROBUST: Findings are robust to methodological variations")
    elif checks_passed >= 2:
        report.append("\n★★ MODERATELY ROBUST: Findings are generally robust")
    else:
        report.append("\n★ CAUTION: Some robustness concerns - interpret with care")
    
    return '\n'.join(report)


def main():
    """Main function to run all robustness checks."""
    logger.info("=" * 70)
    logger.info("RUNNING ROBUSTNESS CHECKS")
    logger.info("=" * 70)
    
    # Load data
    anthro, age_pred, gender_pred, self_decl = load_data()
    
    # Run analyses
    # 1. Threshold sensitivity
    threshold_results = run_threshold_sensitivity(anthro, age_pred)
    
    # 2. Self-declared analysis
    self_declared_results = run_self_declared_analysis(anthro, self_decl)
    
    # 3. Bootstrap CIs
    bootstrap_results = run_bootstrap_cis(anthro, age_pred, threshold=0.6, n_bootstrap=10000)
    
    # Generate report
    report = generate_robustness_report(threshold_results, self_declared_results, bootstrap_results)
    
    # Save report
    report_path = RESULTS_DIR / "robustness_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\nReport saved to {report_path}")
    
    # Print report (handle Unicode for Windows)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        # Fall back to ASCII-safe version
        safe_report = report.encode('ascii', 'replace').decode('ascii')
        print("\n" + safe_report)
    
    # Save detailed results as JSON for further analysis
    import json
    
    results_dict = {
        'threshold_sensitivity': threshold_results,
        'self_declared': self_declared_results,
        'bootstrap': bootstrap_results,
        'generated': datetime.now().isoformat()
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj
    
    results_dict = convert_numpy(results_dict)
    
    json_path = RESULTS_DIR / "robustness_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Detailed results saved to {json_path}")
    
    return threshold_results, self_declared_results, bootstrap_results


if __name__ == "__main__":
    results = main()

