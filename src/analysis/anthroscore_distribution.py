"""
AnthroScore Distribution Analysis Module.

This module analyzes the distribution of AnthroScore values,
addressing concerns about floor effects and alternative aggregations.

Key insights from CRITICISM.md:
- Most users have AnthroScore ≈ 0 (50th and 75th percentile = 0)
- Floor effects may limit our ability to detect demographic effects
- Alternative aggregations (max, std, pct_nonzero) may be more informative
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def analyze_distribution(
    data_path: Path = None,
    df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Analyze the distribution of AnthroScore.
    
    Args:
        data_path: Path to user_anthroscores.parquet
        df: DataFrame with anthroscore columns
        
    Returns:
        Dict with distribution analysis results
    """
    if df is None:
        if data_path is None:
            data_path = Path("Data/features/user_anthroscores.parquet")
        df = pd.read_parquet(data_path)
    
    if 'anthroscore_mean' not in df.columns:
        logger.error("anthroscore_mean column not found")
        return {}
    
    scores = df['anthroscore_mean'].dropna()
    
    results = {
        'n': len(scores),
        'basic_stats': {
            'mean': scores.mean(),
            'std': scores.std(),
            'median': scores.median(),
            'min': scores.min(),
            'max': scores.max(),
            'skewness': scores.skew(),
            'kurtosis': scores.kurtosis()
        },
        'percentiles': {
            '5th': scores.quantile(0.05),
            '10th': scores.quantile(0.10),
            '25th': scores.quantile(0.25),
            '50th': scores.quantile(0.50),
            '75th': scores.quantile(0.75),
            '90th': scores.quantile(0.90),
            '95th': scores.quantile(0.95),
            '99th': scores.quantile(0.99)
        },
        'floor_effects': {
            'n_zero': (scores == 0).sum(),
            'pct_zero': (scores == 0).mean(),
            'n_near_zero': (scores < 0.01).sum(),
            'pct_near_zero': (scores < 0.01).mean(),
            'n_positive': (scores > 0).sum(),
            'pct_positive': (scores > 0).mean()
        },
        'distribution_shape': {}
    }
    
    # Test for normality
    if len(scores) > 5000:
        # Use smaller sample for normality test
        sample = scores.sample(5000, random_state=42)
    else:
        sample = scores
    
    try:
        shapiro_stat, shapiro_p = scipy_stats.shapiro(sample)
        results['distribution_shape']['shapiro_stat'] = shapiro_stat
        results['distribution_shape']['shapiro_p'] = shapiro_p
        results['distribution_shape']['normal'] = shapiro_p > 0.05
    except Exception as e:
        logger.warning(f"Shapiro test failed: {e}")
    
    # Classify floor effect severity
    pct_zero = results['floor_effects']['pct_zero']
    if pct_zero > 0.7:
        results['floor_effects']['severity'] = 'SEVERE'
    elif pct_zero > 0.5:
        results['floor_effects']['severity'] = 'MODERATE'
    elif pct_zero > 0.3:
        results['floor_effects']['severity'] = 'MILD'
    else:
        results['floor_effects']['severity'] = 'MINIMAL'
    
    return results


def compare_aggregations(
    data_path: Path = None,
    df: pd.DataFrame = None
) -> Dict[str, Dict]:
    """
    Compare different aggregation methods for AnthroScore.
    
    Instead of just mean, we try:
    - max: Peak anthropomorphization
    - std: Variability in anthropomorphization
    - pct_nonzero: Proportion of comments with any anthropomorphization
    
    Args:
        data_path: Path to user_anthroscores.parquet
        df: DataFrame with anthroscore columns
        
    Returns:
        Dict with results for each aggregation method
    """
    if df is None:
        if data_path is None:
            data_path = Path("Data/features/user_anthroscores.parquet")
        df = pd.read_parquet(data_path)
    
    results = {}
    
    # Mean (baseline)
    if 'anthroscore_mean' in df.columns:
        scores = df['anthroscore_mean'].dropna()
        results['mean'] = {
            'available': True,
            'mean': scores.mean(),
            'std': scores.std(),
            'pct_zero': (scores == 0).mean(),
            'pct_positive': (scores > 0).mean()
        }
    
    # Max (if available or can be computed)
    if 'anthroscore_max' in df.columns:
        max_scores = df['anthroscore_max'].dropna()
        results['max'] = {
            'available': True,
            'mean': max_scores.mean(),
            'std': max_scores.std(),
            'pct_zero': (max_scores == 0).mean(),
            'pct_positive': (max_scores > 0).mean()
        }
    else:
        results['max'] = {'available': False}
    
    # Std (if available)
    if 'anthroscore_std' in df.columns:
        std_scores = df['anthroscore_std'].dropna()
        results['std'] = {
            'available': True,
            'mean': std_scores.mean(),
            'std': std_scores.std()
        }
    else:
        results['std'] = {'available': False}
    
    # Count (number of comments per user)
    if 'anthroscore_count' in df.columns:
        counts = df['anthroscore_count'].dropna()
        results['count'] = {
            'available': True,
            'mean': counts.mean(),
            'std': counts.std(),
            'min': counts.min(),
            'max': counts.max()
        }
    else:
        results['count'] = {'available': False}
    
    # Percentage nonzero (computed from mean if we assume binary)
    # Note: This is an approximation - proper calculation needs comment-level data
    if 'anthroscore_mean' in df.columns:
        # Users with any positive mean have at least some anthropomorphization
        pct_any = (df['anthroscore_mean'] > 0).mean()
        results['pct_nonzero'] = {
            'available': True,
            'pct_users_with_any': pct_any
        }
    
    return results


def analyze_nonzero_subsample(
    merged_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Analyze demographic effects only among users with non-zero AnthroScore.
    
    This addresses the floor effect concern by focusing on users
    who actually anthropomorphize.
    
    Args:
        merged_df: DataFrame with anthroscore and demographics
        
    Returns:
        Dict with analysis results
    """
    if merged_df is None:
        merged = pd.read_parquet("Data/features/full_merged_dataset.parquet")
        demo = pd.read_parquet("Data/features/demographics.parquet")
        merged_df = merged.merge(demo[['author', 'age_bucket', 'gender']], on='author', how='left')
    
    if 'anthroscore_mean' not in merged_df.columns:
        return {'error': 'anthroscore_mean not found'}
    
    # Filter to non-zero AnthroScore
    nonzero_df = merged_df[merged_df['anthroscore_mean'] > 0].copy()
    
    results = {
        'n_total': len(merged_df),
        'n_nonzero': len(nonzero_df),
        'pct_nonzero': len(nonzero_df) / len(merged_df),
        'subsample_analysis': {}
    }
    
    # Demographics of non-zero users
    if 'age_bucket' in nonzero_df.columns:
        age_dist = nonzero_df['age_bucket'].value_counts(normalize=True)
        results['subsample_analysis']['age_distribution'] = age_dist.to_dict()
    
    if 'gender' in nonzero_df.columns:
        gender_dist = nonzero_df['gender'].value_counts(normalize=True)
        results['subsample_analysis']['gender_distribution'] = gender_dist.to_dict()
    
    # Run regression on non-zero subset
    # Convert age to 3-bucket
    def to_3bucket(age):
        if pd.isna(age):
            return None
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        return 'adult'
    
    nonzero_df['age_3bucket'] = nonzero_df['age_bucket'].apply(to_3bucket)
    
    # Filter to complete cases
    complete_df = nonzero_df[
        nonzero_df['age_3bucket'].notna() &
        nonzero_df['gender'].notna() &
        (nonzero_df['gender'] != 'unknown')
    ]
    
    if len(complete_df) >= 100:
        try:
            from statsmodels.formula.api import ols
            
            model = ols(
                "anthroscore_mean ~ C(age_3bucket) + C(gender)",
                data=complete_df
            ).fit()
            
            results['subsample_analysis']['regression'] = {
                'n': int(model.nobs),
                'r2': model.rsquared,
                'adj_r2': model.rsquared_adj,
                'f_pvalue': model.f_pvalue,
                'coefficients': dict(model.params),
                'pvalues': dict(model.pvalues)
            }
        except Exception as e:
            results['subsample_analysis']['regression'] = {'error': str(e)}
    else:
        results['subsample_analysis']['regression'] = {
            'error': f'Insufficient data: {len(complete_df)} complete cases'
        }
    
    return results


def compute_alternative_metrics(
    comments_path: Path = None
) -> pd.DataFrame:
    """
    Compute alternative AnthroScore aggregations from comment-level data.
    
    Args:
        comments_path: Path to comments with anthroscores
        
    Returns:
        DataFrame with user-level alternative metrics
    """
    if comments_path is None:
        comments_path = Path("Data/features/comments_with_anthroscores.parquet")
    
    if not comments_path.exists():
        logger.warning(f"Comments file not found: {comments_path}")
        return pd.DataFrame()
    
    comments = pd.read_parquet(comments_path)
    
    if 'author' not in comments.columns:
        logger.error("author column not found in comments")
        return pd.DataFrame()
    
    # Find the anthroscore column
    anthro_col = None
    for col in ['anthroscore', 'score', 'anthroscore_mean']:
        if col in comments.columns:
            anthro_col = col
            break
    
    if anthro_col is None:
        logger.error("No anthroscore column found in comments")
        return pd.DataFrame()
    
    # Aggregate by user
    user_metrics = comments.groupby('author').agg({
        anthro_col: ['mean', 'max', 'std', 'count', lambda x: (x > 0).mean()]
    }).reset_index()
    
    # Flatten column names
    user_metrics.columns = [
        'author', 
        'anthroscore_mean', 
        'anthroscore_max', 
        'anthroscore_std', 
        'anthroscore_count',
        'anthroscore_pct_nonzero'
    ]
    
    return user_metrics


def generate_distribution_report(output_path: Path = None) -> str:
    """
    Generate comprehensive distribution analysis report.
    
    Args:
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANTHROSCORE DISTRIBUTION ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    
    # Basic distribution
    lines.append("1. DISTRIBUTION CHARACTERISTICS")
    lines.append("-" * 50)
    
    dist = analyze_distribution()
    
    if dist:
        lines.append(f"N users: {dist['n']:,}")
        lines.append("")
        
        lines.append("Basic Statistics:")
        for stat, value in dist['basic_stats'].items():
            lines.append(f"  {stat}: {value:.6f}")
        lines.append("")
        
        lines.append("Percentiles:")
        for pct, value in dist['percentiles'].items():
            lines.append(f"  {pct}: {value:.6f}")
        lines.append("")
        
        lines.append("Floor Effects:")
        lines.append(f"  Exactly zero: {dist['floor_effects']['n_zero']:,} "
                    f"({dist['floor_effects']['pct_zero']:.1%})")
        lines.append(f"  Near zero (<0.01): {dist['floor_effects']['n_near_zero']:,} "
                    f"({dist['floor_effects']['pct_near_zero']:.1%})")
        lines.append(f"  Positive: {dist['floor_effects']['n_positive']:,} "
                    f"({dist['floor_effects']['pct_positive']:.1%})")
        lines.append(f"  Severity: {dist['floor_effects']['severity']}")
    lines.append("")
    
    # Alternative aggregations
    lines.append("2. ALTERNATIVE AGGREGATIONS")
    lines.append("-" * 50)
    
    aggs = compare_aggregations()
    
    for method, result in aggs.items():
        if result.get('available', False):
            lines.append(f"\n{method.upper()}:")
            for key, value in result.items():
                if key != 'available':
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")
    lines.append("")
    
    # Non-zero subsample analysis
    lines.append("3. NON-ZERO SUBSAMPLE ANALYSIS")
    lines.append("-" * 50)
    
    try:
        nonzero = analyze_nonzero_subsample()
        
        if 'error' not in nonzero:
            lines.append(f"Total users: {nonzero['n_total']:,}")
            lines.append(f"Users with non-zero AnthroScore: {nonzero['n_nonzero']:,} "
                        f"({nonzero['pct_nonzero']:.1%})")
            
            if 'regression' in nonzero.get('subsample_analysis', {}):
                reg = nonzero['subsample_analysis']['regression']
                if 'error' not in reg:
                    lines.append(f"\nRegression on non-zero subsample:")
                    lines.append(f"  N: {reg['n']}")
                    lines.append(f"  R²: {reg['r2']:.6f}")
                    lines.append(f"  F p-value: {reg['f_pvalue']:.4f}")
                else:
                    lines.append(f"\nRegression error: {reg['error']}")
    except Exception as e:
        lines.append(f"Error in subsample analysis: {e}")
    lines.append("")
    
    # Interpretation
    lines.append("4. INTERPRETATION")
    lines.append("-" * 50)
    lines.append("The distribution shows significant floor effects:")
    lines.append("  - Most users have zero or near-zero AnthroScore")
    lines.append("  - This limits variance available for demographics to explain")
    lines.append("  - The null demographic effects may partially reflect this")
    lines.append("")
    lines.append("Recommendations:")
    lines.append("  1. Report floor effects prominently in limitations")
    lines.append("  2. Consider analyzing non-zero users separately")
    lines.append("  3. Try alternative aggregations (max, pct_nonzero)")
    lines.append("  4. Acknowledge that demographics may not predict WHO")
    lines.append("     anthropomorphizes, even if they might predict HOW MUCH")
    
    report = '\n'.join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Distribution report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = generate_distribution_report()
    print(report)

