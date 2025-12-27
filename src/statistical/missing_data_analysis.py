"""
Missing Data Analysis Module.

This module analyzes patterns of missingness in demographic data
and assesses potential selection bias.

Key concerns from CRITICISM.md:
- 32% missing age, 40.5% missing gender
- These users are excluded from regression
- Missingness may not be random (MNAR)
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def analyze_missingness(
    demographics_path: Path = None,
    anthroscores_path: Path = None
) -> Dict[str, Any]:
    """
    Analyze patterns of missing demographic data.
    
    Args:
        demographics_path: Path to demographics parquet
        anthroscores_path: Path to anthroscores parquet
        
    Returns:
        Dict with missingness analysis results
    """
    if demographics_path is None:
        demographics_path = Path("Data/features/demographics.parquet")
    if anthroscores_path is None:
        anthroscores_path = Path("Data/features/user_anthroscores.parquet")
    
    demo = pd.read_parquet(demographics_path)
    anthro = pd.read_parquet(anthroscores_path)
    
    n_total = len(demo)
    
    results = {
        'n_total': n_total,
        'n_missing_age': 0,
        'n_missing_gender': 0,
        'pct_missing_age': 0.0,
        'pct_missing_gender': 0.0,
        'missingness_correlation_anthroscore': None,
        'mcar_test': {},
        'recommendations': []
    }
    
    # Count missing
    n_missing_age = demo['age_bucket'].isna().sum()
    n_unknown_gender = (demo['gender'].isna() | (demo['gender'] == 'unknown')).sum()
    
    results['n_missing_age'] = n_missing_age
    results['n_missing_gender'] = n_unknown_gender
    results['pct_missing_age'] = n_missing_age / n_total
    results['pct_missing_gender'] = n_unknown_gender / n_total
    
    # Merge with AnthroScore
    merged = demo.merge(anthro, on='author', how='inner')
    
    # Test if missingness is related to AnthroScore
    has_age = merged['age_bucket'].notna()
    
    if has_age.sum() > 100 and (~has_age).sum() > 100:
        scores_with = merged.loc[has_age, 'anthroscore_mean'].dropna()
        scores_without = merged.loc[~has_age, 'anthroscore_mean'].dropna()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(scores_with, scores_without)
        
        results['missingness_correlation_anthroscore'] = {
            'mean_with_age': scores_with.mean(),
            'mean_without_age': scores_without.mean(),
            'mean_diff': scores_with.mean() - scores_without.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'mnar': p_value < 0.05  # Missing Not At Random
        }
        
        if p_value < 0.05:
            results['recommendations'].append(
                "WARNING: Missingness is associated with outcome (MNAR). "
                "Consider including 'unknown' as category or use selection models."
            )
        else:
            results['recommendations'].append(
                "Missingness appears unrelated to outcome (MAR). "
                "Exclusion of missing cases is less biased."
            )
    
    # Gender missingness analysis
    has_gender = (merged['gender'].notna()) & (merged['gender'] != 'unknown')
    
    if has_gender.sum() > 100 and (~has_gender).sum() > 100:
        scores_with_g = merged.loc[has_gender, 'anthroscore_mean'].dropna()
        scores_without_g = merged.loc[~has_gender, 'anthroscore_mean'].dropna()
        
        t_stat_g, p_value_g = stats.ttest_ind(scores_with_g, scores_without_g)
        
        results['gender_missingness_correlation'] = {
            'mean_with_gender': scores_with_g.mean(),
            'mean_without_gender': scores_without_g.mean(),
            't_statistic': t_stat_g,
            'p_value': p_value_g,
            'mnar': p_value_g < 0.05
        }
    
    # Cross-tabulation of missingness
    results['missingness_pattern'] = {
        'both_present': (demo['age_bucket'].notna() & 
                        (demo['gender'].notna()) & 
                        (demo['gender'] != 'unknown')).sum(),
        'age_only': (demo['age_bucket'].notna() & 
                    ((demo['gender'].isna()) | (demo['gender'] == 'unknown'))).sum(),
        'gender_only': (demo['age_bucket'].isna() & 
                       (demo['gender'].notna()) & 
                       (demo['gender'] != 'unknown')).sum(),
        'both_missing': (demo['age_bucket'].isna() & 
                        ((demo['gender'].isna()) | (demo['gender'] == 'unknown'))).sum()
    }
    
    return results


def compare_with_unknown_category(
    merged_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Compare regression results including "unknown" as a category vs excluding.
    
    Args:
        merged_df: DataFrame with demographics and anthroscore
        
    Returns:
        Dict with comparison results
    """
    from statsmodels.formula.api import ols
    
    if merged_df is None:
        merged = pd.read_parquet("Data/features/full_merged_dataset.parquet")
        demo = pd.read_parquet("Data/features/demographics.parquet")
        merged_df = merged.merge(demo[['author', 'age_bucket', 'gender']], on='author', how='left')
    
    results = {
        'excluding_missing': {},
        'including_unknown': {},
        'comparison': {}
    }
    
    # Convert age to 3-bucket
    def to_3bucket(age):
        if pd.isna(age):
            return 'unknown'
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        return 'adult'
    
    df = merged_df.copy()
    df['age_3bucket'] = df['age_bucket'].apply(to_3bucket)
    df['gender_clean'] = df['gender'].fillna('unknown')
    
    # Model 1: Excluding missing
    df_complete = df[
        (df['age_3bucket'] != 'unknown') & 
        (df['gender_clean'] != 'unknown') &
        df['anthroscore_mean'].notna()
    ]
    
    if len(df_complete) >= 100:
        try:
            model1 = ols(
                "anthroscore_mean ~ C(age_3bucket) + C(gender_clean)",
                data=df_complete
            ).fit()
            
            results['excluding_missing'] = {
                'n': int(model1.nobs),
                'r2': model1.rsquared,
                'adj_r2': model1.rsquared_adj
            }
        except Exception as e:
            results['excluding_missing'] = {'error': str(e)}
    
    # Model 2: Including unknown as category
    df_all = df[df['anthroscore_mean'].notna()]
    
    if len(df_all) >= 100:
        try:
            model2 = ols(
                "anthroscore_mean ~ C(age_3bucket) + C(gender_clean)",
                data=df_all
            ).fit()
            
            results['including_unknown'] = {
                'n': int(model2.nobs),
                'r2': model2.rsquared,
                'adj_r2': model2.rsquared_adj,
                'unknown_age_coef': model2.params.get('C(age_3bucket)[T.unknown]', np.nan),
                'unknown_gender_coef': model2.params.get('C(gender_clean)[T.unknown]', np.nan)
            }
        except Exception as e:
            results['including_unknown'] = {'error': str(e)}
    
    # Compare
    if 'r2' in results['excluding_missing'] and 'r2' in results['including_unknown']:
        results['comparison'] = {
            'n_increase': results['including_unknown']['n'] - results['excluding_missing']['n'],
            'r2_change': results['including_unknown']['r2'] - results['excluding_missing']['r2']
        }
    
    return results


def generate_missing_data_report(output_path: Path = None) -> str:
    """
    Generate comprehensive missing data analysis report.
    
    Args:
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MISSING DATA ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Basic missingness
    lines.append("1. EXTENT OF MISSINGNESS")
    lines.append("-" * 50)
    
    analysis = analyze_missingness()
    
    lines.append(f"Total users: {analysis['n_total']:,}")
    lines.append(f"Missing age: {analysis['n_missing_age']:,} ({analysis['pct_missing_age']:.1%})")
    lines.append(f"Missing/unknown gender: {analysis['n_missing_gender']:,} ({analysis['pct_missing_gender']:.1%})")
    lines.append("")
    
    if 'missingness_pattern' in analysis:
        pattern = analysis['missingness_pattern']
        lines.append("Missingness pattern:")
        lines.append(f"  Both present: {pattern['both_present']:,}")
        lines.append(f"  Age only: {pattern['age_only']:,}")
        lines.append(f"  Gender only: {pattern['gender_only']:,}")
        lines.append(f"  Both missing: {pattern['both_missing']:,}")
    lines.append("")
    
    # Relationship to outcome
    lines.append("2. MISSINGNESS RELATIONSHIP TO ANTHROSCORE")
    lines.append("-" * 50)
    
    if analysis.get('missingness_correlation_anthroscore'):
        corr = analysis['missingness_correlation_anthroscore']
        lines.append(f"Mean AnthroScore (with age): {corr['mean_with_age']:.4f}")
        lines.append(f"Mean AnthroScore (without age): {corr['mean_without_age']:.4f}")
        lines.append(f"Difference: {corr['mean_diff']:.4f}")
        lines.append(f"T-test p-value: {corr['p_value']:.4f}")
        lines.append(f"MNAR: {'Yes' if corr['mnar'] else 'No'}")
    lines.append("")
    
    # Recommendations
    lines.append("3. RECOMMENDATIONS")
    lines.append("-" * 50)
    for rec in analysis.get('recommendations', []):
        lines.append(f"  • {rec}")
    lines.append("")
    
    # Comparison with unknown category
    lines.append("4. MODEL COMPARISON")
    lines.append("-" * 50)
    
    try:
        comparison = compare_with_unknown_category()
        
        if 'n' in comparison.get('excluding_missing', {}):
            lines.append(f"Excluding missing: N={comparison['excluding_missing']['n']:,}, "
                        f"R²={comparison['excluding_missing']['r2']:.6f}")
        
        if 'n' in comparison.get('including_unknown', {}):
            lines.append(f"Including unknown: N={comparison['including_unknown']['n']:,}, "
                        f"R²={comparison['including_unknown']['r2']:.6f}")
        
        if comparison.get('comparison'):
            lines.append(f"N increase: +{comparison['comparison']['n_increase']:,}")
            lines.append(f"R² change: {comparison['comparison']['r2_change']:+.6f}")
    except Exception as e:
        lines.append(f"Error in comparison: {e}")
    lines.append("")
    
    # Final interpretation
    lines.append("5. INTERPRETATION")
    lines.append("-" * 50)
    lines.append("Missing data is substantial (~40%) but appears to be:")
    lines.append("  - Missing at Random (MAR) if unrelated to AnthroScore")
    lines.append("  - Missing Not at Random (MNAR) if related to AnthroScore")
    lines.append("")
    lines.append("Including 'unknown' as a category allows using full sample")
    lines.append("but may introduce interpretation challenges.")
    lines.append("")
    lines.append("Current approach (exclusion) is conservative but reduces power.")
    
    report = '\n'.join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Missing data report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = generate_missing_data_report()
    print(report)

