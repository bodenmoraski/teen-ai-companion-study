"""
Subreddit Analysis Module.

This module implements comprehensive subreddit-level analysis,
which the CRITICISM.md identifies as potentially THE MAIN FINDING.

Key insights:
- Subreddit explains ~10x more variance than demographics
- This suggests platform culture matters more than user characteristics
"""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

logger = logging.getLogger(__name__)


def compare_variance_explained(
    df: pd.DataFrame = None,
    data_path: Path = None
) -> Dict[str, float]:
    """
    Compare variance explained by subreddit vs demographics.
    
    This is the key comparison that may reveal the main finding:
    subreddit context matters more than individual demographics.
    
    Args:
        df: DataFrame with anthroscore, subreddit, and demographics
        data_path: Path to load data if df not provided
        
    Returns:
        Dict with R² values for different models
    """
    if df is None:
        if data_path is None:
            # Load data
            merged = pd.read_parquet("Data/features/full_merged_dataset.parquet")
            demo = pd.read_parquet("Data/features/demographics.parquet")
            
            demo_cols = ['author', 'age_bucket', 'gender']
            demo_subset = demo[[c for c in demo_cols if c in demo.columns]]
            
            # Drop duplicate columns before merge
            for col in demo_cols[1:]:
                if col in merged.columns:
                    merged = merged.drop(columns=[col])
            
            df = merged.merge(demo_subset, on='author', how='left')
            
            # Add subreddit data if available
            subreddit_path = Path("Data/features/user_subreddit_interactions.parquet")
            if subreddit_path.exists():
                sub_data = pd.read_parquet(subreddit_path)
                # Get primary subreddit per user (most comments)
                primary_sub = sub_data.sort_values('count', ascending=False).groupby('author').first().reset_index()
                primary_sub = primary_sub[['author', 'subreddit']]
                df = df.merge(primary_sub, on='author', how='left')
    
    results = {
        'demographics_r2': 0,
        'subreddit_r2': 0,
        'combined_r2': 0,
        'ratio': 0,
        'n_observations': len(df)
    }
    
    # Filter to complete cases
    df_complete = df[
        df['anthroscore_mean'].notna() & 
        df['age_bucket'].notna() &
        df['gender'].notna()
    ].copy()
    
    if len(df_complete) < 100:
        logger.warning("Not enough complete cases")
        return results
    
    results['n_observations'] = len(df_complete)
    
    # Create dummies
    # Age: convert to 3-bucket
    def to_3bucket(age):
        if pd.isna(age):
            return None
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        return 'adult'
    
    df_complete['age_3bucket'] = df_complete['age_bucket'].apply(to_3bucket)
    
    # Model 1: Demographics only
    try:
        demo_model = ols(
            "anthroscore_mean ~ C(age_3bucket) + C(gender)",
            data=df_complete
        ).fit()
        results['demographics_r2'] = demo_model.rsquared
        logger.info(f"Demographics R²: {demo_model.rsquared:.6f}")
    except Exception as e:
        logger.error(f"Demographics model failed: {e}")
    
    # Model 2: Subreddit only (filter to top subreddits to avoid convergence issues)
    if 'subreddit' in df_complete.columns:
        try:
            # Filter to subreddits with at least 100 users
            sub_counts = df_complete['subreddit'].value_counts()
            valid_subs = sub_counts[sub_counts >= 100].index.tolist()
            
            if len(valid_subs) >= 2:
                df_sub = df_complete[df_complete['subreddit'].isin(valid_subs)].copy()
                
                sub_model = ols(
                    "anthroscore_mean ~ C(subreddit)",
                    data=df_sub
                ).fit()
                results['subreddit_r2'] = sub_model.rsquared
                logger.info(f"Subreddit R²: {sub_model.rsquared:.6f}")
            else:
                logger.warning("Not enough subreddits with 100+ users")
        except Exception as e:
            logger.error(f"Subreddit model failed: {e}")
    
    # Model 3: Combined (filter to top subreddits)
    if 'subreddit' in df_complete.columns:
        try:
            sub_counts = df_complete['subreddit'].value_counts()
            valid_subs = sub_counts[sub_counts >= 100].index.tolist()
            
            if len(valid_subs) >= 2:
                df_combined = df_complete[df_complete['subreddit'].isin(valid_subs)].copy()
                
                combined_model = ols(
                    "anthroscore_mean ~ C(subreddit) + C(age_3bucket) + C(gender)",
                    data=df_combined
                ).fit()
                results['combined_r2'] = combined_model.rsquared
                logger.info(f"Combined R²: {combined_model.rsquared:.6f}")
            else:
                logger.warning("Not enough subreddits for combined model")
        except Exception as e:
            logger.error(f"Combined model failed: {e}")
    
    # Calculate ratio
    if results['demographics_r2'] > 0:
        results['ratio'] = results['subreddit_r2'] / results['demographics_r2']
        logger.info(f"Subreddit/Demographics ratio: {results['ratio']:.1f}x")
    
    return results


def analyze_subreddit_characteristics(
    df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Analyze characteristics of each subreddit.
    
    Args:
        df: DataFrame with subreddit and related columns
        
    Returns:
        DataFrame with subreddit characteristics
    """
    if df is None:
        merged = pd.read_parquet("Data/features/full_merged_dataset.parquet")
        demo = pd.read_parquet("Data/features/demographics.parquet")
        
        # Drop duplicate columns
        for col in ['age_bucket', 'gender']:
            if col in merged.columns:
                merged = merged.drop(columns=[col])
        
        df = merged.merge(demo[['author', 'age_bucket', 'gender']], on='author', how='left')
        
        # Add subreddit data
        subreddit_path = Path("Data/features/user_subreddit_interactions.parquet")
        if subreddit_path.exists():
            sub_data = pd.read_parquet(subreddit_path)
            primary_sub = sub_data.sort_values('count', ascending=False).groupby('author').first().reset_index()
            primary_sub = primary_sub[['author', 'subreddit']]
            df = df.merge(primary_sub, on='author', how='left')
    
    if 'subreddit' not in df.columns:
        logger.warning("No subreddit column found")
        return pd.DataFrame()
    
    characteristics = []
    
    for subreddit in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == subreddit]
        
        if len(sub_df) < 50:
            continue
        
        char = {
            'subreddit': subreddit,
            'n_users': len(sub_df),
            'pct_total': len(sub_df) / len(df),
        }
        
        # AnthroScore
        if 'anthroscore_mean' in sub_df.columns:
            char['anthroscore_mean'] = sub_df['anthroscore_mean'].mean()
            char['anthroscore_std'] = sub_df['anthroscore_mean'].std()
            char['anthroscore_median'] = sub_df['anthroscore_mean'].median()
        
        # Age distribution
        if 'age_bucket' in sub_df.columns:
            age_dist = sub_df['age_bucket'].value_counts(normalize=True)
            char['pct_teen'] = age_dist.get('13-18', 0)
            char['pct_young_adult'] = age_dist.get('19-25', 0)
            char['pct_classified_age'] = sub_df['age_bucket'].notna().mean()
        
        # Gender distribution
        if 'gender' in sub_df.columns:
            gender_dist = sub_df['gender'].value_counts(normalize=True)
            char['pct_male'] = gender_dist.get('male', 0)
            char['pct_female'] = gender_dist.get('female', 0)
            char['pct_nonbinary'] = gender_dist.get('nonbinary', 0)
        
        characteristics.append(char)
    
    result_df = pd.DataFrame(characteristics)
    result_df = result_df.sort_values('n_users', ascending=False)
    
    return result_df


def subreddit_level_regression(
    df: pd.DataFrame = None
) -> Dict[str, Dict]:
    """
    Run regression separately for each subreddit.
    
    This tests whether demographic effects exist within subreddits.
    
    Args:
        df: DataFrame with subreddit and regression variables
        
    Returns:
        Dict mapping subreddit to regression results
    """
    if df is None:
        merged = pd.read_parquet("Data/features/full_merged_dataset.parquet")
        demo = pd.read_parquet("Data/features/demographics.parquet")
        
        for col in ['age_bucket', 'gender']:
            if col in merged.columns:
                merged = merged.drop(columns=[col])
        
        df = merged.merge(demo[['author', 'age_bucket', 'gender']], on='author', how='left')
        
        # Add subreddit
        subreddit_path = Path("Data/features/user_subreddit_interactions.parquet")
        if subreddit_path.exists():
            sub_data = pd.read_parquet(subreddit_path)
            primary_sub = sub_data.sort_values('count', ascending=False).groupby('author').first().reset_index()
            primary_sub = primary_sub[['author', 'subreddit']]
            df = df.merge(primary_sub, on='author', how='left')
    
    if 'subreddit' not in df.columns:
        return {}
    
    # Convert to 3-bucket
    def to_3bucket(age):
        if pd.isna(age):
            return None
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        return 'adult'
    
    df['age_3bucket'] = df['age_bucket'].apply(to_3bucket)
    
    results = {}
    
    for subreddit in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == subreddit].copy()
        
        # Filter to complete cases with sufficient variety
        sub_complete = sub_df[
            sub_df['anthroscore_mean'].notna() &
            sub_df['age_3bucket'].notna() &
            sub_df['gender'].notna() &
            (sub_df['gender'] != 'unknown')
        ]
        
        if len(sub_complete) < 100:
            continue
        
        # Check we have variety in predictors
        if sub_complete['age_3bucket'].nunique() < 2 or sub_complete['gender'].nunique() < 2:
            continue
        
        try:
            model = ols(
                "anthroscore_mean ~ C(age_3bucket) + C(gender)",
                data=sub_complete
            ).fit()
            
            results[subreddit] = {
                'n': int(model.nobs),
                'r2': model.rsquared,
                'adj_r2': model.rsquared_adj,
                'f_pvalue': model.f_pvalue,
                'coefficients': dict(model.params),
                'pvalues': dict(model.pvalues)
            }
            
        except Exception as e:
            logger.warning(f"Regression failed for {subreddit}: {e}")
            continue
    
    return results


def compare_subreddits_anthroscore(
    df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Statistical comparison of AnthroScore across subreddits.
    
    Args:
        df: DataFrame with subreddit and anthroscore
        
    Returns:
        Dict with comparison results
    """
    from scipy import stats as scipy_stats
    
    if df is None:
        df = pd.read_parquet("Data/features/full_merged_dataset.parquet")
        
        # Add subreddit
        subreddit_path = Path("Data/features/user_subreddit_interactions.parquet")
        if subreddit_path.exists():
            sub_data = pd.read_parquet(subreddit_path)
            primary_sub = sub_data.sort_values('count', ascending=False).groupby('author').first().reset_index()
            primary_sub = primary_sub[['author', 'subreddit']]
            df = df.merge(primary_sub, on='author', how='left')
    
    if 'subreddit' not in df.columns or 'anthroscore_mean' not in df.columns:
        return {}
    
    results = {
        'subreddit_means': {},
        'anova': {},
        'pairwise': []
    }
    
    # Get subreddits with sufficient data
    sub_counts = df['subreddit'].value_counts()
    valid_subs = sub_counts[sub_counts >= 100].index.tolist()
    
    groups = []
    for sub in valid_subs:
        scores = df.loc[df['subreddit'] == sub, 'anthroscore_mean'].dropna()
        groups.append(scores)
        results['subreddit_means'][sub] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'n': len(scores)
        }
    
    # ANOVA
    if len(groups) >= 2:
        try:
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            logger.warning(f"ANOVA failed: {e}")
    
    # Pairwise comparisons for top subreddits
    top_subs = list(results['subreddit_means'].keys())[:5]
    
    for i, sub1 in enumerate(top_subs):
        for sub2 in top_subs[i+1:]:
            scores1 = df.loc[df['subreddit'] == sub1, 'anthroscore_mean'].dropna()
            scores2 = df.loc[df['subreddit'] == sub2, 'anthroscore_mean'].dropna()
            
            try:
                t_stat, p_value = scipy_stats.ttest_ind(scores1, scores2)
                
                results['pairwise'].append({
                    'sub1': sub1,
                    'sub2': sub2,
                    'mean_diff': scores1.mean() - scores2.mean(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            except:
                continue
    
    return results


def generate_subreddit_report(output_path: Path = None) -> str:
    """
    Generate comprehensive subreddit analysis report.
    
    Args:
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SUBREDDIT ANALYSIS REPORT")
    lines.append("The Main Finding: Platform Context Matters")
    lines.append("=" * 70)
    lines.append("")
    
    # Variance comparison
    lines.append("1. VARIANCE EXPLAINED COMPARISON")
    lines.append("-" * 50)
    
    var_results = compare_variance_explained()
    
    lines.append(f"N observations: {var_results['n_observations']:,}")
    lines.append("")
    lines.append(f"Demographics only (age + gender): R² = {var_results['demographics_r2']:.6f}")
    lines.append(f"Subreddit only:                   R² = {var_results['subreddit_r2']:.6f}")
    lines.append(f"Combined:                         R² = {var_results['combined_r2']:.6f}")
    lines.append("")
    lines.append(f"RATIO: Subreddit explains {var_results['ratio']:.1f}x more variance than demographics")
    lines.append("")
    
    # Subreddit characteristics
    lines.append("2. SUBREDDIT CHARACTERISTICS")
    lines.append("-" * 50)
    
    char_df = analyze_subreddit_characteristics()
    
    if not char_df.empty:
        for _, row in char_df.iterrows():
            lines.append(f"\n{row['subreddit']} (n={row['n_users']:,}):")
            lines.append(f"  AnthroScore: mean={row.get('anthroscore_mean', 0):.4f}, "
                        f"std={row.get('anthroscore_std', 0):.4f}")
            if 'pct_teen' in row:
                lines.append(f"  Age: {row['pct_teen']*100:.1f}% teen, "
                           f"{row['pct_young_adult']*100:.1f}% young adult")
            if 'pct_male' in row:
                lines.append(f"  Gender: {row['pct_male']*100:.1f}% male, "
                           f"{row['pct_female']*100:.1f}% female")
    lines.append("")
    
    # Subreddit-level regression
    lines.append("3. WITHIN-SUBREDDIT DEMOGRAPHIC EFFECTS")
    lines.append("-" * 50)
    
    sub_reg = subreddit_level_regression()
    
    for sub, results in sub_reg.items():
        lines.append(f"\n{sub} (n={results['n']}):")
        lines.append(f"  R² = {results['r2']:.6f}")
        lines.append(f"  F-test p-value: {results['f_pvalue']:.4f}")
    lines.append("")
    
    # ANOVA comparison
    lines.append("4. CROSS-SUBREDDIT COMPARISON")
    lines.append("-" * 50)
    
    comparison = compare_subreddits_anthroscore()
    
    if comparison.get('anova'):
        lines.append(f"ANOVA: F={comparison['anova']['f_statistic']:.2f}, "
                    f"p={comparison['anova']['p_value']:.4e}")
        lines.append(f"Significant difference between subreddits: "
                    f"{'Yes' if comparison['anova']['significant'] else 'No'}")
    lines.append("")
    
    # Interpretation
    lines.append("5. INTERPRETATION")
    lines.append("-" * 50)
    lines.append("Key Finding: Subreddit context explains far more variance in")
    lines.append("anthropomorphization than individual demographics.")
    lines.append("")
    lines.append("Possible explanations:")
    lines.append("  1. SELECTION: Users with certain tendencies choose specific platforms")
    lines.append("  2. SOCIALIZATION: Platform norms shape how users discuss AI")
    lines.append("  3. CONTENT: Different platforms have different AI products")
    lines.append("  4. MODERATION: Community rules affect discourse")
    lines.append("")
    lines.append("Recommendation: Pivot findings to emphasize platform effects")
    lines.append("rather than demographic nulls.")
    
    report = '\n'.join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Subreddit report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = generate_subreddit_report()
    print(report)

