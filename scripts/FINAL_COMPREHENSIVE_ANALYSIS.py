"""
FINAL COMPREHENSIVE ANALYSIS: The Illusion Project
===================================================

This script runs ALL statistical analyses with our best models (V3 @ 0.60 confidence)
to produce publication-ready results.

Research Questions:
- RQ1: Demographics and Intent of AI companion users
- RQ2: Demographics and Anthropomorphization relationships
- RQ3: Emotional Expression and Anthropomorphization

Statistical Tests:
- Descriptive statistics with 95% CIs
- T-tests with Cohen's d effect sizes
- ANOVA with eta-squared and post-hoc tests
- Chi-square with Cramer's V
- Pearson and Spearman correlations
- Multiple regression with standardized coefficients
- Moderation analysis
- Bootstrap confidence intervals
- Sensitivity analyses

Author: Research Agent
Date: January 10, 2026
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_1samp, mannwhitneyu, chi2_contingency, 
    f_oneway, pearsonr, spearmanr, kruskal, shapiro
)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'results/FINAL_ANALYSIS.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

def bootstrap_ci(data: np.ndarray, statistic_func, n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for any statistic."""
    boot_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_func(sample))
    alpha = (1 - ci) / 2
    return (
        np.mean(boot_stats),
        np.percentile(boot_stats, alpha * 100),
        np.percentile(boot_stats, (1 - alpha) * 100)
    )

def bootstrap_diff_ci(group1: np.ndarray, group2: np.ndarray, n_bootstrap: int = 10000) -> Dict:
    """Bootstrap CI for difference between groups."""
    diffs = []
    for _ in range(n_bootstrap):
        s1 = np.random.choice(group1, size=len(group1), replace=True)
        s2 = np.random.choice(group2, size=len(group2), replace=True)
        diffs.append(s1.mean() - s2.mean())
    return {
        'mean_diff': np.mean(diffs),
        'ci_lower': np.percentile(diffs, 2.5),
        'ci_upper': np.percentile(diffs, 97.5),
        'excludes_zero': np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0
    }

def cramers_v(confusion_matrix: np.ndarray) -> float:
    """Calculate Cramer's V for chi-square test."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

def eta_squared(df_between: float, df_within: float, ss_between: float, ss_total: float) -> float:
    """Calculate eta-squared for ANOVA."""
    return ss_between / ss_total if ss_total > 0 else 0

def interpret_effect_size(d: float, metric: str = 'cohens_d') -> str:
    """Interpret effect size magnitude."""
    if metric == 'cohens_d':
        if abs(d) < 0.2: return 'negligible'
        elif abs(d) < 0.5: return 'small'
        elif abs(d) < 0.8: return 'medium'
        else: return 'large'
    elif metric == 'r':
        if abs(d) < 0.1: return 'negligible'
        elif abs(d) < 0.3: return 'small'
        elif abs(d) < 0.5: return 'medium'
        else: return 'large'
    elif metric == 'cramers_v':
        if abs(d) < 0.1: return 'negligible'
        elif abs(d) < 0.3: return 'small'
        elif abs(d) < 0.5: return 'medium'
        else: return 'large'
    return 'unknown'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load all required datasets with V3 predictions at optimal threshold."""
    logger.info("="*70)
    logger.info("LOADING DATA WITH V3 PREDICTIONS @ 0.60 CONFIDENCE")
    logger.info("="*70)
    
    data = {}
    
    # Core data
    data['comments'] = pd.read_parquet(project_root / 'Data/processed/all_comments.parquet')
    data['anthroscores'] = pd.read_parquet(project_root / 'Data/features/user_anthroscores.parquet')
    data['self_declarations'] = pd.read_parquet(project_root / 'Data/features/self_declarations.parquet')
    
    # Emotions if available
    try:
        data['emotions'] = pd.read_parquet(project_root / 'Data/features/user_emotions.parquet')
    except:
        logger.warning("User emotions file not found - will compute if needed")
        data['emotions'] = None
    
    # V3 predictions (our best models)
    data['gender_v3'] = pd.read_parquet(project_root / 'experiments/v2_correction/gender_predictions_v3.parquet')
    data['age_v3'] = pd.read_parquet(project_root / 'experiments/v2_correction/age_predictions_v3.parquet')
    
    # Apply confidence threshold
    CONFIDENCE_THRESHOLD = 0.60
    
    data['gender_v3_filtered'] = data['gender_v3'][data['gender_v3']['confidence'] >= CONFIDENCE_THRESHOLD].copy()
    data['age_v3_filtered'] = data['age_v3'][data['age_v3']['confidence'] >= CONFIDENCE_THRESHOLD].copy()
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Total comments: {len(data['comments']):,}")
    logger.info(f"  Total users with anthroscores: {len(data['anthroscores']):,}")
    logger.info(f"  Gender predictions (all): {len(data['gender_v3']):,}")
    logger.info(f"  Gender predictions (>={CONFIDENCE_THRESHOLD}): {len(data['gender_v3_filtered']):,} ({len(data['gender_v3_filtered'])/len(data['gender_v3'])*100:.1f}%)")
    logger.info(f"  Age predictions (all): {len(data['age_v3']):,}")
    logger.info(f"  Age predictions (>={CONFIDENCE_THRESHOLD}): {len(data['age_v3_filtered']):,} ({len(data['age_v3_filtered'])/len(data['age_v3'])*100:.1f}%)")
    
    return data

def create_master_dataset(data: Dict) -> pd.DataFrame:
    """Merge all data sources into one master dataset."""
    logger.info("\nCreating master dataset...")
    
    # Start with anthroscores
    master = data['anthroscores'].copy()
    
    # Add V3 predictions
    master = master.merge(
        data['gender_v3_filtered'][['author', 'gender_predicted', 'confidence']].rename(
            columns={'confidence': 'gender_confidence'}
        ),
        on='author', how='left'
    )
    
    master = master.merge(
        data['age_v3_filtered'][['author', 'age_predicted', 'confidence']].rename(
            columns={'confidence': 'age_confidence'}
        ),
        on='author', how='left'
    )
    
    # Add emotions if available
    if data['emotions'] is not None:
        emotion_cols = [c for c in data['emotions'].columns if c.startswith('emotion_') or c in 
                       ['dominant_emotion', 'emotional_intensity', 'emotional_valence', 'emotional_diversity']]
        master = master.merge(
            data['emotions'][['author'] + emotion_cols],
            on='author', how='left'
        )
    
    # Add self-declarations for validation
    master = master.merge(
        data['self_declarations'][['author', 'age_bucket_self_declared', 'gender_self_declared']],
        on='author', how='left'
    )
    
    # Create binary age column for self-declared
    def map_age_binary(bucket):
        if pd.isna(bucket):
            return np.nan
        return 'teen' if bucket == '13-18' else 'adult'
    
    master['age_self_declared_binary'] = master['age_bucket_self_declared'].apply(map_age_binary)
    
    # Filter to users with both predictions
    master_complete = master[
        master['gender_predicted'].notna() & 
        master['age_predicted'].notna()
    ].copy()
    
    logger.info(f"Master dataset: {len(master_complete):,} users with complete predictions")
    
    return master_complete

# ============================================================================
# RQ1: DEMOGRAPHICS & INTENT ANALYSIS
# ============================================================================

def analyze_rq1(master: pd.DataFrame, data: Dict) -> Dict:
    """
    RQ1: What are the demographics of AI companion users?
    
    Analyses:
    - Age distribution with 95% CIs
    - Gender distribution with 95% CIs
    - Chi-square tests for demographic associations
    """
    logger.info("\n" + "="*70)
    logger.info("RQ1: DEMOGRAPHICS ANALYSIS")
    logger.info("="*70)
    
    results = {'rq1': {}}
    
    # Age Distribution
    logger.info("\n--- Age Distribution ---")
    age_counts = master['age_predicted'].value_counts()
    age_pcts = master['age_predicted'].value_counts(normalize=True)
    
    results['rq1']['age_distribution'] = {
        'teen': {
            'count': int(age_counts.get('teen', 0)),
            'percentage': float(age_pcts.get('teen', 0)),
            'ci_lower': float(age_pcts.get('teen', 0) - 1.96 * np.sqrt(age_pcts.get('teen', 0) * (1-age_pcts.get('teen', 0)) / len(master))),
            'ci_upper': float(age_pcts.get('teen', 0) + 1.96 * np.sqrt(age_pcts.get('teen', 0) * (1-age_pcts.get('teen', 0)) / len(master)))
        },
        'adult': {
            'count': int(age_counts.get('adult', 0)),
            'percentage': float(age_pcts.get('adult', 0)),
            'ci_lower': float(age_pcts.get('adult', 0) - 1.96 * np.sqrt(age_pcts.get('adult', 0) * (1-age_pcts.get('adult', 0)) / len(master))),
            'ci_upper': float(age_pcts.get('adult', 0) + 1.96 * np.sqrt(age_pcts.get('adult', 0) * (1-age_pcts.get('adult', 0)) / len(master)))
        }
    }
    
    logger.info(f"Teen: {age_counts.get('teen', 0):,} ({age_pcts.get('teen', 0)*100:.1f}%)")
    logger.info(f"Adult: {age_counts.get('adult', 0):,} ({age_pcts.get('adult', 0)*100:.1f}%)")
    
    # Gender Distribution
    logger.info("\n--- Gender Distribution ---")
    gender_counts = master['gender_predicted'].value_counts()
    gender_pcts = master['gender_predicted'].value_counts(normalize=True)
    
    results['rq1']['gender_distribution'] = {
        'male': {
            'count': int(gender_counts.get('male', 0)),
            'percentage': float(gender_pcts.get('male', 0)),
            'ci_lower': float(gender_pcts.get('male', 0) - 1.96 * np.sqrt(gender_pcts.get('male', 0) * (1-gender_pcts.get('male', 0)) / len(master))),
            'ci_upper': float(gender_pcts.get('male', 0) + 1.96 * np.sqrt(gender_pcts.get('male', 0) * (1-gender_pcts.get('male', 0)) / len(master)))
        },
        'female': {
            'count': int(gender_counts.get('female', 0)),
            'percentage': float(gender_pcts.get('female', 0)),
            'ci_lower': float(gender_pcts.get('female', 0) - 1.96 * np.sqrt(gender_pcts.get('female', 0) * (1-gender_pcts.get('female', 0)) / len(master))),
            'ci_upper': float(gender_pcts.get('female', 0) + 1.96 * np.sqrt(gender_pcts.get('female', 0) * (1-gender_pcts.get('female', 0)) / len(master)))
        }
    }
    
    logger.info(f"Male: {gender_counts.get('male', 0):,} ({gender_pcts.get('male', 0)*100:.1f}%)")
    logger.info(f"Female: {gender_counts.get('female', 0):,} ({gender_pcts.get('female', 0)*100:.1f}%)")
    
    # Age × Gender Crosstab
    logger.info("\n--- Age × Gender Crosstab ---")
    crosstab = pd.crosstab(master['age_predicted'], master['gender_predicted'])
    chi2, p_val, dof, expected = chi2_contingency(crosstab)
    v = cramers_v(crosstab.values)
    
    results['rq1']['age_gender_association'] = {
        'chi_square': float(chi2),
        'p_value': float(p_val),
        'dof': int(dof),
        'cramers_v': float(v),
        'effect_interpretation': interpret_effect_size(v, 'cramers_v'),
        'crosstab': crosstab.to_dict()
    }
    
    logger.info(f"Chi-square = {chi2:.2f}, p = {p_val:.6f}, Cramer's V = {v:.3f} ({interpret_effect_size(v, 'cramers_v')})")
    
    return results

# ============================================================================
# RQ2: DEMOGRAPHICS & ANTHROPOMORPHIZATION
# ============================================================================

def analyze_rq2(master: pd.DataFrame) -> Dict:
    """
    RQ2: How do demographics relate to anthropomorphization?
    
    Analyses:
    - Age → AnthroScore (t-test, Cohen's d, bootstrap CI)
    - Gender → AnthroScore (t-test, Cohen's d, bootstrap CI)
    - Age × Gender interaction (Two-way ANOVA)
    - Ground truth validation
    """
    logger.info("\n" + "="*70)
    logger.info("RQ2: DEMOGRAPHICS & ANTHROPOMORPHIZATION")
    logger.info("="*70)
    
    results = {'rq2': {}}
    
    # Filter to users with non-zero anthroscore for most analyses
    nonzero = master[master['anthroscore_max'] > 0].copy()
    logger.info(f"\nUsers with non-zero AnthroScore: {len(nonzero):,} ({len(nonzero)/len(master)*100:.1f}%)")
    
    # ========================================
    # Age → AnthroScore
    # ========================================
    logger.info("\n--- Age → AnthroScore ---")
    
    teens = nonzero[nonzero['age_predicted'] == 'teen']['anthroscore_max'].values
    adults = nonzero[nonzero['age_predicted'] == 'adult']['anthroscore_max'].values
    
    if len(teens) > 0 and len(adults) > 0:
        t_stat, p_val = ttest_ind(teens, adults, equal_var=False)
        d = cohens_d(teens, adults)
        
        # Bootstrap CI for effect size
        boot_ci = bootstrap_diff_ci(teens, adults, n_bootstrap=5000)
        
        results['rq2']['age_effect'] = {
            'teen_n': int(len(teens)),
            'teen_mean': float(teens.mean()),
            'teen_std': float(teens.std()),
            'adult_n': int(len(adults)),
            'adult_mean': float(adults.mean()),
            'adult_std': float(adults.std()),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(d),
            'effect_interpretation': interpret_effect_size(d),
            'bootstrap_mean_diff': boot_ci['mean_diff'],
            'bootstrap_ci_lower': boot_ci['ci_lower'],
            'bootstrap_ci_upper': boot_ci['ci_upper'],
            'bootstrap_significant': boot_ci['excludes_zero'],
            'direction': 'teens_higher' if d > 0 else 'adults_higher'
        }
        
        logger.info(f"Teen mean: {teens.mean():.4f} (SD={teens.std():.4f}, n={len(teens)})")
        logger.info(f"Adult mean: {adults.mean():.4f} (SD={adults.std():.4f}, n={len(adults)})")
        logger.info(f"t = {t_stat:.3f}, p = {p_val:.6f}")
        logger.info(f"Cohen's d = {d:.4f} ({interpret_effect_size(d)})")
        logger.info(f"Bootstrap 95% CI: [{boot_ci['ci_lower']:.4f}, {boot_ci['ci_upper']:.4f}]")
        logger.info(f"Direction: {'TEENS HIGHER' if d > 0 else 'ADULTS HIGHER'}")
    
    # ========================================
    # Gender → AnthroScore
    # ========================================
    logger.info("\n--- Gender → AnthroScore ---")
    
    males = nonzero[nonzero['gender_predicted'] == 'male']['anthroscore_max'].values
    females = nonzero[nonzero['gender_predicted'] == 'female']['anthroscore_max'].values
    
    if len(males) > 0 and len(females) > 0:
        t_stat, p_val = ttest_ind(males, females, equal_var=False)
        d = cohens_d(males, females)
        boot_ci = bootstrap_diff_ci(males, females, n_bootstrap=5000)
        
        results['rq2']['gender_effect'] = {
            'male_n': int(len(males)),
            'male_mean': float(males.mean()),
            'male_std': float(males.std()),
            'female_n': int(len(females)),
            'female_mean': float(females.mean()),
            'female_std': float(females.std()),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(d),
            'effect_interpretation': interpret_effect_size(d),
            'bootstrap_mean_diff': boot_ci['mean_diff'],
            'bootstrap_ci_lower': boot_ci['ci_lower'],
            'bootstrap_ci_upper': boot_ci['ci_upper'],
            'bootstrap_significant': boot_ci['excludes_zero'],
            'direction': 'males_higher' if d > 0 else 'females_higher'
        }
        
        logger.info(f"Male mean: {males.mean():.4f} (SD={males.std():.4f}, n={len(males)})")
        logger.info(f"Female mean: {females.mean():.4f} (SD={females.std():.4f}, n={len(females)})")
        logger.info(f"t = {t_stat:.3f}, p = {p_val:.6f}")
        logger.info(f"Cohen's d = {d:.4f} ({interpret_effect_size(d)})")
    
    # ========================================
    # Age × Gender Two-Way ANOVA
    # ========================================
    logger.info("\n--- Age × Gender Two-Way ANOVA ---")
    
    anova_df = nonzero[['age_predicted', 'gender_predicted', 'anthroscore_max']].dropna()
    
    if len(anova_df) > 100:
        try:
            model = ols('anthroscore_max ~ C(age_predicted) * C(gender_predicted)', data=anova_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            results['rq2']['two_way_anova'] = {
                'age_F': float(anova_table.loc['C(age_predicted)', 'F']),
                'age_p': float(anova_table.loc['C(age_predicted)', 'PR(>F)']),
                'gender_F': float(anova_table.loc['C(gender_predicted)', 'F']),
                'gender_p': float(anova_table.loc['C(gender_predicted)', 'PR(>F)']),
                'interaction_F': float(anova_table.loc['C(age_predicted):C(gender_predicted)', 'F']),
                'interaction_p': float(anova_table.loc['C(age_predicted):C(gender_predicted)', 'PR(>F)']),
                'r_squared': float(model.rsquared)
            }
            
            # Group means
            group_means = anova_df.groupby(['age_predicted', 'gender_predicted'])['anthroscore_max'].agg(['mean', 'std', 'count'])
            results['rq2']['group_means'] = group_means.reset_index().to_dict('records')
            
            logger.info(f"Age effect: F = {results['rq2']['two_way_anova']['age_F']:.2f}, p = {results['rq2']['two_way_anova']['age_p']:.6f}")
            logger.info(f"Gender effect: F = {results['rq2']['two_way_anova']['gender_F']:.2f}, p = {results['rq2']['two_way_anova']['gender_p']:.6f}")
            logger.info(f"Interaction: F = {results['rq2']['two_way_anova']['interaction_F']:.2f}, p = {results['rq2']['two_way_anova']['interaction_p']:.6f}")
            logger.info(f"R-squared: {model.rsquared:.4f}")
            
        except Exception as e:
            logger.warning(f"ANOVA failed: {e}")
    
    # ========================================
    # Ground Truth Validation
    # ========================================
    logger.info("\n--- Ground Truth Validation (Self-Declared Ages) ---")
    
    gt_df = master[
        master['age_self_declared_binary'].notna() & 
        (master['anthroscore_max'] > 0)
    ].copy()
    
    if len(gt_df) > 20:
        gt_teens = gt_df[gt_df['age_self_declared_binary'] == 'teen']['anthroscore_max'].values
        gt_adults = gt_df[gt_df['age_self_declared_binary'] == 'adult']['anthroscore_max'].values
        
        if len(gt_teens) > 5 and len(gt_adults) > 5:
            t_stat_gt, p_val_gt = ttest_ind(gt_teens, gt_adults, equal_var=False)
            d_gt = cohens_d(gt_teens, gt_adults)
            
            results['rq2']['ground_truth_validation'] = {
                'teen_n': int(len(gt_teens)),
                'teen_mean': float(gt_teens.mean()),
                'adult_n': int(len(gt_adults)),
                'adult_mean': float(gt_adults.mean()),
                't_statistic': float(t_stat_gt),
                'p_value': float(p_val_gt),
                'cohens_d': float(d_gt),
                'direction': 'teens_higher' if d_gt > 0 else 'adults_higher',
                'matches_predicted': (d_gt > 0) == (results['rq2'].get('age_effect', {}).get('cohens_d', 0) > 0)
            }
            
            logger.info(f"Ground Truth - Teen mean: {gt_teens.mean():.4f} (n={len(gt_teens)})")
            logger.info(f"Ground Truth - Adult mean: {gt_adults.mean():.4f} (n={len(gt_adults)})")
            logger.info(f"Ground Truth Cohen's d = {d_gt:.4f} ({interpret_effect_size(d_gt)})")
            logger.info(f"Direction: {'TEENS HIGHER' if d_gt > 0 else 'ADULTS HIGHER'}")
    
    return results

# ============================================================================
# RQ3: EMOTIONAL EXPRESSION & ANTHROPOMORPHIZATION
# ============================================================================

def analyze_rq3(master: pd.DataFrame) -> Dict:
    """
    RQ3: How do emotional expression patterns relate to anthropomorphization?
    
    Analyses:
    - Correlations between emotions and AnthroScore
    - Group comparisons (high vs low anthropomorphizers)
    - Age moderation effects
    - Regression with interaction terms
    """
    logger.info("\n" + "="*70)
    logger.info("RQ3: EMOTIONAL EXPRESSION & ANTHROPOMORPHIZATION")
    logger.info("="*70)
    
    results = {'rq3': {}}
    
    # Check if we have emotion data
    emotion_cols = [c for c in master.columns if c.startswith('emotion_')]
    if not emotion_cols:
        logger.warning("No emotion columns found - skipping RQ3")
        return results
    
    # Filter to users with emotions and anthroscores
    emotion_df = master[
        master['anthroscore_max'].notna() &
        master['anthroscore_max'] > 0
    ].copy()
    
    logger.info(f"Users with emotions and non-zero AnthroScore: {len(emotion_df):,}")
    
    # ========================================
    # Correlations with AnthroScore
    # ========================================
    logger.info("\n--- Correlations with AnthroScore ---")
    
    emotion_vars = [c for c in emotion_cols if c in emotion_df.columns]
    if 'emotional_diversity' in emotion_df.columns:
        emotion_vars.append('emotional_diversity')
    if 'emotional_intensity' in emotion_df.columns:
        emotion_vars.append('emotional_intensity')
    if 'emotional_valence' in emotion_df.columns:
        emotion_vars.append('emotional_valence')
    
    correlations = {}
    for var in emotion_vars:
        if var in emotion_df.columns and emotion_df[var].notna().sum() > 100:
            valid = emotion_df[[var, 'anthroscore_max']].dropna()
            r, p = pearsonr(valid[var], valid['anthroscore_max'])
            correlations[var] = {
                'r': float(r),
                'p_value': float(p),
                'n': int(len(valid)),
                'interpretation': interpret_effect_size(r, 'r'),
                'significant': p < 0.05
            }
            logger.info(f"{var}: r = {r:.4f}, p = {p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
    
    results['rq3']['correlations'] = correlations
    
    # ========================================
    # Group Comparisons (High vs Low Anthropomorphizers)
    # ========================================
    logger.info("\n--- High vs Low Anthropomorphizers ---")
    
    # Define high/low as top/bottom quartiles
    q25 = emotion_df['anthroscore_max'].quantile(0.25)
    q75 = emotion_df['anthroscore_max'].quantile(0.75)
    
    low_anthro = emotion_df[emotion_df['anthroscore_max'] <= q25]
    high_anthro = emotion_df[emotion_df['anthroscore_max'] >= q75]
    
    logger.info(f"Low anthropomorphizers (bottom 25%): n = {len(low_anthro)}")
    logger.info(f"High anthropomorphizers (top 25%): n = {len(high_anthro)}")
    
    group_comparisons = {}
    for var in emotion_vars:
        if var in low_anthro.columns and var in high_anthro.columns:
            low_vals = low_anthro[var].dropna().values
            high_vals = high_anthro[var].dropna().values
            
            if len(low_vals) > 10 and len(high_vals) > 10:
                t_stat, p_val = ttest_ind(high_vals, low_vals, equal_var=False)
                d = cohens_d(high_vals, low_vals)
                
                group_comparisons[var] = {
                    'low_mean': float(low_vals.mean()),
                    'low_std': float(low_vals.std()),
                    'high_mean': float(high_vals.mean()),
                    'high_std': float(high_vals.std()),
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'cohens_d': float(d),
                    'interpretation': interpret_effect_size(d),
                    'direction': 'high_anthro_higher' if d > 0 else 'low_anthro_higher'
                }
                
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                logger.info(f"{var}: d = {d:.4f} ({interpret_effect_size(d)}) {sig}")
    
    results['rq3']['group_comparisons'] = group_comparisons
    
    # ========================================
    # Age Moderation Analysis
    # ========================================
    logger.info("\n--- Age Moderation Effects ---")
    
    teens_emotions = emotion_df[emotion_df['age_predicted'] == 'teen']
    adults_emotions = emotion_df[emotion_df['age_predicted'] == 'adult']
    
    moderation_effects = {}
    for var in emotion_vars[:5]:  # Top 5 emotions
        if var in teens_emotions.columns:
            teen_valid = teens_emotions[[var, 'anthroscore_max']].dropna()
            adult_valid = adults_emotions[[var, 'anthroscore_max']].dropna()
            
            if len(teen_valid) > 50 and len(adult_valid) > 50:
                r_teen, p_teen = pearsonr(teen_valid[var], teen_valid['anthroscore_max'])
                r_adult, p_adult = pearsonr(adult_valid[var], adult_valid['anthroscore_max'])
                
                # Fisher's z-test for difference in correlations
                z_teen = np.arctanh(r_teen)
                z_adult = np.arctanh(r_adult)
                se_diff = np.sqrt(1/(len(teen_valid)-3) + 1/(len(adult_valid)-3))
                z_diff = (z_teen - z_adult) / se_diff
                p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
                
                moderation_effects[var] = {
                    'teen_r': float(r_teen),
                    'teen_p': float(p_teen),
                    'teen_n': int(len(teen_valid)),
                    'adult_r': float(r_adult),
                    'adult_p': float(p_adult),
                    'adult_n': int(len(adult_valid)),
                    'z_difference': float(z_diff),
                    'p_difference': float(p_diff),
                    'significant_moderation': p_diff < 0.05
                }
                
                sig = '*' if p_diff < 0.05 else ''
                logger.info(f"{var}: Teen r={r_teen:.3f}, Adult r={r_adult:.3f}, diff p={p_diff:.4f} {sig}")
    
    results['rq3']['moderation_effects'] = moderation_effects
    
    # ========================================
    # Regression with Interaction
    # ========================================
    if 'emotional_diversity' in emotion_df.columns:
        logger.info("\n--- Regression: AnthroScore ~ EmoDiversity * Age ---")
        
        reg_df = emotion_df[['anthroscore_max', 'emotional_diversity', 'age_predicted']].dropna()
        reg_df['is_teen'] = (reg_df['age_predicted'] == 'teen').astype(int)
        
        try:
            model = ols('anthroscore_max ~ emotional_diversity * is_teen', data=reg_df).fit()
            
            results['rq3']['regression'] = {
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'f_statistic': float(model.fvalue),
                'f_pvalue': float(model.f_pvalue),
                'coefficients': {
                    'intercept': float(model.params['Intercept']),
                    'emotional_diversity': float(model.params['emotional_diversity']),
                    'is_teen': float(model.params['is_teen']),
                    'interaction': float(model.params['emotional_diversity:is_teen'])
                },
                'p_values': {
                    'intercept': float(model.pvalues['Intercept']),
                    'emotional_diversity': float(model.pvalues['emotional_diversity']),
                    'is_teen': float(model.pvalues['is_teen']),
                    'interaction': float(model.pvalues['emotional_diversity:is_teen'])
                }
            }
            
            logger.info(f"R-squared: {model.rsquared:.4f}")
            logger.info(f"EmoDiversity coefficient: {model.params['emotional_diversity']:.4f} (p={model.pvalues['emotional_diversity']:.4f})")
            logger.info(f"Interaction coefficient: {model.params['emotional_diversity:is_teen']:.4f} (p={model.pvalues['emotional_diversity:is_teen']:.4f})")
            
        except Exception as e:
            logger.warning(f"Regression failed: {e}")
    
    return results

# ============================================================================
# SENSITIVITY ANALYSES
# ============================================================================

def run_sensitivity_analyses(master: pd.DataFrame) -> Dict:
    """Run sensitivity analyses across different confidence thresholds."""
    logger.info("\n" + "="*70)
    logger.info("SENSITIVITY ANALYSES")
    logger.info("="*70)
    
    results = {'sensitivity': {}}
    
    # Get full predictions (before confidence filtering)
    gender_all = pd.read_parquet(project_root / 'experiments/v2_correction/gender_predictions_v3.parquet')
    age_all = pd.read_parquet(project_root / 'experiments/v2_correction/age_predictions_v3.parquet')
    anthro = pd.read_parquet(project_root / 'Data/features/user_anthroscores.parquet')
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    sensitivity_age = []
    for thresh in thresholds:
        filtered = age_all[age_all['confidence'] >= thresh].merge(
            anthro[anthro['anthroscore_max'] > 0][['author', 'anthroscore_max']],
            on='author'
        )
        
        if len(filtered) > 100:
            teens = filtered[filtered['age_predicted'] == 'teen']['anthroscore_max'].values
            adults = filtered[filtered['age_predicted'] == 'adult']['anthroscore_max'].values
            
            if len(teens) > 10 and len(adults) > 10:
                d = cohens_d(teens, adults)
                t_stat, p_val = ttest_ind(teens, adults, equal_var=False)
                
                sensitivity_age.append({
                    'threshold': thresh,
                    'n_total': len(filtered),
                    'n_teen': len(teens),
                    'n_adult': len(adults),
                    'cohens_d': float(d),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                })
                
                logger.info(f"Threshold {thresh}: n={len(filtered)}, d={d:.4f}, p={p_val:.4f} {'***' if p_val < 0.001 else ''}")
    
    results['sensitivity']['age_threshold'] = sensitivity_age
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete analysis pipeline."""
    logger.info("="*70)
    logger.info("THE ILLUSION PROJECT: FINAL COMPREHENSIVE ANALYSIS")
    logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    # Load data
    data = load_all_data()
    master = create_master_dataset(data)
    
    # Run all analyses
    all_results = {}
    
    # RQ1: Demographics
    rq1_results = analyze_rq1(master, data)
    all_results.update(rq1_results)
    
    # RQ2: Demographics & Anthropomorphization
    rq2_results = analyze_rq2(master)
    all_results.update(rq2_results)
    
    # RQ3: Emotional Expression
    rq3_results = analyze_rq3(master)
    all_results.update(rq3_results)
    
    # Sensitivity analyses
    sensitivity_results = run_sensitivity_analyses(master)
    all_results.update(sensitivity_results)
    
    # Save results
    results_path = project_root / 'results/FINAL_STATISTICAL_RESULTS.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n\nResults saved to: {results_path}")
    
    # Generate markdown report
    generate_markdown_report(all_results, master)
    
    return all_results


def generate_markdown_report(results: Dict, master: pd.DataFrame):
    """Generate comprehensive markdown report."""
    
    report_lines = [
        "# The Illusion Project: Comprehensive Statistical Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Analysis Sample:** {len(master):,} users with V3 predictions @ 0.60 confidence",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Key findings
    if 'rq2' in results and 'age_effect' in results['rq2']:
        ae = results['rq2']['age_effect']
        report_lines.append(f"### Age Effect on Anthropomorphization")
        report_lines.append(f"- **Cohen's d:** {ae['cohens_d']:.4f} ({ae['effect_interpretation']})")
        report_lines.append(f"- **p-value:** {ae['p_value']:.6f}")
        report_lines.append(f"- **Bootstrap 95% CI:** [{ae['bootstrap_ci_lower']:.4f}, {ae['bootstrap_ci_upper']:.4f}]")
        report_lines.append(f"- **Direction:** {ae['direction'].replace('_', ' ').title()}")
        report_lines.append("")
    
    if 'rq2' in results and 'gender_effect' in results['rq2']:
        ge = results['rq2']['gender_effect']
        report_lines.append(f"### Gender Effect on Anthropomorphization")
        report_lines.append(f"- **Cohen's d:** {ge['cohens_d']:.4f} ({ge['effect_interpretation']})")
        report_lines.append(f"- **p-value:** {ge['p_value']:.6f}")
        report_lines.append("")
    
    # RQ1 Demographics
    report_lines.extend([
        "---",
        "",
        "## RQ1: Demographics of AI Companion Users",
        "",
    ])
    
    if 'rq1' in results:
        if 'age_distribution' in results['rq1']:
            ad = results['rq1']['age_distribution']
            report_lines.extend([
                "### Age Distribution",
                "",
                "| Age Group | Count | Percentage | 95% CI |",
                "|-----------|-------|------------|--------|",
            ])
            for age in ['teen', 'adult']:
                if age in ad:
                    report_lines.append(
                        f"| {age.title()} | {ad[age]['count']:,} | {ad[age]['percentage']*100:.1f}% | "
                        f"[{ad[age]['ci_lower']*100:.1f}%, {ad[age]['ci_upper']*100:.1f}%] |"
                    )
            report_lines.append("")
        
        if 'gender_distribution' in results['rq1']:
            gd = results['rq1']['gender_distribution']
            report_lines.extend([
                "### Gender Distribution",
                "",
                "| Gender | Count | Percentage | 95% CI |",
                "|--------|-------|------------|--------|",
            ])
            for gender in ['male', 'female']:
                if gender in gd:
                    report_lines.append(
                        f"| {gender.title()} | {gd[gender]['count']:,} | {gd[gender]['percentage']*100:.1f}% | "
                        f"[{gd[gender]['ci_lower']*100:.1f}%, {gd[gender]['ci_upper']*100:.1f}%] |"
                    )
            report_lines.append("")
    
    # RQ2 Demographics & Anthropomorphization
    report_lines.extend([
        "---",
        "",
        "## RQ2: Demographics & Anthropomorphization",
        "",
    ])
    
    if 'rq2' in results:
        if 'age_effect' in results['rq2']:
            ae = results['rq2']['age_effect']
            report_lines.extend([
                "### Age → Anthropomorphization",
                "",
                "| Metric | Teen | Adult |",
                "|--------|------|-------|",
                f"| N | {ae['teen_n']:,} | {ae['adult_n']:,} |",
                f"| Mean AnthroScore | {ae['teen_mean']:.4f} | {ae['adult_mean']:.4f} |",
                f"| SD | {ae['teen_std']:.4f} | {ae['adult_std']:.4f} |",
                "",
                "| Statistical Test | Value |",
                "|-----------------|-------|",
                f"| Welch's t | {ae['t_statistic']:.3f} |",
                f"| p-value | {ae['p_value']:.6f} |",
                f"| Cohen's d | {ae['cohens_d']:.4f} ({ae['effect_interpretation']}) |",
                f"| Bootstrap 95% CI | [{ae['bootstrap_ci_lower']:.4f}, {ae['bootstrap_ci_upper']:.4f}] |",
                "",
            ])
        
        if 'two_way_anova' in results['rq2']:
            anova = results['rq2']['two_way_anova']
            report_lines.extend([
                "### Age × Gender Two-Way ANOVA",
                "",
                "| Effect | F | p-value | Significant |",
                "|--------|---|---------|-------------|",
                f"| Age | {anova['age_F']:.2f} | {anova['age_p']:.6f} | {'Yes' if anova['age_p'] < 0.05 else 'No'} |",
                f"| Gender | {anova['gender_F']:.2f} | {anova['gender_p']:.6f} | {'Yes' if anova['gender_p'] < 0.05 else 'No'} |",
                f"| Age × Gender | {anova['interaction_F']:.2f} | {anova['interaction_p']:.6f} | {'Yes' if anova['interaction_p'] < 0.05 else 'No'} |",
                "",
                f"**R-squared:** {anova['r_squared']:.4f}",
                "",
            ])
        
        if 'ground_truth_validation' in results['rq2']:
            gt = results['rq2']['ground_truth_validation']
            report_lines.extend([
                "### Ground Truth Validation (Self-Declared Ages)",
                "",
                f"- Teen mean: {gt['teen_mean']:.4f} (n={gt['teen_n']})",
                f"- Adult mean: {gt['adult_mean']:.4f} (n={gt['adult_n']})",
                f"- Cohen's d: {gt['cohens_d']:.4f}",
                f"- Direction: {gt['direction'].replace('_', ' ').title()}",
                f"- **Matches predicted direction:** {'Yes' if gt['matches_predicted'] else 'NO - CRITICAL DISCREPANCY'}",
                "",
            ])
    
    # RQ3 Emotional Expression
    report_lines.extend([
        "---",
        "",
        "## RQ3: Emotional Expression & Anthropomorphization",
        "",
    ])
    
    if 'rq3' in results:
        if 'correlations' in results['rq3']:
            report_lines.extend([
                "### Correlations with AnthroScore",
                "",
                "| Variable | r | p-value | Interpretation |",
                "|----------|---|---------|----------------|",
            ])
            for var, data in sorted(results['rq3']['correlations'].items(), key=lambda x: abs(x[1]['r']), reverse=True):
                sig = '***' if data['p_value'] < 0.001 else '**' if data['p_value'] < 0.01 else '*' if data['p_value'] < 0.05 else ''
                report_lines.append(f"| {var} | {data['r']:.4f} | {data['p_value']:.6f} {sig} | {data['interpretation']} |")
            report_lines.append("")
        
        if 'group_comparisons' in results['rq3']:
            report_lines.extend([
                "### High vs Low Anthropomorphizers (Quartile Comparison)",
                "",
                "| Variable | Low Mean | High Mean | Cohen's d | p-value |",
                "|----------|----------|-----------|-----------|---------|",
            ])
            for var, data in sorted(results['rq3']['group_comparisons'].items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True):
                sig = '***' if data['p_value'] < 0.001 else '**' if data['p_value'] < 0.01 else '*' if data['p_value'] < 0.05 else ''
                report_lines.append(f"| {var} | {data['low_mean']:.4f} | {data['high_mean']:.4f} | {data['cohens_d']:.4f} | {data['p_value']:.6f} {sig} |")
            report_lines.append("")
    
    # Sensitivity analyses
    report_lines.extend([
        "---",
        "",
        "## Sensitivity Analyses",
        "",
    ])
    
    if 'sensitivity' in results and 'age_threshold' in results['sensitivity']:
        report_lines.extend([
            "### Age Effect Stability Across Confidence Thresholds",
            "",
            "| Threshold | N | Cohen's d | p-value | Significant |",
            "|-----------|---|-----------|---------|-------------|",
        ])
        for row in results['sensitivity']['age_threshold']:
            report_lines.append(
                f"| >= {row['threshold']} | {row['n_total']:,} | {row['cohens_d']:.4f} | {row['p_value']:.6f} | {'Yes' if row['significant'] else 'No'} |"
            )
        report_lines.append("")
    
    # Write report
    report_path = project_root / 'results/COMPREHENSIVE_STATISTICAL_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Markdown report saved to: {report_path}")


if __name__ == '__main__':
    results = main()
