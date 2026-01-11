"""
ADVANCED ANALYSES: Mediation, Moderation, and Deep Statistical Testing
=======================================================================

This script performs advanced statistical analyses beyond the basic tests.

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
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_analysis_data():
    """Load and prepare data for advanced analyses."""
    
    anthroscores = pd.read_parquet(project_root / 'Data/features/user_anthroscores.parquet')
    gender_v3 = pd.read_parquet(project_root / 'experiments/v2_correction/gender_predictions_v3.parquet')
    age_v3 = pd.read_parquet(project_root / 'experiments/v2_correction/age_predictions_v3.parquet')
    self_decl = pd.read_parquet(project_root / 'Data/features/self_declarations.parquet')
    
    try:
        emotions = pd.read_parquet(project_root / 'Data/features/user_emotions.parquet')
    except:
        emotions = None
    
    # Merge with 0.60 confidence threshold
    df = anthroscores.merge(
        gender_v3[gender_v3['confidence'] >= 0.6][['author', 'gender_predicted', 'confidence']].rename(
            columns={'confidence': 'gender_conf'}),
        on='author', how='inner'
    )
    df = df.merge(
        age_v3[age_v3['confidence'] >= 0.6][['author', 'age_predicted', 'confidence']].rename(
            columns={'confidence': 'age_conf'}),
        on='author', how='inner'
    )
    
    if emotions is not None:
        df = df.merge(emotions, on='author', how='left')
    
    df = df.merge(self_decl[['author', 'age_bucket_self_declared', 'gender_self_declared']], 
                  on='author', how='left')
    
    # Create numeric variables
    df['is_teen'] = (df['age_predicted'] == 'teen').astype(int)
    df['is_female'] = (df['gender_predicted'] == 'female').astype(int)
    df['has_anthro'] = (df['anthroscore_max'] > 0).astype(int)
    
    return df

# ============================================================================
# EFFECT SIZE ANALYSIS
# ============================================================================

def comprehensive_effect_sizes(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive effect sizes with different measures."""
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE EFFECT SIZE ANALYSIS")
    logger.info("="*60)
    
    results = {}
    
    # Filter to non-zero anthropomorphization
    nonzero = df[df['anthroscore_max'] > 0].copy()
    
    # Age effect - multiple measures
    teens = nonzero[nonzero['age_predicted'] == 'teen']['anthroscore_max']
    adults = nonzero[nonzero['age_predicted'] == 'adult']['anthroscore_max']
    
    if len(teens) > 0 and len(adults) > 0:
        # Cohen's d
        pooled_std = np.sqrt(((len(teens)-1)*teens.var() + (len(adults)-1)*adults.var()) / 
                            (len(teens)+len(adults)-2))
        d = (teens.mean() - adults.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Hedges' g (corrected for small samples)
        correction = 1 - 3/(4*(len(teens)+len(adults)-2) - 1)
        g = d * correction
        
        # Glass's delta (using adult as control)
        delta = (teens.mean() - adults.mean()) / adults.std() if adults.std() > 0 else 0
        
        # Common Language Effect Size (probability of superiority)
        # P(randomly selected teen > randomly selected adult)
        cles_values = []
        np.random.seed(42)
        for _ in range(1000):
            t_sample = np.random.choice(teens.values, 100, replace=True)
            a_sample = np.random.choice(adults.values, 100, replace=True)
            cles_values.append((t_sample > a_sample).mean())
        cles = np.mean(cles_values)
        
        # Point-biserial correlation
        x = np.concatenate([np.ones(len(teens)), np.zeros(len(adults))])
        y = np.concatenate([teens.values, adults.values])
        r_pb, p_pb = pearsonr(x, y)
        
        results['age_effect'] = {
            'cohens_d': float(d),
            'hedges_g': float(g),
            'glass_delta': float(delta),
            'cles': float(cles),  # 0.5 = no effect, >0.5 = teens higher
            'point_biserial_r': float(r_pb),
            'point_biserial_p': float(p_pb)
        }
        
        logger.info(f"\nAge Effect (Teen vs Adult):")
        logger.info(f"  Cohen's d: {d:.4f}")
        logger.info(f"  Hedges' g: {g:.4f}")
        logger.info(f"  Glass's delta: {delta:.4f}")
        logger.info(f"  CLES (P(teen > adult)): {cles:.4f}")
        logger.info(f"  Point-biserial r: {r_pb:.4f} (p={p_pb:.6f})")
    
    # Gender effect
    males = nonzero[nonzero['gender_predicted'] == 'male']['anthroscore_max']
    females = nonzero[nonzero['gender_predicted'] == 'female']['anthroscore_max']
    
    if len(males) > 0 and len(females) > 0:
        pooled_std = np.sqrt(((len(males)-1)*males.var() + (len(females)-1)*females.var()) / 
                            (len(males)+len(females)-2))
        d = (males.mean() - females.mean()) / pooled_std if pooled_std > 0 else 0
        correction = 1 - 3/(4*(len(males)+len(females)-2) - 1)
        g = d * correction
        
        # CLES
        cles_values = []
        np.random.seed(42)
        for _ in range(1000):
            m_sample = np.random.choice(males.values, 100, replace=True)
            f_sample = np.random.choice(females.values, 100, replace=True)
            cles_values.append((m_sample > f_sample).mean())
        cles = np.mean(cles_values)
        
        results['gender_effect'] = {
            'cohens_d': float(d),
            'hedges_g': float(g),
            'cles': float(cles),  # <0.5 means females higher
        }
        
        logger.info(f"\nGender Effect (Male vs Female):")
        logger.info(f"  Cohen's d: {d:.4f}")
        logger.info(f"  Hedges' g: {g:.4f}")
        logger.info(f"  CLES (P(male > female)): {cles:.4f}")
    
    return results

# ============================================================================
# MULTIPLE REGRESSION
# ============================================================================

def multiple_regression_analysis(df: pd.DataFrame) -> Dict:
    """Run comprehensive multiple regression models."""
    logger.info("\n" + "="*60)
    logger.info("MULTIPLE REGRESSION ANALYSIS")
    logger.info("="*60)
    
    results = {}
    
    # Filter to non-zero and get emotion columns
    reg_df = df[df['anthroscore_max'] > 0].copy()
    emotion_cols = [c for c in reg_df.columns if c.startswith('emotion_') and c != 'dominant_emotion']
    
    # Model 1: Demographics only
    logger.info("\nModel 1: Demographics → AnthroScore")
    try:
        model1 = ols('anthroscore_max ~ is_teen + is_female', data=reg_df).fit()
        results['model1_demographics'] = {
            'r_squared': float(model1.rsquared),
            'adj_r_squared': float(model1.rsquared_adj),
            'f_statistic': float(model1.fvalue),
            'f_pvalue': float(model1.f_pvalue),
            'coefficients': {
                'intercept': float(model1.params['Intercept']),
                'is_teen': float(model1.params['is_teen']),
                'is_female': float(model1.params['is_female'])
            },
            'p_values': {
                'intercept': float(model1.pvalues['Intercept']),
                'is_teen': float(model1.pvalues['is_teen']),
                'is_female': float(model1.pvalues['is_female'])
            },
            'ci_95': {
                'is_teen': [float(model1.conf_int().loc['is_teen', 0]), 
                           float(model1.conf_int().loc['is_teen', 1])],
                'is_female': [float(model1.conf_int().loc['is_female', 0]), 
                             float(model1.conf_int().loc['is_female', 1])]
            }
        }
        logger.info(f"  R²: {model1.rsquared:.4f}")
        logger.info(f"  is_teen: B={model1.params['is_teen']:.4f} (p={model1.pvalues['is_teen']:.4f})")
        logger.info(f"  is_female: B={model1.params['is_female']:.4f} (p={model1.pvalues['is_female']:.4f})")
    except Exception as e:
        logger.warning(f"Model 1 failed: {e}")
    
    # Model 2: Demographics + Interaction
    logger.info("\nModel 2: Demographics with Interaction")
    try:
        model2 = ols('anthroscore_max ~ is_teen * is_female', data=reg_df).fit()
        results['model2_interaction'] = {
            'r_squared': float(model2.rsquared),
            'adj_r_squared': float(model2.rsquared_adj),
            'coefficients': {
                'intercept': float(model2.params['Intercept']),
                'is_teen': float(model2.params['is_teen']),
                'is_female': float(model2.params['is_female']),
                'interaction': float(model2.params['is_teen:is_female'])
            },
            'p_values': {
                'is_teen': float(model2.pvalues['is_teen']),
                'is_female': float(model2.pvalues['is_female']),
                'interaction': float(model2.pvalues['is_teen:is_female'])
            }
        }
        logger.info(f"  R²: {model2.rsquared:.4f}")
        logger.info(f"  Interaction: B={model2.params['is_teen:is_female']:.4f} (p={model2.pvalues['is_teen:is_female']:.4f})")
    except Exception as e:
        logger.warning(f"Model 2 failed: {e}")
    
    # Model 3: Full model with emotions
    if emotion_cols and len(emotion_cols) > 0:
        logger.info("\nModel 3: Full Model (Demographics + Emotions)")
        
        emotion_vars = ['emotion_anger', 'emotion_sadness', 'emotion_fear', 'emotion_joy']
        emotion_vars = [c for c in emotion_vars if c in reg_df.columns]
        
        if emotion_vars:
            formula = f'anthroscore_max ~ is_teen + is_female + {" + ".join(emotion_vars)}'
            try:
                model3 = ols(formula, data=reg_df.dropna(subset=emotion_vars)).fit()
                
                results['model3_full'] = {
                    'r_squared': float(model3.rsquared),
                    'adj_r_squared': float(model3.rsquared_adj),
                    'n': int(model3.nobs),
                    'coefficients': {k: float(v) for k, v in model3.params.items()},
                    'p_values': {k: float(v) for k, v in model3.pvalues.items()},
                    'significant_predictors': [k for k, v in model3.pvalues.items() if v < 0.05 and k != 'Intercept']
                }
                
                logger.info(f"  R²: {model3.rsquared:.4f}")
                logger.info(f"  Significant predictors: {results['model3_full']['significant_predictors']}")
            except Exception as e:
                logger.warning(f"Model 3 failed: {e}")
    
    return results

# ============================================================================
# SUBGROUP ANALYSES
# ============================================================================

def subgroup_analyses(df: pd.DataFrame) -> Dict:
    """Analyze anthropomorphization patterns within subgroups."""
    logger.info("\n" + "="*60)
    logger.info("SUBGROUP ANALYSES")
    logger.info("="*60)
    
    results = {}
    nonzero = df[df['anthroscore_max'] > 0].copy()
    
    # Four demographic subgroups
    groups = {
        'teen_male': nonzero[(nonzero['age_predicted'] == 'teen') & (nonzero['gender_predicted'] == 'male')],
        'teen_female': nonzero[(nonzero['age_predicted'] == 'teen') & (nonzero['gender_predicted'] == 'female')],
        'adult_male': nonzero[(nonzero['age_predicted'] == 'adult') & (nonzero['gender_predicted'] == 'male')],
        'adult_female': nonzero[(nonzero['age_predicted'] == 'adult') & (nonzero['gender_predicted'] == 'female')]
    }
    
    logger.info("\n--- Subgroup Means and Effect Sizes ---")
    
    subgroup_stats = {}
    for name, group in groups.items():
        if len(group) > 10:
            subgroup_stats[name] = {
                'n': int(len(group)),
                'mean': float(group['anthroscore_max'].mean()),
                'std': float(group['anthroscore_max'].std()),
                'median': float(group['anthroscore_max'].median()),
                'pct_high': float((group['anthroscore_max'] >= group['anthroscore_max'].quantile(0.75)).mean())
            }
            logger.info(f"{name}: n={len(group)}, M={group['anthroscore_max'].mean():.4f}, SD={group['anthroscore_max'].std():.4f}")
    
    results['subgroup_statistics'] = subgroup_stats
    
    # Pairwise comparisons
    logger.info("\n--- Pairwise Comparisons ---")
    comparisons = [
        ('teen_male', 'adult_male'),
        ('teen_female', 'adult_female'),
        ('teen_male', 'teen_female'),
        ('adult_male', 'adult_female')
    ]
    
    pairwise = {}
    for g1, g2 in comparisons:
        if g1 in groups and g2 in groups and len(groups[g1]) > 10 and len(groups[g2]) > 10:
            v1 = groups[g1]['anthroscore_max'].values
            v2 = groups[g2]['anthroscore_max'].values
            
            t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            d = (v1.mean() - v2.mean()) / np.sqrt((v1.var() + v2.var()) / 2)
            
            pairwise[f'{g1}_vs_{g2}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'significant': p_val < 0.05
            }
            
            sig = '*' if p_val < 0.05 else ''
            logger.info(f"{g1} vs {g2}: d={d:.4f}, p={p_val:.4f} {sig}")
    
    results['pairwise_comparisons'] = pairwise
    
    return results

# ============================================================================
# PREVALENCE ANALYSIS
# ============================================================================

def prevalence_analysis(df: pd.DataFrame) -> Dict:
    """Analyze prevalence of anthropomorphization across groups."""
    logger.info("\n" + "="*60)
    logger.info("PREVALENCE ANALYSIS")
    logger.info("="*60)
    
    results = {}
    
    # Define high anthropomorphization (top quartile of non-zero)
    nonzero = df[df['anthroscore_max'] > 0]
    high_threshold = nonzero['anthroscore_max'].quantile(0.75)
    df['high_anthro'] = (df['anthroscore_max'] >= high_threshold).astype(int)
    
    logger.info(f"High anthropomorphization threshold: {high_threshold:.4f}")
    
    # Prevalence by age
    logger.info("\n--- Prevalence by Age ---")
    age_prev = df.groupby('age_predicted')['high_anthro'].agg(['mean', 'sum', 'count']).reset_index()
    age_prev['ci_lower'] = age_prev.apply(
        lambda r: r['mean'] - 1.96 * np.sqrt(r['mean']*(1-r['mean'])/r['count']), axis=1)
    age_prev['ci_upper'] = age_prev.apply(
        lambda r: r['mean'] + 1.96 * np.sqrt(r['mean']*(1-r['mean'])/r['count']), axis=1)
    
    results['prevalence_by_age'] = age_prev.to_dict('records')
    
    for _, row in age_prev.iterrows():
        logger.info(f"{row['age_predicted']}: {row['mean']*100:.1f}% [{row['ci_lower']*100:.1f}%, {row['ci_upper']*100:.1f}%]")
    
    # Prevalence by gender
    logger.info("\n--- Prevalence by Gender ---")
    gender_prev = df.groupby('gender_predicted')['high_anthro'].agg(['mean', 'sum', 'count']).reset_index()
    gender_prev['ci_lower'] = gender_prev.apply(
        lambda r: r['mean'] - 1.96 * np.sqrt(r['mean']*(1-r['mean'])/r['count']), axis=1)
    gender_prev['ci_upper'] = gender_prev.apply(
        lambda r: r['mean'] + 1.96 * np.sqrt(r['mean']*(1-r['mean'])/r['count']), axis=1)
    
    results['prevalence_by_gender'] = gender_prev.to_dict('records')
    
    for _, row in gender_prev.iterrows():
        logger.info(f"{row['gender_predicted']}: {row['mean']*100:.1f}% [{row['ci_lower']*100:.1f}%, {row['ci_upper']*100:.1f}%]")
    
    # Odds ratios
    logger.info("\n--- Odds Ratios ---")
    
    # Age odds ratio
    teen_prev = df[df['age_predicted'] == 'teen']['high_anthro'].mean()
    adult_prev = df[df['age_predicted'] == 'adult']['high_anthro'].mean()
    
    if teen_prev > 0 and adult_prev > 0 and teen_prev < 1 and adult_prev < 1:
        teen_odds = teen_prev / (1 - teen_prev)
        adult_odds = adult_prev / (1 - adult_prev)
        or_age = teen_odds / adult_odds
        
        results['odds_ratio_age'] = {
            'teen_vs_adult': float(or_age),
            'interpretation': 'teens more likely' if or_age > 1 else 'adults more likely'
        }
        logger.info(f"Age OR (teen vs adult): {or_age:.4f}")
    
    # Gender odds ratio
    male_prev = df[df['gender_predicted'] == 'male']['high_anthro'].mean()
    female_prev = df[df['gender_predicted'] == 'female']['high_anthro'].mean()
    
    if male_prev > 0 and female_prev > 0 and male_prev < 1 and female_prev < 1:
        male_odds = male_prev / (1 - male_prev)
        female_odds = female_prev / (1 - female_prev)
        or_gender = female_odds / male_odds
        
        results['odds_ratio_gender'] = {
            'female_vs_male': float(or_gender),
            'interpretation': 'females more likely' if or_gender > 1 else 'males more likely'
        }
        logger.info(f"Gender OR (female vs male): {or_gender:.4f}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("="*70)
    logger.info("ADVANCED STATISTICAL ANALYSES")
    logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    df = load_analysis_data()
    logger.info(f"\nLoaded {len(df):,} users with complete data")
    
    all_results = {}
    
    # Effect sizes
    effect_results = comprehensive_effect_sizes(df)
    all_results['effect_sizes'] = effect_results
    
    # Multiple regression
    regression_results = multiple_regression_analysis(df)
    all_results['regression'] = regression_results
    
    # Subgroup analyses
    subgroup_results = subgroup_analyses(df)
    all_results['subgroups'] = subgroup_results
    
    # Prevalence
    prevalence_results = prevalence_analysis(df)
    all_results['prevalence'] = prevalence_results
    
    # Save
    results_path = project_root / 'results/ADVANCED_ANALYSES_RESULTS.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n\nAdvanced results saved to: {results_path}")
    
    return all_results

if __name__ == '__main__':
    results = main()
