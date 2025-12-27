"""
NeurIPS-Level Statistical Analysis Module.

This module implements all the statistical rigor required for NeurIPS:
1. Hierarchical regression with controls
2. Robustness checks (bootstrap, sensitivity, temporal)
3. Method comparison and ablation
4. Multiple comparison corrections
5. Model diagnostics
6. 3-bucket age simplification
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import warnings

logger = logging.getLogger(__name__)

# =============================================================================
# 3-BUCKET AGE SCHEME
# =============================================================================

def convert_to_3_buckets(age_bucket_5: str) -> str:
    """Convert 5-bucket age to 3-bucket."""
    if pd.isna(age_bucket_5):
        return None
    
    mapping = {
        '13-18': 'teen',      # 13-18
        '19-25': 'young_adult',  # 19-30
        '26-40': 'adult',     # 31+
        '41-60': 'adult',
        '61-80': 'adult',
    }
    return mapping.get(age_bucket_5, None)


def convert_age_score_to_3_buckets(score: float, thresholds: Dict = None) -> str:
    """Convert age score to 3-bucket using calibrated thresholds."""
    if thresholds is None:
        # Calibrated from self-declarations
        # teen: mean=-0.30, young_adult: mean=-0.20, adult: mean=-0.05
        thresholds = {
            'teen': (-float('inf'), -0.22),
            'young_adult': (-0.22, -0.10),
            'adult': (-0.10, float('inf')),
        }
    
    for bucket, (low, high) in thresholds.items():
        if low <= score < high:
            return bucket
    return 'adult'


# =============================================================================
# CONTROL VARIABLES
# =============================================================================

def prepare_regression_with_controls(
    df: pd.DataFrame,
    age_scheme: str = '3_bucket',  # '3_bucket' or '5_bucket'
    exclude_unknown_gender: bool = False,
    reference_gender: str = 'male'
) -> pd.DataFrame:
    """
    Prepare regression data with all control variables.
    
    Controls added:
    - subreddit (fixed effects)
    - comment_count (activity level)
    - avg_comment_length (text complexity proxy)
    """
    logger.info(f"Preparing regression data with controls (age={age_scheme}, exclude_unknown={exclude_unknown_gender})")
    
    reg_df = df.copy()
    
    # Filter to valid observations
    reg_df = reg_df[
        reg_df['age_bucket'].notna() & 
        reg_df['anthroscore_mean'].notna()
    ].copy()
    
    if exclude_unknown_gender:
        reg_df = reg_df[reg_df['gender'] != 'unknown'].copy()
        logger.info(f"Excluded unknown gender: {len(reg_df)} remaining")
    
    # Create age variables
    if age_scheme == '3_bucket':
        reg_df['age_3bucket'] = reg_df['age_bucket'].apply(convert_to_3_buckets)
        age_col = 'age_3bucket'
        age_ref = 'adult'
    else:
        reg_df['age_bucket_safe'] = reg_df['age_bucket'].str.replace('-', '_', regex=False)
        age_col = 'age_bucket_safe'
        age_ref = '26_40'
    
    # Create age dummies
    age_dummies = pd.get_dummies(reg_df[age_col], prefix='age', dtype=int)
    ref_col = f'age_{age_ref}'
    if ref_col in age_dummies.columns:
        age_dummies = age_dummies.drop(ref_col, axis=1)
    reg_df = pd.concat([reg_df, age_dummies], axis=1)
    
    # Create gender dummies
    gender_dummies = pd.get_dummies(reg_df['gender'], prefix='gender', dtype=int)
    ref_gender_col = f'gender_{reference_gender}'
    if ref_gender_col in gender_dummies.columns:
        gender_dummies = gender_dummies.drop(ref_gender_col, axis=1)
    reg_df = pd.concat([reg_df, gender_dummies], axis=1)
    
    # Control variables
    # 1. Subreddit fixed effects
    if 'subreddit' in df.columns:
        # Get most common subreddit as reference
        sub_counts = df['subreddit'].value_counts()
        ref_sub = sub_counts.index[0] if len(sub_counts) > 0 else None
        subreddit_dummies = pd.get_dummies(reg_df['subreddit'], prefix='sub', dtype=int)
        if ref_sub and f'sub_{ref_sub}' in subreddit_dummies.columns:
            subreddit_dummies = subreddit_dummies.drop(f'sub_{ref_sub}', axis=1)
        reg_df = pd.concat([reg_df, subreddit_dummies], axis=1)
    
    # 2. Comment count (log-transformed for normality)
    if 'anthroscore_count' in reg_df.columns:
        reg_df['log_comment_count'] = np.log1p(reg_df['anthroscore_count'])
    
    # Create interaction terms (only for dummy columns with numeric values)
    # Get only the dummy columns (created by get_dummies, which are numeric)
    age_dummy_cols = [c for c in age_dummies.columns]
    gender_dummy_cols = [c for c in gender_dummies.columns]
    
    for age_term in age_dummy_cols:
        for gender_term in gender_dummy_cols:
            interaction_name = f'{age_term}_x_{gender_term}'
            if age_term in reg_df.columns and gender_term in reg_df.columns:
                reg_df[interaction_name] = reg_df[age_term].astype(int) * reg_df[gender_term].astype(int)
    
    logger.info(f"Prepared {len(reg_df)} observations")
    return reg_df


# =============================================================================
# HIERARCHICAL REGRESSION
# =============================================================================

def run_hierarchical_regression(
    df: pd.DataFrame,
    age_scheme: str = '3_bucket',
    exclude_unknown_gender: bool = True
) -> Dict[str, Any]:
    """
    Run hierarchical regression to show incremental variance explained.
    
    Steps:
    1. Controls only
    2. Controls + Age
    3. Controls + Age + Gender
    4. Controls + Age + Gender + Interactions
    """
    logger.info("Running hierarchical regression")
    
    reg_df = prepare_regression_with_controls(df, age_scheme, exclude_unknown_gender)
    
    if len(reg_df) < 100:
        return {"error": "Insufficient data"}
    
    # Get variable lists
    control_terms = [c for c in reg_df.columns if c.startswith('sub_') or c == 'log_comment_count']
    age_terms = [c for c in reg_df.columns if c.startswith('age_') and '_x_' not in c and
                 c not in ['age_bucket', 'age_bucket_safe', 'age_3bucket', 'age_bucket_community',
                          'age_bucket_llm', 'age_bucket_self_declared', 'age_community_score']]
    gender_terms = [c for c in reg_df.columns if c.startswith('gender_') and '_x_' not in c and
                   c not in ['gender', 'gender_community', 'gender_self_declared', 
                            'gender_community_score']]
    interaction_terms = [c for c in reg_df.columns if '_x_' in c]
    
    results = {
        'n_observations': len(reg_df),
        'control_terms': control_terms,
        'age_terms': age_terms,
        'gender_terms': gender_terms,
        'interaction_terms': interaction_terms,
        'regression_df': reg_df,
        'models': {}
    }
    
    # Step 1: Controls only
    if control_terms:
        formula1 = "anthroscore_mean ~ " + " + ".join(control_terms)
        try:
            model1 = ols(formula1, data=reg_df).fit()
            results['models']['controls_only'] = model1
            logger.info(f"Step 1 (Controls): R² = {model1.rsquared:.6f}")
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
    
    # Step 2: Controls + Age
    if control_terms and age_terms:
        formula2 = "anthroscore_mean ~ " + " + ".join(control_terms + age_terms)
        try:
            model2 = ols(formula2, data=reg_df).fit()
            results['models']['controls_age'] = model2
            delta_r2 = model2.rsquared - results['models'].get('controls_only', model2).rsquared
            logger.info(f"Step 2 (+ Age): R² = {model2.rsquared:.6f}, ΔR² = {delta_r2:.6f}")
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
    
    # Step 3: Controls + Age + Gender
    if control_terms and age_terms and gender_terms:
        formula3 = "anthroscore_mean ~ " + " + ".join(control_terms + age_terms + gender_terms)
        try:
            model3 = ols(formula3, data=reg_df).fit()
            results['models']['controls_age_gender'] = model3
            prev_r2 = results['models'].get('controls_age', results['models'].get('controls_only', model3)).rsquared
            delta_r2 = model3.rsquared - prev_r2
            logger.info(f"Step 3 (+ Gender): R² = {model3.rsquared:.6f}, ΔR² = {delta_r2:.6f}")
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
    
    # Step 4: Full model with interactions
    if control_terms and age_terms and gender_terms and interaction_terms:
        formula4 = "anthroscore_mean ~ " + " + ".join(
            control_terms + age_terms + gender_terms + interaction_terms
        )
        try:
            model4 = ols(formula4, data=reg_df).fit()
            results['models']['full'] = model4
            prev_r2 = results['models'].get('controls_age_gender', model4).rsquared
            delta_r2 = model4.rsquared - prev_r2
            logger.info(f"Step 4 (+ Interactions): R2 = {model4.rsquared:.6f}, dR2 = {delta_r2:.6f}")
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
    
    # Step 5: Model with robust standard errors (for heteroscedasticity)
    if age_terms and gender_terms:
        formula_robust = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms)
        try:
            model_robust = ols(formula_robust, data=reg_df).fit(cov_type='HC3')
            results['models']['robust_hc3'] = model_robust
            logger.info(f"Robust model (HC3): R2 = {model_robust.rsquared:.6f}")
        except Exception as e:
            logger.error(f"Robust model failed: {e}")
    
    return results


# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================

def bootstrap_coefficients(
    df: pd.DataFrame,
    formula: str,
    n_iterations: int = 1000,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for regression coefficients.
    """
    logger.info(f"Running bootstrap with {n_iterations} iterations")
    
    n = len(df)
    coef_samples = []
    
    for i in range(n_iterations):
        # Sample with replacement
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample_df = df.iloc[sample_idx]
        
        try:
            model = ols(formula, data=sample_df).fit()
            coef_samples.append(model.params)
        except:
            continue
    
    if len(coef_samples) < 100:
        logger.warning("Too few successful bootstrap iterations")
        return pd.DataFrame()
    
    coef_df = pd.DataFrame(coef_samples)
    
    alpha = 1 - confidence_level
    results = pd.DataFrame({
        'coef_mean': coef_df.mean(),
        'coef_std': coef_df.std(),
        f'ci_lower_{int(confidence_level*100)}': coef_df.quantile(alpha/2),
        f'ci_upper_{int(confidence_level*100)}': coef_df.quantile(1 - alpha/2),
    })
    
    logger.info(f"Bootstrap complete: {len(coef_samples)} successful iterations")
    return results


def sensitivity_analysis_thresholds(
    df: pd.DataFrame,
    threshold_variations: List[float] = [0.9, 0.95, 1.0, 1.05, 1.1]
) -> pd.DataFrame:
    """
    Sensitivity analysis: vary age thresholds and check stability.
    """
    logger.info("Running sensitivity analysis on age thresholds")
    
    base_thresholds = {
        'teen': (-float('inf'), -0.22),
        'young_adult': (-0.22, -0.10),
        'adult': (-0.10, float('inf')),
    }
    
    results = []
    
    for variation in threshold_variations:
        # Scale thresholds
        scaled_thresholds = {}
        for bucket, (low, high) in base_thresholds.items():
            scaled_low = low * variation if low != -float('inf') else low
            scaled_high = high * variation if high != float('inf') else high
            scaled_thresholds[bucket] = (scaled_low, scaled_high)
        
        # Apply thresholds
        df_temp = df.copy()
        df_temp['age_3bucket_sens'] = df_temp['age_community_score'].apply(
            lambda x: convert_age_score_to_3_buckets(x, scaled_thresholds) if pd.notna(x) else None
        )
        
        # Check accuracy against ground truth
        mask = df_temp['age_bucket_self_declared'].notna() & df_temp['age_3bucket_sens'].notna()
        if mask.sum() > 0:
            gt_3bucket = df_temp.loc[mask, 'age_bucket_self_declared'].apply(convert_to_3_buckets)
            accuracy = (gt_3bucket == df_temp.loc[mask, 'age_3bucket_sens']).mean()
        else:
            accuracy = np.nan
        
        results.append({
            'threshold_scale': variation,
            'accuracy': accuracy,
            'n_classified': df_temp['age_3bucket_sens'].notna().sum()
        })
    
    return pd.DataFrame(results)


def temporal_stability_analysis(
    df: pd.DataFrame,
    date_column: str = 'created_utc'
) -> Dict[str, Any]:
    """
    Check if results are stable across time periods.
    """
    logger.info("Running temporal stability analysis")
    
    if date_column not in df.columns:
        logger.warning(f"Date column {date_column} not found")
        return {}
    
    # Convert to datetime
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp[date_column], unit='s')
    df_temp['year'] = df_temp['date'].dt.year
    
    results = {}
    
    for year in df_temp['year'].unique():
        year_df = df_temp[df_temp['year'] == year]
        if len(year_df) > 100:
            # Run simple regression
            year_reg = prepare_regression_with_controls(
                year_df, 
                age_scheme='3_bucket', 
                exclude_unknown_gender=True
            )
            
            if len(year_reg) > 50:
                age_terms = [c for c in year_reg.columns if c.startswith('age_') and '_x_' not in c and
                            c not in ['age_bucket', 'age_bucket_safe', 'age_3bucket', 'age_bucket_community',
                                     'age_bucket_llm', 'age_bucket_self_declared', 'age_community_score']]
                gender_terms = [c for c in year_reg.columns if c.startswith('gender_') and '_x_' not in c and
                               c not in ['gender', 'gender_community', 'gender_self_declared', 
                                        'gender_community_score']]
                
                if age_terms and gender_terms:
                    formula = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms)
                    try:
                        model = ols(formula, data=year_reg).fit()
                        results[int(year)] = {
                            'n': len(year_reg),
                            'r_squared': model.rsquared,
                            'significant_terms': [p for p in model.pvalues.index if model.pvalues[p] < 0.05]
                        }
                    except:
                        pass
    
    return results


def subreddit_level_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run analysis separately for each subreddit.
    """
    logger.info("Running subreddit-level analysis")
    
    if 'subreddit' not in df.columns:
        return {}
    
    results = {}
    
    for subreddit in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == subreddit]
        if len(sub_df) > 200:
            sub_reg = prepare_regression_with_controls(
                sub_df,
                age_scheme='3_bucket',
                exclude_unknown_gender=True
            )
            
            if len(sub_reg) > 50:
                age_terms = [c for c in sub_reg.columns if c.startswith('age_') and '_x_' not in c and
                            c not in ['age_bucket', 'age_bucket_safe', 'age_3bucket', 'age_bucket_community',
                                     'age_bucket_llm', 'age_bucket_self_declared', 'age_community_score']]
                gender_terms = [c for c in sub_reg.columns if c.startswith('gender_') and '_x_' not in c and
                               c not in ['gender', 'gender_community', 'gender_self_declared', 
                                        'gender_community_score']]
                
                if age_terms and gender_terms:
                    formula = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms)
                    try:
                        model = ols(formula, data=sub_reg).fit()
                        results[subreddit] = {
                            'n': len(sub_reg),
                            'r_squared': model.rsquared,
                            'gender_female_coef': model.params.get('gender_female', np.nan),
                            'gender_female_p': model.pvalues.get('gender_female', np.nan)
                        }
                    except:
                        pass
    
    return results


# =============================================================================
# METHOD COMPARISON
# =============================================================================

def compare_classification_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare accuracy of different classification methods.
    """
    logger.info("Comparing classification methods")
    
    results = []
    
    # Self-declaration (baseline - 100% accuracy by definition)
    self_decl_count = df['age_bucket_self_declared'].notna().sum()
    results.append({
        'method': 'Self-Declaration',
        'coverage': self_decl_count / len(df),
        'n_classified': self_decl_count,
        'accuracy': 1.0,  # Ground truth
        'cohens_kappa': 1.0
    })
    
    # Community embeddings
    comm_mask = df['age_bucket_community'].notna() & df['age_bucket_self_declared'].notna()
    if comm_mask.sum() > 0:
        comm_acc = (df.loc[comm_mask, 'age_bucket_community'] == 
                   df.loc[comm_mask, 'age_bucket_self_declared']).mean()
        try:
            comm_kappa = cohen_kappa_score(
                df.loc[comm_mask, 'age_bucket_self_declared'],
                df.loc[comm_mask, 'age_bucket_community']
            )
        except:
            comm_kappa = np.nan
        
        results.append({
            'method': 'Community Embeddings',
            'coverage': df['age_bucket_community'].notna().sum() / len(df),
            'n_classified': df['age_bucket_community'].notna().sum(),
            'accuracy': comm_acc,
            'cohens_kappa': comm_kappa
        })
    
    # LLM
    llm_mask = df['age_bucket_llm'].notna() & df['age_bucket_self_declared'].notna()
    if llm_mask.sum() > 0:
        llm_acc = (df.loc[llm_mask, 'age_bucket_llm'] == 
                  df.loc[llm_mask, 'age_bucket_self_declared']).mean()
        try:
            llm_kappa = cohen_kappa_score(
                df.loc[llm_mask, 'age_bucket_self_declared'],
                df.loc[llm_mask, 'age_bucket_llm']
            )
        except:
            llm_kappa = np.nan
        
        results.append({
            'method': 'LLM Classification',
            'coverage': df['age_bucket_llm'].notna().sum() / len(df),
            'n_classified': df['age_bucket_llm'].notna().sum(),
            'accuracy': llm_acc,
            'cohens_kappa': llm_kappa
        })
    
    # Ensemble (final age_bucket)
    ensemble_mask = df['age_bucket'].notna() & df['age_bucket_self_declared'].notna()
    if ensemble_mask.sum() > 0:
        ensemble_acc = (df.loc[ensemble_mask, 'age_bucket'] == 
                       df.loc[ensemble_mask, 'age_bucket_self_declared']).mean()
        try:
            ensemble_kappa = cohen_kappa_score(
                df.loc[ensemble_mask, 'age_bucket_self_declared'],
                df.loc[ensemble_mask, 'age_bucket']
            )
        except:
            ensemble_kappa = np.nan
        
        results.append({
            'method': 'Ensemble',
            'coverage': df['age_bucket'].notna().sum() / len(df),
            'n_classified': df['age_bucket'].notna().sum(),
            'accuracy': ensemble_acc,
            'cohens_kappa': ensemble_kappa
        })
    
    return pd.DataFrame(results)


def method_agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Cohen's κ between all pairs of methods.
    """
    methods = ['age_bucket_self_declared', 'age_bucket_community', 'age_bucket_llm', 'age_bucket']
    method_names = ['Self-Declaration', 'Community', 'LLM', 'Ensemble']
    
    agreement_matrix = np.zeros((len(methods), len(methods)))
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                mask = df[m1].notna() & df[m2].notna()
                if mask.sum() > 10:
                    try:
                        kappa = cohen_kappa_score(df.loc[mask, m1], df.loc[mask, m2])
                        agreement_matrix[i, j] = kappa
                    except:
                        agreement_matrix[i, j] = np.nan
                else:
                    agreement_matrix[i, j] = np.nan
    
    return pd.DataFrame(agreement_matrix, index=method_names, columns=method_names)


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================

def apply_multiple_comparison_correction(
    pvalues: pd.Series,
    method: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Apply multiple comparison correction to p-values.
    """
    n_tests = len(pvalues)
    
    results = pd.DataFrame({
        'p_uncorrected': pvalues
    })
    
    if method == 'bonferroni':
        results['p_corrected'] = np.minimum(pvalues * n_tests, 1.0)
        results['correction'] = 'Bonferroni'
    elif method == 'fdr':
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues.iloc[sorted_idx]
        
        fdr_corrected = np.zeros(n_tests)
        for i, (idx, pval) in enumerate(zip(sorted_idx, sorted_pvals)):
            fdr_corrected[idx] = min(pval * n_tests / (i + 1), 1.0)
        
        results['p_corrected'] = fdr_corrected
        results['correction'] = 'FDR (BH)'
    
    results['significant_corrected'] = results['p_corrected'] < 0.05
    
    return results


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================

def run_model_diagnostics(model) -> Dict[str, Any]:
    """
    Run diagnostic tests on fitted model.
    """
    results = {}
    
    # 1. Heteroscedasticity (Breusch-Pagan test)
    try:
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
        results['breusch_pagan_stat'] = bp_stat
        results['breusch_pagan_p'] = bp_pvalue
        results['heteroscedasticity'] = 'Yes' if bp_pvalue < 0.05 else 'No'
    except Exception as e:
        results['breusch_pagan_error'] = str(e)
    
    # 2. Residual normality (Jarque-Bera)
    try:
        jb_stat, jb_pvalue, _, _ = stats.jarque_bera(model.resid)
        results['jarque_bera_stat'] = jb_stat
        results['jarque_bera_p'] = jb_pvalue
        results['residuals_normal'] = 'Yes' if jb_pvalue > 0.05 else 'No'
    except Exception as e:
        results['jarque_bera_error'] = str(e)
    
    # 3. Influential observations (Cook's distance)
    try:
        influence = OLSInfluence(model)
        cooks_d = influence.cooks_distance[0]
        n_influential = (cooks_d > 4/len(cooks_d)).sum()
        results['n_influential_obs'] = n_influential
        results['max_cooks_d'] = cooks_d.max()
    except Exception as e:
        results['cooks_d_error'] = str(e)
    
    # 4. VIF for multicollinearity
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = model.model.exog
        vif_data = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X, i)
            vif_data.append(vif)
        results['max_vif'] = max(vif_data)
        results['multicollinearity'] = 'High' if max(vif_data) > 10 else 'Low'
    except Exception as e:
        results['vif_error'] = str(e)
    
    return results


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_full_neurips_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    run_bootstrap: bool = True,
    n_bootstrap: int = 500  # Reduced for speed
) -> Dict[str, Any]:
    """
    Run complete NeurIPS-level analysis.
    """
    logger.info("=" * 70)
    logger.info("RUNNING FULL NEURIPS-LEVEL ANALYSIS")
    logger.info("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Method comparison
    logger.info("\n=== PHASE 1: Method Comparison ===")
    method_comparison = compare_classification_methods(df)
    results['method_comparison'] = method_comparison
    logger.info(f"\n{method_comparison.to_string()}")
    
    # 2. Agreement matrix
    agreement = method_agreement_matrix(df)
    results['agreement_matrix'] = agreement
    logger.info(f"\nAgreement matrix:\n{agreement.to_string()}")
    
    # 3. Create 3-bucket age
    logger.info("\n=== PHASE 2: 3-Bucket Age Classification ===")
    df['age_3bucket'] = df['age_bucket'].apply(convert_to_3_buckets)
    df['age_3bucket_community'] = df['age_bucket_community'].apply(convert_to_3_buckets)
    
    # Calculate 3-bucket accuracy for ensemble
    mask_3b = df['age_bucket_self_declared'].notna() & df['age_3bucket'].notna()
    if mask_3b.sum() > 0:
        gt_3b = df.loc[mask_3b, 'age_bucket_self_declared'].apply(convert_to_3_buckets)
        acc_3b = (gt_3b == df.loc[mask_3b, 'age_3bucket']).mean()
        results['age_3bucket_accuracy'] = acc_3b
        logger.info(f"3-bucket ensemble accuracy: {acc_3b:.1%}")
    
    # Calculate 3-bucket accuracy for community embeddings specifically
    mask_comm = df['age_bucket_self_declared'].notna() & df['age_3bucket_community'].notna()
    if mask_comm.sum() > 0:
        gt_3b_comm = df.loc[mask_comm, 'age_bucket_self_declared'].apply(convert_to_3_buckets)
        acc_3b_comm = (gt_3b_comm == df.loc[mask_comm, 'age_3bucket_community']).mean()
        results['age_3bucket_community_accuracy'] = acc_3b_comm
        logger.info(f"3-bucket community embedding accuracy: {acc_3b_comm:.1%}")
    
    # 4. Hierarchical regression
    logger.info("\n=== PHASE 3: Hierarchical Regression ===")
    hier_results = run_hierarchical_regression(df, age_scheme='3_bucket', exclude_unknown_gender=True)
    results['hierarchical_regression'] = hier_results
    
    # 5. Robustness checks
    logger.info("\n=== PHASE 4: Robustness Checks ===")
    
    # Sensitivity analysis
    sensitivity = sensitivity_analysis_thresholds(df)
    results['sensitivity'] = sensitivity
    logger.info(f"\nSensitivity analysis:\n{sensitivity.to_string()}")
    
    # Temporal stability
    temporal = temporal_stability_analysis(df)
    results['temporal'] = temporal
    if temporal:
        for year, data in temporal.items():
            logger.info(f"Year {year}: N={data['n']}, R²={data['r_squared']:.6f}")
    
    # Subreddit-level
    subreddit = subreddit_level_analysis(df)
    results['subreddit'] = subreddit
    if subreddit:
        for sub, data in subreddit.items():
            logger.info(f"{sub}: N={data['n']}, R²={data['r_squared']:.6f}")
    
    # 6. Multiple comparison correction
    logger.info("\n=== PHASE 5: Multiple Comparison Corrections ===")
    if 'full' in hier_results.get('models', {}):
        full_model = hier_results['models']['full']
        corrected_pvals = apply_multiple_comparison_correction(full_model.pvalues, method='fdr')
        results['corrected_pvalues'] = corrected_pvals
        n_sig_before = (full_model.pvalues < 0.05).sum()
        n_sig_after = corrected_pvals['significant_corrected'].sum()
        logger.info(f"Significant before correction: {n_sig_before}")
        logger.info(f"Significant after FDR correction: {n_sig_after}")
    
    # 7. Model diagnostics
    logger.info("\n=== PHASE 6: Model Diagnostics ===")
    if 'full' in hier_results.get('models', {}):
        diagnostics = run_model_diagnostics(hier_results['models']['full'])
        results['diagnostics'] = diagnostics
        for key, value in diagnostics.items():
            logger.info(f"  {key}: {value}")
    
    # 8. Bootstrap (optional, time-consuming)
    if run_bootstrap and 'regression_df' in hier_results:
        logger.info("\n=== PHASE 7: Bootstrap CIs ===")
        reg_df = hier_results['regression_df']
        
        age_terms = hier_results.get('age_terms', [])
        gender_terms = hier_results.get('gender_terms', [])
        
        if age_terms and gender_terms:
            formula = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms)
            bootstrap_results = bootstrap_coefficients(reg_df, formula, n_iterations=n_bootstrap)
            results['bootstrap'] = bootstrap_results
            logger.info(f"\nBootstrap CIs (n={n_bootstrap}):\n{bootstrap_results.to_string()}")
    
    # 9. Generate comprehensive report
    logger.info("\n=== GENERATING FINAL REPORT ===")
    generate_neurips_report(results, output_dir / 'neurips_analysis_report.txt')
    
    return results


def generate_neurips_report(results: Dict, output_path: Path):
    """Generate comprehensive NeurIPS-level report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("NEURIPS-LEVEL ANALYSIS REPORT")
    lines.append("Teen-AI Companion Anthropomorphization Study")
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append("=" * 80)
    lines.append("")
    
    # Method comparison
    lines.append("1. METHOD COMPARISON")
    lines.append("-" * 40)
    if 'method_comparison' in results:
        lines.append(results['method_comparison'].to_string())
    lines.append("")
    
    # Agreement matrix
    lines.append("2. INTER-METHOD AGREEMENT (Cohen's κ)")
    lines.append("-" * 40)
    if 'agreement_matrix' in results:
        lines.append(results['agreement_matrix'].to_string())
    lines.append("")
    
    # 3-bucket accuracy
    lines.append("3. SIMPLIFIED AGE CLASSIFICATION (3-BUCKET)")
    lines.append("-" * 40)
    if 'age_3bucket_accuracy' in results:
        lines.append(f"Ensemble accuracy: {results['age_3bucket_accuracy']:.1%}")
    if 'age_3bucket_community_accuracy' in results:
        lines.append(f"Community embedding accuracy: {results['age_3bucket_community_accuracy']:.1%}")
    lines.append("")
    
    # Hierarchical regression
    lines.append("4. HIERARCHICAL REGRESSION RESULTS")
    lines.append("-" * 40)
    if 'hierarchical_regression' in results:
        hier = results['hierarchical_regression']
        for model_name, model in hier.get('models', {}).items():
            if model is not None and hasattr(model, 'rsquared'):
                lines.append(f"\n{model_name}:")
                lines.append(f"  R² = {model.rsquared:.6f}")
                lines.append(f"  Adj R² = {model.rsquared_adj:.6f}")
                lines.append(f"  N = {int(model.nobs)}")
                
                # Significant coefficients
                sig_coeffs = [p for p in model.pvalues.index if model.pvalues[p] < 0.05]
                if sig_coeffs:
                    lines.append(f"  Significant (p<0.05): {', '.join(sig_coeffs)}")
    lines.append("")
    
    # Robustness
    lines.append("5. ROBUSTNESS CHECKS")
    lines.append("-" * 40)
    
    if 'sensitivity' in results:
        lines.append("\nSensitivity Analysis (varying thresholds):")
        lines.append(results['sensitivity'].to_string())
    
    if 'temporal' in results:
        lines.append("\nTemporal Stability:")
        for year, data in results['temporal'].items():
            lines.append(f"  {year}: N={data['n']}, R²={data['r_squared']:.6f}")
    
    if 'subreddit' in results:
        lines.append("\nSubreddit-Level Analysis:")
        for sub, data in results['subreddit'].items():
            lines.append(f"  {sub}: N={data['n']}, R²={data['r_squared']:.6f}")
    lines.append("")
    
    # Multiple comparison
    lines.append("6. MULTIPLE COMPARISON CORRECTIONS")
    lines.append("-" * 40)
    if 'corrected_pvalues' in results:
        cp = results['corrected_pvalues']
        lines.append(f"Before correction: {(cp['p_uncorrected'] < 0.05).sum()} significant")
        lines.append(f"After FDR correction: {cp['significant_corrected'].sum()} significant")
    lines.append("")
    
    # Diagnostics
    lines.append("7. MODEL DIAGNOSTICS")
    lines.append("-" * 40)
    if 'diagnostics' in results:
        for key, value in results['diagnostics'].items():
            lines.append(f"  {key}: {value}")
    lines.append("")
    
    # Bootstrap
    if 'bootstrap' in results:
        lines.append("8. BOOTSTRAP CONFIDENCE INTERVALS")
        lines.append("-" * 40)
        lines.append(results['bootstrap'].to_string())
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Report saved to {output_path}")

