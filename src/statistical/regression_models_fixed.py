"""
Fixed regression models for research question RQ2.

This version fixes the column naming issue where hyphens in age bucket names
(e.g., 'age_13-18') are interpreted as minus signs by patsy formula parser.

FIX: Replace hyphens with underscores in column names (e.g., 'age_13_18').
"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

logger = logging.getLogger(__name__)


def prepare_regression_data_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for regression analysis with FIXED column naming.
    
    FIX: Uses underscores instead of hyphens in dummy variable names
    to avoid patsy formula parsing issues.
    
    Args:
        df: Merged dataset with demographics and features
        
    Returns:
        DataFrame prepared for regression
    """
    logger.info("Preparing data for regression analysis (FIXED version)")
    
    # Copy dataframe
    reg_df = df.copy()
    
    # Filter to users with age and AnthroScore
    reg_df = reg_df[
        reg_df['age_bucket'].notna() & 
        reg_df['anthroscore_mean'].notna()
    ].copy()
    
    if len(reg_df) == 0:
        logger.warning("No valid observations after filtering")
        return reg_df
    
    # FIX: Convert age bucket values to patsy-safe names (replace - with _)
    reg_df['age_bucket_safe'] = reg_df['age_bucket'].str.replace('-', '_', regex=False)
    
    # Create dummy variables for age buckets (reference: 26_40)
    age_dummies = pd.get_dummies(reg_df['age_bucket_safe'], prefix='age', dtype=int)
    
    # Drop reference category if it exists
    ref_col = 'age_26_40'
    if ref_col in age_dummies.columns:
        age_dummies = age_dummies.drop(ref_col, axis=1)
    
    reg_df = pd.concat([reg_df, age_dummies], axis=1)
    
    # Create dummy variables for gender (reference: unknown/None)
    if 'gender' in reg_df.columns:
        reg_df['gender_encoded'] = reg_df['gender'].fillna('unknown')
        gender_dummies = pd.get_dummies(reg_df['gender_encoded'], prefix='gender', dtype=int)
        
        # Drop reference category
        if 'gender_unknown' in gender_dummies.columns:
            gender_dummies = gender_dummies.drop('gender_unknown', axis=1)
        
        reg_df = pd.concat([reg_df, gender_dummies], axis=1)
        
        # Create interaction terms (age × gender) - also with safe names
        age_dummy_cols = [c for c in age_dummies.columns]
        gender_dummy_cols = [c for c in gender_dummies.columns]
        
        for age_col in age_dummy_cols:
            for gender_col in gender_dummy_cols:
                # Create interaction term with patsy-safe name
                interaction_name = f'{age_col}_x_{gender_col}'
                reg_df[interaction_name] = reg_df[age_col] * reg_df[gender_col]
    
    logger.info(f"Prepared {len(reg_df)} observations for regression")
    
    # Log column info for debugging
    age_cols = [c for c in reg_df.columns if c.startswith('age_') and 
                c not in ['age_bucket', 'age_bucket_safe', 'age_bucket_community', 
                         'age_bucket_llm', 'age_bucket_self_declared', 'age_community_score']]
    gender_cols = [c for c in reg_df.columns if c.startswith('gender_') and 
                   c not in ['gender', 'gender_encoded', 'gender_community', 
                            'gender_self_declared', 'gender_community_score']]
    
    logger.debug(f"Age dummy columns: {age_cols}")
    logger.debug(f"Gender dummy columns: {gender_cols}")
    
    return reg_df


def run_rq2_regression_fixed(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run regression model for RQ2 with FIXED column naming.
    
    Model: AnthroScore ~ Age + Gender + (Age×Gender)
    
    Args:
        df: Merged dataset with demographics and features
        
    Returns:
        Dictionary with model results and statistics
    """
    logger.info("Running RQ2 regression model (FIXED version)")
    
    # Prepare data
    reg_df = prepare_regression_data_fixed(df)
    
    if len(reg_df) < 50:
        logger.warning("Insufficient data for regression analysis")
        return {"error": "Insufficient data", "n_observations": len(reg_df)}
    
    # Get column names for formula building
    # Age terms: age_13_18, age_19_25, age_41_60, age_61_80 (reference: age_26_40)
    age_terms = [c for c in reg_df.columns if c.startswith('age_') and 
                 c not in ['age_bucket', 'age_bucket_safe', 'age_bucket_community',
                          'age_bucket_llm', 'age_bucket_self_declared', 'age_community_score', 
                          'age_26_40'] and '_x_' not in c]
    
    # Gender terms: gender_female, gender_male, gender_nonbinary (reference: gender_unknown)
    gender_terms = [c for c in reg_df.columns if c.startswith('gender_') and 
                    c not in ['gender', 'gender_encoded', 'gender_community',
                             'gender_self_declared', 'gender_community_score', 
                             'gender_unknown'] and '_x_' not in c]
    
    # Interaction terms
    interaction_terms = [c for c in reg_df.columns if '_x_' in c]
    
    logger.info(f"Age terms: {age_terms}")
    logger.info(f"Gender terms: {gender_terms}")
    logger.info(f"Interaction terms: {len(interaction_terms)} terms")
    
    results = {
        "n_observations": len(reg_df),
        "age_terms": age_terms,
        "gender_terms": gender_terms,
        "regression_df": reg_df
    }
    
    # Model 1: Age only
    if age_terms:
        formula1 = "anthroscore_mean ~ " + " + ".join(age_terms)
        logger.info(f"Model 1 formula: {formula1}")
        try:
            model1 = ols(formula1, data=reg_df).fit()
            results["model1_age_only"] = model1
            logger.info(f"Fitted Model 1: Age only (R² = {model1.rsquared:.4f})")
        except Exception as e:
            logger.error(f"Error fitting Model 1: {e}")
            results["model1_age_only"] = None
            results["model1_error"] = str(e)
    else:
        logger.warning("No age terms available for Model 1")
        results["model1_age_only"] = None
    
    # Model 2: Age + Gender
    if age_terms and gender_terms:
        formula2 = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms)
        logger.info(f"Model 2 formula: {formula2}")
        try:
            model2 = ols(formula2, data=reg_df).fit()
            results["model2_age_gender"] = model2
            logger.info(f"Fitted Model 2: Age + Gender (R² = {model2.rsquared:.4f})")
        except Exception as e:
            logger.error(f"Error fitting Model 2: {e}")
            results["model2_age_gender"] = None
            results["model2_error"] = str(e)
    else:
        results["model2_age_gender"] = None
    
    # Model 3: Age + Gender + (Age×Gender)
    if age_terms and gender_terms and interaction_terms:
        formula3 = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms + interaction_terms)
        logger.info(f"Model 3 formula: {formula3[:100]}...")
        try:
            model3 = ols(formula3, data=reg_df).fit()
            results["model3_full"] = model3
            logger.info(f"Fitted Model 3: Full model (R² = {model3.rsquared:.4f})")
        except Exception as e:
            logger.error(f"Error fitting Model 3: {e}")
            results["model3_full"] = None
            results["model3_error"] = str(e)
    else:
        results["model3_full"] = None
    
    return results


def calculate_effect_sizes(model, reg_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate effect sizes for regression model.
    
    Args:
        model: Fitted OLS model
        reg_df: Regression dataframe
        
    Returns:
        Dictionary with effect size metrics
    """
    effect_sizes = {}
    
    try:
        # Partial eta-squared (η²) for each predictor
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        for predictor in anova_table.index:
            if predictor != 'Residual':
                ss_effect = anova_table.loc[predictor, 'sum_sq']
                ss_error = anova_table.loc['Residual', 'sum_sq']
                eta_squared = ss_effect / (ss_effect + ss_error)
                effect_sizes[f'{predictor}_eta2'] = eta_squared
    except Exception as e:
        logger.warning(f"Could not calculate eta-squared: {e}")
    
    # Cohen's f² for model
    # f² = R² / (1 - R²)
    if model.rsquared < 1:
        f_squared = model.rsquared / (1 - model.rsquared)
    else:
        f_squared = np.inf
    effect_sizes['cohens_f2'] = f_squared
    
    # Cohen's d for significant coefficients (approximate)
    # d ≈ 2t / sqrt(df)
    try:
        for param in model.params.index:
            if param != 'Intercept':
                t_val = model.tvalues[param]
                df = model.df_resid
                if df > 0:
                    cohens_d = 2 * abs(t_val) / np.sqrt(df)
                    effect_sizes[f'{param}_cohens_d'] = cohens_d
    except Exception as e:
        logger.warning(f"Could not calculate Cohen's d: {e}")
    
    return effect_sizes


def generate_regression_tables_fixed(results: Dict[str, Any], output_path) -> None:
    """
    Generate comprehensive regression tables for NeurIPS-level publication.
    
    Includes: coefficients, CI, p-values, effect sizes, model fit, diagnostics.
    
    Args:
        results: Dictionary with regression results
        output_path: Path to save tables
    """
    logger.info(f"Generating comprehensive regression tables to {output_path}")
    
    tables = []
    
    model_configs = [
        ("Model 1: Age Only", "model1_age_only"),
        ("Model 2: Age + Gender", "model2_age_gender"),
        ("Model 3: Full Model (Age + Gender + Interactions)", "model3_full")
    ]
    
    for model_name, model_key in model_configs:
        model = results.get(model_key)
        
        if model is None:
            error = results.get(f"{model_key.replace('model', 'model')}_error", "Not fitted")
            tables.append(f"\n{model_name}\n" + "=" * 70)
            tables.append(f"\nModel not fitted: {error}")
            continue
            
        # Get regression dataframe for effect sizes
        reg_df = results.get('regression_df')
        
        # Create comprehensive summary table
        conf_int = model.conf_int(alpha=0.05)  # 95% CI
        summary_df = pd.DataFrame({
            'Coefficient': model.params,
            'Std Error': model.bse,
            't-value': model.tvalues,
            'p-value': model.pvalues,
            'CI Lower (95%)': conf_int[0],
            'CI Upper (95%)': conf_int[1]
        })
        summary_df['Significance'] = summary_df['p-value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        )
        
        # Calculate effect sizes
        try:
            effect_sizes = calculate_effect_sizes(model, reg_df) if reg_df is not None else {}
        except Exception as e:
            logger.warning(f"Could not calculate effect sizes: {e}")
            effect_sizes = {}
        
        tables.append(f"\n{model_name}\n" + "=" * 70)
        tables.append("\nCoefficients:")
        tables.append(summary_df.to_string())
        
        # Model fit statistics
        tables.append(f"\n\nModel Fit Statistics:")
        tables.append(f"  R-squared: {model.rsquared:.4f}")
        tables.append(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
        tables.append(f"  AIC: {model.aic:.2f}")
        tables.append(f"  BIC: {model.bic:.2f}")
        tables.append(f"  F-statistic: {model.fvalue:.4f}")
        tables.append(f"  F-statistic p-value: {model.f_pvalue:.4e}")
        tables.append(f"  Log-likelihood: {model.llf:.2f}")
        tables.append(f"  N observations: {int(model.nobs)}")
        tables.append(f"  Degrees of freedom (residual): {int(model.df_resid)}")
        
        # Effect sizes
        if effect_sizes:
            tables.append(f"\nEffect Sizes:")
            if 'cohens_f2' in effect_sizes:
                f2 = effect_sizes['cohens_f2']
                tables.append(f"  Cohen's f2: {f2:.6f}")
                # Interpret f²
                if f2 < 0.02:
                    interp = "negligible"
                elif f2 < 0.15:
                    interp = "small"
                elif f2 < 0.35:
                    interp = "medium"
                else:
                    interp = "large"
                tables.append(f"    Interpretation: {interp} effect")
        
        # Model comparison (if multiple models)
        if model_key != "model1_age_only":
            prev_model = results.get("model1_age_only")
            if prev_model is not None:
                try:
                    # Likelihood ratio test
                    df_diff = model.df_model - prev_model.df_model
                    if df_diff > 0:
                        lr_stat = -2 * (prev_model.llf - model.llf)
                        lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df_diff)
                        tables.append(f"\nModel Comparison (vs. Model 1):")
                        tables.append(f"  Likelihood ratio test: {lr_stat:.4f}")
                        tables.append(f"  Degrees of freedom: {int(df_diff)}")
                        tables.append(f"  p-value: {lr_pvalue:.4e}")
                        tables.append(f"  Delta R-squared: {model.rsquared - prev_model.rsquared:.6f}")
                except Exception as e:
                    logger.warning(f"Could not compute model comparison: {e}")
    
    # Write comprehensive tables
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RQ2: Regression Analysis Results (NeurIPS-Level Statistics)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total observations: {results.get('n_observations', 'N/A')}\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference categories: age_26_40 (age), gender_unknown (gender)\n\n")
        f.write("\n".join(tables))
        
        # Add interpretation notes
        f.write("\n\n" + "=" * 70)
        f.write("\nInterpretation Notes:\n")
        f.write("=" * 70 + "\n")
        f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n")
        f.write("CI: 95% confidence intervals\n")
        f.write("Effect sizes: Cohen's f2 < 0.02 (negligible), 0.02-0.15 (small), 0.15-0.35 (medium), >0.35 (large)\n")
        f.write("AIC/BIC: Lower is better (model comparison)\n")
        f.write("\nNote: Age bucket column names use underscores (e.g., age_13_18) for compatibility.\n")
    
    logger.info("Comprehensive regression tables generated successfully")


# Alias for backwards compatibility
prepare_regression_data = prepare_regression_data_fixed
run_rq2_regression = run_rq2_regression_fixed
generate_regression_tables = generate_regression_tables_fixed

