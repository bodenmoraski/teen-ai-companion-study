"""
Statistical regression models for research question RQ2.

This module implements regression analysis to examine relationships between
demographics and anthropomorphization (AnthroScore).
"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

logger = logging.getLogger(__name__)


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for regression analysis.
    
    Args:
        df: Merged dataset with demographics and features
        
    Returns:
        DataFrame prepared for regression
    """
    logger.info("Preparing data for regression analysis")
    
    # Copy dataframe
    reg_df = df.copy()
    
    # Filter to users with age and AnthroScore
    reg_df = reg_df[
        reg_df['age_bucket'].notna() & 
        reg_df['anthroscore_mean'].notna()
    ].copy()
    
    # Create dummy variables for age buckets (reference: 26-40)
    age_dummies = pd.get_dummies(reg_df['age_bucket'], prefix='age')
    if 'age_26-40' in age_dummies.columns:
        age_dummies = age_dummies.drop('age_26-40', axis=1)  # Reference category
    reg_df = pd.concat([reg_df, age_dummies], axis=1)
    
    # Create dummy variables for gender (reference: unknown/None)
    if 'gender' in reg_df.columns:
        reg_df['gender_encoded'] = reg_df['gender'].fillna('unknown')
        gender_dummies = pd.get_dummies(reg_df['gender_encoded'], prefix='gender')
        if 'gender_unknown' in gender_dummies.columns:
            gender_dummies = gender_dummies.drop('gender_unknown', axis=1)
        reg_df = pd.concat([reg_df, gender_dummies], axis=1)
    
    # Create interaction terms (age × gender)
    if 'gender_encoded' in reg_df.columns:
        for age_col in age_dummies.columns:
            for gender_col in gender_dummies.columns:
                reg_df[f'{age_col}_x_{gender_col}'] = (
                    reg_df[age_col] * reg_df[gender_col]
                )
    
    logger.info(f"Prepared {len(reg_df)} observations for regression")
    return reg_df


def run_rq2_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run regression model for RQ2: How do demographics correlate with anthropomorphization?
    
    Model: AnthroScore ~ Age + Gender + (Age×Gender)
    
    Args:
        df: Prepared regression dataframe
        
    Returns:
        Dictionary with model results and statistics
    """
    logger.info("Running RQ2 regression model")
    
    # Prepare data
    reg_df = prepare_regression_data(df)
    
    if len(reg_df) < 50:
        logger.warning("Insufficient data for regression analysis")
        return {"error": "Insufficient data"}
    
    # Build formula
    age_terms = [col for col in reg_df.columns if col.startswith('age_') and col != 'age_26-40']
    gender_terms = [col for col in reg_df.columns if col.startswith('gender_') and col != 'gender_unknown']
    
    # Model 1: Age only
    formula1 = "anthroscore_mean ~ " + " + ".join(age_terms)
    try:
        model1 = ols(formula1, data=reg_df).fit()
        logger.info("Fitted Model 1: Age only")
    except Exception as e:
        logger.error(f"Error fitting Model 1: {e}")
        model1 = None
    
    # Model 2: Age + Gender
    if gender_terms:
        formula2 = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms)
        try:
            model2 = ols(formula2, data=reg_df).fit()
            logger.info("Fitted Model 2: Age + Gender")
        except Exception as e:
            logger.error(f"Error fitting Model 2: {e}")
            model2 = None
    else:
        model2 = None
    
    # Model 3: Age + Gender + (Age×Gender)
    interaction_terms = [col for col in reg_df.columns if '_x_' in col]
    if interaction_terms:
        formula3 = "anthroscore_mean ~ " + " + ".join(age_terms + gender_terms + interaction_terms)
        try:
            model3 = ols(formula3, data=reg_df).fit()
            logger.info("Fitted Model 3: Age + Gender + Interactions")
        except Exception as e:
            logger.error(f"Error fitting Model 3: {e}")
            model3 = None
    else:
        model3 = None
    
    results = {
        "model1_age_only": model1,
        "model2_age_gender": model2,
        "model3_full": model3,
        "n_observations": len(reg_df),
        "age_terms": age_terms,
        "gender_terms": gender_terms,
        "regression_df": reg_df  # Include for effect size calculations
    }
    
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
    # Partial eta-squared (η²) for each predictor
    # η² = SS_effect / (SS_effect + SS_error)
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    effect_sizes = {}
    for predictor in anova_table.index:
        if predictor != 'Residual':
            ss_effect = anova_table.loc[predictor, 'sum_sq']
            ss_error = anova_table.loc['Residual', 'sum_sq']
            eta_squared = ss_effect / (ss_effect + ss_error)
            effect_sizes[f'{predictor}_eta2'] = eta_squared
    
    # Cohen's f² for model
    # f² = R² / (1 - R²)
    f_squared = model.rsquared / (1 - model.rsquared) if model.rsquared < 1 else np.inf
    effect_sizes['cohens_f2'] = f_squared
    
    # Cohen's d for significant coefficients (approximate)
    # d ≈ 2t / sqrt(df)
    for param in model.params.index:
        if param != 'Intercept':
            t_val = model.tvalues[param]
            df = model.df_resid
            cohens_d = 2 * abs(t_val) / np.sqrt(df) if df > 0 else np.nan
            effect_sizes[f'{param}_cohens_d'] = cohens_d
    
    return effect_sizes


def generate_regression_tables(results: Dict[str, Any], output_path) -> None:
    """
    Generate comprehensive regression tables for NeurIPS-level publication.
    
    Includes: coefficients, CI, p-values, effect sizes, model fit, diagnostics.
    
    Args:
        results: Dictionary with regression results
        output_path: Path to save tables
    """
    logger.info(f"Generating comprehensive regression tables to {output_path}")
    
    tables = []
    
    for model_name, model in [
        ("Model 1: Age Only", results.get("model1_age_only")),
        ("Model 2: Age + Gender", results.get("model2_age_gender")),
        ("Model 3: Full Model (Age + Gender + Interactions)", results.get("model3_full"))
    ]:
        if model is not None:
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
            except:
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
            tables.append(f"  N observations: {model.nobs}")
            tables.append(f"  Degrees of freedom (residual): {model.df_resid}")
            
            # Effect sizes
            if effect_sizes:
                tables.append(f"\nEffect Sizes:")
                if 'cohens_f2' in effect_sizes:
                    tables.append(f"  Cohen's f²: {effect_sizes['cohens_f2']:.4f}")
                    # Interpret f²
                    f2 = effect_sizes['cohens_f2']
                    if f2 < 0.02:
                        interp = "small"
                    elif f2 < 0.15:
                        interp = "medium"
                    else:
                        interp = "large"
                    tables.append(f"    Interpretation: {interp} effect")
            
            # Model comparison (if multiple models)
            if model_name != "Model 1: Age Only":
                prev_model = results.get("model1_age_only")
                if prev_model is not None:
                    # Likelihood ratio test
                    try:
                        lr_stat = -2 * (prev_model.llf - model.llf)
                        lr_pvalue = 1 - stats.chi2.cdf(lr_stat, model.df_model - prev_model.df_model)
                        tables.append(f"\nModel Comparison (vs. Model 1):")
                        tables.append(f"  Likelihood ratio test: {lr_stat:.4f}")
                        tables.append(f"  p-value: {lr_pvalue:.4e}")
                        tables.append(f"  ΔR²: {model.rsquared - prev_model.rsquared:.4f}")
                    except:
                        pass
    
    # Write comprehensive tables
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RQ2: Regression Analysis Results (NeurIPS-Level Statistics)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total observations: {results.get('n_observations', 'N/A')}\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(tables))
        
        # Add interpretation notes
        f.write("\n\n" + "=" * 70)
        f.write("\nInterpretation Notes:\n")
        f.write("=" * 70 + "\n")
        f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n")
        f.write("CI: 95% confidence intervals\n")
        f.write("Effect sizes: Cohen's f² < 0.02 (small), 0.02-0.15 (medium), >0.15 (large)\n")
        f.write("AIC/BIC: Lower is better (model comparison)\n")
    
    logger.info("Comprehensive regression tables generated")

