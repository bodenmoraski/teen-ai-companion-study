#!/usr/bin/env python3
"""
Missing Statistical Analyses
=============================

This script runs the missing analyses identified in the research plan:
1. Three-way ANOVA: Age × Gender × Intent → AnthroScore
2. Mediation analysis: Does intent mediate the age → anthropomorphization relationship?
3. Nonlinear effects testing: Quadratic terms, threshold detection

Output: results/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_full_dataset() -> pd.DataFrame:
    """Load the full merged dataset with all variables."""
    logger.info("Loading full merged dataset...")
    
    # Try the full merged dataset first
    full_path = DATA_DIR / "full_merged_dataset.parquet"
    if full_path.exists():
        df = pd.read_parquet(full_path)
        logger.info(f"Loaded full_merged_dataset: {len(df):,} records")
        
        # Check if intent data is missing - if so, merge it
        if 'intent_category' not in df.columns and 'dominant_intent' not in df.columns:
            intent_path = DATA_DIR / "intent_topics.parquet"
            if intent_path.exists():
                intent = pd.read_parquet(intent_path)
                intent_col = 'dominant_intent' if 'dominant_intent' in intent.columns else 'intent_category'
                if intent_col in intent.columns:
                    df = df.merge(intent[['author', intent_col]], on='author', how='left')
                    df = df.rename(columns={intent_col: 'intent_category'})
                    logger.info(f"Merged intent data: {df['intent_category'].notna().sum():,} users with intent")
        
        return df
    
    # Otherwise merge manually
    logger.info("Merging datasets manually...")
    
    anthro = pd.read_parquet(DATA_DIR / "user_anthroscores.parquet")
    
    age_pred = pd.read_parquet(DATA_DIR / "ultimate_predictor" / "ultimate_predictions.parquet")
    age_col = 'age_bucket_predicted' if 'age_bucket_predicted' in age_pred.columns else 'age_bucket'
    age_pred = age_pred.rename(columns={age_col: 'age_bucket', 'confidence': 'age_confidence'})
    
    gender_pred = pd.read_parquet(DATA_DIR / "ultimate_predictor" / "gender_predictions.parquet")
    gender_col = 'gender_predicted' if 'gender_predicted' in gender_pred.columns else 'gender'
    gender_pred = gender_pred.rename(columns={gender_col: 'gender', 'confidence': 'gender_confidence'})
    
    intent = pd.read_parquet(DATA_DIR / "intent_topics.parquet")
    
    # Merge
    df = anthro.merge(age_pred[['author', 'age_bucket', 'age_confidence']], on='author', how='left')
    df = df.merge(gender_pred[['author', 'gender', 'gender_confidence']], on='author', how='left')
    # Get intent column
    intent_col = None
    for col in ['dominant_intent', 'intent_category', 'topic']:
        if col in intent.columns:
            intent_col = col
            break
    
    if intent_col:
        df = df.merge(intent[['author', intent_col]], on='author', how='left')
        if intent_col != 'intent_category':
            df = df.rename(columns={intent_col: 'intent_category'})
    else:
        logger.warning("No intent column found in intent_topics.parquet")
    
    logger.info(f"Merged dataset: {len(df):,} records")
    return df


def prepare_data_for_anova(df: pd.DataFrame, min_confidence: float = 0.6) -> pd.DataFrame:
    """Prepare data for ANOVA analysis."""
    logger.info(f"Preparing data for ANOVA (min_confidence={min_confidence})...")
    
    # Filter by confidence
    if 'age_confidence' in df.columns:
        df = df[df['age_confidence'] >= min_confidence].copy()
    
    # Drop missing values
    required_cols = ['age_bucket', 'gender', 'anthroscore_max']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    
    # Standardize age bucket names
    if 'age_bucket' in df.columns:
        df['age_bucket'] = df['age_bucket'].str.lower()
        df['is_teen'] = df['age_bucket'].isin(['13-18', 'teen', '13-17'])
        df['age_group'] = df['is_teen'].map({True: 'Teen', False: 'Adult'})
    
    # Standardize gender names
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.lower()
        df['gender_group'] = df['gender'].map({'male': 'Male', 'female': 'Female'})
        df = df[df['gender_group'].notna()]
    
    # Standardize intent categories
    if 'intent_category' in df.columns:
        # Keep top categories, group others
        top_intents = df['intent_category'].value_counts().head(5).index.tolist()
        df['intent_group'] = df['intent_category'].apply(
            lambda x: x if x in top_intents else 'Other'
        )
    
    logger.info(f"Prepared data: {len(df):,} records")
    return df


def run_three_way_anova(df: pd.DataFrame) -> Dict:
    """
    Run three-way ANOVA: Age × Gender × Intent → AnthroScore
    
    Tests:
    1. Main effects of Age, Gender, Intent
    2. Two-way interactions
    3. Three-way interaction
    """
    logger.info("=" * 70)
    logger.info("THREE-WAY ANOVA: Age × Gender × Intent → AnthroScore")
    logger.info("=" * 70)
    
    # Prepare data
    df_anova = prepare_data_for_anova(df)
    
    # Check if we have intent data
    has_intent = 'intent_group' in df_anova.columns and df_anova['intent_group'].notna().sum() > 0
    
    results = {
        'n_observations': len(df_anova),
        'groups': {}
    }
    
    # Group sizes
    logger.info("\nGroup sizes:")
    if 'age_group' in df_anova.columns:
        age_counts = df_anova['age_group'].value_counts()
        results['groups']['age'] = age_counts.to_dict()
        logger.info(f"  Age: {age_counts.to_dict()}")
    
    if 'gender_group' in df_anova.columns:
        gender_counts = df_anova['gender_group'].value_counts()
        results['groups']['gender'] = gender_counts.to_dict()
        logger.info(f"  Gender: {gender_counts.to_dict()}")
    
    if has_intent:
        intent_counts = df_anova['intent_group'].value_counts()
        results['groups']['intent'] = intent_counts.to_dict()
        logger.info(f"  Intent: {intent_counts.to_dict()}")
    
    # Run three-way ANOVA (or two-way if no intent data)
    if has_intent and len(df_anova['intent_group'].dropna().unique()) > 1:
        formula = 'anthroscore_max ~ C(age_group) * C(gender_group) * C(intent_group)'
        analysis_type = 'three_way'
    else:
        formula = 'anthroscore_max ~ C(age_group) * C(gender_group)'
        analysis_type = 'two_way'
        logger.warning("Intent data not available/sufficient - running two-way ANOVA")
    
    logger.info(f"\nRunning {analysis_type} ANOVA...")
    logger.info(f"Formula: {formula}")
    
    try:
        model = ols(formula, data=df_anova).fit()
        anova_table = anova_lm(model, typ=2)
        
        results['anova_type'] = analysis_type
        results['r_squared'] = model.rsquared
        results['adj_r_squared'] = model.rsquared_adj
        results['f_statistic'] = model.fvalue
        results['f_pvalue'] = model.f_pvalue
        
        # Extract effects
        logger.info("\n" + "-" * 70)
        logger.info("ANOVA Results:")
        logger.info("-" * 70)
        
        effects = {}
        for idx in anova_table.index:
            if idx != 'Residual':
                row = anova_table.loc[idx]
                f_val = row['F']
                p_val = row['PR(>F)']
                ss = row['sum_sq']
                df_effect = row['df']
                
                # Calculate partial eta-squared
                ss_error = anova_table.loc['Residual', 'sum_sq']
                eta_sq = ss / (ss + ss_error)
                
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                
                effects[idx] = {
                    'F': float(f_val),
                    'p': float(p_val),
                    'df': float(df_effect),
                    'sum_sq': float(ss),
                    'partial_eta_sq': float(eta_sq),
                    'significant': p_val < 0.05
                }
                
                logger.info(f"  {idx:50s}: F={f_val:8.3f}, p={p_val:.4e}, η²={eta_sq:.4f} {sig}")
        
        results['effects'] = effects
        
        # Group means
        logger.info("\n" + "-" * 70)
        logger.info("Group Means (AnthroScore):")
        logger.info("-" * 70)
        
        group_means = df_anova.groupby(['age_group', 'gender_group'])['anthroscore_max'].agg(['mean', 'std', 'count'])
        # Convert MultiIndex to string keys for JSON serialization
        group_means_dict = {}
        for idx in group_means.index:
            key = f"{idx[0]}_{idx[1]}"
            group_means_dict[key] = {
                'mean': float(group_means.loc[idx, 'mean']),
                'std': float(group_means.loc[idx, 'std']),
                'count': int(group_means.loc[idx, 'count'])
            }
        results['group_means'] = group_means_dict
        logger.info(f"\n{group_means}")
        
    except Exception as e:
        logger.error(f"ANOVA failed: {e}")
        results['error'] = str(e)
    
    return results


def run_mediation_analysis(df: pd.DataFrame) -> Dict:
    """
    Mediation analysis: Does intent mediate the Age → AnthroScore relationship?
    
    Tests Baron & Kenny (1986) steps:
    1. Path c: Age → AnthroScore (total effect)
    2. Path a: Age → Intent 
    3. Path b: Intent → AnthroScore (controlling for Age)
    4. Path c': Age → AnthroScore (controlling for Intent) - direct effect
    
    Indirect effect = a * b
    Mediation if c' < c and indirect effect significant
    """
    logger.info("=" * 70)
    logger.info("MEDIATION ANALYSIS: Does Intent Mediate Age → AnthroScore?")
    logger.info("=" * 70)
    
    # Prepare data
    df_med = prepare_data_for_anova(df, min_confidence=0.6)
    
    # Need numeric encoding for regression
    df_med['age_numeric'] = df_med['is_teen'].astype(int)  # 1 = teen, 0 = adult
    
    # Check for intent data
    if 'intent_group' not in df_med.columns or df_med['intent_group'].isna().all():
        logger.warning("No intent data available for mediation analysis")
        return {'error': 'No intent data available'}
    
    # Create dummy for specific intent (e.g., character creation)
    # First find the most common intent
    intent_counts = df_med['intent_group'].value_counts()
    logger.info(f"Intent distribution:\n{intent_counts}")
    
    # Use character creation or the most relevant intent
    target_intent = None
    for intent in ['character_creation', 'Character Creation', 'character creation']:
        if intent in df_med['intent_group'].values:
            target_intent = intent
            break
    
    if target_intent is None:
        target_intent = intent_counts.index[0] if len(intent_counts) > 0 else None
    
    if target_intent is None:
        return {'error': 'Could not identify target intent'}
    
    df_med['target_intent'] = (df_med['intent_group'] == target_intent).astype(int)
    df_med = df_med.dropna(subset=['age_numeric', 'target_intent', 'anthroscore_max'])
    
    results = {
        'n_observations': len(df_med),
        'target_intent': target_intent,
    }
    
    logger.info(f"\nTarget intent for mediation: {target_intent}")
    logger.info(f"Sample size: {len(df_med):,}")
    
    # Step 1: Path c (Total effect) - Age → AnthroScore
    logger.info("\n" + "-" * 70)
    logger.info("Step 1: Total Effect (Age → AnthroScore)")
    logger.info("-" * 70)
    
    X_c = sm.add_constant(df_med['age_numeric'])
    model_c = sm.OLS(df_med['anthroscore_max'], X_c).fit()
    
    c = model_c.params['age_numeric']
    c_se = model_c.bse['age_numeric']
    c_p = model_c.pvalues['age_numeric']
    
    results['path_c'] = {'coef': float(c), 'se': float(c_se), 'p': float(c_p)}
    logger.info(f"  c = {c:.4f} (SE = {c_se:.4f}), p = {c_p:.4f}")
    
    # Step 2: Path a - Age → Intent
    logger.info("\n" + "-" * 70)
    logger.info("Step 2: Path a (Age → Intent)")
    logger.info("-" * 70)
    
    X_a = sm.add_constant(df_med['age_numeric'])
    model_a = sm.OLS(df_med['target_intent'], X_a).fit()
    
    a = model_a.params['age_numeric']
    a_se = model_a.bse['age_numeric']
    a_p = model_a.pvalues['age_numeric']
    
    results['path_a'] = {'coef': float(a), 'se': float(a_se), 'p': float(a_p)}
    logger.info(f"  a = {a:.4f} (SE = {a_se:.4f}), p = {a_p:.4f}")
    
    # Step 3: Path b and c' - Intent → AnthroScore (controlling for Age)
    logger.info("\n" + "-" * 70)
    logger.info("Step 3: Paths b and c' (Intent → AnthroScore, controlling Age)")
    logger.info("-" * 70)
    
    X_bc = sm.add_constant(df_med[['age_numeric', 'target_intent']])
    model_bc = sm.OLS(df_med['anthroscore_max'], X_bc).fit()
    
    b = model_bc.params['target_intent']
    b_se = model_bc.bse['target_intent']
    b_p = model_bc.pvalues['target_intent']
    
    c_prime = model_bc.params['age_numeric']
    c_prime_se = model_bc.bse['age_numeric']
    c_prime_p = model_bc.pvalues['age_numeric']
    
    results['path_b'] = {'coef': float(b), 'se': float(b_se), 'p': float(b_p)}
    results['path_c_prime'] = {'coef': float(c_prime), 'se': float(c_prime_se), 'p': float(c_prime_p)}
    
    logger.info(f"  b = {b:.4f} (SE = {b_se:.4f}), p = {b_p:.4f}")
    logger.info(f"  c' = {c_prime:.4f} (SE = {c_prime_se:.4f}), p = {c_prime_p:.4f}")
    
    # Calculate indirect effect (a * b)
    indirect = a * b
    
    # Sobel test for indirect effect significance
    sobel_se = np.sqrt((a**2 * b_se**2) + (b**2 * a_se**2))
    sobel_z = indirect / sobel_se
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    
    results['indirect_effect'] = {
        'coef': float(indirect),
        'sobel_z': float(sobel_z),
        'sobel_p': float(sobel_p)
    }
    
    # Proportion mediated
    if c != 0:
        prop_mediated = indirect / c
    else:
        prop_mediated = 0
    
    results['proportion_mediated'] = float(prop_mediated)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MEDIATION SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"\nTotal effect (c):     {c:.4f}")
    logger.info(f"Direct effect (c'):   {c_prime:.4f}")
    logger.info(f"Indirect effect (ab): {indirect:.4f}")
    logger.info(f"Proportion mediated:  {prop_mediated:.1%}")
    logger.info(f"\nSobel test: z = {sobel_z:.3f}, p = {sobel_p:.4f}")
    
    # Interpretation
    if sobel_p < 0.05:
        if c_prime_p >= 0.05 and c_p < 0.05:
            mediation_type = "FULL MEDIATION"
        elif c_prime_p < 0.05:
            mediation_type = "PARTIAL MEDIATION"
        else:
            mediation_type = "INDIRECT EFFECT ONLY"
    else:
        mediation_type = "NO MEDIATION"
    
    results['mediation_type'] = mediation_type
    
    logger.info(f"\nConclusion: {mediation_type}")
    
    if mediation_type in ["FULL MEDIATION", "PARTIAL MEDIATION"]:
        logger.info(f"  → Intent ({target_intent}) {'fully' if mediation_type == 'FULL MEDIATION' else 'partially'} "
                   f"mediates the relationship between age and anthropomorphization")
    
    return results


def run_nonlinear_effects(df: pd.DataFrame) -> Dict:
    """
    Test for nonlinear effects:
    1. Quadratic age effects
    2. Threshold detection (is there a specific age where effect changes?)
    3. Polynomial regression comparison
    """
    logger.info("=" * 70)
    logger.info("NONLINEAR EFFECTS ANALYSIS")
    logger.info("=" * 70)
    
    # Load self-declared ages for continuous age variable
    self_decl = pd.read_parquet(DATA_DIR / "self_declarations.parquet")
    
    # Check for numeric age column
    age_col = 'age_self_declared'
    if age_col not in self_decl.columns:
        logger.warning("No continuous age data available")
        return {'error': 'No continuous age data available'}
    
    # Filter to valid ages
    self_decl = self_decl[self_decl[age_col].notna()].copy()
    self_decl['age'] = pd.to_numeric(self_decl[age_col], errors='coerce')
    self_decl = self_decl[(self_decl['age'] >= 10) & (self_decl['age'] <= 80)]
    
    # Merge with anthroscores
    anthro = pd.read_parquet(DATA_DIR / "user_anthroscores.parquet")
    df_nl = anthro.merge(self_decl[['author', 'age']], on='author', how='inner')
    df_nl = df_nl.dropna(subset=['age', 'anthroscore_max'])
    
    logger.info(f"Sample size with continuous age: {len(df_nl):,}")
    logger.info(f"Age range: {df_nl['age'].min():.0f} - {df_nl['age'].max():.0f}")
    
    if len(df_nl) < 50:
        logger.warning("Insufficient data for nonlinear analysis")
        return {'error': 'Insufficient data', 'n': len(df_nl)}
    
    results = {
        'n_observations': len(df_nl),
        'age_range': [float(df_nl['age'].min()), float(df_nl['age'].max())],
    }
    
    # Standardize age for stability
    df_nl['age_centered'] = df_nl['age'] - df_nl['age'].mean()
    df_nl['age_squared'] = df_nl['age_centered'] ** 2
    
    # 1. Linear model
    logger.info("\n" + "-" * 70)
    logger.info("Model 1: Linear (Age → AnthroScore)")
    logger.info("-" * 70)
    
    model_linear = ols('anthroscore_max ~ age_centered', data=df_nl).fit()
    logger.info(f"  R² = {model_linear.rsquared:.4f}")
    logger.info(f"  Age coefficient: {model_linear.params['age_centered']:.4f} (p = {model_linear.pvalues['age_centered']:.4f})")
    
    results['linear'] = {
        'r_squared': float(model_linear.rsquared),
        'aic': float(model_linear.aic),
        'bic': float(model_linear.bic),
        'age_coef': float(model_linear.params['age_centered']),
        'age_p': float(model_linear.pvalues['age_centered'])
    }
    
    # 2. Quadratic model
    logger.info("\n" + "-" * 70)
    logger.info("Model 2: Quadratic (Age + Age² → AnthroScore)")
    logger.info("-" * 70)
    
    model_quadratic = ols('anthroscore_max ~ age_centered + age_squared', data=df_nl).fit()
    logger.info(f"  R² = {model_quadratic.rsquared:.4f}")
    logger.info(f"  Age coefficient: {model_quadratic.params['age_centered']:.4f} (p = {model_quadratic.pvalues['age_centered']:.4f})")
    logger.info(f"  Age² coefficient: {model_quadratic.params['age_squared']:.4f} (p = {model_quadratic.pvalues['age_squared']:.4f})")
    
    results['quadratic'] = {
        'r_squared': float(model_quadratic.rsquared),
        'aic': float(model_quadratic.aic),
        'bic': float(model_quadratic.bic),
        'age_coef': float(model_quadratic.params['age_centered']),
        'age_p': float(model_quadratic.pvalues['age_centered']),
        'age_sq_coef': float(model_quadratic.params['age_squared']),
        'age_sq_p': float(model_quadratic.pvalues['age_squared'])
    }
    
    # Model comparison
    logger.info("\n" + "-" * 70)
    logger.info("Model Comparison")
    logger.info("-" * 70)
    
    # F-test for nested models
    rss_linear = model_linear.ssr
    rss_quadratic = model_quadratic.ssr
    df_diff = 1  # One additional parameter
    n = len(df_nl)
    
    f_stat = ((rss_linear - rss_quadratic) / df_diff) / (rss_quadratic / (n - 3))
    f_p = 1 - stats.f.cdf(f_stat, df_diff, n - 3)
    
    results['model_comparison'] = {
        'f_statistic': float(f_stat),
        'p_value': float(f_p),
        'aic_improvement': float(model_linear.aic - model_quadratic.aic),
        'bic_improvement': float(model_linear.bic - model_quadratic.bic)
    }
    
    logger.info(f"  F-test (quadratic vs linear): F = {f_stat:.3f}, p = {f_p:.4f}")
    logger.info(f"  AIC improvement: {model_linear.aic - model_quadratic.aic:.2f}")
    logger.info(f"  BIC improvement: {model_linear.bic - model_quadratic.bic:.2f}")
    
    if f_p < 0.05:
        logger.info("  ✓ Quadratic term is significant - nonlinear effect detected")
        results['nonlinear_detected'] = True
    else:
        logger.info("  ✗ Quadratic term not significant - linear model sufficient")
        results['nonlinear_detected'] = False
    
    # 3. Threshold detection using piecewise regression
    logger.info("\n" + "-" * 70)
    logger.info("Threshold Detection")
    logger.info("-" * 70)
    
    # Test different thresholds
    thresholds = [16, 18, 21, 25, 30]
    threshold_results = []
    
    for thresh in thresholds:
        df_nl['above_threshold'] = (df_nl['age'] >= thresh).astype(int)
        df_nl['age_x_above'] = df_nl['age_centered'] * df_nl['above_threshold']
        
        try:
            model_thresh = ols('anthroscore_max ~ age_centered + above_threshold + age_x_above', 
                              data=df_nl).fit()
            
            interaction_p = model_thresh.pvalues['age_x_above']
            
            threshold_results.append({
                'threshold': thresh,
                'interaction_p': float(interaction_p),
                'r_squared': float(model_thresh.rsquared),
                'aic': float(model_thresh.aic)
            })
            
            sig = "*" if interaction_p < 0.05 else ""
            logger.info(f"  Threshold at {thresh}: interaction p = {interaction_p:.4f} {sig}")
        except Exception as e:
            logger.warning(f"  Threshold {thresh} failed: {e}")
    
    results['threshold_tests'] = threshold_results
    
    # Find best threshold (if any significant)
    sig_thresholds = [t for t in threshold_results if t['interaction_p'] < 0.05]
    if sig_thresholds:
        best = min(sig_thresholds, key=lambda x: x['aic'])
        results['best_threshold'] = best
        logger.info(f"\n  Best threshold: {best['threshold']} (AIC = {best['aic']:.2f})")
    else:
        logger.info("\n  No significant thresholds detected")
    
    return results


def generate_analyses_report(anova_results: Dict, 
                              mediation_results: Dict,
                              nonlinear_results: Dict) -> str:
    """Generate comprehensive report of all missing analyses."""
    logger.info("Generating analyses report...")
    
    report = []
    report.append("=" * 80)
    report.append("MISSING STATISTICAL ANALYSES REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Section 1: Three-Way ANOVA
    report.append("\n" + "=" * 80)
    report.append("SECTION 1: THREE-WAY ANOVA (Age x Gender x Intent -> AnthroScore)")
    report.append("=" * 80)
    
    if 'error' not in anova_results:
        report.append(f"\nN observations: {anova_results.get('n_observations', 'N/A'):,}")
        report.append(f"R-squared: {anova_results.get('r_squared', 'N/A'):.4f}")
        report.append(f"Adjusted R-squared: {anova_results.get('adj_r_squared', 'N/A'):.4f}")
        
        report.append("\n" + "-" * 70)
        report.append("Effects:")
        report.append("-" * 70)
        
        for effect, stats_dict in anova_results.get('effects', {}).items():
            sig = "***" if stats_dict['p'] < 0.001 else "**" if stats_dict['p'] < 0.01 else "*" if stats_dict['p'] < 0.05 else ""
            report.append(f"  {effect:45s}: F={stats_dict['F']:8.3f}, p={stats_dict['p']:.4e}, "
                         f"eta2={stats_dict['partial_eta_sq']:.4f} {sig}")
    else:
        report.append(f"\nError: {anova_results.get('error', 'Unknown')}")
    
    # Section 2: Mediation Analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 2: MEDIATION ANALYSIS (Intent as Mediator)")
    report.append("=" * 80)
    
    if 'error' not in mediation_results:
        report.append(f"\nN observations: {mediation_results.get('n_observations', 'N/A'):,}")
        report.append(f"Target intent: {mediation_results.get('target_intent', 'N/A')}")
        
        report.append("\n" + "-" * 70)
        report.append("Path Coefficients:")
        report.append("-" * 70)
        
        for path in ['path_c', 'path_a', 'path_b', 'path_c_prime']:
            if path in mediation_results:
                p_data = mediation_results[path]
                path_label = {
                    'path_c': 'c (Total: Age -> AnthroScore)',
                    'path_a': 'a (Age -> Intent)',
                    'path_b': 'b (Intent -> AnthroScore | Age)',
                    'path_c_prime': "c' (Direct: Age -> AnthroScore | Intent)"
                }.get(path, path)
                
                sig = "*" if p_data['p'] < 0.05 else ""
                report.append(f"  {path_label:45s}: B={p_data['coef']:+.4f} (p={p_data['p']:.4f}) {sig}")
        
        if 'indirect_effect' in mediation_results:
            ie = mediation_results['indirect_effect']
            sig = "*" if ie['sobel_p'] < 0.05 else ""
            report.append(f"\n  Indirect effect (a*b): {ie['coef']:.4f}")
            report.append(f"  Sobel test: z = {ie['sobel_z']:.3f}, p = {ie['sobel_p']:.4f} {sig}")
            report.append(f"  Proportion mediated: {mediation_results.get('proportion_mediated', 0):.1%}")
        
        report.append(f"\n  Conclusion: {mediation_results.get('mediation_type', 'N/A')}")
    else:
        report.append(f"\nError: {mediation_results.get('error', 'Unknown')}")
    
    # Section 3: Nonlinear Effects
    report.append("\n" + "=" * 80)
    report.append("SECTION 3: NONLINEAR EFFECTS ANALYSIS")
    report.append("=" * 80)
    
    if 'error' not in nonlinear_results:
        report.append(f"\nN observations: {nonlinear_results.get('n_observations', 'N/A'):,}")
        report.append(f"Age range: {nonlinear_results.get('age_range', ['N/A', 'N/A'])}")
        
        report.append("\n" + "-" * 70)
        report.append("Model Comparison:")
        report.append("-" * 70)
        
        if 'linear' in nonlinear_results:
            lin = nonlinear_results['linear']
            report.append(f"  Linear model:    R2 = {lin['r_squared']:.4f}, AIC = {lin['aic']:.2f}")
        
        if 'quadratic' in nonlinear_results:
            quad = nonlinear_results['quadratic']
            sig = "*" if quad['age_sq_p'] < 0.05 else ""
            report.append(f"  Quadratic model: R2 = {quad['r_squared']:.4f}, AIC = {quad['aic']:.2f}")
            report.append(f"  Quadratic term:  B = {quad['age_sq_coef']:.6f} (p = {quad['age_sq_p']:.4f}) {sig}")
        
        if 'model_comparison' in nonlinear_results:
            mc = nonlinear_results['model_comparison']
            report.append(f"\n  F-test (quad vs linear): F = {mc['f_statistic']:.3f}, p = {mc['p_value']:.4f}")
        
        if nonlinear_results.get('nonlinear_detected'):
            report.append("\n  Conclusion: NONLINEAR EFFECT DETECTED")
        else:
            report.append("\n  Conclusion: No significant nonlinear effect")
        
        if 'threshold_tests' in nonlinear_results:
            report.append("\n" + "-" * 70)
            report.append("Threshold Analysis:")
            report.append("-" * 70)
            for t in nonlinear_results['threshold_tests']:
                sig = "*" if t['interaction_p'] < 0.05 else ""
                report.append(f"  Age {t['threshold']:2d}: interaction p = {t['interaction_p']:.4f} {sig}")
    else:
        report.append(f"\nError: {nonlinear_results.get('error', 'Unknown')}")
    
    # Overall Summary
    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    
    key_findings = []
    
    if 'effects' in anova_results:
        sig_effects = [k for k, v in anova_results['effects'].items() if v['significant']]
        if sig_effects:
            key_findings.append(f"Significant ANOVA effects: {', '.join(sig_effects)}")
    
    if mediation_results.get('mediation_type') in ['FULL MEDIATION', 'PARTIAL MEDIATION']:
        key_findings.append(f"{mediation_results['mediation_type']} via {mediation_results.get('target_intent', 'intent')}")
    
    if nonlinear_results.get('nonlinear_detected'):
        key_findings.append("Nonlinear age effect detected")
    
    if key_findings:
        report.append("\nKey Findings:")
        for i, finding in enumerate(key_findings, 1):
            report.append(f"  {i}. {finding}")
    else:
        report.append("\nNo additional significant findings from these analyses.")
    
    return '\n'.join(report)


def main():
    """Run all missing analyses."""
    logger.info("=" * 70)
    logger.info("RUNNING MISSING STATISTICAL ANALYSES")
    logger.info("=" * 70)
    
    # Load data
    df = load_full_dataset()
    
    # Run analyses
    # 1. Three-way ANOVA
    anova_results = run_three_way_anova(df)
    
    # 2. Mediation analysis
    mediation_results = run_mediation_analysis(df)
    
    # 3. Nonlinear effects
    nonlinear_results = run_nonlinear_effects(df)
    
    # Generate report
    report = generate_analyses_report(anova_results, mediation_results, nonlinear_results)
    
    # Save report
    report_path = RESULTS_DIR / "missing_analyses_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\nReport saved to {report_path}")
    
    # Print report
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        safe_report = report.encode('ascii', 'replace').decode('ascii')
        print("\n" + safe_report)
    
    # Save detailed results as JSON
    import json
    
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
    
    results_dict = {
        'anova': convert_numpy(anova_results),
        'mediation': convert_numpy(mediation_results),
        'nonlinear': convert_numpy(nonlinear_results),
        'generated': datetime.now().isoformat()
    }
    
    json_path = RESULTS_DIR / "missing_analyses_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    logger.info(f"Detailed results saved to {json_path}")
    
    return anova_results, mediation_results, nonlinear_results


if __name__ == "__main__":
    results = main()

