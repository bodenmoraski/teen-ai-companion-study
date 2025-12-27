"""
Measurement Error Correction Module.

This module implements techniques to account for classification error
in demographic variables, including:
1. Reliability coefficient calculation
2. Attenuation bias correction
3. Measurement-error-corrected regression

References:
- Fuller, W.A. (1987). Measurement Error Models
- Carroll et al. (2006). Measurement Error in Nonlinear Models
"""
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_reliability(
    data_path: Path = None,
    method: str = 'accuracy'
) -> Dict[str, float]:
    """
    Calculate reliability coefficients for demographic classifications.
    
    Reliability is estimated as the proportion of correctly classified observations
    (accuracy) when compared against ground truth (self-declarations).
    
    For categorical variables, reliability = accuracy of classification.
    This is then used to estimate attenuation bias.
    
    Args:
        data_path: Path to demographics parquet file
        method: Method for calculating reliability ('accuracy' or 'kappa')
        
    Returns:
        Dict mapping variable name to reliability coefficient
    """
    if data_path is None:
        data_path = Path("Data/features/demographics.parquet")
    
    if not data_path.exists():
        logger.warning(f"Demographics file not found at {data_path}")
        # Return theoretical values based on known accuracy
        return {
            'age_5bucket': 0.376,  # 37.6% accuracy from report
            'age_3bucket': 0.463,  # 46.3% accuracy from report
            'gender': 0.40,  # Estimated
        }
    
    df = pd.read_parquet(data_path)
    
    reliability = {}
    
    # Age reliability (5-bucket)
    mask_age_5 = df['age_bucket_self_declared'].notna() & df['age_bucket_community'].notna()
    if mask_age_5.sum() > 50:
        acc_5 = (df.loc[mask_age_5, 'age_bucket_self_declared'] == 
                df.loc[mask_age_5, 'age_bucket_community']).mean()
        reliability['age_5bucket'] = acc_5
        logger.info(f"Age 5-bucket reliability: {acc_5:.3f}")
    else:
        reliability['age_5bucket'] = 0.376  # From report
    
    # Age reliability (3-bucket)
    # Convert to 3-bucket for comparison
    def to_3bucket(age):
        if pd.isna(age):
            return None
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        else:
            return 'adult'
    
    if mask_age_5.sum() > 50:
        gt_3 = df.loc[mask_age_5, 'age_bucket_self_declared'].apply(to_3bucket)
        pred_3 = df.loc[mask_age_5, 'age_bucket_community'].apply(to_3bucket)
        acc_3 = (gt_3 == pred_3).mean()
        reliability['age_3bucket'] = acc_3
        logger.info(f"Age 3-bucket reliability: {acc_3:.3f}")
    else:
        reliability['age_3bucket'] = 0.463  # From report
    
    # Gender reliability
    mask_gender = df['gender_self_declared'].notna() & df['gender_community'].notna()
    if 'gender_self_declared' in df.columns and 'gender_community' in df.columns:
        if mask_gender.sum() > 50:
            acc_gender = (df.loc[mask_gender, 'gender_self_declared'] == 
                         df.loc[mask_gender, 'gender_community']).mean()
            reliability['gender'] = acc_gender
            logger.info(f"Gender reliability: {acc_gender:.3f}")
        else:
            reliability['gender'] = 0.40  # Estimated
    else:
        reliability['gender'] = 0.40  # Estimated
    
    return reliability


def correct_for_attenuation(
    observed_coefficient: float,
    reliability_x: float,
    reliability_y: float = 1.0
) -> float:
    """
    Correct regression coefficient for attenuation bias due to measurement error.
    
    The observed correlation/coefficient is attenuated by measurement error:
    r_observed = r_true * sqrt(reliability_x * reliability_y)
    
    For regression coefficients with error only in X:
    beta_true = beta_observed / reliability_x
    
    Args:
        observed_coefficient: The observed (attenuated) coefficient
        reliability_x: Reliability of the independent variable (0 < r < 1)
        reliability_y: Reliability of the dependent variable (default 1.0)
        
    Returns:
        Corrected (disattenuated) coefficient
    """
    if reliability_x <= 0 or reliability_x > 1:
        raise ValueError(f"reliability_x must be in (0, 1], got {reliability_x}")
    if reliability_y <= 0 or reliability_y > 1:
        raise ValueError(f"reliability_y must be in (0, 1], got {reliability_y}")
    
    # For regression: beta_true = beta_observed / reliability_x
    # This assumes classical measurement error model
    corrected = observed_coefficient / reliability_x
    
    return corrected


def correct_r_squared_for_attenuation(
    observed_r_squared: float,
    reliability_x: float
) -> float:
    """
    Correct R² for attenuation due to measurement error.
    
    R²_true = R²_observed / reliability_x
    
    Args:
        observed_r_squared: The observed R²
        reliability_x: Reliability of the independent variable(s)
        
    Returns:
        Corrected R² (may exceed 1.0 if observed R² is close to reliability)
    """
    if reliability_x <= 0:
        raise ValueError("reliability_x must be positive")
    
    corrected = observed_r_squared / reliability_x
    
    # Cap at 1.0 for interpretability
    return min(corrected, 1.0)


def calculate_attenuation_factor(reliability: float) -> float:
    """
    Calculate the attenuation factor for a given reliability.
    
    The attenuation factor tells us by how much the true effect
    is reduced due to measurement error.
    
    Args:
        reliability: Reliability coefficient (0 < r ≤ 1)
        
    Returns:
        Attenuation factor (same as reliability for simple case)
    """
    return reliability


def run_measurement_error_analysis(
    model_results: Dict,
    reliability: Dict[str, float]
) -> Dict:
    """
    Run comprehensive measurement error analysis on regression results.
    
    Args:
        model_results: Dict containing model results with coefficients
        reliability: Dict of reliability coefficients for each variable
        
    Returns:
        Dict containing corrected estimates and uncertainty bounds
    """
    results = {
        'original': {},
        'corrected': {},
        'attenuation_factors': {},
        'interpretation': []
    }
    
    # Age reliability
    age_reliability = reliability.get('age_3bucket', 0.463)
    
    results['attenuation_factors']['age'] = age_reliability
    
    # If we have model coefficients, correct them
    if 'coefficients' in model_results:
        for coef_name, coef_value in model_results['coefficients'].items():
            results['original'][coef_name] = coef_value
            
            if 'age' in coef_name.lower():
                corrected = correct_for_attenuation(coef_value, age_reliability)
                results['corrected'][coef_name] = corrected
            elif 'gender' in coef_name.lower():
                gender_reliability = reliability.get('gender', 0.40)
                corrected = correct_for_attenuation(coef_value, gender_reliability)
                results['corrected'][coef_name] = corrected
            else:
                results['corrected'][coef_name] = coef_value
    
    # Add interpretation
    results['interpretation'].append(
        f"Age classification reliability: {age_reliability:.1%} "
        f"(attenuation factor = {age_reliability:.2f})"
    )
    results['interpretation'].append(
        f"True effects may be up to {1/age_reliability:.1f}x larger than observed"
    )
    results['interpretation'].append(
        "Null results may reflect measurement error masking true effects"
    )
    
    return results


def bootstrap_corrected_coefficients(
    df: pd.DataFrame,
    formula: str,
    reliability: Dict[str, float],
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for measurement-error-corrected coefficients.
    
    Args:
        df: DataFrame for regression
        formula: Regression formula
        reliability: Dict of reliability coefficients
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        DataFrame with corrected coefficients and CIs
    """
    from statsmodels.formula.api import ols
    
    n = len(df)
    corrected_samples = []
    
    age_reliability = reliability.get('age_3bucket', 0.463)
    gender_reliability = reliability.get('gender', 0.40)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample_df = df.iloc[sample_idx]
        
        try:
            model = ols(formula, data=sample_df).fit()
            
            # Correct coefficients
            corrected_params = {}
            for name, value in model.params.items():
                if 'age' in name.lower():
                    corrected_params[name] = correct_for_attenuation(value, age_reliability)
                elif 'gender' in name.lower():
                    corrected_params[name] = correct_for_attenuation(value, gender_reliability)
                else:
                    corrected_params[name] = value
            
            corrected_samples.append(corrected_params)
        except:
            continue
    
    if len(corrected_samples) < 100:
        logger.warning("Too few successful bootstrap iterations")
        return pd.DataFrame()
    
    corrected_df = pd.DataFrame(corrected_samples)
    
    results = pd.DataFrame({
        'corrected_mean': corrected_df.mean(),
        'corrected_std': corrected_df.std(),
        'ci_lower_95': corrected_df.quantile(0.025),
        'ci_upper_95': corrected_df.quantile(0.975),
    })
    
    return results


def generate_measurement_error_report(output_path: Path = None) -> str:
    """
    Generate a comprehensive measurement error analysis report.
    
    Returns:
        Report as string
    """
    reliability = calculate_reliability()
    
    lines = []
    lines.append("=" * 60)
    lines.append("MEASUREMENT ERROR ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append("1. RELIABILITY COEFFICIENTS")
    lines.append("-" * 40)
    for var, rel in reliability.items():
        lines.append(f"  {var}: {rel:.3f} ({rel:.1%} accuracy)")
    lines.append("")
    
    lines.append("2. ATTENUATION IMPLICATIONS")
    lines.append("-" * 40)
    for var, rel in reliability.items():
        correction_factor = 1 / rel if rel > 0 else float('inf')
        lines.append(f"  {var}:")
        lines.append(f"    - True effects may be {correction_factor:.1f}x larger")
        lines.append(f"    - {(1-rel)*100:.1f}% of signal lost to noise")
    lines.append("")
    
    lines.append("3. INTERPRETATION GUIDANCE")
    lines.append("-" * 40)
    lines.append("  - Null results (p > 0.05) may reflect:")
    lines.append("    a) True null effect (demographics don't matter)")
    lines.append("    b) Effect masked by measurement error")
    lines.append("  - With ~46% accuracy, substantial effects could be hidden")
    lines.append("  - Should interpret as 'too small to detect' not 'no effect'")
    lines.append("")
    
    lines.append("4. RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append("  - Report both naive and corrected coefficients")
    lines.append("  - Conduct power analysis with measurement error")
    lines.append("  - Frame conclusions tentatively")
    lines.append("  - Acknowledge measurement limitations prominently")
    
    report = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Measurement error report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate report
    report = generate_measurement_error_report()
    print(report)
    
    # Test attenuation correction
    print("\n" + "=" * 60)
    print("EXAMPLE: Correcting observed coefficient")
    print("=" * 60)
    
    observed = 0.01  # Observed coefficient
    reliability = 0.463  # 46.3% accuracy
    
    corrected = correct_for_attenuation(observed, reliability)
    print(f"Observed coefficient: {observed:.4f}")
    print(f"Reliability: {reliability:.3f}")
    print(f"Corrected coefficient: {corrected:.4f}")
    print(f"Correction factor: {corrected/observed:.2f}x")

