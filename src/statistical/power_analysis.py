"""
Power Analysis Module with Measurement Error.

This module implements statistical power analysis accounting for
measurement error in predictor variables.

Key functions:
1. calculate_power: Power for detecting effects given measurement error
2. minimum_detectable_effect: MDE at 80% power
3. power_curve: Power across range of effect sizes
"""
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_power(
    n: int,
    effect_size: float,
    reliability: float = 1.0,
    alpha: float = 0.05,
    n_predictors: int = 5
) -> float:
    """
    Calculate statistical power to detect an effect given measurement error.
    
    Power is reduced by measurement error because:
    1. True effect is attenuated: observed_effect = true_effect * reliability
    2. This reduces the signal-to-noise ratio
    
    Args:
        n: Sample size
        effect_size: Cohen's f² (true effect size, before attenuation)
        reliability: Reliability of predictor measurement (0 < r ≤ 1)
        alpha: Significance level
        n_predictors: Number of predictors in model
        
    Returns:
        Statistical power (0 to 1)
    """
    if n <= n_predictors + 1:
        return 0.0
    
    if reliability <= 0 or reliability > 1:
        raise ValueError(f"Reliability must be in (0, 1], got {reliability}")
    
    # Attenuated effect size
    # f²_observed = f²_true * reliability²
    # (effect size is squared, so reliability is also squared)
    attenuated_effect = effect_size * (reliability ** 2)
    
    # Degrees of freedom
    df1 = n_predictors  # numerator df (effect)
    df2 = n - n_predictors - 1  # denominator df (error)
    
    if df2 <= 0:
        return 0.0
    
    # Non-centrality parameter
    # λ = f² * (df2 + df1 + 1) ≈ f² * n for large n
    ncp = attenuated_effect * (n)
    
    # Critical F value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # Power = P(F > f_crit | λ)
    # Use non-central F distribution
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
    
    return power


def minimum_detectable_effect(
    n: int,
    power: float = 0.80,
    reliability: float = 1.0,
    alpha: float = 0.05,
    n_predictors: int = 5
) -> float:
    """
    Calculate minimum detectable effect size (MDE) given measurement error.
    
    The MDE is the smallest true effect size that can be detected with
    the specified power, accounting for attenuation due to measurement error.
    
    Args:
        n: Sample size
        power: Desired power (default 0.80)
        reliability: Reliability of predictor measurement
        alpha: Significance level
        n_predictors: Number of predictors
        
    Returns:
        Minimum detectable true effect size (Cohen's f²)
    """
    # Binary search for MDE
    low, high = 0.0001, 1.0
    
    for _ in range(50):  # Max iterations
        mid = (low + high) / 2
        calc_power = calculate_power(n, mid, reliability, alpha, n_predictors)
        
        if calc_power < power:
            low = mid
        else:
            high = mid
        
        if abs(calc_power - power) < 0.001:
            break
    
    return mid


def power_curve(
    n: int,
    effect_sizes: np.ndarray = None,
    reliability: float = 1.0,
    alpha: float = 0.05,
    n_predictors: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate power curve across range of effect sizes.
    
    Args:
        n: Sample size
        effect_sizes: Array of effect sizes to evaluate
        reliability: Reliability of predictor measurement
        alpha: Significance level
        n_predictors: Number of predictors
        
    Returns:
        Tuple of (effect_sizes, powers)
    """
    if effect_sizes is None:
        effect_sizes = np.logspace(-4, -1, 50)  # f² from 0.0001 to 0.1
    
    powers = np.array([
        calculate_power(n, es, reliability, alpha, n_predictors)
        for es in effect_sizes
    ])
    
    return effect_sizes, powers


def compare_power_with_without_error(
    n: int,
    effect_size: float,
    reliability: float,
    alpha: float = 0.05,
    n_predictors: int = 5
) -> Dict[str, float]:
    """
    Compare power with and without measurement error.
    
    Args:
        n: Sample size
        effect_size: True effect size (Cohen's f²)
        reliability: Reliability of predictor measurement
        alpha: Significance level
        n_predictors: Number of predictors
        
    Returns:
        Dict with power comparison
    """
    power_no_error = calculate_power(n, effect_size, 1.0, alpha, n_predictors)
    power_with_error = calculate_power(n, effect_size, reliability, alpha, n_predictors)
    
    return {
        'effect_size': effect_size,
        'reliability': reliability,
        'power_no_error': power_no_error,
        'power_with_error': power_with_error,
        'power_reduction': power_no_error - power_with_error,
        'power_ratio': power_with_error / power_no_error if power_no_error > 0 else 0
    }


def effect_size_benchmarks() -> Dict[str, float]:
    """
    Return Cohen's f² benchmarks for effect sizes.
    
    Returns:
        Dict with small, medium, large effect sizes
    """
    return {
        'small': 0.02,   # f² = 0.02
        'medium': 0.15,  # f² = 0.15
        'large': 0.35,   # f² = 0.35
    }


def interpret_effect_size(f_squared: float) -> str:
    """
    Interpret an effect size in terms of Cohen's benchmarks.
    
    Args:
        f_squared: Cohen's f² effect size
        
    Returns:
        Interpretation string
    """
    if f_squared < 0.02:
        return "negligible"
    elif f_squared < 0.15:
        return "small"
    elif f_squared < 0.35:
        return "medium"
    else:
        return "large"


def generate_power_analysis_report(
    n: int = 27000,
    reliability: float = 0.463,
    output_path: Path = None
) -> str:
    """
    Generate comprehensive power analysis report.
    
    Args:
        n: Sample size
        reliability: Reliability of demographic classification
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    benchmarks = effect_size_benchmarks()
    
    lines = []
    lines.append("=" * 70)
    lines.append("POWER ANALYSIS REPORT")
    lines.append("Accounting for Measurement Error in Demographics")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("PARAMETERS")
    lines.append("-" * 40)
    lines.append(f"  Sample size: {n:,}")
    lines.append(f"  Classification reliability: {reliability:.1%}")
    lines.append(f"  Alpha: 0.05 (two-tailed)")
    lines.append(f"  Number of predictors: 5")
    lines.append("")
    
    lines.append("EFFECT SIZE BENCHMARKS (Cohen's f²)")
    lines.append("-" * 40)
    for name, f2 in benchmarks.items():
        lines.append(f"  {name.capitalize()}: f² = {f2}")
    lines.append("")
    
    lines.append("POWER TO DETECT EFFECTS")
    lines.append("-" * 40)
    lines.append(f"{'Effect Size':<15} {'No Error':<15} {'With Error':<15} {'Reduction':<15}")
    lines.append("-" * 60)
    
    for name, f2 in benchmarks.items():
        comparison = compare_power_with_without_error(n, f2, reliability)
        lines.append(
            f"{name.capitalize():<15} "
            f"{comparison['power_no_error']:.1%}{'':<8} "
            f"{comparison['power_with_error']:.1%}{'':<8} "
            f"{comparison['power_reduction']:+.1%}"
        )
    lines.append("")
    
    # Minimum detectable effect
    lines.append("MINIMUM DETECTABLE EFFECT (MDE) at 80% Power")
    lines.append("-" * 40)
    
    mde_no_error = minimum_detectable_effect(n, 0.80, 1.0)
    mde_with_error = minimum_detectable_effect(n, 0.80, reliability)
    
    lines.append(f"  Without measurement error: f² = {mde_no_error:.4f} ({interpret_effect_size(mde_no_error)})")
    lines.append(f"  With measurement error:    f² = {mde_with_error:.4f} ({interpret_effect_size(mde_with_error)})")
    lines.append(f"  MDE inflation:             {mde_with_error/mde_no_error:.1f}x")
    lines.append("")
    
    # Current study context
    lines.append("STUDY CONTEXT")
    lines.append("-" * 40)
    observed_f2 = 0.0007  # From report
    lines.append(f"  Observed effect size (f²): {observed_f2:.4f}")
    lines.append(f"  Interpretation: {interpret_effect_size(observed_f2)}")
    lines.append("")
    
    power_observed = calculate_power(n, observed_f2, reliability)
    lines.append(f"  Power to detect observed effect: {power_observed:.1%}")
    lines.append("")
    
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    lines.append("  The observed effect (f² ≈ 0.0007) is far below even 'small' effects (f² = 0.02).")
    lines.append("  However, measurement error (46.3% accuracy) reduces power substantially.")
    lines.append("")
    lines.append("  Key conclusions:")
    lines.append(f"  1. With {n:,} observations and {reliability:.1%} reliability,")
    lines.append(f"     we can only detect true effects ≥ f² = {mde_with_error:.4f} at 80% power")
    lines.append("  2. A 'small' effect (f² = 0.02) would be detected with >99% power even with error")
    lines.append("  3. The null result likely reflects genuinely negligible effects,")
    lines.append("     not effects masked by measurement error")
    lines.append("")
    lines.append("  Recommendation: Frame as 'effects too small to matter' rather than")
    lines.append("  'no effects exist' - we cannot rule out very tiny effects")
    
    report = '\n'.join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Power analysis report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate report
    report = generate_power_analysis_report()
    print(report)
    
    # Example calculations
    print("\n" + "=" * 70)
    print("ADDITIONAL EXAMPLES")
    print("=" * 70)
    
    n = 27000
    reliability = 0.463
    
    # Power for different effect sizes
    print("\nPower for different true effect sizes:")
    for f2 in [0.001, 0.005, 0.01, 0.02, 0.05]:
        power = calculate_power(n, f2, reliability)
        print(f"  f² = {f2:.3f}: power = {power:.1%}")

