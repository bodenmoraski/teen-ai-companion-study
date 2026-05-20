"""
VALIDATE PAPER STATISTICS
===========================

This script validates that all statistics reported in the paper
match the actual data. It generates results for BOTH:

1. INCLUSIVE SAMPLE: All high-confidence users (N ≈ 16,347)
2. CONDITIONAL SAMPLE: Users with mean AnthroIndex > 1 (n ≈ 5,160)

Run this script to verify reproducibility of all paper claims.

Dependencies: pandas, numpy, scipy (no statsmodels required)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, pearsonr, spearmanr,
    chi2_contingency, levene, f_oneway
)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIDENCE_THRESHOLD = 0.60

PATHS = {
    "anthroscore_v3": BASE_DIR / "experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet",
    "all_comments": BASE_DIR / "Data/processed/all_comments.parquet",
    "user_emotions": BASE_DIR / "Data/features/user_emotions.parquet",
    "gender_predictions_v3": BASE_DIR / "experiments/v2_correction/gender_predictions_v3.parquet",
    "age_predictions_v3": BASE_DIR / "experiments/v2_correction/age_predictions_v3.parquet",
    "gender_predictions_v4": BASE_DIR / "experiments/v2_correction/gender_predictions_v4.parquet",
    "age_predictions_v4": BASE_DIR / "experiments/v2_correction/age_predictions_v4.parquet",
    "output": BASE_DIR / "results/PAPER_VALIDATION_REPORT.md",
    "output_json": BASE_DIR / "results/PAPER_VALIDATION_RESULTS.json",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cohens_d(group1, group2):
    """Calculate Cohen's d with Hedges' correction."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    # Hedges' g correction
    correction = 1 - (3 / (4*(n1+n2) - 9))
    return d * correction


def interpret_d(d):
    """Interpret Cohen's d."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def eta_squared_from_anova(groups):
    """Calculate eta-squared from ANOVA."""
    # One-way ANOVA
    f_stat, p_val = f_oneway(*groups)

    # Calculate SS
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
    ss_total = ss_between + ss_within

    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_val),
        'eta_squared': float(eta_sq),
        'ss_between': float(ss_between),
        'ss_total': float(ss_total),
    }


def two_way_anova_manual(df, dv_col, factor1_col, factor2_col):
    """Manual two-way ANOVA calculation."""
    grand_mean = df[dv_col].mean()
    ss_total = ((df[dv_col] - grand_mean) ** 2).sum()
    n_total = len(df)

    results = {}

    # Main effect of factor 1
    f1_means = df.groupby(factor1_col)[dv_col].mean()
    f1_counts = df.groupby(factor1_col)[dv_col].count()
    ss_f1 = sum(f1_counts[lvl] * (f1_means[lvl] - grand_mean)**2 for lvl in f1_means.index)

    # Main effect of factor 2
    f2_means = df.groupby(factor2_col)[dv_col].mean()
    f2_counts = df.groupby(factor2_col)[dv_col].count()
    ss_f2 = sum(f2_counts[lvl] * (f2_means[lvl] - grand_mean)**2 for lvl in f2_means.index)

    # Interaction
    cell_means = df.groupby([factor1_col, factor2_col])[dv_col].mean()
    cell_counts = df.groupby([factor1_col, factor2_col])[dv_col].count()

    ss_interaction = 0
    for (f1, f2), mean in cell_means.items():
        expected = grand_mean + (f1_means[f1] - grand_mean) + (f2_means[f2] - grand_mean)
        ss_interaction += cell_counts[(f1, f2)] * (mean - expected)**2

    # Residual
    ss_residual = ss_total - ss_f1 - ss_f2 - ss_interaction

    # Degrees of freedom
    df_f1 = len(f1_means) - 1
    df_f2 = len(f2_means) - 1
    df_interaction = df_f1 * df_f2
    df_residual = n_total - len(f1_means) * len(f2_means)

    # Mean squares and F
    ms_f1 = ss_f1 / df_f1 if df_f1 > 0 else 0
    ms_f2 = ss_f2 / df_f2 if df_f2 > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0

    f_f1 = ms_f1 / ms_residual if ms_residual > 0 else 0
    f_f2 = ms_f2 / ms_residual if ms_residual > 0 else 0
    f_interaction = ms_interaction / ms_residual if ms_residual > 0 else 0

    # P-values
    p_f1 = 1 - stats.f.cdf(f_f1, df_f1, df_residual) if df_residual > 0 else 1
    p_f2 = 1 - stats.f.cdf(f_f2, df_f2, df_residual) if df_residual > 0 else 1
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_residual) if df_residual > 0 else 1

    # R-squared
    r_squared = (ss_f1 + ss_f2 + ss_interaction) / ss_total if ss_total > 0 else 0

    return {
        'factor1': {
            'ss': float(ss_f1),
            'df': int(df_f1),
            'f': float(f_f1),
            'p': float(p_f1),
            'eta_squared': float(ss_f1 / ss_total) if ss_total > 0 else 0,
        },
        'factor2': {
            'ss': float(ss_f2),
            'df': int(df_f2),
            'f': float(f_f2),
            'p': float(p_f2),
            'eta_squared': float(ss_f2 / ss_total) if ss_total > 0 else 0,
        },
        'interaction': {
            'ss': float(ss_interaction),
            'df': int(df_interaction),
            'f': float(f_interaction),
            'p': float(p_interaction),
            'eta_squared': float(ss_interaction / ss_total) if ss_total > 0 else 0,
        },
        'r_squared': float(r_squared),
        'ss_total': float(ss_total),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load all required data."""
    print("Loading data...")
    data = {}

    # AnthroScore V3 (comment-level)
    if PATHS['anthroscore_v3'].exists():
        data['anthro_v3'] = pd.read_parquet(PATHS['anthroscore_v3'])
        print(f"  AnthroScore V3: {len(data['anthro_v3']):,} comments")
    else:
        raise FileNotFoundError(f"AnthroScore V3 not found: {PATHS['anthroscore_v3']}")

    # Comments (to get author info)
    if PATHS['all_comments'].exists():
        data['comments'] = pd.read_parquet(PATHS['all_comments'])
        print(f"  Comments: {len(data['comments']):,}")

    # Merge anthro_v3 with author info
    if 'comments' in data:
        data['anthro_v3'] = data['anthro_v3'].merge(
            data['comments'][['id', 'author', 'subreddit']].astype({'id': str}),
            left_on='comment_id', right_on='id', how='left'
        )

    # Demographics predictions - use V4 (matches paper statistics)
    # Note: The paper was generated using V4 demographics predictions
    for pred_type in ['gender', 'age']:
        v4_path = PATHS[f'{pred_type}_predictions_v4']
        v3_path = PATHS[f'{pred_type}_predictions_v3']

        if v4_path.exists():
            data[f'{pred_type}_pred'] = pd.read_parquet(v4_path)
            print(f"  {pred_type.title()} predictions (V4): {len(data[f'{pred_type}_pred']):,}")
        elif v3_path.exists():
            data[f'{pred_type}_pred'] = pd.read_parquet(v3_path)
            print(f"  {pred_type.title()} predictions (V3 fallback): {len(data[f'{pred_type}_pred']):,}")

    # User emotions
    if PATHS['user_emotions'].exists():
        data['emotions'] = pd.read_parquet(PATHS['user_emotions'])
        print(f"  Emotions: {len(data['emotions']):,} users")

    return data


def create_user_dataset(data):
    """Create user-level dataset with all features."""
    print("\nCreating user-level dataset...")

    # Aggregate V3 to user level (ALL scores, including 1)
    anthro_v3 = data['anthro_v3']
    valid_scores = anthro_v3[anthro_v3['score'] > 0]

    user_v3 = valid_scores.groupby('author').agg(
        anthro_v3_mean=('score', 'mean'),
        anthro_v3_max=('score', 'max'),
        anthro_v3_count=('score', 'count'),
    ).reset_index()

    print(f"  User-level AnthroScore: {len(user_v3):,} users")

    # Merge demographics
    df = user_v3.copy()

    if 'gender_pred' in data:
        gender_cols = ['author']
        if 'gender_predicted' in data['gender_pred'].columns:
            gender_cols.append('gender_predicted')
        if 'confidence' in data['gender_pred'].columns:
            gender_cols.append('confidence')
        df = df.merge(
            data['gender_pred'][gender_cols].rename(
                columns={'confidence': 'gender_conf'}
            ), on='author', how='left'
        )

    if 'age_pred' in data:
        age_cols = ['author']
        if 'age_predicted' in data['age_pred'].columns:
            age_cols.append('age_predicted')
        if 'confidence' in data['age_pred'].columns:
            age_cols.append('confidence')
        df = df.merge(
            data['age_pred'][age_cols].rename(
                columns={'confidence': 'age_conf'}
            ), on='author', how='left'
        )

    if 'emotions' in data:
        df = df.merge(data['emotions'], on='author', how='left')

    print(f"  Full dataset: {len(df):,} users")

    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_age_effect(df, sample_name):
    """Analyze age effect on anthropomorphization."""
    teens = df[df['age_predicted'] == 'teen']['anthro_v3_mean'].dropna().values
    adults = df[df['age_predicted'] == 'adult']['anthro_v3_mean'].dropna().values

    results = {
        'sample': sample_name,
        'teen_n': len(teens),
        'adult_n': len(adults),
        'teen_mean': float(np.mean(teens)) if len(teens) > 0 else None,
        'teen_sd': float(np.std(teens, ddof=1)) if len(teens) > 1 else None,
        'adult_mean': float(np.mean(adults)) if len(adults) > 0 else None,
        'adult_sd': float(np.std(adults, ddof=1)) if len(adults) > 1 else None,
    }

    if len(teens) > 1 and len(adults) > 1:
        t_stat, t_p = ttest_ind(teens, adults, equal_var=False)
        u_stat, u_p = mannwhitneyu(teens, adults, alternative='two-sided')
        d = cohens_d(teens, adults)

        results['welch_t'] = float(t_stat)
        results['welch_p'] = float(t_p)
        results['mann_whitney_u'] = float(u_stat)
        results['mann_whitney_p'] = float(u_p)
        results['cohens_d'] = float(d)
        results['d_interpretation'] = interpret_d(d)
        results['direction'] = 'adults higher' if np.mean(adults) > np.mean(teens) else 'teens higher'

    return results


def analyze_gender_effect(df, sample_name):
    """Analyze gender effect on anthropomorphization."""
    males = df[df['gender_predicted'] == 'male']['anthro_v3_mean'].dropna().values
    females = df[df['gender_predicted'] == 'female']['anthro_v3_mean'].dropna().values

    results = {
        'sample': sample_name,
        'male_n': len(males),
        'female_n': len(females),
        'male_mean': float(np.mean(males)) if len(males) > 0 else None,
        'male_sd': float(np.std(males, ddof=1)) if len(males) > 1 else None,
        'female_mean': float(np.mean(females)) if len(females) > 0 else None,
        'female_sd': float(np.std(females, ddof=1)) if len(females) > 1 else None,
    }

    if len(males) > 1 and len(females) > 1:
        t_stat, t_p = ttest_ind(males, females, equal_var=False)
        d = cohens_d(males, females)

        results['welch_t'] = float(t_stat)
        results['welch_p'] = float(t_p)
        results['cohens_d'] = float(d)
        results['d_interpretation'] = interpret_d(d)
        results['direction'] = 'females higher' if np.mean(females) > np.mean(males) else 'males higher'

    return results


def analyze_anova(df, sample_name):
    """Run two-way ANOVA."""
    df_clean = df[['anthro_v3_mean', 'age_predicted', 'gender_predicted']].dropna()

    if len(df_clean) < 10:
        return {'sample': sample_name, 'error': 'Insufficient data'}

    anova_results = two_way_anova_manual(
        df_clean,
        'anthro_v3_mean',
        'age_predicted',
        'gender_predicted'
    )

    return {
        'sample': sample_name,
        'n': len(df_clean),
        'r_squared': anova_results['r_squared'],
        'age_effect': anova_results['factor1'],
        'gender_effect': anova_results['factor2'],
        'interaction': anova_results['interaction'],
    }


def analyze_subgroup_means(df, sample_name):
    """Calculate subgroup means."""
    results = {'sample': sample_name, 'subgroups': {}}

    for age in ['teen', 'adult']:
        for gender in ['male', 'female']:
            mask = (df['age_predicted'] == age) & (df['gender_predicted'] == gender)
            subset = df[mask]['anthro_v3_mean'].dropna()
            results['subgroups'][f'{age}_{gender}'] = {
                'n': int(len(subset)),
                'mean': float(subset.mean()) if len(subset) > 0 else None,
                'sd': float(subset.std()) if len(subset) > 1 else None,
            }

    return results


def analyze_binary_threshold(df, sample_name, threshold=3):
    """Analyze binary outcome: ever produced comment scoring >= threshold."""
    results = {'sample': sample_name, 'threshold': threshold}

    df_copy = df.copy()
    df_copy['high_anthro'] = df_copy['anthro_v3_max'] >= threshold

    # Overall rate
    results['overall_rate'] = float(df_copy['high_anthro'].mean())
    results['overall_n'] = int(df_copy['high_anthro'].sum())

    # By age
    teen_mask = df_copy['age_predicted'] == 'teen'
    adult_mask = df_copy['age_predicted'] == 'adult'

    results['teen_rate'] = float(df_copy[teen_mask]['high_anthro'].mean()) if teen_mask.sum() > 0 else None
    results['adult_rate'] = float(df_copy[adult_mask]['high_anthro'].mean()) if adult_mask.sum() > 0 else None

    # Chi-square and OR for age
    ct_age = pd.crosstab(df_copy['age_predicted'], df_copy['high_anthro'])
    if ct_age.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(ct_age)
        results['age_chi2'] = float(chi2)
        results['age_p'] = float(p)

        if 'teen' in ct_age.index and 'adult' in ct_age.index:
            teen_no, teen_yes = ct_age.loc['teen', False], ct_age.loc['teen', True]
            adult_no, adult_yes = ct_age.loc['adult', False], ct_age.loc['adult', True]
            if teen_yes > 0 and teen_no > 0 and adult_no > 0:
                or_age = (adult_yes / adult_no) / (teen_yes / teen_no)
                results['age_odds_ratio'] = float(or_age)

    # By gender
    male_mask = df_copy['gender_predicted'] == 'male'
    female_mask = df_copy['gender_predicted'] == 'female'

    results['male_rate'] = float(df_copy[male_mask]['high_anthro'].mean()) if male_mask.sum() > 0 else None
    results['female_rate'] = float(df_copy[female_mask]['high_anthro'].mean()) if female_mask.sum() > 0 else None

    ct_gender = pd.crosstab(df_copy['gender_predicted'], df_copy['high_anthro'])
    if ct_gender.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(ct_gender)
        results['gender_chi2'] = float(chi2)
        results['gender_p'] = float(p)

        if 'male' in ct_gender.index and 'female' in ct_gender.index:
            male_no, male_yes = ct_gender.loc['male', False], ct_gender.loc['male', True]
            female_no, female_yes = ct_gender.loc['female', False], ct_gender.loc['female', True]
            if male_yes > 0 and male_no > 0 and female_no > 0:
                or_gender = (female_yes / female_no) / (male_yes / male_no)
                results['gender_odds_ratio'] = float(or_gender)

    return results


def analyze_emotions(df, sample_name):
    """Analyze emotion correlations."""
    emotion_cols = [col for col in df.columns if 'emotion' in col.lower()]

    if not emotion_cols:
        return {'sample': sample_name, 'error': 'No emotion columns found'}

    results = {'sample': sample_name, 'correlations': {}}

    for col in emotion_cols:
        try:
            valid = df[['anthro_v3_mean', col]].dropna()
            # Ensure numeric dtype
            valid = valid.copy()
            valid['anthro_v3_mean'] = pd.to_numeric(valid['anthro_v3_mean'], errors='coerce')
            valid[col] = pd.to_numeric(valid[col], errors='coerce')
            valid = valid.dropna()

            if len(valid) > 30:
                x = valid['anthro_v3_mean'].values.astype(float)
                y = valid[col].values.astype(float)
                r, p = pearsonr(x, y)
                rho, rho_p = spearmanr(x, y)
                results['correlations'][col] = {
                    'pearson_r': float(r),
                    'pearson_p': float(p),
                    'spearman_rho': float(rho),
                    'spearman_p': float(rho_p),
                }
        except Exception as e:
            results['correlations'][col] = {'error': str(e)}

    return results


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation():
    """Run complete validation."""
    print("=" * 60)
    print("PAPER STATISTICS VALIDATION")
    print("=" * 60)

    # Load data
    data = load_data()
    df = create_user_dataset(data)

    # Check columns
    print(f"\nAvailable columns: {list(df.columns)[:10]}...")

    # Define samples
    has_conf = 'gender_conf' in df.columns and 'age_conf' in df.columns

    if has_conf:
        # INCLUSIVE: All high-confidence users
        df_inclusive = df[
            (df['gender_conf'] >= CONFIDENCE_THRESHOLD) &
            (df['age_conf'] >= CONFIDENCE_THRESHOLD)
        ].copy()
    else:
        # Without confidence, use all users with predictions
        df_inclusive = df[
            df['gender_predicted'].notna() &
            df['age_predicted'].notna()
        ].copy()

    # CONDITIONAL: Users with mean AnthroIndex > 1
    df_conditional = df_inclusive[df_inclusive['anthro_v3_mean'] > 1].copy()

    print(f"\n=== SAMPLE SIZES ===")
    print(f"  INCLUSIVE (all high-conf): N = {len(df_inclusive):,}")
    print(f"  CONDITIONAL (anthro > 1):  n = {len(df_conditional):,}")

    results = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'inclusive_n': len(df_inclusive),
            'conditional_n': len(df_conditional),
        }
    }

    # Run analyses on BOTH samples
    print("\n=== RUNNING ANALYSES ===")

    for sample_name, df_sample in [('inclusive', df_inclusive), ('conditional', df_conditional)]:
        print(f"\n--- {sample_name.upper()} SAMPLE (N={len(df_sample):,}) ---")

        results[sample_name] = {}

        # Age effect
        age_results = analyze_age_effect(df_sample, sample_name)
        results[sample_name]['age_effect'] = age_results
        if age_results.get('teen_mean') and age_results.get('adult_mean'):
            print(f"  Age: Teens M={age_results['teen_mean']:.3f}, Adults M={age_results['adult_mean']:.3f}, d={age_results.get('cohens_d', 'N/A'):.3f}")

        # Gender effect
        gender_results = analyze_gender_effect(df_sample, sample_name)
        results[sample_name]['gender_effect'] = gender_results
        if gender_results.get('male_mean') and gender_results.get('female_mean'):
            print(f"  Gender: Males M={gender_results['male_mean']:.3f}, Females M={gender_results['female_mean']:.3f}, d={gender_results.get('cohens_d', 'N/A'):.3f}")

        # ANOVA
        anova_results = analyze_anova(df_sample, sample_name)
        results[sample_name]['anova'] = anova_results
        if 'r_squared' in anova_results:
            print(f"  ANOVA R² = {anova_results['r_squared']:.4f}")
            if 'age_effect' in anova_results:
                print(f"    Age η² = {anova_results['age_effect']['eta_squared']:.4f}")
            if 'gender_effect' in anova_results:
                print(f"    Gender η² = {anova_results['gender_effect']['eta_squared']:.4f}")

        # Subgroups
        results[sample_name]['subgroups'] = analyze_subgroup_means(df_sample, sample_name)

        # Binary threshold (only for inclusive)
        if sample_name == 'inclusive':
            results[sample_name]['binary_threshold'] = analyze_binary_threshold(df_sample, sample_name)
            bt = results[sample_name]['binary_threshold']
            if bt.get('adult_rate') and bt.get('teen_rate'):
                print(f"  Binary (>= 3): Adults {bt['adult_rate']:.1%} vs Teens {bt['teen_rate']:.1%}")
                if 'age_odds_ratio' in bt:
                    print(f"    Age OR = {bt['age_odds_ratio']:.2f}")

        # Emotions (only for conditional to match paper)
        if sample_name == 'conditional':
            results[sample_name]['emotions'] = analyze_emotions(df_sample, sample_name)

    # Save results
    print("\n=== SAVING RESULTS ===")

    # JSON
    PATHS['output_json'].parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS['output_json'], 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON: {PATHS['output_json']}")

    # Markdown report
    generate_report(results, PATHS['output'])
    print(f"  Report: {PATHS['output']}")

    return results


def generate_report(results, output_path):
    """Generate markdown validation report."""
    md = f"""# Paper Statistics Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Confidence Threshold:** {CONFIDENCE_THRESHOLD}

---

## Sample Sizes

| Sample | Description | N |
|--------|-------------|---|
| Inclusive | All high-confidence users | {results['metadata']['inclusive_n']:,} |
| Conditional | Users with mean AnthroIndex > 1 | {results['metadata']['conditional_n']:,} |

---

## Key Statistics Comparison

### Age Effect

| Metric | Inclusive | Conditional | Paper Claims |
|--------|-----------|-------------|--------------|
"""

    inc_age = results['inclusive']['age_effect']
    cond_age = results['conditional']['age_effect']

    def fmt(v, decimals=3):
        if v is None:
            return 'N/A'
        return f"{v:.{decimals}f}"

    md += f"| Teen N | {inc_age['teen_n']:,} | {cond_age['teen_n']:,} | - |\n"
    md += f"| Adult N | {inc_age['adult_n']:,} | {cond_age['adult_n']:,} | - |\n"
    md += f"| Teen Mean | {fmt(inc_age.get('teen_mean'))} | {fmt(cond_age.get('teen_mean'))} | 1.108 (inc) / 1.383 (cond) |\n"
    md += f"| Adult Mean | {fmt(inc_age.get('adult_mean'))} | {fmt(cond_age.get('adult_mean'))} | 1.279 (inc) / 1.602 (cond) |\n"
    md += f"| Teen SD | {fmt(inc_age.get('teen_sd'))} | {fmt(cond_age.get('teen_sd'))} | 0.290 (inc) / 0.439 (cond) |\n"
    md += f"| Adult SD | {fmt(inc_age.get('adult_sd'))} | {fmt(cond_age.get('adult_sd'))} | 0.493 (inc) / 0.575 (cond) |\n"
    md += f"| Welch's t | {fmt(inc_age.get('welch_t'), 2)} | {fmt(cond_age.get('welch_t'), 2)} | -18.42 (inc) / -12.96 (cond) |\n"
    md += f"| Cohen's d | {fmt(inc_age.get('cohens_d'))} | {fmt(cond_age.get('cohens_d'))} | -0.51 (inc) / -0.46 (cond) |\n"
    md += f"| Direction | {inc_age.get('direction', 'N/A')} | {cond_age.get('direction', 'N/A')} | adults higher |\n"

    md += """
### Gender Effect

| Metric | Inclusive | Conditional | Paper Claims |
|--------|-----------|-------------|--------------|
"""

    inc_gender = results['inclusive']['gender_effect']
    cond_gender = results['conditional']['gender_effect']

    md += f"| Male Mean | {fmt(inc_gender.get('male_mean'))} | {fmt(cond_gender.get('male_mean'))} | 1.121 (inc) / 1.433 (cond) |\n"
    md += f"| Female Mean | {fmt(inc_gender.get('female_mean'))} | {fmt(cond_gender.get('female_mean'))} | 1.225 (inc) / 1.466 (cond) |\n"
    md += f"| Cohen's d | {fmt(inc_gender.get('cohens_d'))} | {fmt(cond_gender.get('cohens_d'))} | -0.31 (inc) / -0.07 (cond) |\n"

    md += """
### Two-Way ANOVA

| Effect | Inclusive η² | Conditional η² | Paper Claims |
|--------|--------------|----------------|--------------|
"""

    inc_anova = results['inclusive'].get('anova', {})
    cond_anova = results['conditional'].get('anova', {})

    for effect_name, effect_key, paper_val in [
        ('Age', 'age_effect', '.029 (inc)'),
        ('Gender', 'gender_effect', '.008 (inc)'),
        ('Interaction', 'interaction', 'n.s.')
    ]:
        inc_eta = inc_anova.get(effect_key, {}).get('eta_squared')
        cond_eta = cond_anova.get(effect_key, {}).get('eta_squared')
        md += f"| {effect_name} | {fmt(inc_eta, 4)} | {fmt(cond_eta, 4)} | {paper_val} |\n"

    md += f"""
### Model R²

| Sample | R² | Paper Claims |
|--------|-----|--------------|
| Inclusive | {fmt(inc_anova.get('r_squared'), 4)} | .041 |
| Conditional | {fmt(cond_anova.get('r_squared'), 4)} | ~4% |

"""

    if 'binary_threshold' in results['inclusive']:
        bt = results['inclusive']['binary_threshold']
        md += f"""### Binary Threshold Analysis (Inclusive, ≥3)

| Metric | Computed | Paper Claims |
|--------|----------|--------------|
| Overall Rate | {bt.get('overall_rate', 0):.1%} | 7.2% |
| Teen Rate | {fmt(bt.get('teen_rate'))} ({bt.get('teen_rate', 0)*100:.1f}%) | 6.2% |
| Adult Rate | {fmt(bt.get('adult_rate'))} ({bt.get('adult_rate', 0)*100:.1f}%) | 11.7% |
| Age OR | {fmt(bt.get('age_odds_ratio'), 2)} | 1.88 |
| Female Rate | {fmt(bt.get('female_rate'))} ({bt.get('female_rate', 0)*100:.1f}%) | 12.3% |
| Male Rate | {fmt(bt.get('male_rate'))} ({bt.get('male_rate', 0)*100:.1f}%) | 6.1% |
| Gender OR | {fmt(bt.get('gender_odds_ratio'), 2)} | 2.05 |

"""

    md += """### Subgroup Means (Conditional)

| Subgroup | N | Mean | SD |
|----------|---|------|-----|
"""

    for key, vals in results['conditional']['subgroups']['subgroups'].items():
        md += f"| {key.replace('_', ' ').title()} | {vals['n']:,} | {fmt(vals.get('mean'))} | {fmt(vals.get('sd'))} |\n"

    md += """
---

## Emotion Correlations (Conditional Sample)

| Emotion | Pearson r | p-value | Paper Claims |
|---------|-----------|---------|--------------|
"""

    if 'emotions' in results['conditional'] and 'correlations' in results['conditional']['emotions']:
        paper_claims = {
            'emotion_joy': '+0.100',
            'Emotion_joy': '+0.100',
            'emotion_neutral': '-0.129',
            'Emotion_neutral': '-0.129',
        }
        for emotion, vals in sorted(results['conditional']['emotions']['correlations'].items(),
                                     key=lambda x: -abs(x[1]['pearson_r'])):
            paper = paper_claims.get(emotion, '-')
            md += f"| {emotion} | {vals['pearson_r']:+.3f} | {vals['pearson_p']:.4f} | {paper} |\n"

    md += """
---

## Validation Summary

Review the computed values against paper claims:
- **Match (±5%)**: Values are consistent
- **Minor difference**: May be due to rounding or exact sample definition
- **Major difference**: Requires investigation

---

*Generated by validate_paper_statistics.py*
"""

    with open(output_path, 'w') as f:
        f.write(md)


if __name__ == "__main__":
    run_validation()
