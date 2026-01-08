#!/usr/bin/env python3
"""
RQ3: Emotional Expression & Anthropomorphization Analysis
=========================================================

Comprehensive analysis of how emotional expression patterns relate to 
anthropomorphization of AI companions.

Analyses:
- Option A: Emotional intensity correlations with AnthroScore
- Option B: Emotional profile differences (high vs low anthropomorphizers)
- Option C: Age-moderated emotional relationships
- Option D: Emotional valence and AI companion attachment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("Data/features")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_and_merge_data():
    """Load and merge all required datasets."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load user emotions
    emotions = pd.read_parquet(DATA_DIR / "user_emotions.parquet")
    print(f"User emotions: {len(emotions):,} users")
    
    # Load anthroscores
    anthro = pd.read_parquet(DATA_DIR / "user_anthroscores.parquet")
    print(f"AnthroScores: {len(anthro):,} users")
    
    # Load demographics (with age/gender predictions)
    demo = pd.read_parquet(DATA_DIR / "demographics_v2.parquet")
    print(f"Demographics: {len(demo):,} users")
    
    # Merge all
    df = emotions.merge(anthro, on='author', how='inner')
    df = df.merge(demo[['author', 'age_bucket', 'gender', 'confidence']], 
                  on='author', how='left')
    
    # Rename for consistency
    df = df.rename(columns={
        'age_bucket': 'age_group', 
        'confidence': 'age_confidence',
        'anthroscore_mean': 'mean_anthroscore',
        'anthroscore_max': 'max_anthroscore',
        'anthroscore_std': 'std_anthroscore'
    })
    
    print(f"\nMerged dataset: {len(df):,} users")
    print(f"  With age predictions: {df['age_group'].notna().sum():,}")
    print(f"  With gender predictions: {df['gender'].notna().sum():,}")
    
    return df


def calculate_derived_metrics(df):
    """Calculate derived emotional metrics."""
    # Emotional intensity (sum of all non-neutral emotions)
    df['emotional_intensity'] = (
        df['emotion_joy'] + df['emotion_sadness'] + df['emotion_anger'] + 
        df['emotion_fear'] + df['emotion_surprise'] + df['emotion_disgust']
    )
    
    # Emotional valence (positive - negative)
    df['emotional_valence'] = df['emotion_joy'] - (
        df['emotion_sadness'] + df['emotion_anger'] + 
        df['emotion_fear'] + df['emotion_disgust']
    )
    
    # Emotional diversity (entropy-like measure)
    emotions = ['emotion_joy', 'emotion_sadness', 'emotion_anger', 
                'emotion_fear', 'emotion_surprise', 'emotion_disgust']
    emotion_matrix = df[emotions].values
    # Normalize to probabilities
    emotion_sums = emotion_matrix.sum(axis=1, keepdims=True)
    emotion_probs = np.divide(emotion_matrix, emotion_sums, 
                              where=emotion_sums > 0, 
                              out=np.zeros_like(emotion_matrix))
    # Calculate entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.nansum(emotion_probs * np.log2(emotion_probs + 1e-10), axis=1)
    df['emotional_diversity'] = entropy
    
    # High/Low anthropomorphizer classification (median split on non-zero users)
    # Many users have 0, so we use median split on the full data with duplicates='drop'
    try:
        df['anthro_group'] = pd.qcut(df['mean_anthroscore'], q=2, labels=['Low', 'High'], duplicates='drop')
    except ValueError:
        # If still fails, use median cutoff manually
        median = df['mean_anthroscore'].median()
        df['anthro_group'] = df['mean_anthroscore'].apply(lambda x: 'High' if x > median else 'Low')
    
    # Tertile split - handle duplicates
    try:
        df['anthro_tertile'] = pd.qcut(df['mean_anthroscore'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    except ValueError:
        # Use quantile cutoffs manually
        q33 = df['mean_anthroscore'].quantile(0.33)
        q66 = df['mean_anthroscore'].quantile(0.66)
        df['anthro_tertile'] = df['mean_anthroscore'].apply(
            lambda x: 'Low' if x <= q33 else ('Medium' if x <= q66 else 'High')
        )
    
    return df


def option_a_correlations(df):
    """
    Option A: Emotional Expression & Anthropomorphization Correlations
    """
    print("\n" + "=" * 70)
    print("OPTION A: EMOTIONAL INTENSITY CORRELATIONS")
    print("=" * 70)
    
    results = {}
    
    # Variables to correlate with AnthroScore
    emotion_vars = [
        'emotion_joy', 'emotion_sadness', 'emotion_anger', 'emotion_fear',
        'emotion_surprise', 'emotion_disgust', 'emotion_neutral',
        'emotional_intensity', 'emotional_valence', 'emotional_diversity'
    ]
    
    # Filter to non-zero anthro users for meaningful analysis
    df_nonzero = df[df['mean_anthroscore'] > 0].copy()
    print(f"\nAnalyzing {len(df_nonzero):,} users with non-zero AnthroScore")
    
    print("\n" + "-" * 50)
    print("Pearson Correlations with Mean AnthroScore:")
    print("-" * 50)
    
    correlations = []
    for var in emotion_vars:
        r, p = pearsonr(df_nonzero['mean_anthroscore'], df_nonzero[var])
        rho, p_spearman = spearmanr(df_nonzero['mean_anthroscore'], df_nonzero[var])
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        
        correlations.append({
            'variable': var.replace('emotion_', '').replace('emotional_', ''),
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spearman,
            'significant': p < 0.05
        })
        
        print(f"  {var:25s}: r = {r:+.4f} (p = {p:.4f}) {sig}")
    
    results['correlations'] = pd.DataFrame(correlations)
    
    # Highlight strongest correlations
    print("\n" + "-" * 50)
    print("STRONGEST CORRELATIONS (by absolute r):")
    print("-" * 50)
    
    top_corrs = sorted(correlations, key=lambda x: abs(x['pearson_r']), reverse=True)[:5]
    for c in top_corrs:
        direction = "POSITIVE" if c['pearson_r'] > 0 else "NEGATIVE"
        print(f"  {c['variable']:20s}: r = {c['pearson_r']:+.4f} ({direction})")
    
    return results


def option_b_group_differences(df):
    """
    Option B: Emotional Profile Differences (High vs Low Anthropomorphizers)
    """
    print("\n" + "=" * 70)
    print("OPTION B: EMOTIONAL PROFILE DIFFERENCES")
    print("=" * 70)
    
    results = {}
    
    # Filter to non-zero anthro
    df_nonzero = df[df['mean_anthroscore'] > 0].copy()
    
    emotions = ['emotion_joy', 'emotion_sadness', 'emotion_anger', 
                'emotion_fear', 'emotion_surprise', 'emotion_disgust',
                'emotional_intensity', 'emotional_valence', 'emotional_diversity']
    
    # Use quartiles for High vs Low comparison (top 25% vs bottom 25%)
    q25 = df_nonzero['mean_anthroscore'].quantile(0.25)
    q75 = df_nonzero['mean_anthroscore'].quantile(0.75)
    
    high = df_nonzero[df_nonzero['mean_anthroscore'] >= q75]
    low = df_nonzero[df_nonzero['mean_anthroscore'] <= q25]
    
    print(f"\nHigh Anthropomorphizers: {len(high):,} users")
    print(f"Low Anthropomorphizers:  {len(low):,} users")
    
    print("\n" + "-" * 50)
    print("T-Test Comparisons (High vs Low Anthropomorphizers):")
    print("-" * 50)
    
    comparisons = []
    for emotion in emotions:
        t_stat, p_val = ttest_ind(high[emotion], low[emotion])
        
        # Cohen's d effect size
        pooled_std = np.sqrt((high[emotion].std()**2 + low[emotion].std()**2) / 2)
        cohens_d = (high[emotion].mean() - low[emotion].mean()) / pooled_std if pooled_std > 0 else 0
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        comparisons.append({
            'emotion': emotion.replace('emotion_', '').replace('emotional_', ''),
            'high_mean': high[emotion].mean(),
            'low_mean': low[emotion].mean(),
            'difference': high[emotion].mean() - low[emotion].mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'significant': p_val < 0.05
        })
        
        high_mean = high[emotion].mean()
        low_mean = low[emotion].mean()
        direction = "HIGH > LOW" if high_mean > low_mean else "LOW > HIGH"
        
        name = emotion.replace('emotion_', '').replace('emotional_', '')
        print(f"  {name:20s}: High={high_mean:.4f}, Low={low_mean:.4f}, d={cohens_d:+.3f} {sig} ({direction})")
    
    results['comparisons'] = pd.DataFrame(comparisons)
    
    # Dominant emotion distribution
    print("\n" + "-" * 50)
    print("Dominant Emotion Distribution:")
    print("-" * 50)
    
    high_dist = high['dominant_emotion'].value_counts(normalize=True)
    low_dist = low['dominant_emotion'].value_counts(normalize=True)
    
    print(f"\n{'Emotion':<15} {'High Anthro':>12} {'Low Anthro':>12} {'Difference':>12}")
    print("-" * 55)
    
    all_emotions = set(high_dist.index) | set(low_dist.index)
    for emotion in sorted(all_emotions):
        h = high_dist.get(emotion, 0) * 100
        l = low_dist.get(emotion, 0) * 100
        diff = h - l
        print(f"{emotion:<15} {h:>11.1f}% {l:>11.1f}% {diff:>+11.1f}%")
    
    # Chi-square test
    contingency = pd.crosstab(df_nonzero['anthro_group'], df_nonzero['dominant_emotion'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test: X2 = {chi2:.2f}, p = {p:.4f}")
    
    results['chi_square'] = {'chi2': chi2, 'p': p, 'dof': dof}
    
    return results


def option_c_age_moderation(df):
    """
    Option C: Age-Moderated Emotional Relationships
    """
    print("\n" + "=" * 70)
    print("OPTION C: AGE-MODERATED EMOTIONAL RELATIONSHIPS")
    print("=" * 70)
    
    results = {}
    
    # Filter to users with age predictions and non-zero anthro
    df_analysis = df[(df['age_group'].notna()) & (df['mean_anthroscore'] > 0)].copy()
    
    # Create teen indicator
    df_analysis['is_teen'] = df_analysis['age_group'] == '13-18'
    
    teens = df_analysis[df_analysis['is_teen']]
    adults = df_analysis[~df_analysis['is_teen']]
    
    print(f"\nTeens (13-18): {len(teens):,} users")
    print(f"Adults (19+):  {len(adults):,} users")
    
    emotions = ['emotional_intensity', 'emotional_valence', 'emotional_diversity',
                'emotion_joy', 'emotion_sadness', 'emotion_anger']
    
    print("\n" + "-" * 50)
    print("Correlations by Age Group:")
    print("-" * 50)
    
    age_comparisons = []
    for emotion in emotions:
        # Teen correlation
        r_teen, p_teen = pearsonr(teens['mean_anthroscore'], teens[emotion])
        # Adult correlation
        r_adult, p_adult = pearsonr(adults['mean_anthroscore'], adults[emotion])
        
        # Fisher's z-test for difference between correlations
        z_teen = np.arctanh(r_teen)
        z_adult = np.arctanh(r_adult)
        se_diff = np.sqrt(1/(len(teens)-3) + 1/(len(adults)-3))
        z_diff = (z_teen - z_adult) / se_diff
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
        
        age_comparisons.append({
            'emotion': emotion.replace('emotion_', '').replace('emotional_', ''),
            'teen_r': r_teen,
            'teen_p': p_teen,
            'adult_r': r_adult,
            'adult_p': p_adult,
            'z_difference': z_diff,
            'p_difference': p_diff,
            'stronger_for': 'Teens' if abs(r_teen) > abs(r_adult) else 'Adults'
        })
        
        name = emotion.replace('emotion_', '').replace('emotional_', '')
        t_sig = "*" if p_teen < 0.05 else ""
        a_sig = "*" if p_adult < 0.05 else ""
        d_sig = "*" if p_diff < 0.05 else ""
        
        print(f"  {name:20s}: Teens r={r_teen:+.4f}{t_sig}, Adults r={r_adult:+.4f}{a_sig}, diff p={p_diff:.4f}{d_sig}")
    
    results['age_comparisons'] = pd.DataFrame(age_comparisons)
    
    # Interaction model
    print("\n" + "-" * 50)
    print("Regression with Interaction Terms:")
    print("-" * 50)
    
    # Standardize for regression
    df_analysis['anthro_z'] = (df_analysis['mean_anthroscore'] - df_analysis['mean_anthroscore'].mean()) / df_analysis['mean_anthroscore'].std()
    df_analysis['teen_numeric'] = df_analysis['is_teen'].astype(int)
    
    for emotion in ['emotional_intensity', 'emotional_valence']:
        df_analysis['emotion_z'] = (df_analysis[emotion] - df_analysis[emotion].mean()) / df_analysis[emotion].std()
        
        # Fit interaction model
        model = ols(f'emotion_z ~ anthro_z * teen_numeric', data=df_analysis).fit()
        
        name = emotion.replace('emotional_', '').upper()
        print(f"\n{name}:")
        print(f"  AnthroScore main effect:    B = {model.params['anthro_z']:.4f}, p = {model.pvalues['anthro_z']:.4f}")
        print(f"  Teen main effect:           B = {model.params['teen_numeric']:.4f}, p = {model.pvalues['teen_numeric']:.4f}")
        print(f"  Interaction (Anthro x Teen): B = {model.params['anthro_z:teen_numeric']:.4f}, p = {model.pvalues['anthro_z:teen_numeric']:.4f}")
        
        if model.pvalues['anthro_z:teen_numeric'] < 0.05:
            print(f"  *** SIGNIFICANT INTERACTION EFFECT! ***")
    
    return results


def option_d_valence_analysis(df):
    """
    Option D: Emotional Valence & AI Companion Attachment
    """
    print("\n" + "=" * 70)
    print("OPTION D: EMOTIONAL VALENCE & AI COMPANION ATTACHMENT")
    print("=" * 70)
    
    results = {}
    
    # Filter to non-zero anthro
    df_nonzero = df[df['mean_anthroscore'] > 0].copy()
    
    # Classify valence
    df_nonzero['valence_category'] = pd.cut(
        df_nonzero['emotional_valence'],
        bins=[-np.inf, -0.1, 0.1, np.inf],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    print(f"\nValence Distribution:")
    print(df_nonzero['valence_category'].value_counts())
    
    # AnthroScore by valence category
    print("\n" + "-" * 50)
    print("AnthroScore by Emotional Valence Category:")
    print("-" * 50)
    
    for cat in ['Positive', 'Neutral', 'Negative']:
        subset = df_nonzero[df_nonzero['valence_category'] == cat]
        mean_anthro = subset['mean_anthroscore'].mean()
        std_anthro = subset['mean_anthroscore'].std()
        print(f"  {cat:10s}: Mean AnthroScore = {mean_anthro:.4f} (SD = {std_anthro:.4f}), n = {len(subset):,}")
    
    # ANOVA
    groups = [df_nonzero[df_nonzero['valence_category'] == cat]['mean_anthroscore'] 
              for cat in ['Positive', 'Neutral', 'Negative']]
    f_stat, p_val = f_oneway(*groups)
    print(f"\nOne-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")
    
    results['anova'] = {'f_stat': f_stat, 'p_val': p_val}
    
    # Detailed valence correlation
    r, p = pearsonr(df_nonzero['mean_anthroscore'], df_nonzero['emotional_valence'])
    print(f"\nCorrelation (AnthroScore vs Valence): r = {r:+.4f}, p = {p:.4f}")
    
    if r > 0:
        print("  -> Higher anthropomorphization is associated with MORE POSITIVE emotional expression")
    else:
        print("  -> Higher anthropomorphization is associated with MORE NEGATIVE emotional expression")
    
    results['valence_correlation'] = {'r': r, 'p': p}
    
    # By age group
    print("\n" + "-" * 50)
    print("Valence-Anthro Relationship by Age:")
    print("-" * 50)
    
    df_with_age = df_nonzero[df_nonzero['age_group'].notna()]
    
    for age in df_with_age['age_group'].unique():
        subset = df_with_age[df_with_age['age_group'] == age]
        if len(subset) > 30:
            r, p = pearsonr(subset['mean_anthroscore'], subset['emotional_valence'])
            sig = "*" if p < 0.05 else ""
            print(f"  {age:10s}: r = {r:+.4f} (p = {p:.4f}) {sig}, n = {len(subset):,}")
    
    # Max AnthroScore analysis (peak anthropomorphization)
    print("\n" + "-" * 50)
    print("Max AnthroScore Analysis (Peak Anthropomorphization):")
    print("-" * 50)
    
    r_max, p_max = pearsonr(df_nonzero['max_anthroscore'], df_nonzero['emotional_valence'])
    print(f"  Max AnthroScore vs Valence: r = {r_max:+.4f}, p = {p_max:.4f}")
    
    r_max_int, p_max_int = pearsonr(df_nonzero['max_anthroscore'], df_nonzero['emotional_intensity'])
    print(f"  Max AnthroScore vs Intensity: r = {r_max_int:+.4f}, p = {p_max_int:.4f}")
    
    return results


def generate_summary(all_results):
    """Generate a summary of the most interesting findings."""
    print("\n" + "=" * 70)
    print("SUMMARY: MOST INTERESTING & SURPRISING FINDINGS")
    print("=" * 70)
    
    findings = []
    
    # Option A findings
    if 'option_a' in all_results:
        corrs = all_results['option_a']['correlations']
        sig_corrs = corrs[corrs['significant']]
        for _, row in sig_corrs.iterrows():
            effect = "MORE" if row['pearson_r'] > 0 else "LESS"
            findings.append({
                'source': 'Option A',
                'finding': f"Users who anthropomorphize more show {effect} {row['variable']}",
                'effect_size': abs(row['pearson_r']),
                'p_value': row['pearson_p'],
                'metric': f"r = {row['pearson_r']:+.4f}"
            })
    
    # Option B findings
    if 'option_b' in all_results:
        comps = all_results['option_b']['comparisons']
        sig_comps = comps[comps['significant']]
        for _, row in sig_comps.iterrows():
            direction = "higher" if row['difference'] > 0 else "lower"
            findings.append({
                'source': 'Option B',
                'finding': f"High anthropomorphizers have {direction} {row['emotion']} than low anthropomorphizers",
                'effect_size': abs(row['cohens_d']),
                'p_value': row['p_value'],
                'metric': f"d = {row['cohens_d']:+.3f}"
            })
    
    # Sort by effect size
    findings.sort(key=lambda x: x['effect_size'], reverse=True)
    
    print("\n" + "=" * 70)
    print("TOP FINDINGS (sorted by effect size):")
    print("=" * 70)
    
    for i, f in enumerate(findings[:10], 1):
        stars = "***" if f['p_value'] < 0.001 else "**" if f['p_value'] < 0.01 else "*"
        print(f"\n{i}. [{f['source']}] {f['finding']}")
        print(f"   {f['metric']}, p = {f['p_value']:.4f} {stars}")
    
    return findings


def save_comprehensive_report(all_results, findings, df):
    """Save a comprehensive report of all findings."""
    report_path = RESULTS_DIR / "RQ3_EMOTIONAL_ANALYSIS_REPORT.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RQ3: EMOTIONAL EXPRESSION & ANTHROPOMORPHIZATION ANALYSIS\n")
        f.write("Comprehensive Results Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("How do emotional expression patterns relate to anthropomorphization of AI companions?\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total users analyzed: {len(df):,}\n")
        f.write(f"Users with non-zero AnthroScore: {(df['mean_anthroscore'] > 0).sum():,}\n")
        f.write(f"Users with age predictions: {df['age_group'].notna().sum():,}\n")
        f.write(f"Users with gender predictions: {df['gender'].notna().sum():,}\n\n")
        
        # Option A
        f.write("-" * 80 + "\n")
        f.write("OPTION A: EMOTIONAL INTENSITY CORRELATIONS\n")
        f.write("-" * 80 + "\n")
        if 'option_a' in all_results:
            for _, row in all_results['option_a']['correlations'].iterrows():
                sig = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['significant'] else ""
                f.write(f"  {row['variable']:20s}: r = {row['pearson_r']:+.4f} (p = {row['pearson_p']:.4f}) {sig}\n")
        f.write("\n")
        
        # Option B
        f.write("-" * 80 + "\n")
        f.write("OPTION B: GROUP DIFFERENCES (HIGH vs LOW ANTHROPOMORPHIZERS)\n")
        f.write("-" * 80 + "\n")
        if 'option_b' in all_results:
            for _, row in all_results['option_b']['comparisons'].iterrows():
                sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['significant'] else ""
                f.write(f"  {row['emotion']:20s}: d = {row['cohens_d']:+.3f} (p = {row['p_value']:.4f}) {sig}\n")
        f.write("\n")
        
        # Top findings
        f.write("=" * 80 + "\n")
        f.write("TOP 10 MOST INTERESTING FINDINGS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, finding in enumerate(findings[:10], 1):
            stars = "***" if finding['p_value'] < 0.001 else "**" if finding['p_value'] < 0.01 else "*"
            f.write(f"{i}. {finding['finding']}\n")
            f.write(f"   Source: {finding['source']}\n")
            f.write(f"   Effect: {finding['metric']}, p = {finding['p_value']:.4f} {stars}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        
        # Auto-generate interpretation based on findings
        if findings:
            f.write("Key patterns emerging from this analysis:\n\n")
            
            # Check for emotion patterns
            joy_findings = [f for f in findings if 'joy' in f['finding'].lower()]
            if joy_findings:
                f.write("1. JOY: " + joy_findings[0]['finding'] + "\n")
            
            sad_findings = [f for f in findings if 'sadness' in f['finding'].lower()]
            if sad_findings:
                f.write("2. SADNESS: " + sad_findings[0]['finding'] + "\n")
            
            intensity_findings = [f for f in findings if 'intensity' in f['finding'].lower()]
            if intensity_findings:
                f.write("3. INTENSITY: " + intensity_findings[0]['finding'] + "\n")
            
            valence_findings = [f for f in findings if 'valence' in f['finding'].lower()]
            if valence_findings:
                f.write("4. VALENCE: " + valence_findings[0]['finding'] + "\n")
    
    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    """Run all RQ3 analyses."""
    print("\n" + "=" * 70)
    print("RQ3: EMOTIONAL EXPRESSION & ANTHROPOMORPHIZATION")
    print("Comprehensive Analysis")
    print("=" * 70)
    
    # Load data
    df = load_and_merge_data()
    
    # Calculate derived metrics
    df = calculate_derived_metrics(df)
    
    # Run all analyses
    all_results = {}
    
    # Option A: Correlations
    all_results['option_a'] = option_a_correlations(df)
    
    # Option B: Group differences
    all_results['option_b'] = option_b_group_differences(df)
    
    # Option C: Age moderation
    all_results['option_c'] = option_c_age_moderation(df)
    
    # Option D: Valence analysis
    all_results['option_d'] = option_d_valence_analysis(df)
    
    # Generate summary
    findings = generate_summary(all_results)
    
    # Save report
    save_comprehensive_report(all_results, findings, df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return all_results, findings


if __name__ == "__main__":
    all_results, findings = main()

