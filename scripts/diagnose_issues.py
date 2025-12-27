"""
Comprehensive diagnostic script to understand why classifications are failing.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
print("=" * 70)

# Load data
df = pd.read_parquet('data/features/demographics.parquet')
print(f"\nTotal users: {len(df):,}")

# ============================================
# AGE ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("AGE CLASSIFICATION ANALYSIS")
print("=" * 70)

# Check age scores by self-declared bucket
mask = df['age_bucket_self_declared'].notna() & df['age_community_score'].notna()
subset = df[mask].copy()
print(f"\nUsers with BOTH self-declared age AND community score: {len(subset)}")

print("\n=== AGE SCORE BY SELF-DECLARED BUCKET ===")
print("(Ideal: scores should INCREASE with age if embeddings work)")
for bucket in ['13-18', '19-25', '26-40', '41-60', '61-80']:
    data = subset[subset['age_bucket_self_declared']==bucket]['age_community_score']
    if len(data) > 0:
        print(f"  {bucket}: mean={data.mean():.4f}, std={data.std():.4f}, n={len(data)}")

# Check if there's ANY correlation between age score and actual age
# Map buckets to numeric
age_to_num = {'13-18': 15, '19-25': 22, '26-40': 33, '41-60': 50, '61-80': 70}
subset['age_numeric'] = subset['age_bucket_self_declared'].map(age_to_num)
corr = subset['age_community_score'].corr(subset['age_numeric'])
print(f"\nCorrelation between age score and actual age: {corr:.4f}")

# ============================================
# GENDER ANALYSIS  
# ============================================
print("\n" + "=" * 70)
print("GENDER CLASSIFICATION ANALYSIS")
print("=" * 70)

# Check gender scores by self-declared gender
mask = df['gender_self_declared'].notna() & df['gender_community_score'].notna()
subset = df[mask].copy()
print(f"\nUsers with BOTH self-declared gender AND community score: {len(subset)}")

print("\n=== GENDER SCORE BY SELF-DECLARED GENDER ===")
print("(Ideal: males should be positive, females should be negative)")
for gender in ['male', 'female', 'nonbinary']:
    data = subset[subset['gender_self_declared']==gender]['gender_community_score']
    if len(data) > 0:
        print(f"  {gender}: mean={data.mean():.4f}, std={data.std():.4f}, n={len(data)}")

# Check the ENTIRE distribution of gender scores
print("\n=== OVERALL GENDER SCORE DISTRIBUTION ===")
all_scores = df['gender_community_score'].dropna()
print(f"  Total: mean={all_scores.mean():.4f}, std={all_scores.std():.4f}")
print(f"  Min: {all_scores.min():.4f}, Max: {all_scores.max():.4f}")
print(f"  Percentiles: 10th={np.percentile(all_scores, 10):.4f}, 50th={np.percentile(all_scores, 50):.4f}, 90th={np.percentile(all_scores, 90):.4f}")

# ============================================
# CHECK SEED PAIR VALIDITY
# ============================================
print("\n" + "=" * 70)
print("SEED PAIR ANALYSIS")
print("=" * 70)

# Load subreddit interactions
usi_path = Path('data/features/user_subreddit_interactions.parquet')
if usi_path.exists():
    usi = pd.read_parquet(usi_path)
    usi['subreddit_lower'] = usi['subreddit'].str.lower()
    all_subs = set(usi['subreddit_lower'].unique())
    
    print(f"\nTotal unique subreddits in data: {len(all_subs):,}")
    
    # Check age seed pairs
    age_seeds = [
        ("teenagers", "redditforgrownups"),
        ("teenrelationships", "relationship_advice"),
        ("highschool", "college"),
        ("genz", "genx"),
    ]
    print("\n=== AGE SEED PAIRS ===")
    for term1, term2 in age_seeds:
        t1_count = usi[usi['subreddit_lower']==term1]['author'].nunique()
        t2_count = usi[usi['subreddit_lower']==term2]['author'].nunique()
        in_data = term1 in all_subs and term2 in all_subs
        status = "OK" if in_data else "MISSING"
        print(f"  [{status}] ({term1}: {t1_count} users, {term2}: {t2_count} users)")
    
    # Check gender seed pairs
    gender_seeds = [
        ("askwomen", "askmen"),
        ("twoxchromosomes", "mensrights"),
        ("thegirlsurvivalguide", "oney"),
    ]
    print("\n=== GENDER SEED PAIRS ===")
    for term1, term2 in gender_seeds:
        t1_count = usi[usi['subreddit_lower']==term1]['author'].nunique()
        t2_count = usi[usi['subreddit_lower']==term2]['author'].nunique()
        in_data = term1 in all_subs and term2 in all_subs
        status = "OK" if in_data and t1_count > 0 and t2_count > 0 else "MISSING"
        print(f"  [{status}] ({term1}: {t1_count} users, {term2}: {t2_count} users)")
else:
    print("\nWARNING: user_subreddit_interactions.parquet not found")

# ============================================
# THE ROOT CAUSE ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("ROOT CAUSE ANALYSIS")
print("=" * 70)

# Problem 1: Percentile-based age bucketing creates EQUAL distribution
print("\n*** PROBLEM 1: Age buckets are 20%/20%/20%/20%/20% ***")
print("The percentile-based approach divides into EQUAL groups, not CALIBRATED ones")
age_counts = df['age_bucket_community'].value_counts(dropna=False)
print("Distribution:")
print(age_counts)

# Problem 2: Gender dimension might be inverted or too weak
print("\n*** PROBLEM 2: Gender scores don't separate genders ***")
mask = df['gender_self_declared'].notna() & df['gender_community_score'].notna()
subset = df[mask]
male_scores = subset[subset['gender_self_declared']=='male']['gender_community_score']
female_scores = subset[subset['gender_self_declared']=='female']['gender_community_score']
# Check overlap
from scipy import stats
if len(male_scores) > 10 and len(female_scores) > 10:
    t_stat, p_val = stats.ttest_ind(male_scores, female_scores)
    print(f"t-test males vs females: t={t_stat:.4f}, p={p_val:.4f}")
    effect_size = (male_scores.mean() - female_scores.mean()) / np.sqrt((male_scores.std()**2 + female_scores.std()**2)/2)
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    if abs(effect_size) < 0.2:
        print("INTERPRETATION: Effect is NEGLIGIBLE - embeddings don't separate genders")
    elif abs(effect_size) < 0.5:
        print("INTERPRETATION: Effect is SMALL - some separation but not much")
    else:
        print("INTERPRETATION: Effect is MEDIUM+ - embeddings do separate genders")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("""
1. AGE: Don't use percentile-based buckets (creates equal 20% distribution)
   Instead: Use fixed thresholds calibrated on self-declarations

2. GENDER: The scores don't separate genders well
   - Males and females both have similar negative scores
   - Need to check if seed pairs are valid in Word2Vec vocabulary
   - May need different/more seed pairs

3. VALIDATION: Gender is 0% because validation filters to male/female/nonbinary
   but community mostly predicts 'unknown' for these users

4. LLM: Has more reasonable distribution (3430 19-25, 1020 13-18, 520 26-40)
   Consider weighting LLM more heavily in ensemble
""")

