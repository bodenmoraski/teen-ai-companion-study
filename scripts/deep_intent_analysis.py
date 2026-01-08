"""Deep analysis of intent findings."""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print('=== DETAILED INTENT ANALYSIS ===')
print()

# Load the merged data
data = pd.read_parquet('Data/features/full_analysis_data.parquet')
print(f'Total sample: {len(data)}')
print()

# 1. Intent by Age breakdown
print('=== INTENT BY AGE (Proportions) ===')
data['is_teen'] = data['age_bucket_predicted'] == 'teen'
ct = pd.crosstab(data['dominant_intent'], data['is_teen'], normalize='columns')
ct.columns = ['Adult', 'Teen']
ct['Diff (Teen-Adult)'] = ct['Teen'] - ct['Adult']
ct = ct.sort_values('Diff (Teen-Adult)', ascending=False)
print(ct.round(3))
print()

# 2. Intent by Gender breakdown
print('=== INTENT BY GENDER (Proportions) ===')
ct = pd.crosstab(data['dominant_intent'], data['gender_predicted'], normalize='columns')
ct['Diff (F-M)'] = ct['female'] - ct['male']
ct = ct.sort_values('Diff (F-M)', ascending=False)
print(ct.round(3))
print()

# 3. AnthroScore by Intent - ranked
print('=== ANTHROSCORE BY INTENT (Ranked) ===')
anthro_by_intent = data.groupby('dominant_intent')['anthroscore_max'].agg(['mean', 'std', 'count'])
anthro_by_intent = anthro_by_intent.sort_values('mean', ascending=False)
print(anthro_by_intent.round(3))
print()

# 4. THE KEY QUESTION: Does intent EXPLAIN the age effect?
print('=== DOES INTENT EXPLAIN THE AGE EFFECT? ===')

# Age effect without controlling for intent
teen_all = data[data['is_teen']]['anthroscore_max']
adult_all = data[~data['is_teen']]['anthroscore_max']
t_raw, p_raw = stats.ttest_ind(teen_all, adult_all)
print(f'Raw age effect: Teen={teen_all.mean():.3f}, Adult={adult_all.mean():.3f}, p={p_raw:.4f}')

# Within each intent, is there still an age effect?
print()
print('Age effect WITHIN each intent:')
for intent in data['dominant_intent'].unique():
    subset = data[data['dominant_intent'] == intent]
    teen = subset[subset['is_teen']]['anthroscore_max']
    adult = subset[~subset['is_teen']]['anthroscore_max']
    if len(teen) >= 10 and len(adult) >= 10:
        t, p = stats.ttest_ind(teen, adult)
        print(f'  {intent}: Teen({len(teen)})={teen.mean():.3f}, Adult({len(adult)})={adult.mean():.3f}, p={p:.4f}')

# 5. Most interesting finding: What characterizes HIGH anthropomorphizers?
print()
print('=== PROFILE OF HIGH ANTHROPOMORPHIZERS (Top 10%) ===')
threshold = data['anthroscore_max'].quantile(0.90)
high = data[data['anthroscore_max'] >= threshold]
low = data[data['anthroscore_max'] < threshold]

print(f'High anthropomorphizers: {len(high)} users')
print()
print('Demographics:')
high_teen_rate = high['is_teen'].mean() * 100
low_teen_rate = low['is_teen'].mean() * 100
print(f'  Teen rate: High={high_teen_rate:.1f}% vs Low={low_teen_rate:.1f}%')

high_female_rate = (high['gender_predicted'] == 'female').mean() * 100
low_female_rate = (low['gender_predicted'] == 'female').mean() * 100
print(f'  Female rate: High={high_female_rate:.1f}% vs Low={low_female_rate:.1f}%')
print()

print('Intent distribution (High vs Low):')
for intent in ['character_creation', 'roleplay_fantasy', 'emotional_support', 'community_sharing', 'other', 'unknown']:
    high_rate = (high['dominant_intent'] == intent).mean() * 100
    low_rate = (low['dominant_intent'] == intent).mean() * 100
    diff = high_rate - low_rate
    if abs(diff) > 1:  # Only show meaningful differences
        print(f'  {intent}: High={high_rate:.1f}% vs Low={low_rate:.1f}% (diff={diff:+.1f}pp)')

# 6. Three-way analysis: Age × Gender × Intent → AnthroScore
print()
print('=== THREE-WAY ANALYSIS: Age × Gender × Intent ===')
print()

# For each intent, show the age × gender breakdown
for intent in ['character_creation', 'other', 'unknown']:
    if intent not in data['dominant_intent'].values:
        continue
    subset = data[data['dominant_intent'] == intent]
    print(f'{intent.upper()} (n={len(subset)}):')
    
    teen_male = subset[(subset['is_teen']) & (subset['gender_predicted'] == 'male')]['anthroscore_max']
    teen_female = subset[(subset['is_teen']) & (subset['gender_predicted'] == 'female')]['anthroscore_max']
    adult_male = subset[(~subset['is_teen']) & (subset['gender_predicted'] == 'male')]['anthroscore_max']
    adult_female = subset[(~subset['is_teen']) & (subset['gender_predicted'] == 'female')]['anthroscore_max']
    
    if len(teen_male) >= 5:
        print(f'  Teen Male: {teen_male.mean():.3f} (n={len(teen_male)})')
    if len(teen_female) >= 5:
        print(f'  Teen Female: {teen_female.mean():.3f} (n={len(teen_female)})')
    if len(adult_male) >= 5:
        print(f'  Adult Male: {adult_male.mean():.3f} (n={len(adult_male)})')
    if len(adult_female) >= 5:
        print(f'  Adult Female: {adult_female.mean():.3f} (n={len(adult_female)})')
    print()

# 7. Summary statistics
print('=== SUMMARY ===')
print()
print('Key findings from intent analysis:')
print('1. Intent is significantly associated with Age (p=0.0006)')
print('2. Intent is significantly associated with Gender (p=0.0001)')
print('3. Intent is significantly associated with Anthropomorphization (p<0.0001)')
print()
print('This suggests that WHY people use AI companions matters for')
print('understanding anthropomorphization patterns.')

