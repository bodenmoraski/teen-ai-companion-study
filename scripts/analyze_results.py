"""Analyze the demographic results with new age predictions."""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print('=== DEEP DIVE: PEAK ANTHROPOMORPHIZATION ===')
print()

# Load data
preds = pd.read_parquet('Data/features/ultimate_predictor/ultimate_predictions.parquet')
anthro = pd.read_parquet('Data/features/user_anthroscores.parquet')
merged = preds.merge(anthro, on='author')
merged['is_teen'] = merged['age_bucket_predicted'] == 'teen'
data = merged[(merged['confidence'] >= 0.6) & (merged['anthroscore_mean'] != 0)]

# Effect size for MAX
teen_max = data[data['is_teen']]['anthroscore_max']
non_teen_max = data[~data['is_teen']]['anthroscore_max']
pooled_std = np.sqrt(((len(teen_max)-1)*teen_max.var() + (len(non_teen_max)-1)*non_teen_max.var()) / (len(teen_max)+len(non_teen_max)-2))
d = (teen_max.mean() - non_teen_max.mean()) / pooled_std

print(f'MAX ANTHROSCORE:')
print(f'  Teen: {teen_max.mean():.3f} (std={teen_max.std():.3f})')
print(f'  Non-Teen: {non_teen_max.mean():.3f} (std={non_teen_max.std():.3f})')
print(f'  Difference: {teen_max.mean() - non_teen_max.mean():.3f}')
print(f"  Cohen's d: {d:.3f} (small-medium effect)")
print()

# Look at comment count - maybe teens just have more comments?
print('=== CONTROLLING FOR ACTIVITY ===')
print(f'Mean comment count - Teen: {data[data.is_teen]["anthroscore_count"].mean():.1f}')
print(f'Mean comment count - Non-Teen: {data[~data.is_teen]["anthroscore_count"].mean():.1f}')

# Check if effect holds when controlling for comment count
# Divide max by count (normalized)
data['normalized_max'] = data['anthroscore_max'] / np.log1p(data['anthroscore_count'])
teen_norm = data[data['is_teen']]['normalized_max']
non_teen_norm = data[~data['is_teen']]['normalized_max']
t, p = stats.ttest_ind(teen_norm, non_teen_norm)
print()
print(f'NORMALIZED MAX (controlling for activity):')
print(f'  t={t:.3f}, p={p:.6f}')

# Binary: are teens more likely to have ANY high anthropomorphization?
print()
print('=== PROPORTION WITH HIGH ANTHROPOMORPHIZATION ===')
for threshold in [0.5, 1.0, 1.5, 2.0]:
    teen_high = (teen_max >= threshold).mean()
    non_teen_high = (non_teen_max >= threshold).mean()
    # Chi-square
    contingency = [[
        (data.is_teen & (data.anthroscore_max >= threshold)).sum(),
        (data.is_teen & (data.anthroscore_max < threshold)).sum()
    ], [
        (~data.is_teen & (data.anthroscore_max >= threshold)).sum(),
        (~data.is_teen & (data.anthroscore_max < threshold)).sum()
    ]]
    chi2, p = stats.chi2_contingency(contingency)[:2]
    print(f'Max >= {threshold}: Teen {teen_high*100:.1f}% vs Non-Teen {non_teen_high*100:.1f}% (p={p:.4f})')

# What about the 3-class breakdown?
print()
print('=== MAX ANTHROSCORE BY AGE GROUP ===')
for age in ['teen', 'young_adult', 'adult']:
    subset = data[data['age_bucket_predicted'] == age]['anthroscore_max']
    print(f'{age}: mean={subset.mean():.3f}, median={subset.median():.3f}, n={len(subset)}')

# Regression with all controls
print()
print('=== REGRESSION ANALYSIS ===')
import statsmodels.formula.api as smf

# Prepare for regression
data['age_teen'] = data['is_teen'].astype(int)
data['log_count'] = np.log1p(data['anthroscore_count'])

# Model 1: Simple
model1 = smf.ols('anthroscore_max ~ age_teen', data=data).fit()
print('Model 1 (simple):')
print(f'  Teen coefficient: {model1.params["age_teen"]:.4f} (p={model1.pvalues["age_teen"]:.4f})')

# Model 2: Control for activity
model2 = smf.ols('anthroscore_max ~ age_teen + log_count', data=data).fit()
print('Model 2 (controlling for activity):')
print(f'  Teen coefficient: {model2.params["age_teen"]:.4f} (p={model2.pvalues["age_teen"]:.4f})')
print(f'  R-squared: {model2.rsquared:.4f}')

