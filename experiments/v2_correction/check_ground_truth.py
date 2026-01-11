"""
Check ground truth relationship between age and anthropomorphization.
This validates the core finding we need to replicate.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

base_path = Path(__file__).parent.parent.parent

# Load existing features
features_path = base_path / 'Data/features/ultimate_predictor/all_features.parquet'
if features_path.exists():
    features = pd.read_parquet(features_path)
    print(f'All features shape: {features.shape}')
    print(f'Columns (first 20): {features.columns.tolist()[:20]}')
    print()

# Load self-declarations
self_decl = pd.read_parquet(base_path / 'Data/features/self_declarations.parquet')

# Load anthroscores
anthro = pd.read_parquet(base_path / 'Data/features/user_anthroscores.parquet')

# Check overlap with age labels
age_labeled = self_decl[self_decl['age_bucket_self_declared'].notna()]

if features_path.exists():
    features_with_age = features.merge(age_labeled, on='author', how='inner')
    print(f'Users with features AND known age: {len(features_with_age)}')

# Check overlap with gender labels
gender_labeled = self_decl[self_decl['gender_self_declared'].notna()]
if features_path.exists():
    features_with_gender = features.merge(gender_labeled[['author', 'gender_self_declared']], on='author', how='inner')
    print(f'Users with features AND known gender: {len(features_with_gender)}')

# Merge for analysis
age_anthro = age_labeled.merge(anthro, on='author', how='inner')
print()
print(f'Users with known age AND anthroscores: {len(age_anthro)}')

# Critical check: ground truth relationship
def map_to_binary(bucket):
    if bucket == '13-18':
        return 'teen'
    return 'adult'

age_anthro['age_binary'] = age_anthro['age_bucket_self_declared'].apply(map_to_binary)

# Filter to non-zero anthroscores (matching the original analysis)
age_anthro_nonzero = age_anthro[age_anthro['anthroscore_max'] != 0]

print()
print('='*60)
print('GROUND TRUTH CHECK (ALL USERS WITH KNOWN AGE)')
print('='*60)

teens = age_anthro[age_anthro['age_binary'] == 'teen']['anthroscore_max']
adults = age_anthro[age_anthro['age_binary'] == 'adult']['anthroscore_max']

print(f'Teen mean max AnthroScore: {teens.mean():.4f} (n={len(teens)})')
print(f'Adult mean max AnthroScore: {adults.mean():.4f} (n={len(adults)})')

d = (teens.mean() - adults.mean()) / np.sqrt((teens.std()**2 + adults.std()**2) / 2)
t, p = stats.ttest_ind(teens, adults)
print(f"Cohen's d (teen - adult): {d:.4f}")
print(f'p-value: {p:.6f}')
direction = 'TEENS HIGHER' if d > 0 else 'ADULTS HIGHER'
print(f'Direction: {direction}')

print()
print('='*60)
print('GROUND TRUTH CHECK (NON-ZERO ANTHROSCORE ONLY)')
print('='*60)

teens_nz = age_anthro_nonzero[age_anthro_nonzero['age_binary'] == 'teen']['anthroscore_max']
adults_nz = age_anthro_nonzero[age_anthro_nonzero['age_binary'] == 'adult']['anthroscore_max']

print(f'Teen mean max AnthroScore: {teens_nz.mean():.4f} (n={len(teens_nz)})')
print(f'Adult mean max AnthroScore: {adults_nz.mean():.4f} (n={len(adults_nz)})')

d_nz = (teens_nz.mean() - adults_nz.mean()) / np.sqrt((teens_nz.std()**2 + adults_nz.std()**2) / 2)
t_nz, p_nz = stats.ttest_ind(teens_nz, adults_nz)
print(f"Cohen's d (teen - adult): {d_nz:.4f}")
print(f'p-value: {p_nz:.6f}')
direction_nz = 'TEENS HIGHER' if d_nz > 0 else 'ADULTS HIGHER'
print(f'Direction: {direction_nz}')

print()
print('='*60)
print('SUMMARY: THE GROUND TRUTH')
print('='*60)
print(f'Ground truth shows: {direction} (d = {d:.4f})')
print(f'V1 model predicted: TEENS HIGHER (d = +0.111)')
print()
if d < 0:
    print('*** CRITICAL: V1 model is WRONG on directionality! ***')
    print('V2 must align with ground truth: ADULTS anthropomorphize more.')
else:
    print('V1 model matches ground truth direction.')

# Also check gender ground truth
print()
print('='*60)
print('GENDER DISTRIBUTION IN TRAINING DATA')
print('='*60)
print(self_decl['gender_self_declared'].value_counts())
male_count = (self_decl['gender_self_declared'] == 'male').sum()
female_count = (self_decl['gender_self_declared'] == 'female').sum()
imbalance_ratio = male_count / female_count
print(f'\nImbalance ratio (male/female): {imbalance_ratio:.2f}:1')
print(f'Recommended scale_pos_weight for XGBoost: {imbalance_ratio:.2f}')
