"""Combined Age x Gender Analysis."""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

print('=== COMBINED AGE x GENDER ANALYSIS ===')
print()

# Load all data
age_preds = pd.read_parquet('Data/features/ultimate_predictor/ultimate_predictions.parquet')
gender_preds = pd.read_parquet('Data/features/ultimate_predictor/gender_predictions.parquet')
anthro = pd.read_parquet('Data/features/user_anthroscores.parquet')

# Rename columns to avoid confusion
age_preds = age_preds.rename(columns={'confidence': 'age_confidence'})
gender_preds = gender_preds.rename(columns={'confidence': 'gender_confidence'})

# Merge all
data = age_preds.merge(gender_preds[['author', 'gender_predicted', 'gender_confidence']], on='author')
data = data.merge(anthro, on='author')

# Filter to high confidence on BOTH
data = data[(data['age_confidence'] >= 0.6) & 
            (data['gender_confidence'] >= 0.6) & 
            (data['anthroscore_mean'] != 0)]

print(f'Users with high-confidence age AND gender AND non-zero AnthroScore: {len(data)}')
print()

# Create binary age
data['is_teen'] = data['age_bucket_predicted'] == 'teen'

# Cross-tabulation
print('=== CROSS-TABULATION ===')
ct = pd.crosstab(data['age_bucket_predicted'], data['gender_predicted'])
print(ct)
print()

# Group means
print('=== MEAN MAX ANTHROSCORE BY AGE x GENDER ===')
grouped = data.groupby(['age_bucket_predicted', 'gender_predicted'])['anthroscore_max'].agg(['mean', 'count'])
print(grouped)
print()

# Regression with interaction
print('=== REGRESSION ANALYSIS ===')
data['is_teen_int'] = data['is_teen'].astype(int)
data['is_female'] = (data['gender_predicted'] == 'female').astype(int)

# Model 1: Main effects only
model1 = smf.ols('anthroscore_max ~ is_teen_int + is_female', data=data).fit()
print('Model 1 (main effects):')
print(f'  Teen: coef={model1.params["is_teen_int"]:.4f}, p={model1.pvalues["is_teen_int"]:.4f}')
print(f'  Female: coef={model1.params["is_female"]:.4f}, p={model1.pvalues["is_female"]:.4f}')
print(f'  R-squared: {model1.rsquared:.4f}')
print()

# Model 2: With interaction
model2 = smf.ols('anthroscore_max ~ is_teen_int * is_female', data=data).fit()
print('Model 2 (with interaction):')
print(f'  Teen: coef={model2.params["is_teen_int"]:.4f}, p={model2.pvalues["is_teen_int"]:.4f}')
print(f'  Female: coef={model2.params["is_female"]:.4f}, p={model2.pvalues["is_female"]:.4f}')
print(f'  Teen x Female: coef={model2.params["is_teen_int:is_female"]:.4f}, p={model2.pvalues["is_teen_int:is_female"]:.4f}')
print(f'  R-squared: {model2.rsquared:.4f}')
print()

# Specific subgroup analysis
print('=== SUBGROUP COMPARISON ===')
teen_male = data[(data['is_teen']) & (data['gender_predicted']=='male')]['anthroscore_max']
teen_female = data[(data['is_teen']) & (data['gender_predicted']=='female')]['anthroscore_max']
adult_male = data[(~data['is_teen']) & (data['gender_predicted']=='male')]['anthroscore_max']
adult_female = data[(~data['is_teen']) & (data['gender_predicted']=='female')]['anthroscore_max']

print(f'Teen Male: mean={teen_male.mean():.3f}, n={len(teen_male)}')
print(f'Teen Female: mean={teen_female.mean():.3f}, n={len(teen_female)}')
print(f'Adult Male: mean={adult_male.mean():.3f}, n={len(adult_male)}')
print(f'Adult Female: mean={adult_female.mean():.3f}, n={len(adult_female)}')
print()

# Most anthropomorphizing group?
print('=== KEY FINDING ===')
groups = {
    'Teen Male': teen_male.mean(),
    'Teen Female': teen_female.mean(),
    'Adult Male': adult_male.mean(),
    'Adult Female': adult_female.mean()
}
sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
print('Groups ranked by mean max AnthroScore:')
for i, (group, mean) in enumerate(sorted_groups, 1):
    print(f'  {i}. {group}: {mean:.3f}')
print()

# Statistical tests between key groups
print('=== STATISTICAL SIGNIFICANCE ===')

# Teen vs Adult (overall)
teen_all = data[data['is_teen']]['anthroscore_max']
adult_all = data[~data['is_teen']]['anthroscore_max']
t, p = stats.ttest_ind(teen_all, adult_all)
print(f'Teen vs Adult (overall): t={t:.3f}, p={p:.4f}')

# Male vs Female (overall)
male_all = data[data['gender_predicted']=='male']['anthroscore_max']
female_all = data[data['gender_predicted']=='female']['anthroscore_max']
t, p = stats.ttest_ind(male_all, female_all)
print(f'Male vs Female (overall): t={t:.3f}, p={p:.4f}')

# Teen Female vs Adult Male (most vs least?)
t, p = stats.ttest_ind(teen_female, adult_male)
print(f'Teen Female vs Adult Male: t={t:.3f}, p={p:.4f}')

# Teen Male vs Teen Female
t, p = stats.ttest_ind(teen_male, teen_female)
print(f'Teen Male vs Teen Female: t={t:.3f}, p={p:.4f}')

# Two-way ANOVA
print()
print('=== TWO-WAY ANOVA ===')
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('anthroscore_max ~ C(is_teen) + C(is_female) + C(is_teen):C(is_female)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

