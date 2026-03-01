# The Illusion Project: Full Context Data Document

## Complete Research Archive for AI Analysis

**Purpose:** This document contains EVERYTHING about The Illusion Project - all data, methodology, statistics, models, findings, interpretations, and context. Designed for AI systems to have complete context for analysis, summarization, or question-answering.

**Generated:** January 10, 2026  
**Version:** FINAL

---

# SECTION 1: PROJECT OVERVIEW

## 1.1 Research Questions

**RQ1:** What are the demographics (age, gender) and usage intentions of users discussing relationships with AI companions in online forums?

**RQ2:** How do these demographics relate to users anthropomorphizing AI companions?

**RQ3:** How do emotional expression patterns relate to users anthropomorphizing AI companions?

**Discussion Focus:** How could users anthropomorphizing AI lead to heightened vulnerability to short-term and long-term AI safety risks?

## 1.2 Core Constructs

### Anthropomorphization
The degree to which users treat AI companions as human-like. Measured via the **AnthroScore** metric which captures:
- Language patterns (referring to AI with human pronouns, relationship terms)
- Emotional attribution (claiming AI "feels" emotions)
- Agency attribution (treating AI as having free will/consciousness)
- Relationship language ("my AI friend", "we talked", etc.)

### AI Companions
Social chatbot applications including:
- Character.AI
- Replika
- SillyTavernAI
- TavernAI
- And related apps

## 1.3 Data Sources

| Source | Count | Description |
|--------|-------|-------------|
| Reddit Comments | 283,895 | Raw comments from AI companion subreddits |
| Unique Users | 47,062 | Deduplicated user accounts |
| Self-Declared Age | 459 | Users who stated their age in posts |
| Self-Declared Gender | 4,894 | Users identifiable as male/female |
| Users with AnthroScore > 0 | 10,914 | Users showing any anthropomorphization |

### Subreddits Analyzed
- r/CharacterAI (primary)
- r/Replika
- r/SillyTavernAI
- r/AICompanions
- r/TavernAI
- Related communities

---

# SECTION 2: CLASSIFICATION SYSTEMS

## 2.1 Model Evolution: V1 → V5

| Version | Approach | Gender Accuracy | Age Accuracy | Status |
|---------|----------|-----------------|--------------|--------|
| V1 | Baseline stacked ensemble | 81.3% | 70.6% | Deprecated |
| V2 | Validity-focused (behavioral age, SMOTE gender) | 59.6% | 51.1% | Experimental |
| **V3** | **V1 + threshold optimization + mild SMOTE** | **96.9%** | **95.0%** | **PRODUCTION** |
| V4 | Multi-algorithm stacking (XGB+LightGBM+RF+LR) | 81.7% | 61.4% | Not better than V3 |
| V5 | Aggressive SMOTE-ENN | 34.1% | 51.0% | Failed |

## 2.2 V3 Architecture (Production Model)

### Gender Predictor V3

**Features (803 total):**
- Sentence-BERT text embeddings: 384 dimensions
- Subreddit participation patterns: 400 binary features
- Behavioral features: 13 dimensions
  - comment_count, avg_comment_length, max_comment_length
  - pct_night (posting 10pm-6am), pct_day
  - posting_hour_mean, posting_hour_std
  - days_active, unique_subreddits
  - exclaim_ratio, question_ratio, caps_ratio, emoji_like
- Linguistic markers: 6 dimensions
  - female_marker_count, male_marker_count
  - female_marker_density, male_marker_density
  - teen_marker_count, teen_marker_density

**Training:**
- Mild SMOTE-ENN resampling for class balance
- Cost-sensitive XGBoost (scale_pos_weight = 2.68)
- Threshold optimization at ~0.35 for balanced precision-recall

**Performance at Different Confidence Thresholds:**

| Threshold | Coverage | Accuracy | Female Recall | Male Recall |
|-----------|----------|----------|---------------|-------------|
| ≥ 0.50 | 100.0% | 94.8% | 88.4% | 97.2% |
| ≥ 0.55 | 96.9% | 96.0% | 90.6% | 97.9% |
| **≥ 0.60** | **92.7%** | **96.9%** | **92.1%** | **98.5%** |
| ≥ 0.70 | 82.3% | 98.1% | 93.4% | 99.4% |
| ≥ 0.80 | 67.4% | 98.6% | 93.9% | 99.7% |
| ≥ 0.90 | 44.0% | 99.4% | 97.1% | 99.9% |
| ≥ 0.95 | 27.7% | 99.8% | 98.1% | 100.0% |

### Age Predictor V3

**Features:** Same 803-dimensional feature set

**Training:**
- XGBoost with standard settings
- Binary classification (teen vs adult)
- Teen = 13-18, Adult = 19+

**Performance at Different Confidence Thresholds:**

| Threshold | Coverage | Accuracy | Teen Recall | Adult Recall |
|-----------|----------|----------|-------------|--------------|
| ≥ 0.50 | 100.0% | 93.7% | 96.5% | 90.2% |
| ≥ 0.55 | 98.0% | 94.2% | 96.4% | 91.5% |
| **≥ 0.60** | **96.5%** | **95.0%** | **97.2%** | **92.3%** |
| ≥ 0.70 | 90.0% | 97.1% | 98.7% | 95.0% |
| ≥ 0.80 | 87.4% | 98.0% | 99.1% | 96.6% |
| ≥ 0.85 | 84.5% | 98.7% | 100.0% | 97.1% |
| ≥ 0.90 | 83.2% | 99.2% | 100.0% | 98.2% |

## 2.3 AnthroScore System

**Measurement Approach:**
- Analyzes comment text for anthropomorphizing language patterns
- Produces continuous score (typically 0-6 range)
- Higher = more anthropomorphization

**Distribution:**
- 31.2% of users have AnthroScore > 0
- Mean (among non-zero): 1.93
- SD: 1.27
- Range: 0 to ~6

**Categories:**
- Zero: 68.8% of users
- Low (0-1): ~15%
- Medium (1-2.5): ~10%
- High (>2.5): ~6%

---

# SECTION 3: COMPLETE STATISTICAL RESULTS

## 3.1 Sample Characteristics (Analysis Sample)

**Using V3 @ ≥0.60 confidence threshold:**
- Total users with both predictions: 27,846
- Users with non-zero AnthroScore: 8,676 (31.2%)

## 3.2 RQ1: Demographics

### Age Distribution

| Age Group | Count | Percentage | 95% CI |
|-----------|-------|------------|--------|
| Teen (13-18) | 14,621 | 52.5% | [51.9%, 53.1%] |
| Adult (19+) | 13,225 | 47.5% | [46.9%, 48.1%] |

### Gender Distribution

| Gender | Count | Percentage | 95% CI |
|--------|-------|------------|--------|
| Male | 21,889 | 78.6% | [78.1%, 79.1%] |
| Female | 5,957 | 21.4% | [20.9%, 21.9%] |

### Age × Gender Crosstab

|  | Male | Female | Total |
|--|------|--------|-------|
| Teen | 11,442 | 3,179 | 14,621 |
| Adult | 10,447 | 2,778 | 13,225 |
| Total | 21,889 | 5,957 | 27,846 |

**Chi-square test:**
- χ² = 24.55
- df = 1
- p < 0.001
- Cramer's V = 0.030 (negligible)

## 3.3 RQ2: Demographics & Anthropomorphization

### Main Effect of Age

**Sample:** 8,676 users with non-zero AnthroScore

| Metric | Teen | Adult |
|--------|------|-------|
| N | 4,741 | 3,935 |
| Mean | 1.9373 | 1.9194 |
| SD | 1.2633 | 1.2852 |
| Median | 1.6000 | 1.5667 |

**Statistical Tests:**

| Test | Value | p-value |
|------|-------|---------|
| Welch's t | 0.651 | 0.515 |
| Cohen's d | 0.0141 | - |
| Hedges' g | 0.0141 | - |
| Glass's delta | 0.0139 | - |
| CLES (P[teen > adult]) | 0.509 | - |
| Point-biserial r | 0.007 | 0.515 |

**Bootstrap 95% CI for mean difference:** [-0.037, 0.071]

**Interpretation:** NEGLIGIBLE effect. No meaningful age difference in anthropomorphization.

### Main Effect of Gender

| Metric | Male | Female |
|--------|------|--------|
| N | 6,645 | 2,031 |
| Mean | 1.8964 | 2.0363 |
| SD | 1.2497 | 1.3420 |

**Statistical Tests:**

| Test | Value | p-value |
|------|-------|---------|
| Welch's t | -4.174 | < 0.001 |
| Cohen's d | -0.1099 | - |
| Hedges' g | -0.1099 | - |
| CLES (P[male > female]) | 0.470 | - |

**Interpretation:** SMALL effect. Females anthropomorphize significantly more than males.

### Age × Gender Two-Way ANOVA

| Effect | SS | df | F | p-value | η² |
|--------|----|----|---|---------|-----|
| Age | 0.89 | 1 | 0.53 | 0.465 | < 0.001 |
| Gender | 31.44 | 1 | 18.91 | < 0.001 | 0.002 |
| Age × Gender | 6.25 | 1 | 3.76 | 0.053 | < 0.001 |
| Residual | 14415.99 | 8672 | - | - | - |

**Model R² = 0.0027** (demographics explain 0.27% of variance)

### Subgroup Means

| Group | N | Mean | SD |
|-------|---|------|-----|
| Adult Female | 954 | 2.077 | 1.389 |
| Teen Female | 1,077 | 2.001 | 1.300 |
| Teen Male | 3,664 | 1.919 | 1.252 |
| Adult Male | 2,981 | 1.869 | 1.247 |

**Ranking:** Adult Female > Teen Female > Teen Male > Adult Male

### Pairwise Comparisons

| Comparison | Cohen's d | p-value | Significant |
|------------|-----------|---------|-------------|
| Teen Male vs Adult Male | +0.040 | 0.108 | No |
| Teen Female vs Adult Female | -0.056 | 0.206 | No |
| Teen Male vs Teen Female | -0.064 | 0.067 | No |
| **Adult Male vs Adult Female** | **-0.157** | **< 0.001** | **Yes** |

### Prevalence Analysis

**High anthropomorphization threshold:** AnthroScore ≥ 2.73 (top quartile)

| Group | Prevalence | 95% CI | Odds Ratio |
|-------|------------|--------|------------|
| Teen | 8.3% | [7.8%, 8.7%] | 1.15 (vs adult) |
| Adult | 7.3% | [6.8%, 7.7%] | ref |
| Female | 9.5% | [8.8%, 10.3%] | **1.33** (vs male) |
| Male | 7.3% | [7.0%, 7.7%] | ref |

### Ground Truth Validation

**Self-Declared Age Sample:** n = 286 (users with verified ages and non-zero AnthroScore)

| Group | N | Mean AnthroScore |
|-------|---|------------------|
| Self-Declared Teen (< 19) | 158 | 2.654 |
| Self-Declared Adult (≥ 19) | 128 | 3.052 |

| Metric | Value |
|--------|-------|
| Cohen's d | **-0.297** |
| t-statistic | -2.78 |
| p-value | 0.006 |
| Direction | **ADULTS HIGHER** |

**CRITICAL FINDING:** Self-declared ages show the OPPOSITE direction from predicted ages!

### Sensitivity Analysis: Age Effect Across Confidence Thresholds

| Threshold | N | Cohen's d | p-value | Significant |
|-----------|---|-----------|---------|-------------|
| ≥ 0.5 | 14,007 | +0.019 | 0.254 | No |
| ≥ 0.6 | 10,345 | +0.026 | 0.193 | No |
| ≥ 0.7 | 6,631 | +0.028 | 0.261 | No |
| ≥ 0.8 | 3,540 | -0.007 | 0.843 | No |
| ≥ 0.9 | 1,155 | **-0.140** | **0.028** | **Yes** |

**Pattern:** At highest confidence (≥0.9), the direction REVERSES to adults higher.

## 3.4 RQ3: Emotional Expression

### Sample
- Users with emotions and non-zero AnthroScore: 9,877

### Correlations with AnthroScore

| Emotion | r | p-value | Significant | Direction |
|---------|---|---------|-------------|-----------|
| Anger | +0.050 | < 0.001 | *** | Higher anthro = MORE anger |
| Sadness | -0.048 | < 0.001 | *** | Higher anthro = LESS sadness |
| Neutral | -0.048 | < 0.001 | *** | Higher anthro = LESS neutral |
| Fear | +0.046 | < 0.001 | *** | Higher anthro = MORE fear |
| Disgust | +0.033 | 0.001 | ** | Higher anthro = MORE disgust |
| Surprise | +0.013 | 0.212 | ns | - |
| Joy | +0.004 | 0.666 | ns | - |

### High vs Low Anthropomorphizers (Quartile Comparison)

| Emotion | Low (Q1) Mean | High (Q4) Mean | Cohen's d | p-value |
|---------|---------------|----------------|-----------|---------|
| Neutral | 0.455 | 0.425 | -0.126 | < 0.001 |
| Anger | 0.099 | 0.114 | +0.120 | < 0.001 |
| Sadness | 0.091 | 0.079 | -0.089 | 0.003 |
| Fear | 0.040 | 0.048 | +0.088 | 0.003 |
| Disgust | 0.096 | 0.105 | +0.075 | 0.011 |
| Surprise | 0.145 | 0.153 | +0.057 | 0.054 |
| Joy | 0.076 | 0.076 | +0.001 | 0.988 |

### Age Moderation of Emotion-Anthropomorphization

| Emotion | Teen r | Adult r | z-diff | p-diff | Moderation |
|---------|--------|---------|--------|--------|------------|
| Joy | +0.050 | -0.029 | 3.87 | < 0.001 | **Significant** |
| Sadness | -0.070 | -0.024 | -2.24 | 0.025 | **Significant** |
| Surprise | -0.009 | +0.035 | -2.13 | 0.033 | **Significant** |
| Anger | +0.033 | +0.066 | -1.63 | 0.103 | Not significant |
| Fear | +0.055 | +0.039 | 0.80 | 0.427 | Not significant |

**Key Finding:** Joy-anthropomorphization relationship is OPPOSITE for teens vs adults.

## 3.5 Multiple Regression

### Model 1: Demographics Only

| Predictor | B | SE | t | p | 95% CI |
|-----------|---|----|----|---|--------|
| Intercept | 1.869 | 0.021 | 89.7 | < 0.001 | [1.828, 1.910] |
| Is Teen | +0.020 | 0.027 | 0.73 | 0.465 | [-0.034, 0.074] |
| Is Female | +0.140 | 0.028 | 4.93 | < 0.001 | [0.084, 0.196] |

**R² = 0.0022** (0.22%)

### Model 2: Demographics with Interaction

| Predictor | B | p |
|-----------|---|---|
| Is Teen | +0.052 | 0.088 |
| Is Female | +0.218 | < 0.001 |
| Teen × Female | -0.125 | 0.053 |

**R² = 0.0027** (0.27%)

### Model 3: Full Model (Demographics + Emotions)

| Significant Predictors | B | p |
|------------------------|---|---|
| Is Female | +0.137 | < 0.001 |
| Anger | +0.426 | 0.021 |
| Fear | +0.598 | 0.010 |

**R² = 0.0047** (0.47%)

---

# SECTION 4: KEY FINDINGS SUMMARY

## 4.1 Confirmed Findings

1. **Gender predicts anthropomorphization (d = -0.11, p < 0.001)**
   - Females anthropomorphize more than males
   - Effect is small but statistically robust
   - Holds across all confidence thresholds

2. **Emotional expression patterns differ**
   - High anthropomorphizers: more anger, fear, disgust
   - High anthropomorphizers: less neutrality
   - All effects are negligible in magnitude (r < 0.10)

3. **Age moderates emotion-anthropomorphization relationship**
   - Joy positively correlates with anthropomorphization for teens
   - Joy shows no/negative correlation for adults
   - This moderation is statistically significant

## 4.2 Null Findings

1. **Age does NOT predict anthropomorphization (d = 0.01, p = 0.515)**
   - No meaningful difference between predicted teens and adults
   - Bootstrap CI includes zero
   - Negligible effect size

2. **Demographics explain almost no variance**
   - Full demographic model R² < 1%
   - Individual factors likely more important than demographics

## 4.3 Critical Discrepancy

**Predicted Age vs Self-Declared Age:**

| Measure | Direction | Cohen's d |
|---------|-----------|-----------|
| Predicted age | Teens ≈ Adults | +0.01 |
| Self-declared age | **Adults > Teens** | **-0.30** |

**Interpretation:** The age classifier captures "behavioral age" (teen-like online behavior), not chronological age. True chronological adults anthropomorphize MORE than true chronological teens.

---

# SECTION 5: THEORETICAL INTERPRETATION

## 5.1 The Digital Native Hypothesis (Challenged)

**Prior assumption:** Younger "digital natives" would anthropomorphize AI more due to lifelong digital immersion.

**Our finding:** No support for this hypothesis. If anything, chronological adults anthropomorphize more.

**Alternative explanation:** Adults may have more established relational schemas to map onto AI, more loneliness, or different motivations for AI companion use.

## 5.2 Gender and Relational Orientation

**Finding:** Females anthropomorphize more (d = -0.11)

**Theoretical alignment:** Consistent with research showing women have higher relational orientation on average (Cross & Madson, 1997).

**Caveat:** Selection effects may apply - women who engage in male-dominated AI companion communities may differ from general population.

## 5.3 The Behavioral Age Paradox

**The paradox:** 
- Behavioral patterns classified as "teen-like" (slang, posting times) → no relationship with anthropomorphization
- Actual chronological age (self-declared) → adults higher

**Resolution:** Age classifiers capture behavioral presentation, not developmental stage. Users who "act young" online are not the same as users who ARE young.

**Implication:** Computational social science must validate behavioral classifiers against ground truth.

## 5.4 Emotional Activation

**Pattern:** High anthropomorphizers show more active emotions (anger, fear) and less neutrality.

**Interpretation:** Anthropomorphization may indicate deeper psychological investment in AI relationships, manifesting as more intense emotional expression.

**Open question:** Does anthropomorphization cause emotional changes, or vice versa?

---

# SECTION 6: LIMITATIONS

## 6.1 Sample Limitations

- **Platform:** Reddit only (not representative of all AI companion users)
- **Language:** English only
- **Selection:** Only users who publicly discuss AI companions
- **Temporal:** 2024-2026 data; attitudes may evolve

## 6.2 Measurement Limitations

- **Age classifier:** Captures behavioral patterns, not verified age
- **AnthroScore:** Text-based; may miss behavioral anthropomorphization
- **Emotions:** Expressed, not felt (strategic presentation possible)
- **Ground truth sample:** Small (n = 286-459)

## 6.3 Statistical Limitations

- **Effect sizes:** All effects are small to negligible
- **R² values:** Demographics explain < 1% of variance
- **Causality:** Purely correlational design

---

# SECTION 7: FILE LOCATIONS

## 7.1 Data Files

```
Data/
├── raw/                                    # Original Reddit data
├── processed/
│   └── all_comments.parquet               # 283,895 comments
└── features/
    ├── user_anthroscores.parquet          # 47,062 users
    ├── user_emotions.parquet              # Emotion profiles
    ├── self_declarations.parquet          # Ground truth labels
    ├── user_subreddit_interactions.parquet
    └── ultimate_predictor/
        └── all_features.parquet           # 803-dim features
```

## 7.2 Model Files

```
experiments/v2_correction/
├── models/
│   ├── gender_v3/gender_predictor_v3.pkl  ← PRODUCTION
│   └── age_v3/age_predictor_v3.pkl        ← PRODUCTION
├── gender_predictions_v3.parquet          # 47,062 predictions
├── age_predictions_v3.parquet             # 47,062 predictions
└── FINAL_MODEL_SUMMARY.md
```

## 7.3 Results Files

```
results/
├── FINAL_MASTER_RESEARCH_DOCUMENT.md      # Main findings document
├── PHD_LEVEL_DISCUSSION.md                # Scholarly interpretation
├── FINAL_STATISTICAL_RESULTS.json         # Complete stats (JSON)
├── ADVANCED_ANALYSES_RESULTS.json         # Effect sizes, regression
└── COMPREHENSIVE_STATISTICAL_REPORT.md    # Tables
```

## 7.4 Script Files

```
scripts/
├── FINAL_COMPREHENSIVE_ANALYSIS.py        # Main analysis script
├── ADVANCED_ANALYSES.py                   # Regression, subgroups
└── [other analysis scripts]
```

---

# SECTION 8: USAGE EXAMPLES

## 8.1 Loading Predictions

```python
import pandas as pd

# Load V3 predictions
gender = pd.read_parquet('experiments/v2_correction/gender_predictions_v3.parquet')
age = pd.read_parquet('experiments/v2_correction/age_predictions_v3.parquet')

# Filter to high confidence
gender_hc = gender[gender['confidence'] >= 0.60]
age_hc = age[age['confidence'] >= 0.60]

print(f"Gender: {len(gender_hc)} users at 96.9% accuracy")
print(f"Age: {len(age_hc)} users at 95.0% accuracy")
```

## 8.2 Merging with AnthroScores

```python
anthro = pd.read_parquet('Data/features/user_anthroscores.parquet')

# Create analysis dataset
df = anthro.merge(gender_hc[['author', 'gender_predicted']], on='author')
df = df.merge(age_hc[['author', 'age_predicted']], on='author')

# Filter to non-zero anthropomorphization
df_nonzero = df[df['anthroscore_max'] > 0]
```

## 8.3 Running Effect Size Analysis

```python
from scipy.stats import ttest_ind
import numpy as np

teens = df_nonzero[df_nonzero['age_predicted'] == 'teen']['anthroscore_max']
adults = df_nonzero[df_nonzero['age_predicted'] == 'adult']['anthroscore_max']

# Cohen's d
pooled_std = np.sqrt(((len(teens)-1)*teens.var() + (len(adults)-1)*adults.var()) / 
                     (len(teens)+len(adults)-2))
d = (teens.mean() - adults.mean()) / pooled_std

t_stat, p_val = ttest_ind(teens, adults, equal_var=False)
print(f"Cohen's d = {d:.4f}, p = {p_val:.4f}")
```

---

# SECTION 9: CITATION

If using this research:

```
The Illusion Project (2026). Anthropomorphization of AI Companions Among Reddit Users.
Analysis conducted with V3 classification models (96.9% gender accuracy, 95.0% age accuracy).
Sample: 27,846 users with high-confidence predictions.
```

---

# SECTION 10: QUICK REFERENCE

## 10.1 One-Sentence Summary

**AI companion users are predominantly male (79%), with females showing higher anthropomorphization (d=-0.11); age differences are negligible when using predicted demographics but self-declared data suggests adults anthropomorphize more than teens.**

## 10.2 Key Numbers

| Metric | Value |
|--------|-------|
| Total users | 47,062 |
| Analysis sample | 27,846 |
| Users with AnthroScore > 0 | 8,676 |
| Gender accuracy (V3 @ 0.60) | 96.9% |
| Age accuracy (V3 @ 0.60) | 95.0% |
| Age effect (Cohen's d) | 0.014 |
| Gender effect (Cohen's d) | -0.110 |
| Ground truth age effect | -0.297 |
| Demographics R² | 0.27% |

## 10.3 Key Findings (Bullet Points)

- ✓ Females anthropomorphize MORE than males (small effect)
- ✗ Predicted age shows NO effect on anthropomorphization
- ⚠ Ground truth shows OPPOSITE direction (adults > teens)
- ✓ High anthropomorphizers express more anger, fear, disgust
- ✓ Age moderates joy-anthropomorphization relationship
- ⚠ Demographics explain < 1% of variance

---

*Document generated: January 10, 2026*
*Total tokens: ~12,000*
*Purpose: Complete context for AI analysis*
