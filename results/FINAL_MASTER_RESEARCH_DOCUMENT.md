# The Illusion Project: Final Research Findings

## Anthropomorphization of AI Companions Among Reddit Users

**Generated:** January 10, 2026  
**Version:** FINAL - V3 Models with Confidence Filtering  
**Classification Accuracy:** 96.9% (Gender), 95.0% (Age) at ≥0.60 confidence threshold

---

# Executive Summary

## Overview

This study examines how Reddit users **anthropomorphize AI companions** (treat AI as human-like) and how this relates to demographics and emotional expression patterns. Using machine learning classifiers validated at 95-97% accuracy, we analyzed 47,062 unique users who discuss AI companion apps.

## Key Findings

| Finding | Statistic | Effect Size | p-value | Interpretation |
|---------|-----------|-------------|---------|----------------|
| **Gender predicts anthropomorphization** | t = -4.17 | d = -0.11 | < 0.001 | Females anthropomorphize more |
| **Age effect is negligible** | t = 0.65 | d = 0.01 | 0.515 | No meaningful difference |
| **Ground truth contradicts model direction** | d = -0.30 | small | 0.006 | Self-declared adults anthropomorphize more |
| **Anger correlates with anthropomorphization** | r = 0.050 | negligible | < 0.001 | Higher anthropomorphizers express more anger |
| **Neutrality inversely correlated** | r = -0.048 | negligible | < 0.001 | Higher anthropomorphizers less neutral |

## The Core Narrative

**AI companion users are predominantly male (78.6%) with a roughly even age split (52.5% predicted teen, 47.5% adult).** Females show significantly higher anthropomorphization than males, though the effect is small (d = -0.11). Critically, **the relationship between age and anthropomorphization depends on measurement method**: predicted age shows negligible differences, while self-declared age (ground truth) shows adults anthropomorphize more than teens.

High anthropomorphizers express **more anger, fear, and disgust** and **less neutrality** in their discussions about AI companions.

---

# Methodology

## Data Sources

| Source | N | Description |
|--------|---|-------------|
| Reddit Comments | 283,895 | Comments from AI companion subreddits |
| Unique Users | 47,062 | Users who posted about AI companions |
| Users with Labels (Gender) | 4,894 | Self-declared gender for validation |
| Users with Labels (Age) | 459 | Self-declared age for validation |

## Classification System: V3 Models

### Model Performance Summary

| Model | Version | Overall Accuracy | Minority Recall | Confidence Threshold |
|-------|---------|------------------|-----------------|----------------------|
| Gender | V3 | **96.9%** | 92.1% (female) | ≥ 0.60 |
| Age | V3 | **95.0%** | 97.2% (teen) | ≥ 0.60 |

### Technical Architecture

**Gender Predictor V3:**
- Features: Sentence-BERT embeddings (384D) + Subreddit patterns (400D) + Behavioral (13D) + Linguistic markers (6D)
- Training: Mild SMOTE-ENN resampling, cost-sensitive XGBoost
- Threshold optimization for balanced precision-recall

**Age Predictor V3:**
- Features: Full feature set including text embeddings
- Training: XGBoost with standard settings
- Binary classification (teen vs adult)

### Confidence Filtering

At ≥0.60 confidence threshold:
- Gender: 82.0% of users retained (38,588 / 47,062)
- Age: 72.1% of users retained (33,909 / 47,062)
- Combined: 27,846 users with both predictions

---

# Research Question 1: Demographics

## Who uses AI companions?

### Age Distribution (V3 @ ≥0.60 confidence)

| Age Group | Count | Percentage | 95% CI |
|-----------|-------|------------|--------|
| Teen (13-18) | 14,621 | 52.5% | [51.9%, 53.1%] |
| Adult (19+) | 13,225 | 47.5% | [46.9%, 48.1%] |

**Key Finding:** The age distribution is more balanced than previously estimated. Earlier models with lower accuracy showed 81% teen; with improved V3 models, the split is approximately 52-48%.

### Gender Distribution (V3 @ ≥0.60 confidence)

| Gender | Count | Percentage | 95% CI |
|--------|-------|------------|--------|
| Male | 21,889 | 78.6% | [78.1%, 79.1%] |
| Female | 5,957 | 21.4% | [20.9%, 21.9%] |

**Key Finding:** AI companion users are predominantly male (~4:1 ratio).

### Age × Gender Association

| Test | Value |
|------|-------|
| Chi-square | 24.55 |
| p-value | < 0.001 |
| Cramer's V | 0.030 (negligible) |

**Interpretation:** While statistically significant (due to large sample size), the association between age and gender is negligible in practical terms.

---

# Research Question 2: Demographics & Anthropomorphization

## How do demographics relate to treating AI as human-like?

### Sample

- Users with non-zero AnthroScore: 8,676 (31.2% of sample)
- Mean AnthroScore: 1.93 (SD = 1.27)

### Age Effect

| Metric | Teen | Adult | Difference |
|--------|------|-------|------------|
| N | 4,741 | 3,935 | - |
| Mean AnthroScore | 1.937 | 1.919 | +0.018 |
| SD | 1.263 | 1.285 | - |

| Statistical Test | Value |
|-----------------|-------|
| Welch's t | 0.651 |
| p-value | 0.515 |
| Cohen's d | 0.014 (negligible) |
| Hedges' g | 0.014 (negligible) |
| CLES (P[teen > adult]) | 0.509 |
| Point-biserial r | 0.007 |
| 95% Bootstrap CI | [-0.037, 0.071] |

**Key Finding:** There is **no meaningful age difference** in anthropomorphization. The effect size is negligible (d = 0.014), and the bootstrap confidence interval includes zero.

### Gender Effect

| Metric | Male | Female | Difference |
|--------|------|--------|------------|
| N | 6,645 | 2,031 | - |
| Mean AnthroScore | 1.896 | 2.036 | -0.140 |
| SD | 1.250 | 1.342 | - |

| Statistical Test | Value |
|-----------------|-------|
| Welch's t | -4.17 |
| p-value | < 0.001 |
| Cohen's d | -0.110 (small) |
| Hedges' g | -0.110 (small) |
| CLES (P[male > female]) | 0.470 |

**Key Finding:** **Females anthropomorphize significantly more than males**, though the effect is small (d = -0.11). Females have a 53% probability of scoring higher than males on any random comparison.

### Age × Gender Interaction (Two-Way ANOVA)

| Effect | F | p-value | η² | Significant |
|--------|---|---------|-----|-------------|
| Age | 0.53 | 0.465 | < 0.001 | No |
| Gender | 18.91 | < 0.001 | 0.002 | **Yes** |
| Age × Gender | 3.76 | 0.053 | < 0.001 | Borderline |

**R-squared:** 0.27% (demographics explain very little variance)

### Subgroup Analysis

| Group | N | Mean | SD |
|-------|---|------|-----|
| Adult Female | 954 | 2.077 | 1.389 |
| Teen Female | 1,077 | 2.001 | 1.300 |
| Teen Male | 3,664 | 1.919 | 1.252 |
| Adult Male | 2,981 | 1.869 | 1.247 |

**Ranking:** Adult Female > Teen Female > Teen Male > Adult Male

**Key Pairwise Comparison:**
- Adult Male vs Adult Female: d = -0.157, p < 0.001 (**significant**)
- All other comparisons: not significant

### Prevalence of High Anthropomorphization

Using top quartile threshold (AnthroScore ≥ 2.73):

| Group | Prevalence | 95% CI | Odds Ratio |
|-------|------------|--------|------------|
| Teen | 8.3% | [7.8%, 8.7%] | 1.15 (vs adult) |
| Adult | 7.3% | [6.8%, 7.7%] | ref |
| Female | 9.5% | [8.8%, 10.3%] | **1.33 (vs male)** |
| Male | 7.3% | [7.0%, 7.7%] | ref |

**Key Finding:** Females are 33% more likely to be high anthropomorphizers than males (OR = 1.33).

---

# Critical Validity Check: Ground Truth Validation

## The Discrepancy

| Measurement | Direction | Cohen's d | p-value |
|-------------|-----------|-----------|---------|
| Predicted Age | Teens slightly higher | +0.014 | 0.515 |
| Self-Declared Age | **Adults higher** | **-0.297** | 0.006 |

### Self-Declared Age Analysis (n = 286)

| Group | N | Mean AnthroScore |
|-------|---|------------------|
| Self-Declared Teen (< 19) | 158 | 2.654 |
| Self-Declared Adult (≥ 19) | 128 | **3.052** |

**Critical Finding:** When using verified self-declared ages, **adults anthropomorphize MORE than teens** (d = -0.30, small effect). This contradicts the direction suggested by our age classifier.

### Interpretation

The age predictor captures **behavioral patterns** that correlate with teen-like online behavior (e.g., slang usage, posting times), not necessarily chronological age. Users who "behave like teens" may anthropomorphize slightly more, but **actual chronological adults anthropomorphize more than actual teens**.

### Methodological Implication

Claims about "teen" vs "adult" anthropomorphization should be interpreted as claims about **predicted behavioral age group**, not verified chronological age. For analyses requiring true age information, self-declared samples should be used.

---

# Research Question 3: Emotional Expression & Anthropomorphization

## How do emotional patterns relate to anthropomorphization?

### Sample
- Users with emotions and non-zero AnthroScore: 9,877

### Correlations with AnthroScore

| Emotion | r | p-value | Direction | Interpretation |
|---------|---|---------|-----------|----------------|
| Anger | +0.050 | < 0.001 | Higher anthropomorphizers express MORE anger | Negligible |
| Fear | +0.046 | < 0.001 | Higher anthropomorphizers express MORE fear | Negligible |
| Disgust | +0.033 | 0.001 | Higher anthropomorphizers express MORE disgust | Negligible |
| Sadness | -0.048 | < 0.001 | Higher anthropomorphizers express LESS sadness | Negligible |
| Neutral | -0.048 | < 0.001 | Higher anthropomorphizers are LESS neutral | Negligible |
| Surprise | +0.013 | 0.212 | Not significant | - |
| Joy | +0.004 | 0.666 | Not significant | - |

### High vs Low Anthropomorphizers (Quartile Comparison)

| Emotion | Low Anthro (Q1) | High Anthro (Q4) | Cohen's d | p-value |
|---------|-----------------|------------------|-----------|---------|
| Neutral | 0.455 | 0.425 | -0.126 | < 0.001 |
| Anger | 0.099 | 0.114 | +0.120 | < 0.001 |
| Sadness | 0.091 | 0.079 | -0.089 | 0.003 |
| Fear | 0.040 | 0.048 | +0.088 | 0.003 |
| Disgust | 0.096 | 0.105 | +0.075 | 0.011 |

**Key Finding:** High anthropomorphizers express more **anger, fear, and disgust** while being **less neutral** in their discussions.

### Age Moderation of Emotion-Anthropomorphization Relationship

| Emotion | Teen r | Adult r | Difference p | Significant? |
|---------|--------|---------|--------------|--------------|
| Joy | +0.050 | -0.029 | < 0.001 | **Yes** |
| Sadness | -0.070 | -0.024 | 0.025 | **Yes** |
| Surprise | -0.009 | +0.035 | 0.033 | **Yes** |
| Anger | +0.033 | +0.066 | 0.103 | No |
| Fear | +0.055 | +0.039 | 0.427 | No |

**Key Finding:** The relationship between emotions and anthropomorphization differs by age group:
- **For teens:** Joy is positively correlated with anthropomorphization
- **For adults:** Joy shows no relationship or slight negative correlation
- This moderation effect is statistically significant (p < 0.001)

---

# Multiple Regression Analysis

## Predicting Anthropomorphization

### Model 1: Demographics Only

| Predictor | B | 95% CI | p-value |
|-----------|---|--------|---------|
| Intercept | 1.869 | - | < 0.001 |
| Is Teen | +0.020 | [-0.034, 0.074] | 0.465 |
| Is Female | +0.140 | [0.084, 0.196] | < 0.001 |

**R² = 0.22%** (demographics explain very little variance)

### Model 2: Demographics with Interaction

| Predictor | B | p-value |
|-----------|---|---------|
| Is Teen | +0.052 | 0.088 |
| Is Female | +0.218 | < 0.001 |
| Teen × Female | -0.125 | 0.053 (borderline) |

**R² = 0.27%**

**Interpretation:** There is a marginally significant interaction: the female advantage in anthropomorphization may be smaller for teens than adults.

### Model 3: Full Model (Demographics + Emotions)

| Significant Predictors | B | p-value |
|------------------------|---|---------|
| Is Female | +0.137 | < 0.001 |
| Anger | +0.426 | 0.021 |
| Fear | +0.598 | 0.010 |

**R² = 0.47%**

**Interpretation:** Even controlling for emotional expression, females still show higher anthropomorphization. Anger and fear independently predict higher anthropomorphization.

---

# Summary of Key Findings

## What We Found

1. **Gender matters, age doesn't (for predicted demographics)**
   - Females anthropomorphize significantly more than males (d = -0.11)
   - Predicted age shows no meaningful difference (d = 0.01)

2. **Ground truth reveals opposite age pattern**
   - Self-declared adults actually anthropomorphize more than teens (d = -0.30)
   - This suggests the age classifier captures behavioral patterns, not chronological age

3. **Emotional expression patterns differ**
   - High anthropomorphizers express more anger, fear, and disgust
   - They are less emotionally neutral
   - Age moderates the joy-anthropomorphization relationship

4. **Demographics explain very little variance**
   - R² < 1% for all demographic models
   - Other unmeasured factors drive anthropomorphization

## Effect Size Summary

| Finding | Cohen's d / r | Interpretation |
|---------|---------------|----------------|
| Gender → Anthro | d = -0.11 | Small |
| Age → Anthro (predicted) | d = 0.01 | Negligible |
| Age → Anthro (self-declared) | d = -0.30 | Small |
| Anger → Anthro | r = 0.05 | Negligible |
| Neutral → Anthro | r = -0.05 | Negligible |

---

# Limitations

## Methodological Limitations

1. **Classification vs. Ground Truth**
   - Age classifier captures behavioral patterns, not verified chronological age
   - Self-declared age sample is small (n = 286) and may be biased

2. **Effect Sizes**
   - Most effects are negligible to small (d < 0.30)
   - Demographics explain < 1% of variance in anthropomorphization

3. **Observational Design**
   - Cannot establish causation
   - Confounds not controlled

4. **Platform Bias**
   - Reddit users only
   - English language only
   - Users who publicly discuss AI companions

## Sample Characteristics

- Predominantly male (78.6%)
- Roughly balanced age (52.5% teen, 47.5% adult based on predictions)
- Only 31.2% of users have non-zero AnthroScore

---

# Implications & Future Directions

## Theoretical Implications

1. **Gender differences in AI relationships**
   - Women may form more human-like relationships with AI companions
   - This could reflect broader patterns in relationship formation

2. **The behavioral vs. chronological age paradox**
   - Users who "act young" online may anthropomorphize more
   - But actual older users anthropomorphize more than actual young users
   - This suggests complex developmental and generational factors

3. **Emotional correlates**
   - Anthropomorphization is associated with more active (angry, fearful) emotional expression
   - This may indicate deeper psychological investment

## Practical Implications

1. **AI companion design**
   - Consider gender differences in user experience
   - Age-based targeting may be less relevant than behavioral patterns

2. **AI safety**
   - Users who anthropomorphize more show different emotional patterns
   - This may have implications for user well-being

## Future Research

1. **Longitudinal studies** to establish temporal relationships
2. **Qualitative analysis** of high anthropomorphizers
3. **Cross-platform validation** (Discord, TikTok, etc.)
4. **Mental health correlates** of anthropomorphization patterns
5. **Cultural variation** in AI companion relationships

---

# Technical Appendix

## Files Generated

```
results/
├── FINAL_STATISTICAL_RESULTS.json        # Complete statistical output
├── ADVANCED_ANALYSES_RESULTS.json        # Additional analyses
├── COMPREHENSIVE_STATISTICAL_REPORT.md   # Intermediate report
└── FINAL_MASTER_RESEARCH_DOCUMENT.md     # This document

experiments/v2_correction/
├── models/
│   ├── gender_v3/gender_predictor_v3.pkl
│   └── age_v3/age_predictor_v3.pkl
├── gender_predictions_v3.parquet
├── age_predictions_v3.parquet
└── FINAL_MODEL_SUMMARY.md
```

## Statistical Tools

- Python 3.10+
- scipy.stats for t-tests, ANOVA, chi-square
- statsmodels for regression, effect sizes
- Bootstrap: 5,000 iterations for confidence intervals

## Citation

If using this research, please cite:

> The Illusion Project (2026). Anthropomorphization of AI Companions Among Reddit Users. 
> Analysis conducted with V3 classification models at 95-97% accuracy.

---

*Document generated: January 10, 2026*  
*Analysis sample: 27,846 users with V3 predictions @ 0.60 confidence*
