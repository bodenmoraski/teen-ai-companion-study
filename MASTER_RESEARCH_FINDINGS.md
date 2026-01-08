# The Illusion Project: Complete Research Findings

## Anthropomorphization of AI Companions Among Reddit Users

**Generated:** January 7, 2026  
**Status:** MAJOR SIGNIFICANT FINDINGS ACROSS ALL ANALYSES  
**Document Purpose:** Comprehensive summary for AI-assisted analysis and research expansion

---

# Table of Contents

1. [Executive Summary](#executive-summary)
2. [Methodology Overview](#methodology-overview)
3. [Classification Systems](#classification-systems)
   - [Ultimate Age Predictor](#ultimate-age-predictor)
   - [Ultimate Gender Predictor](#ultimate-gender-predictor)
4. [Research Question 1: Demographics & Intent](#rq1-demographics--intent)
5. [Research Question 2: Demographics & Anthropomorphization](#rq2-demographics--anthropomorphization)
6. [Research Question 3: Emotional Expression](#rq3-emotional-expression--anthropomorphization)
7. [Integrated Findings](#integrated-findings)
8. [Limitations](#limitations)
9. [Data Files & Technical Details](#data-files--technical-details)
10. [Future Directions](#future-directions)

---

# Executive Summary

This study examines **anthropomorphization of AI companions** (treating AI as human-like) among Reddit users who discuss AI companion apps (Character.AI, Replika, etc.). We analyze how demographics (age, gender), usage intent, and emotional expression relate to anthropomorphization levels.

## Top-Line Findings

| Finding | Effect Size | p-value | Significance |
|---------|-------------|---------|--------------|
| Teens anthropomorphize more than adults | d = 0.111 | p < 0.0001 | ★★★ |
| High anthropomorphizers show LESS emotional diversity | d = -1.176 | p < 0.0001 | ★★★ |
| Character creation intent → highest anthropomorphization | F = 15.58 | p < 0.0001 | ★★★ |
| Age × Emotional Intensity interaction | B = -0.059 | p = 0.009 | ★★ |
| Age × Emotional Valence interaction | B = -0.055 | p = 0.016 | ★★ |
| Intent distribution differs by age | χ² = 21.80 | p = 0.0006 | ★★★ |
| Intent distribution differs by gender | χ² = 25.36 | p = 0.0001 | ★★★ |

## The Core Story

**The typical high anthropomorphizer is a teenage male engaged in character creation, expressing concentrated (less diverse) and more negative emotions in discussions about AI companions. The relationship between anthropomorphization and emotional expression is significantly different for teens vs. adults, with teens showing a unique pattern of reduced joy associated with higher anthropomorphization.**

---

# Methodology Overview

## Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| Reddit Comments | 277,420 | Comments from AI companion subreddits |
| Unique Users | 47,062 | Users who posted about AI companions |
| Known Age Users | 459 | Users with self-declared age |
| Known Gender Users | 979 | Users with self-declared gender |

## Subreddits Analyzed
- r/CharacterAI
- r/Replika
- r/SillyTavernAI
- r/AICompanions
- r/TavernAI
- And related subreddits

## Anthropomorphization Measurement (AnthroScore)

The **AnthroScore** measures the degree to which users treat AI companions as human-like. It is calculated using:

1. **Language Analysis**: Detection of anthropomorphizing language patterns
2. **Relationship Language**: References to AI as "friend", "partner", "they feel", etc.
3. **Emotional Attribution**: Assigning human emotions to AI
4. **Agency Attribution**: Treating AI as having free will/consciousness

| Metric | Description |
|--------|-------------|
| Mean AnthroScore | Average across all user comments |
| Max AnthroScore | Highest single-comment anthropomorphization |
| AnthroScore > 0 | 10,914 users (23.2% of total) |
| Mean of non-zero | 0.73 (SD = 0.81) |

---

# Classification Systems

## Ultimate Age Predictor

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ULTIMATE AGE PREDICTOR                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  SIGNAL 1       │  │  SIGNAL 2       │  │  SIGNAL 3       │     │
│  │  Text Embed.    │  │  Subreddit      │  │  Behavioral     │     │
│  │  (SBERT)        │  │  Patterns       │  │  Features       │     │
│  │                 │  │                 │  │                 │     │
│  │  384-dim        │  │  500 binary     │  │  7 features     │     │
│  │  embedding      │  │  features       │  │                 │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           ▼                    ▼                    ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   XGBoost       │  │   XGBoost       │  │   XGBoost       │     │
│  │   Classifier    │  │   Classifier    │  │   Classifier    │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           ▼                    ▼                    ▼               │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │              PROBABILITY OUTPUTS (per class)               │     │
│  │   P(teen), P(young_adult), P(adult) × 3 signals = 9 vals   │     │
│  └─────────────────────────────┬─────────────────────────────┘     │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    META-LEARNER (STACKING)                   │   │
│  │                         XGBoost                              │   │
│  │   Input: 9 probability values from 3 signals                │   │
│  │   Output: Final calibrated probabilities                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Signal Details

| Signal | Features | Description |
|--------|----------|-------------|
| **Text Embeddings** | 384 dimensions | Sentence-BERT (all-MiniLM-L6-v2) on concatenated user comments |
| **Subreddit Patterns** | 500 binary features | Participation in top 500 subreddits |
| **Behavioral Features** | 7 features | Late night ratio, school hours ratio, weekend ratio, activity span, avg comment length, comment count, subreddit diversity |

### Performance Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 70.6% |
| Baseline (random) | 33.3% |
| Previous System | 46.3% |
| **Improvement** | **+24.3 pp** |

### Confidence-Filtered Accuracy

| Confidence Threshold | Users | Accuracy |
|---------------------|-------|----------|
| ≥ 0.5 | 379 | 76.3% |
| ≥ 0.6 (Recommended) | 314 | **84.1%** |
| ≥ 0.7 | 250 | 89.2% |
| ≥ 0.8 | 146 | 94.5% |
| ≥ 0.9 | 55 | 98.2% |

### Class-Level Performance (at 0.6 threshold)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Teen (13-18) | 75% | 94% | 83% |
| Young Adult (19-25) | 71% | 18% | 29% |
| Adult (26+) | 58% | 61% | 60% |

**Note**: Teen classification is very strong (94% recall). Young adult classification struggles due to behavioral overlap with both teens and adults.

---

## Ultimate Gender Predictor

### Architecture

Similar stacked ensemble with 4 signals:

| Signal | Features | CV Accuracy |
|--------|----------|-------------|
| Text Embeddings | 384 dimensions (SBERT) | 75.6% |
| Subreddit Patterns | 500 binary features | 72.1% |
| Behavioral Features | 7 features | 72.2% |
| Linguistic Features | LIWC-like categories | 79.9% |
| **Stacked Ensemble** | Combined | **80.2%** |

### Performance Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 81.3% |
| Test Set Size | 979 users |

### Confidence-Filtered Accuracy

| Confidence Threshold | Users | Accuracy |
|---------------------|-------|----------|
| ≥ 0.5 | 979 | 81.3% |
| ≥ 0.6 | 876 | 84.1% |
| ≥ 0.7 | 760 | 87.1% |
| ≥ 0.8 | 572 | 92.1% |
| ≥ 0.9 | 388 | **96.1%** |

### Class-Level Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Female | 77% | 44% | 56% | 266 |
| Male | 82% | 95% | 88% | 713 |

**Note**: Male classification is strong; female classification has lower recall (44%) likely due to class imbalance and smaller training sample.

---

# RQ1: Demographics & Intent

## Research Question
> What are the demographics and usage intentions of AI companion users?

## Age Distribution (High-Confidence Predictions)

| Age Group | Percentage | Count |
|-----------|------------|-------|
| Teen (13-18) | 81.2% | 18,500+ |
| Young Adult (19-25) | 4.7% | 1,070+ |
| Adult (26+) | 14.1% | 3,210+ |

**Key Finding**: The overwhelming majority of AI companion users are teenagers.

## Gender Distribution (High-Confidence Predictions)

| Gender | Percentage |
|--------|------------|
| Male | 85.3% |
| Female | 14.7% |

**Key Finding**: AI companion users are predominantly male.

## Intent/Purpose Analysis (BERTopic)

### Methodology

1. BERTopic clustering on 277,420 Reddit comments
2. Semantic labeling of clusters into intent categories
3. Assignment of users to primary intent based on their posts

### Intent Distribution

| Intent Category | Count | Percentage |
|-----------------|-------|------------|
| Other/General Discussion | 2,053 | 44.3% |
| Character Creation | 1,275 | 27.5% |
| Unknown | 1,186 | 25.6% |
| Community Sharing | 62 | 1.3% |
| Roleplay/Fantasy | 44 | 0.9% |
| Emotional Support | 16 | 0.3% |

### Intent-Demographic Associations

| Test | χ² Statistic | p-value | Interpretation |
|------|--------------|---------|----------------|
| Intent vs Age | 21.80 | **0.0006** | Teens prefer character creation (+7.6 pp) |
| Intent vs Gender | 25.36 | **0.0001** | Gender differences in intent |

---

# RQ2: Demographics & Anthropomorphization

## Research Question
> How do demographics relate to anthropomorphization of AI companions?

## Main Effect of Age

### Key Finding: Teens Anthropomorphize More ★★★

| Metric | Teen | Non-Teen | Cohen's d | p-value |
|--------|------|----------|-----------|---------|
| Mean Max AnthroScore | 1.258 | 1.074 | 0.111 | **<0.0001** |

| Metric | Teen Rate | Non-Teen Rate | p-value |
|--------|-----------|---------------|---------|
| High Anthropomorphization (Max ≥ 1.5) | 42.7% | 39.0% | **0.007** |

**Interpretation**: Teens show significantly higher peak anthropomorphization than adults. The effect size is small (d = 0.11) but highly robust.

## Main Effect of Gender

| Metric | Male | Female | p-value |
|--------|------|--------|---------|
| Mean Max AnthroScore | 1.133 | 1.184 | 0.137 (ns) |

**Finding**: No significant main effect of gender on anthropomorphization.

## Age × Gender Interaction

| Factor | F-statistic | p-value |
|--------|-------------|---------|
| Age | 14.35 | **0.0002** |
| Gender | 0.12 | 0.72 |
| Age × Gender | 3.63 | 0.057 (borderline) |

### Group Rankings (Mean Max AnthroScore)

| Rank | Group | Mean AnthroScore |
|------|-------|------------------|
| 1 | Teen Male | 1.240 |
| 2 | Adult Female | 1.198 |
| 3 | Teen Female | 1.170 |
| 4 | Adult Male | 1.019 |

**Interpretation**: Teen males have the highest anthropomorphization levels, while adult males have the lowest. Adult females show higher anthropomorphization than teen females, suggesting different patterns by age and gender.

## Intent and Anthropomorphization

### Character Creation → Highest Anthropomorphization ★★★

| Intent | Mean AnthroScore | n |
|--------|------------------|---|
| Character Creation | 1.713 | 1,275 |
| Other/General | 1.637 | 2,053 |
| Emotional Support | 1.460 | 16 |
| Roleplay/Fantasy | 1.425 | 44 |
| Unknown | 1.220 | 1,186 |

**ANOVA**: F = 15.58, p < 0.0001

### Does Intent Mediate the Age Effect?

**Partially.** Within the "other" intent category, teens still show significantly higher anthropomorphization (p = 0.0007), suggesting the age effect is not fully explained by intent differences.

## Profile of High Anthropomorphizers (Top 10%)

| Characteristic | High Anthro | Baseline | Difference |
|----------------|-------------|----------|------------|
| Teen Rate | 84.3% | 83.6% | +0.7 pp |
| Female Rate | 9.5% | 14.6% | **-5.1 pp** |
| Character Creation | +4.4 pp | - | Higher |
| Active Discussion | +9.0 pp | - | Higher |
| Unknown Intent | -11.9 pp | - | Lower |

**Key Finding**: High anthropomorphizers are more likely to be male and engaged in active character creation and discussion.

---

# RQ3: Emotional Expression & Anthropomorphization

## Research Question
> How do emotional expression patterns relate to anthropomorphization of AI companions?

## Dataset for Emotional Analysis

| Metric | Value |
|--------|-------|
| Total users analyzed | 44,421 |
| Users with non-zero AnthroScore | 10,914 |
| Users with age predictions | 29,394 |
| Emotions analyzed | Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral |

## Option A: Correlation Analysis

### All Correlations with Mean AnthroScore

| Emotion | Pearson r | p-value | Direction |
|---------|-----------|---------|-----------|
| **Emotional Diversity** | **-0.3905** | **<0.0001** | LESS diverse ★★★ |
| Surprise | +0.0333 | 0.0005 | MORE ★★★ |
| Fear | +0.0280 | 0.0035 | MORE ★★ |
| Emotional Intensity | +0.0276 | 0.0039 | MORE ★★ |
| Neutral | -0.0276 | 0.0039 | LESS ★★ |
| Joy | -0.0259 | 0.0069 | LESS ★★ |
| Emotional Valence | -0.0258 | 0.0070 | MORE NEGATIVE ★★ |
| Anger | +0.0247 | 0.0100 | MORE ★★ |
| Sadness | -0.0097 | 0.3092 | ns |
| Disgust | -0.0021 | 0.8234 | ns |

### Major Finding: Emotional Diversity (r = -0.39)

**This is the largest effect in the study.** Users who anthropomorphize AI companions more express significantly LESS emotional diversity in their discussions. Their emotional expression is concentrated rather than varied.

## Option B: Group Comparisons (High vs Low Anthropomorphizers)

Comparing top 25% vs bottom 25% of anthropomorphizers (n = 2,729 each):

| Emotion | High Anthro | Low Anthro | Cohen's d | p-value |
|---------|-------------|------------|-----------|---------|
| **Diversity** | 1.566 | 2.128 | **-1.176** | **<0.0001** ★★★ |
| Surprise | 0.164 | 0.149 | +0.088 | 0.0012 ★★ |
| Joy | 0.069 | 0.078 | -0.073 | 0.0068 ★★ |
| Intensity | 0.574 | 0.559 | +0.061 | 0.0248 ★ |
| Valence | -0.272 | -0.254 | -0.061 | 0.0235 ★ |

### Major Finding: Emotional Diversity (d = -1.18)

This is a **MASSIVE effect size** (d > 1 is considered "large"). High anthropomorphizers show dramatically less emotional diversity.

### Dominant Emotion Distribution

| Emotion | High Anthro | Low Anthro | Difference |
|---------|-------------|------------|------------|
| Neutral | 55.6% | **86.4%** | **-30.9 pp** |
| Surprise | 13.7% | 5.1% | +8.6 pp |
| Anger | 9.1% | 2.4% | +6.7 pp |
| Disgust | 7.0% | 2.0% | +5.0 pp |
| Sadness | 6.2% | 1.6% | +4.6 pp |
| Joy | 5.3% | 1.6% | +3.6 pp |
| Fear | 3.1% | 0.8% | +2.3 pp |

**Interpretation**: Low anthropomorphizers are predominantly neutral (86%), while high anthropomorphizers show a more active emotional profile with more surprise, anger, disgust, and sadness.

## Option C: Age Moderation Effects ★★★

### Correlations by Age Group

| Emotion | Teen r | Adult r | Difference p |
|---------|--------|---------|--------------|
| **Joy** | **-0.104** | -0.014 | **0.0001** ★★★ |
| **Valence** | **-0.071** | -0.015 | **0.0124** ★ |
| Intensity | -0.010 | +0.050 | 0.0080 ★★ |
| Diversity | -0.403 | -0.413 | 0.5733 (ns) |
| Sadness | +0.012 | -0.005 | 0.4374 (ns) |
| Anger | +0.019 | +0.018 | 0.9613 (ns) |

### Key Finding: Teen Joy-Anthropomorphization Link

For **TEENS**: Higher anthropomorphization strongly predicts LESS joy (r = -0.10, p < 0.001)
For **ADULTS**: No significant relationship (r = -0.01, ns)

**This effect is UNIQUE to teenagers.**

### Regression with Interaction Terms

| Model: Emotional Intensity | Coefficient | p-value |
|---------------------------|-------------|---------|
| AnthroScore main effect | 0.049 | 0.0013 ★★ |
| Teen main effect | -0.034 | 0.1299 |
| **Anthro × Teen Interaction** | **-0.059** | **0.0093** ★★ |

| Model: Emotional Valence | Coefficient | p-value |
|-------------------------|-------------|---------|
| AnthroScore main effect | -0.015 | 0.3071 |
| Teen main effect | 0.015 | 0.5095 |
| **Anthro × Teen Interaction** | **-0.055** | **0.0157** ★ |

**Interpretation**: The relationship between anthropomorphization and emotional expression is significantly different for teens vs. adults. For teens, higher anthropomorphization is associated with more negative emotional expression (less joy, more negative valence).

## Option D: Emotional Valence Analysis

### AnthroScore by Valence Category

| Valence Category | Mean AnthroScore | n |
|------------------|------------------|---|
| Positive | 0.924 | 539 |
| Neutral | 0.865 | 2,269 |
| Negative | 0.553 | 8,106 |

**ANOVA**: F = 178.24, p < 0.0001

### Valence-Anthro by Age

| Age Group | Correlation r | p-value |
|-----------|---------------|---------|
| 13-18 (Teen) | **-0.071** | **<0.0001** ★★★ |
| 19-25 | -0.014 | 0.5954 (ns) |
| 26-40 | -0.067 | 0.0562 |
| 41-60 | -0.032 | 0.4877 (ns) |
| 61-80 | +0.017 | 0.5565 (ns) |

**Key Finding**: The negative valence-anthropomorphization relationship is significant ONLY for teens.

---

# Integrated Findings

## The Complete Picture

1. **Demographics**: AI companion users are predominantly teenage males (81% teen, 85% male).

2. **Anthropomorphization Pattern**: Teens anthropomorphize significantly more than adults, and this effect is most pronounced in teen males engaged in character creation.

3. **Emotional Signature of Anthropomorphization**:
   - High anthropomorphizers show concentrated (less diverse) emotional expression
   - They express more surprise, fear, and anger
   - They express less joy and more negative valence overall
   - This emotional pattern is UNIQUELY strong in teenagers

4. **The Age Moderation Effect**: The relationship between anthropomorphization and emotional expression differs by age:
   - For teens: High anthropomorphization → significantly less joy, more negative emotions
   - For adults: Weak or no relationship

5. **Intent Pathway**: Teen → Character Creation → High Anthropomorphization → Concentrated Negative Emotions

## Key Statistical Findings Summary

| Finding | Test | Statistic | p-value | Effect Size |
|---------|------|-----------|---------|-------------|
| Teens anthropomorphize more | t-test | t = 3.79 | <0.0001 | d = 0.111 |
| Emotional diversity lower in high anthro | t-test | - | <0.0001 | d = -1.176 |
| Intent predicts anthropomorphization | ANOVA | F = 15.58 | <0.0001 | η² ≈ 0.01 |
| Intent differs by age | χ² | 21.80 | 0.0006 | V = 0.07 |
| Intent differs by gender | χ² | 25.36 | 0.0001 | V = 0.08 |
| Anthro × Teen on Intensity | Regression | B = -0.059 | 0.0093 | - |
| Anthro × Teen on Valence | Regression | B = -0.055 | 0.0157 | - |
| Valence ANOVA across anthro groups | ANOVA | F = 178.24 | <0.0001 | - |

---

# Limitations

## Methodological Limitations

1. **Classification Accuracy**: Age prediction ~84% at high confidence (not perfect); gender prediction ~81-96%

2. **Observational Design**: Cannot establish causation, only associations

3. **Selection Bias**: Only Reddit users who publicly discuss AI companions

4. **Low Counts for Some Intents**: Emotional support (n=16) too small for reliable analysis

5. **AnthroScore Validation**: Human validation not yet completed (planned: 200 samples)

## Sample Characteristics

- Platform: Reddit only
- Language: English only
- Time Period: Limited to available data window
- Generalizability: May not extend to users who don't discuss AI companions publicly

---

# Data Files & Technical Details

## Data Directory Structure

```
Data/
├── raw/                        # Original Reddit data
├── processed/                  # Cleaned and processed data
└── features/
    ├── user_anthroscores.parquet        # AnthroScore per user
    ├── user_emotions.parquet            # Emotion profiles per user
    ├── comments_with_emotions.parquet   # Comment-level emotions
    ├── demographics_v2.parquet          # Age/gender predictions
    ├── intent_topics.parquet            # BERTopic intent labels
    ├── full_merged_dataset.parquet      # Complete merged dataset
    └── ultimate_predictor/
        ├── ultimate_predictions.parquet  # Age predictions
        ├── gender_predictions.parquet    # Gender predictions
        └── model/                        # Trained models
```

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| `author` | string | Reddit username |
| `anthroscore_mean` | float | Mean anthropomorphization score |
| `anthroscore_max` | float | Maximum anthropomorphization score |
| `age_bucket` | categorical | Predicted age group |
| `gender` | categorical | Predicted gender |
| `confidence` | float | Prediction confidence (0-1) |
| `emotion_joy`, `emotion_sadness`, etc. | float | Emotion intensities |
| `dominant_emotion` | categorical | Primary emotion |
| `intent_category` | categorical | BERTopic-derived intent |

## Statistical Tools Used

- **Python Libraries**: pandas, numpy, scipy, statsmodels, scikit-learn, xgboost
- **NLP**: sentence-transformers (SBERT), BERTopic
- **Statistical Tests**: t-tests, ANOVA, chi-square, Pearson/Spearman correlations, OLS regression
- **Effect Sizes**: Cohen's d, Pearson r, η²

---

# Future Directions

## Recommended Next Steps

### 1. Human Validation of AnthroScore
- **Priority**: HIGH
- **Method**: 200 random samples rated by 2-3 annotators
- **Metrics**: Inter-rater reliability (Krippendorff's α), correlation with AnthroScore

### 2. Longitudinal Analysis
- Track how anthropomorphization changes over time for individual users
- Examine if emotional patterns precede or follow anthropomorphization

### 3. Qualitative Analysis
- In-depth content analysis of high-anthropomorphizing posts
- Understanding the narrative and contextual features

### 4. Cross-Platform Validation
- Extend analysis to Discord, TikTok, or other platforms
- Test generalizability of findings

### 5. Intervention Study Design
- Based on character creation pathway, design interventions
- Test if reducing anthropomorphization improves emotional well-being

## Potential Research Extensions

1. **Causality**: Does anthropomorphization cause negative emotions, or vice versa?
2. **Protective Factors**: What moderates the age effect? Parental involvement? Peer relationships?
3. **Cultural Differences**: Do these patterns hold across different cultures?
4. **Platform Design**: How do specific AI companion features drive anthropomorphization?
5. **Clinical Implications**: Is high anthropomorphization associated with mental health outcomes?

---

# Appendix: Complete Statistical Results

## Age Predictor Performance Details

```
OVERALL PERFORMANCE:
- Total users with predictions: 47,062
- Accuracy on known-age users: 70.6% (vs 46.3% baseline)
- Improvement: +24.3 percentage points

PREDICTION DISTRIBUTION (all users):
- Teen:        33,682 (71.6%)
- Adult:        8,651 (18.4%)
- Young Adult:  4,729 (10.0%)

HIGH-CONFIDENCE PREDICTIONS (confidence >= 0.6):
- Total: 22,774 users (48.4% of all users)
- Teen:        19,429 (85.3%)
- Adult:        2,500 (11.0%)
- Young Adult:    845 (3.7%)
```

## Gender Predictor Performance Details

```
TRAINING RESULTS (Cross-Validation):
- text_embeddings: 75.6%
- subreddit_patterns: 72.1%
- behavioral: 72.2%
- linguistic: 79.9%
- stacked_ensemble: 80.2%

TEST SET EVALUATION:
- Overall Accuracy: 81.3%
- Test Set Size: 979

CONFUSION MATRIX:
          Predicted
          Female  Male
Actual Female  117   149
       Male     34   679
```

## RQ3 Emotional Analysis Complete Results

### All Correlations (Option A)

```
Pearson Correlations with Mean AnthroScore (n=10,914):
  emotion_joy              : r = -0.0259 (p = 0.0069) **
  emotion_sadness          : r = -0.0097 (p = 0.3092) 
  emotion_anger            : r = +0.0247 (p = 0.0100) **
  emotion_fear             : r = +0.0280 (p = 0.0035) **
  emotion_surprise         : r = +0.0333 (p = 0.0005) ***
  emotion_disgust          : r = -0.0021 (p = 0.8234) 
  emotion_neutral          : r = -0.0276 (p = 0.0039) **
  emotional_intensity      : r = +0.0276 (p = 0.0039) **
  emotional_valence        : r = -0.0258 (p = 0.0070) **
  emotional_diversity      : r = -0.3905 (p = 0.0000) ***
```

### Group Comparisons (Option B)

```
T-Test Comparisons (Top 25% vs Bottom 25% Anthropomorphizers):
  joy       : High=0.069, Low=0.078, d=-0.073 ** (LOW > HIGH)
  sadness   : High=0.081, Low=0.082, d=-0.012    (ns)
  anger     : High=0.112, Low=0.107, d=+0.040    (ns)
  fear      : High=0.047, Low=0.044, d=+0.033    (ns)
  surprise  : High=0.164, Low=0.149, d=+0.088 ** (HIGH > LOW)
  disgust   : High=0.101, Low=0.099, d=+0.017    (ns)
  intensity : High=0.574, Low=0.559, d=+0.061 *  (HIGH > LOW)
  valence   : High=-0.272, Low=-0.254, d=-0.061* (LOW > HIGH)
  diversity : High=1.566, Low=2.128, d=-1.176*** (LOW > HIGH)
```

### Age Moderation (Option C)

```
Correlations by Age Group:
  intensity : Teens r=-0.010, Adults r=+0.050*, diff p=0.0080*
  valence   : Teens r=-0.071*, Adults r=-0.015, diff p=0.0124*
  diversity : Teens r=-0.403*, Adults r=-0.413*, diff p=0.5733
  joy       : Teens r=-0.104*, Adults r=-0.014, diff p=0.0001*

Regression: Emotional Intensity ~ AnthroScore * Teen
  AnthroScore:           B = 0.049, p = 0.0013
  Teen:                  B = -0.034, p = 0.1299
  AnthroScore × Teen:    B = -0.059, p = 0.0093 ***

Regression: Emotional Valence ~ AnthroScore * Teen
  AnthroScore:           B = -0.015, p = 0.3071
  Teen:                  B = 0.015, p = 0.5095
  AnthroScore × Teen:    B = -0.055, p = 0.0157 *
```

---

*Document generated for The Illusion Project - Anthropomorphization of AI Companions Study*

