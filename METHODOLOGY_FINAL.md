# Final Methodology Documentation

**The Illusion Project: Anthropomorphization of AI Companions**

This document chronicles the **final methodology** used for each component of the research. It is intended as a reference for writing the research paper.

---

## Table of Contents

1. [Data Collection](#1-data-collection)
2. [Age Classification](#2-age-classification)
3. [Gender Classification](#3-gender-classification)
4. [AnthroScore V3 (Anthropomorphization)](#4-anthroscore-v3-anthropomorphization)
5. [Emotion Analysis](#5-emotion-analysis)
6. [Statistical Analysis](#6-statistical-analysis)

---

## 1. Data Collection

### Source
- **Arctic Shift** Reddit archive (https://arctic-shift.photon-reddit.com/)
- Public Reddit API for supplementary data

### Subreddits Collected
| Subreddit | Focus | Comments |
|-----------|-------|----------|
| r/CharacterAI | Character-based AI companions | 118,686 |
| r/Replika | AI companion app (relationship-focused) | 3,658 |
| r/AICompanions | General AI companion discussion | 2,196 |

### Total Dataset
- **283,895 comments** collected
- **47,062 unique users**
- Date range: [Insert dates from your data]

### Data Cleaning
1. Removed bot accounts (AutoModerator, common bots)
2. Removed `[deleted]` and `[removed]` content
3. Removed comments with < 20 characters
4. Deduplicated by comment ID

### Implementation
- **File:** `src/data_collection/preprocess.py`
- **Output:** `Data/processed/all_comments.parquet`

---

## 2. Age Classification

### Final Approach: LLM-Based Classification

We use **GPT-4o-mini** (via OpenAI API) to classify users into binary age groups based on their comment history.

### Age Categories
| Category | Age Range | Label |
|----------|-----------|-------|
| Teen | 13-18 years | `teen` |
| Adult | 19+ years | `adult` |

### Classification Process

1. **Input:** Up to 20 most recent comments from a user
2. **Context truncation:** Each comment limited to 200 characters
3. **Prompt:** LLM analyzes vocabulary, topics, life stage indicators, cultural references, writing maturity
4. **Output:** Age bucket, confidence score (0-1), reasoning

### Prompt Template
```
Analyze these Reddit comments from a single user and estimate their most likely age bucket.

Consider:
- Vocabulary and language complexity
- Topics and interests mentioned
- Life stage indicators (school, work, family references)
- Cultural references
- Writing style and maturity

[Comments inserted here]

Age buckets: 13-18, 19-25, 26-40, 41-60, 61-80

Respond with JSON only:
{"age_bucket": "...", "confidence": 0.0-1.0, "reasoning": "..."}
```

### Binary Conversion
- 13-18 → `teen`
- 19-25, 26-40, 41-60, 61-80 → `adult`

### Confidence Threshold
- **Threshold:** ≥ 0.60
- Users below threshold excluded from demographic analyses
- Sensitivity analyses conducted at thresholds 0.50, 0.55, 0.60, 0.65, 0.70

### Validation
- Validated against self-declared ages from comment text
- Accuracy metrics: [Insert if available]

### Implementation
- **File:** `src/demographics/llm_classifier.py`
- **Model:** `gpt-4o-mini` (LLM_AGE_MODEL in config)
- **Temperature:** Configurable (default 0.1 for consistency)
- **Output:** `experiments/v2_correction/age_predictions_v4.parquet`

---

## 3. Gender Classification

### Final Approach: LLM-Based Classification

Same architecture as age classification, using **GPT-4o-mini**.

### Gender Categories
| Category | Label |
|----------|-------|
| Male | `male` |
| Female | `female` |

### Classification Process

1. **Input:** Up to 20 most recent comments from a user
2. **Prompt:** LLM analyzes language patterns, topics, self-references
3. **Output:** Gender prediction, confidence score (0-1), reasoning

### Confidence Threshold
- **Threshold:** ≥ 0.60
- Same sensitivity analysis approach as age

### Known Limitations
- Binary classification (does not capture non-binary identities)
- Based on linguistic patterns, not self-identification
- Reddit-specific language may introduce biases

### Implementation
- **File:** `src/demographics/llm_classifier.py` (same module, different prompt)
- **Output:** `experiments/v2_correction/gender_predictions_v4.parquet`

---

## 4. AnthroScore V3 (Anthropomorphization)

### Evolution from V2

| Version | Method | Correlation with Expert |
|---------|--------|------------------------|
| V1 | Original AnthroScore (pronoun probability) | Unknown |
| V2 | MLM-based (Twitter-RoBERTa) | r = 0.11 |
| **V3** | **LLM-based (GPT-4.1-nano)** | **r = 0.59** |

### Final Approach: LLM-Based Classification

We use **GPT-4.1-nano** to directly classify anthropomorphization on a 1-5 ordinal scale.

### Classification Scale

| Score | Label | Description | Examples |
|-------|-------|-------------|----------|
| 1 | None | AI treated as pure software/tool | "The app is buggy", "Clear the cache" |
| 2 | Minimal | Slight humanization but still AI | "It's pretty smart", "The bot understood" |
| 3 | Moderate | Some human attributes/emotions | "She seemed confused", uses he/she |
| 4 | High | Genuine feelings/personality | "He really cares", "she gets jealous" |
| 5 | Extreme | Full human-equivalent relationship | "We're in love", "they're my everything" |

### Key Indicators (from prompt)

**Higher anthropomorphization:**
- Pronouns: "he/she/they" instead of "it"
- Emotions: "happy", "sad", "jealous", "caring"
- Relationship language: "friend", "partner"
- Agency: "decided to", "wanted to", "chose to"

**Lower anthropomorphization:**
- Pronouns: "it", "the bot", "the app"
- Technical language: "glitch", "bug", "settings"
- Functional framing: "tool", "software"

### Production Pipeline

1. **Pre-filtering:** Auto-score simple comments as 1 to reduce API costs
   - Comments < 20 characters
   - Comments with no AI-related keywords
   
2. **LLM Classification:**
   - Model: GPT-4.1-nano
   - Temperature: 1.0 (default for GPT-4.1)
   - Max tokens: 200
   - Response format: JSON
   
3. **Retry Logic:** 
   - 3 retries with exponential backoff
   - Handle rate limits (429 errors)

4. **Parallelization:**
   - Async processing with 15 concurrent requests
   - Batch size: 5000 comments per checkpoint

### Validation

**Expert Labels:**
- 100 comments labeled by GPT-5-mini (acting as expert)
- Used for validation, not training

**Validation Metrics:**

| Metric | V3 (LLM) | V2 (MLM) |
|--------|----------|----------|
| Expert correlation (r) | **0.59** | 0.11 |
| Head-to-head accuracy | **83%** | 16% |
| Within-1 accuracy | **96%** | N/A |
| Cohen's Kappa | 0.58 | N/A |

### User-Level Aggregation

For statistical analyses, we aggregate comment-level scores to user-level:
- **Mean:** Average anthropomorphization across all comments
- **Max:** Highest anthropomorphization score
- **Count:** Number of scored comments

### Cost
- **Total:** ~$8-10 for 283,895 comments
- **Model:** GPT-4.1-nano (~$0.10 per 1M tokens)

### Implementation
- **File:** `experiments/anthroscore_v3/anthroscore_llm.py`
- **Production pipeline:** `experiments/anthroscore_v3/run_full_dataset_optimized.py`
- **Output:** `experiments/anthroscore_v3/anthroscore_v3_full.parquet`

---

## 5. Emotion Analysis

### Approach: Pre-trained Transformer Classifier

We use a pre-trained **DistilRoBERTa** model fine-tuned on emotion classification.

### Model
- **Name:** `j-hartmann/emotion-english-distilroberta-base`
- **Source:** Hugging Face Hub
- **Training data:** Multiple emotion datasets

### Emotion Categories

| Emotion | Description |
|---------|-------------|
| Joy | Happiness, excitement, positive affect |
| Sadness | Grief, disappointment, negative affect |
| Anger | Frustration, irritation, hostility |
| Fear | Anxiety, worry, apprehension |
| Surprise | Astonishment, unexpectedness |
| Disgust | Revulsion, distaste |
| Neutral | No strong emotional content |

### Classification Process

1. **Input:** Individual comment text
2. **Preprocessing:** Truncation to model max length (512 tokens)
3. **Output:** Probability distribution across all 7 emotions
4. **Dominant emotion:** Argmax of probabilities

### User-Level Aggregation

For each user, we compute:
- Mean probability for each emotion across all comments
- Dominant emotion (most frequent across comments)
- Dominant emotion score (mean probability of dominant)

### Implementation
- **File:** `src/analysis/emotion_analysis.py`
- **Device:** GPU if available, else CPU
- **Batch size:** 32
- **Output:** `Data/features/user_emotions.parquet`

---

## 6. Statistical Analysis

### Research Questions

| RQ | Question | DV | IV |
|----|----------|----|----|
| RQ1 | Demographics of AI companion users | - | Age, Gender |
| RQ2 | Demographics → Anthropomorphization | AnthroScore V3 | Age, Gender |
| RQ3 | Emotions → Anthropomorphization | AnthroScore V3 | 7 emotion scores |

### Statistical Tests Employed

#### For Group Comparisons

| Test | Purpose | When Used |
|------|---------|-----------|
| Welch's t-test | Mean comparison (unequal variance) | Age, Gender effects |
| Mann-Whitney U | Non-parametric alternative | Robustness checks |
| Brunner-Munzel | Robust to heteroscedasticity | Variance sensitivity |
| Two-way ANOVA | Age × Gender interaction | Interaction effects |
| Tukey HSD | Pairwise comparisons | Post-hoc tests |
| Chi-square | Categorical associations | Age × Gender distribution |

#### For Correlations

| Test | Purpose | When Used |
|------|---------|-----------|
| Pearson r | Linear correlation | Emotion correlations |
| Spearman ρ | Rank correlation | Robustness check |
| Fisher z-test | Compare correlations | Age moderation |

#### For Regression

| Model | Purpose | When Used |
|-------|---------|-----------|
| OLS regression | Continuous outcome | Full models |
| Logistic regression | Binary outcome | High anthropomorphizer (yes/no) |
| Robust regression (Huber) | Outlier-resistant | Sensitivity check |

### Effect Sizes

| Measure | Interpretation | Used For |
|---------|----------------|----------|
| Cohen's d (Hedges' g) | Small: 0.2, Medium: 0.5, Large: 0.8 | Group comparisons |
| η² (eta-squared) | Variance explained | ANOVA effects |
| Cramér's V | Association strength | Categorical relationships |
| Odds Ratio | Relative odds | Logistic regression |
| CLES | Probability of superiority | Intuitive effect size |

### Confidence Intervals

- **Method:** Bootstrap (n = 5,000-10,000 resamples)
- **Confidence level:** 95%
- **For proportions:** Wilson score interval

### Multiple Comparison Correction

- **Method:** Bonferroni correction for post-hoc tests
- **Method:** Tukey HSD for pairwise comparisons

### Implementation
- **Main script:** `scripts/COMPREHENSIVE_V3_ANALYSIS.py`
- **Extended checks:** `scripts/EXTENDED_ANALYSIS.py`
- **Deep dive:** `scripts/DEEP_DIVE_ANALYSIS.py`
- **Output:** `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md`

---

## Software & Dependencies

### Python Version
- Python 3.10+

### Key Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation |
| numpy | Latest | Numerical operations |
| scipy | Latest | Statistical tests |
| statsmodels | Latest | Regression, ANOVA |
| scikit-learn | Latest | ML utilities |
| transformers | Latest | Emotion classification |
| openai | Latest | GPT API access |
| sentence-transformers | Latest | Semantic similarity |
| matplotlib, seaborn | Latest | Visualization |

### API Requirements
- **OpenAI API key** for LLM-based classification
- Estimated cost: ~$10-20 for full pipeline

---

## Reproducibility Notes

1. **Random seeds:** Set where applicable for reproducibility
2. **Data versioning:** Parquet files with timestamps
3. **Checkpointing:** Intermediate results saved for resumability
4. **Logging:** All processes logged with timestamps

---

*Document created for The Illusion Project*  
*Last updated: January 2026*
