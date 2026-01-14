# Deep Dive Analysis: Why Adults Anthropomorphize More

**Generated:** 2026-01-13 23:31

---

## Executive Summary

This analysis explores potential explanations for why adults anthropomorphize AI companions more than teens.

---

## 1. Loneliness / Isolation Indicators

We searched for loneliness-related language patterns in comments.

### Prevalence by Age

| Group | % with Loneliness Indicators |
|-------|------------------------------|
| Teens | 2.2% |
| Adults | 3.6% |

**χ²** = 174.52, p = 0.0000

**Direction:** ADULTS MORE LONELY

### Correlation with Anthropomorphization

- **Pearson r:** 0.036
- **High anthropomorphizers loneliness:** 4.0%
- **Low anthropomorphizers loneliness:** 2.3%
- **Ratio:** 1.71x


---

## 2. Relationship Language Analysis

We measured semantic similarity to human relationship language.

### Age Effect

| Group | Relationship Score |
|-------|-------------------|
| Teens | 0.221 |
| Adults | 0.213 |

**t** = 1.27, p = 0.2030

**Direction:** teens use more

### Correlation with Anthropomorphization

**Pearson r** = 0.235


---

## 3. Linguistic Feature Analysis

| Feature | Teen Mean | Adult Mean | Cohen's d |
|---------|-----------|------------|----------|
| Word Count | 20.24 | 33.45 | 0.381 * |
| Words Per Sentence | 10.92 | 12.29 | 0.137 * |
| First Person Pct | 6.19 | 6.01 | -0.028  |
| Third Person Pct | 1.83 | 1.74 | -0.021  |
| Exclamations | 0.15 | 0.17 | 0.037 * |
| Questions | 0.29 | 0.28 | -0.021  |

---

## 4. Content Pattern Analysis

What topics do high vs low anthropomorphizers discuss?

| Content Type | High Anthro % | Low Anthro % | Ratio |
|--------------|---------------|--------------|-------|
| Romantic | 19.6% | 2.3% | 8.58x * |
| Emotional Support | 2.9% | 1.1% | 2.70x * |
| Friendship | 10.4% | 3.9% | 2.65x * |
| Roleplay | 23.4% | 11.2% | 2.09x * |
| Creative | 16.9% | 8.6% | 1.97x * |
| Technical | 18.4% | 17.0% | 1.09x * |

---

## 5. Subreddit Analysis

Does the age effect persist within individual subreddits?

| Subreddit | Teen Mean | Adult Mean | Cohen's d | Sig. |
|-----------|-----------|------------|-----------|------|
| CharacterAI | 1.97 | 2.04 | 0.126 | * |
| replika | 2.06 | 2.32 | 0.349 | * |
| AICompanions | 1.80 | 2.27 | 0.614 | * |

---

## Key Findings

### Why Adults Anthropomorphize More:

1. **Loneliness Hypothesis:** [Result based on analysis]
2. **Relationship Language:** [Result based on analysis]
3. **Content Differences:** [Result based on analysis]
4. **Effect Persists Across Subreddits:** The age effect is not explained by different platform preferences

---

## Visualizations

Generated in `results/deep_dive/`:
- `loneliness_by_group.png`
- `relationship_language_by_group.png`
- `content_patterns_comparison.png`
- `linguistic_features_ratio.png`

