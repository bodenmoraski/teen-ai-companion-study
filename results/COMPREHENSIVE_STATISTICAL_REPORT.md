# The Illusion Project: Comprehensive Statistical Results

**Generated:** 2026-01-10 23:55:03
**Analysis Sample:** 27,846 users with V3 predictions @ 0.60 confidence

---

## Executive Summary

### Age Effect on Anthropomorphization
- **Cohen's d:** 0.0141 (negligible)
- **p-value:** 0.515293
- **Bootstrap 95% CI:** [-0.0374, 0.0709]
- **Direction:** Teens Higher

### Gender Effect on Anthropomorphization
- **Cohen's d:** -0.1099 (negligible)
- **p-value:** 0.000031

---

## RQ1: Demographics of AI Companion Users

### Age Distribution

| Age Group | Count | Percentage | 95% CI |
|-----------|-------|------------|--------|
| Teen | 14,621 | 52.5% | [51.9%, 53.1%] |
| Adult | 13,225 | 47.5% | [46.9%, 48.1%] |

### Gender Distribution

| Gender | Count | Percentage | 95% CI |
|--------|-------|------------|--------|
| Male | 21,889 | 78.6% | [78.1%, 79.1%] |
| Female | 5,957 | 21.4% | [20.9%, 21.9%] |

---

## RQ2: Demographics & Anthropomorphization

### Age → Anthropomorphization

| Metric | Teen | Adult |
|--------|------|-------|
| N | 4,741 | 3,935 |
| Mean AnthroScore | 1.9373 | 1.9194 |
| SD | 1.2633 | 1.2852 |

| Statistical Test | Value |
|-----------------|-------|
| Welch's t | 0.651 |
| p-value | 0.515293 |
| Cohen's d | 0.0141 (negligible) |
| Bootstrap 95% CI | [-0.0374, 0.0709] |

### Age × Gender Two-Way ANOVA

| Effect | F | p-value | Significant |
|--------|---|---------|-------------|
| Age | 0.53 | 0.465142 | No |
| Gender | 18.91 | 0.000014 | Yes |
| Age × Gender | 3.76 | 0.052614 | No |

**R-squared:** 0.0027

### Ground Truth Validation (Self-Declared Ages)

- Teen mean: 2.6544 (n=158)
- Adult mean: 3.0517 (n=128)
- Cohen's d: -0.2970
- Direction: Adults Higher
- **Matches predicted direction:** NO - CRITICAL DISCREPANCY

---

## RQ3: Emotional Expression & Anthropomorphization

### Correlations with AnthroScore

| Variable | r | p-value | Interpretation |
|----------|---|---------|----------------|
| emotion_anger | 0.0498 | 0.000001 *** | negligible |
| emotion_sadness | -0.0476 | 0.000004 *** | negligible |
| emotion_neutral | -0.0475 | 0.000004 *** | negligible |
| emotion_fear | 0.0457 | 0.000009 *** | negligible |
| emotion_disgust | 0.0331 | 0.001320 ** | negligible |
| emotion_surprise | 0.0129 | 0.211585  | negligible |
| emotion_joy | 0.0044 | 0.666435  | negligible |

### High vs Low Anthropomorphizers (Quartile Comparison)

| Variable | Low Mean | High Mean | Cohen's d | p-value |
|----------|----------|-----------|-----------|---------|
| emotion_neutral | 0.4546 | 0.4248 | -0.1264 | 0.000021 *** |
| emotion_anger | 0.0986 | 0.1139 | 0.1201 | 0.000051 *** |
| emotion_sadness | 0.0911 | 0.0794 | -0.0894 | 0.002711 ** |
| emotion_fear | 0.0399 | 0.0479 | 0.0880 | 0.002988 ** |
| emotion_disgust | 0.0955 | 0.1047 | 0.0752 | 0.011346 * |
| emotion_surprise | 0.1445 | 0.1534 | 0.0572 | 0.053896  |
| emotion_joy | 0.0758 | 0.0759 | 0.0005 | 0.987521  |

---

## Sensitivity Analyses

### Age Effect Stability Across Confidence Thresholds

| Threshold | N | Cohen's d | p-value | Significant |
|-----------|---|-----------|---------|-------------|
| >= 0.5 | 14,007 | 0.0194 | 0.253916 | No |
| >= 0.6 | 10,345 | 0.0258 | 0.192964 | No |
| >= 0.7 | 6,631 | 0.0280 | 0.260593 | No |
| >= 0.8 | 3,540 | -0.0068 | 0.843213 | No |
| >= 0.9 | 1,155 | -0.1404 | 0.027526 | Yes |
