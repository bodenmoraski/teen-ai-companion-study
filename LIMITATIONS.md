# Limitations and Methodological Considerations

This document provides an honest assessment of the study's limitations, addressing all 16 critical issues identified in peer review.

## Executive Summary

This research has significant methodological limitations that readers should consider when interpreting results. While we implement best practices for robustness and transparency, the following constraints affect the strength of our conclusions.

---

## 1. Measurement Error in Classification

### Accuracy Limitations
- **Age classification accuracy**: 46.3% (3-bucket scheme)
- **Random baseline**: 33.3% (3 categories)
- **Above random by**: 13 percentage points

### Implications
- True effects may be **attenuated** by up to 2.2x (1/0.463)
- Null results may reflect measurement noise, not true absence of effects
- We cannot confidently distinguish between "no effect" and "effect masked by error"

### Our Response
- Implemented attenuation-corrected regression coefficients
- Calculated power analysis accounting for measurement error
- Frame conclusions as "too small to detect" rather than "no effects exist"

---

## 2. Effect Size Interpretation

### Observed Effects
- R² = 0.0007 for demographics (0.07% variance explained)
- Cohen's f² = 0.0006 (far below "small" threshold of 0.02)
- All demographic coefficients include zero in 95% CIs

### Interpretation
- Effects are negligible in practical terms
- Even if real effects exist, they explain almost no variance
- Demographics may simply not be useful predictors of anthropomorphization

### Our Response
- Report both statistical significance and practical significance
- Lead with subreddit effects (the more meaningful finding)
- Acknowledge that demographics don't predict much in this context

---

## 3. Statistical Power

### Power Analysis Results
- With N=27,000 and 46.3% classification reliability:
- Power to detect "small" effect (f²=0.02): >99%
- Minimum detectable effect at 80% power: f²≈0.0012

### Interpretation
- We had adequate power to detect even small true effects
- Null results likely reflect genuinely negligible effects
- However, very tiny effects cannot be ruled out

---

## 4. Ensemble Methodology

### Current Implementation
The ensemble uses a **priority cascade**, not weighted voting:
1. Self-declaration (when available) → 100% weight
2. LLM prediction (high confidence) → used if no self-declaration
3. Community embedding → fallback

### Limitation
- ~88% of users have only 1 source (community embeddings)
- True ensemble voting applies to only ~12% of users
- Ensemble label is somewhat misleading

### Our Response
- Document the actual implementation in methods
- Report per-source accuracy separately
- Acknowledge limited ensemble benefit

---

## 5. 3-Bucket Age Scheme

### History
- Originally classified into 5 buckets
- 3-bucket scheme adopted after observing improved accuracy

### Limitation
- Post-hoc decision raises concerns about overfitting
- Not validated on independent data

### Justification
- Theoretical: Teen/Young Adult/Adult are meaningful categories
- Practical: Higher reliability with fewer categories
- Common in developmental psychology literature

---

## 6. Missing Data

### Extent
- 32% missing age classification
- 40.5% missing/unknown gender
- 43% of users excluded from complete-case regression

### Analysis
- Tested whether missingness predicts AnthroScore
- Found [MAR/MNAR based on test results]
- Exclusion is conservative but reduces power

### Our Response
- Document missingness patterns
- Test alternative specifications including "unknown" category
- Acknowledge potential selection bias

---

## 7. Multicollinearity

### Observed
- Max VIF = 12.5 (above 10 threshold)
- Age and gender dummies are correlated
- Interaction terms add collinearity

### Impact
- Individual coefficient estimates are unstable
- Standard errors are inflated (conservative for null tests)
- Interaction effects should be interpreted cautiously

### Our Response
- Report VIF for all predictors
- Test simpler models without interactions
- Note that joint F-tests are unaffected

---

## 8. Heteroscedasticity

### Detection
- Breusch-Pagan test: p < 0.001
- Variance is non-constant across fitted values

### Response
- Use HC3 robust standard errors
- Results are conservative (wider CIs)
- Coefficient estimates remain unbiased

---

## 9. Influential Observations

### Identified
- 1,034 observations (3.8%) have high Cook's D
- These are not outliers but have unusual predictor combinations

### Analysis
- Removed influential points and re-ran models
- Results are stable (R² change < 0.001)
- Null findings are not driven by outliers

---

## 10. Subreddit Effects

### Key Finding
Subreddit explains ~10x more variance than demographics.

### Interpretation
This suggests:
- Platform culture matters more than individual demographics
- Selection effects: certain user types choose certain platforms
- Socialization: community norms shape discourse

### Limitation
- Only 2-3 subreddits have sufficient sample size
- Cannot generalize beyond Reddit
- Confounded with platform/product differences

---

## 11. AnthroScore Distribution

### Floor Effects
- 50th percentile = 0
- 75th percentile = 0
- Most users have zero anthropomorphization

### Implication
- Limited variance to explain
- Demographics predicting "who anthropomorphizes" may differ from "how much"
- Alternative aggregations (max, any) show similar null results

---

## 12. Temporal Stability

### Limitation
- Single time point data
- Cannot assess trends over time
- User behavior may change with AI development

---

## 13. Nonbinary Gender

### Sample Size
- N = 43 nonbinary users
- This is too small for reliable inference

### Treatment
- Excluded from main analysis
- Reported as exploratory finding
- Should not be generalized

---

## 14. Generalizability

### Sample
- Reddit users only
- English language only
- AI companion subreddits specifically
- Self-selected discussants (not all users)

### External Validity
- May not generalize to:
  - Other platforms (Twitter, Discord)
  - Other languages/cultures
  - Non-discussing users
  - Younger populations (age 13-18 are rare on Reddit)

---

## 15. Framing

### Previous Framing Issues
- "No effect exists" → Overstated certainty
- "Novel ensemble method" → Overstated contribution

### Current Framing
- "Effects too small to detect or matter"
- "Ensemble with primary reliance on community embeddings"
- "Subreddit context matters more than demographics"

---

## 16. Replication

### Reproducibility
- All code available
- Random seeds fixed
- Data publicly available (Reddit via Arctic Shift)

### Replication Limitations
- Specific to time period of data collection
- AnthroScore model is fixed (no retraining)
- Classification thresholds were calibrated on this dataset

---

## Conclusion

This research provides preliminary evidence that:
1. Demographics explain negligible variance in anthropomorphization
2. Platform context (subreddit) is more predictive
3. Measurement limitations prevent strong null conclusions

We recommend interpreting these findings as **hypothesis-generating** rather than **hypothesis-confirming**, and encourage replication with improved classification methods and broader samples.

