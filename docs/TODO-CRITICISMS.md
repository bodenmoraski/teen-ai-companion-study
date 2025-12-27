# TODO: Fixing Critical Methodological Issues

**Status:** NOT STARTED  
**Goal:** Address all 16 critical issues identified in CRITICISM.md  
**Approach:** Foundationally improve methodology, not just acknowledge limitations

---

## Priority 1: Measurement Quality (CRITICAL)

### T1: Improve Classification Accuracy
- [ ] **T1.1**: Increase LLM coverage from 10.6% to >30% (use gpt-4.1-nano for cost)
- [ ] **T1.2**: Calibrate classification thresholds using cross-validation (not arbitrary percentiles)
- [ ] **T1.3**: Improve community embedding: test multiple seed pairs, average projections
- [ ] **T1.4**: Ensure weighted voting applies to users with multiple sources (>30% of users)
- [ ] **T1.5**: Test: Achieve >50% accuracy for 3-bucket age classification (current: 46.3%)
- [ ] **T1.6**: Test: LLM coverage increased from 10.6% to >30%

### T2: Address Measurement Error in Statistical Models
- [ ] **T2.1**: Implement measurement-error-corrected regression (attenuation bias correction)
- [ ] **T2.2**: Calculate reliability coefficients for classification
- [ ] **T2.3**: Use latent variable models to account for classification uncertainty
- [ ] **T2.4**: Test: Measurement error correction changes coefficient estimates meaningfully
- [ ] **T2.5**: Test: Corrected confidence intervals are wider than naive intervals

### T3: Power Analysis and Detectable Effect Sizes
- [ ] **T3.1**: Calculate statistical power for detecting effects given measurement error
- [ ] **T3.2**: Simulate minimum detectable effect sizes (MDE)
- [ ] **T3.3**: Calculate attenuation factor due to classification error
- [ ] **T3.4**: Test: Power analysis shows >80% power to detect small effects (f²=0.02) if they exist
- [ ] **T3.5**: Test: Attenuation simulations show true effects would be X× larger than observed

---

## Priority 2: Statistical Rigor (CRITICAL)

### T4: Ensure Ensemble Actually Ensembles
- [ ] **T4.1**: Document that weighted_age_vote() exists but rarely applies (most users have only 1 source)
- [ ] **T4.2**: Increase users with multiple sources by expanding LLM coverage (see T1)
- [ ] **T4.3**: Calculate and report: % of users with 1/2/3 sources
- [ ] **T4.4**: Test: >30% of users have 2+ sources for actual ensemble voting
- [ ] **T4.5**: Test: Ensemble accuracy > community embedding alone when multiple sources available

### T5: Address Multicollinearity
- [ ] **T5.1**: Calculate VIF for all predictors
- [ ] **T5.2**: Remove highly correlated predictors (VIF > 10)
- [ ] **T5.3**: Use regularization (Ridge/Lasso) if all predictors needed
- [ ] **T5.4**: Test: Max VIF < 5 after fixes
- [ ] **T5.5**: Test: Coefficient estimates are stable (bootstrap stability)

### T6: Handle Heteroscedasticity Properly
- [ ] **T6.1**: Generate residual plots (residuals vs fitted, vs predictors)
- [ ] **T6.2**: Try variance-stabilizing transformations (log, sqrt, Box-Cox)
- [ ] **T6.3**: Test for normality of residuals (Jarque-Bera, Shapiro-Wilk)
- [ ] **T6.4**: Test: Heteroscedasticity test p > 0.05 after transformation OR robust SEs shown
- [ ] **T6.5**: Test: Residual plots show no clear patterns

### T7: Analyze Influential Observations
- [ ] **T7.1**: Remove influential observations (Cook's D > 4/n)
- [ ] **T7.2**: Re-run models with and without outliers
- [ ] **T7.3**: Characterize influential points (demographics, AnthroScore, subreddit)
- [ ] **T7.4**: Test: Results are stable after removing outliers (R² change < 0.001)
- [ ] **T7.5**: Test: Influential points are characterized and reported

---

## Priority 3: Missing Data and Selection Bias (HIGH)

### T8: Address Missing Data
- [ ] **T8.1**: Analyze missingness patterns (MNAR test)
- [ ] **T8.2**: Compare AnthroScore for classified vs unclassified users
- [ ] **T8.3**: Use multiple imputation or include "unknown" as category
- [ ] **T8.4**: Test: Missingness is not associated with AnthroScore (p > 0.05)
- [ ] **T8.5**: Test: Results similar with imputation vs exclusion

### T9: Justify 3-Bucket Age Scheme
- [ ] **T9.1**: Make 3-bucket age the PRIMARY classification scheme
- [ ] **T9.2**: Provide a priori justification (theoretical or practical)
- [ ] **T9.3**: Validate 3-bucket on independent set if possible
- [ ] **T9.4**: Test: 3-bucket age is primary method in demographics.parquet
- [ ] **T9.5**: Test: Justification is documented in methods section

---

## Priority 4: Explore Real Findings (HIGH)

### T10: Analyze Subreddit Effects (THE REAL FINDING)
- [ ] **T10.1**: Develop theory for why subreddit matters
- [ ] **T10.2**: Analyze subreddit characteristics (size, norms, topics, moderation)
- [ ] **T10.3**: Compare subreddit effect size to demographic effect sizes
- [ ] **T10.4**: Test: Subreddit explains >10× more variance than demographics
- [ ] **T10.5**: Test: Subreddit-level analysis is comprehensive (all subreddits with N>100)

### T11: Analyze AnthroScore Distribution
- [ ] **T11.1**: Generate histogram of AnthroScore distribution
- [ ] **T11.2**: Analyze users with zero vs non-zero AnthroScore
- [ ] **T11.3**: Try alternative aggregations (max, std, percent non-zero)
- [ ] **T11.4**: Test: Distribution analysis shows floor effects or other patterns
- [ ] **T11.5**: Test: Alternative aggregations improve model fit (R² > 0.01)

---

## Priority 5: Robustness and Validation (MEDIUM)

### T12: Comprehensive Robustness Checks
- [ ] **T12.1**: Leave-one-subreddit-out validation
- [ ] **T12.2**: Different train/test splits for classification
- [ ] **T12.3**: Alternative seed pairs for community embeddings
- [ ] **T12.4**: Test: Results stable across robustness checks (R² variation < 0.0005)
- [ ] **T12.5**: Test: All robustness checks documented

### T13: Fix Temporal Stability Analysis
- [ ] **T13.1**: Add date column from comments to merged dataset
- [ ] **T13.2**: Run temporal stability checks (by year, by month)
- [ ] **T13.3**: Test: Temporal stability analysis runs without errors
- [ ] **T13.4**: Test: Results are stable across time periods (R² variation < 0.001)

### T14: Handle Nonbinary Gender Appropriately
- [ ] **T14.1**: Remove nonbinary from main analysis OR acknowledge n=43 limitation
- [ ] **T14.2**: Report nonbinary findings as exploratory/pilot only
- [ ] **T14.3**: Consider grouping with other category for analysis
- [ ] **T14.4**: Test: Main analysis excludes n<100 groups OR explicitly acknowledges limitation
- [ ] **T14.5**: Test: Nonbinary findings are marked as exploratory

---

## Priority 6: Framing and Interpretation (MEDIUM)

### T15: Honest Framing
- [ ] **T15.1**: Reframe as "effects too small to detect or matter" not "no effects exist"
- [ ] **T15.2**: Acknowledge measurement limitations prominently
- [ ] **T15.3**: Emphasize subreddit effects as main finding
- [ ] **T15.4**: Test: Abstract/intro mentions measurement limitations
- [ ] **T15.5**: Test: Subreddit effects discussed before demographic nulls

### T16: Comprehensive Reporting
- [ ] **T16.1**: Report both corrected and uncorrected p-values
- [ ] **T16.2**: Report VIF for all predictors
- [ ] **T16.3**: Report residual diagnostics (plots, tests)
- [ ] **T16.4**: Test: All diagnostic statistics reported in tables
- [ ] **T16.5**: Test: Limitations section addresses all 16 criticisms

---

## Test Suite Requirements

Each fix MUST have:
1. **Functional test**: Code works correctly (e.g., ensemble uses weighted voting)
2. **Validation test**: Improvement is measurable (e.g., accuracy increased)
3. **Regression test**: Fix doesn't break existing functionality
4. **Documentation test**: Changes are documented and justified

---

## Success Criteria

After fixes:
- [ ] Classification accuracy >50% (3-bucket) with documented improvement
- [ ] Measurement error accounted for in models (attenuation correction applied)
- [ ] Power analysis calculated and MDE documented
- [ ] Ensemble has multiple sources for >30% of users (not just 1%)
- [ ] VIF < 5 for all predictors OR use regularization
- [ ] Missing data analyzed (MNAR test, comparison of classified vs unclassified)
- [ ] Subreddit effects thoroughly analyzed as THE MAIN FINDING
- [ ] All robustness checks documented and pass
- [ ] Framing is honest: "too small to detect" not "no effects exist"
- [ ] All 16 criticisms addressed with tests passing

---

## Completion Log

| Task | Status | Tests Passing | Notes |
|------|--------|---------------|-------|
| T1: Improve Classification | ⚪ Not Started | 0/2 | |
| T2: Measurement Error Correction | ⚪ Not Started | 0/2 | |
| T3: Power Analysis | ⚪ Not Started | 0/2 | |
| T4: Fix Ensemble | ⚪ Not Started | 0/2 | |
| T5: Multicollinearity | ⚪ Not Started | 0/2 | |
| T6: Heteroscedasticity | ⚪ Not Started | 0/2 | |
| T7: Influential Observations | ⚪ Not Started | 0/2 | |
| T8: Missing Data | ⚪ Not Started | 0/2 | |
| T9: 3-Bucket Justification | ⚪ Not Started | 0/2 | |
| T10: Subreddit Analysis | ⚪ Not Started | 0/2 | |
| T11: AnthroScore Distribution | ⚪ Not Started | 0/2 | |
| T12: Robustness | ⚪ Not Started | 0/2 | |
| T13: Temporal Stability | ⚪ Not Started | 0/2 | |
| T14: Nonbinary Gender | ⚪ Not Started | 0/2 | |
| T15: Framing | ⚪ Not Started | 0/2 | |
| T16: Reporting | ⚪ Not Started | 0/2 | |

