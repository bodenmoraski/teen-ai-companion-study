# Implementation Plan: Fixing Critical Methodological Issues

**Goal:** Foundationally address all 16 criticisms with testable improvements  
**Approach:** Build comprehensive test suite FIRST, then implement fixes that pass tests

---

## Phase 0: Test Framework Setup (FIRST - CRITICAL)

### Step 0.1: Create Test Infrastructure
**Purpose:** Ensure we can validate fixes actually work

**Implementation:**
1. Create `tests/test_criticism_fixes.py` with test framework
2. Define test fixtures (load current data, create validation sets)
3. Define baseline metrics (current accuracy, R², VIF, etc.)
4. Define success criteria (what metrics must improve to pass)

**Tests to Write:**
```python
# Example test structure
def test_classification_accuracy_improves():
    """Test that classification accuracy improves above 55%"""
    current_accuracy = get_current_accuracy()
    new_accuracy = get_new_accuracy()
    assert new_accuracy > 0.55
    assert new_accuracy > current_accuracy

def test_ensemble_uses_weighted_voting():
    """Test that ensemble uses weighted voting, not priority cascade"""
    ensemble_method = get_ensemble_implementation()
    assert uses_weighted_voting(ensemble_method)
    assert weighted_voting_applies_to_more_than_50_percent_users()
```

**Success Criteria:**
- [ ] Test framework runs all tests
- [ ] Baseline metrics captured
- [ ] All tests fail initially (proves tests work)

---

## Phase 1: Improve Classification Quality (Highest Impact)

### Step 1.1: Increase Multi-Source Coverage for True Ensemble

**Current Problem:** Weighted voting code exists (`weighted_age_vote()` in `ensemble_classifier.py`) but rarely applies because most users have only 1 source (community embeddings). Self-declarations: 1%, LLM: 10.6%. For 88% of users, there's nothing to ensemble.

**Fix:** Increase coverage of secondary methods so weighted voting actually applies

**Implementation:**
1. **Increase LLM coverage:**
   - Current: 10.6% of users
   - Target: >30% of users
   - Use gpt-4.1-nano for cost efficiency (~$0.10 per 1000 users)
   
2. **Improve community embedding quality:**
   - Test multiple seed pair combinations
   - Average projections from multiple seed pairs for stability
   
3. **Calculate and report source distribution:**
```python
def report_source_distribution(demo_df):
    has_self = demo_df['age_bucket_self_declared'].notna()
    has_comm = demo_df['age_bucket_community'].notna()
    has_llm = demo_df['age_bucket_llm'].notna()
    
    n_sources = has_self.astype(int) + has_comm.astype(int) + has_llm.astype(int)
    
    print(f"1 source: {(n_sources == 1).mean():.1%}")
    print(f"2 sources: {(n_sources == 2).mean():.1%}")
    print(f"3 sources: {(n_sources == 3).mean():.1%}")
```

**Tests:**
- [ ] `test_llm_coverage_increased()`: LLM covers >30% of users (up from 10.6%)
- [ ] `test_multi_source_coverage()`: >30% of users have 2+ sources
- [ ] `test_weighted_voting_applied()`: For users with 2+ sources, final prediction uses voting

**Files to Modify:**
- `scripts/phase2_with_api_data.py`: Run LLM on more users
- `src/demographics/llm_classifier.py`: Optimize for cost/coverage

---

### Step 1.2: Improve Classification Accuracy

**Current Problem:** 46.3% accuracy is marginal (sensitivity shows 50.7% achievable)  
**Goal:** >50% accuracy for 3-bucket age (realistic given constraints)

**Implementation:**
1. **Cross-validation for threshold calibration:**
   - Current: Arbitrary percentile thresholds
   - Fix: Use 5-fold CV on self-declared users to optimize thresholds
   - Maximize accuracy, not just use percentiles
   
2. **Better seed pairs:**
   - Test alternative age subreddits:
     - Young: r/teenagers, r/teenrelationships, r/highschool
     - Old: r/RedditForGrownups, r/over30, r/AskOldPeople
   - Average projections from multiple seed pairs
   
3. **Optimal threshold from sensitivity analysis:**
   - Current sensitivity shows 50.7% accuracy at 0.90 threshold scale
   - Use this insight to set optimal thresholds

**Tests:**
- [ ] `test_accuracy_above_50_percent()`: 3-bucket accuracy > 0.50 (up from 46.3%)
- [ ] `test_thresholds_cv_optimized()`: Document that thresholds were selected via CV
- [ ] `test_multiple_seed_pairs()`: At least 2 seed pair combinations tested

**Files to Create/Modify:**
- `src/demographics/threshold_calibration.py`: CV-based threshold selection
- `src/demographics/community_embedding_improved.py`: Multiple seed pairs

---

### Step 1.3: Measurement Error Correction

**Current Problem:** Classification error attenuates true effects  
**Fix:** Account for measurement error in regression

**Implementation:**
1. **Calculate reliability:**
   - Reliability = test-retest agreement
   - Or reliability = agreement between methods
   - Use to calculate attenuation factor

2. **Attenuation-corrected regression:**
   - True correlation = observed correlation / reliability
   - Use measurement-error models (e.g., errors-in-variables regression)
   - Or use latent variable models

3. **Power analysis with measurement error:**
   - Calculate power given classification accuracy
   - Show minimum detectable effect (MDE) accounting for attenuation

**Tests:**
- [ ] `test_reliability_calculated()`: Reliability coefficients computed
- [ ] `test_attenuation_factor_calculated()`: Attenuation factor < 1 (shows bias)
- [ ] `test_corrected_coefficients_larger()`: Corrected coefficients > naive coefficients
- [ ] `test_power_analysis_includes_error()`: Power analysis accounts for measurement error

**Files to Create:**
- `src/statistical/measurement_error_correction.py`: Attenuation correction
- `src/statistical/power_analysis_with_error.py`: Power analysis with measurement error

---

## Phase 2: Statistical Rigor Fixes

### Step 2.1: Fix Multicollinearity

**Current Problem:** VIF = 12.5 (high multicollinearity)  
**Fix:** Remove highly correlated predictors or use regularization

**Implementation:**
1. **Calculate VIF for all predictors:**
   - Report VIF in regression output
   - Identify which predictors are problematic

2. **Options:**
   - **Option A:** Remove predictors with VIF > 10
   - **Option B:** Use Ridge regression (L2 regularization)
   - **Option C:** Use principal components for age/gender

3. **Prefer Option A first** (simplest, most interpretable)

**Tests:**
- [ ] `test_vif_calculated_all_predictors()`: VIF reported for all predictors
- [ ] `test_max_vif_below_5()`: Max VIF < 5 after fixes
- [ ] `test_coefficients_stable()`: Bootstrap shows stable coefficients (low variance)

**Files to Modify:**
- `src/statistical/regression_models.py`: Add VIF calculation, remove high-VIF predictors
- `src/statistical/neurips_analysis.py`: Include VIF in diagnostics

---

### Step 2.2: Handle Heteroscedasticity Properly

**Current Problem:** Heteroscedasticity detected but only partially addressed  
**Fix:** Try transformations or properly document

**Implementation:**
1. **Generate residual plots:**
   - Residuals vs fitted values
   - Residuals vs each predictor
   - Q-Q plot for normality

2. **Try transformations:**
   - Log-transform AnthroScore (if all positive, or log1p)
   - Square-root transform
   - Box-Cox transform

3. **Choose best transformation:**
   - Minimize heteroscedasticity test p-value
   - Maximize model fit (R²)
   - Keep interpretability

4. **If no transformation helps, keep HC3 robust SEs but document**

**Tests:**
- [ ] `test_residual_plots_generated()`: Residual plots created
- [ ] `test_heteroscedasticity_reduced()`: BP test p > 0.05 after transformation OR robust SEs documented
- [ ] `test_residuals_normal()`: Residuals approximately normal (JB test p > 0.05)

**Files to Create/Modify:**
- `src/statistical/residual_diagnostics.py`: Generate residual plots and tests
- `src/statistical/neurips_analysis.py`: Try transformations, report diagnostics

---

### Step 2.3: Analyze Influential Observations

**Current Problem:** 1,034 influential observations not analyzed  
**Fix:** Remove and re-analyze

**Implementation:**
1. **Remove influential observations:**
   - Cook's D > 4/n threshold
   - Re-run all models

2. **Characterize influential points:**
   - Demographics (age, gender)
   - AnthroScore (extreme values?)
   - Subreddit distribution
   - Comment count/activity

3. **Compare results:**
   - R² with and without outliers
   - Coefficient estimates
   - Significance tests

**Tests:**
- [ ] `test_influential_observations_identified()`: Influential points identified (Cook's D)
- [ ] `test_results_stable_after_removal()`: R² change < 0.001 after removing outliers
- [ ] `test_influential_points_characterized()`: Characteristics reported (demographics, scores, etc.)

**Files to Create/Modify:**
- `src/statistical/influence_analysis.py`: Remove outliers, characterize them
- `src/statistical/neurips_analysis.py`: Run models with and without outliers

---

## Phase 3: Missing Data and Selection Bias

### Step 3.1: Address Missing Data

**Current Problem:** 43% of users excluded, potential selection bias  
**Fix:** Analyze missingness, use imputation or include "unknown"

**Implementation:**
1. **Analyze missingness:**
   - Test if missingness is associated with AnthroScore (t-test)
   - Analyze patterns (which demographics are missing more?)
   
2. **Options:**
   - **Option A:** Include "unknown" as category (simplest)
   - **Option B:** Multiple imputation (more sophisticated)
   - **Option C:** Analyze separately (classified vs unclassified)

3. **Prefer Option A first** (interpretable, doesn't assume MAR)

**Tests:**
- [ ] `test_missingness_analyzed()`: Missingness patterns reported
- [ ] `test_missingness_not_associated_with_anthroscore()`: T-test p > 0.05
- [ ] `test_results_with_unknown_category()`: Models run with "unknown" category
- [ ] `test_results_similar_with_imputation()`: Imputation vs exclusion give similar results

**Files to Create/Modify:**
- `src/statistical/missing_data_analysis.py`: Analyze missingness patterns
- `src/statistical/regression_models.py`: Include "unknown" as category

---

### Step 3.2: Make 3-Bucket Age Primary

**Current Problem:** 3-bucket created post-hoc  
**Fix:** Make it the primary classification scheme

**Implementation:**
1. **Justify a priori:**
   - Theoretical: Teen vs Young Adult vs Adult (meaningful categories)
   - Practical: Better accuracy than 5-bucket
   - Document justification

2. **Use as primary:**
   - Classify directly into 3 buckets (not convert from 5)
   - Save 3-bucket in demographics.parquet
   - Use 3-bucket in all analyses

3. **Validate:**
   - Validate on independent set if possible
   - Or use CV for validation

**Tests:**
- [ ] `test_3bucket_is_primary()`: 3-bucket age is in demographics.parquet as primary
- [ ] `test_3bucket_justification_documented()`: Justification written in methods
- [ ] `test_3bucket_validated()`: Validation on independent set or CV

**Files to Modify:**
- `src/demographics/community_embedding.py`: Classify directly to 3 buckets
- `data/features/demographics.parquet`: Use 3-bucket as primary
- `docs/methodology.md`: Document 3-bucket justification

---

## Phase 4: Explore Real Findings

### Step 4.1: Analyze Subreddit Effects (THE MAIN FINDING)

**Current Problem:** Subreddit explains 10× more variance but unexplored  
**Fix:** Comprehensive subreddit analysis

**Implementation:**
1. **Develop theory:**
   - Why would subreddit matter? (culture, norms, topics, moderation)
   - Literature review on platform effects

2. **Characterize subreddits:**
   - Size (subscriber count, active users)
   - Topics (BERTopic analysis per subreddit)
   - Moderation style (strictness, rules)
   - User demographics (if available)

3. **Compare effect sizes:**
   - Subreddit R² vs demographic R²
   - Interaction: subreddit × demographics

4. **Reframe main finding:**
   - "Subreddit context matters more than demographics"
   - Discuss implications for AI companion research

**Tests:**
- [ ] `test_subreddit_analysis_comprehensive()`: All subreddits with N>100 analyzed
- [ ] `test_subreddit_explains_more_variance()`: Subreddit R² > 10× demographic R²
- [ ] `test_subreddit_characteristics_reported()`: Subreddit characteristics documented

**Files to Create:**
- `src/analysis/subreddit_characteristics.py`: Analyze subreddit features
- `results/tables/subreddit_analysis.txt`: Comprehensive subreddit comparison

---

### Step 4.2: Analyze AnthroScore Distribution

**Current Problem:** Most scores are zero, floor effects  
**Fix:** Analyze distribution, try alternative aggregations

**Implementation:**
1. **Distribution analysis:**
   - Histogram of AnthroScore
   - Percentile analysis
   - Test for floor effects

2. **Alternative aggregations:**
   - Max AnthroScore (peak anthropomorphization)
   - Std AnthroScore (variability)
   - Percent non-zero (any anthropomorphization)
   
3. **Compare model fit:**
   - Which aggregation explains most variance?
   - Which makes most theoretical sense?

**Tests:**
- [ ] `test_anthroscore_distribution_analyzed()`: Distribution plots and stats generated
- [ ] `test_alternative_aggregations_tested()`: Max, std, percent non-zero calculated
- [ ] `test_alternative_aggregation_improves_fit()`: At least one aggregation gives R² > 0.01

**Files to Create:**
- `src/analysis/anthroscore_distribution.py`: Distribution analysis
- `src/analysis/alternative_aggregations.py`: Try different aggregation methods

---

## Phase 5: Robustness and Validation

### Step 5.1: Comprehensive Robustness Checks

**Current Problem:** Robustness checks incomplete  
**Fix:** Expand robustness suite

**Implementation:**
1. **Leave-one-subreddit-out:**
   - Remove each subreddit, re-run analysis
   - Check if results stable

2. **Train/test splits:**
   - 5-fold CV for classification
   - Report accuracy for each fold

3. **Alternative seed pairs:**
   - Test different seed pairs for embeddings
   - Show results are stable

4. **Threshold sensitivity:**
   - Already done, but expand range

**Tests:**
- [ ] `test_robustness_leave_one_subreddit_out()`: Results stable (R² variation < 0.0005)
- [ ] `test_robustness_cross_validation()`: CV accuracy reported for classification
- [ ] `test_robustness_alternative_seed_pairs()`: Results stable across seed pairs

**Files to Create/Modify:**
- `src/statistical/robustness_checks_expanded.py`: Comprehensive robustness suite
- `src/statistical/neurips_analysis.py`: Include all robustness checks

---

### Step 5.2: Fix Temporal Stability

**Current Problem:** Temporal analysis failed (no date column)  
**Fix:** Add date, run analysis

**Implementation:**
1. **Add date column:**
   - Extract from comments (created_utc)
   - Merge into user-level dataset

2. **Temporal analysis:**
   - By year (2024 vs 2025)
   - By month (if enough data)
   - Stability test: R² similar across time

**Tests:**
- [ ] `test_temporal_stability_runs()`: Temporal analysis completes without errors
- [ ] `test_temporal_stability_reported()`: Results by time period reported
- [ ] `test_temporal_results_stable()`: R² variation < 0.001 across time

**Files to Modify:**
- `scripts/run_neurips_analysis.py`: Add date column from comments
- `src/statistical/neurips_analysis.py`: Fix temporal stability function

---

### Step 5.3: Handle Nonbinary Gender

**Current Problem:** n=43 is unreliable  
**Fix:** Remove from main analysis or explicitly acknowledge

**Implementation:**
1. **Option A:** Remove from main analysis
   - Keep in descriptive stats
   - Report separately as exploratory

2. **Option B:** Acknowledge limitation
   - Mark as exploratory/pilot
   - Don't claim significance
   - Discuss in limitations

3. **Prefer Option A** (cleaner main analysis)

**Tests:**
- [ ] `test_nonbinary_excluded_or_acknowledged()`: Main analysis excludes n<100 OR explicitly marks as exploratory
- [ ] `test_nonbinary_findings_marked()`: Nonbinary findings clearly marked as exploratory

**Files to Modify:**
- `src/statistical/regression_models.py`: Exclude nonbinary from main analysis
- `results/tables/regression_results.txt`: Mark nonbinary as exploratory

---

## Phase 6: Framing and Reporting

### Step 6.1: Honest Framing

**Current Problem:** Framing overstates certainty  
**Fix:** Reframe honestly

**Implementation:**
1. **Abstract/Introduction:**
   - Mention measurement limitations
   - Frame as "effects too small to detect" not "no effects"
   - Lead with subreddit findings

2. **Results:**
   - Discuss subreddit effects first
   - Then demographic nulls
   - Always with measurement caveats

3. **Discussion:**
   - Acknowledge limitations prominently
   - Discuss what CAN and CAN'T be concluded

**Tests:**
- [ ] `test_abstract_mentions_limitations()`: Abstract/intro mentions measurement limitations
- [ ] `test_subreddit_discussed_first()`: Subreddit effects discussed before demographic nulls
- [ ] `test_framing_honest()`: Uses "too small to detect" not "no effects exist"

**Files to Modify:**
- `docs/paper/abstract.md`: Reframe abstract
- `docs/paper/introduction.md`: Lead with subreddit findings
- `docs/paper/discussion.md`: Honest limitations discussion

---

### Step 6.2: Comprehensive Reporting

**Current Problem:** Not all diagnostics reported  
**Fix:** Report everything

**Implementation:**
1. **Tables:**
   - VIF for all predictors
   - Residual diagnostics (BP test, JB test)
   - Bootstrap CIs
   - Power analysis results

2. **Figures:**
   - Residual plots
   - Distribution plots
   - Robustness check summaries

**Tests:**
- [ ] `test_all_diagnostics_reported()`: VIF, residual tests, bootstrap CIs in tables
- [ ] `test_limitations_section_complete()`: Limitations section addresses all 16 criticisms

**Files to Create/Modify:**
- `src/statistical/generate_tables.py`: Include all diagnostics
- `docs/paper/limitations.md`: Address all criticisms

---

## Implementation Strategy

### Order of Implementation

1. **Week 1: Test Framework + Classification Improvements**
   - Phase 0: Test framework
   - Phase 1: Classification accuracy + ensemble

2. **Week 2: Statistical Rigor**
   - Phase 2: Multicollinearity, heteroscedasticity, influential observations

3. **Week 3: Missing Data + Real Findings**
   - Phase 3: Missing data, 3-bucket justification
   - Phase 4: Subreddit analysis, AnthroScore distribution

4. **Week 4: Robustness + Framing**
   - Phase 5: Robustness checks, temporal stability, nonbinary
   - Phase 6: Framing and reporting

### Testing Philosophy

**For EVERY fix:**
1. Write test FIRST (TDD approach)
2. Test should FAIL initially (proves it catches the problem)
3. Implement fix
4. Test should PASS (proves fix works)
5. Document the fix

### Success Metrics

Each phase must achieve:
- All tests passing
- Measurable improvement (accuracy up, VIF down, etc.)
- Documentation updated
- No regressions (existing tests still pass)

---

## Risk Mitigation

**Risk:** Fixes break existing functionality  
**Mitigation:** Run full test suite after each change

**Risk:** Improvements are marginal  
**Mitigation:** Define clear success criteria (e.g., accuracy >55%)

**Risk:** Takes too long  
**Mitigation:** Prioritize high-impact fixes first (classification, measurement error)

---

## Final Validation

Before considering fixes complete:
- [ ] All 16 criticisms addressed
- [ ] All tests passing
- [ ] Measurable improvements documented
- [ ] Paper rewritten with honest framing
- [ ] Results reproducible (code, data, seeds)

