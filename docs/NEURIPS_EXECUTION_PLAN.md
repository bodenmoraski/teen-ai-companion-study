# NeurIPS-Worthy Execution Plan
## Teen-AI Companion Relationships on Reddit

**Date:** December 26, 2025  
**Current Status:** 7/10 - Good foundation, critical gaps remain  
**Target:** 9.5/10 - NeurIPS-ready submission

---

## Executive Summary

You have a **strong research foundation** with:
- ✅ Large dataset (283k comments, 47k users, 100k+ subreddits)
- ✅ Innovative 3-method ensemble for demographics
- ✅ Complete analysis pipeline (AnthroScore, BERTopic, emotions)
- ✅ Statistical framework in place

**Critical gaps preventing NeurIPS acceptance:**
1. ⚠️ **Regression output broken** (file shows only "Total observations: 18390")
2. ⚠️ **No manual validation** (only self-declaration validation exists)
3. ⚠️ **No method comparison/ablation study**
4. ⚠️ **No robustness checks**
5. ⚠️ **RQ3 incomplete** (simplified emotion analysis, not true mirroring)

**Timeline to NeurIPS:** 8-12 weeks of focused work

---

## Phase 1: IMMEDIATE FIXES (Week 1) 🔴 CRITICAL

### 1.1 Fix Regression Output (Day 1-2)

**Problem:** `results/tables/regression_results.txt` is essentially empty - only shows observation count.

**Root Cause Analysis:**
- Models may not be fitting (check logs)
- `generate_regression_tables()` may be failing silently
- Data preparation may have issues

**Action Items:**
```bash
# 1. Debug regression execution
python -c "
import pandas as pd
from src.statistical.regression_models import run_rq2_regression, generate_regression_tables
df = pd.read_parquet('data/features/full_merged_dataset.parquet')
results = run_rq2_regression(df)
print('Models fitted:', {k: v is not None for k, v in results.items() if 'model' in k})
print('N observations:', results.get('n_observations'))
"

# 2. Check for errors in targeted_phase3_phase4.log
# 3. Manually test regression with sample data
# 4. Fix any issues in generate_regression_tables()
```

**Expected Output:**
- Full regression tables with coefficients, CI, p-values
- Effect sizes (Cohen's f², η²)
- Model fit statistics (R², AIC, BIC, F-stat)
- Model comparison (likelihood ratio tests)

**Deliverable:** `results/tables/regression_results.txt` with complete NeurIPS-level statistics

---

### 1.2 Run Existing Validation (Day 2-3)

**Action Items:**
```bash
# Run validation using self-declarations as ground truth
python scripts/validate_demographics.py
```

**Expected Output:**
- Accuracy, precision, recall, F1 for each method
- Confusion matrices
- Cohen's κ for inter-method agreement
- Saved to `results/validation/`

**Deliverable:** Validation report showing method performance on self-declared users

---

### 1.3 Method Comparison Analysis (Day 3-5)

**Create new script:** `scripts/method_comparison_analysis.py`

**What to analyze:**
1. **Coverage analysis:**
   - % of users classified by each method
   - Overlap between methods
   - Users classified by 1, 2, or 3 methods

2. **Agreement analysis:**
   - Cohen's κ between each pair of methods
   - Agreement when multiple methods available
   - Disagreement patterns (which buckets disagree most?)

3. **Ablation study:**
   - Remove each method, see impact on coverage/accuracy
   - Ensemble vs. individual methods

4. **Confidence calibration:**
   - Are high-confidence predictions more accurate?
   - Confidence distribution by method

**Deliverable:** `results/method_comparison_report.txt` with tables and analysis

---

## Phase 2: VALIDATION (Weeks 2-4) 🔴 CRITICAL

### 2.1 Design Manual Annotation Protocol (Week 2, Day 1-2)

**Create:** `data/annotations/annotation_protocol.md`

**Protocol should include:**
1. **Age classification guidelines:**
   - How to interpret comment history
   - What signals indicate each age bucket
   - Edge cases (e.g., mature 17-year-old vs. immature 19-year-old)
   - Examples for each bucket

2. **Gender classification guidelines:**
   - How to identify self-declarations
   - How to interpret community participation
   - Handling nonbinary/unknown cases

3. **Annotation interface:**
   - CSV with columns: `user_id`, `comment_sample`, `annotator_id`, `age_bucket`, `gender`, `confidence`, `notes`
   - Or use annotation tool (Prodigy, Label Studio, etc.)

**Deliverable:** Complete annotation protocol document

---

### 2.2 Create Annotation Sample (Week 2, Day 3-5)

**Create script:** `scripts/create_annotation_sample.py`

**Sampling strategy:**
- **Stratified sample:** Ensure representation across:
  - All age buckets (even if small n)
  - All gender categories
  - Different confidence levels
  - Different method combinations
- **Target:** 300-500 users (minimum 200 for statistical power)
- **Include:** User ID, sample comments (5-10 most recent), current classifications

**Output format:**
```csv
user_id,comment_sample_1,comment_sample_2,...,age_bucket_current,gender_current,confidence_current
```

**Deliverable:** `data/annotations/annotation_sample.csv`

---

### 2.3 Manual Annotation (Weeks 2-4)

**Requirements:**
- **2-3 annotators** (independent)
- **200-500 users** (aim for 300+)
- **Inter-annotator agreement** target: Krippendorff's α > 0.8 (age), Cohen's κ > 0.7 (gender)

**Process:**
1. Annotators work independently
2. Calculate inter-rater reliability after first 50 users
3. Resolve disagreements through discussion
4. Update protocol if needed
5. Complete remaining annotations

**Deliverable:** `data/annotations/manual_annotations.csv` with all annotator labels

---

### 2.4 Validation Analysis (Week 4)

**Create script:** `scripts/validate_against_manual_annotations.py`

**Analysis:**
1. **Inter-annotator reliability:**
   - Krippendorff's α for age (ordinal)
   - Cohen's κ for gender (nominal)
   - Per-bucket agreement

2. **Method accuracy (vs. ground truth):**
   - Accuracy, precision, recall, F1 for each method
   - Confusion matrices
   - Per-bucket performance

3. **Ensemble performance:**
   - How well does ensemble match ground truth?
   - Which method combinations work best?

4. **Error analysis:**
   - What types of users are misclassified?
   - Systematic biases?

**Deliverable:** `results/validation/manual_validation_report.txt` with full metrics

---

## Phase 3: ROBUSTNESS & RIGOR (Weeks 5-6) 🟡 HIGH PRIORITY

### 3.1 Sensitivity Analysis (Week 5, Day 1-3)

**Create script:** `scripts/sensitivity_analysis.py`

**What to vary:**
1. **Community embedding parameters:**
   - Different seed pairs
   - Different percentile thresholds for age buckets
   - Different vector dimensions

2. **Ensemble weights:**
   - What if we change weights?
   - Remove one method entirely

3. **LLM classification:**
   - Different models (GPT-4o-mini vs. GPT-4o)
   - Different prompts
   - Different confidence thresholds

**Output:** Report showing how results change with parameter variations

**Deliverable:** `results/robustness/sensitivity_analysis.txt`

---

### 3.2 Temporal Stability (Week 5, Day 4-5)

**Create script:** `scripts/temporal_analysis.py`

**Analysis:**
- Split data by time period (e.g., Q1 2024, Q2 2024, etc.)
- Run demographics classification on each period
- Compare distributions over time
- Check if AnthroScore patterns are stable

**Questions:**
- Do age/gender distributions change over time?
- Are results consistent across time periods?
- Any seasonal effects?

**Deliverable:** `results/robustness/temporal_stability.txt`

---

### 3.3 Subreddit-Level Analysis (Week 6, Day 1-3)

**Create script:** `scripts/subreddit_analysis.py`

**Analysis:**
- Analyze each subreddit separately (r/CharacterAI, r/Replika, r/AICompanions)
- Compare demographics across subreddits
- Compare AnthroScore patterns
- Check if results generalize across platforms

**Questions:**
- Are users in r/CharacterAI different from r/Replika?
- Do findings hold across all subreddits?
- Any platform-specific patterns?

**Deliverable:** `results/robustness/subreddit_comparison.txt`

---

### 3.4 Bias Analysis (Week 6, Day 4-5)

**Create script:** `scripts/bias_analysis.py`

**Check for:**
1. **Selection bias:**
   - Are certain groups underrepresented?
   - Who is missing from our dataset?

2. **Classification bias:**
   - Are certain groups systematically misclassified?
   - Gender bias in age classification?
   - Age bias in gender classification?

3. **Measurement bias:**
   - Does AnthroScore work equally well for all groups?
   - Are emotion classifications biased?

**Deliverable:** `results/robustness/bias_analysis.txt`

---

## Phase 4: COMPLETE RQ3 (Weeks 7-8) 🟡 MEDIUM PRIORITY

### 4.1 Assess RQ3 Feasibility (Week 7, Day 1)

**Current state:** Simplified emotion analysis (comment-level emotions, not true mirroring)

**What's needed for true RQ3:**
- User-AI conversation pairs
- Sentence-level emotion parsing (user vs. AI)
- Temporal emotion trajectories
- Similarity metrics (cosine similarity, DTW)

**Feasibility check:**
- Can we collect AI companion responses from Reddit?
- Do users share screenshots/conversations?
- Is there enough data for mirroring analysis?

**Decision point:** If not feasible, reframe RQ3 or acknowledge limitation

---

### 4.2 Implement True RQ3 (Week 7-8, if feasible)

**If feasible, create:** `scripts/rq3_emotional_mirroring.py`

**Steps:**
1. Extract user-AI conversation pairs from comments
2. Parse sentences (user vs. AI)
3. Classify emotions for each sentence
4. Calculate emotion trajectories over time
5. Compute similarity metrics (cosine, DTW)
6. Analyze mirroring patterns by demographics

**Deliverable:** `results/rq3_emotional_mirroring.txt` with analysis

---

### 4.3 Alternative: Reframe RQ3 (Week 7-8, if not feasible)

**Alternative research questions:**
- "How do users describe their AI companions' emotions?"
- "What emotional patterns emerge in AI companion discussions?"
- "How do user emotions relate to their AI companion usage patterns?"

**Action:** Update methodology section, run analysis, update results

---

## Phase 5: STATISTICAL RIGOR (Week 9) 🟡 HIGH PRIORITY

### 5.1 Model Diagnostics (Week 9, Day 1-3)

**Create script:** `scripts/model_diagnostics.py`

**Check regression assumptions:**
1. **Normality of residuals:** Q-Q plots, Shapiro-Wilk test
2. **Homoscedasticity:** Breusch-Pagan test, residual plots
3. **Linearity:** Residual vs. fitted plots
4. **Multicollinearity:** VIF scores
5. **Outliers:** Cook's distance, leverage

**If assumptions violated:**
- Use robust standard errors
- Transform variables if needed
- Remove outliers (with justification)
- Use alternative models (e.g., quantile regression)

**Deliverable:** `results/statistics/model_diagnostics.txt` + diagnostic plots

---

### 5.2 Multiple Comparison Correction (Week 9, Day 4-5)

**Apply corrections:**
- Bonferroni correction for multiple tests
- FDR (Benjamini-Hochberg) for exploratory analyses
- Report both corrected and uncorrected p-values

**Update regression tables** with corrected p-values

**Deliverable:** Updated regression results with corrections

---

## Phase 6: PAPER WRITING (Weeks 10-12) 🔴 CRITICAL

### 6.1 Paper Structure (Week 10)

**NeurIPS format:** 8 pages + references (main text)

**Sections:**
1. **Abstract** (150 words)
2. **Introduction** (1 page)
   - Motivation (64% of teens use AI chatbots)
   - Research questions
   - Contributions
3. **Related Work** (1 page)
   - AI companions and teens
   - Anthropomorphization
   - Reddit demographics
4. **Methodology** (2 pages)
   - Data collection
   - 3-method ensemble (novel contribution)
   - AnthroScore, BERTopic, emotions
   - Statistical analysis
5. **Results** (2 pages)
   - Demographics (RQ1)
   - Anthropomorphization patterns (RQ2)
   - Emotional dynamics (RQ3)
6. **Discussion** (1 page)
   - Key findings
   - Limitations
   - Policy implications
7. **Conclusion** (0.5 pages)

**Deliverable:** Paper outline and first draft

---

### 6.2 Figures & Tables (Week 11)

**Required figures:**
1. Age distribution (already exists, refine)
2. AnthroScore by demographics (already exists, refine)
3. Method comparison (NEW)
4. Validation results (NEW)
5. Emotion distribution (already exists, refine)
6. Robustness checks (NEW - if space allows)

**Required tables:**
1. Descriptive statistics (refine)
2. Regression results (FIX - currently broken)
3. Validation metrics (NEW)
4. Method comparison (NEW)

**Deliverable:** All publication-ready figures and tables

---

### 6.3 Writing & Revision (Weeks 11-12)

**Writing process:**
1. Write methods section (most technical)
2. Write results section (data-driven)
3. Write introduction & related work
4. Write discussion & conclusion
5. Polish abstract
6. Get feedback from co-authors/colleagues
7. Revise based on feedback

**Quality checklist:**
- [ ] Clear narrative flow
- [ ] All claims supported by data
- [ ] Limitations honestly discussed
- [ ] Contributions clearly stated
- [ ] Related work comprehensive
- [ ] Figures/tables clear and informative
- [ ] Code/data available (reproducibility)

**Deliverable:** Final paper draft ready for submission

---

## Phase 7: REPRODUCIBILITY (Week 12) 🟡 REQUIRED

### 7.1 Code Documentation

**Ensure:**
- All functions have docstrings
- All scripts have usage instructions
- Configuration files documented
- README updated with full instructions

---

### 7.2 Reproducibility Package

**Create:**
- `REPRODUCIBILITY.md` with:
  - Environment setup
  - Data access instructions
  - Step-by-step execution guide
  - Expected outputs
- `requirements.txt` with exact versions
- Random seeds set everywhere
- Docker container (optional but recommended)

---

### 7.3 Data Documentation

**Create:**
- `DATA_DOCUMENTATION.md` with:
  - Data sources
  - Collection dates
  - Preprocessing steps
  - Schema documentation
  - Privacy considerations

---

## Priority Matrix

### 🔴 CRITICAL (Must do for NeurIPS):
1. Fix regression output (Week 1)
2. Manual validation (Weeks 2-4)
3. Paper writing (Weeks 10-12)

### 🟡 HIGH PRIORITY (Strongly recommended):
4. Method comparison (Week 1)
5. Robustness checks (Weeks 5-6)
6. Statistical rigor (Week 9)

### 🟢 MEDIUM PRIORITY (Nice to have):
7. Complete RQ3 (Weeks 7-8)
8. Reproducibility package (Week 12)

---

## Success Metrics

### Minimum Viable (Risky):
- ✅ Regression output fixed
- ✅ Manual validation (200 users, 2 annotators, α > 0.7)
- ✅ Method comparison analysis
- ✅ Basic robustness checks
- ✅ Paper draft complete

### Recommended (Higher Quality):
- ✅ Regression output fixed
- ✅ Manual validation (300+ users, 3 annotators, α > 0.8)
- ✅ Comprehensive method comparison
- ✅ Full robustness checks (sensitivity, temporal, subreddit, bias)
- ✅ Model diagnostics
- ✅ Complete RQ3 (if feasible)
- ✅ High-quality paper with all figures/tables

---

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Immediate Fixes | Fixed regression, validation report, method comparison |
| 2-4 | Validation | Annotation protocol, manual annotations, validation report |
| 5-6 | Robustness | Sensitivity, temporal, subreddit, bias analyses |
| 7-8 | RQ3 | Complete or reframe emotional mirroring |
| 9 | Statistical Rigor | Model diagnostics, multiple comparison correction |
| 10-12 | Paper Writing | Full paper draft, figures, tables |
| 12 | Reproducibility | Documentation, reproducibility package |

**Total: 12 weeks (3 months)**

---

## Risk Mitigation

### Risk 1: Manual validation shows low accuracy
**Mitigation:** Start validation early (Week 2), iterate on methods if needed

### Risk 2: Regression still broken after fixes
**Mitigation:** Debug thoroughly, consider alternative statistical approaches

### Risk 3: Not enough time for all robustness checks
**Mitigation:** Prioritize critical checks (sensitivity, bias), skip less critical ones

### Risk 4: RQ3 not feasible
**Mitigation:** Reframe RQ3 early (Week 7), don't waste time on impossible analysis

### Risk 5: Paper writing takes longer than expected
**Mitigation:** Start writing early (Week 10), get feedback early, iterate quickly

---

## Next Steps (IMMEDIATE)

1. **TODAY:** Debug regression output
2. **THIS WEEK:** Run existing validation, create method comparison analysis
3. **NEXT WEEK:** Design annotation protocol, create annotation sample
4. **WEEK 3:** Start manual annotation

**Let's make this NeurIPS-worthy!** 🎯

