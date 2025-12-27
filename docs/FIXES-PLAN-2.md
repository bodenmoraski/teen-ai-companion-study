# FIXES PLAN 2: NeurIPS Reviewer Destruction Prevention

**Date:** December 26, 2025  
**Role:** Hostile NeurIPS Reviewer  
**Goal:** Make this paper rejection-proof

---

## 🔴 CRITICAL ISSUES THAT WILL GET US REJECTED

### ISSUE 1: Effect Sizes Are Negligible (FATAL)
**Reviewer Comment:** "The authors report R² = 0.0009. Demographics explain 0.09% of variance in anthropomorphization. Even with statistically significant p-values, this has no practical significance. The entire premise that demographics predict anthropomorphization is not supported."

**Root Cause:** 
- We're using "unknown" as reference category (40% of users)
- "Unknown" gender users may be fundamentally different (lurkers vs active)
- We're not controlling for confounds (subreddit, comment length, time)

**Fix:**
1. Change reference category to most common known group
2. Add control variables (subreddit, comment count, account age)
3. Use hierarchical regression to show incremental variance explained
4. Consider that null result IS a finding - frame appropriately

---

### ISSUE 2: Age Classification Still Below Target (HIGH)
**Reviewer Comment:** "Age classification accuracy of 37.6% is only marginally better than random (20%). With 5 buckets, this level of noise propagates into all downstream analyses, potentially explaining the null results."

**Root Cause:**
- Community embeddings alone aren't enough
- Self-declarations are rare (459 users)
- LLM classification is underutilized (only 5000 users)

**Fix:**
1. Collapse 5 buckets to 3 (teen/young adult/adult) for higher accuracy
2. Weight LLM predictions more heavily when available
3. Use ensemble confidence to filter uncertain classifications
4. Report sensitivity analysis with different classification schemes

---

### ISSUE 3: No Robustness Checks (HIGH)
**Reviewer Comment:** "No sensitivity analysis, no bootstrap confidence intervals, no temporal stability checks. How do we know these results aren't artifacts of arbitrary parameter choices?"

**Fix:**
1. Bootstrap CIs for all key estimates
2. Sensitivity analysis on age thresholds
3. Temporal split validation (2024 vs 2025)
4. Subreddit-level analysis (CharacterAI vs Replika)
5. Leave-one-out stability for demographic methods

---

### ISSUE 4: No Method Comparison (HIGH)
**Reviewer Comment:** "The authors claim a novel 3-method ensemble but provide no evidence it outperforms individual methods. Where is the ablation study?"

**Fix:**
1. Compare accuracy: Self-declaration vs Community vs LLM vs Ensemble
2. Calculate agreement (Cohen's κ) between methods
3. Show that ensemble has higher coverage AND accuracy
4. Ablation: What happens if we remove each method?

---

### ISSUE 5: Missing Control Variables (MEDIUM)
**Reviewer Comment:** "The regression doesn't control for obvious confounds like subreddit (CharacterAI vs Replika users differ), posting frequency, or account characteristics."

**Fix:**
1. Add subreddit fixed effects
2. Add comment count (activity level)
3. Add temporal controls (month/year)
4. Add text length controls

---

### ISSUE 6: Model Diagnostics Missing (MEDIUM)
**Reviewer Comment:** "No residual plots, no tests for heteroscedasticity, no influence diagnostics. Basic regression assumptions aren't verified."

**Fix:**
1. Generate residual vs fitted plots
2. Run Breusch-Pagan test for heteroscedasticity
3. Calculate Cook's distance for influential points
4. Use robust standard errors if needed

---

### ISSUE 7: RQ3 Still Incomplete (MEDIUM)
**Reviewer Comment:** "RQ3 asks about emotional mirroring but the paper only reports emotion distributions. This question is not answered."

**Fix:**
1. Reframe RQ3 as emotion EXPRESSION analysis (what we can measure)
2. Or remove RQ3 and focus on RQ1-RQ2
3. Add emotion × demographics analysis

---

### ISSUE 8: Multiple Comparison Problem (MEDIUM)
**Reviewer Comment:** "With 20+ coefficients tested, some will be significant by chance. No multiple comparison correction is applied."

**Fix:**
1. Apply Bonferroni or FDR correction
2. Focus on pre-registered hypotheses
3. Report both corrected and uncorrected p-values

---

### ISSUE 9: Unknown Gender Reference Category (MEDIUM)
**Reviewer Comment:** "Using 'unknown' gender as reference makes interpretation confusing. Unknown users are likely different in systematic ways."

**Fix:**
1. Exclude unknown gender users from gender analysis
2. Or use male as reference (largest known group)
3. Run separate analyses: all users vs known-gender-only

---

### ISSUE 10: No Cross-Validation (LOW)
**Reviewer Comment:** "Results are from a single train-test split. How do we know they generalize?"

**Fix:**
1. 5-fold cross-validation on classification
2. Bootstrap validation on regression
3. Report CI ranges

---

## Implementation Priority

| Priority | Issue | Time | Impact |
|----------|-------|------|--------|
| 1 | Add control variables | 2h | HIGH |
| 2 | 3-bucket age simplification | 1h | HIGH |
| 3 | Robustness checks | 2h | HIGH |
| 4 | Method comparison | 1h | HIGH |
| 5 | Multiple comparison correction | 30m | MEDIUM |
| 6 | Model diagnostics | 1h | MEDIUM |
| 7 | Known-gender-only analysis | 30m | MEDIUM |
| 8 | Bootstrap CIs | 1h | MEDIUM |
| 9 | Reframe RQ3 | 30m | LOW |

---

## Success Criteria

After fixes:
- [ ] R² improves to > 0.01 (at least 1% variance explained)
- [ ] OR null result is properly framed as finding
- [ ] Age accuracy > 50% with 3 buckets
- [ ] All robustness checks documented
- [ ] Method comparison shows ensemble value
- [ ] Multiple comparison corrections applied
- [ ] Model diagnostics pass
- [ ] Clear, interpretable findings

