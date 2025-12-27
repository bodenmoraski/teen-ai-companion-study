# FIXES PLAN: Resolving Critical NeurIPS Issues

**Date:** December 26, 2025  
**Status:** Active Development  
**Priority:** CRITICAL - Must fix before any further analysis

---

## Executive Summary

After deep investigation, I identified 4 ROOT CAUSES for why our classifications are failing:

| Root Cause | Problem | Impact | Fix |
|------------|---------|--------|-----|
| 1. Percentile age bucketing | Creates equal 20% per bucket | ~20% accuracy (random) | Use calibrated fixed thresholds |
| 2. Gender seed pair "oney" | Only 1 user | Gender dimension is broken | Replace with better subreddit |
| 3. Gender score shift | Both genders have negative scores | Can't separate them | Center scores before classifying |
| 4. Ensemble ignoring self-declaration | LLM/community override self-reports | Loses ground truth | Self-declaration must be authoritative |

---

## ROOT CAUSE #1: Age Percentile Bucketing

### Problem
The current `age_score_to_bucket_fixed()` uses percentile-based thresholds:
```python
p20, p40, p60, p80 = np.percentile(all_scores, [20, 40, 60, 80])
```

This creates EXACTLY 20%/20%/20%/20%/20% distribution regardless of actual ages.

### Evidence
```
age_bucket_community distribution:
  61-80: 5904 (20.0%)
  13-18: 5903 (20.0%)
  26-40: 5903 (20.0%)
  19-25: 5903 (20.0%)
  41-60: 5903 (20.0%)
```

### Fix
Use FIXED thresholds calibrated on self-declarations:
```
Self-declared age scores:
  13-18: mean=-0.30
  19-25: mean=-0.23
  26-40: mean=-0.12
  41-60: mean=-0.20
  61-80: mean=0.13
```

New calibrated thresholds:
- score < -0.26 → 13-18
- score < -0.17 → 19-25
- score < -0.05 → 26-40
- score < 0.10 → 41-60
- else → 61-80

---

## ROOT CAUSE #2: Gender Seed Pair Failure

### Problem
The seed pair `("thegirlsurvivalguide", "oney")` has only 1 user in "oney":
```
thegirlsurvivalguide: 336 users
oney: 1 user
```

This makes the seed pair nearly useless and biases the gender dimension.

### Fix
Replace with better subreddits that have more users:
```
Better options in data:
  askmen: 763 users
  mensrights: 165 users
  men: ? users (check)
  malefashionadvice: ? users (check)
```

**New seed pairs:**
1. ("askwomen", "askmen") - 442 vs 763 users ✓
2. ("twoxchromosomes", "mensrights") - 499 vs 165 users ✓
3. ("askwomenadvice", "malelivingspace") - check availability

---

## ROOT CAUSE #3: Gender Score Shift

### Problem
Both males and females have NEGATIVE mean scores:
```
male: mean=-0.1819
female: mean=-0.2352
```

The entire distribution is shifted negative, so threshold of 0 doesn't work.

### Fix
**Option A: Center scores** - Subtract the population mean before classifying
**Option B: Use relative thresholds** - Classify based on score percentiles within self-declared population
**Option C: Use optimal threshold** - Find the score that best separates males from females

The optimal separation point is around -0.21 (midpoint between -0.18 and -0.24).

---

## ROOT CAUSE #4: Self-Declaration Not Being Used

### Problem
The ensemble gives equal weight to all methods, even when self-declaration exists.
Self-declarations are GROUND TRUTH but get diluted by noisy community/LLM predictions.

### Fix
Self-declaration should be AUTHORITATIVE:
```python
if self_declared is not None:
    return self_declared  # No ensemble needed
else:
    return ensemble_vote(community, llm)
```

---

## EFFECT SIZE ROOT CAUSE

### Problem
Effect sizes (R² ≈ 0) might be because:
1. True null effect (demographics don't predict anthropomorphization)
2. Measurement error in demographics obscures real effects
3. Missing control variables

### After Demographic Fixes
We should expect R² to increase if there's a real effect.
If R² stays near 0, we report a "null result" - demographics don't predict anthropomorphization.

---

## Implementation Order

### Phase 1: Diagnostic Tests (1 hour)
1. Write comprehensive tests for all classification functions
2. Verify current behavior
3. Establish baselines

### Phase 2: Fix Age Classification (2 hours)
1. Implement calibrated thresholds based on self-declarations
2. Test against ground truth
3. Verify accuracy improves

### Phase 3: Fix Gender Classification (2 hours)
1. Find better seed pairs
2. Center gender scores
3. Test against ground truth
4. Verify accuracy improves

### Phase 4: Fix Ensemble Logic (1 hour)
1. Make self-declarations authoritative
2. Update ensemble weights
3. Re-run classification

### Phase 5: Re-run Analysis (2 hours)
1. Regenerate demographics with fixes
2. Re-run regression analysis
3. Check if effect sizes improve

### Phase 6: Validation (1 hour)
1. Run full validation suite
2. Generate updated metrics
3. Document improvements

---

## Success Criteria

| Metric | Current | Target | NeurIPS Minimum |
|--------|---------|--------|-----------------|
| Age accuracy | 34.75% | 50%+ | 60%+ |
| Age Cohen's κ | 0.035 | 0.30+ | 0.40+ |
| Gender accuracy | 0% | 50%+ | 60%+ |
| Gender Cohen's κ | 0.00 | 0.30+ | 0.40+ |
| Age-AnthroScore corr | 0.0003 | 0.05+ | 0.10+ (or null finding) |

---

## Files to Modify

1. `src/demographics/community_embedding_fixed.py` - Fix thresholds and seed pairs
2. `src/demographics/ensemble_classifier.py` - Make self-declaration authoritative
3. `scripts/rerun_with_fixes.py` - Re-run full pipeline
4. `tests/test_classification_fixes.py` - New comprehensive tests

---

## Next Step

Immediately proceed to FIXES-TODO.md and start implementing fixes.

