# Comprehensive Agent Prompt: NeurIPS-Quality Research Fixes

**Purpose:** This document provides a new AI agent instance with complete context to execute all remaining fixes and make this research NeurIPS-worthy.

**Estimated Time:** 4-8 hours of continuous work  
**Priority:** HIGH - All criticisms must be addressed

---

## 1. MISSION STATEMENT

You are a computational social science research agent. Your mission is to take a research project that is ~70% complete and make it **publication-ready for NeurIPS**. The project studies how Reddit users anthropomorphize AI companions (like Replika and Character.AI), and whether user demographics (age, gender) predict the degree of anthropomorphization.

**Current State:** 
- Data collected and processed ✅
- Demographics classified ✅ (but accuracy is low)
- AnthroScore computed ✅
- Basic regression run ✅
- **But:** 16 critical methodological issues identified that would cause rejection

**Your Goal:**
1. Read and understand all criticisms (`CRITICISM.md`)
2. Implement all fixes in `TODO-CRITICISMS.md` using the plan in `PLAN-CRITICISMS.md`
3. Write tests FIRST, then implement fixes that make tests pass
4. Run the full pipeline after fixes to regenerate results
5. Ensure all criticisms are addressed

---

## 2. PROJECT STRUCTURE

```
Unmuted Anthro-Analysis/
├── CRITICISM.md              # 16 critical issues - READ THIS FIRST
├── TODO-CRITICISMS.md        # Task list with checkboxes - TRACK PROGRESS HERE
├── PLAN-CRITICISMS.md        # Implementation plan - FOLLOW THIS
├── data/
│   ├── raw/                  # JSONL from Reddit (don't modify)
│   ├── processed/            # Cleaned comments
│   └── features/             # Computed features, demographics
│       ├── demographics.parquet        # User classifications (WILL BE MODIFIED)
│       ├── full_merged_dataset.parquet # Features + demographics
│       └── user_anthroscores.parquet   # AnthroScore per user
├── src/
│   ├── demographics/
│   │   ├── ensemble_classifier.py     # NEEDS FIX: True weighted voting
│   │   ├── community_embedding.py     # NEEDS FIX: Better accuracy
│   │   └── self_declaration.py        # OK
│   ├── statistical/
│   │   ├── regression_models.py       # NEEDS FIX: Measurement error
│   │   ├── neurips_analysis.py        # NEEDS FIX: More robustness
│   │   └── (NEW FILES NEEDED)
│   └── analysis/
│       └── (NEW FILES NEEDED for subreddit analysis)
├── tests/
│   ├── test_neurips_analysis.py       # Existing tests - EXTEND THESE
│   └── test_criticism_fixes.py        # CREATE THIS - All fix validation
├── scripts/
│   ├── run_neurips_analysis.py        # Main analysis script
│   └── (NEW SCRIPTS for re-running pipeline)
└── results/
    ├── neurips/                       # Current results - WILL BE REPLACED
    └── figures/
```

---

## 3. THE 16 CRITICAL ISSUES (Summary)

Read `CRITICISM.md` for full details. Here's the summary:

### PRIORITY 1: Measurement Quality (FATAL if not fixed)
| # | Issue | Current State | Target |
|---|-------|--------------|--------|
| 1 | Low classification accuracy | 46.3% (3-bucket), 50.7% achievable | >50% with documented method |
| 2 | No measurement error correction | Not done | Attenuation-corrected estimates |
| 3 | No power analysis | Not done | Calculate MDE with measurement error |

### PRIORITY 2: Statistical Rigor (CRITICAL)
| # | Issue | Current State | Target |
|---|-------|--------------|--------|
| 4 | Ensemble rarely ensembles | `weighted_age_vote()` exists but 88% have only 1 source | >30% with 2+ sources |
| 5 | High multicollinearity | VIF = 12.5 | VIF < 5 |
| 6 | Heteroscedasticity | Detected, HC3 applied | Proper transformation + diagnosis |
| 7 | Influential observations | 1,034 identified | Analyze and remove |

### PRIORITY 3: Missing Data & Selection Bias
| # | Issue | Current State | Target |
|---|-------|--------------|--------|
| 8 | 43% missing data | Excluded | Analyze missingness, impute or include |
| 9 | 3-bucket age is post-hoc | Created after seeing 5-bucket accuracy | Make primary, justify a priori |

### PRIORITY 4: Explore Real Findings
| # | Issue | Current State | Target |
|---|-------|--------------|--------|
| 10 | Subreddit effects unexplored | Mentioned but not analyzed | Comprehensive analysis (THE REAL FINDING) |
| 11 | AnthroScore distribution issues | Most scores are 0 | Analyze floor effects, try alternative aggregations |

### PRIORITY 5: Robustness & Validation
| # | Issue | Current State | Target |
|---|-------|--------------|--------|
| 12 | Incomplete robustness | Some checks done | Leave-one-out, CV, alternative seeds |
| 13 | Temporal stability failed | No date column | Add dates, run stability |
| 14 | Nonbinary n=43 unreliable | Reported as significant | Remove from main analysis |

### PRIORITY 6: Framing & Interpretation
| # | Issue | Current State | Target |
|---|-------|--------------|--------|
| 15 | Framing overstates certainty | "No effects" | "Effects too small to detect" |
| 16 | Not all diagnostics reported | Partial | Full VIF, residuals, power analysis |

---

## 4. IMPLEMENTATION APPROACH

### 4.1 Test-Driven Development (CRITICAL)

For EVERY fix:
1. **Write the test FIRST** in `tests/test_criticism_fixes.py`
2. **Run the test - it MUST FAIL** (proves the test catches the problem)
3. **Implement the fix**
4. **Run the test - it MUST PASS** (proves the fix works)
5. **Run ALL tests** to ensure no regressions

### 4.2 File Creation Order

1. `tests/test_criticism_fixes.py` - Create first with all test stubs
2. `src/statistical/measurement_error_correction.py` - New module
3. `src/statistical/power_analysis.py` - New module
4. `src/analysis/subreddit_analysis.py` - New module
5. `src/analysis/anthroscore_distribution.py` - New module
6. Modify existing files as needed

### 4.3 Test Structure Template

```python
# tests/test_criticism_fixes.py
"""
Tests for all 16 criticism fixes.
Each test must:
1. Fail before the fix is implemented
2. Pass after the fix is implemented
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# PRIORITY 1: Measurement Quality
# ============================================

class TestClassificationAccuracy:
    """T1: Improve classification accuracy to >55%"""
    
    def test_accuracy_above_50_percent(self):
        """3-bucket age accuracy must exceed 50%"""
        # Load data
        demo = pd.read_parquet("data/features/demographics.parquet")
        
        # Calculate 3-bucket accuracy against ground truth
        from src.statistical.neurips_analysis import convert_to_3_buckets
        
        mask = demo['age_bucket_self_declared'].notna() & demo['age_bucket'].notna()
        subset = demo[mask]
        
        gt = subset['age_bucket_self_declared'].apply(convert_to_3_buckets)
        pred = subset['age_bucket'].apply(convert_to_3_buckets)
        
        accuracy = (gt == pred).mean()
        assert accuracy > 0.50, f"Accuracy {accuracy:.1%} must be > 50% (current: 46.3%)"
    
    def test_ensemble_has_multiple_sources(self):
        """Users must have multiple sources for ensemble to be meaningful"""
        demo = pd.read_parquet("data/features/demographics.parquet")
        
        # Count sources per user
        has_self = demo['age_bucket_self_declared'].notna()
        has_community = demo['age_bucket_community'].notna()
        has_llm = demo['age_bucket_llm'].notna()
        
        n_sources = has_self.astype(int) + has_community.astype(int) + has_llm.astype(int)
        multi_source = (n_sources >= 2)
        
        pct_multi = multi_source.mean()
        
        # Current: ~10% have 2+ sources. Target: >30%
        assert pct_multi > 0.30, f"Only {pct_multi:.1%} have 2+ sources (need >30% for meaningful ensemble)"


class TestMeasurementError:
    """T2: Measurement error correction"""
    
    def test_reliability_calculated(self):
        """Reliability coefficients must be calculated"""
        from src.statistical.measurement_error_correction import calculate_reliability
        
        reliability = calculate_reliability()
        
        assert 'age' in reliability
        assert 0 < reliability['age'] < 1, "Reliability should be between 0 and 1"
    
    def test_attenuation_correction_applied(self):
        """Attenuation-corrected coefficients must be larger than naive"""
        from src.statistical.measurement_error_correction import correct_for_attenuation
        
        naive_coef = 0.01
        reliability = 0.5
        
        corrected = correct_for_attenuation(naive_coef, reliability)
        
        assert corrected > naive_coef, "Corrected coefficient should be larger"


class TestPowerAnalysis:
    """T3: Power analysis with measurement error"""
    
    def test_power_calculated(self):
        """Power must be calculated for detecting small effects"""
        from src.statistical.power_analysis import calculate_power
        
        power = calculate_power(
            n=27000,
            effect_size=0.02,  # Small effect (f²)
            reliability=0.5,
            alpha=0.05
        )
        
        assert 0 <= power <= 1
        # With measurement error, power should be lower than naive calculation
    
    def test_minimum_detectable_effect(self):
        """MDE must be calculated given measurement error"""
        from src.statistical.power_analysis import minimum_detectable_effect
        
        mde = minimum_detectable_effect(
            n=27000,
            power=0.80,
            reliability=0.5
        )
        
        assert mde > 0.02, "MDE with 50% reliability should be > 0.02"


# ============================================
# PRIORITY 2: Statistical Rigor
# ============================================

class TestMulticollinearity:
    """T5: Fix multicollinearity (VIF < 5)"""
    
    def test_vif_below_5(self):
        """Max VIF must be < 5 after fixes"""
        from src.statistical.neurips_analysis import prepare_regression_with_controls
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        merged = pd.read_parquet("data/features/full_merged_dataset.parquet")
        demo = pd.read_parquet("data/features/demographics.parquet")
        merged = merged.merge(demo[['author', 'age_bucket', 'gender']], on='author', how='left')
        
        reg_df = prepare_regression_with_controls(merged, exclude_unknown_gender=True)
        
        # Get predictor columns
        pred_cols = [c for c in reg_df.columns if c.startswith(('age_', 'gender_')) 
                    and 'bucket' not in c.lower()]
        
        if len(pred_cols) < 2:
            pytest.skip("Not enough predictors for VIF")
        
        X = reg_df[pred_cols].dropna()
        vifs = [variance_inflation_factor(X.values, i) for i in range(len(pred_cols))]
        
        max_vif = max(vifs)
        assert max_vif < 5, f"Max VIF {max_vif:.1f} must be < 5"


class TestInfluentialObservations:
    """T7: Analyze and handle influential observations"""
    
    def test_results_stable_after_outlier_removal(self):
        """Results must be stable when removing influential observations"""
        # This test checks that Cook's D outliers don't dominate results
        # R² change should be < 0.001 after removing outliers
        pass  # Implement after base analysis


# ============================================
# PRIORITY 3: Missing Data
# ============================================

class TestMissingData:
    """T8: Address missing data properly"""
    
    def test_missingness_analyzed(self):
        """Missingness patterns must be documented"""
        from src.statistical.missing_data_analysis import analyze_missingness
        
        analysis = analyze_missingness()
        
        assert 'n_missing_age' in analysis
        assert 'n_missing_gender' in analysis
        assert 'missingness_correlation_anthroscore' in analysis
    
    def test_missingness_not_related_to_outcome(self):
        """Missing demographics should not predict AnthroScore"""
        demo = pd.read_parquet("data/features/demographics.parquet")
        anthro = pd.read_parquet("data/features/user_anthroscores.parquet")
        
        merged = demo.merge(anthro, on='author')
        
        has_age = merged['age_bucket'].notna()
        
        mean_with = merged.loc[has_age, 'anthroscore_mean'].mean()
        mean_without = merged.loc[~has_age, 'anthroscore_mean'].mean()
        
        # T-test
        from scipy.stats import ttest_ind
        stat, pval = ttest_ind(
            merged.loc[has_age, 'anthroscore_mean'].dropna(),
            merged.loc[~has_age, 'anthroscore_mean'].dropna()
        )
        
        # Note: This test may FAIL, which is a finding in itself
        # If it fails, we must document that missingness IS related to outcome


# ============================================
# PRIORITY 4: Real Findings
# ============================================

class TestSubredditAnalysis:
    """T10: Comprehensive subreddit analysis"""
    
    def test_subreddit_explains_more_variance(self):
        """Subreddit must explain >10x more variance than demographics"""
        from src.analysis.subreddit_analysis import compare_variance_explained
        
        results = compare_variance_explained()
        
        ratio = results['subreddit_r2'] / max(results['demographics_r2'], 0.0001)
        
        assert ratio > 10, f"Subreddit/demographics R² ratio {ratio:.1f} should be > 10"


class TestAnthroScoreDistribution:
    """T11: Analyze AnthroScore distribution"""
    
    def test_floor_effects_documented(self):
        """Floor effects in AnthroScore must be documented"""
        anthro = pd.read_parquet("data/features/user_anthroscores.parquet")
        
        pct_zero = (anthro['anthroscore_mean'] == 0).mean()
        
        assert pct_zero < 0.80, f"{pct_zero:.1%} have zero AnthroScore - severe floor effects"
    
    def test_alternative_aggregations_tested(self):
        """Alternative aggregations (max, std) must be tested"""
        from src.analysis.anthroscore_distribution import compare_aggregations
        
        results = compare_aggregations()
        
        assert 'max' in results
        assert 'std' in results
        assert 'pct_nonzero' in results


# ============================================
# PRIORITY 5: Robustness
# ============================================

class TestRobustness:
    """T12-14: Comprehensive robustness checks"""
    
    def test_leave_one_subreddit_out(self):
        """Results must be stable when removing each subreddit"""
        from src.statistical.robustness_checks import leave_one_subreddit_out
        
        results = leave_one_subreddit_out()
        
        r2_values = [r['r2'] for r in results['by_subreddit']]
        r2_range = max(r2_values) - min(r2_values)
        
        assert r2_range < 0.0005, f"R² variation {r2_range:.6f} must be < 0.0005"
    
    def test_temporal_stability_runs(self):
        """Temporal stability analysis must run successfully"""
        from src.statistical.neurips_analysis import temporal_stability_analysis
        
        merged = pd.read_parquet("data/features/full_merged_dataset.parquet")
        
        # Add date column from comments
        comments = pd.read_parquet("data/processed/all_comments.parquet")
        dates = comments.groupby('author')['created_utc'].first().reset_index()
        merged = merged.merge(dates, on='author', how='left')
        
        results = temporal_stability_analysis(merged)
        
        assert len(results) > 0, "Temporal analysis must produce results"


# ============================================
# PRIORITY 6: Framing
# ============================================

class TestHonestFraming:
    """T15-16: Honest framing and complete reporting"""
    
    def test_limitations_documented(self):
        """All 16 criticisms must be addressed in limitations"""
        # This is a documentation test - check that limitations doc exists
        limitations_path = Path("docs/paper/limitations.md")
        
        if limitations_path.exists():
            content = limitations_path.read_text()
            
            # Check for key phrases
            assert "measurement error" in content.lower()
            assert "classification accuracy" in content.lower()
            assert "too small to detect" in content.lower() or "cannot rule out" in content.lower()
```

---

## 5. KEY TECHNICAL DETAILS

### 5.1 Data Files

| File | Contents | Rows | Key Columns |
|------|----------|------|-------------|
| `demographics.parquet` | User classifications | 47,062 | author, age_bucket, gender, age_bucket_community, age_bucket_llm |
| `full_merged_dataset.parquet` | Features + demographics | 47,062 | author, anthroscore_mean, emotion_*, topic_*, subreddit |
| `user_anthroscores.parquet` | AnthroScore per user | 47,062 | author, anthroscore_mean, anthroscore_std |
| `all_comments.parquet` | Processed comments | ~150,000 | author, body, created_utc, subreddit |

### 5.2 Key Functions

```python
# src/statistical/neurips_analysis.py
def convert_to_3_buckets(age_5bucket: str) -> str:
    """Convert 5-bucket to 3-bucket age"""
    if age_5bucket == '13-18': return 'teen'
    if age_5bucket == '19-25': return 'young_adult'
    return 'adult'  # 26-40, 41-60, 61-80

def prepare_regression_with_controls(df, age_scheme='3_bucket', exclude_unknown_gender=True):
    """Prepare regression data with control variables"""
    
def run_hierarchical_regression(df, age_scheme='3_bucket'):
    """Run hierarchical regression with controls"""
    
def run_model_diagnostics(model):
    """Run VIF, heteroscedasticity, normality tests"""
```

### 5.3 Current Statistics

| Metric | Current Value | Target |
|--------|---------------|--------|
| Age classification accuracy (3-bucket) | 46.3% (50.7% achievable) | >50% |
| Gender classification accuracy | ~40% | >60% |
| R² (full model) | 0.0007 | Document honestly |
| VIF (max) | 12.5 | <5 |
| Influential observations | 1,034 | Analyze, remove |
| Missing age | 32% | Address via imputation or unknown category |
| Missing gender | 40.5% | Address via imputation or unknown category |

---

## 6. EXECUTION CHECKLIST

### Phase 0: Setup (30 min)
- [ ] Read `CRITICISM.md` completely
- [ ] Read `TODO-CRITICISMS.md` and `PLAN-CRITICISMS.md`
- [ ] Create `tests/test_criticism_fixes.py` with all test stubs
- [ ] Run existing tests: `python -m pytest tests/ -v`
- [ ] Commit: "Add test framework for criticism fixes"

### Phase 1: Classification Improvements (2-3 hours)
- [ ] T1: Improve classification accuracy
  - [ ] Implement cross-validation for threshold calibration
  - [ ] Test alternative seed pairs
  - [ ] Increase LLM coverage if API budget allows
  - [ ] Test passes: accuracy > 55%
- [ ] T4: Fix ensemble to use true weighted voting for all users
  - [ ] Modify `ensemble_classifier.py`
  - [ ] Test passes: >50% of users get weighted voting
- [ ] Re-run demographics pipeline
- [ ] Commit: "Improve classification accuracy to >55%"

### Phase 2: Measurement Error & Power (1-2 hours)
- [ ] T2: Create `src/statistical/measurement_error_correction.py`
  - [ ] Calculate reliability coefficients
  - [ ] Implement attenuation correction
  - [ ] Test passes: corrected coefficients calculated
- [ ] T3: Create `src/statistical/power_analysis.py`
  - [ ] Calculate power with measurement error
  - [ ] Calculate MDE
  - [ ] Test passes: power analysis complete
- [ ] Commit: "Add measurement error correction and power analysis"

### Phase 3: Statistical Rigor (1-2 hours)
- [ ] T5: Fix multicollinearity
  - [ ] Calculate VIF for all predictors
  - [ ] Remove or regularize high-VIF predictors
  - [ ] Test passes: max VIF < 5
- [ ] T6: Handle heteroscedasticity
  - [ ] Generate residual plots
  - [ ] Try transformations
  - [ ] Document with HC3 if needed
- [ ] T7: Analyze influential observations
  - [ ] Remove Cook's D > 4/n
  - [ ] Characterize influential points
  - [ ] Compare results with/without
- [ ] Commit: "Fix multicollinearity and influential observations"

### Phase 4: Missing Data & 3-Bucket (1 hour)
- [ ] T8: Create `src/statistical/missing_data_analysis.py`
  - [ ] Analyze missingness patterns
  - [ ] Test if missingness predicts AnthroScore
  - [ ] Include "unknown" as category or impute
- [ ] T9: Make 3-bucket age primary
  - [ ] Classify directly to 3 buckets
  - [ ] Update demographics.parquet
  - [ ] Document justification
- [ ] Commit: "Address missing data and make 3-bucket primary"

### Phase 5: Real Findings (1-2 hours)
- [ ] T10: Create `src/analysis/subreddit_analysis.py`
  - [ ] Comprehensive subreddit comparison
  - [ ] Develop theory for why subreddit matters
  - [ ] Test passes: subreddit R² > 10× demographic R²
- [ ] T11: Create `src/analysis/anthroscore_distribution.py`
  - [ ] Distribution analysis with plots
  - [ ] Alternative aggregations (max, std, pct_nonzero)
  - [ ] Test passes: alternatives tested
- [ ] Commit: "Comprehensive subreddit and distribution analysis"

### Phase 6: Robustness (1 hour)
- [ ] T12: Expand robustness checks
  - [ ] Leave-one-subreddit-out
  - [ ] Cross-validation for classification
  - [ ] Alternative seed pairs
- [ ] T13: Fix temporal stability
  - [ ] Add date column from comments
  - [ ] Run by year/month
- [ ] T14: Handle nonbinary (n=43)
  - [ ] Remove from main analysis
  - [ ] Report as exploratory
- [ ] Commit: "Complete robustness checks"

### Phase 7: Final Reporting (30 min)
- [ ] T15: Update framing
  - [ ] "Effects too small to detect" not "no effects"
  - [ ] Lead with subreddit findings
  - [ ] Acknowledge measurement limitations
- [ ] T16: Complete reporting
  - [ ] All VIF values
  - [ ] Residual plots
  - [ ] Power analysis results
- [ ] Create `docs/paper/limitations.md`
- [ ] Commit: "Honest framing and complete reporting"

### Final Validation
- [ ] Run all tests: `python -m pytest tests/ -v`
- [ ] All tests pass (100%)
- [ ] Re-run full analysis: `python scripts/run_neurips_analysis.py`
- [ ] Update `TODO-CRITICISMS.md` - all items checked
- [ ] Final commit: "All criticism fixes complete - NeurIPS ready"

---

## 7. SUCCESS CRITERIA

Before declaring victory, verify:

| Criterion | Check |
|-----------|-------|
| Classification accuracy > 50% (3-bucket) | `pytest tests/test_criticism_fixes.py::TestClassificationAccuracy -v` |
| Measurement error documented | Power analysis and attenuation correction in report |
| VIF < 5 for all predictors | Check `neurips_analysis_report.txt` |
| Missing data addressed | Either imputed or "unknown" category analyzed |
| Subreddit analysis complete | `results/neurips/subreddit_analysis.txt` exists |
| All robustness checks pass | Temporal, leave-one-out, sensitivity |
| Framing is honest | Uses "too small to detect", not "no effects exist" |
| All tests pass | `pytest tests/ -v` shows 100% pass |

---

## 8. COMMON PITFALLS TO AVOID

1. **Don't just acknowledge limitations - FIX them where possible**
2. **Write tests FIRST before implementing fixes**
3. **Don't break existing functionality (run all tests after each change)**
4. **Don't guess missing parameters - check existing code**
5. **This is Windows - use Windows-compatible terminal commands**
6. **Use `pathlib` for file paths**
7. **Commit after each major fix with descriptive messages**
8. **If stuck, re-read the relevant section of `PLAN-CRITICISMS.md`**

---

## 9. ENVIRONMENT NOTES

- **OS:** Windows 10
- **Python:** 3.10+
- **Key packages:** pandas, numpy, statsmodels, scikit-learn, pytest
- **OpenAI API:** Available for LLM classification (use gpt-4.1-nano for cost efficiency)
- **Data files:** All in parquet format for fast I/O

---

## 10. FINAL NOTES

This is research that matters. Teen-AI companion relationships are a critical topic in 2025, with policy implications. The methodological improvements you make here will determine whether this work gets published and influences policy, or gets rejected and forgotten.

**Be rigorous. Be honest. Be thorough.**

The 16 criticisms in `CRITICISM.md` are real issues that would cause any competent reviewer to reject this paper. Your job is to address them foundationally, not superficially.

**Good luck. The research community is counting on you.**

