# Quick Reference Guide

**For:** Future AI agents and researchers  
**See:** `COMPREHENSIVE_PROJECT_SUMMARY.md` for full details

---

## What Is This?

Research project studying teen-AI companion relationships on Reddit. Analyzes 283k comments from 47k users to understand demographics, anthropomorphization, and emotional dynamics.

---

## Key Commands

```bash
# Main pipeline
python scripts/phase1_data_collection.py          # Data collection
python scripts/phase2_with_api_data.py            # Demographics
python scripts/targeted_phase3_phase4.py          # Analysis + Stats

# Utilities
python scripts/check_progress.py                  # Check status
python scripts/validate_demographics.py           # Validation
```

---

## Key Files

**Data:**
- `data/features/full_merged_dataset.parquet` - FINAL DATASET (all features)

**Results:**
- `results/tables/regression_results.txt` - RQ2 regression
- `results/figures/` - All plots

**Code:**
- `src/demographics/` - Age/gender classification (3 methods)
- `src/analysis/` - AnthroScore, BERTopic, emotions
- `src/statistical/` - Regression, stats, visualization

---

## Current Status

- ✅ Phase 1: Complete (283k comments, 47k users)
- ✅ Phase 2: Complete (18.5k age-classified, 17.4k gender-classified)
- 🟡 Phase 3/4: In progress (targeted re-run)
- ⚠️ Validation: Pending (critical for NeurIPS)

---

## Critical Gaps

1. **NO VALIDATION** - Need manual annotation, inter-rater reliability
2. **INCOMPLETE RQ3** - Emotional mirroring not fully implemented
3. **REGRESSION OUTPUT EMPTY** - Need to debug
4. **NO METHOD COMPARISON** - Need to compare 3 demographic methods
5. **NO ROBUSTNESS CHECKS** - Need sensitivity analysis

---

## NeurIPS Readiness: 7/10

**Good:** Large dataset, innovative methodology, complete pipeline, statistical rigor  
**Missing:** Validation, method comparison, robustness checks, complete RQ3

---

## What Could Tear Down Research

1. Validation failure (classifications wrong)
2. Selection bias (Reddit not representative)
3. Community embedding invalidity (seed pairs wrong)
4. Statistical issues (assumptions violated)
5. Ethical concerns (privacy/consent)

---

**For full details, see `COMPREHENSIVE_PROJECT_SUMMARY.md`**

