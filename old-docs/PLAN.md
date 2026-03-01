# Execution Plan

## Status: ALL PHASES COMPLETE ✅

### Phase 1: Human Validation Setup ✅ COMPLETE
**Status**: ✅ Complete

| Task | Status | Output |
|------|--------|--------|
| 1.1 Stratified sample | ✅ Complete | `Data/annotations/sample_for_annotation.parquet` |
| 1.2 Guidelines | ✅ Complete | `Data/annotations/ANNOTATION_GUIDELINES.md` |
| 1.3 Spreadsheet | ✅ Complete | `Data/annotations/annotation_sheet.csv` |
| 1.4 IRR calculator | ✅ Complete | `scripts/calculate_irr.py` |

### Phase 2: Robustness Checks ✅ COMPLETE
**Status**: ✅ Complete

| Task | Status | Output |
|------|--------|--------|
| 2.1 Multiple thresholds | ✅ Complete | `results/robustness/robustness_report.txt` |
| 2.2 Self-declared only | ✅ Complete | `results/robustness/robustness_report.txt` |
| 2.3 Bootstrap CIs | ✅ Complete | `results/robustness/robustness_report.txt` |

### Phase 3: Missing Analyses ✅ COMPLETE
**Status**: ✅ Complete

| Task | Status | Output |
|------|--------|--------|
| 3.1 Three-way ANOVA | ✅ Complete | `results/missing_analyses_report.txt` |
| 3.2 Mediation | ✅ Complete | `results/missing_analyses_report.txt` |
| 3.4 Nonlinear | ✅ Complete | `results/missing_analyses_report.txt` |

---

### Phase 4: AnthroScore V3 (LLM-Based) ✅ COMPLETE
**Status**: ✅ Complete
**Date**: 2026-01-11

#### Summary
We developed and validated an LLM-based approach to measuring anthropomorphization that dramatically outperforms the existing MLM-based AnthroScore.

#### Key Results
| Metric | LLM (GPT-4.1-nano) | MLM (RoBERTa) |
|--------|-------------------|---------------|
| Correlation with Expert | **r = 0.590*** | r = 0.107 (n.s.) |
| Head-to-head wins | **83%** | 16% |
| Within-1 accuracy | **96%** | N/A |

**The LLM approach is 453% better correlated with expert judgments!**

#### Files Created
| File | Description |
|------|-------------|
| `experiments/anthroscore_v3/README.md` | Methodology overview |
| `experiments/anthroscore_v3/RESULTS.md` | Full validation results |
| `experiments/anthroscore_v3/anthroscore_llm.py` | LLM scorer implementation |
| `experiments/anthroscore_v3/create_test_set.py` | Test set creation |
| `experiments/anthroscore_v3/label_with_expert.py` | Expert labeling (GPT-4o) |
| `experiments/anthroscore_v3/validate_agreement.py` | Validation pipeline |
| `experiments/anthroscore_v3/compare_to_mlm.py` | LLM vs MLM comparison |
| `experiments/anthroscore_v3/run_full_dataset.py` | Full dataset processor |
| `experiments/anthroscore_v3/test_set_fully_labeled.parquet` | 100 labeled samples |
| `experiments/anthroscore_v3/validation_results.json` | Agreement metrics |

#### Cost Estimate
- Full dataset (283k comments): ~$17, ~39 hours
- Per-comment cost: ~$0.00006

#### Recommendation
**Replace MLM-based AnthroScore with LLM-based scoring** for all future analyses.

---
*Last Updated: 2026-01-11*

