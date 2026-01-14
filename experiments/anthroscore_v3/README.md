# AnthroScore V3: LLM-Based Anthropomorphization Measurement

This folder contains the AnthroScore V3 implementation and validation materials.

## Core Files

| File | Purpose |
|------|---------|
| `anthroscore_llm.py` | Main LLM-based scorer (synchronous) |
| `anthroscore_llm_parallel.py` | Parallel/async version for batch processing |
| `run_full_dataset_optimized.py` | Production pipeline for scoring full dataset |

## Data Files

| File | Description |
|------|-------------|
| `anthroscore_v3_full.parquet` | **Final scored dataset** (283,895 comments) |
| `test_set_expert_labeled.parquet` | Expert-labeled validation set (100 comments) |

## Validation Materials

| File | Purpose |
|------|---------|
| `human_validation_sample.csv` | Sample for human annotation (100 comments) |
| `HUMAN_VALIDATION_GUIDE.md` | Instructions for human annotators |
| `HUMAN_ANNOTATION_GUIDE.md` | Detailed annotation guidelines |
| `validation_results.json` | GPT-4.1-nano validation metrics |

## Results

| File | Contents |
|------|----------|
| `RESULTS.md` | Summary of validation results |
| `mlm_comparison_results.json` | V3 vs V2 comparison metrics |

## Key Metrics

- **Expert correlation:** r = 0.59 (vs r = 0.11 for V2)
- **Head-to-head accuracy:** 83% (vs 16% for V2)
- **Within-1 accuracy:** 96%
- **Total cost:** ~$8-10 for 283,895 comments

## Archive

The `archive/` folder contains development scripts, test files, and experimental approaches that were explored during V3 development.

---

*AnthroScore V3: Validated, Publication-Quality Measurement*
