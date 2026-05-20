# Reproducibility Checklist

This document helps researchers verify they have all necessary files and can reproduce the paper's results.

## Quick Verification

Run this command to verify all essential files are present:

```bash
python scripts/validate_paper_statistics.py
```

Expected output should show:
- Inclusive sample: N = 16,347
- Conditional sample: n = 5,160
- All statistics matching paper claims

---

## Essential Files Checklist

### Core Data
- [ ] `Data/processed/all_comments.parquet` (~31 MB)
  - 283,895 cleaned Reddit comments
  - Required for all analyses

### AnthroScore Files
- [ ] `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet` (~14 MB)
  - LLM-scored anthropomorphization (1-5 scale)
  - Primary outcome variable

### Demographics Predictions
- [ ] `experiments/v2_correction/age_predictions_v4.parquet` (~2 MB)
  - Age predictions (teen/adult)
  - **This version produces paper statistics**

- [ ] `experiments/v2_correction/gender_predictions_v4.parquet` (~2 MB)
  - Gender predictions (male/female)
  - **This version produces paper statistics**

### Feature Files
- [ ] `Data/features/user_emotions.parquet` (~3.6 MB)
  - User-level emotion scores (7 emotions)

- [ ] `Data/features/user_anthroscores.parquet` (~900 KB)
  - User-level AnthroScore aggregates

- [ ] `Data/features/self_declarations.parquet` (~680 KB)
  - Self-reported demographics (ground truth)

### Human Validation Data
- [ ] `Validations/HUMAN_VALIDATION_ANSWER_KEY - Answer Key.csv`
- [ ] `Validations/STEPHANIE VALIDATION - Annotations (1).csv`
- [ ] `Validations/AFIA VALIDATION - Annotations (3).csv`
- [ ] `Validations/BODEN VALIDATION - Annotations (3).csv`

---

## Expected Statistics

When running `scripts/COMPREHENSIVE_V3_ANALYSIS.py`, you should get:

### Sample Sizes
| Sample | Expected N |
|--------|-----------|
| Inclusive (high-conf) | 16,347 |
| Conditional (anthro > 1) | 5,160 |

### Age Effect (Conditional Sample)
| Metric | Expected Value |
|--------|---------------|
| Teen Mean | 1.383 |
| Adult Mean | 1.602 |
| Cohen's d | -0.456 |

### Age Effect (Inclusive Sample)
| Metric | Expected Value |
|--------|---------------|
| Teen Mean | 1.108 |
| Adult Mean | 1.279 |
| Cohen's d | -0.507 |

### Gender Effect (Inclusive Sample)
| Metric | Expected Value |
|--------|---------------|
| Male Mean | 1.121 |
| Female Mean | 1.225 |
| Cohen's d | -0.307 |

---

## Version Information

### Component Versions

| Component | Version | File Pattern |
|-----------|---------|--------------|
| AnthroScore | V3 | `anthroscore_v3_*.parquet` |
| Demographics | V4 | `*_predictions_v4.parquet` |
| Emotion Model | DistilRoBERTa | `j-hartmann/emotion-english-distilroberta-base` |

### Why V4 Demographics?

The script `COMPREHENSIVE_V3_ANALYSIS.py` is named after **AnthroScore V3** (the anthropomorphization scorer), but uses **Demographics V4** (the stacked ensemble classifier). V4 demographics produce the paper's exact statistics.

---

## Troubleshooting

### Wrong Sample Sizes

If you see different sample sizes:
1. Verify you're using `*_predictions_v4.parquet` (not v3)
2. Check confidence threshold is 0.60
3. Ensure `anthroscore_v3_improved_final.parquet` is used

### Missing Files

If parquet files can't be read:
```bash
pip install pyarrow
```

### Statistics Don't Match

Run the validation script to compare:
```bash
python scripts/validate_paper_statistics.py
```

This generates `results/PAPER_VALIDATION_REPORT.md` with detailed comparisons.

---

## File Sizes Reference

| File | Expected Size |
|------|--------------|
| `all_comments.parquet` | ~31 MB |
| `anthroscore_v3_improved_final.parquet` | ~14 MB |
| `anthroscore_v3_full.parquet` | ~23 MB |
| `age_predictions_v4.parquet` | ~2 MB |
| `gender_predictions_v4.parquet` | ~2 MB |
| `user_emotions.parquet` | ~3.6 MB |
| `comments_with_emotions.parquet` | ~45 MB |

Total essential data: ~120 MB

---

## Contact

If you encounter issues reproducing results, please open a GitHub issue with:
1. The error message or unexpected output
2. Your Python version (`python --version`)
3. Output of `pip list | grep -E "pandas|numpy|scipy"`
