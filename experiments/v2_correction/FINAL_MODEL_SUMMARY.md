# Final Model Summary: V1 → V5 Evolution

## Executive Summary

After extensive experimentation (V1 → V5), **V3 is our recommended production model** with confidence filtering at ≥0.60.

### Key Results with V3 @ Confidence ≥0.60:

| Task | Accuracy | Minority Recall | Coverage |
|------|----------|-----------------|----------|
| Gender | **96.9%** | 92.1% (female) | 92.7% |
| Age | **95.0%** | 97.2% (teen) | 96.5% |

---

## Version Comparison

### Gender Prediction

| Version | Accuracy | Female Recall | Male Recall | Approach |
|---------|----------|---------------|-------------|----------|
| V1 (Baseline) | 81.3% | 44.0% | 95.0% | XGBoost ensemble |
| V3 | **94.8%** | **88.4%** | 97.2% | V1 + threshold optimization |
| V4 | 81.7% | 54.7% | 91.8% | Multi-algorithm stacking |
| V5 | 34.1% | 100.0% | 9.5% | Aggressive SMOTE-ENN (failed) |

**Winner: V3** - Threshold optimization on original distribution provides the best balance.

### Age Prediction

| Version | Accuracy | Teen Recall | Adult Recall | Approach |
|---------|----------|-------------|--------------|----------|
| V1 (Baseline) | 70.6% | N/A | N/A | Full stacked ensemble |
| V3 | **93.7%** | **96.5%** | 90.2% | Uses text embeddings |
| V4 | 61.4% | N/A | N/A | Multi-algorithm stacking |
| V5 | 51.0% | 59.8% | 40.0% | Behavioral-only (failed) |

**Winner: V3** - Text embeddings are essential for age prediction accuracy.

---

## Confidence-Filtered Performance (V3)

### Gender V3 with Confidence Thresholds

| Threshold | Coverage | Accuracy | F-Recall | M-Recall | Macro-F1 |
|-----------|----------|----------|----------|----------|----------|
| ≥ 0.50 | 100.0% | 94.8% | 88.4% | 97.2% | 93.4% |
| ≥ 0.55 | 96.9% | 96.0% | 90.6% | 97.9% | 94.8% |
| **≥ 0.60** | **92.7%** | **96.9%** | **92.1%** | **98.5%** | **95.8%** |
| ≥ 0.70 | 82.3% | 98.1% | 93.4% | 99.4% | 97.2% |
| ≥ 0.80 | 67.4% | 98.6% | 93.9% | 99.7% | 97.7% |
| ≥ 0.90 | 44.0% | 99.4% | 97.1% | 99.9% | 99.0% |
| ≥ 0.95 | 27.7% | 99.8% | 98.1% | 100.0% | 99.5% |

### Age V3 with Confidence Thresholds

| Threshold | Coverage | Accuracy | Teen Recall | Adult Recall |
|-----------|----------|----------|-------------|--------------|
| ≥ 0.50 | 100.0% | 93.7% | 96.5% | 90.2% |
| ≥ 0.55 | 98.0% | 94.2% | 96.4% | 91.5% |
| **≥ 0.60** | **96.5%** | **95.0%** | **97.2%** | **92.3%** |
| ≥ 0.70 | 90.0% | 97.1% | 98.7% | 95.0% |
| ≥ 0.80 | 87.4% | 98.0% | 99.1% | 96.6% |
| ≥ 0.85 | 84.5% | 98.7% | 100.0% | 97.1% |
| ≥ 0.90 | 83.2% | 99.2% | 100.0% | 98.2% |

---

## Key Learnings

### What Worked

1. **Threshold Optimization (Gender)**: Moving the classification threshold from 0.5 to ~0.35 dramatically improved female recall without destroying accuracy.

2. **Text Embeddings (Age)**: Sentence-BERT embeddings capture linguistic patterns essential for age prediction. Behavioral features alone are not sufficient.

3. **Confidence Filtering**: All models show significantly better accuracy at higher confidence thresholds. The 0.60 threshold provides the best tradeoff (>90% coverage with ~95%+ accuracy).

### What Didn't Work

1. **Behavioral-Only Age Model (V2/V5)**: Removing text embeddings to avoid "stereotype bias" reduced accuracy to ~50% (random). The model needs linguistic features.

2. **Aggressive SMOTE-ENN (V5)**: Over-resampling the minority class led to a model that predicts everything as the minority class.

3. **Pure Stacking (V4)**: Multi-algorithm stacking without specialized techniques (threshold optimization) didn't outperform V3.

---

## Final Recommendations

### For Maximum Accuracy (Recommended)
- Use V3 with confidence threshold ≥ 0.60
- Gender: 96.9% accuracy, 92.1% female recall, 92.7% coverage
- Age: 95.0% accuracy, 97.2% teen recall, 96.5% coverage

### For Maximum Statistical Power
- Use V3 with confidence threshold ≥ 0.50 (all predictions)
- Gender: 94.8% accuracy, 88.4% female recall, 100% coverage
- Age: 93.7% accuracy, 96.5% teen recall, 100% coverage

### For Highest Precision Analyses
- Use V3 with confidence threshold ≥ 0.80
- Gender: 98.6% accuracy, 93.9% female recall, 67.4% coverage
- Age: 98.0% accuracy, 99.1% teen recall, 87.4% coverage

---

## Model Files

Production models are located in:
- `experiments/v2_correction/models/gender_v3/gender_predictor_v3.pkl`
- `experiments/v2_correction/models/age_v3/age_predictor_v3.pkl`

Predictions are in:
- `experiments/v2_correction/gender_predictions_v3.parquet`
- `experiments/v2_correction/age_predictions_v3.parquet`

---

*Generated: 2026-01-10*
