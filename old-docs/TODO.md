# TODO: Research Improvements

## Priority 1: Human Validation Setup ✅ COMPLETE
- [x] 1.1 Create stratified sample of 200 comments for annotation
- [x] 1.2 Generate annotation guidelines document
- [x] 1.3 Create annotation spreadsheet (CSV)
- [x] 1.4 Create inter-rater reliability calculator script

**Output:** `Data/annotations/` folder with all materials

## Priority 2: Robustness Checks ✅ COMPLETE
- [x] 2.1 Re-run key analyses at thresholds 0.5, 0.7, 0.8
- [x] 2.2 Run analyses with self-declared demographics only
- [x] 2.3 Bootstrap confidence intervals for key effect sizes

**Output:** `results/robustness/robustness_report.txt`

## Priority 3: Missing Statistical Analyses ✅ COMPLETE
- [x] 3.1 Three-way ANOVA: Age × Gender × Intent → AnthroScore
- [x] 3.2 Mediation analysis: Intent as mediator of Age → AnthroScore
- [x] 3.4 Nonlinear effects testing (quadratic, thresholds)

**Output:** `results/missing_analyses_report.txt`

## Priority 4: V2 Model Corrections ✅ COMPLETE
- [x] 4.1 Research SOTA solutions for noisy labels and class imbalance
- [x] 4.2 Build Age Predictor V2 (behavioral-only, golden set trained)
- [x] 4.3 Build Gender Predictor V2 (SMOTE + cost-sensitive + threshold optimization)
- [x] 4.4 Validate Age V2 against ground truth direction
- [x] 4.5 Validate Gender V2 female recall improvement
- [x] 4.6 Generate comprehensive comparison report

**Output:** `experiments/v2_correction/` folder with models and reports

## Priority 5: V3-V5 Model Optimization ✅ COMPLETE
- [x] 5.1 Build V3 with balanced improvements (no tradeoffs)
- [x] 5.2 Analyze V3 with confidence thresholds
- [x] 5.3 Build V4 with multi-algorithm stacking (XGBoost + LightGBM + RF)
- [x] 5.4 Build V5 hybrid with aggressive SMOTE-ENN
- [x] 5.5 Compare all versions and select best
- [x] 5.6 Document confidence-filtered performance
- [x] 5.7 Update MASTER_RESEARCH_FINDINGS.md

**Final Results (V3 with confidence ≥ 0.60):**
- Gender: 96.9% accuracy, 92.1% female recall (92.7% coverage) ✅
- Age: 95.0% accuracy, 97.2% teen recall (96.5% coverage) ✅

**Key Finding:** V3 with confidence filtering is production-ready:
- At 90% threshold: Gender 99.4% accuracy, Age 99.2% accuracy

**Output:** 
- `experiments/v2_correction/FINAL_MODEL_SUMMARY.md`
- `experiments/v2_correction/models/gender_v3/`
- `experiments/v2_correction/models/age_v3/`

---
*Completed: 2026-01-10*

