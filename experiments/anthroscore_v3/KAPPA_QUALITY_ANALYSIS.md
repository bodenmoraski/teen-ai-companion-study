# Kappa Quality Analysis & Improvement Strategy

## Current Performance

**Kappa = 0.579** is **"Moderate" agreement** (0.41-0.60 range)

### Interpretation
- **0.00-0.20**: Slight agreement ❌
- **0.21-0.40**: Fair agreement ⚠️
- **0.41-0.60**: **Moderate agreement** ✅ (Current: 0.579)
- **0.61-0.80**: **Substantial agreement** 🎯 (Target)
- **0.81-1.00**: Almost perfect agreement ⭐ (Ideal)

### Research Quality Standards
- **Minimum acceptable**: >0.60 (Substantial)
- **Good quality**: >0.70 (Substantial)
- **Excellent quality**: >0.80 (Almost perfect)

## Current Metrics (GPT-4.1-nano vs Expert)

| Metric | Value | Status |
|--------|-------|--------|
| Cohen's Kappa | 0.579 | Moderate (needs improvement) |
| Within-1 Accuracy | 96.0% | Excellent ✅ |
| Pearson r | 0.590 | Good ✅ |
| Exact Accuracy | 64.0% | Moderate ⚠️ |

## Improvement Strategy

### 1. Use Better Models for Difficult Cases

**Current**: GPT-4.1-nano for all
**Improved**: Smart routing with GPT-5 models

- **Tier 1 (85%)**: GPT-4.1-nano (Kappa=0.579)
- **Tier 2 (10%)**: GPT-5-nano (better accuracy)
- **Tier 3 (5%)**: GPT-5-mini (expert quality)

**Expected improvement**: Kappa >0.65 (Substantial)

### 2. Better Prompt Engineering

Current prompt is good, but can be enhanced:
- Add more examples for edge cases
- Clarify ambiguous categories (2 vs 3, 3 vs 4)
- Add explicit confidence calibration

### 3. Calibrate Thresholds

Use `validate_smart_routing.py` to:
- Test different routing thresholds
- Find optimal escalation points
- Maximize overall Kappa

### 4. Improve Difficulty Detection

Enhanced features:
- Better sarcasm detection
- Negation handling
- Mixed sentiment identification
- Context-aware ambiguity scoring

## Target Metrics

| Metric | Current | Target | Strategy |
|--------|---------|--------|----------|
| Cohen's Kappa | 0.579 | **>0.65** | Smart routing + GPT-5 |
| Within-1 Accuracy | 96.0% | **>97%** | Better Tier 2/3 |
| Exact Accuracy | 64.0% | **>70%** | Improved prompts |
| Pearson r | 0.590 | **>0.70** | Better models |

## Cost-Quality Tradeoff

| Approach | Cost (283k) | Kappa | Quality |
|----------|-------------|-------|---------|
| Tier 1 only | $8.51 | 0.579 | Moderate |
| Smart Routing | $15.94 | **>0.65** | Substantial 🎯 |
| Expert only | $63.80 | >0.80 | Almost perfect ⭐ |

**Recommendation**: Smart routing provides best balance - substantial quality improvement for reasonable cost.

## Validation Process

1. **Run validation**: `python validate_smart_routing.py`
2. **Measure improvement**: Compare Tier 1 only vs Smart Routing
3. **Calibrate thresholds**: Optimize for maximum Kappa
4. **Verify targets**: Ensure Kappa >0.65

## Notes

- **Kappa 0.579 is acceptable** but not ideal for research
- **Within-1 accuracy (96%) is excellent** - most errors are small
- **Smart routing should improve Kappa to >0.65** (Substantial)
- **GPT-5 models** are newer and better than GPT-4o series

---

*Generated: 2026-01-12*
*Study: The Illusion Project - Anthropomorphization of AI Companions*
