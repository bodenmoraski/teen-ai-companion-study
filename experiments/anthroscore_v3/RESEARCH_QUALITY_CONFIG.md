# Research-Quality Smart Routing Configuration

## Overview

This document outlines the research-quality configuration for the smart routing AnthroScore system, ensuring maximum quality while maintaining cost efficiency.

## Validated Models

Based on validation results (see `RESULTS.md`):

| Tier | Model | Validation | Use Case |
|------|-------|------------|----------|
| **Tier 1** | GPT-4.1-nano | Kappa=0.579, r=0.590 | ~85% of comments (easy cases) |
| **Tier 2** | GPT-4o-mini | Better than nano | ~10% of comments (ambiguous) |
| **Tier 3** | GPT-4o | Expert (gold standard) | ~5% of comments (critical) |

### Why These Models?

1. **Tier 1 (GPT-4.1-nano)**: 
   - Validated against expert labels: 96% within-1 accuracy
   - Cost-effective: ~$0.00003 per comment
   - Sufficient for clear, unambiguous cases

2. **Tier 2 (GPT-4o-mini)**:
   - Better than nano for ambiguous cases
   - Still cost-effective: ~$0.00015 per comment
   - Good balance for moderate difficulty

3. **Tier 3 (GPT-4o)**:
   - Validated as expert/gold standard
   - Highest quality: used for critical cases
   - More expensive: ~$0.0008 per comment

## Calibrated Thresholds

Thresholds should be calibrated using `validate_smart_routing.py`:

```python
CONFIDENCE_THRESHOLD_TIER2 = 0.65  # Escalate to Tier 2
CONFIDENCE_THRESHOLD_TIER3 = 0.45  # Escalate to Tier 3 (expert)
```

**Calibration Process**:
1. Run `validate_smart_routing.py` on test set
2. Test different threshold combinations
3. Select thresholds that maximize Cohen's Kappa
4. Update `smart_routing_scorer.py` with calibrated values

## Enhanced Difficulty Detection

The optimized version includes:

### Text Features
- **Length**: Very short (<30 chars) or very long (>1500 chars) = harder
- **Pronoun mixing**: Multiple pronoun types = ambiguous
- **Hedges**: "maybe", "perhaps", "might" = uncertainty
- **Sarcasm**: "lol", "/s", "jk" = harder to interpret
- **Negation**: Can flip meaning
- **Mixed sentiment**: Positive + negative = ambiguous

### Confidence Estimation
- Analyzes Tier 1 reasoning for low-confidence phrases
- Considers text difficulty signals
- Accounts for extreme scores (1 or 5)
- Weights user importance

## Importance Weighting

Users are prioritized based on:

1. **Research Sample** (2.5x): Users in research analyses
2. **Known Demographics** (1.5x): Users with known age/gender
3. **High Engagement** (1.3x): Users with >50 comments
4. **Extreme Scores** (1.4x): Scores of 1 or 5 need verification

## Quality Metrics

Target metrics (from validation):

| Metric | Tier 1 Only | Smart Routing | Target |
|--------|------------|--------------|--------|
| Cohen's Kappa | 0.579 | >0.65 | Maximize |
| Within-1 Accuracy | 96% | >97% | Maximize |
| Pearson r | 0.590 | >0.70 | Maximize |
| MAE | 0.41 | <0.35 | Minimize |

## Validation Process

### Step 1: Validate Routing Quality
```bash
python experiments/anthroscore_v3/validate_smart_routing.py
```

This will:
- Compare Tier 1 only vs Smart Routing vs Expert only
- Measure improvement from routing
- Analyze which cases benefit from escalation

### Step 2: Calibrate Thresholds
The validation script will:
- Test different threshold combinations
- Find optimal values that maximize quality
- Report recommended thresholds

### Step 3: Update Configuration
Update `smart_routing_scorer.py` with calibrated thresholds.

## Cost-Quality Tradeoff

| Approach | Cost (283k) | Quality (Kappa) | Time |
|----------|-------------|----------------|------|
| Tier 1 only | $8.51 | 0.579 | 4 hours |
| Smart Routing | $15.94 | >0.65 | 4 hours |
| Expert only | $226.84 | 0.85+ | 4 hours |

**Recommendation**: Smart routing provides best balance:
- 75% cost savings vs expert-only
- Significant quality improvement vs Tier 1 only
- Same processing time (parallel)

## Research-Quality Checklist

Before running on full dataset:

- [ ] Models validated against expert labels
- [ ] Thresholds calibrated on test set
- [ ] Difficulty detection tested
- [ ] Importance weighting configured
- [ ] Validation metrics meet targets
- [ ] Cost estimates verified
- [ ] Parallel processing enabled

## Files

- `smart_routing_scorer_optimized.py` - Research-quality version
- `validate_smart_routing.py` - Validation and calibration
- `smart_routing_scorer.py` - Main implementation (update with calibrated thresholds)
- `run_smart_routing.py` - Batch processing script

## Next Steps

1. **Run validation**: `python validate_smart_routing.py`
2. **Calibrate thresholds**: Use recommended values
3. **Test on sample**: Verify quality improvements
4. **Run full dataset**: Process all 283k comments

---

*Generated: 2026-01-12*
*Study: The Illusion Project - Anthropomorphization of AI Companions*
