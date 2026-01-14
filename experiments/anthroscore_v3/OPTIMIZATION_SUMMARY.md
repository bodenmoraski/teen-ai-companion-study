# AnthroScore Optimization Summary

## Goal
Find optimal model/prompt/temperature configuration for maximum agreement with expert evaluators.

## Current Status
- **Kappa = 0.579** (Moderate agreement)
- **Target: >0.60** (Substantial agreement) or **>0.80** (Almost perfect)

## Optimization Script
`optimize_anthroscore_v2.py` tests:
- **Models**: GPT-5-nano, GPT-5-mini, GPT-4.1-nano, GPT-4.1-mini
- **Prompts**: Current, Detailed, Examples, Focused
- **Temperatures**: 0.0, 0.1, 0.2

## Next Steps

1. **Fix expert labeling** (GPT-5 API parameter issue)
2. **Run optimization** to find best configuration
3. **Update AnthroScore** with optimal settings
4. **Validate improvement** on test set

## Expected Improvements

Using GPT-5 models + optimized prompts should achieve:
- **Kappa >0.65** (Substantial agreement)
- **Within-1 accuracy >97%**
- **Pearson r >0.70**

---

*Status: In progress - fixing API issues*
