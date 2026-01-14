# Smart Routing AnthroScore System

## Overview

A cost-optimized, adaptive scoring system that uses different LLM models based on comment difficulty and user importance. Achieves **~75% cost savings** vs using expert models everywhere while maintaining quality.

## Strategy

### Three-Tier System

1. **Tier 1 (Cheap)**: GPT-5-nano for obvious cases (~85%)
   - Clear, unambiguous comments
   - Standard users
   - Cost: ~$0.00003 per comment

2. **Tier 2 (Medium)**: GPT-5-mini for ambiguous cases (~10%)
   - Mixed signals, hedges, questions
   - Moderate difficulty
   - Cost: ~$0.00015 per comment

3. **Tier 3 (Expert)**: GPT-5-mini with detailed prompt for critical cases (~5%)
   - Very ambiguous or extreme scores
   - Important users (research sample, high engagement)
   - Cost: ~$0.00022 per comment

## Routing Logic

### Initial Assessment
- **Text difficulty**: Length, ambiguity, sarcasm indicators, complexity
- **User importance**: Research sample membership, engagement level
- **Score extremity**: Scores of 1 or 5 may need verification

### Escalation Triggers
After Tier 1 scoring, escalate if:
- Low confidence indicators in reasoning ("unclear", "ambiguous", "might be")
- High ambiguity score (>0.5)
- Extreme scores (1 or 5) with ambiguity
- Important users (research sample) with moderate confidence

## Cost Analysis

For **283,544 comments**:

| Approach | Cost | Savings |
|----------|------|---------|
| Expert-only (Tier 3) | $63.80 | - |
| Smart Routing | $15.94 | **75.0%** |
| Cheap-only (Tier 1) | $8.51 | 86.7% |

**Smart routing provides the best balance**: 75% savings vs expert-only, while ensuring difficult/important cases get proper attention.

## Usage

### Dry Run (Estimate Only)
```bash
python experiments/anthroscore_v3/run_smart_routing.py --dry-run
```

### Full Processing
```bash
python experiments/anthroscore_v3/run_smart_routing.py --run
```

### Test Individual Comments
```python
from smart_routing_scorer import SmartRoutingScorer

scorer = SmartRoutingScorer()
result = scorer.score_comment(
    text="She seemed confused? Maybe? I'm not sure...",
    username="user123",
    comment_count=50
)

print(f"Score: {result.score}/5")
print(f"Tier: {result.tier_used}")
print(f"Cost: ${result.total_cost_usd:.6f}")
```

## Output Format

The system produces a parquet file with:
- `anthroscore_smart`: Score (1-5)
- `smart_tier`: Tier used (1, 2, or 3)
- `smart_model`: Model name
- `smart_routing_reason`: Why this tier was chosen
- `smart_confidence`: Confidence score
- `smart_cost_usd`: Cost for this comment
- `smart_time_ms`: Processing time
- `smart_escalated`: Whether escalation occurred

## Advantages

1. **Cost-effective**: 75% savings vs expert-only
2. **Quality-preserving**: Difficult cases still get expert attention
3. **Adaptive**: Automatically routes based on difficulty signals
4. **Transparent**: Tracks routing decisions and costs
5. **Resumable**: Checkpoint system for long runs

## Comparison to Alternatives

### vs. Cheap-Only (Tier 1)
- **Smart routing**: $15.94, handles difficult cases
- **Cheap-only**: $8.51, but misses nuance in ambiguous cases
- **Verdict**: Worth the extra $7.43 for quality

### vs. Expert-Only (Tier 3)
- **Smart routing**: $15.94, 75% savings
- **Expert-only**: $63.80, no quality gain for easy cases
- **Verdict**: Smart routing is clearly better

### vs. Fixed Split (e.g., 80/20)
- **Smart routing**: Adaptive based on actual difficulty
- **Fixed split**: Wastes expert model on easy cases, misses some hard ones
- **Verdict**: Adaptive routing is superior

## Future Improvements

1. **Confidence calibration**: Better confidence estimates from Tier 1
2. **Learning from disagreements**: Track which escalations were necessary
3. **User-level optimization**: Cache scores for repeated patterns
4. **Batch processing**: Process multiple comments in parallel
5. **Cost tracking**: Real-time cost monitoring and alerts

---

*Generated: 2026-01-12*
*Study: The Illusion Project - Anthropomorphization of AI Companions*
