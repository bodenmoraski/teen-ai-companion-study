# AnthroScore V3: LLM-Enhanced Anthropomorphization Scoring

## Executive Summary

This experiment aims to improve AnthroScore accuracy by leveraging modern LLMs (like GPT-4.1-nano) 
to directly classify anthropomorphization levels, rather than relying solely on masked language 
model pronoun probabilities.

## Current Approach Limitations (V2)

The current AnthroScore V2 uses:
- **Method**: Mask entity references → compute log(P_human_pronoun / P_nonhuman_pronoun)
- **Model**: RoBERTa-Twitter (masked language model)
- **Score Range**: Continuous (log probability ratio, typically -10 to +10)

### Known Limitations:
1. **Pronoun-centric**: Only captures pronoun choice, misses other anthropomorphization signals
   - "I love chatting with my bot" → Low score (no pronoun)
   - "She's just an algorithm" → High score (uses "she") but sentiment is dehumanizing
2. **Context-blind**: Can't understand sarcasm, irony, or complex sentiment
3. **Entity detection issues**: Struggles with custom names, ambiguous references
4. **Single dimension**: Only human vs non-human, no nuanced levels

## Proposed V3 Approach

### Core Idea
Use an LLM to directly classify anthropomorphization on a 1-5 scale, matching the 
annotation guidelines already developed for human annotators.

### Why This Could Work:
1. **Semantic understanding**: LLMs understand context, sarcasm, nuance
2. **Aligned with human judgment**: Can be prompted with same guidelines as human annotators
3. **Cost-effective**: GPT-4.1-nano is ~$0.10/1M input tokens, $0.30/1M output tokens
4. **Fast**: Modern APIs handle 1000s of requests/minute with batching
5. **Reliable**: Structured output ensures valid responses

### Cost Analysis (Estimated)
- **Current dataset**: ~16,000 users with ~200,000 comments
- **Per comment**: ~100 tokens input, ~50 tokens output
- **Total tokens**: ~20M input + 10M output = ~$5 total for entire dataset
- **Per run**: Extremely cheap for research-quality results

## Methodology

### Step 1: Create Gold Standard Test Set
- Sample 100 diverse comments (stratified by current AnthroScore)
- Have LLM (GPT-5.0-instant or similar smart model) label with detailed reasoning
- These become "expert labels" for validation

### Step 2: Develop Classification Prompt
```
Rate the anthropomorphization level (1-5):
1 = None (AI treated as software/tool)
2 = Minimal (slight humanization)
3 = Moderate (some human attributes)
4 = High (genuine feelings/personality)
5 = Extreme (full human-equivalent relationship)

Comment: "{text}"
```

### Step 3: Validate Cheap Model vs Expert
- Run GPT-4.1-nano on test set
- Measure agreement (Cohen's Kappa, accuracy, correlation)
- Target: Kappa > 0.7 (substantial agreement)

### Step 4: If Successful, Batch Process Full Dataset
- Use batch API for cost efficiency
- Compute both old and new scores for comparison
- Analyze correlations with demographics, emotions, etc.

## Files in This Directory
- `create_test_set.py` - Sample comments for labeling
- `label_with_expert.py` - Use GPT-5.0-instant to create gold labels
- `anthroscore_llm.py` - LLM-based scorer implementation
- `validate_agreement.py` - Measure LLM vs expert agreement
- `run_full_analysis.py` - Process full dataset if validation passes
- `test_set_labeled.parquet` - Gold standard test set with labels
- `RESULTS.md` - Analysis results and conclusions

## Success Criteria
1. **Cohen's Kappa ≥ 0.7** between GPT-4.1-nano and expert labels
2. **Cost < $10** for full dataset processing
3. **Processing time < 1 hour** for full dataset
4. **Clear improvement** in face validity (scores make intuitive sense)

## Risk Mitigation
- If cheap model fails validation, try GPT-4.1-mini or GPT-4o-mini
- Hybrid approach: Use LLM for ambiguous cases only
- Ensemble: Combine MLM + LLM scores

---

*Created: 2026-01-11*
*Study: The Illusion Project*
