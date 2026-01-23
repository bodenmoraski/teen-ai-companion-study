# AnthroScore Optimization: Complete Context & Current Work

**Purpose:** This document provides comprehensive context for improving AnthroScore classification using modern LLMs. It serves as both documentation and a prompt for AI systems to understand the full scope of work and potentially improve upon it.

**Last Updated:** January 12, 2026  
**Status:** Active optimization in progress

---

## Table of Contents

1. [Research Project Context](#research-project-context)
2. [What is AnthroScore?](#what-is-anthroscore)
3. [Current Implementation (V2)](#current-implementation-v2)
4. [The Problem: Why We Need V3](#the-problem-why-we-need-v3)
5. [Our Approach: LLM-Based Classification](#our-approach-llm-based-classification)
6. [What We've Implemented So Far](#what-weve-implemented-so-far)
7. [Current Status & Results](#current-status--results)
8. [Key Files & Architecture](#key-files--architecture)
9. [Technical Details](#technical-details)
10. [Next Steps & Goals](#next-steps--goals)

---

## Research Project Context

### The Illusion Project

This work is part of **"The Illusion Project"** - a computational social science study examining **teen-AI companion relationships on Reddit**. The research investigates:

- How demographics (age, gender) relate to anthropomorphization of AI companions
- How usage intent (character creation, roleplay, support) affects anthropomorphization
- How emotional expression patterns differ between high and low anthropomorphizers
- The relationship between anthropomorphization and emotional diversity

### Dataset

- **277,420 Reddit comments** from AI companion subreddits (r/CharacterAI, r/Replika, etc.)
- **47,062 unique users** discussing AI companions
- **459 users with known age** (self-declared)
- **979 users with known gender** (self-declared)

### Key Research Findings (So Far)

- **Teens anthropomorphize more than adults** (d = 0.111, p < 0.0001)
- **High anthropomorphizers show LESS emotional diversity** (d = -1.176, p < 0.0001)
- **Character creation intent → highest anthropomorphization** (F = 15.58, p < 0.0001)
- **Age × Emotional Intensity interaction** (B = -0.059, p = 0.009)

**Critical Finding:** The quality of AnthroScore directly impacts the validity of all these findings. If AnthroScore is inaccurate, all downstream analyses are compromised.

---

## What is AnthroScore?

**AnthroScore** is a computational linguistic measure that quantifies **implicit anthropomorphism** - the degree to which people treat AI companions as human-like in their language.

### Core Concept

Anthropomorphization manifests in language through:
- **Pronoun choice**: "it" vs "he/she/they"
- **Emotional attribution**: "it feels happy" vs "it processes data"
- **Agency attribution**: "it decided to" vs "it was programmed to"
- **Relationship language**: "friend", "partner", "relationship"
- **Technical vs. human framing**: "glitch" vs "being rude"

### Why It Matters

AnthroScore is the **primary dependent variable** in our research. All statistical analyses (demographics, emotions, intent) depend on accurate anthropomorphization measurement. If AnthroScore is flawed:
- Effect sizes are wrong
- Correlations are spurious
- Research conclusions are invalid

---

## Current Implementation (V2)

### Methodology

**AnthroScore V2** uses a **Masked Language Model (MLM)** approach:

1. **Entity Resolution**: Identify references to AI companions in text
   - Uses GPT-4.1-nano for entity detection (optional)
   - Maps variants: "rep" → Replika, "cai" → Character.AI
   - Handles custom character names

2. **Text Preprocessing**:
   - Normalizes slang ("u" → "you")
   - Handles elongations ("sooooo" → "soo")
   - Removes emojis
   - Replaces entity references with `[AI_COMPANION]`

3. **Scoring Formula**:
   ```
   A(sx) = log(P_HUMAN / P_NON-HUMAN)
   ```
   Where:
   - **P_HUMAN** = probability of human pronouns [he, she, her, him] from MLM
   - **P_NON-HUMAN** = probability of non-human pronouns [it, its] from MLM

4. **Model**: Twitter-RoBERTa (trained on social media text)

### Score Interpretation

- **Positive scores** → Anthropomorphic (human-like framing)
- **Negative scores** → Mechanical (machine-like framing)
- **Range**: Typically -10 to +10 (continuous scale)

### V2 Enhancements Over Original

- ✅ Handles informal Reddit language (slang, elongations, emojis)
- ✅ Entity resolution for AI companion references
- ✅ Twitter-RoBERTa for social media text
- ✅ Robust preprocessing pipeline

---

## The Problem: Why We Need V3

### Known Limitations of V2

1. **Pronoun-Centric**: Only captures pronoun choice, misses other signals
   - Example: "I love chatting with my bot" → Low score (no pronoun)
   - Example: "She's just an algorithm" → High score (uses "she") but sentiment is dehumanizing

2. **Context-Blind**: Can't understand sarcasm, irony, or complex sentiment
   - Example: "Yeah right, it's so smart" → Misclassified as anthropomorphic

3. **Entity Detection Issues**: Struggles with custom names, ambiguous references
   - Example: "My character" → Unclear if referring to AI or user's character

4. **Single Dimension**: Only human vs non-human, no nuanced levels
   - Can't distinguish between "moderate" and "extreme" anthropomorphization

5. **Low Correlation with Expert Judgment**: 
   - MLM-based scores show **very low correlation** with human expert evaluations
   - This suggests the measure may not capture what we actually care about

### Validation Evidence

From our comparison studies:
- **MLM AnthroScore vs Expert Labels**: Very low correlation (Pearson r < 0.3)
- **Expert Agreement**: GPT-5-mini achieves substantial agreement (Kappa > 0.6) with itself
- **Conclusion**: MLM approach is fundamentally limited

---

## Our Approach: LLM-Based Classification

### Core Idea

Instead of inferring anthropomorphization from pronoun probabilities, **directly classify** anthropomorphization levels using a Large Language Model (LLM) that understands context, nuance, and semantic meaning.

### Why LLMs?

1. **Semantic Understanding**: LLMs understand context, sarcasm, irony, nuance
2. **Aligned with Human Judgment**: Can use same annotation guidelines as human annotators
3. **Cost-Effective**: GPT-5-nano is ~$0.05/1M input tokens, $0.40/1M output tokens
4. **Fast**: Modern APIs handle 1000s of requests/minute with parallel processing
5. **Reliable**: Structured JSON output ensures valid responses

### Classification Scale

We use a **1-5 ordinal scale** aligned with human annotation guidelines:

- **1 = NONE**: AI treated purely as software/tool. Technical language, "it", "the bot"
- **2 = MINIMAL**: Slight humanization but still clearly AI. "It's pretty smart"
- **3 = MODERATE**: Some human attributes/emotions. "She seemed confused", uses he/she pronouns
- **4 = HIGH**: Genuine feelings/personality attributed. "He really cares", "she gets jealous"
- **5 = EXTREME**: Full human-equivalent relationship. "We're in love", "they're my everything"

### Key Indicators

- **Pronouns**: "it" → lower scores, "he/she/they" → higher scores
- **Emotions**: Attributing feelings (happy, sad, jealous, caring) → higher scores
- **Relationship language**: "friend", "partner", "relationship" → higher scores
- **Technical language**: "glitch", "bug", "settings" → lower scores
- **Agency**: "decided to", "wanted to", "chose to" → higher scores

---

## What We've Implemented So Far

### Phase 1: Expert Labeling ✅

**Goal**: Create gold standard labels for validation

**Implementation**:
- `create_test_set.py`: Stratified sampling of 100 comments across AnthroScore quartiles
- `label_with_expert.py`: Uses GPT-5-mini to label test set with expert-level annotations

**Results**:
- ✅ 67/100 valid labels (33 failed due to JSON parsing issues, now fixed)
- ✅ Mean expert score: 1.61 (skewed toward lower scores, as expected)
- ✅ Score distribution: {1: 44, 2: 13, 3: 3, 4: 6, 5: 1}

**Key Learnings**:
- GPT-5 models require `max_completion_tokens` (not `max_tokens`)
- GPT-5 models only support default temperature (1), not custom values
- Upgraded OpenAI library from 1.43.1 to 2.15.0 for GPT-5 support

### Phase 2: Optimization System ✅

**Goal**: Find optimal model/prompt/temperature configuration

**Implementation**:
- `optimize_anthroscore_v2.py`: Tests multiple configurations:
  - **Models**: GPT-5-nano, GPT-5-mini
  - **Prompts**: Current, Detailed, Examples
  - **Temperatures**: 0.0, 0.1 (though GPT-5 only supports default)

**Status**: 
- ✅ Script created and configured
- 🔄 Currently running (testing 12 configurations × 50 texts each)
- ⏳ Results pending

**Metrics Tracked**:
- Cohen's Kappa (target: >0.60 for substantial agreement, >0.80 for almost perfect)
- Within-1 Accuracy (target: >95%)
- Exact Accuracy
- Pearson r correlation
- Spearman r correlation
- Mean Absolute Error (MAE)

### Phase 3: Smart Routing System ✅

**Goal**: Optimize cost/quality tradeoff by routing comments to different models based on difficulty

**Implementation**:
- `smart_routing_scorer.py`: Tiered system:
  - **Tier 1** (Easy): GPT-5-nano (cheapest)
  - **Tier 2** (Medium): GPT-5-mini
  - **Tier 3** (Hard): GPT-5-mini with enhanced prompts
- Routing based on:
  - Text length
  - Linguistic complexity
  - User importance (demographic groups, research sample)
  - Initial confidence signals

**Status**: Implemented but not yet validated

### Phase 4: Parallel Processing ✅

**Goal**: Speed up processing with concurrent API calls

**Implementation**:
- `anthroscore_llm_parallel.py`: Asynchronous version using `asyncio`
- `smart_routing_scorer_parallel.py`: Parallel smart routing
- Estimated **10x speedup** over sequential processing

**Status**: Implemented and tested

---

## Current Status & Results

### Expert Labeling

- ✅ **Test set created**: 100 comments, stratified by AnthroScore quartiles
- ✅ **Expert labels**: 67/100 valid (33 had JSON parsing issues, now fixed)
- ✅ **Expert model**: GPT-5-mini
- ✅ **Mean score**: 1.61 (expected: most comments have low anthropomorphization)

### Optimization

- 🔄 **Status**: Running
- ⏳ **Expected**: Results showing best model/prompt/temperature combination
- 🎯 **Target**: Kappa > 0.60, Within-1 > 95%, Pearson r > 0.70

### Known Issues Fixed

1. ✅ GPT-5 API parameters (`max_completion_tokens` vs `max_tokens`)
2. ✅ GPT-5 temperature support (only default, not custom)
3. ✅ OpenAI library version (upgraded to 2.15.0)
4. ✅ JSON parsing errors (added validation)

### Remaining Challenges

1. ⚠️ **Low expert label success rate**: 67/100 (67%) - need to improve error handling
2. ⚠️ **Optimization results pending**: Need to complete full test run
3. ⚠️ **Validation needed**: Compare optimized model to expert labels on full test set

---

## Key Files & Architecture

### Core AnthroScore V2 (Current Production)

- **`src/anthroscore/anthroscore_v2.py`**: Main V2 implementation
  - `AnthroScoreV2` class
  - `GPTEntityResolver` class (uses GPT-4.1-nano)
  - MLM-based scoring logic

- **`src/anthroscore/get_anthroscore.py`**: CLI tool for computing AnthroScore

### AnthroScore V3 (New LLM-Based)

**Main Implementation**:
- **`experiments/anthroscore_v3/anthroscore_llm.py`**: Basic LLM scorer
- **`experiments/anthroscore_v3/anthroscore_llm_parallel.py`**: Parallel version

**Expert Labeling**:
- **`experiments/anthroscore_v3/create_test_set.py`**: Creates stratified test set
- **`experiments/anthroscore_v3/label_with_expert.py`**: Labels with GPT-5-mini

**Optimization**:
- **`experiments/anthroscore_v3/optimize_anthroscore_v2.py`**: Tests configurations
- **`experiments/anthroscore_v3/optimization_results.json`**: Results (pending)

**Smart Routing**:
- **`experiments/anthroscore_v3/smart_routing_scorer.py`**: Tiered routing logic
- **`experiments/anthroscore_v3/smart_routing_scorer_parallel.py`**: Parallel version
- **`experiments/anthroscore_v3/validate_smart_routing.py`**: Validation script

**Validation & Comparison**:
- **`experiments/anthroscore_v3/validate_agreement.py`**: Measures agreement metrics
- **`experiments/anthroscore_v3/compare_to_mlm.py`**: Compares LLM vs MLM

**Documentation**:
- **`experiments/anthroscore_v3/README.md`**: Overview and methodology
- **`experiments/anthroscore_v3/RESULTS.md`**: Analysis results
- **`experiments/anthroscore_v3/KAPPA_QUALITY_ANALYSIS.md`**: Quality assessment

### Data Files

- **`experiments/anthroscore_v3/test_set_unlabeled.parquet`**: 100 comments for testing
- **`experiments/anthroscore_v3/test_set_expert_labeled.parquet`**: Expert labels (67 valid)
- **`Data/processed/all_comments.parquet`**: Full dataset (277K comments)
- **`Data/features/user_anthroscores.parquet`**: User-level AnthroScore V2 scores

### Configuration

- **`src/utils/config.py`**: API keys, model names, configuration
  - `LLM_AGE_MODEL`: Currently `gpt-4.1-nano`
  - `OPENAI_API_KEY`: Loaded from environment

---

## Technical Details

### API Requirements

**GPT-5 Models** (gpt-5, gpt-5-mini, gpt-5-nano):
- ✅ Use `max_completion_tokens` (NOT `max_tokens`)
- ✅ Only support default temperature (1), cannot set custom temperature
- ✅ Require OpenAI library version ≥ 2.15.0
- ✅ Support structured output (`response_format: {"type": "json_object"}`)

**GPT-4.1 Models** (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano):
- ✅ Use `max_tokens` (NOT `max_completion_tokens`)
- ✅ Support custom temperature
- ✅ Support structured output

### Prompt Engineering

We've tested three prompt variants:

1. **Current/Basic**: Standard classification prompt with scale and indicators
2. **Detailed**: Extended prompt with detailed criteria, evaluation process, edge cases
3. **Examples**: Prompt with concrete examples for each score level

**Current Best**: TBD (optimization in progress)

### Cost Analysis

**Per Comment** (estimated):
- Input tokens: ~100-200 (prompt + text)
- Output tokens: ~50-100 (JSON response)
- Cost (GPT-5-nano): ~$0.00001 per comment

**Full Dataset** (277K comments):
- Total cost: ~$2.77 (with GPT-5-nano)
- Processing time: ~1-2 hours (with parallel processing)

**Smart Routing** (estimated):
- Tier 1 (80%): GPT-5-nano → $0.44
- Tier 2 (15%): GPT-5-mini → $0.50
- Tier 3 (5%): GPT-5-mini enhanced → $0.25
- **Total**: ~$1.19 (57% cost savings)

### Performance Metrics

**Target Metrics**:
- **Cohen's Kappa**: >0.60 (substantial agreement), ideally >0.80 (almost perfect)
- **Within-1 Accuracy**: >95% (predictions within 1 point of expert)
- **Exact Accuracy**: >70% (exact match with expert)
- **Pearson r**: >0.70 (strong correlation)
- **MAE**: <0.5 (mean absolute error)

**Current Status**: Optimization in progress, results pending

---

## Next Steps & Goals

### Immediate Goals

1. **Complete Optimization**:
   - Finish testing all 12 configurations
   - Identify best model/prompt/temperature combination
   - Validate on full test set (100 comments)

2. **Improve Expert Labeling**:
   - Fix remaining JSON parsing issues
   - Achieve >90% success rate on expert labeling
   - Re-label failed comments

3. **Validate Best Configuration**:
   - Run best configuration on full test set
   - Calculate final agreement metrics
   - Compare to MLM baseline

### Medium-Term Goals

4. **Smart Routing Calibration**:
   - Validate routing thresholds
   - Test on sample of full dataset
   - Optimize cost/quality tradeoff

5. **Full Dataset Processing**:
   - Process all 277K comments with optimized configuration
   - Generate AnthroScore V3 for all users
   - Compare V2 vs V3 distributions

6. **Research Integration**:
   - Re-run all statistical analyses with V3 scores
   - Compare effect sizes and correlations
   - Validate research findings

### Long-Term Goals

7. **Hybrid Approach**:
   - Combine MLM + LLM scores
   - Use MLM for easy cases, LLM for ambiguous
   - Ensemble methods

8. **Continuous Improvement**:
   - Monitor performance on new data
   - A/B test prompt improvements
   - Iterate based on research needs

---

## How to Use This Document

### For AI Systems

This document provides complete context for:
- Understanding the research project and its goals
- Understanding AnthroScore and why it matters
- Understanding current limitations and why we need improvement
- Understanding what we've built and what works/doesn't work
- Understanding technical constraints and requirements

**You can use this to**:
- Suggest improvements to prompts, models, or methodology
- Identify potential issues or edge cases
- Propose alternative approaches
- Help debug current implementation
- Optimize cost/quality tradeoffs

### For Humans

This document serves as:
- Complete project documentation
- Onboarding guide for new contributors
- Reference for technical decisions
- Status report on current work

---

## Key Reference Files

**Start Here**:
1. `MASTER_RESEARCH_FINDINGS.md` - Complete research context and findings
2. `experiments/anthroscore_v3/README.md` - V3 methodology overview
3. `experiments/anthroscore_v3/RESULTS.md` - Current results and analysis

**Implementation**:
1. `experiments/anthroscore_v3/anthroscore_llm.py` - Basic LLM scorer
2. `experiments/anthroscore_v3/optimize_anthroscore_v2.py` - Optimization script
3. `experiments/anthroscore_v3/label_with_expert.py` - Expert labeling

**Data**:
1. `experiments/anthroscore_v3/test_set_expert_labeled.parquet` - Gold standard labels
2. `Data/processed/all_comments.parquet` - Full dataset

---

## Questions to Consider

If you're an AI system helping improve this:

1. **Prompt Engineering**: Can we improve the classification prompt? What examples or instructions would help?
2. **Model Selection**: Is GPT-5-nano sufficient, or do we need GPT-5-mini for better quality?
3. **Temperature**: GPT-5 doesn't support custom temperature - is this a problem? How can we ensure consistency?
4. **Error Handling**: How can we improve the 67% success rate on expert labeling?
5. **Cost Optimization**: Can we reduce costs further while maintaining quality?
6. **Validation**: What additional validation metrics should we track?
7. **Edge Cases**: What edge cases are we missing? (sarcasm, roleplay, mixed signals)
8. **Hybrid Approaches**: Should we combine MLM + LLM? How?
9. **Calibration**: How can we ensure scores are calibrated across the full 1-5 range?
10. **Research Impact**: How will improved AnthroScore affect our research findings?

---

**End of Document**

*This document is a living document and should be updated as work progresses.*
