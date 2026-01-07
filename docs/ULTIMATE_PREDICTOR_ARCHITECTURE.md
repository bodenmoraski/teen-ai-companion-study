# Ultimate Age Predictor Architecture

**Created:** January 6, 2026  
**Status:** Ready to Run  
**Goal:** Achieve >70% accuracy (vs. 46.3% baseline)

---

## Executive Summary

This document describes a state-of-the-art multi-signal stacked ensemble for predicting Reddit user age. Instead of arbitrary threshold projections, we train actual ML classifiers on known-age users and combine multiple signals using meta-learning.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ULTIMATE AGE PREDICTOR                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  SIGNAL 1       │  │  SIGNAL 2       │  │  SIGNAL 3       │     │
│  │  Text Embed.    │  │  Subreddit      │  │  Behavioral     │     │
│  │  (SBERT)        │  │  Patterns       │  │  Features       │     │
│  │                 │  │                 │  │                 │     │
│  │  384-dim        │  │  500 binary     │  │  7 features     │     │
│  │  embedding      │  │  features       │  │                 │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           ▼                    ▼                    ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   XGBoost       │  │   XGBoost       │  │   XGBoost       │     │
│  │   Classifier    │  │   Classifier    │  │   Classifier    │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           ▼                    ▼                    ▼               │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │              PROBABILITY OUTPUTS (per class)               │     │
│  │   P(teen), P(young_adult), P(adult) × 3 signals = 9 vals   │     │
│  └─────────────────────────────┬─────────────────────────────┘     │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    META-LEARNER (STACKING)                   │   │
│  │                         XGBoost                              │   │
│  │                                                              │   │
│  │   Input: 9 probability values from 3 signals                │   │
│  │   Output: Final calibrated probabilities                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why This is Better Than the Old Approach

### Old Approach (46.3% accuracy)

1. **Community Embedding**: Word2Vec on subreddit co-occurrence
2. **Projection**: Project user onto "age axis" defined by seed pairs
3. **Thresholds**: Arbitrary cutoffs (-0.3, -0.1, 0.1, 0.3)
4. **Ensemble**: Simple weighted voting

**Problems:**
- Thresholds not learned from data
- Only uses 1D projection (ignores other dimensions)
- No actual ML training
- Wastes information in embeddings

### New Approach (Target: >70%)

1. **Multiple Signals**: Text + Subreddit + Behavior
2. **Trained Classifiers**: XGBoost on each signal
3. **Learned Boundaries**: Classifiers learn optimal decision boundaries
4. **Meta-Learning**: Stacking combines signals optimally
5. **Calibrated Probabilities**: Can filter by confidence

**Improvements:**
- Uses ALL information in features (not just 1D projection)
- Learns from actual labeled data (459 users)
- Multiple complementary signals
- Stacking outperforms voting
- Confidence filtering for quality

---

## Signal Details

### Signal 1: Text Embeddings

**Model**: Sentence-BERT (all-MiniLM-L6-v2)  
**Dimension**: 384  
**Input**: Concatenated user comments (up to 50)

**Rationale**: Language style, vocabulary, and topics vary by age. BERT captures semantic meaning that age-related patterns can be learned from.

**Expected Contribution**: High - text is rich signal

### Signal 2: Subreddit Patterns

**Features**: Binary participation in top 500 subreddits  
**Method**: Trained classifier (not projection!)

**Rationale**: Different age groups participate in different communities. Instead of projecting onto an arbitrary axis, we let XGBoost learn which subreddit combinations predict age.

**Expected Contribution**: Medium-High - strong signal but sparse

### Signal 3: Behavioral Features

**Features**:
- `late_night_ratio`: Posts between midnight-4am (teens post late)
- `school_hours_ratio`: Posts 9am-3pm (lower if in school)
- `weekend_ratio`: Weekend vs weekday posting
- `activity_span_days`: Time between first and last post
- `avg_comment_length`: Writing maturity indicator
- `comment_count`: Activity level
- `subreddit_diversity`: Exploration patterns

**Rationale**: Age groups have different online behavior patterns. These features capture behavioral signatures.

**Expected Contribution**: Medium - helpful auxiliary signal

### Signal 4: LLM (Optional Enhancement)

**Model**: GPT-4.1-nano or similar  
**Method**: Few-shot prompting with probability output

**Rationale**: LLMs can pick up on subtle life stage indicators. Used for users where other signals are weak.

**Cost**: ~$0.001 per user → ~$50 for all users

**Status**: Implemented but not integrated into stacking (can add later)

---

## Stacking Strategy

### Why Stacking > Voting

**Voting**: Fixed weights, ignores signal reliability per-user
**Stacking**: Learns optimal combination, adapts to signal quality

### How It Works

1. Train each signal classifier with 5-fold CV
2. Get out-of-fold predictions (probabilities) for each signal
3. Stack probabilities as features for meta-learner
4. Train meta-learner (XGBoost) to combine optimally
5. Meta-learner learns: "When Signal 1 is confident but Signal 2 disagrees, trust Signal 1"

### Expected Improvement

Literature shows stacking typically adds 2-5% accuracy over best single model.

---

## Training Strategy

### Data Split
- **Train**: 80% of 459 known-age users (~367 users)
- **Test**: 20% held out (~92 users)

### Cross-Validation
- 5-fold stratified CV during training
- Prevents overfitting on small training set

### Class Balance
- Use stratified sampling to maintain class distribution
- XGBoost handles class imbalance with `scale_pos_weight`

---

## Confidence Filtering

The system outputs calibrated probabilities. Users can be filtered by confidence:

| Threshold | Expected Accuracy | Expected Coverage |
|-----------|------------------|-------------------|
| ≥50% | ~60% | ~80% |
| ≥60% | ~70% | ~60% |
| ≥70% | ~75% | ~40% |
| ≥80% | ~80% | ~20% |

**Recommendation**: Use ≥60% threshold for good balance of accuracy and coverage.

---

## Possible Enhancements

### 1. More Training Data
- **External datasets**: Look for publicly available Reddit age datasets
- **Active collection**: Use regex to find more self-declarations in historical Reddit data
- **Arctic Shift**: Query more subreddits known to have age declarations (e.g., r/teenagers posts)

### 2. Better Text Features
- **Fine-tuned BERT**: Train on age prediction task
- **Age-specific vocabulary**: Extract age-indicative words/phrases
- **Writing style features**: Punctuation, capitalization, slang

### 3. Temporal Features
- **Account age**: Newer accounts may be younger users
- **First subreddits joined**: Early participation patterns
- **Posting frequency changes**: How activity evolves

### 4. Additional Signals
- **Username analysis**: Some patterns correlate with age
- **Comment score patterns**: How community responds
- **Thread participation depth**: Lurker vs active patterns

---

## How to Run

```bash
# Install dependencies if needed
pip install sentence-transformers xgboost scikit-learn pandas numpy

# Run the predictor
python scripts/run_ultimate_predictor.py
```

**Output Files**:
- `Data/features/ultimate_predictor/ultimate_predictions.parquet`
- `Data/features/ultimate_predictor/model/ultimate_predictor.pkl`
- `results/ultimate_predictor_report.txt`

---

## Expected Results

Based on similar approaches in literature:

| Method | Accuracy |
|--------|----------|
| Random (3 classes) | 33.3% |
| Old approach (projection + thresholds) | 46.3% |
| Single best signal (text embeddings) | ~55-60% |
| **Stacked ensemble (this system)** | **65-75%** |

If we achieve >65% accuracy, this is a **significant improvement** that makes the demographic analysis much more defensible.

---

## Fallback Plan

If the predictor doesn't achieve target accuracy:

1. **Use confidence filtering**: Even if overall accuracy is 60%, high-confidence predictions may be 75%+
2. **Subset analysis**: Use only users with ≥70% confidence for regression
3. **Report uncertainty**: Use probabilistic predictions in downstream analysis
4. **Combine with known-age**: Weight known-age users higher in analysis

---

## Next Steps

1. **Run the predictor**: `python scripts/run_ultimate_predictor.py`
2. **Evaluate results**: Check `results/ultimate_predictor_report.txt`
3. **Iterate if needed**: Add more features or training data
4. **Integrate**: Use predictions in demographic regression analysis

---

## References

1. Chew et al. (2021). Predicting Age Groups of Reddit Users. JMIR.
2. Text2Gender (2023). BERT-based age/gender prediction. arXiv:2305.08633.
3. XGBoost stacking: Standard ensemble learning technique.
4. Sentence-BERT: Reimers & Gurevych (2019). Sentence-BERT.

