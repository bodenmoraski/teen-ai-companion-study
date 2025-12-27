# Architecture Improvement Plan

**Purpose:** Ideas to think through during your trip (no PC/data needed)  
**Focus:** Fundamental improvements beyond threshold tuning

---

## 1. CLASSIFICATION ARCHITECTURE IMPROVEMENTS

### 1.1 Train a Proper ML Classifier

**Current Problem:**  
Community embeddings use hand-crafted seed pairs and arbitrary projections.

**Better Approach:**
```
Train: XGBoost/LightGBM classifier
Input: User's subreddit embedding vector (100-dim from Word2Vec)
Target: 3-bucket age from self-declared users (459 training examples)
Method: Stratified 5-fold CV to avoid overfitting
```

**Why This Works:**
- Uses ALL dimensions of embedding, not just one projection axis
- Learns optimal decision boundaries from data
- Can include additional features (comment length, activity patterns)
- Regularization prevents overfitting on small training set

**Design Questions to Think About:**
- What additional features could help? (posting time patterns, vocabulary complexity)
- Should we use transfer learning from a pre-trained user representation model?
- How to handle class imbalance (most are teens in this dataset)?

---

### 1.2 Contrastive Learning for User Embeddings

**Current Problem:**  
Word2Vec on subreddit co-occurrence doesn't directly optimize for demographic prediction.

**Better Approach:**
```
Use contrastive learning:
- Positive pairs: Users with same self-declared age
- Negative pairs: Users with different self-declared ages
- Learn embedding that clusters similar-age users

Model: Simple neural network with triplet loss
Input: Bag of subreddits (or comment text)
Output: User embedding optimized for age separation
```

**Why This Works:**
- Directly optimizes for the task we care about
- Can use the 459 self-declared users as anchors
- Modern contrastive methods (SimCLR, CLIP) are very effective

---

### 1.3 Multi-Task Learning

**Idea:** Train one model that predicts both age AND gender AND other attributes jointly.

**Benefits:**
- Shared representations improve all predictions
- More training signal from available labels
- Gender prediction might help age prediction (correlated patterns)

---

## 2. LLM CLASSIFICATION IMPROVEMENTS

### 2.1 Better Prompting Strategy

**Current:**
```
"Analyze these Reddit comments and estimate age bucket..."
```

**Improvements to Consider:**
1. **Chain-of-thought prompting:**
   ```
   "First, identify specific clues about the user's life stage.
   Then, list evidence for each age bracket.
   Finally, make your prediction."
   ```

2. **Few-shot examples:**
   Include 3-4 examples of users with known ages

3. **Calibrated confidence:**
   Ask for probability distribution over age buckets, not just best guess

---

### 2.2 Comment Selection Strategy

**Current:** Just take first 20 comments

**Better:**
- Select most informative comments (mention age-related topics)
- Diverse sampling across time
- Filter out very short or uninformative comments

---

### 2.3 Ensemble of LLM Calls

Run 3 LLM calls with different prompts/temperatures, vote on result.
More expensive but more reliable.

---

## 3. ENSEMBLE ARCHITECTURE IMPROVEMENTS

### 3.1 Stacking Instead of Voting

**Current:** Weighted average of predictions

**Better:** Train a meta-learner
```
Features for meta-learner:
- Community embedding score
- Community embedding confidence (distance from threshold)
- LLM prediction (one-hot)
- LLM confidence
- Number of subreddits user participates in
- Comment count
- Account age

Target: Correct age bucket

Method: Logistic regression or small neural net
```

**Why:** Learns optimal weighting per situation, not fixed weights.

---

### 3.2 Calibrated Probabilities

Instead of hard predictions, output calibrated probabilities:
- P(teen) = 0.4, P(young_adult) = 0.35, P(adult) = 0.25

This allows:
- Uncertainty quantification in downstream analysis
- Probabilistic regression instead of categorical

---

## 4. OUTCOME VARIABLE IMPROVEMENTS

### 4.1 Better AnthroScore Aggregation

**Current Problem:**  
75% of users have AnthroScore = 0 (floor effect)

**Ideas:**
1. **Binary:** Did user EVER anthropomorphize? (0/1)
2. **Maximum:** Highest AnthroScore across comments
3. **Trend:** Is anthropomorphization increasing over time?
4. **Contextual:** AnthroScore for emotional vs. informational comments

---

### 4.2 Alternative Dependent Variables

Instead of just AnthroScore mean:
- **Emotional intensity** when discussing AI
- **Topic diversity** (BERTopic clusters)
- **Engagement level** (comment length, frequency)
- **Sentiment trajectory** over conversation threads

---

## 5. DATA COLLECTION IMPROVEMENTS

### 5.1 Expand to More Platforms

Reddit is just one slice. Consider:
- Discord servers (different demographics?)
- TikTok comments
- Twitter/X discussions
- YouTube comments on AI companion videos

---

### 5.2 Longitudinal Design

**Current:** Cross-sectional snapshot

**Better:** Track same users over time
- How does anthropomorphization change with experience?
- Are there developmental trajectories?
- Effect of AI model updates on behavior

---

### 5.3 Ground Truth Expansion

**Current:** 459 self-declared ages

**Ideas:**
- Survey/interview a subset of users directly
- Use profile information from other platforms
- Crowdsource age annotations based on comment content

---

## 6. STATISTICAL MODEL IMPROVEMENTS

### 6.1 Measurement Error Models (SIMEX/MCSIMEX)

Instead of simple attenuation correction, use simulation-extrapolation:
- Artificially add more measurement error
- Fit model at multiple error levels
- Extrapolate back to zero error

This is more robust than simple correction.

---

### 6.2 Bayesian Regression with Priors

```python
# Informative priors based on literature
age_effect ~ Normal(0, 0.05)  # Expect small effects
gender_effect ~ Normal(0, 0.05)
subreddit_effect ~ Normal(0, 0.2)  # Expect larger platform effects
```

Benefits:
- Better uncertainty quantification
- Can incorporate prior knowledge
- Handles small samples better

---

### 6.3 Hierarchical Model

```
User nested within Subreddit

AnthroScore_ij = β0 + β1*Age_i + β2*Gender_i + u_j + ε_ij

where:
- i = user
- j = subreddit
- u_j = random subreddit intercept
```

This properly models the subreddit clustering.

---

## 7. QUICK WINS (Implement First When You Return)

1. **Apply optimized thresholds** ✅ (running now)
2. **Train simple XGBoost on embeddings** (1-2 hours)
3. **Better LLM prompt** (30 min)
4. **Binary anthropomorphization variable** (1 hour)
5. **Hierarchical regression** (2 hours)

---

## 8. MEDIUM-TERM PROJECTS

1. **Contrastive learning for user embeddings** (1 week)
2. **Meta-learner ensemble** (2-3 days)
3. **Longitudinal tracking** (ongoing)

---

## 9. QUESTIONS TO PONDER ON YOUR TRIP

1. **What's the actual research question?**
   - "Do teens anthropomorphize more?" → Currently null
   - Maybe reframe: "What predicts anthropomorphization?"
   - Subreddit/platform might be the real story

2. **What would change policy?**
   - If we found teens DO anthropomorphize more, what intervention?
   - Platform design changes? Age verification?

3. **What's the minimal viable finding?**
   - "Platform context matters more than demographics" is actually publishable
   - Focus energy on making that finding rigorous

4. **Is classification even the right approach?**
   - Maybe analyze known-age users only (the 459)
   - Smaller N but 100% accurate demographics

---

## 10. WHEN YOU RETURN

1. Check if LLM classification completed (`improvement_run_output.log`)
2. Run tests: `python -m pytest tests/ -v`
3. Re-run full analysis with improved demographics
4. Decide on next architecture improvement to implement

---

Good luck on your trip! 🚀

