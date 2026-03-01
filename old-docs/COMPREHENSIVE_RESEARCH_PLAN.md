# Comprehensive Research Implementation Plan
## Teen-AI Companion Relationships on Reddit

**Generated:** December 25, 2025  
**Status:** Ready for execution  
**Philosophy:** Maximally ambitious, AI-powered research pipeline

---

## Executive Summary

This plan leverages cutting-edge 2025 methodologies to study teen-AI companion relationships at scale. Key innovations include:

1. **Multi-bucket age classification** (5 groups) using a novel hybrid approach
2. **Dual gender methodology** (self-report + community embedding with validation)
3. **Streamlined RQ3** using comment-level emotion analysis (no VLM needed)
4. **Full automation** via structured AI agent prompts

**Why This Matters (2025 Context):**
- 64% of US teens use AI chatbots, 30% daily (Pew Research, Dec 2025)
- 72% of teens have used AI companions (Common Sense Media, July 2025)
- Character.AI banned under-18 from chat (Nov 25, 2025)
- NY and CA enacted AI companion safety laws (2025)
- FTC launched inquiry into AI companions for minors (2025)

---

## Research Questions (Refined)

| RQ | Question | Primary Method |
|----|----------|----------------|
| **RQ1a** | What is the age distribution of users discussing AI companions? | 5-bucket classifier (hybrid Chew V2 + LLM) |
| **RQ1b** | What is the gender distribution? | Self-report + Community embedding |
| **RQ1c** | What are the dominant interaction patterns? | BERTopic clustering |
| **RQ2** | How do demographics correlate with anthropomorphization? | AnthroScore V2 × demographics regression |
| **RQ3** | Do users mirror the emotional patterns of their AI companions? | Comment-level emotion trajectories |

---

## Data Pipeline

### Sources
```
┌─────────────────────────────────────────────────────────┐
│                    Arctic Shift API                      │
├─────────────────────────────────────────────────────────┤
│  r/CharacterAI (current: 6,570 comments)                │
│  r/Replika (to collect)                                  │
│  r/replika_ai, r/AICompanions, r/SocialChatbots         │
│  Time Range: Jan 2024 - Dec 2025 (maximize temporal span)│
└─────────────────────────────────────────────────────────┘
```

### Data Schema (JSONL)
```json
{
  "id": "comment_id",
  "author": "username",
  "body": "comment text",
  "created_utc": 1234567890,
  "subreddit": "CharacterAI",
  "link_id": "t3_post_id",
  "parent_id": "t1_parent_comment_or_t3_post",
  "score": 42,
  "author_flair_text": "16F | Replika User"
}
```

### Processing Steps
1. **Deduplication** - Remove exact duplicates by `id`
2. **Bot filtering** - Remove AutoModerator, known bots
3. **Deleted filtering** - Remove [deleted]/[removed] authors
4. **Text quality** - Min 20 characters, max 10,000
5. **Author aggregation** - Group comments by author for feature extraction

---

## Component 1: Age Classification (5 Buckets)

### The Challenge
Chew V2 is binary (13-20 vs 21-54). We need 5 buckets:
- **13-18** (minors)
- **19-25** (young adults)
- **26-40** (adults)
- **41-60** (middle-aged)
- **61-80** (older adults)

### Hybrid Solution

**Step 1: Extract self-declarations**
```python
# Regex patterns for explicit age mentions
patterns = [
    r'\b(?:I am|I\'m|im)\s*(\d{1,2})\b',
    r'\b(\d{1,2})\s*(?:years?\s*old|yo|y\.?o\.?)\b',
    r'\b(\d{1,2})\s*[MF]\b',  # "16F", "23M"
    r'\bas a\s*(\d{1,2})\b',
]
```

**Step 2: Community embedding age dimension**

Based on [Toronto CSS Lab methodology](http://csslab.cs.toronto.edu/reddit/):
- Seed pairs: r/teenagers ↔ r/RedditForGrownups
- Augment with: r/teenrelationships ↔ r/relationship_advice
- Project user subreddit vectors onto age dimension

**Step 3: LLM-assisted classification**

For users without self-declaration or clear community signal:
```python
prompt = """
Analyze this Reddit user's comment history and estimate their most likely age bucket.
Consider: vocabulary, topics, life stage indicators, cultural references.

Comments:
{comments}

Age buckets: 13-18, 19-25, 26-40, 41-60, 61-80
Respond with JSON: {"age_bucket": "...", "confidence": 0.0-1.0, "reasoning": "..."}
"""
```

**Step 4: Ensemble confidence scoring**
```python
final_age = weighted_vote(
    self_declaration_age,      # weight: 1.0 (if available)
    community_embedding_age,   # weight: 0.7
    llm_estimated_age,         # weight: 0.6
)
confidence = calculate_agreement(sources)
```

### Validation
- Compare against 850k self-declaration dataset from [arxiv:2502.05049](https://arxiv.org/abs/2502.05049)
- Manual annotation of 50 random users by research team
- Inter-rater reliability (Krippendorff's α > 0.8)

---

## Component 2: Gender Classification

### Approach A: Self-Report Filtering
```python
# Pattern matching in posts/comments
gender_patterns = {
    'male': [r'\b(\d{1,2})\s*M\b', r'\bmale\b', r'\bguy\b', r'\bman\b', r'\bhe/him\b'],
    'female': [r'\b(\d{1,2})\s*F\b', r'\bfemale\b', r'\bgirl\b', r'\bwoman\b', r'\bshe/her\b'],
    'nonbinary': [r'\bthey/them\b', r'\benby\b', r'\bnon-?binary\b', r'\bnb\b'],
}

# Also check author_flair_text field
```

### Approach B: Community Embedding
```python
# Seed pairs for gender dimension
seed_pairs = [
    ('TwoXChromosomes', 'MensRights'),
    ('AskWomen', 'AskMen'),
    ('girlsgonewild', 'ladybonersgw'),  # NSFW validation pairs
]

# User vector projection
gender_score = project_user_vector(user_subreddits, gender_dimension)
```

### Validation Strategy
1. Run both methods on full dataset
2. Calculate agreement (Cohen's κ)
3. For disagreements: manual review sample
4. Report both results in paper with discussion of tradeoffs

---

## Component 3: Interaction Pattern Clustering (BERTopic)

### Pipeline
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# 1. Use domain-appropriate embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Configure for Reddit data
topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=50,
    nr_topics='auto',
    calculate_probabilities=True
)

# 3. Fit on comment bodies
topics, probs = topic_model.fit_transform(comments)

# 4. Extract interpretable labels
topic_info = topic_model.get_topic_info()
```

### Expected Topics (Based on Prior Research)
- Emotional support seeking
- Romantic/relationship themes
- Roleplay/creative writing
- Technical troubleshooting
- Platform comparison (CAI vs Replika)
- Grief/loss processing
- Daily companionship
- Mental health discussions

---

## Component 4: AnthroScore V2 Analysis

### Execution
```python
from anthroscore_v2 import AnthroScoreV2

scorer = AnthroScoreV2(use_twitter_model=True, device='cuda')

# Batch process all comments
df['anthroscore'] = df['body'].apply(
    lambda x: scorer.compute_score(x)['mean_score']
)

# Aggregate to user level
user_anthro = df.groupby('author')['anthroscore'].agg(['mean', 'std', 'count'])
```

### Analysis Plan
1. **Descriptive stats** by subreddit, age bucket, gender
2. **Regression models**: AnthroScore ~ Age + Gender + Topic + (Age×Gender)
3. **Temporal trends**: Has anthropomorphization changed over 2024-2025?
4. **Platform comparison**: Character.AI vs Replika framing differences

---

## Component 5: Emotional Dynamics (Simplified RQ3)

### Key Insight
We don't need VLM/OCR for screenshots. Users describe their AI's emotions in comments:
> "My Replika was so happy today when I told her about my promotion"
> "He seemed sad when I said I was busy"

### Approach: Comment-Level Emotion Analysis

**Step 1: Emotion classification with RoBERTa**
```python
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Extract emotions for user AND described AI
emotions = emotion_classifier(comment)
```

**Step 2: Parse AI-directed vs user emotions**
```python
# Separate sentences about AI vs about self
ai_sentences = extract_sentences_about_ai(comment)
user_sentences = extract_sentences_about_self(comment)

user_emotions = classify_emotions(user_sentences)
ai_described_emotions = classify_emotions(ai_sentences)
```

**Step 3: Emotional mirroring metric**
```python
# Calculate similarity between user emotions and AI-described emotions
mirroring_score = cosine_similarity(user_emotions, ai_described_emotions)

# Or correlation over time (longitudinal users)
temporal_correlation = calculate_dtw(user_trajectory, ai_trajectory)
```

### Alternative: Sentiment Trajectory
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Track sentiment over user's posting history
user_sentiment_trajectory = [
    analyzer.polarity_scores(comment)['compound']
    for comment in user_comments_sorted_by_time
]
```

---

## Implementation Architecture

### Repository Structure
```
illusion-project/
├── data/
│   ├── raw/                    # JSONL from Arctic Shift
│   ├── processed/              # Cleaned, deduplicated
│   ├── features/               # Extracted features
│   └── annotations/            # Manual validation data
├── src/
│   ├── data_collection/
│   │   ├── arctic_shift.py     # API wrapper
│   │   └── preprocess.py       # Cleaning pipeline
│   ├── demographics/
│   │   ├── age_classifier.py   # 5-bucket hybrid
│   │   ├── gender_classifier.py
│   │   └── community_embedding.py
│   ├── analysis/
│   │   ├── anthroscore_runner.py
│   │   ├── bertopic_clustering.py
│   │   └── emotion_analysis.py
│   ├── statistical/
│   │   ├── regression_models.py
│   │   └── correlation_analysis.py
│   └── utils/
│       ├── reddit_api.py
│       └── visualization.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_demographics_analysis.ipynb
│   ├── 03_anthropomorphization.ipynb
│   └── 04_emotional_dynamics.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
├── config/
│   └── config.yaml
├── requirements.txt
├── README.md
└── AGENT_PROMPTS.md           # Structured prompts for AI agent
```

---

## Execution Phases

### Phase 1: Data Collection & Preprocessing (Day 1)
- [ ] Collect r/Replika data via Arctic Shift
- [ ] Collect additional subreddits (r/AICompanions, etc.)
- [ ] Standardize all data to JSONL schema
- [ ] Run preprocessing pipeline
- [ ] Generate descriptive statistics

### Phase 2: Demographics Extraction (Day 1-2)
- [ ] Implement self-declaration regex patterns
- [ ] Collect user subreddit participation data
- [ ] Build community embedding for age/gender dimensions
- [ ] Run LLM classification for uncertain users
- [ ] Create demographic feature dataset

### Phase 3: Core Analysis (Day 2-3)
- [ ] Run AnthroScore V2 on all comments
- [ ] Run BERTopic clustering
- [ ] Run emotion analysis
- [ ] Aggregate features to user level
- [ ] Merge all feature sets

### Phase 4: Statistical Analysis (Day 3-4)
- [ ] Descriptive statistics by demographic groups
- [ ] Regression models (RQ2)
- [ ] Emotional mirroring analysis (RQ3)
- [ ] Sensitivity analyses
- [ ] Generate all figures/tables

### Phase 5: Validation & Writing (Day 4-5)
- [ ] Manual annotation of sample
- [ ] Inter-rater reliability
- [ ] Draft results section
- [ ] Create visualizations
- [ ] Compile supplementary materials

---

## Cost Estimates

| Component | Method | Estimated Cost |
|-----------|--------|----------------|
| AnthroScore V2 | Local GPU | $0 |
| BERTopic | Local GPU | $0 |
| Emotion classification | Local GPU | $0 |
| Age LLM classification | GPT-4o-mini API | ~$2-5 (10k users) |
| Community embedding | Local compute | $0 |
| Self-declaration regex | Local compute | $0 |
| **Total** | | **~$5** |

---

## Key Methodological Innovations

### 1. Hybrid Age Classification
- First to combine self-declaration, community embedding, AND LLM estimation
- Enables 5-bucket classification (vs. binary in prior work)
- Full confidence scoring with transparency

### 2. Dual Gender Validation
- Running both self-report and community embedding
- Reporting agreement and discussing limitations
- More rigorous than either method alone

### 3. Simplified Emotional Dynamics
- Avoids heavy VLM compute by analyzing comment descriptions
- Captures user reports of AI emotional states
- Enables mirroring analysis without image processing

### 4. Temporal Scope
- Jan 2024 - Dec 2025 captures key policy changes
- Pre/post Character.AI under-18 ban comparison
- Captures rapid evolution of AI companion space

---

## Expected Contributions

1. **First large-scale demographic analysis** of AI companion users across age/gender
2. **Novel 5-bucket age methodology** validated against multiple sources
3. **Quantitative anthropomorphization patterns** by demographic group
4. **Evidence on emotional mirroring** in AI companion relationships
5. **Policy-relevant findings** during critical regulatory moment

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Low LLM age accuracy | Validate against self-declarations; report confidence |
| Gender misclassification | Dual-method validation; report disagreement rates |
| Computational limits | Use efficient models (distilroberta, MiniLM) |
| API costs | GPT-4o-mini; batch processing; caching |
| IRB concerns | Aggregate analysis only; no individual identification |

---

## Next Steps

1. **Confirm this plan looks good**
2. **Set up repository structure**
3. **Create detailed agent prompts for each component**
4. **Begin Phase 1 execution**

---

## References

1. Cheng et al. (2024). AnthroScore: A Computational Linguistic Measure of Anthropomorphism. EACL.
2. Chew et al. (2021). Predicting Age Groups of Reddit Users. JMIR Public Health.
3. Waller & Anderson (2025). Uncovering the Sociodemographic Fabric of Reddit. arXiv:2502.05049.
4. Toronto CSS Lab (2021). Social media's social structure. Nature.
5. Pew Research (2025). Teens, Social Media and AI Chatbots 2025.
6. Common Sense Media (2025). Talk, Trust, and Trade-offs: How and Why Teens Use AI Companions.
