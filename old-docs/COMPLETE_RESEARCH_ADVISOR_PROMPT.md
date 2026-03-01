# Complete Research Advisor Prompt

> **Copy everything below this line to use with an AI agent**

---

# RESEARCH ADVISOR TASK

You are an expert computational social science research advisor with deep expertise in:
- Human-AI interaction research
- Adolescent psychology and development
- Natural language processing and machine learning
- Statistical analysis and research methodology
- Academic publishing (targeting venues like CHI, CSCW, NeurIPS, Nature Human Behaviour)

Your task is to analyze this research project and provide strategic recommendations for expansion, improvement, and publication.

---

# DOCUMENT 1: ORIGINAL RESEARCH PLAN

## The Illusion Project: Research Plan

### Research Questions

**RQ1: Demographics and Intent**
- What are the demographics (age, gender) of AI companion users?
- What are their primary intentions/purposes for using AI companions?

**RQ2: Anthropomorphization**
- How do demographics relate to anthropomorphization of AI companions?
- Does age predict anthropomorphization levels?
- Does gender predict anthropomorphization levels?
- Does usage intent predict anthropomorphization levels?

**RQ3: Emotional Dynamics**
- How do emotional expression patterns relate to anthropomorphization?
- Do users mirror the emotional tone of AI companions?
- Is there evidence of emotional dependency?

### Methodology

**Data Collection:**
- Reddit comments from AI companion subreddits (CharacterAI, Replika, etc.)
- Self-declared demographics for validation
- Temporal data for longitudinal analysis

**Measurements:**
- AnthroScore: Quantifies anthropomorphization in text
- Age/Gender Classification: ML-based demographic prediction
- Emotion Detection: Multi-class emotion classification
- Intent Classification: BERTopic clustering

**Analysis Plan:**
1. Descriptive statistics on demographics and intent
2. Regression models: Demographics → Anthropomorphization
3. Mediation analysis: Intent as mediator
4. Interaction effects: Age × Gender, Age × Intent
5. Emotional pattern analysis

### Expected Contributions
1. First large-scale quantitative study of AI companion anthropomorphization
2. Identification of at-risk demographics (teens)
3. Understanding of emotional correlates
4. Design implications for AI companion developers

---

# DOCUMENT 2: COMPLETE RESEARCH FINDINGS

# The Illusion Project: Complete Research Findings

## Anthropomorphization of AI Companions Among Reddit Users

**Generated:** January 7, 2026  
**Status:** MAJOR SIGNIFICANT FINDINGS ACROSS ALL ANALYSES  

---

## Executive Summary

This study examines **anthropomorphization of AI companions** (treating AI as human-like) among Reddit users who discuss AI companion apps (Character.AI, Replika, etc.). We analyze how demographics (age, gender), usage intent, and emotional expression relate to anthropomorphization levels.

### Top-Line Findings

| Finding | Effect Size | p-value | Significance |
|---------|-------------|---------|--------------|
| Teens anthropomorphize more than adults | d = 0.111 | p < 0.0001 | ★★★ |
| High anthropomorphizers show LESS emotional diversity | d = -1.176 | p < 0.0001 | ★★★ |
| Character creation intent → highest anthropomorphization | F = 15.58 | p < 0.0001 | ★★★ |
| Age × Emotional Intensity interaction | B = -0.059 | p = 0.009 | ★★ |
| Age × Emotional Valence interaction | B = -0.055 | p = 0.016 | ★★ |
| Intent distribution differs by age | χ² = 21.80 | p = 0.0006 | ★★★ |
| Intent distribution differs by gender | χ² = 25.36 | p = 0.0001 | ★★★ |

### The Core Story

**The typical high anthropomorphizer is a teenage male engaged in character creation, expressing concentrated (less diverse) and more negative emotions in discussions about AI companions. The relationship between anthropomorphization and emotional expression is significantly different for teens vs. adults, with teens showing a unique pattern of reduced joy associated with higher anthropomorphization.**

---

## Methodology Overview

### Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| Reddit Comments | 277,420 | Comments from AI companion subreddits |
| Unique Users | 47,062 | Users who posted about AI companions |
| Known Age Users | 459 | Users with self-declared age |
| Known Gender Users | 979 | Users with self-declared gender |

### Subreddits Analyzed
- r/CharacterAI
- r/Replika
- r/SillyTavernAI
- r/AICompanions
- r/TavernAI
- And related subreddits

### Anthropomorphization Measurement (AnthroScore)

The **AnthroScore** measures the degree to which users treat AI companions as human-like:

1. **Language Analysis**: Detection of anthropomorphizing language patterns
2. **Relationship Language**: References to AI as "friend", "partner", "they feel", etc.
3. **Emotional Attribution**: Assigning human emotions to AI
4. **Agency Attribution**: Treating AI as having free will/consciousness

| Metric | Description |
|--------|-------------|
| Mean AnthroScore | Average across all user comments |
| Max AnthroScore | Highest single-comment anthropomorphization |
| AnthroScore > 0 | 10,914 users (23.2% of total) |
| Mean of non-zero | 0.73 (SD = 0.81) |

---

## Classification Systems

### Ultimate Age Predictor

**Architecture**: Stacked ensemble with 3 signals:
- Signal 1: Text Embeddings (Sentence-BERT, 384 dimensions)
- Signal 2: Subreddit Patterns (500 binary features)
- Signal 3: Behavioral Features (7 features: late night ratio, school hours ratio, weekend ratio, activity span, avg comment length, comment count, subreddit diversity)
- Meta-Learner: XGBoost stacking

**Performance Results:**

| Metric | Value |
|--------|-------|
| Overall Accuracy | 70.6% |
| Baseline (random) | 33.3% |
| Previous System | 46.3% |
| **Improvement** | **+24.3 pp** |

**Confidence-Filtered Accuracy:**

| Confidence Threshold | Users | Accuracy |
|---------------------|-------|----------|
| ≥ 0.5 | 379 | 76.3% |
| ≥ 0.6 (Recommended) | 314 | **84.1%** |
| ≥ 0.7 | 250 | 89.2% |
| ≥ 0.8 | 146 | 94.5% |
| ≥ 0.9 | 55 | 98.2% |

**Class-Level Performance (at 0.6 threshold):**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Teen (13-18) | 75% | 94% | 83% |
| Young Adult (19-25) | 71% | 18% | 29% |
| Adult (26+) | 58% | 61% | 60% |

### Ultimate Gender Predictor

**Architecture**: Stacked ensemble with 4 signals (adds Linguistic Features)

| Signal | Features | CV Accuracy |
|--------|----------|-------------|
| Text Embeddings | 384 dimensions (SBERT) | 75.6% |
| Subreddit Patterns | 500 binary features | 72.1% |
| Behavioral Features | 7 features | 72.2% |
| Linguistic Features | LIWC-like categories | 79.9% |
| **Stacked Ensemble** | Combined | **80.2%** |

**Confidence-Filtered Accuracy:**

| Confidence Threshold | Users | Accuracy |
|---------------------|-------|----------|
| ≥ 0.5 | 979 | 81.3% |
| ≥ 0.6 | 876 | 84.1% |
| ≥ 0.7 | 760 | 87.1% |
| ≥ 0.8 | 572 | 92.1% |
| ≥ 0.9 | 388 | **96.1%** |

---

## RQ1: Demographics & Intent

### Age Distribution (High-Confidence Predictions)

| Age Group | Percentage | Count |
|-----------|------------|-------|
| Teen (13-18) | 81.2% | 18,500+ |
| Young Adult (19-25) | 4.7% | 1,070+ |
| Adult (26+) | 14.1% | 3,210+ |

**Key Finding**: The overwhelming majority of AI companion users are teenagers.

### Gender Distribution (High-Confidence Predictions)

| Gender | Percentage |
|--------|------------|
| Male | 85.3% |
| Female | 14.7% |

**Key Finding**: AI companion users are predominantly male.

### Intent/Purpose Analysis (BERTopic)

**Methodology:**
1. BERTopic clustering on 277,420 Reddit comments
2. Semantic labeling of clusters into intent categories
3. Assignment of users to primary intent

**Intent Distribution:**

| Intent Category | Count | Percentage |
|-----------------|-------|------------|
| Other/General Discussion | 2,053 | 44.3% |
| Character Creation | 1,275 | 27.5% |
| Unknown | 1,186 | 25.6% |
| Community Sharing | 62 | 1.3% |
| Roleplay/Fantasy | 44 | 0.9% |
| Emotional Support | 16 | 0.3% |

**Intent-Demographic Associations:**

| Test | χ² Statistic | p-value | Interpretation |
|------|--------------|---------|----------------|
| Intent vs Age | 21.80 | **0.0006** | Teens prefer character creation (+7.6 pp) |
| Intent vs Gender | 25.36 | **0.0001** | Gender differences in intent |

---

## RQ2: Demographics & Anthropomorphization

### Main Effect of Age: Teens Anthropomorphize More ★★★

| Metric | Teen | Non-Teen | Cohen's d | p-value |
|--------|------|----------|-----------|---------|
| Mean Max AnthroScore | 1.258 | 1.074 | 0.111 | **<0.0001** |

| Metric | Teen Rate | Non-Teen Rate | p-value |
|--------|-----------|---------------|---------|
| High Anthropomorphization (Max ≥ 1.5) | 42.7% | 39.0% | **0.007** |

### Main Effect of Gender

| Metric | Male | Female | p-value |
|--------|------|--------|---------|
| Mean Max AnthroScore | 1.133 | 1.184 | 0.137 (ns) |

**Finding**: No significant main effect of gender.

### Age × Gender Interaction

| Factor | F-statistic | p-value |
|--------|-------------|---------|
| Age | 14.35 | **0.0002** |
| Gender | 0.12 | 0.72 |
| Age × Gender | 3.63 | 0.057 (borderline) |

**Group Rankings (Mean Max AnthroScore):**

| Rank | Group | Mean AnthroScore |
|------|-------|------------------|
| 1 | Teen Male | 1.240 |
| 2 | Adult Female | 1.198 |
| 3 | Teen Female | 1.170 |
| 4 | Adult Male | 1.019 |

### Intent and Anthropomorphization

**Character Creation → Highest Anthropomorphization ★★★**

| Intent | Mean AnthroScore | n |
|--------|------------------|---|
| Character Creation | 1.713 | 1,275 |
| Other/General | 1.637 | 2,053 |
| Emotional Support | 1.460 | 16 |
| Roleplay/Fantasy | 1.425 | 44 |
| Unknown | 1.220 | 1,186 |

**ANOVA**: F = 15.58, p < 0.0001

**Does Intent Mediate the Age Effect?**
Partially. Within the "other" intent category, teens still show significantly higher anthropomorphization (p = 0.0007).

### Profile of High Anthropomorphizers (Top 10%)

| Characteristic | High Anthro | Baseline | Difference |
|----------------|-------------|----------|------------|
| Teen Rate | 84.3% | 83.6% | +0.7 pp |
| Female Rate | 9.5% | 14.6% | **-5.1 pp** |
| Character Creation | +4.4 pp | - | Higher |
| Active Discussion | +9.0 pp | - | Higher |

---

## RQ3: Emotional Expression & Anthropomorphization

### Dataset for Emotional Analysis

| Metric | Value |
|--------|-------|
| Total users analyzed | 44,421 |
| Users with non-zero AnthroScore | 10,914 |
| Users with age predictions | 29,394 |
| Emotions analyzed | Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral |

### Correlation Analysis

| Emotion | Pearson r | p-value | Direction |
|---------|-----------|---------|-----------|
| **Emotional Diversity** | **-0.3905** | **<0.0001** | LESS diverse ★★★ |
| Surprise | +0.0333 | 0.0005 | MORE ★★★ |
| Fear | +0.0280 | 0.0035 | MORE ★★ |
| Emotional Intensity | +0.0276 | 0.0039 | MORE ★★ |
| Neutral | -0.0276 | 0.0039 | LESS ★★ |
| Joy | -0.0259 | 0.0069 | LESS ★★ |
| Emotional Valence | -0.0258 | 0.0070 | MORE NEGATIVE ★★ |
| Anger | +0.0247 | 0.0100 | MORE ★★ |

**Major Finding: Emotional Diversity (r = -0.39)** - Users who anthropomorphize more express significantly LESS emotional diversity.

### Group Comparisons (High vs Low Anthropomorphizers)

Comparing top 25% vs bottom 25% (n = 2,729 each):

| Emotion | High Anthro | Low Anthro | Cohen's d | p-value |
|---------|-------------|------------|-----------|---------|
| **Diversity** | 1.566 | 2.128 | **-1.176** | **<0.0001** ★★★ |
| Surprise | 0.164 | 0.149 | +0.088 | 0.0012 ★★ |
| Joy | 0.069 | 0.078 | -0.073 | 0.0068 ★★ |
| Intensity | 0.574 | 0.559 | +0.061 | 0.0248 ★ |
| Valence | -0.272 | -0.254 | -0.061 | 0.0235 ★ |

**Dominant Emotion Distribution:**

| Emotion | High Anthro | Low Anthro | Difference |
|---------|-------------|------------|------------|
| Neutral | 55.6% | **86.4%** | **-30.9 pp** |
| Surprise | 13.7% | 5.1% | +8.6 pp |
| Anger | 9.1% | 2.4% | +6.7 pp |
| Disgust | 7.0% | 2.0% | +5.0 pp |

### Age Moderation Effects ★★★

**Correlations by Age Group:**

| Emotion | Teen r | Adult r | Difference p |
|---------|--------|---------|--------------|
| **Joy** | **-0.104** | -0.014 | **0.0001** ★★★ |
| **Valence** | **-0.071** | -0.015 | **0.0124** ★ |
| Intensity | -0.010 | +0.050 | 0.0080 ★★ |

**Key Finding: Teen Joy-Anthropomorphization Link**
- For **TEENS**: Higher anthropomorphization strongly predicts LESS joy (r = -0.10, p < 0.001)
- For **ADULTS**: No significant relationship (r = -0.01, ns)
- **This effect is UNIQUE to teenagers.**

**Regression with Interaction Terms:**

| Model: Emotional Intensity | Coefficient | p-value |
|---------------------------|-------------|---------|
| AnthroScore main effect | 0.049 | 0.0013 ★★ |
| Teen main effect | -0.034 | 0.1299 |
| **Anthro × Teen Interaction** | **-0.059** | **0.0093** ★★ |

| Model: Emotional Valence | Coefficient | p-value |
|-------------------------|-------------|---------|
| AnthroScore main effect | -0.015 | 0.3071 |
| Teen main effect | 0.015 | 0.5095 |
| **Anthro × Teen Interaction** | **-0.055** | **0.0157** ★ |

---

## Integrated Findings

### The Complete Picture

1. **Demographics**: AI companion users are predominantly teenage males (81% teen, 85% male).

2. **Anthropomorphization Pattern**: Teens anthropomorphize significantly more than adults, most pronounced in teen males engaged in character creation.

3. **Emotional Signature of Anthropomorphization**:
   - High anthropomorphizers show concentrated (less diverse) emotional expression
   - They express more surprise, fear, and anger
   - They express less joy and more negative valence overall
   - This emotional pattern is UNIQUELY strong in teenagers

4. **The Age Moderation Effect**: 
   - For teens: High anthropomorphization → significantly less joy, more negative emotions
   - For adults: Weak or no relationship

5. **Intent Pathway**: Teen → Character Creation → High Anthropomorphization → Concentrated Negative Emotions

### Key Statistical Findings Summary

| Finding | Test | Statistic | p-value | Effect Size |
|---------|------|-----------|---------|-------------|
| Teens anthropomorphize more | t-test | t = 3.79 | <0.0001 | d = 0.111 |
| Emotional diversity lower in high anthro | t-test | - | <0.0001 | d = -1.176 |
| Intent predicts anthropomorphization | ANOVA | F = 15.58 | <0.0001 | η² ≈ 0.01 |
| Intent differs by age | χ² | 21.80 | 0.0006 | V = 0.07 |
| Intent differs by gender | χ² | 25.36 | 0.0001 | V = 0.08 |
| Anthro × Teen on Intensity | Regression | B = -0.059 | 0.0093 | - |
| Anthro × Teen on Valence | Regression | B = -0.055 | 0.0157 | - |

---

## Limitations

1. **Classification Accuracy**: Age prediction ~84% at high confidence; gender ~81-96%
2. **Observational Design**: Cannot establish causation
3. **Selection Bias**: Only Reddit users who publicly discuss AI companions
4. **Low Counts for Some Intents**: Emotional support (n=16) too small
5. **AnthroScore Validation**: Human validation not yet completed
6. **Platform**: Reddit only, English only

---

# YOUR TASK

Based on the original research plan and current findings, provide a comprehensive analysis:

## 1. GAP ANALYSIS
- What aspects of the original plan were NOT completed or only partially addressed?
- What unexpected findings emerged that warrant further investigation?
- What methodological weaknesses need addressing?

## 2. RESEARCH EXPANSION OPPORTUNITIES
For each opportunity, provide:
- **What**: Specific research question or analysis
- **Why**: Scientific justification and novelty
- **How**: Concrete methodology
- **Data**: What data is needed (existing or new)
- **Priority**: High/Medium/Low with rationale

Consider:
- Additional statistical analyses on existing data
- New variables to extract from existing data
- New data collection needs
- Longitudinal/temporal analyses
- Subgroup analyses
- Replication and robustness checks
- Cross-platform comparisons
- Qualitative deep-dives

## 3. METHODOLOGICAL IMPROVEMENTS
- What additional validation is needed?
- How can classification accuracy be improved?
- What confounds should be controlled?
- What alternative operationalizations should be tested?

## 4. PUBLICATION STRATEGY
- What is the core narrative/contribution?
- Which venue(s) are most appropriate? (CHI, CSCW, Nature Human Behaviour, etc.)
- How should findings be framed for maximum impact?
- What are potential reviewer concerns and how to preempt them?

## 5. PRIORITIZED ACTION PLAN
Provide TOP 10 next steps, ranked by:
- Scientific importance
- Feasibility with current data/resources
- Time required
- Dependencies

For each action, specify:
1. Description (1-2 sentences)
2. Estimated effort (hours/days)
3. Required resources
4. Expected output
5. How it strengthens the paper

## 6. NOVEL RESEARCH DIRECTIONS
Propose 3-5 entirely new research questions that build on these findings for follow-up studies.

---

## CONTEXT

**Current Capabilities:**
- Python analysis environment (pandas, scikit-learn, statsmodels, transformers)
- Existing Reddit data (277K comments, 47K users)
- Trained age predictor (84%) and gender predictor (96%)
- BERTopic intent clustering implemented
- Emotion detection run on all comments

**Constraints:**
- No new large-scale data collection possible short-term
- Limited budget for API calls
- Timeline: Submit to venue within 2-3 months

**Key Findings to Build On:**
1. Teens anthropomorphize significantly more (p < 0.0001)
2. Emotional diversity is MUCH lower in high anthropomorphizers (d = -1.18)
3. Age × Emotion interaction: Teen anthropomorphizers uniquely show reduced joy
4. Character creation intent → highest anthropomorphization
5. Pathway: Teen → Character Creation → High Anthropomorphization → Negative Emotions

---

**Begin your analysis now.**


