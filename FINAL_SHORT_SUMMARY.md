# The Illusion Project: Executive Summary

**Anthropomorphization of AI Companions on Reddit**

*A Computational Social Science Study*

---

## Research Question

**Do demographics (age, gender) and emotional expression patterns predict how users anthropomorphize AI companions?**

---

## Key Finding

> **Adults anthropomorphize AI companions significantly MORE than teens.**
>
> This contradicts the "digital native" assumption that tech-savvy teens would form more human-like relationships with AI.

---

## Sample

| Metric | Value |
|--------|-------|
| Total comments | 283,895 |
| Unique users | 47,062 |
| Analysis sample (high-confidence) | 15,281 |
| Subreddits | CharacterAI, Replika, AICompanions |

---

## Primary Results

### RQ1: Who Uses AI Companions?

| Demographic | Percentage |
|-------------|------------|
| **Male** | 81.4% |
| **Female** | 18.6% |
| **Teen (13-18)** | 80.4% |
| **Adult (19+)** | 19.6% |

*AI companion communities are predominantly young and male.*

---

### RQ2: Demographics → Anthropomorphization

| Effect | Cohen's d | p-value | Direction |
|--------|-----------|---------|-----------|
| **Age** | -0.501 (medium) | < 0.0001 | Adults higher |
| **Gender** | -0.292 (small) | < 0.0001 | Females higher |
| **Interaction** | - | 0.013 | Significant |

**Subgroup ranking (highest to lowest):**
1. Adult Female (M = 2.35)
2. Adult Male (M = 2.21)
3. Teen Female (M = 2.11)
4. Teen Male (M = 2.02)

**Binary Analysis:**
- Adults are **2.0x more likely** to be "high anthropomorphizers"
- Females are **2.2x more likely** to be "high anthropomorphizers"

---

### RQ3: Emotions → Anthropomorphization

| Emotion | Correlation (r) | Direction |
|---------|-----------------|-----------|
| **Joy** | +0.115 | Positive |
| **Neutral** | -0.116 | Negative |
| Fear | +0.047 | Positive |
| Anger | +0.039 | Positive |
| Sadness | +0.037 | Positive |

*High anthropomorphizers express more joy and less neutral content.*

**Age Moderation:**
- Joy–Anthro relationship is **stronger for adults** (r = 0.14 vs 0.10)
- Anger–Anthro relationship is **stronger for teens** (r = 0.05 vs 0.02)

---

## Why Do Adults Anthropomorphize More?

| Hypothesis | Evidence | Support |
|------------|----------|---------|
| Adults are lonelier | 1.6x more loneliness language | ✅ |
| Adults seek romance/emotion | 2x more romantic content | ✅ |
| Adults are more invested | 65% longer comments | ✅ |
| Not just platform selection | Effect persists within subreddits | ✅ |

**Key Insight:** Adults appear to use AI companions to fulfill emotional and relational needs, while teens use them more for entertainment.

---

## Methodology Highlights

### AnthroScore V3

- **Method:** LLM-based classification (GPT-4.1-nano)
- **Scale:** 1 (None) to 5 (Extreme)
- **Validation:** r = 0.59 with expert labels (vs r = 0.11 for old method)
- **Improvement:** 5.4x better correlation with expert judgment

### Demographics

- **Method:** LLM-based classification (GPT-4o-mini)
- **Confidence threshold:** ≥ 60%
- **Binary categories:** Teen/Adult, Male/Female

### Robustness

- Effect consistent across all confidence thresholds (0.50–0.70)
- Non-parametric tests confirm all findings
- Effect persists within each subreddit

---

## Implications

### For AI Safety

1. **Vulnerable population:** Lonely adults may be most susceptible to AI attachment
2. **Platform matters:** Relationship-focused AI (Replika) shows highest anthropomorphization
3. **Education focus:** AI literacy may need to target adults, not just teens

### For Research

1. **Measurement matters:** LLM-based measures dramatically outperform older methods
2. **Age effects reversed:** Better measurement revealed opposite of initial findings
3. **Small variance explained:** Demographics explain only ~5% — psychological factors matter more

---

## Limitations

1. Reddit sample only — may not generalize
2. Cross-sectional design — no causal claims
3. Binary demographics — does not capture full identity spectrum
4. LLM-based validation — not human ground truth

---

## Files

| Document | Purpose |
|----------|---------|
| `COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` | Full statistical report |
| `METHODOLOGY_FINAL.md` | Detailed methodology |
| `results/deep_dive/` | Exploratory analysis |
| `results/extended_analysis/` | Robustness visualizations |

---

## Bottom Line

**Adults, particularly adult females, anthropomorphize AI companions the most — likely because they seek emotional connection. This challenges assumptions about "digital native" teens and has implications for AI safety and design.**

---

*The Illusion Project | January 2026*
