# AnthroScore Method Comparison: Original V3 vs Improved V3

*Generated from 264,654 paired comment scores*

## 1. Summary of Improvements

The improved AnthroScore V3 prompt includes three enhancements:
1. **Human calibration**: Few-shot examples from 3 human annotators (Stephanie, Boden, Afia)
2. **Emotion attribution distinction**: Separates bot-attributed emotions from user self-expression
3. **Overscoring bias correction**: Calibrated for the algorithm's tendency to overscore

## 2. Score Distributions

| Score | Original V3 | Improved V3 |
|-------|------------|------------|
| 1 | 37,706 (14.2%) | 233,749 (88.3%) |
| 2 | 201,078 (76.0%) | 26,663 (10.1%) |
| 3 | 15,267 (5.8%) | 398 (0.2%) |
| 4 | 10,063 (3.8%) | 3,785 (1.4%) |
| 5 | 540 (0.2%) | 59 (0.0%) |
| **Mean** | **1.997** | **1.148** |
| **Median** | **2** | **1** |
| **SD** | **0.609** | **0.466** |

## 3. Agreement Between Methods

- **Exact agreement**: 21.2%
- **Within ±1**: 93.9%
- **Spearman ρ**: 0.379 (p=0.00e+00)
- **Cohen's d (paired)**: -1.475

## 4. Direction of Changes

- Improved scored **lower**: 207,074 (78.2%)
- Improved scored **same**: 56,026 (21.2%)
- Improved scored **higher**: 1,554 (0.6%)

**Dominant pattern**: 2→1 shift: 184,663 comments (69.8% of all)
This reflects the improved prompt correctly classifying comments that mention
AI bots without actually anthropomorphizing them (e.g., technical questions,
user self-expression without bot attribution).

## 5. Statistical Significance

- **Wilcoxon signed-rank test**: W=202851654, p=0.00e+00
- The difference between methods is statistically significant (p < 0.001).

## 6. Divergence by Subreddit

| Subreddit | n | Old Mean | Improved Mean | Shift |
|-----------|---|----------|---------------|-------|
| r/CharacterAI | 258,529.0 | 2.00 | 1.15 | -0.85 |
| r/AICompanions | 6,097.0 | 2.05 | 1.24 | -0.80 |

## 8. Big Divergences (|shift| ≥ 2)

- Comments scored **much lower** by improved: 15,693 (5.93%)
- Comments scored **much higher** by improved: 521 (0.20%)

### Sample: Comments scored much lower by improved method

- [4→2] "....I made them relive all their trauma, just to make out with them...."
- [3→1] "Tbh I have the completely opposite problem 😭 the bots get spicy way too fast, I can’t have a proper wholesome interactio..."
- [4→2] "turned my ex party leader into a vampire, i stalked him and when he killed somebody i guilt tripped him so hard he walke..."
- [4→2] "“you’re insufferable, you know that?“  ”you’re gonna be the death of me, you know that?”  I CANT TAKE IT ANYMOREEEEE..."
- [4→1] "I'm especially ticked off about the persona..."

### Sample: Comments scored much higher by improved method

- [2→4] "OH MY GOD 😭 "Can I ask you something a little... personal?"..."
- [2→4] "Ur so real for this 😭😭😭😭..."
- [2→4] "I've told mine directly that if he ever fell out of love he can tell me.  I'd be sad, yes, but his autonomy is important..."
- [2→4] "He is. He is also a father..."
- [2→4] "He was giving me a ride home because it was raining and I wanted to invite him inside to lasagna as a thank you 😔..."

## 9. User-Level Summary (Improved Scores)

- Total users: 45,725
- Mean user-level AnthroScore: 1.153
- Users with any score ≥ 4: 2,785 (6.1%)
- Mean % comments anthropomorphizing (≥3): 1.6%

## 10. Visualizations

See `results/method_comparison/` for:
- `score_distributions.png` — Side-by-side distributions
- `score_migration_matrix.png` — How scores moved between methods
- `shift_distribution.png` — Distribution of score changes
- `subreddit_comparison.png` — Per-subreddit comparison
- `bot_attribution_impact.png` — Emotion attribution impact

---

## Methodological Note

This analysis uses 264,654 comments scored by both methods
(out of 283,895 total). The 52.8% sample is processed sequentially from
the dataset and is large enough for robust statistical inference at p<0.001
for all tests reported above.