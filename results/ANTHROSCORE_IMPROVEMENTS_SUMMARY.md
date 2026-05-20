# AnthroScore V3 Improvements: Summary of Changes & Results

## What Changed

We made three targeted improvements to the AnthroScore V3 LLM-based scoring algorithm:

1. **Human Calibration** — Incorporated 151 validation items scored by three independent human annotators (Stephanie, Boden, Afia) as few-shot calibration examples in the LLM prompt. The calibration identified a systematic overscoring bias in the original algorithm (algorithm mean 3.08 vs human mean 2.31 on validation items) and corrected for it.

2. **Emotion Attribution Distinction** — Added explicit prompt instructions and examples to distinguish *bot-attributed emotions* ("she gets jealous", "he really cares") from *user self-expression* ("I'm happy", "I love the app"). The original prompt conflated these, inflating scores for comments that merely mentioned emotional states without anthropomorphizing the AI.

3. **Enneagram-Based Emotional Chain Detection** — Replaced flat keyword matching with Enneagram personality-type emotional chain patterns, enabling detection of deeper emotional dynamics (e.g., sequences of attachment → fear of loss → idealization) rather than isolated emotion words.

## Validation Against Human Annotators (n=93)

|  | Exact Agreement | Within ±1 | Weighted κ |
|--|----------------|-----------|------------|
| Human–Human (avg) | 43.0% | 74.6% | 0.379 |
| Human–Old Algorithm | 35.8% | 70.6% | 0.306 |
| **Human–New Algorithm** | **47.7%** | **77.1%** | **0.383** |
| Consensus–Old Algorithm | 37.6% | 74.2% | 0.329 |
| **Consensus–New Algorithm** | **49.5%** | **81.7%** | **0.466** |

The improved algorithm now matches human-human inter-rater reliability (weighted κ = 0.383 vs 0.379) and exceeds it when compared against human consensus (weighted κ = 0.466).

## Main Dataset Results (274,191 comments, 96.6% coverage)

The dominant change was a 2→1 reclassification: 69% of all comments shifted from "minimal" to "none." The original algorithm was assigning score 2 to comments that mentioned AI bots in passing without actually anthropomorphizing them.

| Score | Original V3 | Improved V3 |
|-------|------------|-------------|
| 1 (None) | 14.2% | 88.3% |
| 2 (Minimal) | 76.0% | 10.1% |
| 3 (Moderate) | 5.8% | 0.2% |
| 4 (High) | 3.8% | 1.4% |
| 5 (Extreme) | 0.2% | 0.0% |
| **Mean** | **1.997** | **1.148** |

## Confirmatory Replication (45,704 comments, Jan 2024)

A separate dataset from r/CharacterAI collected from January 2024 (15 months before the main dataset's Apr–Jul 2025 window) was scored with the same improved algorithm.

| | Main (2025) | Confirmatory (2024) |
|--|------------|-------------------|
| Mean | 1.147 | 1.122 |
| Score 1 | 88.3% | 90.7% |
| Score 4–5 | 1.4% | 1.3% |
| **Cohen's d** | | **0.056 (negligible)** |

The score distributions are nearly identical across time periods, confirming that the improved methodology produces stable, replicable results.

## Files

| File | Description |
|------|-------------|
| `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet` | Improved scores for main dataset |
| `Data/confirmatory/confirmatory_scored.parquet` | Scored confirmatory dataset |
| `Data/features/user_anthroscores_improved.parquet` | User-level aggregated scores |
| `results/method_comparison/comparison_report.md` | Detailed comparison report |
| `results/method_comparison/*.png` | Visualizations (distributions, migration matrix, etc.) |
| `src/anthroscore/human_calibration.py` | Human calibration module |
| `src/anthroscore/enneagram_chains.py` | Enneagram emotional chain detection |
| `src/analysis/emotion_analysis.py` | Bot-attribution emotion analysis |
