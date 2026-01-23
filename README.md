# The Illusion Project: Anthropomorphization of AI Companions

Study of demographics (age, gender) and anthropomorphization in AI companion discussions on Reddit.

## Key documentation

| Document | Description |
|----------|-------------|
| [**METHODOLOGY_FINAL.md**](METHODOLOGY_FINAL.md) | Final methodology: data, age/gender classification, AnthroScore V3, **validation & accuracy** (95% age, 97% gender @ ≥0.6 confidence), handling of prediction vs self-declared contradictions |
| [**COMPREHENSIVE_V3_ANALYSIS_RESULTS.md**](COMPREHENSIVE_V3_ANALYSIS_RESULTS.md) | Full statistical analysis, extended checks, deep-dive results |
| [**FINAL_SHORT_SUMMARY.md**](FINAL_SHORT_SUMMARY.md) | Short summary for papers |
| [docs/QUICK_START.md](docs/QUICK_START.md) | Setup and run instructions |
| [experiments/v2_correction/FINAL_MODEL_SUMMARY.md](experiments/v2_correction/FINAL_MODEL_SUMMARY.md) | Age/gender classifier validation and confidence–accuracy tables |

## Structure

- **`scripts/`** — Analysis: `COMPREHENSIVE_V3_ANALYSIS`, `EXTENDED_ANALYSIS`, `DEEP_DIVE_ANALYSIS`
- **`src/`** — Data collection, demographics, AnthroScore, emotion analysis, statistics
- **`experiments/`** — AnthroScore V3 LLM scorer; v2_correction demographic models
- **`results/`** — Outputs (extended_analysis, deep_dive, JSON)
- **`old-docs/`** — Archived planning and legacy docs

## Quick start

See [docs/QUICK_START.md](docs/QUICK_START.md). Install deps with `pip install -r requirements.txt`.
