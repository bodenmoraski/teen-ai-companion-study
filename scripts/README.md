# Analysis Scripts

This folder contains the **current, active analysis scripts** for The Illusion Project.

## Active Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `COMPREHENSIVE_V3_ANALYSIS.py` | Main statistical analysis (RQ1, RQ2, RQ3) | `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` |
| `EXTENDED_ANALYSIS.py` | Robustness checks, binary analysis, variance tests | Appends to main results |
| `DEEP_DIVE_ANALYSIS.py` | Exploratory analysis: Why adults > teens | `results/deep_dive/` |

## How to Run

```bash
# Run the complete analysis pipeline
python scripts/COMPREHENSIVE_V3_ANALYSIS.py
python scripts/EXTENDED_ANALYSIS.py
python scripts/DEEP_DIVE_ANALYSIS.py
```

## Archive

The `archive/` folder contains legacy scripts from earlier analysis iterations. These are preserved for reference but are **not part of the current pipeline**.

---

*The Illusion Project - AnthroScore V3*
