# The Illusion Project: Anthropomorphization of AI Companions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data for the research paper:

> **Anthropomorphism in AI Companion Communities: Age, Gender, and Emotional Correlates**
>
> *Investigating how demographics and emotions relate to anthropomorphization of AI chatbots in online communities.*

### Key Findings

- **Adults anthropomorphize AI companions significantly more than teens** (d = -0.46, contrary to common assumptions)
- **Women anthropomorphize more than men** (d = -0.31)
- **Joy is positively associated with anthropomorphization** (r = +0.10); **neutral expression is negatively associated** (r = -0.13)
- Demographics explain ~5% of variance in anthropomorphic tendency

---

## Quick Start

### Prerequisites

- Python 3.10+
- ~200 MB disk space for data files

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/illusion-project.git
cd illusion-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Paper Results

```bash
# Validate all paper statistics (recommended first step)
python scripts/validate_paper_statistics.py

# Run main statistical analysis
python scripts/COMPREHENSIVE_V3_ANALYSIS.py

# Run extended robustness checks
python scripts/EXTENDED_ANALYSIS.py
```

See [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md) for detailed verification steps.

---

## Repository Structure

```
.
├── Data/                                   # Data files
│   ├── processed/
│   │   └── all_comments.parquet           # 283,895 Reddit comments
│   ├── features/
│   │   ├── user_emotions.parquet          # User-level emotion scores
│   │   ├── user_anthroscores.parquet      # User-level AnthroIndex
│   │   └── self_declarations.parquet      # Ground truth demographics
│   ├── annotations/                        # Human validation data
│   └── confirmatory/                       # Confirmatory dataset
│
├── experiments/
│   ├── anthroscore_v3/                     # AnthroIndex scoring
│   │   ├── anthroscore_v3_improved_final.parquet  # Main scores
│   │   └── RESULTS.md                      # Validation results
│   └── v2_correction/                      # Demographics classifier
│       ├── age_predictions_v4.parquet     # Age predictions (paper)
│       ├── gender_predictions_v4.parquet  # Gender predictions (paper)
│       └── FINAL_MODEL_SUMMARY.md         # Model comparison
│
├── scripts/                                # Analysis scripts
│   ├── COMPREHENSIVE_V3_ANALYSIS.py       # Main analysis (paper results)
│   ├── validate_paper_statistics.py       # Reproducibility validation
│   ├── EXTENDED_ANALYSIS.py               # Robustness checks
│   ├── DEEP_DIVE_ANALYSIS.py              # Exploratory analysis
│   ├── generate_paper_figures.py          # Publication figures
│   └── utilities/                          # Helper scripts
│
├── src/                                    # Source code
│   ├── data_collection/                   # Reddit data collection
│   ├── demographics/                       # Age/gender classification
│   ├── anthroscore/                        # Anthropomorphization scoring
│   ├── analysis/                           # Analysis modules
│   └── statistical/                        # Statistical methods
│
├── results/                                # Analysis outputs
│   ├── COMPREHENSIVE_V3_ANALYSIS_RESULTS.md
│   ├── PAPER_VALIDATION_REPORT.md
│   └── paper_figures/                      # Publication figures
│
├── Validations/                            # Human annotation data
├── tests/                                  # Unit tests
├── docs/                                   # Additional documentation
│
├── METHODOLOGY_FINAL.md                    # Detailed methodology
├── REPRODUCTION_GUIDE.md                   # Step-by-step reproduction
├── REPRODUCIBILITY_CHECKLIST.md            # Verification checklist
└── requirements.txt                        # Python dependencies
```

---

## Methodology

### Data Collection

We collected 283,895 public Reddit comments from three AI companion communities:

| Subreddit | Comments | Retention |
|-----------|----------|-----------|
| r/CharacterAI | 269,040 | 67.7% |
| r/Replika | 8,380 | 83.8% |
| r/AICompanions | 6,475 | 86.0% |

**Date range:** January 2024 - December 2025

### Demographic Classification

Stacked ensemble classifier using LLM + ML features:

| Task | Accuracy | Coverage |
|------|----------|----------|
| Age (teen vs adult) | 95.0% | 96.5% |
| Gender (male vs female) | 96.9% | 92.7% |

### AnthroIndex (Anthropomorphization Score)

LLM-based classifier (GPT-4.1-nano) scoring comments 1-5:

| Score | Description | Example |
|-------|-------------|---------|
| 1 | Pure software/tool | "The app is buggy" |
| 2 | Minimal humanization | "It's pretty smart" |
| 3 | Moderate (pronouns) | "She seemed confused" |
| 4 | High (emotions) | "He really cares" |
| 5 | Human-equivalent | "We're in love" |

**Validation:** r = 0.59 with expert labels, 96% within-1-point accuracy

---

## Key Results

### Demographics and Anthropomorphization

| Comparison | Mean Difference | Effect Size |
|------------|-----------------|-------------|
| Adults vs Teens | Adults higher | d = -0.46 |
| Women vs Men | Women higher | d = -0.31 |

### Emotion Correlations

| Emotion | Correlation |
|---------|-------------|
| Joy | r = +0.10 |
| Neutral | r = -0.13 |
| Fear | r = +0.08 |
| Anger | r = +0.07 |

---

## Reproducibility

### Version Clarification

The script `COMPREHENSIVE_V3_ANALYSIS.py` uses:
- **AnthroScore V3**: LLM-based anthropomorphization scoring
- **Demographics V4**: Stacked ensemble classifier (produces paper statistics)

### Validation

```bash
python scripts/validate_paper_statistics.py
```

This confirms exact reproduction of:
- Sample sizes: N=16,347 (inclusive), n=5,160 (conditional)
- Effect sizes: d=-0.51 (age inclusive), d=-0.46 (age conditional)
- All means, SDs, and statistical tests

### Essential Data Files

| File | Size | Purpose |
|------|------|---------|
| `Data/processed/all_comments.parquet` | 31 MB | Core dataset |
| `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet` | 14 MB | AnthroIndex scores |
| `experiments/v2_correction/age_predictions_v4.parquet` | 2 MB | Age predictions |
| `experiments/v2_correction/gender_predictions_v4.parquet` | 2 MB | Gender predictions |
| `Data/features/user_emotions.parquet` | 4 MB | Emotion scores |

---

## Citation

```bibtex
@article{illusionproject2026,
  title={Anthropomorphism in AI Companion Communities: Age, Gender, and Emotional Correlates},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Ethical Considerations

- Uses publicly available Reddit data
- No personally identifiable information included
- User privacy protected through paraphrasing
- IRB exempt (45 C.F.R. 46.104, 2018)

## Contact

For questions, please open a GitHub issue.

---

*The Illusion Project: Understanding Human-AI Relationships*
