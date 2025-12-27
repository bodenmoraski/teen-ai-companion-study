# Teen-AI Companion Relationships on Reddit

A computational social science study examining how adolescents interact with AI companions, using NeurIPS-level statistical methodology.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This research analyzes ~250,000 Reddit comments to understand teen-AI companion relationships through:

- **Multi-method demographic classification** combining self-declaration, community embeddings, and LLM inference
- **AnthroScore V2** for measuring anthropomorphization tendencies
- **Rigorous statistical analysis** with measurement error correction and power analysis

### Research Questions

| RQ | Question | Method |
|----|----------|--------|
| **RQ1a** | Age distribution of AI companion users | 5-bucket hybrid classification |
| **RQ1b** | Gender distribution | Self-report + community embedding |
| **RQ2** | Demographics × anthropomorphization | OLS regression with corrections |

## Project Structure

```
├── src/
│   ├── analysis/           # Core analysis modules
│   ├── anthroscore/        # AnthroScore V2 implementation
│   ├── demographics/       # Age/gender classification
│   ├── statistical/        # Regression, power analysis, robustness
│   └── utils/              # Configuration and utilities
├── scripts/                # Execution scripts
├── tests/                  # Test suite
├── Data/                   # Data directory (not tracked)
│   ├── raw/                # Raw JSONL from Arctic Shift
│   ├── processed/          # Cleaned data
│   └── features/           # Extracted features
├── results/                # Analysis outputs
│   ├── figures/            # Publication figures
│   ├── neurips/            # NeurIPS-level reports
│   └── tables/             # Statistical tables
└── docs/                   # Documentation
```

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key (for LLM-based classification)

### Setup

```bash
# Clone repository
git clone https://github.com/bodenmoraski/teen-ai-companion-study.git
cd teen-ai-companion-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Verify installation
python verify_setup.py
```

## Usage

### Full Pipeline

```bash
# Run complete NeurIPS-level analysis
python scripts/run_comprehensive_analysis.py
```

### Individual Phases

```bash
# Phase 1: Data collection
python scripts/phase1_data_collection.py

# Phase 2: Demographics classification
python scripts/phase2_with_api_data.py

# Phase 3-4: Analysis and statistics
python scripts/targeted_phase3_phase4.py
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_criticism_fixes.py -v
```

## Methodology

### Demographic Classification

We use a **three-method ensemble** approach:

1. **Self-declaration extraction** - Pattern matching for explicit age/gender mentions
2. **Community embedding** - Word2Vec on subreddit participation patterns
3. **LLM inference** - GPT-4.1-nano for ambiguous cases

### Statistical Rigor

Our analysis addresses common methodological concerns:

- **Measurement error correction** via reliability coefficients
- **Power analysis** with minimum detectable effect calculation
- **Multicollinearity** checking (VIF < 5)
- **Heteroscedasticity** robust standard errors
- **Influential observations** analysis (Cook's D)

See [`LIMITATIONS.md`](LIMITATIONS.md) for honest framing of all limitations.

## Key Findings

Results are documented in:
- `results/neurips/comprehensive_summary.txt` - Full analysis summary
- `results/tables/` - Statistical tables
- `results/figures/` - Publication-ready figures

## Documentation

| Document | Description |
|----------|-------------|
| [`COMPREHENSIVE_RESEARCH_PLAN.md`](COMPREHENSIVE_RESEARCH_PLAN.md) | Full methodology |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Honest limitations disclosure |
| [`ARCHITECTURE_IMPROVEMENT_PLAN.md`](ARCHITECTURE_IMPROVEMENT_PLAN.md) | Future improvements |

## Citation

If you use this work, please cite:

```bibtex
@article{teen_ai_companion_2025,
  title={Teen-AI Companion Relationships on Reddit: A Computational Social Science Study},
  author={[Your Name]},
  year={2025},
  note={NeurIPS Submission}
}
```

## References

1. Cheng et al. (2024). AnthroScore: A Computational Linguistic Measure of Anthropomorphism. *EACL*.
2. Chew et al. (2021). Predicting Age Groups of Reddit Users. *JMIR Public Health*.
3. Waller & Anderson (2025). Uncovering the Sociodemographic Fabric of Reddit. *arXiv:2502.05049*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
