# The Illusion Project

## Anthropomorphization of AI Companions: A Computational Social Science Study

This research project examines how Reddit users anthropomorphize AI companion applications (Character.AI, Replika, etc.), with a focus on demographic differences, usage intent, and emotional expression patterns.

---

## 🔬 Research Overview

### Research Questions

1. **RQ1 (Demographics & Intent)**: What are the demographics and usage intentions of AI companion users?
2. **RQ2 (Anthropomorphization)**: How do demographics relate to anthropomorphization of AI companions?
3. **RQ3 (Emotional Expression)**: How do emotional expression patterns relate to anthropomorphization?

### Key Findings

| Finding | Effect Size | p-value |
|---------|-------------|---------|
| Teens anthropomorphize more than adults | d = 0.111 | p < 0.0001 |
| High anthropomorphizers show LESS emotional diversity | d = -1.176 | p < 0.0001 |
| Character creation → highest anthropomorphization | F = 15.58 | p < 0.0001 |
| Age × Emotional Expression interaction effects | B = -0.06 | p < 0.01 |

**The Core Story**: Teen males engaged in character creation show the highest anthropomorphization levels and exhibit concentrated (less diverse), more negative emotional expression patterns.

---

## 📊 Methodology

### Data
- **Source**: Reddit (AI companion subreddits)
- **Comments**: 277,420
- **Unique Users**: 47,062
- **Known Age Users**: 459 (for validation)
- **Known Gender Users**: 979 (for validation)

### Classification Systems

| System | Accuracy | Method |
|--------|----------|--------|
| Age Predictor | 84.1% (high-conf) | Stacked Ensemble (SBERT + Subreddits + Behavior) |
| Gender Predictor | 96.1% (high-conf) | Stacked Ensemble (SBERT + Subreddits + Linguistic) |

### Analysis Tools
- **AnthroScore**: Measures anthropomorphization in text
- **BERTopic**: Intent/purpose clustering
- **Emotion Detection**: 7-category emotion classification

---

## 📁 Project Structure

```
├── MASTER_RESEARCH_FINDINGS.md    # Complete findings document
├── src/
│   ├── analysis/                  # Core analysis modules
│   ├── demographics/              # Age/gender prediction
│   ├── anthroscore/               # Anthropomorphization scoring
│   └── statistical/               # Statistical analysis
├── scripts/
│   ├── run_ultimate_predictor.py  # Age prediction
│   ├── run_ultimate_gender_predictor.py  # Gender prediction
│   └── rq3_emotional_analysis.py  # Emotional analysis
├── results/                       # Analysis outputs
├── Data/                          # Data files (not in git)
└── docs/                          # Documentation
```

---

## 🚀 Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas, numpy, scipy
- scikit-learn, xgboost
- sentence-transformers
- bertopic

### Run Analysis

```bash
# Age prediction
python scripts/run_ultimate_predictor.py

# Gender prediction  
python scripts/run_ultimate_gender_predictor.py

# Emotional analysis (RQ3)
python scripts/rq3_emotional_analysis.py
```

---

## 📄 Key Documents

- **[MASTER_RESEARCH_FINDINGS.md](MASTER_RESEARCH_FINDINGS.md)** - Complete findings with all statistics
- **[COMPREHENSIVE_RESEARCH_PLAN.md](COMPREHENSIVE_RESEARCH_PLAN.md)** - Original research plan
- **[docs/ULTIMATE_PREDICTOR_ARCHITECTURE.md](docs/ULTIMATE_PREDICTOR_ARCHITECTURE.md)** - Classification system design

---

## 📈 Results Summary

### Demographics (RQ1)
- 81% teenagers, 85% male
- Primary intent: Character creation (28%), General discussion (44%)

### Anthropomorphization (RQ2)
- Teens significantly higher anthropomorphization (p < 0.0001)
- Character creation intent → highest anthropomorphization
- Teen males are the highest-risk demographic

### Emotional Expression (RQ3)
- High anthropomorphizers: less emotional diversity (d = -1.18)
- More surprise, fear, anger; less joy
- **Age interaction**: Teen anthropomorphization uniquely linked to reduced joy

---

## ⚠️ Limitations

- Classification accuracy ~84-96% (not perfect)
- Observational design (no causation)
- Reddit-only sample
- English language only

---

## 📜 License

MIT License - See [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- AnthroScore methodology adapted from existing research
- Sentence-BERT by Reimers & Gurevych (2019)
- Reddit data via Arctic Shift API
