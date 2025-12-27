# Quick Start Guide - Full Robust Methodology

## What's Been Fixed

✅ **Arctic Shift API** - Now working correctly  
✅ **Community Embeddings** - Fully implemented  
✅ **Complete Ensemble** - All three methods integrated

## Installation

```bash
# Install/update dependencies (includes gensim for embeddings)
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

## Quick Test

Test the API connection:
```bash
python scripts/test_arctic_shift.py
```

## Run the Pipeline

### Option 1: Run All Phases
```bash
python scripts/run_all_phases.py
```

### Option 2: Run Individually

**Phase 1: Collect Additional Subreddits** (should work now!)
```bash
python scripts/phase1_data_collection.py
```

**Phase 2: Demographics with Full Methodology**
```bash
python scripts/phase2_demographics.py
```
Now includes:
- Self-declaration extraction
- **Community embeddings** ✅ NEW
- LLM classification
- Ensemble classifier

**Phase 3: Core Analysis**
```bash
python scripts/phase3_core_analysis.py
```

**Phase 4: Statistical Analysis**
```bash
python scripts/phase4_statistical_analysis.py
```

## What You Get

After running all phases:

1. **Demographics Dataset** (`data/features/demographics.parquet`)
   - Age classifications from 3 methods combined
   - Gender classifications from 2 methods combined
   - Confidence scores

2. **Feature Dataset** (`data/features/full_merged_dataset.parquet`)
   - All demographics
   - AnthroScore per user
   - Topic distributions
   - Emotion profiles

3. **Results** (`results/tables/` and `results/figures/`)
   - Descriptive statistics
   - Regression models
   - Publication-ready figures

## Methodology Now Complete

```
Age Classification:
├── Self-Declaration (regex) ✅
├── Community Embeddings (Word2Vec) ✅ NEW
├── LLM (GPT-4o-mini) ✅
└── Ensemble (weighted) ✅

Gender Classification:
├── Self-Declaration (regex) ✅
├── Community Embeddings (Word2Vec) ✅ NEW
└── Ensemble ✅
```

## Notes

- **Community embeddings** work best when users participate in multiple subreddits
- Current dataset (r/CharacterAI only) may have limited diversity
- Running Phase 1 to collect additional subreddits will improve embeddings
- All methods are combined in ensemble for robustness

## Troubleshooting

**ImportError: No module named 'gensim'**
```bash
pip install gensim>=4.3.0
```

**API errors**
- Check internet connection
- API may be temporarily unavailable (retry later)
- Rate limiting is handled automatically

**Memory issues in Phase 3**
- Reduce batch sizes in scripts
- Use CPU instead of GPU (will be slower)

