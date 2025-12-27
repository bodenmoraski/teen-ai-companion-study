# Execution Guide: Complete Research Pipeline

This guide provides step-by-step instructions for executing the complete research pipeline.

## Prerequisites

1. **Python 3.10+** installed
2. **All dependencies installed**: `pip install -r requirements.txt`
3. **spaCy model**: `python -m spacy download en_core_web_sm`
4. **OpenAI API Key** in `.env` file (for LLM classification)

## Quick Start

### Option 1: Run All Phases Automatically

```bash
python scripts/run_all_phases.py
```

This will execute all phases sequentially. Note: This will take significant time and compute resources.

### Option 2: Run Phases Individually

Run each phase separately for better control and error handling:

#### Phase 1: Data Collection & Preprocessing
```bash
python scripts/phase1_data_collection.py
```

**Output:**
- `data/processed/all_comments.parquet` - Cleaned, preprocessed comments
- `data/processed/collection_statistics.txt` - Collection statistics

#### Phase 2: Demographics Extraction
```bash
python scripts/phase2_demographics.py
```

**Requirements:** OpenAI API key configured in `.env`

**Output:**
- `data/features/demographics.parquet` - User demographics (age, gender)
- `data/features/self_declarations.parquet` - Self-declared demographics
- `data/features/llm_classifications.parquet` - LLM-based classifications

**Note:** This phase uses OpenAI API and may incur costs. It processes up to 5000 users by default for cost control.

#### Phase 3: Core Analysis
```bash
python scripts/phase3_core_analysis.py
```

**Requirements:**
- GPU recommended for AnthroScore V2 (will use CPU if unavailable)
- Significant compute time required

**Output:**
- `data/features/user_anthroscores.parquet` - AnthroScore aggregations
- `data/features/user_topics.parquet` - Topic distributions
- `data/features/user_emotions.parquet` - Emotion distributions
- `data/features/full_merged_dataset.parquet` - Combined feature set

**Note:** This phase is computationally intensive. Consider processing a sample first by modifying `SAMPLE_SIZE` in the script.

#### Phase 4: Statistical Analysis
```bash
python scripts/phase4_statistical_analysis.py
```

**Output:**
- `results/tables/descriptive_statistics.txt` - Descriptive statistics
- `results/tables/regression_results.txt` - Regression model results
- `results/tables/correlation_matrix.csv` - Correlation table
- `results/figures/age_distribution.png` - Age distribution plot
- `results/figures/anthroscore_by_demographics.png` - AnthroScore analysis
- `results/figures/topic_distribution.png` - Topic distribution
- `results/figures/emotion_distribution.png` - Emotion distribution

## Expected Runtime

| Phase | Approximate Time | Notes |
|-------|-----------------|-------|
| Phase 1 | 5-10 minutes | Fast, I/O bound |
| Phase 2 | 1-3 hours | Depends on API rate limits and number of users |
| Phase 3 | 4-24 hours | Highly dependent on dataset size and hardware (GPU recommended) |
| Phase 4 | 5-15 minutes | Fast statistical analysis |

## Troubleshooting

### OpenAI API Errors
- Check that `.env` file has correct `OPENAI_API_KEY`
- Remove quotes if present around the API key
- Verify API key has sufficient credits

### Out of Memory Errors
- Reduce batch sizes in analysis scripts
- Process data in chunks
- Use CPU instead of GPU (slower but uses less memory)

### Missing Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### GPU/CUDA Issues
The pipeline will automatically fall back to CPU if GPU is unavailable. To force CPU:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

## Output Files Overview

### Data Files
- `data/processed/all_comments.parquet` - Cleaned comments (269,040 comments, 43,166 users)
- `data/features/demographics.parquet` - User demographics
- `data/features/full_merged_dataset.parquet` - Complete feature set for analysis

### Results Files
- `results/tables/` - All statistical tables
- `results/figures/` - All publication figures
- `results/models/` - Saved models (if any)

## Next Steps After Execution

1. Review `results/tables/descriptive_statistics.txt` for data overview
2. Check `results/tables/regression_results.txt` for RQ2 findings
3. Examine figures in `results/figures/` for visualizations
4. Use `data/features/full_merged_dataset.parquet` for additional custom analyses

## Notes

- All scripts log to both console and log files (e.g., `phase1_data_collection.log`)
- Progress bars show real-time progress for long-running operations
- Checkpoints are saved after each phase for recovery
- The pipeline is designed to resume from any phase if previous phases completed

