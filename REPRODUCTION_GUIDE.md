# Reproduction Guide

This guide explains how to reproduce the results from the paper.

## Step 1: Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Step 2: Data Files

The following data files are required (included in repository):

### Core Data
- `Data/processed/all_comments.parquet` - 283,895 cleaned Reddit comments
- `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet` - AnthroIndex scores
- `experiments/v2_correction/age_predictions_v3.parquet` - Age predictions
- `experiments/v2_correction/gender_predictions_v3.parquet` - Gender predictions
- `Data/features/user_emotions.parquet` - Emotion scores

### Validation Data
- `Data/annotations/sample_for_annotation.parquet` - Human validation sample
- `Validations/*.csv` - Human annotator responses

## Step 3: Running Analyses

### Main Analysis (Paper Results)

```bash
python scripts/COMPREHENSIVE_V3_ANALYSIS.py
```

**Output:**
- `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` - Formatted report
- `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.json` - Machine-readable results

### Validation Script

```bash
python scripts/validate_paper_statistics.py
```

**Output:**
- `results/PAPER_VALIDATION_REPORT.md` - Comparison with paper claims

### Extended Analysis

```bash
python scripts/EXTENDED_ANALYSIS.py
```

**Output:**
- `results/extended_analysis/` - Robustness checks and visualizations

## Step 4: Understanding the Pipeline

### Data Collection Pipeline
1. `src/data_collection/arctic_shift.py` - Fetches Reddit data from Arctic Shift API
2. `src/data_collection/preprocess.py` - Cleans and standardizes data

### Demographics Classification Pipeline
1. `src/demographics/llm_classifier.py` - LLM-based classification (GPT-4o-mini)
2. `src/demographics/ensemble_classifier.py` - Combines multiple methods
3. Output: `experiments/v2_correction/*_predictions_v3.parquet`

### Anthropomorphization Scoring Pipeline
1. `experiments/anthroscore_v3/anthroscore_llm.py` - LLM-based scoring (GPT-4.1-nano)
2. Output: `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet`

### Emotion Analysis Pipeline
1. `src/analysis/emotion_analysis.py` - DistilRoBERTa emotion classifier
2. Output: `Data/features/user_emotions.parquet`

### Statistical Analysis Pipeline
1. `scripts/COMPREHENSIVE_V3_ANALYSIS.py` - Main analysis
2. `src/statistical/regression_models.py` - Regression models
3. Output: `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.*`

## Key Parameters

| Parameter | Value | File |
|-----------|-------|------|
| Confidence threshold | 0.60 | `scripts/COMPREHENSIVE_V3_ANALYSIS.py` |
| Bootstrap iterations | 10,000 | `scripts/COMPREHENSIVE_V3_ANALYSIS.py` |
| Alpha level | 0.05 | `scripts/COMPREHENSIVE_V3_ANALYSIS.py` |
| AnthroIndex scale | 1-5 | `experiments/anthroscore_v3/anthroscore_llm.py` |

## Version Clarification

The naming can be confusing - here's what each version means:

| Component | Version | File | Description |
|-----------|---------|------|-------------|
| AnthroScore | V3 | `anthroscore_v3_improved_final.parquet` | LLM scoring of anthropomorphization |
| Age Predictions | V4 | `age_predictions_v4.parquet` | Stacked ensemble demographics |
| Gender Predictions | V4 | `gender_predictions_v4.parquet` | Stacked ensemble demographics |

The script `COMPREHENSIVE_V3_ANALYSIS.py` uses **AnthroScore V3** + **Demographics V4**.

## Rerunning from Scratch

To regenerate all features from raw data:

```bash
# 1. Preprocess comments
python src/data_collection/preprocess.py

# 2. Classify demographics (requires OpenAI API)
python experiments/v2_correction/run_v3_models.py

# 3. Score anthropomorphization (requires OpenAI API)
python experiments/anthroscore_v3/run_full_dataset_optimized.py

# 4. Classify emotions
python src/analysis/emotion_analysis.py

# 5. Run analysis
python scripts/COMPREHENSIVE_V3_ANALYSIS.py
```

**Note:** Steps 2-3 require OpenAI API access and incur costs (~$10-20 total).

## Validation Metrics

### AnthroIndex Validation
- Expert correlation: r = 0.59
- Exact accuracy: 64%
- Within-1 accuracy: 96%
- Cohen's κ = 0.58

### Demographics Classifier Validation
- Age accuracy: 95.0% (at ≥0.60 confidence)
- Gender accuracy: 96.9% (at ≥0.60 confidence)

## Troubleshooting

### Missing parquet files
Ensure you have `pyarrow` installed: `pip install pyarrow`

### OpenAI API errors
Check your `.env` file has valid `OPENAI_API_KEY`

### Memory issues
The full dataset requires ~4GB RAM. Use chunked processing if needed.

### Inconsistent results
Some variation is expected due to:
- Bootstrap sampling
- LLM non-determinism (set temperature=0 for consistency)

## Contact

For reproduction issues, please open a GitHub issue.
