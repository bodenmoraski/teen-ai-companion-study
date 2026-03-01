# Setting Up This Repo (Git LFS Data Sync)

This project uses **Git LFS** (Large File Storage) to track essential data files (~122 MB of parquet and xlsx files). Without LFS, you'll get tiny pointer files instead of the actual data.

## Quick Setup (New Machine / Fresh Clone)

```bash
# 1. Install Git LFS (one-time per machine)
git lfs install

# 2. Clone the repo (LFS files download automatically)
git clone https://github.com/bodenmoraski/teen-ai-companion-study.git
cd teen-ai-companion-study

# 3. Verify data files are real (not pointer files)
python -c "import pandas as pd; df = pd.read_parquet('Data/processed/all_comments.parquet'); print(f'Loaded {len(df):,} comments - LFS working!')"
```

## If You Already Cloned Without LFS

```bash
# Install LFS then pull the actual file contents
git lfs install
git lfs pull
```

## What Gets Downloaded via LFS

These are the essential data files tracked by LFS (~122 MB total):

| File | Size | What It Is |
|------|------|------------|
| `Data/processed/all_comments.parquet` | 31 MB | 283,895 Reddit comments (text, author, subreddit) |
| `experiments/anthroscore_v3/anthroscore_v3_full.parquet` | 23 MB | AnthroScore V3 scores for all comments |
| `experiments/anthroscore_v3/test_set_expert_labeled.parquet` | 0.04 MB | Expert-labeled validation set (100 comments) |
| `experiments/v2_correction/age_predictions_v4.parquet` | 2 MB | Age predictions (teen/adult) per user |
| `experiments/v2_correction/gender_predictions_v4.parquet` | 2 MB | Gender predictions per user |
| `Data/features/user_emotions.parquet` | 3.5 MB | User-level emotion scores (7 emotions) |
| `Data/features/user_anthroscores.parquet` | 1.5 MB | User-level AnthroScore V2 (old, for comparison) |
| `Data/features/self_declarations.parquet` | 0.7 MB | Self-declared ages (ground truth) |
| `Data/features/llm_classifications.parquet` | 0.8 MB | LLM demographic classifications |
| `Data/features/full_merged_dataset.parquet` | 2.2 MB | Pre-merged analysis dataset |
| `Data/features/comments_with_emotions.parquet` | 45 MB | Comment-level emotion probabilities |
| `experiments/anthroscore_v3/HUMAN_VALIDATION_BLIND.xlsx` | small | Blind annotation sheet (150 comments) |
| `experiments/anthroscore_v3/HUMAN_VALIDATION_ANSWER_KEY.xlsx` | small | Answer key with true V3 scores |

## What's NOT in the Repo (Too Large / Regenerable)

| Excluded | Size | How to Get It |
|----------|------|---------------|
| Raw Reddit JSONL files | ~1.4 GB | Re-download from [Arctic Shift](https://arctic-shift.photon-reddit.com/) |
| `all_features_v4.parquet` | 107 MB | Re-run feature engineering pipeline |
| `ultimate_predictor/` | 107 MB | Re-run ultimate predictor (superseded) |
| Subreddit-specific parquets | ~31 MB | Filter from `all_comments.parquet` by subreddit column |
| Older demographic predictions (v2/v3/v5) | ~8 MB | Superseded by v4 |

## Python Dependencies

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn openpyxl pyarrow
# For emotion analysis (optional, results already in parquet):
pip install transformers torch sentence-transformers
# For LLM scoring (optional, results already in parquet):
pip install openai
```

## Running the Analysis Scripts

Once LFS data is synced, these scripts run directly:

```bash
# Full statistical analysis (RQ1-RQ3, regression, robustness)
python scripts/COMPREHENSIVE_V3_ANALYSIS.py

# Extended analysis (floor effects, binary analysis, variance tests)
python scripts/EXTENDED_ANALYSIS.py

# Deep dive (loneliness, content patterns, linguistic features)
python scripts/DEEP_DIVE_ANALYSIS.py

# Regenerate human validation spreadsheets
python scripts/generate_human_validation_sheets.py
```

## Verifying Everything Works

```bash
python -c "
import pandas as pd
files = {
    'Comments': 'Data/processed/all_comments.parquet',
    'V3 Scores': 'experiments/anthroscore_v3/anthroscore_v3_full.parquet',
    'Age Preds': 'experiments/v2_correction/age_predictions_v4.parquet',
    'Gender Preds': 'experiments/v2_correction/gender_predictions_v4.parquet',
    'Emotions': 'Data/features/user_emotions.parquet',
}
for name, path in files.items():
    df = pd.read_parquet(path)
    print(f'{name}: {len(df):,} rows OK')
print('All essential data loaded successfully!')
"
```

## Project Structure (Key Paths)

```
├── Data/
│   ├── processed/all_comments.parquet     ← Core comment data
│   ├── features/                          ← Derived features (emotions, demographics)
│   └── annotations/                       ← Human annotation materials
├── experiments/
│   ├── anthroscore_v3/                    ← V3 scores, validation, human validation sheets
│   └── v2_correction/                     ← Demographic predictions
├── scripts/                               ← Analysis scripts (run these)
├── src/                                   ← Source modules (imported by scripts)
├── results/                               ← Output: JSON results, figures, reports
├── FINAL_SHORT_SUMMARY.md                 ← Executive summary of findings
├── COMPREHENSIVE_V3_ANALYSIS_RESULTS.md   ← Full statistical report
└── METHODOLOGY_FINAL.md                   ← Detailed methodology documentation
```
