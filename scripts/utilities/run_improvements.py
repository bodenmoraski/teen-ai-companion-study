#!/usr/bin/env python3
"""
Run all AnthroScore improvements on the full dataset.

1. Human calibration report (3 annotators: Stephanie, Boden, Afia)
2. Bot-attribution emotion detection on all comments
3. N-gram phrase features on all comments
4. Save enriched dataset

No API calls, no GPU — pure CPU/regex work.
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "results" / "improvements_run.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_calibration_report():
    """Phase 1: 3-annotator calibration analysis."""
    logger.info("=" * 70)
    logger.info("PHASE 1: HUMAN CALIBRATION REPORT (3 annotators)")
    logger.info("=" * 70)

    from src.anthroscore.human_calibration import (
        load_validation_data,
        run_calibration,
        print_calibration_report,
    )

    annotations = load_validation_data()
    calibration = run_calibration(annotations, n_examples=12)
    report = print_calibration_report(calibration)

    print("\n" + report + "\n")

    report_path = RESULTS_DIR / "human_calibration_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved calibration report to {report_path}")

    return calibration


def run_bot_attribution(df: pd.DataFrame, text_col: str = "body") -> pd.DataFrame:
    """Phase 2: Bot-attribution detection on all comments."""
    logger.info("=" * 70)
    logger.info("PHASE 2: BOT-ATTRIBUTION EMOTION DETECTION")
    logger.info(f"Processing {len(df):,} comments...")
    logger.info("=" * 70)

    from src.analysis.emotion_analysis import detect_bot_attribution

    start = time.time()
    results = []
    total = len(df)

    for i, text in enumerate(df[text_col].fillna("").astype(str)):
        results.append(detect_bot_attribution(text))
        if (i + 1) % 25000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            logger.info(
                f"  Progress: {i+1:,}/{total:,} ({100*(i+1)/total:.1f}%) | "
                f"Rate: {rate:.0f}/s | ETA: {eta:.0f}s"
            )

    elapsed = time.time() - start
    logger.info(f"Bot-attribution detection complete in {elapsed:.1f}s ({total/elapsed:.0f} comments/s)")

    attr_df = pd.DataFrame(results)
    for col in attr_df.columns:
        df[col] = attr_df[col].values

    bot_count = df["bot_attributed"].sum()
    self_count = df["self_expressed"].sum()
    mixed_count = (df["attribution_type"] == "mixed").sum()
    none_count = (df["attribution_type"] == "none").sum()

    logger.info(f"  Bot-attributed:  {bot_count:,} ({100*bot_count/total:.1f}%)")
    logger.info(f"  Self-expressed:  {self_count:,} ({100*self_count/total:.1f}%)")
    logger.info(f"  Mixed:           {mixed_count:,} ({100*mixed_count/total:.1f}%)")
    logger.info(f"  None:            {none_count:,} ({100*none_count/total:.1f}%)")

    return df


def run_ngram_features(df: pd.DataFrame, text_col: str = "body") -> pd.DataFrame:
    """Phase 3: N-gram phrase features on all comments."""
    logger.info("=" * 70)
    logger.info("PHASE 3: N-GRAM PHRASE FEATURES")
    logger.info(f"Processing {len(df):,} comments...")
    logger.info("=" * 70)

    from src.anthroscore.ngram_features import get_ngram_features

    start = time.time()
    results = []
    total = len(df)

    for i, text in enumerate(df[text_col].fillna("").astype(str)):
        results.append(get_ngram_features(text))
        if (i + 1) % 25000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            logger.info(
                f"  Progress: {i+1:,}/{total:,} ({100*(i+1)/total:.1f}%) | "
                f"Rate: {rate:.0f}/s | ETA: {eta:.0f}s"
            )

    elapsed = time.time() - start
    logger.info(f"N-gram feature extraction complete in {elapsed:.1f}s ({total/elapsed:.0f} comments/s)")

    ngram_df = pd.DataFrame(results)
    for col in ngram_df.columns:
        df[col] = ngram_df[col].values

    anthro_any = (df["ngram_anthro_total"] > 0).sum()
    deanthro_any = (df["ngram_deanthro_total"] > 0).sum()
    mean_signal = df["ngram_net_signal"].mean()

    logger.info(f"  Comments with anthro n-grams:    {anthro_any:,} ({100*anthro_any/total:.1f}%)")
    logger.info(f"  Comments with de-anthro n-grams: {deanthro_any:,} ({100*deanthro_any/total:.1f}%)")
    logger.info(f"  Mean net signal:                 {mean_signal:+.3f}")

    for cat in ["relationship", "emotion_attr", "agency", "consciousness", "pronoun_verb"]:
        col = f"ngram_{cat}"
        if col in df.columns:
            count = int((df[col] > 0).sum())
            if count > 0:
                logger.info(f"    {cat}: {count:,} comments")

    return df


def main():
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info("ANTHROSCORE IMPROVEMENTS - FULL DATASET RUN")
    logger.info("=" * 70)

    # --- Phase 1: Calibration report ---
    calibration = run_calibration_report()

    # --- Load the full dataset ---
    data_path = PROJECT_ROOT / "Data" / "processed" / "all_comments.parquet"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"\nLoading dataset from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} comments, columns: {list(df.columns)}")

    text_col = "body" if "body" in df.columns else df.columns[0]

    # --- Phase 2: Bot-attribution ---
    df = run_bot_attribution(df, text_col)

    # --- Phase 3: N-gram features ---
    df = run_ngram_features(df, text_col)

    # --- Save enriched dataset ---
    output_path = PROJECT_ROOT / "Data" / "features" / "comments_enriched.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved enriched dataset ({len(df):,} rows, {len(df.columns)} cols) to {output_path}")

    # --- Summary stats ---
    summary_path = RESULTS_DIR / "improvements_summary.txt"
    with open(summary_path, "w") as f:
        f.write("ANTHROSCORE IMPROVEMENTS - FULL DATASET SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total comments processed: {len(df):,}\n")
        f.write(f"New columns added: bot_attributed, self_expressed, bot_attribution_score, "
                f"attribution_type, ngram_anthro_total, ngram_deanthro_total, "
                f"ngram_anthro_density, ngram_deanthro_density, ngram_net_signal, "
                f"ngram_relationship, ngram_emotion_attr, ngram_agency, "
                f"ngram_consciousness, ngram_pronoun_verb, "
                f"ngram_technical, ngram_tool_framing\n\n")

        f.write("BOT ATTRIBUTION:\n")
        f.write(f"  Bot-attributed: {df['bot_attributed'].sum():,} "
                f"({100*df['bot_attributed'].mean():.1f}%)\n")
        f.write(f"  Self-expressed: {df['self_expressed'].sum():,} "
                f"({100*df['self_expressed'].mean():.1f}%)\n")
        f.write(f"  Mean attribution score: {df['bot_attribution_score'].mean():.3f}\n\n")

        f.write("N-GRAM FEATURES:\n")
        f.write(f"  Comments with anthro n-grams: {(df['ngram_anthro_total'] > 0).sum():,}\n")
        f.write(f"  Comments with de-anthro n-grams: {(df['ngram_deanthro_total'] > 0).sum():,}\n")
        f.write(f"  Mean net signal: {df['ngram_net_signal'].mean():+.3f}\n")
        f.write(f"  Mean anthro density: {df['ngram_anthro_density'].mean():.4f}\n\n")

        f.write(f"Output: {output_path}\n")
        f.write(f"Calibration report: {RESULTS_DIR / 'human_calibration_report.txt'}\n")

    elapsed = time.time() - overall_start
    logger.info(f"\n{'=' * 70}")
    logger.info(f"ALL PHASES COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Results: {summary_path}")
    logger.info(f"Enriched data: {output_path}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
