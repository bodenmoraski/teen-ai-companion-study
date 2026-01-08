#!/usr/bin/env python3
"""
Runner script for the Ultimate Gender Predictor.

This script orchestrates the full pipeline:
1. Load and prepare data
2. Extract multi-signal features (reuse from age predictor if available)
3. Train stacked ensemble
4. Evaluate on test set (self-declared users)
5. Generate predictions for all users
6. Save results and model

Usage:
    python scripts/run_ultimate_gender_predictor.py

Output:
    - Data/features/ultimate_predictor/gender_predictions.parquet
    - Data/features/ultimate_predictor/gender_model/
    - results/ultimate_gender_predictor_report.txt
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.demographics.ultimate_gender_predictor import UltimateGenderPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ultimate_gender_predictor.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def load_data(data_path: Path):
    """Load all required data."""
    logger.info("Loading data...")
    
    # Load comments
    comments_df = pd.read_parquet(data_path / "processed/all_comments.parquet")
    logger.info(f"  Comments: {len(comments_df)} rows")
    
    # Load user subreddits
    user_subreddits_df = pd.read_parquet(data_path / "features/user_subreddit_interactions.parquet")
    logger.info(f"  User subreddits: {len(user_subreddits_df)} rows")
    
    # Load self-declarations
    self_decl_df = pd.read_parquet(data_path / "features/self_declarations.parquet")
    gender_users = self_decl_df[self_decl_df["gender_self_declared"].notna()]
    logger.info(f"  Users with self-declared gender: {len(gender_users)}")
    
    return comments_df, user_subreddits_df, self_decl_df


def evaluate_on_test_set(predictor, features_df, test_labels, label_col="gender_self_declared"):
    """Evaluate predictor on test set."""
    # Get predictions for test users
    test_features = features_df[features_df["author"].isin(test_labels["author"])]
    predictions = predictor.predict(test_features)
    
    # Merge with true labels
    eval_df = predictions.merge(
        test_labels[["author", label_col]],
        on="author",
        how="inner"
    )
    
    # Filter to binary classes if needed
    if predictor.binary_mode:
        eval_df = eval_df[eval_df[label_col].isin(["male", "female"])]
    
    # Calculate metrics
    y_true = eval_df[label_col]
    y_pred = eval_df["gender_predicted"]
    
    accuracy = accuracy_score(y_true, y_pred)
    
    results = {
        "accuracy": accuracy,
        "n_samples": len(eval_df),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=predictor.label_encoder.classes_)
    }
    
    # Accuracy by confidence threshold
    confidence_results = {}
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        subset = eval_df[eval_df["confidence"] >= thresh]
        if len(subset) > 0:
            acc = accuracy_score(subset[label_col], subset["gender_predicted"])
            confidence_results[thresh] = {"accuracy": acc, "n_samples": len(subset)}
        else:
            confidence_results[thresh] = {"accuracy": 0, "n_samples": 0}
    results["by_confidence"] = confidence_results
    
    return results


def generate_report(train_results, eval_results, output_path):
    """Generate a text report of results."""
    lines = [
        "=" * 70,
        "ULTIMATE GENDER PREDICTOR - RESULTS REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "TRAINING RESULTS (Cross-Validation):",
        "-" * 40,
    ]
    
    for signal, acc in train_results.items():
        lines.append(f"  {signal}: {acc*100:.1f}%")
    
    lines.extend([
        "",
        "TEST SET EVALUATION:",
        "-" * 40,
        f"  Overall Accuracy: {eval_results['accuracy']*100:.1f}%",
        f"  Test Set Size: {eval_results['n_samples']}",
        "",
        "ACCURACY BY CONFIDENCE THRESHOLD:",
        "-" * 40,
    ])
    
    for thresh, result in eval_results["by_confidence"].items():
        lines.append(f"  Confidence >= {thresh}: {result['accuracy']*100:.1f}% (n={result['n_samples']})")
    
    lines.extend([
        "",
        "CLASSIFICATION REPORT:",
        "-" * 40,
        eval_results["classification_report"],
        "",
        "CONFUSION MATRIX:",
        "-" * 40,
        str(eval_results["confusion_matrix"]),
    ])
    
    report = "\n".join(lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")
    return report


def main():
    logger.info("=" * 70)
    logger.info("STARTING ULTIMATE GENDER PREDICTOR PIPELINE")
    logger.info("=" * 70)
    
    # Paths
    data_path = Path("Data")
    output_path = data_path / "features/ultimate_predictor"
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n[Step 1] Loading data...")
    comments_df, user_subreddits_df, self_decl_df = load_data(data_path)
    
    # Get users with known gender (binary only for now)
    known_gender = self_decl_df[
        self_decl_df["gender_self_declared"].isin(["male", "female"])
    ][["author", "gender_self_declared"]]
    logger.info(f"  Users with binary gender: {len(known_gender)}")
    
    # Split into train/test
    logger.info("\n[Step 2] Splitting train/test (80/20)...")
    train_labels, test_labels = train_test_split(
        known_gender,
        test_size=0.2,
        stratify=known_gender["gender_self_declared"],
        random_state=42
    )
    logger.info(f"  Train users: {len(train_labels)}")
    logger.info(f"  Test users: {len(test_labels)}")
    
    # Initialize predictor
    logger.info("\n[Step 3] Initializing Ultimate Gender Predictor...")
    predictor = UltimateGenderPredictor(binary_mode=True)
    
    # Check if we can reuse features from age predictor
    age_features_path = output_path / "all_features.parquet"
    if age_features_path.exists():
        logger.info("\n[Step 4] Loading cached features from age predictor...")
        features_df = pd.read_parquet(age_features_path)
        # Need to add linguistic features
        logger.info("  Adding linguistic marker features...")
        ling_features = predictor.signal4.extract_features(comments_df)
        features_df = features_df.merge(ling_features, on="author", how="left").fillna(0)
    else:
        logger.info("\n[Step 4] Extracting features...")
        features_df = predictor.extract_all_features(comments_df, user_subreddits_df)
        # Save features for reuse
        features_df.to_parquet(age_features_path)
        logger.info(f"  Features saved to {age_features_path}")
    
    # Train
    logger.info("\n[Step 5] Training on train set...")
    predictor.fit(features_df, train_labels)
    train_results = predictor.cv_results
    
    # Evaluate
    logger.info("\n[Step 6] Evaluating on test set...")
    eval_results = evaluate_on_test_set(predictor, features_df, test_labels)
    logger.info(f"  Test accuracy: {eval_results['accuracy']*100:.1f}%")
    
    # Print accuracy by confidence
    logger.info("\n  Accuracy by confidence threshold:")
    for thresh, result in eval_results["by_confidence"].items():
        logger.info(f"    >= {thresh}: {result['accuracy']*100:.1f}% (n={result['n_samples']})")
    
    # Generate predictions for all users
    logger.info("\n[Step 7] Generating predictions for all users...")
    all_predictions = predictor.predict(features_df)
    all_predictions.to_parquet(output_path / "gender_predictions.parquet")
    logger.info(f"  Predictions saved: {len(all_predictions)} users")
    
    # Save model
    logger.info("\n[Step 8] Saving model...")
    predictor.save(output_path / "gender_model")
    
    # Generate report
    logger.info("\n[Step 9] Generating report...")
    report = generate_report(train_results, eval_results, results_path / "ultimate_gender_predictor_report.txt")
    print("\n" + report)
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

