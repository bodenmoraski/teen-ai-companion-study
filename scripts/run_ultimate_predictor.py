#!/usr/bin/env python3
"""
Runner script for the Ultimate Age Predictor.

This script orchestrates the full pipeline:
1. Load and prepare data
2. Extract multi-signal features
3. Train stacked ensemble
4. Evaluate on test set (self-declared users)
5. Generate predictions for all users
6. Save results and model

Usage:
    python scripts/run_ultimate_predictor.py

Output:
    - Data/features/ultimate_predictor/ultimate_predictions.parquet
    - Data/features/ultimate_predictor/model/ultimate_predictor.pkl
    - results/ultimate_predictor_report.txt
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

# Setup logging
log_path = Path(__file__).parent.parent / f"ultimate_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path)
    ]
)
logger = logging.getLogger(__name__)

from src.demographics.ultimate_age_predictor import (
    UltimateAgePredictor,
    map_to_3_bucket,
    AGE_BUCKETS_3
)


def evaluate_on_holdout(
    predictor: UltimateAgePredictor,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    test_users: list,
    author_col: str = "author",
    label_col: str = "age_bucket_self_declared"
) -> dict:
    """
    Evaluate the predictor on held-out test set.
    """
    # Filter to test users
    test_features = features_df[features_df[author_col].isin(test_users)].copy()
    test_labels = labels_df[labels_df[author_col].isin(test_users)].copy()
    
    if len(test_features) == 0:
        logger.warning("No test users found in features!")
        return {}
    
    # Get predictions
    predictions = predictor.predict(test_features, confidence_threshold=0.0)
    
    # Merge with true labels
    eval_df = predictions.merge(
        test_labels[[author_col, label_col]],
        on=author_col,
        how="inner"
    )
    
    # Convert true labels to 3-bucket if using 3-bucket
    if predictor.use_3_bucket:
        eval_df["true_label"] = eval_df[label_col].apply(map_to_3_bucket)
    else:
        eval_df["true_label"] = eval_df[label_col]
    
    # Calculate metrics
    y_true = eval_df["true_label"]
    y_pred = eval_df["age_bucket_predicted"]
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Metrics at different confidence thresholds
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8]
    threshold_results = {}
    
    for thresh in thresholds:
        mask = eval_df["confidence"] >= thresh
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            coverage = mask.sum() / len(eval_df)
            threshold_results[thresh] = {
                "accuracy": acc,
                "coverage": coverage,
                "n_users": mask.sum()
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=predictor.label_encoder.classes_)
    
    return {
        "overall_accuracy": accuracy,
        "n_test": len(eval_df),
        "threshold_results": threshold_results,
        "confusion_matrix": cm,
        "class_labels": list(predictor.label_encoder.classes_),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }


def generate_report(train_results: dict, eval_results: dict, output_path: Path):
    """Generate a text report of the results."""
    
    lines = [
        "=" * 70,
        "ULTIMATE AGE PREDICTOR - RESULTS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "TRAINING SUMMARY",
        "-" * 40,
        f"Training samples: {train_results['n_train']}",
        f"Features used:",
        f"  - Text embedding features: {train_results['feature_counts']['text']}",
        f"  - Subreddit features: {train_results['feature_counts']['subreddit']}",
        f"  - Behavioral features: {train_results['feature_counts']['behavioral']}",
        "",
        "INDIVIDUAL SIGNAL PERFORMANCE (CV)",
        "-" * 40,
    ]
    
    for signal, score in train_results["signal_scores"].items():
        lines.append(f"  {signal}: {score:.1%}")
    
    lines.extend([
        "",
        "TEST SET EVALUATION",
        "-" * 40,
        f"Test samples: {eval_results['n_test']}",
        f"Overall accuracy: {eval_results['overall_accuracy']:.1%}",
        "",
        "ACCURACY BY CONFIDENCE THRESHOLD",
        "-" * 40,
    ])
    
    for thresh, results in eval_results["threshold_results"].items():
        lines.append(
            f"  ≥{thresh:.0%} confidence: "
            f"{results['accuracy']:.1%} accuracy, "
            f"{results['coverage']:.1%} coverage ({results['n_users']} users)"
        )
    
    lines.extend([
        "",
        "CONFUSION MATRIX",
        "-" * 40,
        f"Labels: {eval_results['class_labels']}",
    ])
    
    cm = eval_results["confusion_matrix"]
    for i, row in enumerate(cm):
        lines.append(f"  {eval_results['class_labels'][i]}: {row.tolist()}")
    
    lines.extend([
        "",
        "PER-CLASS METRICS",
        "-" * 40,
    ])
    
    for cls, metrics in eval_results["classification_report"].items():
        if isinstance(metrics, dict) and "precision" in metrics:
            lines.append(
                f"  {cls}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1-score']:.2f}"
            )
    
    lines.extend([
        "",
        "=" * 70,
        "KEY TAKEAWAYS",
        "=" * 70,
        "",
    ])
    
    # Determine key takeaways
    best_thresh_acc = max(
        r["accuracy"] for r in eval_results["threshold_results"].values()
        if r["coverage"] >= 0.3  # At least 30% coverage
    )
    
    if best_thresh_acc >= 0.7:
        lines.append("✅ EXCELLENT: Achieving >70% accuracy at reasonable coverage!")
        lines.append("   This is a significant improvement over the previous 46% baseline.")
    elif best_thresh_acc >= 0.6:
        lines.append("✓ GOOD: Achieving 60-70% accuracy.")
        lines.append("   Use high-confidence predictions for best results.")
    elif best_thresh_acc >= 0.5:
        lines.append("⚠ MODERATE: Achieving 50-60% accuracy.")
        lines.append("   Better than random, but use with caution.")
    else:
        lines.append("❌ NEEDS IMPROVEMENT: Below 50% accuracy.")
        lines.append("   Consider adding more training data or features.")
    
    lines.append("")
    lines.append("RECOMMENDED USAGE:")
    
    # Find best threshold for balance
    best_balance = None
    best_score = 0
    for thresh, results in eval_results["threshold_results"].items():
        # Balance accuracy and coverage (F1-like)
        score = 2 * results["accuracy"] * results["coverage"] / (results["accuracy"] + results["coverage"])
        if score > best_score:
            best_score = score
            best_balance = (thresh, results)
    
    if best_balance:
        lines.append(
            f"  Use confidence threshold ≥{best_balance[0]:.0%} for optimal balance:"
        )
        lines.append(
            f"  → {best_balance[1]['accuracy']:.1%} accuracy, "
            f"{best_balance[1]['coverage']:.1%} coverage"
        )
    
    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Report saved to {output_path}")
    
    # Also print to console
    print("\n".join(lines))


def main():
    """Main function to run the ultimate predictor."""
    
    logger.info("=" * 70)
    logger.info("STARTING ULTIMATE AGE PREDICTOR PIPELINE")
    logger.info("=" * 70)
    
    base_path = Path(__file__).parent.parent
    
    # Paths
    comments_path = base_path / "Data/processed/all_comments.parquet"
    user_subreddits_path = base_path / "Data/features/user_subreddit_interactions.parquet"
    self_declarations_path = base_path / "Data/features/self_declarations.parquet"
    output_path = base_path / "Data/features/ultimate_predictor"
    results_path = base_path / "results/ultimate_predictor_report.txt"
    
    # Check if files exist
    for path, name in [
        (comments_path, "comments"),
        (user_subreddits_path, "user_subreddits"),
        (self_declarations_path, "self_declarations")
    ]:
        if not path.exists():
            logger.error(f"Missing required file: {path}")
            logger.info(f"Please ensure {name} data is available.")
            return
    
    # Load data
    logger.info("\n[Step 1] Loading data...")
    comments_df = pd.read_parquet(comments_path)
    user_subreddits_df = pd.read_parquet(user_subreddits_path)
    self_decl_df = pd.read_parquet(self_declarations_path)
    
    logger.info(f"  Comments: {len(comments_df)} rows")
    logger.info(f"  User subreddits: {len(user_subreddits_df)} users")
    
    # Get users with known age
    known_age_users = self_decl_df[self_decl_df["age_bucket_self_declared"].notna()]["author"].tolist()
    logger.info(f"  Users with self-declared age: {len(known_age_users)}")
    
    if len(known_age_users) < 100:
        logger.error(f"Not enough labeled data: {len(known_age_users)} users (need at least 100)")
        return
    
    # Split into train/test
    logger.info("\n[Step 2] Splitting train/test (80/20)...")
    train_users, test_users = train_test_split(
        known_age_users, 
        test_size=0.2, 
        random_state=42,
        stratify=self_decl_df[self_decl_df["author"].isin(known_age_users)]["age_bucket_self_declared"].apply(map_to_3_bucket)
    )
    logger.info(f"  Train users: {len(train_users)}")
    logger.info(f"  Test users: {len(test_users)}")
    
    # Initialize predictor
    logger.info("\n[Step 3] Initializing Ultimate Age Predictor...")
    predictor = UltimateAgePredictor(use_3_bucket=True)
    
    # Extract features for all users
    logger.info("\n[Step 4] Extracting features...")
    features_df = predictor.extract_all_features(comments_df, user_subreddits_df)
    
    # Train only on train users
    logger.info("\n[Step 5] Training on train set...")
    train_labels = self_decl_df[self_decl_df["author"].isin(train_users)]
    train_results = predictor.fit(features_df, train_labels)
    
    # Evaluate on test set
    logger.info("\n[Step 6] Evaluating on test set...")
    eval_results = evaluate_on_holdout(
        predictor, features_df, self_decl_df, test_users
    )
    
    logger.info(f"  Test accuracy: {eval_results['overall_accuracy']:.1%}")
    
    # Generate predictions for all users
    logger.info("\n[Step 7] Generating predictions for all users...")
    all_predictions = predictor.predict(features_df, confidence_threshold=0.0)
    
    # Save results
    logger.info("\n[Step 8] Saving results...")
    output_path.mkdir(parents=True, exist_ok=True)
    all_predictions.to_parquet(output_path / "ultimate_predictions.parquet")
    predictor.save(output_path / "model")
    
    # Generate report
    results_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(train_results, eval_results, results_path)
    
    # Summary statistics
    high_conf = all_predictions[all_predictions["confidence"] >= 0.6]
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total users processed: {len(all_predictions)}")
    logger.info(f"High confidence predictions (≥60%): {len(high_conf)} ({len(high_conf)/len(all_predictions):.1%})")
    logger.info(f"\nTest set accuracy: {eval_results['overall_accuracy']:.1%}")
    logger.info(f"Best accuracy at ≥70% confidence: {eval_results['threshold_results'].get(0.7, {}).get('accuracy', 'N/A')}")
    
    # Compare to baseline
    logger.info("\n" + "-" * 40)
    logger.info("COMPARISON TO PREVIOUS BASELINE (46.3%)")
    logger.info("-" * 40)
    improvement = eval_results['overall_accuracy'] - 0.463
    if improvement > 0:
        logger.info(f"✅ IMPROVEMENT: +{improvement:.1%} over baseline!")
    else:
        logger.info(f"❌ No improvement over baseline ({improvement:.1%})")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Report saved to: {results_path}")
    logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()

