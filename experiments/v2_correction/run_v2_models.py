"""
V2 MODEL TRAINING AND VALIDATION RUNNER

This script:
1. Trains Age Predictor V2 (validity-first, behavioral-only)
2. Trains Gender Predictor V2 (high female recall)
3. Validates both against ground truth
4. Compares to V1 baselines
5. Generates comprehensive reports

Run from project root:
    python experiments/v2_correction/run_v2_models.py

Author: Research Agent V2
Created: 2026-01-10
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    recall_score, precision_score, f1_score
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.v2_correction.age_predictor_v2 import AgePredictor_V2, map_to_binary
from experiments.v2_correction.gender_predictor_v2 import GenderPredictor_V2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'experiments/v2_correction/v2_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load all necessary data files."""
    logger.info("Loading data...")
    
    data = {
        'comments': pd.read_parquet(project_root / 'Data/processed/all_comments.parquet'),
        'user_subreddits': pd.read_parquet(project_root / 'Data/features/user_subreddit_interactions.parquet'),
        'self_declarations': pd.read_parquet(project_root / 'Data/features/self_declarations.parquet'),
        'anthroscores': pd.read_parquet(project_root / 'Data/features/user_anthroscores.parquet')
    }
    
    # Try to load existing features
    features_path = project_root / 'Data/features/ultimate_predictor/all_features.parquet'
    if features_path.exists():
        data['existing_features'] = pd.read_parquet(features_path)
        logger.info(f"  Loaded existing features: {data['existing_features'].shape}")
    
    logger.info(f"  Comments: {len(data['comments'])} rows")
    logger.info(f"  Self-declarations: {len(data['self_declarations'])} users")
    logger.info(f"  AnthroScores: {len(data['anthroscores'])} users")
    
    return data


def train_age_predictor_v2(data: dict) -> dict:
    """Train and validate Age Predictor V2."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: AGE PREDICTOR V2")
    logger.info("="*70)
    
    # Get users with known age
    self_decl = data['self_declarations']
    age_labeled = self_decl[self_decl['age_bucket_self_declared'].notna()].copy()
    
    # Map to binary for training
    age_labeled['age_binary'] = age_labeled['age_bucket_self_declared'].apply(map_to_binary)
    
    # Split into train/test
    train_users, test_users = train_test_split(
        age_labeled['author'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=age_labeled['age_binary']
    )
    
    logger.info(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
    
    # Initialize and train
    predictor = AgePredictor_V2(use_binary=True)
    
    # Extract features
    logger.info("Extracting features...")
    features = predictor.extract_features(
        data['comments'],
        data['user_subreddits']
    )
    
    # Train on train set only
    train_labels = age_labeled[age_labeled['author'].isin(train_users)]
    
    logger.info("Training model...")
    train_results = predictor.fit(
        features,
        train_labels,
        anthroscore_df=data['anthroscores']
    )
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_features = features[features['author'].isin(test_users)]
    test_labels = age_labeled[age_labeled['author'].isin(test_users)]
    
    predictions = predictor.predict(test_features)
    
    # Merge with true labels
    eval_df = predictions.merge(
        test_labels[['author', 'age_binary']],
        on='author'
    )
    
    accuracy = accuracy_score(eval_df['age_binary'], eval_df['age_predicted'])
    
    logger.info(f"\nTest Set Results:")
    logger.info(f"  Accuracy: {accuracy:.1%}")
    logger.info(f"\n{classification_report(eval_df['age_binary'], eval_df['age_predicted'])}")
    
    # CRITICAL: Validate ground truth direction
    logger.info("\n" + "-"*50)
    logger.info("CRITICAL VALIDATION: Ground Truth Direction")
    logger.info("-"*50)
    
    # Get predictions with anthroscores
    all_preds = predictor.predict(features)
    preds_anthro = all_preds.merge(
        data['anthroscores'][['author', 'anthroscore_max']],
        on='author'
    )
    preds_anthro = preds_anthro[preds_anthro['anthroscore_max'] != 0]  # Non-zero only
    
    teens = preds_anthro[preds_anthro['age_predicted'] == 'teen']['anthroscore_max']
    adults = preds_anthro[preds_anthro['age_predicted'] == 'adult']['anthroscore_max']
    
    d = (teens.mean() - adults.mean()) / np.sqrt((teens.std()**2 + adults.std()**2) / 2)
    t, p = stats.ttest_ind(teens, adults)
    
    logger.info(f"Predicted Teen mean AnthroScore: {teens.mean():.4f} (n={len(teens)})")
    logger.info(f"Predicted Adult mean AnthroScore: {adults.mean():.4f} (n={len(adults)})")
    logger.info(f"Cohen's d: {d:.4f}")
    logger.info(f"p-value: {p:.6f}")
    
    ground_truth_valid = d < 0  # Adults should be HIGHER
    
    if ground_truth_valid:
        logger.info("✓ VALIDATION PASSED: Predicted adults anthropomorphize more")
        logger.info("  V2 model aligns with ground truth direction!")
    else:
        logger.warning("⚠ VALIDATION FAILED: Predicted teens anthropomorphize more")
        logger.warning("  V2 model does NOT align with ground truth!")
    
    # Save model
    save_path = project_root / 'experiments/v2_correction/models/age_v2'
    predictor.save(save_path)
    
    # Save predictions
    all_preds.to_parquet(project_root / 'experiments/v2_correction/age_predictions_v2.parquet')
    
    return {
        'accuracy': accuracy,
        'cohens_d': d,
        'p_value': p,
        'ground_truth_valid': ground_truth_valid,
        'cv_results': train_results['cv_results'],
        'n_train': len(train_users),
        'n_test': len(test_users)
    }


def train_gender_predictor_v2(data: dict) -> dict:
    """Train and validate Gender Predictor V2."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: GENDER PREDICTOR V2")
    logger.info("="*70)
    
    # Get users with known gender
    self_decl = data['self_declarations']
    gender_labeled = self_decl[
        self_decl['gender_self_declared'].isin(['male', 'female'])
    ].copy()
    
    # Split into train/test
    train_users, test_users = train_test_split(
        gender_labeled['author'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=gender_labeled['gender_self_declared']
    )
    
    logger.info(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
    
    # Check for imblearn
    try:
        from imblearn.over_sampling import SMOTE
        has_imblearn = True
        logger.info("imblearn available - will use SMOTE")
    except ImportError:
        has_imblearn = False
        logger.warning("imblearn not available - install with: pip install imbalanced-learn")
    
    # Initialize predictor
    predictor = GenderPredictor_V2(
        target_female_recall=0.70,
        use_smote=has_imblearn,
        use_cost_sensitive=True
    )
    
    # Extract features (reuse existing if available)
    logger.info("Extracting features...")
    existing_text = None
    if 'existing_features' in data:
        text_cols = [c for c in data['existing_features'].columns if c.startswith('text_emb_')]
        if text_cols:
            existing_text = data['existing_features'][['author'] + text_cols]
            logger.info(f"  Reusing {len(text_cols)} text embedding features")
    
    features = predictor.extract_features(
        data['comments'],
        data['user_subreddits'],
        existing_text_features=existing_text
    )
    
    # Train on train set only
    train_labels = gender_labeled[gender_labeled['author'].isin(train_users)]
    
    logger.info("Training model...")
    train_results = predictor.fit(features, train_labels)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_features = features[features['author'].isin(test_users)]
    test_labels = gender_labeled[gender_labeled['author'].isin(test_users)]
    
    predictions = predictor.predict(test_features, use_optimal_threshold=True)
    
    # Merge with true labels
    eval_df = predictions.merge(
        test_labels[['author', 'gender_self_declared']],
        on='author'
    )
    
    # Calculate metrics
    y_true = (eval_df['gender_self_declared'] == 'female').astype(int)
    y_pred = (eval_df['gender_predicted'] == 'female').astype(int)
    
    female_recall = recall_score(y_true, y_pred)
    female_precision = precision_score(y_true, y_pred)
    male_recall = recall_score(1 - y_true, 1 - y_pred)
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    logger.info(f"\nTest Set Results (at threshold {predictor.optimal_threshold:.3f}):")
    logger.info(f"  Female Recall:    {female_recall:.1%}")
    logger.info(f"  Female Precision: {female_precision:.1%}")
    logger.info(f"  Male Recall:      {male_recall:.1%}")
    logger.info(f"  Overall Accuracy: {overall_accuracy:.1%}")
    logger.info(f"  Macro F1:         {macro_f1:.1%}")
    
    logger.info(f"\n{classification_report(eval_df['gender_self_declared'], eval_df['gender_predicted'])}")
    
    # Compare to V1 baseline
    logger.info("\n" + "-"*50)
    logger.info("COMPARISON TO V1 BASELINE")
    logger.info("-"*50)
    v1_female_recall = 0.44  # From MASTER_RESEARCH_FINDINGS
    improvement = female_recall - v1_female_recall
    
    logger.info(f"V1 Female Recall: {v1_female_recall:.1%}")
    logger.info(f"V2 Female Recall: {female_recall:.1%}")
    logger.info(f"Improvement: {improvement:+.1%}")
    
    if improvement > 0:
        logger.info(f"✓ V2 IMPROVED female recall by {improvement:.1%}!")
    else:
        logger.warning(f"⚠ V2 did NOT improve female recall")
    
    # Save model
    save_path = project_root / 'experiments/v2_correction/models/gender_v2'
    predictor.save(save_path)
    
    # Save predictions
    all_preds = predictor.predict(features, use_optimal_threshold=True)
    all_preds.to_parquet(project_root / 'experiments/v2_correction/gender_predictions_v2.parquet')
    
    return {
        'female_recall': female_recall,
        'female_precision': female_precision,
        'male_recall': male_recall,
        'overall_accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'optimal_threshold': predictor.optimal_threshold,
        'v1_female_recall': v1_female_recall,
        'improvement': improvement,
        'cv_results': train_results['cv_results'],
        'n_train': len(train_users),
        'n_test': len(test_users)
    }


def generate_report(age_results: dict, gender_results: dict):
    """Generate comprehensive comparison report."""
    
    report_path = project_root / 'experiments/v2_correction/V2_RESULTS_REPORT.txt'
    
    lines = [
        "="*70,
        "V2 MODEL RESULTS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70,
        "",
        "EXECUTIVE SUMMARY",
        "-"*50,
        "",
    ]
    
    # Age summary
    if age_results['ground_truth_valid']:
        lines.append("[OK] AGE PREDICTOR V2: GROUND TRUTH VALIDATED")
        lines.append(f"  - Predicted adults anthropomorphize more (d = {age_results['cohens_d']:.4f})")
        lines.append(f"  - This matches ground truth direction!")
    else:
        lines.append("[!!] AGE PREDICTOR V2: GROUND TRUTH NOT VALIDATED")
        lines.append(f"  - Cohen's d = {age_results['cohens_d']:.4f}")
    
    lines.append(f"  - Test accuracy: {age_results['accuracy']:.1%}")
    lines.append("")
    
    # Gender summary
    if gender_results['improvement'] > 0:
        lines.append(f"[OK] GENDER PREDICTOR V2: FEMALE RECALL IMPROVED")
        lines.append(f"  - Female recall: {gender_results['v1_female_recall']:.1%} -> {gender_results['female_recall']:.1%}")
        lines.append(f"  - Improvement: {gender_results['improvement']:+.1%}")
    else:
        lines.append("[!!] GENDER PREDICTOR V2: NO IMPROVEMENT")
    
    lines.append(f"  - Male recall maintained at: {gender_results['male_recall']:.1%}")
    lines.append("")
    
    lines.extend([
        "",
        "="*70,
        "DETAILED RESULTS",
        "="*70,
        "",
        "AGE PREDICTOR V2",
        "-"*50,
        "",
        "Architecture Changes from V1:",
        "  - REMOVED: Text embeddings (avoided linguistic stereotype bias)",
        "  - KEPT: Behavioral features (time patterns, activity metrics)",
        "  - KEPT: Subreddit participation patterns",
        "  - Trained on golden set only (self-declared age users)",
        "",
        "Training Results:",
        f"  - Train size: {age_results['n_train']} users",
        f"  - Test size: {age_results['n_test']} users",
        f"  - Test accuracy: {age_results['accuracy']:.1%}",
        "",
        "Signal-Level CV Performance:",
    ])
    
    for signal, score in age_results['cv_results'].items():
        lines.append(f"  - {signal}: {score:.1%}")
    
    lines.extend([
        "",
        "Ground Truth Validation:",
        f"  - Cohen's d (teen - adult): {age_results['cohens_d']:.4f}",
        f"  - p-value: {age_results['p_value']:.6f}",
        f"  - Direction: {'ADULTS HIGHER (CORRECT)' if age_results['cohens_d'] < 0 else 'TEENS HIGHER (WRONG)'}",
        "",
        "",
        "GENDER PREDICTOR V2",
        "-"*50,
        "",
        "Architecture Changes from V1:",
        "  - ADDED: SMOTE oversampling for female class",
        "  - ADDED: Cost-sensitive learning (scale_pos_weight)",
        "  - ADDED: Threshold optimization for female recall",
        "  - ENHANCED: Linguistic feature extraction",
        "",
        "Training Results:",
        f"  - Train size: {gender_results['n_train']} users",
        f"  - Test size: {gender_results['n_test']} users",
        f"  - Optimal threshold: {gender_results['optimal_threshold']:.3f}",
        "",
        "Test Set Metrics:",
        f"  - Female Recall:    {gender_results['female_recall']:.1%} (V1: {gender_results['v1_female_recall']:.1%})",
        f"  - Female Precision: {gender_results['female_precision']:.1%}",
        f"  - Male Recall:      {gender_results['male_recall']:.1%}",
        f"  - Overall Accuracy: {gender_results['overall_accuracy']:.1%}",
        f"  - Macro F1:         {gender_results['macro_f1']:.1%}",
        "",
        "Signal-Level Female Recall:",
    ])
    
    for signal, recall in gender_results['cv_results'].items():
        lines.append(f"  - {signal}: {recall:.1%}")
    
    lines.extend([
        "",
        "",
        "="*70,
        "RECOMMENDATIONS",
        "="*70,
        "",
    ])
    
    if age_results['ground_truth_valid']:
        lines.append("1. AGE PREDICTOR V2: READY FOR USE")
        lines.append("   - Use V2 for all age-related analyses")
        lines.append("   - V2 correctly predicts adults anthropomorphize more")
        lines.append("   - This aligns with ground truth from self-declared users")
    else:
        lines.append("1. AGE PREDICTOR V2: NEEDS REFINEMENT")
        lines.append("   - Ground truth direction not achieved")
        lines.append("   - Consider further feature engineering or model tuning")
    
    lines.append("")
    
    if gender_results['improvement'] > 0.1:  # >10% improvement
        lines.append("2. GENDER PREDICTOR V2: SIGNIFICANT IMPROVEMENT")
        lines.append("   - Use V2 for all gender-related analyses")
        lines.append("   - Much better at identifying female users")
    elif gender_results['improvement'] > 0:
        lines.append("2. GENDER PREDICTOR V2: MODEST IMPROVEMENT")
        lines.append("   - Consider using V2, but validate on your specific use case")
    else:
        lines.append("2. GENDER PREDICTOR V2: NO IMPROVEMENT")
        lines.append("   - Stick with V1 or investigate further")
    
    lines.extend([
        "",
        "",
        "="*70,
        "FILES GENERATED",
        "="*70,
        "",
        "experiments/v2_correction/",
        "├── models/",
        "│   ├── age_v2/age_predictor_v2.pkl",
        "│   └── gender_v2/gender_predictor_v2.pkl",
        "├── age_predictions_v2.parquet",
        "├── gender_predictions_v2.parquet",
        "├── v2_training.log",
        "└── V2_RESULTS_REPORT.txt",
        "",
        "="*70,
    ])
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"\nReport saved to: {report_path}")
    
    # Also save JSON results
    results_json = {
        'age': age_results,
        'gender': gender_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert numpy types to Python types for JSON
    def convert_types(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj
    
    results_json = convert_types(results_json)
    
    with open(project_root / 'experiments/v2_correction/v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
    
    # Print summary to console
    print("\n" + "="*70)
    print("V2 MODELS - FINAL SUMMARY")
    print("="*70)
    print("\n".join(lines[7:30]))


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("V2 MODEL TRAINING AND VALIDATION")
    logger.info("="*70)
    
    # Create output directories
    (project_root / 'experiments/v2_correction/models/age_v2').mkdir(parents=True, exist_ok=True)
    (project_root / 'experiments/v2_correction/models/gender_v2').mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Train Age Predictor V2
    age_results = train_age_predictor_v2(data)
    
    # Train Gender Predictor V2
    gender_results = train_gender_predictor_v2(data)
    
    # Generate report
    generate_report(age_results, gender_results)
    
    logger.info("\n" + "="*70)
    logger.info("V2 TRAINING COMPLETE")
    logger.info("="*70)
    
    return age_results, gender_results


if __name__ == "__main__":
    main()
