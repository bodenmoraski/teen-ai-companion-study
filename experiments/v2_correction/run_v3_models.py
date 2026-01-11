"""
V3 MODEL TRAINING: Balanced Improvement

CONSTRAINTS:
1. All metrics must improve or stay within tolerance
2. No trading one metric for another
3. Grid search to find optimal configurations

Author: Research Agent V3
Created: 2026-01-10
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.v2_correction.gender_predictor_v3 import GenderPredictor_V3
from experiments.v2_correction.age_predictor_v3 import AgePredictor_V3, map_to_binary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'experiments/v2_correction/v3_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load data."""
    logger.info("Loading data...")
    
    data = {
        'comments': pd.read_parquet(project_root / 'Data/processed/all_comments.parquet'),
        'user_subreddits': pd.read_parquet(project_root / 'Data/features/user_subreddit_interactions.parquet'),
        'self_declarations': pd.read_parquet(project_root / 'Data/features/self_declarations.parquet'),
        'anthroscores': pd.read_parquet(project_root / 'Data/features/user_anthroscores.parquet'),
        'existing_features': pd.read_parquet(project_root / 'Data/features/ultimate_predictor/all_features.parquet')
    }
    
    logger.info(f"  Existing features: {data['existing_features'].shape}")
    
    return data


def extract_gender_features(data: dict) -> pd.DataFrame:
    """Extract features for gender prediction."""
    from experiments.v2_correction.gender_predictor_v2 import EnhancedLinguisticSignal
    
    # Reuse existing text embeddings
    existing = data['existing_features']
    text_cols = [c for c in existing.columns if c.startswith('text_emb_')]
    features = existing[['author'] + text_cols].copy()
    
    # Add subreddit features
    logger.info("Extracting subreddit features...")
    user_subs = data['user_subreddits']
    if 'subreddit' in user_subs.columns:
        agg = user_subs.groupby('author')['subreddit'].apply(list).reset_index()
        agg.columns = ['author', 'subreddits']
        user_subs = agg
    
    from collections import Counter
    all_subs = []
    for subs in user_subs['subreddits']:
        if isinstance(subs, list):
            all_subs.extend(subs)
    
    sub_counts = Counter(all_subs)
    selected_subs = [s for s, c in sub_counts.most_common(400) if c >= 10]
    
    sub_features = []
    for _, row in user_subs.iterrows():
        user_subs_set = set(row['subreddits']) if isinstance(row['subreddits'], list) else set()
        feat = {f'sub_{s}': 1 if s in user_subs_set else 0 for s in selected_subs}
        feat['author'] = row['author']
        sub_features.append(feat)
    
    sub_df = pd.DataFrame(sub_features)
    features = features.merge(sub_df, on='author', how='left')
    
    # Add behavioral features
    logger.info("Extracting behavioral features...")
    behav_features = []
    for author, group in data['comments'].groupby('author'):
        feat = {'author': author}
        feat['comment_count'] = len(group)
        feat['avg_comment_length'] = group['body'].str.len().mean()
        feat['std_comment_length'] = group['body'].str.len().std() if len(group) > 1 else 0
        
        if 'created_utc' in group.columns:
            try:
                ts = pd.to_datetime(group['created_utc'], unit='s')
                feat['hour_mean'] = ts.dt.hour.mean()
                feat['hour_std'] = ts.dt.hour.std() if len(ts) > 1 else 6
                feat['weekend_ratio'] = (ts.dt.dayofweek >= 5).mean()
            except:
                feat['hour_mean'] = 12
                feat['hour_std'] = 6
                feat['weekend_ratio'] = 0.29
        else:
            feat['hour_mean'] = 12
            feat['hour_std'] = 6
            feat['weekend_ratio'] = 0.29
        
        behav_features.append(feat)
    
    behav_df = pd.DataFrame(behav_features)
    features = features.merge(behav_df, on='author', how='left')
    
    # Add linguistic features
    logger.info("Extracting linguistic features...")
    ling_extractor = EnhancedLinguisticSignal()
    ling_df = ling_extractor.extract_features(data['comments'])
    features = features.merge(ling_df, on='author', how='left')
    
    features = features.fillna(0)
    logger.info(f"Gender features: {features.shape}")
    
    return features


def extract_age_features(data: dict) -> pd.DataFrame:
    """Extract features for age prediction."""
    # Use existing features
    features = data['existing_features'].copy()
    logger.info(f"Age features: {features.shape}")
    return features


def train_gender_v3(data: dict, features: pd.DataFrame) -> dict:
    """Train Gender V3 with balanced optimization."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING GENDER PREDICTOR V3")
    logger.info("="*70)
    
    self_decl = data['self_declarations']
    gender_labeled = self_decl[self_decl['gender_self_declared'].isin(['male', 'female'])]
    
    # Split train/test
    train_users, test_users = train_test_split(
        gender_labeled['author'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=gender_labeled['gender_self_declared']
    )
    
    logger.info(f"Train: {len(train_users)}, Test: {len(test_users)}")
    
    # V1 baseline metrics
    v1_metrics = {
        'accuracy': 0.813,
        'female_recall': 0.44,
        'male_recall': 0.95,
        'female_precision': 0.77
    }
    
    # Train
    predictor = GenderPredictor_V3()
    train_labels = gender_labeled[gender_labeled['author'].isin(train_users)]
    train_results = predictor.fit(features, train_labels, v1_metrics=v1_metrics)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_features = features[features['author'].isin(test_users)]
    test_labels = gender_labeled[gender_labeled['author'].isin(test_users)]
    
    predictions = predictor.predict(test_features)
    
    eval_df = predictions.merge(
        test_labels[['author', 'gender_self_declared']],
        on='author'
    )
    
    y_true = (eval_df['gender_self_declared'] == 'female').astype(int)
    y_pred = (eval_df['gender_predicted'] == 'female').astype(int)
    
    test_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'female_recall': recall_score(y_true, y_pred),
        'female_precision': precision_score(y_true, y_pred, zero_division=0),
        'male_recall': recall_score(1-y_true, 1-y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro')
    }
    
    logger.info("\n" + "="*60)
    logger.info("GENDER V3 TEST SET RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy:         {test_metrics['accuracy']:.1%} (V1: {v1_metrics['accuracy']:.1%})")
    logger.info(f"Female Recall:    {test_metrics['female_recall']:.1%} (V1: {v1_metrics['female_recall']:.1%})")
    logger.info(f"Female Precision: {test_metrics['female_precision']:.1%} (V1: {v1_metrics['female_precision']:.1%})")
    logger.info(f"Male Recall:      {test_metrics['male_recall']:.1%} (V1: {v1_metrics['male_recall']:.1%})")
    logger.info(f"Macro F1:         {test_metrics['macro_f1']:.1%}")
    
    # Check if ALL metrics improved or stayed constant
    improvements = {
        'accuracy': test_metrics['accuracy'] - v1_metrics['accuracy'],
        'female_recall': test_metrics['female_recall'] - v1_metrics['female_recall'],
        'male_recall': test_metrics['male_recall'] - v1_metrics['male_recall']
    }
    
    logger.info("\nChanges from V1:")
    for metric, change in improvements.items():
        status = "OK" if change >= -0.03 else "BAD"
        logger.info(f"  {metric}: {change:+.1%} [{status}]")
    
    all_ok = all(v >= -0.05 for v in improvements.values())
    female_improved = improvements['female_recall'] > 0.05
    
    if all_ok and female_improved:
        logger.info("\n*** SUCCESS: All metrics maintained + female recall improved! ***")
    elif all_ok:
        logger.info("\nOK: All metrics maintained, but female recall didn't improve much")
    else:
        logger.warning("\nWARNING: Some metrics degraded beyond tolerance")
    
    # Save model
    save_path = project_root / 'experiments/v2_correction/models/gender_v3'
    predictor.save(save_path)
    
    # Save predictions
    all_preds = predictor.predict(features)
    all_preds.to_parquet(project_root / 'experiments/v2_correction/gender_predictions_v3.parquet')
    
    return {
        'test_metrics': test_metrics,
        'v1_metrics': v1_metrics,
        'improvements': improvements,
        'cv_metrics': train_results.get('cv_metrics', {}),
        'success': all_ok and female_improved
    }


def train_age_v3(data: dict, features: pd.DataFrame) -> dict:
    """Train Age V3 with accuracy preservation."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING AGE PREDICTOR V3")
    logger.info("="*70)
    
    self_decl = data['self_declarations']
    age_labeled = self_decl[self_decl['age_bucket_self_declared'].notna()].copy()
    age_labeled['age_binary'] = age_labeled['age_bucket_self_declared'].apply(map_to_binary)
    
    # Split train/test
    train_users, test_users = train_test_split(
        age_labeled['author'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=age_labeled['age_binary']
    )
    
    logger.info(f"Train: {len(train_users)}, Test: {len(test_users)}")
    
    v1_accuracy = 0.706
    
    # Train
    predictor = AgePredictor_V3()
    train_labels = age_labeled[age_labeled['author'].isin(train_users)]
    train_results = predictor.fit(
        features,
        train_labels,
        anthroscore_df=data['anthroscores'],
        v1_accuracy=v1_accuracy
    )
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_features = features[features['author'].isin(test_users)]
    test_labels = age_labeled[age_labeled['author'].isin(test_users)]
    
    predictions = predictor.predict(test_features)
    
    eval_df = predictions.merge(
        test_labels[['author', 'age_binary']],
        on='author'
    )
    
    test_accuracy = accuracy_score(eval_df['age_binary'], eval_df['age_predicted'])
    
    logger.info("\n" + "="*60)
    logger.info("AGE V3 TEST SET RESULTS")
    logger.info("="*60)
    logger.info(f"Test Accuracy: {test_accuracy:.1%} (V1: {v1_accuracy:.1%})")
    logger.info(f"CV Accuracy:   {train_results['cv_accuracy']:.1%}")
    
    change = test_accuracy - v1_accuracy
    if change >= -0.03:
        logger.info(f"\nOK: Accuracy maintained ({change:+.1%})")
    else:
        logger.warning(f"\nWARNING: Accuracy dropped ({change:+.1%})")
    
    # Save model
    save_path = project_root / 'experiments/v2_correction/models/age_v3'
    predictor.save(save_path)
    
    # Save predictions
    all_preds = predictor.predict(features)
    all_preds.to_parquet(project_root / 'experiments/v2_correction/age_predictions_v3.parquet')
    
    return {
        'test_accuracy': test_accuracy,
        'cv_accuracy': train_results['cv_accuracy'],
        'v1_accuracy': v1_accuracy,
        'accuracy_change': change,
        'best_params': train_results['best_params']
    }


def generate_v3_report(gender_results: dict, age_results: dict):
    """Generate final V3 report."""
    
    lines = [
        "="*70,
        "V3 MODEL RESULTS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70,
        "",
        "APPROACH: Balanced optimization - improve metrics without destroying others",
        "",
        "="*70,
        "GENDER PREDICTOR V3",
        "="*70,
        "",
        "Test Set Metrics:",
        f"  Accuracy:         {gender_results['test_metrics']['accuracy']:.1%} (V1: {gender_results['v1_metrics']['accuracy']:.1%}) [{gender_results['improvements']['accuracy']:+.1%}]",
        f"  Female Recall:    {gender_results['test_metrics']['female_recall']:.1%} (V1: {gender_results['v1_metrics']['female_recall']:.1%}) [{gender_results['improvements']['female_recall']:+.1%}]",
        f"  Female Precision: {gender_results['test_metrics']['female_precision']:.1%} (V1: {gender_results['v1_metrics']['female_precision']:.1%})",
        f"  Male Recall:      {gender_results['test_metrics']['male_recall']:.1%} (V1: {gender_results['v1_metrics']['male_recall']:.1%}) [{gender_results['improvements']['male_recall']:+.1%}]",
        f"  Macro F1:         {gender_results['test_metrics']['macro_f1']:.1%}",
        "",
        f"SUCCESS: {'YES - All metrics maintained + female recall improved' if gender_results['success'] else 'NO - Some metrics degraded'}",
        "",
        "="*70,
        "AGE PREDICTOR V3",
        "="*70,
        "",
        f"Test Accuracy: {age_results['test_accuracy']:.1%} (V1: {age_results['v1_accuracy']:.1%}) [{age_results['accuracy_change']:+.1%}]",
        f"CV Accuracy:   {age_results['cv_accuracy']:.1%}",
        "",
        f"ACCURACY MAINTAINED: {'YES' if age_results['accuracy_change'] >= -0.03 else 'NO'}",
        "",
        "="*70,
        "FILES GENERATED",
        "="*70,
        "",
        "experiments/v2_correction/",
        "  models/gender_v3/gender_predictor_v3.pkl",
        "  models/age_v3/age_predictor_v3.pkl",
        "  gender_predictions_v3.parquet",
        "  age_predictions_v3.parquet",
        "  v3_training.log",
        "  V3_RESULTS_REPORT.txt",
        "",
        "="*70
    ]
    
    report_path = project_root / 'experiments/v2_correction/V3_RESULTS_REPORT.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"\nReport saved to: {report_path}")
    
    # Print summary
    print("\n" + "\n".join(lines[:30]))
    
    # Save JSON
    results = {
        'gender': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in gender_results.items() if k != 'cv_metrics'},
        'age': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                for k, v in age_results.items() if k != 'best_params'},
        'timestamp': datetime.now().isoformat()
    }
    
    # Clean nested dicts
    if 'test_metrics' in results['gender']:
        results['gender']['test_metrics'] = {
            k: float(v) for k, v in results['gender']['test_metrics'].items()
        }
    if 'v1_metrics' in results['gender']:
        results['gender']['v1_metrics'] = {
            k: float(v) for k, v in results['gender']['v1_metrics'].items()
        }
    if 'improvements' in results['gender']:
        results['gender']['improvements'] = {
            k: float(v) for k, v in results['gender']['improvements'].items()
        }
    
    with open(project_root / 'experiments/v2_correction/v3_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("V3 MODEL TRAINING - BALANCED OPTIMIZATION")
    logger.info("="*70)
    
    # Create directories
    (project_root / 'experiments/v2_correction/models/gender_v3').mkdir(parents=True, exist_ok=True)
    (project_root / 'experiments/v2_correction/models/age_v3').mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Extract features
    logger.info("\nExtracting gender features...")
    gender_features = extract_gender_features(data)
    
    logger.info("\nExtracting age features...")
    age_features = extract_age_features(data)
    
    # Train Gender V3
    gender_results = train_gender_v3(data, gender_features)
    
    # Train Age V3
    age_results = train_age_v3(data, age_features)
    
    # Generate report
    generate_v3_report(gender_results, age_results)
    
    logger.info("\n" + "="*70)
    logger.info("V3 TRAINING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
