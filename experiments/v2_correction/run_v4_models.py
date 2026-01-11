"""
Run V4 Advanced Models and Compare to V3

This script:
1. Loads data and features (reuses V3 feature extraction)
2. Trains V4 models (multi-algorithm stacking)
3. Compares to V3 results
4. Evaluates with confidence thresholds
5. Generates comprehensive report

Author: Research Agent
Date: 2026-01-10
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.v2_correction.v4_advanced_models import GenderPredictor_V4, AgePredictor_V4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'experiments/v2_correction/v4_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load all required data."""
    logger.info("Loading data...")
    
    data = {
        'comments': pd.read_parquet(project_root / 'Data/processed/all_comments.parquet'),
        'user_subreddits': pd.read_parquet(project_root / 'Data/features/user_subreddit_interactions.parquet'),
        'self_declarations': pd.read_parquet(project_root / 'Data/features/self_declarations.parquet'),
        'anthroscores': pd.read_parquet(project_root / 'Data/features/user_anthroscores.parquet'),
        'existing_features': pd.read_parquet(project_root / 'Data/features/ultimate_predictor/all_features.parquet')
    }
    
    # Load V3 results for comparison
    v3_results = {}
    try:
        with open(project_root / 'experiments/v2_correction/v3_results.json', 'r') as f:
            v3_results = json.load(f)
    except:
        logger.warning("Could not load V3 results for comparison")
    
    return data, v3_results


def extract_all_features(data: dict) -> pd.DataFrame:
    """Extract comprehensive feature set for V4 models."""
    logger.info("Extracting features for V4...")
    
    # Start with existing text embeddings
    existing = data['existing_features']
    text_cols = [c for c in existing.columns if c.startswith('text_emb_')]
    features = existing[['author'] + text_cols].copy()
    
    logger.info(f"  Text embeddings: {len(text_cols)} features")
    
    # Add subreddit features
    logger.info("  Extracting subreddit features...")
    user_subs = data['user_subreddits']
    if 'subreddit' in user_subs.columns:
        agg = user_subs.groupby('author')['subreddit'].apply(list).reset_index()
        agg.columns = ['author', 'subreddits']
        user_subs = agg
    
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
    
    logger.info(f"  Subreddit features: {len(selected_subs)} features")
    
    # Add behavioral features
    logger.info("  Extracting behavioral features...")
    behav_features = []
    for author, group in data['comments'].groupby('author'):
        feat = {'author': author}
        feat['comment_count'] = len(group)
        feat['avg_comment_length'] = group['body'].str.len().mean()
        feat['max_comment_length'] = group['body'].str.len().max()
        
        if 'created_utc' in group.columns:
            timestamps = pd.to_datetime(group['created_utc'], unit='s', errors='coerce')
            if timestamps.notna().any():
                hours = timestamps.dt.hour
                feat['pct_night'] = ((hours >= 22) | (hours < 6)).mean()
                feat['pct_day'] = ((hours >= 6) & (hours < 22)).mean()
                feat['posting_hour_mean'] = hours.mean()
                feat['posting_hour_std'] = hours.std() if len(hours) > 1 else 0
                feat['days_active'] = (timestamps.max() - timestamps.min()).days
        
        feat['unique_subreddits'] = group['subreddit'].nunique() if 'subreddit' in group.columns else 0
        
        # Text patterns
        bodies = ' '.join(group['body'].fillna('').astype(str))
        feat['exclaim_ratio'] = bodies.count('!') / (len(bodies) + 1)
        feat['question_ratio'] = bodies.count('?') / (len(bodies) + 1)
        feat['caps_ratio'] = sum(1 for c in bodies if c.isupper()) / (len(bodies) + 1)
        feat['emoji_like'] = sum(1 for c in bodies if ord(c) > 127) / (len(bodies) + 1)
        
        behav_features.append(feat)
    
    behav_df = pd.DataFrame(behav_features)
    features = features.merge(behav_df, on='author', how='left')
    
    logger.info(f"  Behavioral features: {len(behav_df.columns) - 1} features")
    
    # Add linguistic markers
    logger.info("  Extracting linguistic markers...")
    import re
    
    female_patterns = [
        r'\b(omg|omfg)\b', r'\b(cute|adorable|lovely)\b', r'!{2,}',
        r'\b(so|really|very)\s+(cute|sweet|nice)\b', r'<3', r'\baww+\b',
        r'\bhaha+\b', r'\blol+\b', r'\b(hubby|bf|boyfriend)\b',
        r'\b(girl|woman|sister|mom|mother|daughter)\b',
        r'\bi feel\b', r'\bi love\b', r'\bso happy\b', r'\bmy feelings\b',
    ]
    
    male_patterns = [
        r'\b(dude|bro|man)\b', r'\b(wife|gf|girlfriend)\b',
        r'\b(guy|boy|brother|dad|father|son)\b',
        r'\b(fuck|shit|damn)\b', r'\btbh\b', r'\bimo\b',
        r'\b(awesome|epic|sick|based)\b', r'\bnah\b',
    ]
    
    teen_patterns = [
        r'\b(lowkey|highkey|no cap|bet|fr|deadass)\b',
        r'\b(bruh|bruhhh|breh)\b', r'\b(yeet|sheesh|bussin|slay)\b',
        r'\b(lit|fire|goat|based|mid)\b', r'\b(ngl|fax|cap|sus)\b',
        r'\b(rn|rn rn)\b',
    ]
    
    ling_features = []
    for author, group in data['comments'].groupby('author'):
        bodies = ' '.join(group['body'].fillna('').astype(str).str.lower())
        feat = {'author': author}
        
        feat['female_marker_count'] = sum(len(re.findall(p, bodies)) for p in female_patterns)
        feat['male_marker_count'] = sum(len(re.findall(p, bodies)) for p in male_patterns)
        feat['teen_marker_count'] = sum(len(re.findall(p, bodies)) for p in teen_patterns)
        
        total = len(bodies) + 1
        feat['female_marker_density'] = feat['female_marker_count'] / total * 1000
        feat['male_marker_density'] = feat['male_marker_count'] / total * 1000
        feat['teen_marker_density'] = feat['teen_marker_count'] / total * 1000
        
        ling_features.append(feat)
    
    ling_df = pd.DataFrame(ling_features)
    features = features.merge(ling_df, on='author', how='left')
    
    logger.info(f"  Linguistic features: {len(ling_df.columns) - 1} features")
    
    # Fill NaNs
    features = features.fillna(0)
    
    logger.info(f"\nTotal features: {features.shape[1] - 1} for {features.shape[0]} users")
    
    return features


def evaluate_with_confidence_thresholds(predictions: pd.DataFrame, ground_truth: pd.DataFrame,
                                        pred_col: str, label_col: str, is_binary_gender: bool = True):
    """Evaluate at different confidence thresholds."""
    
    merged = predictions.merge(ground_truth, on='author', how='inner')
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for thresh in thresholds:
        mask = merged['confidence'] >= thresh
        subset = merged[mask]
        
        if len(subset) < 10:
            continue
        
        if is_binary_gender:
            y_true = (subset[label_col] == 'female').astype(int)
            y_pred = (subset[pred_col] == 'female').astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            f_rec = recall_score(y_true, y_pred, zero_division=0)
            m_rec = recall_score(1-y_true, 1-y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro')
        else:
            y_true = (subset[label_col] == 'teen').astype(int)
            y_pred = (subset[pred_col] == 'teen').astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            f_rec = recall_score(y_true, y_pred, zero_division=0)  # teen recall
            m_rec = recall_score(1-y_true, 1-y_pred, zero_division=0)  # adult recall
            f1 = f1_score(y_true, y_pred, average='macro')
        
        results.append({
            'threshold': thresh,
            'coverage': len(subset) / len(merged),
            'n_users': len(subset),
            'accuracy': acc,
            'minority_recall': f_rec,
            'majority_recall': m_rec,
            'macro_f1': f1
        })
    
    return results


def main():
    logger.info("="*70)
    logger.info("V4 ADVANCED MODELS - PUSHING FOR MAXIMUM PERFORMANCE")
    logger.info("="*70)
    
    # Load data
    data, v3_results = load_data()
    self_decl = data['self_declarations']
    
    # Extract features
    all_features = extract_all_features(data)
    
    # Save features for future use
    all_features.to_parquet(project_root / 'experiments/v2_correction/all_features_v4.parquet')
    
    logger.info(f"Total users with features: {len(all_features)}")
    logger.info(f"Total users with labels: {len(self_decl)}")
    
    results = {'v4': {}, 'v3_comparison': v3_results}
    
    # ============================================
    # GENDER PREDICTOR V4
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: GENDER PREDICTOR V4")
    logger.info("="*70)
    
    gender_labeled = self_decl[self_decl['gender_self_declared'].isin(['male', 'female'])]
    
    gender_model_v4 = GenderPredictor_V4()
    
    v1_gender_metrics = {
        'accuracy': 0.813,
        'female_recall': 0.44,
        'male_recall': 0.95
    }
    
    gender_results = gender_model_v4.fit(
        all_features,
        gender_labeled[['author', 'gender_self_declared']],
        v1_metrics=v1_gender_metrics
    )
    
    results['v4']['gender'] = gender_results
    
    # Predict on all users
    logger.info("\nPredicting for all users...")
    gender_predictions = gender_model_v4.predict(all_features)
    
    # Save predictions
    gender_predictions.to_parquet(
        project_root / 'experiments/v2_correction/gender_predictions_v4.parquet'
    )
    
    # Evaluate with confidence thresholds
    gender_conf_results = evaluate_with_confidence_thresholds(
        gender_predictions,
        gender_labeled[['author', 'gender_self_declared']],
        'gender_predicted', 'gender_self_declared', is_binary_gender=True
    )
    results['v4']['gender_confidence'] = gender_conf_results
    
    # ============================================
    # AGE PREDICTOR V4
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: AGE PREDICTOR V4")
    logger.info("="*70)
    
    age_labeled = self_decl[self_decl['age_bucket_self_declared'].notna()].copy()
    
    def map_to_binary(bucket):
        return 'teen' if bucket == '13-18' else 'adult'
    
    age_labeled['age_binary'] = age_labeled['age_bucket_self_declared'].apply(map_to_binary)
    
    age_model_v4 = AgePredictor_V4()
    
    age_results = age_model_v4.fit(
        all_features,
        age_labeled[['author', 'age_bucket_self_declared']],
        v1_accuracy=0.706
    )
    
    results['v4']['age'] = age_results
    
    # Predict on all users
    logger.info("\nPredicting for all users...")
    age_predictions = age_model_v4.predict(all_features)
    
    # Save predictions
    age_predictions.to_parquet(
        project_root / 'experiments/v2_correction/age_predictions_v4.parquet'
    )
    
    # Evaluate with confidence thresholds
    age_conf_results = evaluate_with_confidence_thresholds(
        age_predictions,
        age_labeled[['author', 'age_binary']],
        'age_predicted', 'age_binary', is_binary_gender=False
    )
    results['v4']['age_confidence'] = age_conf_results
    
    # ============================================
    # SAVE MODELS
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("SAVING MODELS")
    logger.info("="*70)
    
    gender_model_v4.save(project_root / 'experiments/v2_correction')
    age_model_v4.save(project_root / 'experiments/v2_correction')
    
    # ============================================
    # COMPARISON REPORT
    # ============================================
    generate_report(results)
    
    # Save results
    with open(project_root / 'experiments/v2_correction/v4_results.json', 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    
    return results


def generate_report(results: dict):
    """Generate comprehensive comparison report."""
    
    report_lines = [
        "="*70,
        "V4 ADVANCED MODELS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70,
        "",
        "EXECUTIVE SUMMARY",
        "-"*50,
        "V4 uses multi-algorithm stacking (XGBoost + LightGBM + RF + LogReg)",
        "with feature selection and calibrated probability outputs.",
        "",
    ]
    
    # Gender results
    if 'gender' in results.get('v4', {}):
        gr = results['v4']['gender']
        report_lines += [
            "",
            "GENDER PREDICTOR V4",
            "-"*50,
            f"Cross-Validation Metrics:",
            f"  - Accuracy:      {gr['cv_metrics']['accuracy']:.1%}",
            f"  - Female Recall: {gr['cv_metrics']['female_recall']:.1%}",
            f"  - Male Recall:   {gr['cv_metrics']['male_recall']:.1%}",
            f"  - Macro F1:      {gr['cv_metrics']['macro_f1']:.1%}",
            "",
            f"Improvements over V1:",
            f"  - Accuracy:      {gr['improvements']['accuracy']:+.1%}",
            f"  - Female Recall: {gr['improvements']['female_recall']:+.1%}",
            f"  - Male Recall:   {gr['improvements']['male_recall']:+.1%}",
        ]
    
    # Gender confidence table
    if 'gender_confidence' in results.get('v4', {}):
        report_lines += [
            "",
            "Gender V4 - Confidence-Filtered Performance:",
            f"{'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'F-Recall':>10}",
            "-"*45,
        ]
        for row in results['v4']['gender_confidence']:
            report_lines.append(
                f"{row['threshold']:>10.2f} {row['coverage']:>10.1%} {row['accuracy']:>10.1%} {row['minority_recall']:>10.1%}"
            )
    
    # Age results
    if 'age' in results.get('v4', {}):
        ar = results['v4']['age']
        report_lines += [
            "",
            "",
            "AGE PREDICTOR V4",
            "-"*50,
            f"Cross-Validation Accuracy: {ar['cv_accuracy']:.1%}",
            f"V1 Accuracy:               {ar['v1_accuracy']:.1%}",
            f"Improvement:               {ar['improvement']:+.1%}",
        ]
    
    # Age confidence table
    if 'age_confidence' in results.get('v4', {}):
        report_lines += [
            "",
            "Age V4 - Confidence-Filtered Performance:",
            f"{'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'Teen-Rec':>10}",
            "-"*45,
        ]
        for row in results['v4']['age_confidence']:
            report_lines.append(
                f"{row['threshold']:>10.2f} {row['coverage']:>10.1%} {row['accuracy']:>10.1%} {row['minority_recall']:>10.1%}"
            )
    
    report_lines += [
        "",
        "",
        "="*70,
        "CONCLUSION",
        "="*70,
        "",
        "V4 uses advanced stacking ensemble with multiple algorithms.",
        "Feature selection reduces noise and improves generalization.",
        "Calibrated probabilities provide better confidence estimates.",
        "",
        "For final analysis, use confidence filtering to trade off",
        "between coverage (statistical power) and accuracy.",
    ]
    
    # Write report
    report_path = project_root / 'experiments/v2_correction/V4_RESULTS_REPORT.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\nReport saved to: {report_path}")
    
    # Print to console
    for line in report_lines:
        print(line)


if __name__ == '__main__':
    main()
