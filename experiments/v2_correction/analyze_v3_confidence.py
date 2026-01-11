"""
Analyze V3 models with confidence thresholds.
Shows accuracy at different confidence levels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

project_root = Path(__file__).parent.parent.parent


def analyze_gender_confidence():
    """Analyze gender predictions at different confidence thresholds."""
    print("="*70)
    print("GENDER V3: CONFIDENCE-FILTERED ACCURACY ANALYSIS")
    print("="*70)
    
    # Load predictions
    preds = pd.read_parquet(project_root / 'experiments/v2_correction/gender_predictions_v3.parquet')
    
    # Load ground truth
    self_decl = pd.read_parquet(project_root / 'Data/features/self_declarations.parquet')
    gender_labeled = self_decl[self_decl['gender_self_declared'].isin(['male', 'female'])]
    
    # Merge
    df = preds.merge(gender_labeled[['author', 'gender_self_declared']], on='author')
    
    print(f"\nTotal labeled users: {len(df)}")
    print(f"Confidence range: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    print(f"\n{'Threshold':>10} {'Users':>8} {'Coverage':>10} {'Accuracy':>10} {'F-Recall':>10} {'M-Recall':>10} {'Macro-F1':>10}")
    print("-"*70)
    
    for thresh in thresholds:
        mask = df['confidence'] >= thresh
        subset = df[mask]
        
        if len(subset) < 10:
            continue
        
        y_true = (subset['gender_self_declared'] == 'female').astype(int)
        y_pred = (subset['gender_predicted'] == 'female').astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f_rec = recall_score(y_true, y_pred, zero_division=0)
        m_rec = recall_score(1-y_true, 1-y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro')
        coverage = len(subset) / len(df)
        
        print(f"{thresh:>10.2f} {len(subset):>8} {coverage:>10.1%} {acc:>10.1%} {f_rec:>10.1%} {m_rec:>10.1%} {f1:>10.1%}")
    
    return df


def analyze_age_confidence():
    """Analyze age predictions at different confidence thresholds."""
    print("\n" + "="*70)
    print("AGE V3: CONFIDENCE-FILTERED ACCURACY ANALYSIS")
    print("="*70)
    
    # Load predictions
    preds = pd.read_parquet(project_root / 'experiments/v2_correction/age_predictions_v3.parquet')
    
    # Load ground truth
    self_decl = pd.read_parquet(project_root / 'Data/features/self_declarations.parquet')
    age_labeled = self_decl[self_decl['age_bucket_self_declared'].notna()].copy()
    
    def map_to_binary(bucket):
        return 'teen' if bucket == '13-18' else 'adult'
    
    age_labeled['age_binary'] = age_labeled['age_bucket_self_declared'].apply(map_to_binary)
    
    # Merge
    df = preds.merge(age_labeled[['author', 'age_binary']], on='author')
    
    print(f"\nTotal labeled users: {len(df)}")
    print(f"Confidence range: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    print(f"\n{'Threshold':>10} {'Users':>8} {'Coverage':>10} {'Accuracy':>10} {'Teen-Rec':>10} {'Adult-Rec':>10}")
    print("-"*70)
    
    for thresh in thresholds:
        mask = df['confidence'] >= thresh
        subset = df[mask]
        
        if len(subset) < 10:
            continue
        
        acc = accuracy_score(subset['age_binary'], subset['age_predicted'])
        
        teen_mask = subset['age_binary'] == 'teen'
        adult_mask = subset['age_binary'] == 'adult'
        
        teen_rec = (subset.loc[teen_mask, 'age_predicted'] == 'teen').mean() if teen_mask.sum() > 0 else 0
        adult_rec = (subset.loc[adult_mask, 'age_predicted'] == 'adult').mean() if adult_mask.sum() > 0 else 0
        
        coverage = len(subset) / len(df)
        
        print(f"{thresh:>10.2f} {len(subset):>8} {coverage:>10.1%} {acc:>10.1%} {teen_rec:>10.1%} {adult_rec:>10.1%}")
    
    return df


if __name__ == "__main__":
    gender_df = analyze_gender_confidence()
    age_df = analyze_age_confidence()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nUse high confidence thresholds for analyses requiring high accuracy.")
    print("Lower thresholds provide more statistical power (larger sample sizes).")
