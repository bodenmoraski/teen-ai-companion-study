"""
Test the V2 community embeddings fix against ground truth.

This script:
1. Runs the V2 classification
2. Validates against self-declarations
3. Compares to V1 results
"""
import sys
from pathlib import Path
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 70)
    print("TESTING COMMUNITY EMBEDDINGS V2 FIXES")
    print("=" * 70)
    
    # Load data
    comments = pd.read_parquet("data/processed/all_comments.parquet")
    demographics = pd.read_parquet("data/features/demographics.parquet")
    api_data_path = Path("data/features/user_subreddit_interactions.parquet")
    
    print(f"\nLoaded {len(comments):,} comments")
    print(f"Loaded {len(demographics):,} users in demographics")
    
    # Run V2 classification
    from src.demographics.community_embedding_v2 import classify_with_community_embeddings_v2
    
    print("\n" + "=" * 70)
    print("RUNNING V2 CLASSIFICATION")
    print("=" * 70)
    
    v2_results = classify_with_community_embeddings_v2(
        comments,
        api_data_path=api_data_path
    )
    
    print(f"\nClassified {len(v2_results):,} users with V2")
    
    # Merge with demographics for validation
    merged = v2_results.merge(demographics, on='author', how='inner', suffixes=('_v2', '_v1'))
    
    print("\n" + "=" * 70)
    print("AGE CLASSIFICATION VALIDATION")
    print("=" * 70)
    
    # Age validation against self-declarations
    age_mask = merged['age_bucket_self_declared'].notna() & merged['age_bucket_community_v2'].notna()
    age_data = merged[age_mask]
    
    if len(age_data) > 0:
        # V2 accuracy
        v2_correct = (age_data['age_bucket_self_declared'] == age_data['age_bucket_community_v2']).sum()
        v2_accuracy = v2_correct / len(age_data)
        
        # V1 accuracy (for comparison)
        v1_mask = age_data['age_bucket_community_v1'].notna()
        v1_correct = (age_data.loc[v1_mask, 'age_bucket_self_declared'] == 
                      age_data.loc[v1_mask, 'age_bucket_community_v1']).sum()
        v1_accuracy = v1_correct / v1_mask.sum() if v1_mask.sum() > 0 else 0
        
        print(f"\nAGE ACCURACY:")
        print(f"  V1 (percentile-based): {v1_accuracy:.1%}")
        print(f"  V2 (calibrated):       {v2_accuracy:.1%}")
        print(f"  Improvement:           {(v2_accuracy - v1_accuracy)*100:+.1f} percentage points")
        
        # Distribution comparison
        print("\nV2 Age Distribution:")
        v2_dist = v2_results['age_bucket_community'].value_counts(normalize=True)
        for bucket in ['13-18', '19-25', '26-40', '41-60', '61-80']:
            if bucket in v2_dist:
                print(f"  {bucket}: {v2_dist[bucket]:.1%}")
        
        # Confusion matrix
        labels = ['13-18', '19-25', '26-40', '41-60', '61-80']
        cm = confusion_matrix(age_data['age_bucket_self_declared'], 
                             age_data['age_bucket_community_v2'], 
                             labels=labels)
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm_df.to_string())
    else:
        print("\nNo age ground truth data for validation")
    
    print("\n" + "=" * 70)
    print("GENDER CLASSIFICATION VALIDATION")
    print("=" * 70)
    
    # Gender validation
    gender_mask = (
        merged['gender_self_declared'].notna() & 
        merged['gender_community_v2'].notna() &
        (merged['gender_community_v2'] != 'unknown')
    )
    gender_data = merged[gender_mask]
    
    if len(gender_data) > 0:
        # V2 accuracy
        v2_correct = (gender_data['gender_self_declared'] == gender_data['gender_community_v2']).sum()
        v2_accuracy = v2_correct / len(gender_data)
        
        print(f"\nGENDER ACCURACY:")
        print(f"  V2 (calibrated): {v2_accuracy:.1%} ({v2_correct}/{len(gender_data)})")
        
        # Gender distribution
        print("\nV2 Gender Distribution:")
        v2_dist = v2_results['gender_community'].value_counts()
        for gender in ['male', 'female', 'unknown']:
            if gender in v2_dist:
                print(f"  {gender}: {v2_dist[gender]:,} ({v2_dist[gender]/len(v2_results):.1%})")
        
        # Check ratio
        if 'female' in v2_dist and 'male' in v2_dist:
            ratio = v2_dist['female'] / max(v2_dist['male'], 1)
            print(f"\nFemale:Male ratio: {ratio:.1f}:1")
            if ratio < 10:
                print("  PASS: Ratio is reasonable (< 10:1)")
            else:
                print("  FAIL: Ratio is still too extreme")
        
        # Confusion matrix
        labels = ['male', 'female']
        subset = gender_data[gender_data['gender_self_declared'].isin(labels)]
        if len(subset) > 0:
            cm = confusion_matrix(subset['gender_self_declared'], 
                                 subset['gender_community_v2'], 
                                 labels=labels)
            print("\nConfusion Matrix (rows=true, cols=predicted):")
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            print(cm_df.to_string())
    else:
        print("\nNo gender ground truth data for validation (all 'unknown')")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nTEST RESULTS:")
    issues = []
    
    # Check age accuracy
    if len(age_data) > 0:
        if v2_accuracy > 0.35:
            print("  [PASS] Age accuracy > 35%")
        else:
            print("  [FAIL] Age accuracy still too low")
            issues.append("age_accuracy")
    
    # Check gender ratio
    v2_dist = v2_results['gender_community'].value_counts()
    if 'female' in v2_dist and 'male' in v2_dist:
        ratio = v2_dist['female'] / max(v2_dist['male'], 1)
        if ratio < 10:
            print("  [PASS] Gender ratio is reasonable")
        else:
            print("  [FAIL] Gender ratio still too extreme")
            issues.append("gender_ratio")
    
    # Check age distribution is not uniform
    age_pcts = v2_results['age_bucket_community'].value_counts(normalize=True)
    if age_pcts.std() > 0.02:
        print("  [PASS] Age distribution is NOT uniform")
    else:
        print("  [FAIL] Age distribution is still uniform")
        issues.append("uniform_age")
    
    if issues:
        print(f"\n{len(issues)} issues remaining: {issues}")
    else:
        print("\nALL TESTS PASSED!")
    
    # Save V2 results
    output_path = Path("data/features/community_embeddings_v2.parquet")
    v2_results.to_parquet(output_path, index=False)
    print(f"\nV2 results saved to {output_path}")
    
    return len(issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

