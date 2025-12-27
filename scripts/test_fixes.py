"""
Test script for regression and community embedding fixes.

Run with: python scripts/test_fixes.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def test_regression_fix():
    """Test the fixed regression module with real data."""
    print("=" * 70)
    print("TESTING REGRESSION FIX")
    print("=" * 70)
    
    from src.statistical.regression_models_fixed import (
        run_rq2_regression_fixed, 
        generate_regression_tables_fixed
    )
    
    # Load real data
    df = pd.read_parquet('data/features/full_merged_dataset.parquet')
    print(f"Loaded {len(df)} users")
    
    # Run regression
    results = run_rq2_regression_fixed(df)
    
    model1_ok = results.get("model1_age_only") is not None
    model2_ok = results.get("model2_age_gender") is not None
    model3_ok = results.get("model3_full") is not None
    
    print(f"\nModel 1 (Age only): {'FITTED' if model1_ok else 'FAILED'}")
    print(f"Model 2 (Age+Gender): {'FITTED' if model2_ok else 'FAILED'}")
    print(f"Model 3 (Full): {'FITTED' if model3_ok else 'FAILED'}")
    print(f"Observations: {results.get('n_observations')}")
    
    if model1_ok:
        model = results['model1_age_only']
        print(f"\nModel 1 Statistics:")
        print(f"  R-squared: {model.rsquared:.6f}")
        print(f"  Adj R-squared: {model.rsquared_adj:.6f}")
        print(f"  F-statistic: {model.fvalue:.4f}")
        print(f"  F-stat p-value: {model.f_pvalue:.4e}")
        print(f"\nCoefficients:")
        for name, coef in model.params.items():
            pval = model.pvalues[name]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f"  {name}: {coef:.4f} (p={pval:.4f}) {sig}")
    
    # Generate tables
    output_path = Path('results/tables/regression_results_FIXED.txt')
    generate_regression_tables_fixed(results, output_path)
    print(f"\nTables saved to {output_path}")
    
    # Verify file has content
    with open(output_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        print(f"Output file has {len(lines)} lines")
        
        # Check for key content
        has_coefficients = 'Coefficient' in content
        has_rsquared = 'R-squared' in content
        has_pvalue = 'p-value' in content
        
        print(f"Has coefficients: {has_coefficients}")
        print(f"Has R-squared: {has_rsquared}")
        print(f"Has p-values: {has_pvalue}")
    
    success = model1_ok and model2_ok and has_coefficients
    print(f"\nREGRESSION FIX: {'SUCCESS' if success else 'FAILED'}")
    return success


def test_community_embedding_fix():
    """Test the fixed community embedding module."""
    print("\n" + "=" * 70)
    print("TESTING COMMUNITY EMBEDDING FIX")
    print("=" * 70)
    
    from src.demographics.community_embedding_fixed import (
        diagnose_seed_pairs,
        classify_with_community_embeddings_fixed,
        AGE_SEED_PAIRS_FIXED,
        GENDER_SEED_PAIRS_FIXED
    )
    
    # Check seed pair availability
    api_data_path = Path('data/features/user_subreddit_interactions.parquet')
    
    if not api_data_path.exists():
        print(f"Data file not found: {api_data_path}")
        return False
    
    print("\n--- Diagnosing Seed Pairs ---")
    diag = diagnose_seed_pairs(api_data_path)
    
    print(f"Total subreddits: {diag['total_subreddits']}")
    
    print("\nAge seed pairs (FIXED - lowercase):")
    age_valid = 0
    for pair, info in diag['age_seed_pairs'].items():
        status = "VALID" if info['valid'] else "INVALID"
        print(f"  {pair}: {status} ({info['term1_users']}, {info['term2_users']} users)")
        if info['valid']:
            age_valid += 1
    
    print("\nGender seed pairs (FIXED - lowercase + oney):")
    gender_valid = 0
    for pair, info in diag['gender_seed_pairs'].items():
        status = "VALID" if info['valid'] else "INVALID"
        print(f"  {pair}: {status} ({info['term1_users']}, {info['term2_users']} users)")
        if info['valid']:
            gender_valid += 1
    
    print(f"\nValid age pairs: {age_valid}/{len(AGE_SEED_PAIRS_FIXED)}")
    print(f"Valid gender pairs: {gender_valid}/{len(GENDER_SEED_PAIRS_FIXED)}")
    
    # Test classification
    if age_valid >= 2 and gender_valid >= 1:
        print("\n--- Testing Classification ---")
        
        # Load comments and run classification on sample
        comments = pd.read_parquet('data/processed/all_comments.parquet')
        print(f"Loaded {len(comments)} comments")
        
        # Run classification
        result = classify_with_community_embeddings_fixed(
            comments,
            api_data_path=api_data_path,
            gender_threshold_percentile=70
        )
        
        print(f"\nClassified {len(result)} users")
        
        # Check outputs
        has_age = result['age_bucket_community'].notna().sum()
        has_gender_score = 'gender_community_score' in result.columns
        non_unknown_gender = (result['gender_community'] != 'unknown').sum()
        
        print(f"Users with age classification: {has_age}")
        print(f"Has gender_community_score column: {has_gender_score}")
        print(f"Users with non-unknown gender: {non_unknown_gender}")
        
        print(f"\nAge distribution:")
        print(result['age_bucket_community'].value_counts())
        
        print(f"\nGender distribution:")
        print(result['gender_community'].value_counts())
        
        # Check gender score stats
        if has_gender_score:
            scores = result['gender_community_score']
            print(f"\nGender score statistics:")
            print(f"  Mean: {scores.mean():.4f}")
            print(f"  Std: {scores.std():.4f}")
            print(f"  Min: {scores.min():.4f}")
            print(f"  Max: {scores.max():.4f}")
        
        success = has_age > 0 and has_gender_score and non_unknown_gender > 0
        print(f"\nCOMMUNITY EMBEDDING FIX: {'SUCCESS' if success else 'PARTIAL'}")
        
        if not success:
            print("\nNOTE: Gender classification may still show all 'unknown' if")
            print("  gender seed pairs are not in the Word2Vec vocabulary.")
            print("  This happens when those subreddits have < min_count users.")
        
        return success
    else:
        print("\nInsufficient valid seed pairs for testing")
        return False


def main():
    """Run all fix tests."""
    print("=" * 70)
    print(" TESTING REGRESSION AND COMMUNITY EMBEDDING FIXES")
    print("=" * 70 + "\n")
    
    regression_ok = test_regression_fix()
    embedding_ok = test_community_embedding_fix()
    
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    print(f"Regression fix: {'PASS' if regression_ok else 'FAIL'}")
    print(f"Community embedding fix: {'PASS' if embedding_ok else 'NEEDS REVIEW'}")
    
    if regression_ok and embedding_ok:
        print("\nALL FIXES VERIFIED!")
        print("\nNext steps:")
        print("1. Update original modules to use fixed versions")
        print("2. Re-run Phase 2 (demographics) with fixed embeddings")
        print("3. Re-run Phase 3/4 (analysis) with fixed regression")
    elif regression_ok:
        print("\nRegression is fixed. Community embedding needs review.")
    else:
        print("\nFixes need more work.")
    
    return regression_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

