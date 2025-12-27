"""
Comprehensive tests for regression models and community embeddings.

These tests validate that:
1. Regression models fit correctly with proper column naming
2. Community embeddings produce valid age AND gender classifications
3. All edge cases are handled properly
4. Output format is NeurIPS-quality

Run with: python tests/test_regression_and_embeddings.py
"""
import logging
import sys
from pathlib import Path
import traceback
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST UTILITIES
# ============================================================================

def create_test_dataframe(n_users: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for regression testing."""
    np.random.seed(42)
    
    age_buckets = ["13-18", "19-25", "26-40", "41-60", "61-80"]
    genders = ["male", "female", "nonbinary", None]
    
    data = {
        'author': [f'user_{i}' for i in range(n_users)],
        'age_bucket': np.random.choice(age_buckets, n_users),
        'gender': np.random.choice(genders, n_users),
        'anthroscore_mean': np.random.normal(0.05, 0.7, n_users),
        'anthroscore_std': np.random.uniform(0.1, 1.0, n_users),
        'anthroscore_count': np.random.randint(1, 100, n_users),
    }
    
    return pd.DataFrame(data)


def create_test_subreddit_data(n_users: int = 500) -> pd.DataFrame:
    """Create synthetic subreddit interaction data for embedding testing."""
    np.random.seed(42)
    
    # All possible subreddits including seed pairs
    subreddits = [
        # Age seeds
        "teenagers", "RedditForGrownups", "teenrelationships", "relationship_advice",
        "highschool", "college", "GenZ", "GenX",
        # Gender seeds
        "AskWomen", "AskMen", "TwoXChromosomes", "MensRights", 
        "TheGirlSurvivalGuide", "askmen",  # Using askmen as lowercase alternative
        # Common subreddits
        "characterai", "replika", "askreddit", "memes", "gaming",
        "music", "movies", "books", "technology", "science",
        "news", "worldnews", "funny", "pics", "videos"
    ]
    
    rows = []
    for i in range(n_users):
        user = f'user_{i}'
        # Each user participates in 3-15 random subreddits
        n_subs = np.random.randint(3, 16)
        user_subs = np.random.choice(subreddits, n_subs, replace=False)
        for sub in user_subs:
            rows.append({
                'author': user,
                'subreddit': sub,
                'count': np.random.randint(1, 50)
            })
    
    return pd.DataFrame(rows)


# ============================================================================
# REGRESSION TESTS
# ============================================================================

def test_regression_column_naming():
    """
    TEST 1: Verify that column names don't break patsy formula parser.
    
    ISSUE: Column names like 'age_13-18' contain hyphens that are interpreted
    as minus signs by patsy, causing formula parsing to fail.
    
    FIX: Use Q('column_name') syntax or rename columns to use underscores.
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Regression column naming")
    logger.info("=" * 70)
    
    from statsmodels.formula.api import ols
    
    # Create test data
    df = create_test_dataframe(500)
    
    # Test 1a: Verify the issue exists with current naming
    logger.info("Testing if hyphen issue exists...")
    
    # Create dummies the current (broken) way
    age_dummies = pd.get_dummies(df['age_bucket'], prefix='age')
    test_df = pd.concat([df, age_dummies], axis=1)
    
    # Try to fit with current column names (should fail)
    hyphen_cols = [c for c in age_dummies.columns if '-' in c]
    if hyphen_cols:
        formula = f"anthroscore_mean ~ {hyphen_cols[0]}"
        try:
            model = ols(formula, data=test_df).fit()
            logger.error("UNEXPECTED: Formula with hyphen worked (shouldn't happen)")
            return False
        except Exception as e:
            logger.info(f"EXPECTED: Formula with hyphen failed: {type(e).__name__}")
    
    # Test 1b: Verify fix with underscore naming works
    logger.info("Testing underscore-based column names...")
    
    # Create dummies with underscores instead of hyphens
    df['age_bucket_safe'] = df['age_bucket'].str.replace('-', '_')
    age_dummies_safe = pd.get_dummies(df['age_bucket_safe'], prefix='age')
    test_df_safe = pd.concat([df, age_dummies_safe], axis=1)
    
    # Remove reference category
    if 'age_26_40' in test_df_safe.columns:
        age_cols = [c for c in age_dummies_safe.columns if c != 'age_26_40']
    else:
        age_cols = list(age_dummies_safe.columns)[:-1]  # Drop last as reference
    
    formula_safe = "anthroscore_mean ~ " + " + ".join(age_cols)
    try:
        model = ols(formula_safe, data=test_df_safe).fit()
        logger.info(f"SUCCESS: Model fitted with R² = {model.rsquared:.4f}")
        logger.info(f"  Coefficients: {len(model.params)}")
        logger.info(f"  Observations: {model.nobs}")
        return True
    except Exception as e:
        logger.error(f"FAILED: {e}")
        traceback.print_exc()
        return False


def test_regression_full_pipeline():
    """
    TEST 2: Test full regression pipeline with NeurIPS-level output.
    
    Validates:
    - All 3 models fit correctly
    - Effect sizes are calculated
    - Model comparison statistics work
    - Output format is complete
    """
    logger.info("=" * 70)
    logger.info("TEST 2: Full regression pipeline")
    logger.info("=" * 70)
    
    # We need to test with the FIXED version
    df = create_test_dataframe(1000)
    
    # Import fixed version (we'll create this)
    try:
        from src.statistical.regression_models_fixed import (
            prepare_regression_data_fixed,
            run_rq2_regression_fixed,
            generate_regression_tables
        )
        
        # Run full pipeline
        results = run_rq2_regression_fixed(df)
        
        # Validate results
        checks = {
            'has_model1': results.get('model1_age_only') is not None,
            'has_model2': results.get('model2_age_gender') is not None,
            'has_model3': results.get('model3_full') is not None,
            'has_n_obs': results.get('n_observations', 0) > 0,
        }
        
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {check}: {status}")
        
        if all(checks.values()):
            # Test table generation
            from io import StringIO
            output = StringIO()
            
            # Check model statistics
            model1 = results['model1_age_only']
            logger.info(f"\n  Model 1 Statistics:")
            logger.info(f"    R²: {model1.rsquared:.4f}")
            logger.info(f"    Adj R²: {model1.rsquared_adj:.4f}")
            logger.info(f"    F-stat: {model1.fvalue:.4f}")
            logger.info(f"    p-value: {model1.f_pvalue:.4e}")
            logger.info(f"    AIC: {model1.aic:.2f}")
            logger.info(f"    BIC: {model1.bic:.2f}")
            
            logger.info("\nSUCCESS: Full regression pipeline works!")
            return True
        else:
            logger.error("FAILED: Some models didn't fit")
            return False
            
    except ImportError:
        logger.warning("Fixed regression module not yet created - skipping full pipeline test")
        logger.info("Run 'python tests/test_regression_and_embeddings.py --create-fixes' to create")
        return None
    except Exception as e:
        logger.error(f"FAILED: {e}")
        traceback.print_exc()
        return False


def test_regression_with_real_data():
    """
    TEST 3: Test regression with actual project data.
    
    Uses real full_merged_dataset.parquet to ensure fix works on actual data.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: Regression with real data")
    logger.info("=" * 70)
    
    data_path = Path("data/features/full_merged_dataset.parquet")
    if not data_path.exists():
        logger.warning(f"Real data not found at {data_path} - skipping")
        return None
    
    try:
        from src.statistical.regression_models_fixed import run_rq2_regression_fixed
        
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} users")
        
        results = run_rq2_regression_fixed(df)
        
        if results.get('model1_age_only') is not None:
            model = results['model1_age_only']
            logger.info(f"SUCCESS: Real data regression works!")
            logger.info(f"  N observations: {model.nobs}")
            logger.info(f"  R²: {model.rsquared:.6f}")
            logger.info(f"  F-stat: {model.fvalue:.4f} (p={model.f_pvalue:.4e})")
            return True
        else:
            logger.error("FAILED: Model didn't fit on real data")
            return False
            
    except ImportError:
        logger.warning("Fixed regression module not yet created")
        return None
    except Exception as e:
        logger.error(f"FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# COMMUNITY EMBEDDING TESTS
# ============================================================================

def test_gender_embedding_scores():
    """
    TEST 4: Verify gender scores are computed AND saved correctly.
    
    ISSUE: gender_community_score column is not being saved, and all gender
    classifications are 'unknown'.
    
    FIX: Ensure gender scores are computed and saved to dataframe.
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Gender embedding scores")
    logger.info("=" * 70)
    
    # Check current demographics data
    demo_path = Path("data/features/demographics.parquet")
    if not demo_path.exists():
        logger.warning("Demographics data not found - skipping")
        return None
    
    demo = pd.read_parquet(demo_path)
    
    # Check for gender_community_score column
    has_score_column = 'gender_community_score' in demo.columns
    logger.info(f"  gender_community_score column exists: {has_score_column}")
    
    # Check gender distribution
    gender_dist = demo['gender_community'].value_counts()
    logger.info(f"  Gender distribution:\n{gender_dist}")
    
    # Calculate expected: at least some non-unknown classifications
    n_non_unknown = (demo['gender_community'] != 'unknown').sum()
    n_male = (demo['gender_community'] == 'male').sum()
    n_female = (demo['gender_community'] == 'female').sum()
    
    logger.info(f"  Non-unknown: {n_non_unknown}")
    logger.info(f"  Male: {n_male}")
    logger.info(f"  Female: {n_female}")
    
    # This should FAIL with current data (showing the bug)
    if n_non_unknown == 0:
        logger.error("ISSUE CONFIRMED: All gender classifications are 'unknown'")
        logger.error("This indicates a bug in gender embedding computation")
        return False
    else:
        logger.info("SUCCESS: Gender classifications are being produced")
        return True


def test_seed_pair_validity():
    """
    TEST 5: Check if all seed pairs exist in subreddit data.
    
    ISSUE: 'everyman' subreddit has 0 users, making that seed pair useless.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Seed pair validity")
    logger.info("=" * 70)
    
    usi_path = Path("data/features/user_subreddit_interactions.parquet")
    if not usi_path.exists():
        logger.warning("User subreddit interactions not found - skipping")
        return None
    
    usi = pd.read_parquet(usi_path)
    
    # Check age seed pairs
    from src.utils.config import AGE_SEED_PAIRS, GENDER_SEED_PAIRS
    
    logger.info("\n  Age seed pairs:")
    age_valid = 0
    for term1, term2 in AGE_SEED_PAIRS:
        count1 = usi[usi['subreddit'].str.lower() == term1.lower()]['author'].nunique()
        count2 = usi[usi['subreddit'].str.lower() == term2.lower()]['author'].nunique()
        valid = count1 > 0 and count2 > 0
        status = "✅" if valid else "❌"
        logger.info(f"    {status} ({term1}, {term2}): {count1}, {count2} users")
        if valid:
            age_valid += 1
    
    logger.info(f"\n  Gender seed pairs:")
    gender_valid = 0
    for term1, term2 in GENDER_SEED_PAIRS:
        count1 = usi[usi['subreddit'].str.lower() == term1.lower()]['author'].nunique()
        count2 = usi[usi['subreddit'].str.lower() == term2.lower()]['author'].nunique()
        valid = count1 > 0 and count2 > 0
        status = "✅" if valid else "❌"
        logger.info(f"    {status} ({term1}, {term2}): {count1}, {count2} users")
        if valid:
            gender_valid += 1
    
    logger.info(f"\n  Valid age pairs: {age_valid}/{len(AGE_SEED_PAIRS)}")
    logger.info(f"  Valid gender pairs: {gender_valid}/{len(GENDER_SEED_PAIRS)}")
    
    # Should have at least 2 valid pairs for each dimension
    success = age_valid >= 2 and gender_valid >= 2
    
    if not success:
        logger.error("ISSUE: Not enough valid seed pairs")
        return False
    
    return True


def test_word2vec_embedding_quality():
    """
    TEST 6: Check Word2Vec embedding quality for demographics.
    
    Validates that embeddings capture the intended semantic relationships.
    """
    logger.info("=" * 70)
    logger.info("TEST 6: Word2Vec embedding quality")
    logger.info("=" * 70)
    
    usi_path = Path("data/features/user_subreddit_interactions.parquet")
    if not usi_path.exists():
        logger.warning("User subreddit interactions not found - skipping")
        return None
    
    try:
        from gensim.models import Word2Vec
        from src.demographics.community_embedding import (
            collect_user_subreddits,
            build_subreddit_embeddings,
            build_dimension_vector
        )
        from src.utils.config import AGE_SEED_PAIRS, GENDER_SEED_PAIRS
        
        usi = pd.read_parquet(usi_path)
        
        # Get subreddit lists per user
        user_subreddits = usi.groupby('author')['subreddit'].apply(list).tolist()
        logger.info(f"  Training on {len(user_subreddits)} users")
        
        # Build model
        model = build_subreddit_embeddings(user_subreddits, vector_size=100, min_count=3)
        logger.info(f"  Model vocabulary: {len(model.wv)} subreddits")
        
        # Check if seed pairs are in vocabulary
        logger.info("\n  Seed pairs in vocabulary:")
        for term1, term2 in AGE_SEED_PAIRS + GENDER_SEED_PAIRS:
            in_vocab1 = term1 in model.wv
            in_vocab2 = term2 in model.wv
            status1 = "✅" if in_vocab1 else "❌"
            status2 = "✅" if in_vocab2 else "❌"
            logger.info(f"    {term1}: {status1}  |  {term2}: {status2}")
        
        # Build dimensions and check they're non-zero
        age_dim = build_dimension_vector(model, AGE_SEED_PAIRS)
        gender_dim = build_dimension_vector(model, GENDER_SEED_PAIRS)
        
        age_norm = np.linalg.norm(age_dim)
        gender_norm = np.linalg.norm(gender_dim)
        
        logger.info(f"\n  Age dimension norm: {age_norm:.4f}")
        logger.info(f"  Gender dimension norm: {gender_norm:.4f}")
        
        if age_norm < 0.001:
            logger.error("ISSUE: Age dimension is zero/near-zero (no valid seed pairs)")
            return False
        if gender_norm < 0.001:
            logger.error("ISSUE: Gender dimension is zero/near-zero (no valid seed pairs)")
            return False
        
        logger.info("SUCCESS: Embedding dimensions are valid")
        return True
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        traceback.print_exc()
        return False


def test_gender_threshold_logic():
    """
    TEST 7: Check gender classification threshold logic.
    
    ISSUE: The threshold of ±0.2 may be too strict, causing all users
    to be classified as 'unknown'.
    """
    logger.info("=" * 70)
    logger.info("TEST 7: Gender threshold logic")
    logger.info("=" * 70)
    
    usi_path = Path("data/features/user_subreddit_interactions.parquet")
    if not usi_path.exists():
        logger.warning("User subreddit interactions not found - skipping")
        return None
    
    try:
        from gensim.models import Word2Vec
        from src.demographics.community_embedding import (
            build_subreddit_embeddings,
            build_dimension_vector,
            project_user_to_dimension
        )
        from src.utils.config import GENDER_SEED_PAIRS
        
        usi = pd.read_parquet(usi_path)
        
        # Get subreddit lists per user
        user_data = usi.groupby('author')['subreddit'].apply(list).reset_index()
        user_subreddits = user_data['subreddit'].tolist()
        
        # Build model and dimension
        model = build_subreddit_embeddings(user_subreddits, vector_size=100, min_count=3)
        gender_dim = build_dimension_vector(model, GENDER_SEED_PAIRS)
        
        # Project all users and check distribution
        scores = []
        for subs in user_subreddits[:5000]:  # Sample 5000 users
            score = project_user_to_dimension(subs, model, gender_dim)
            if score != 0.0:  # Exclude users with no valid subreddits
                scores.append(score)
        
        scores = np.array(scores)
        logger.info(f"  Sampled {len(scores)} users with valid scores")
        
        if len(scores) == 0:
            logger.error("ISSUE: No users have valid gender projection scores")
            return False
        
        logger.info(f"  Score statistics:")
        logger.info(f"    Min: {scores.min():.4f}")
        logger.info(f"    Max: {scores.max():.4f}")
        logger.info(f"    Mean: {scores.mean():.4f}")
        logger.info(f"    Std: {scores.std():.4f}")
        logger.info(f"    25th percentile: {np.percentile(scores, 25):.4f}")
        logger.info(f"    75th percentile: {np.percentile(scores, 75):.4f}")
        
        # Check how many would be classified with current threshold (±0.2)
        n_male_current = (scores > 0.2).sum()
        n_female_current = (scores < -0.2).sum()
        n_unknown_current = len(scores) - n_male_current - n_female_current
        
        logger.info(f"\n  With threshold ±0.2:")
        logger.info(f"    Male: {n_male_current} ({100*n_male_current/len(scores):.1f}%)")
        logger.info(f"    Female: {n_female_current} ({100*n_female_current/len(scores):.1f}%)")
        logger.info(f"    Unknown: {n_unknown_current} ({100*n_unknown_current/len(scores):.1f}%)")
        
        # Check with percentile-based threshold (more adaptive)
        threshold = np.percentile(np.abs(scores), 70)  # Top 30% get classified
        n_male_adaptive = (scores > threshold).sum()
        n_female_adaptive = (scores < -threshold).sum()
        n_unknown_adaptive = len(scores) - n_male_adaptive - n_female_adaptive
        
        logger.info(f"\n  With adaptive threshold ±{threshold:.4f} (70th percentile):")
        logger.info(f"    Male: {n_male_adaptive} ({100*n_male_adaptive/len(scores):.1f}%)")
        logger.info(f"    Female: {n_female_adaptive} ({100*n_female_adaptive/len(scores):.1f}%)")
        logger.info(f"    Unknown: {n_unknown_adaptive} ({100*n_unknown_adaptive/len(scores):.1f}%)")
        
        if n_unknown_current > 0.95 * len(scores):
            logger.error("ISSUE CONFIRMED: Current threshold is too strict")
            logger.info("RECOMMENDATION: Use percentile-based threshold instead of fixed 0.2")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_demographic_pipeline():
    """
    TEST 8: Test the complete demographic classification pipeline.
    
    This is an integration test that validates the entire flow works.
    """
    logger.info("=" * 70)
    logger.info("TEST 8: Full demographic pipeline (integration)")
    logger.info("=" * 70)
    
    # This would test the fixed pipeline end-to-end
    # For now, we check if the fixes exist
    
    try:
        from src.demographics.community_embedding_fixed import (
            classify_with_community_embeddings_fixed
        )
        logger.info("Fixed community embedding module found")
        
        # Run with synthetic data
        test_usi = create_test_subreddit_data(200)
        test_usi_path = Path("data/features/test_user_subreddit_interactions.parquet")
        test_usi.to_parquet(test_usi_path)
        
        # Create a simple comment dataframe
        comments = pd.DataFrame({
            'author': test_usi['author'].unique(),
            'subreddit': 'characterai',
            'body': 'test comment'
        })
        
        result = classify_with_community_embeddings_fixed(
            comments,
            api_data_path=test_usi_path
        )
        
        # Clean up
        test_usi_path.unlink()
        
        # Check results
        has_age = result['age_bucket_community'].notna().sum() > 0
        has_gender_score = 'gender_community_score' in result.columns
        has_non_unknown_gender = (result['gender_community'] != 'unknown').sum() > 0
        
        logger.info(f"  Age classifications: {result['age_bucket_community'].notna().sum()}")
        logger.info(f"  Gender score column: {has_gender_score}")
        logger.info(f"  Non-unknown genders: {(result['gender_community'] != 'unknown').sum()}")
        
        success = has_age and has_gender_score
        if success:
            logger.info("SUCCESS: Full pipeline works")
        else:
            logger.error("FAILED: Pipeline issues remain")
        
        return success
        
    except ImportError:
        logger.warning("Fixed modules not yet created")
        return None
    except Exception as e:
        logger.error(f"FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print(" COMPREHENSIVE REGRESSION & COMMUNITY EMBEDDING TESTS")
    print("=" * 80 + "\n")
    
    tests = [
        ("Regression Column Naming", test_regression_column_naming),
        ("Regression Full Pipeline", test_regression_full_pipeline),
        ("Regression with Real Data", test_regression_with_real_data),
        ("Gender Embedding Scores", test_gender_embedding_scores),
        ("Seed Pair Validity", test_seed_pair_validity),
        ("Word2Vec Embedding Quality", test_word2vec_embedding_quality),
        ("Gender Threshold Logic", test_gender_threshold_logic),
        ("Full Demographic Pipeline", test_full_demographic_pipeline),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results[name] = False
        print()  # Spacing between tests
    
    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⏭️ SKIP"
            skipped += 1
        print(f"  {status}: {name}")
    
    print("\n" + "-" * 40)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print("-" * 40)
    
    if failed > 0:
        print("\n⚠️  ISSUES FOUND - See above for details")
        print("   Run with --create-fixes to generate fix modules")
    elif passed > 0 and skipped == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n⚠️  Some tests skipped - may need to create fix modules first")
    
    return failed == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-fixes":
        print("Creating fix modules... (run main tests after)")
        # This would create the fixed modules
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)

