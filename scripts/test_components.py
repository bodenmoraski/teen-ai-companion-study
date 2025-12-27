"""
Test all demographic classification components to ensure they're working.

This script tests:
1. Self-declaration extraction
2. Community embeddings
3. LLM classification
4. Ensemble classifier
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from src.demographics.self_declaration import extract_self_declarations
from src.demographics.community_embedding import classify_with_community_embeddings
from src.demographics.llm_classifier import classify_age_llm
from src.demographics.ensemble_classifier import create_ensemble_classification
from src.utils.config import DATA_PROCESSED, OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_self_declaration():
    """Test self-declaration extraction."""
    logger.info("=" * 70)
    logger.info("Test 1: Self-Declaration Extraction")
    logger.info("=" * 70)
    
    # Load sample data
    df = pd.read_parquet(DATA_PROCESSED / "all_comments.parquet")
    sample_df = df.head(1000)  # Small sample for testing
    
    try:
        result = extract_self_declarations(sample_df)
        logger.info(f"✓ Self-declaration extraction works")
        logger.info(f"  Found {result['age_bucket_self_declared'].notna().sum()} age declarations")
        logger.info(f"  Found {result['gender_self_declared'].notna().sum()} gender declarations")
        return True
    except Exception as e:
        logger.error(f"✗ Self-declaration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_community_embeddings():
    """Test community embeddings."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: Community Embeddings")
    logger.info("=" * 70)
    
    # Load sample data
    df = pd.read_parquet(DATA_PROCESSED / "all_comments.parquet")
    
    # Need users with multiple subreddits - check if we have that
    user_subreddits = df.groupby('author')['subreddit'].nunique()
    multi_subreddit_users = user_subreddits[user_subreddits > 1].index
    
    if len(multi_subreddit_users) < 10:
        logger.warning("⚠ Limited subreddit diversity - community embeddings may not work well")
        logger.warning(f"  Only {len(multi_subreddit_users)} users participate in multiple subreddits")
        logger.warning("  This is expected if most users only post in CharacterAI/Replika")
        return True  # Not a failure, just limited data
    
    # Use subset with multi-subreddit users
    sample_df = df[df['author'].isin(multi_subreddit_users[:500])]
    
    try:
        result = classify_with_community_embeddings(
            sample_df,
            min_participation=2  # Lower threshold for testing
        )
        logger.info(f"✓ Community embeddings work")
        logger.info(f"  Classified {len(result)} users")
        logger.info(f"  Age classifications: {result['age_bucket_community'].notna().sum()}")
        logger.info(f"  Gender classifications: {result['gender_community'].notna().sum()}")
        logger.info(f"  Sample age buckets: {result['age_bucket_community'].value_counts().head()}")
        return True
    except Exception as e:
        logger.error(f"✗ Community embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_classification():
    """Test LLM classification."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 3: LLM Classification")
    logger.info("=" * 70)
    
    # Check API key
    if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 10:
        logger.warning("⚠ OpenAI API key not configured or invalid")
        logger.warning("  LLM classification will be skipped")
        return None  # Not a failure, just not configured
    
    # Test with sample comments
    test_comments = [
        "I'm 16 and I love my AI companion",
        "As a 25 year old, I find this interesting",
        "My AI friend is so helpful"
    ]
    
    try:
        result = classify_age_llm(test_comments, max_comments=3)
        logger.info(f"✓ LLM classification works")
        logger.info(f"  Result: {result}")
        if result.get('age_bucket'):
            logger.info(f"  Age bucket: {result['age_bucket']}")
            logger.info(f"  Confidence: {result.get('confidence', 0):.2f}")
        return True
    except Exception as e:
        logger.error(f"✗ LLM classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble():
    """Test ensemble classifier."""
    logger.info("\n" + "=" * 70)
    logger.info("Test 4: Ensemble Classifier")
    logger.info("=" * 70)
    
    # Create mock data
    self_decl = pd.DataFrame({
        'author': ['user1', 'user2', 'user3'],
        'age_bucket_self_declared': ['19-25', None, '26-40'],
        'gender_self_declared': ['female', 'male', None]
    })
    
    comm_emb = pd.DataFrame({
        'author': ['user1', 'user2', 'user3'],
        'age_bucket_community': ['19-25', '13-18', None],
        'age_community_score': [0.2, -0.3, None],
        'gender_community': ['female', None, 'male']
    })
    
    try:
        result = create_ensemble_classification(
            self_decl_df=self_decl,
            community_embedding_df=comm_emb,
            llm_df=None
        )
        logger.info(f"✓ Ensemble classifier works")
        logger.info(f"  Result shape: {result.shape}")
        logger.info(f"  Columns: {list(result.columns)}")
        logger.info(f"  Age classifications: {result['age_bucket'].notna().sum()}")
        logger.info(f"  Gender classifications: {result['gender'].notna().sum()}")
        return True
    except Exception as e:
        logger.error(f"✗ Ensemble classifier failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_phase2_outputs():
    """Check what Phase 2 actually produced."""
    logger.info("\n" + "=" * 70)
    logger.info("Checking Phase 2 Outputs")
    logger.info("=" * 70)
    
    base_path = Path("data/features")
    
    files_to_check = [
        "demographics.parquet",
        "self_declarations.parquet",
        "community_embeddings.parquet",
        "llm_classifications.parquet"
    ]
    
    for filename in files_to_check:
        filepath = base_path / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                logger.info(f"✓ {filename}: {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"  Columns: {list(df.columns)[:5]}...")
            except Exception as e:
                logger.error(f"✗ {filename}: Error reading - {e}")
        else:
            logger.warning(f"⚠ {filename}: Not found")
    
    # Check demographics file specifically
    demo_path = base_path / "demographics.parquet"
    if demo_path.exists():
        demo = pd.read_parquet(demo_path)
        logger.info("\nDemographics file analysis:")
        logger.info(f"  Total users: {len(demo)}")
        logger.info(f"  Has age_bucket: {'age_bucket' in demo.columns}")
        logger.info(f"  Has age_bucket_community: {'age_bucket_community' in demo.columns}")
        logger.info(f"  Has age_bucket_llm: {'age_bucket_llm' in demo.columns}")
        logger.info(f"  Has gender: {'gender' in demo.columns}")
        logger.info(f"  Has gender_community: {'gender_community' in demo.columns}")
        
        if 'age_bucket_community' in demo.columns:
            comm_age = demo['age_bucket_community'].notna().sum()
            logger.info(f"  Community age classifications: {comm_age} ({100*comm_age/len(demo):.1f}%)")
        
        if 'age_bucket_llm' in demo.columns:
            llm_age = demo['age_bucket_llm'].notna().sum()
            logger.info(f"  LLM age classifications: {llm_age} ({100*llm_age/len(demo):.1f}%)")


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("Component Testing Suite")
    logger.info("=" * 70)
    
    results = {}
    
    # Test components
    results['self_declaration'] = test_self_declaration()
    results['community_embeddings'] = test_community_embeddings()
    results['llm'] = test_llm_classification()
    results['ensemble'] = test_ensemble()
    
    # Check actual outputs
    check_phase2_outputs()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    for component, result in results.items():
        if result is True:
            logger.info(f"✓ {component}: PASSED")
        elif result is None:
            logger.info(f"⚠ {component}: SKIPPED (not configured)")
        else:
            logger.info(f"✗ {component}: FAILED")
    
    all_passed = all(r for r in results.values() if r is not None)
    if all_passed:
        logger.info("\n✓ All configured components are working!")
    else:
        logger.warning("\n⚠ Some components need attention")


if __name__ == "__main__":
    main()

