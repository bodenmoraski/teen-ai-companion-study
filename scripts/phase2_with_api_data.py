"""
Phase 2: Demographics with full community embeddings.

This version collects user subreddit data from Arctic Shift API first,
then runs community embeddings with the full Reddit participation data.
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_collection.arctic_shift import collect_user_subreddits_batch
from src.demographics.self_declaration import extract_self_declarations
from src.demographics.llm_classifier import classify_users_llm
from src.demographics.community_embedding import classify_with_community_embeddings
from src.demographics.ensemble_classifier import create_ensemble_classification
from src.utils.config import (
    DATA_PROCESSED,
    DATA_FEATURES,
    COLLECTION_START_UTC,
    COLLECTION_END_UTC
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_with_api_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("Phase 2: Demographics with Full Community Embeddings")
    logger.info("=" * 70)
    
    # Load processed comments
    input_path = DATA_PROCESSED / "all_comments.parquet"
    if not input_path.exists():
        logger.error(f"Processed data not found: {input_path}")
        return
    
    logger.info(f"\nLoading processed data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} comments from {df['author'].nunique()} unique authors")
    
    # Step 0: Collect user subreddit interactions (if not already done)
    api_data_path = DATA_FEATURES / "user_subreddit_interactions.parquet"
    
    # Check existing data first
    existing_users = set()
    if api_data_path.exists():
        try:
            existing_df = pd.read_parquet(api_data_path)
            existing_users = set(existing_df['author'].unique())
            logger.info(f"Found existing data: {len(existing_users)} users already collected")
        except Exception as e:
            logger.warning(f"Could not read existing file: {e}")
    
    # Only collect if we don't have enough users yet
    TARGET_TOTAL = 5000
    if len(existing_users) < TARGET_TOTAL:
        logger.info("\n" + "-" * 70)
        logger.info("Step 0: Collecting user subreddit participation (Arctic Shift API)")
        logger.info("-" * 70)
        logger.info("This collects broader Reddit participation for proper community embeddings")
        logger.info("Only collecting for users in our target subreddits")
        
        # Get users from our target subreddits only
        target_subreddits = ['characterai', 'replika', 'aicompanions', 'chai', 'soulmateai']
        df_filtered = df[df['subreddit'].str.lower().isin(target_subreddits)]
        all_authors = df_filtered['author'].unique().tolist()
        
        # Filter out already collected users
        if existing_users:
            authors = [a for a in all_authors if a not in existing_users]
            logger.info(f"Remaining users to collect: {len(authors)} (already have {len(existing_users)})")
        else:
            authors = all_authors
            logger.info(f"Found {len(authors)} unique users in target subreddits")
        
        logger.info(f"Target subreddits: {target_subreddits}")
        
        # Sample a reasonable number of users for community embeddings
        # 5,000 users total is plenty for robust embeddings
        TARGET_TOTAL = 5000
        remaining_needed = max(0, TARGET_TOTAL - len(existing_users))
        
        if len(authors) > remaining_needed and remaining_needed > 0:
            logger.info(f"Sampling {remaining_needed} more users to reach {TARGET_TOTAL} total")
            logger.info(f"  (Already have {len(existing_users)}, need {remaining_needed} more)")
            import numpy as np
            np.random.seed(42)
            authors = np.random.choice(authors, size=remaining_needed, replace=False).tolist()
        elif remaining_needed == 0:
            logger.info(f"Already have {len(existing_users)} users (target: {TARGET_TOTAL}) - skipping collection")
            authors = []
        else:
            logger.info(f"Collecting subreddit interactions for remaining {len(authors)} users")
        
        logger.info(f"Collecting subreddit interactions for {len(authors)} users")
        logger.info("This will take a while due to API rate limits (1 request/second)...")
        estimated_hours = len(authors) / 3600
        estimated_minutes = len(authors) / 60
        logger.info(f"Estimated time: ~{estimated_hours:.1f} hours (~{estimated_minutes:.0f} minutes)")
        
        interactions_df = collect_user_subreddits_batch(
            authors=authors,
            output_path=api_data_path,
            batch_size=50,
            rate_limit=1.0,
            after_utc=COLLECTION_START_UTC,
            before_utc=COLLECTION_END_UTC
        )
        
        if not interactions_df.empty:
            logger.info(f"✓ Collected {len(interactions_df)} interactions")
            logger.info(f"  Unique subreddits: {interactions_df['subreddit'].nunique()}")
    else:
        logger.info(f"\nUsing existing API data: {api_data_path}")
        interactions_df = pd.read_parquet(api_data_path)
        logger.info(f"  {len(interactions_df)} interactions, {interactions_df['subreddit'].nunique()} subreddits")
    
    # Step 1: Self-declaration extraction
    logger.info("\n" + "-" * 70)
    logger.info("Step 1: Extracting self-declarations")
    logger.info("-" * 70)
    
    self_decl_df = extract_self_declarations(df)
    
    # Save self-declarations
    output_path = DATA_FEATURES / "self_declarations.parquet"
    self_decl_df.to_parquet(output_path, index=False)
    logger.info(f"Saved self-declarations to {output_path}")
    
    # Step 2: Community embedding classification (with API data)
    logger.info("\n" + "-" * 70)
    logger.info("Step 2: Community embedding classification (with full Reddit data)")
    logger.info("-" * 70)
    
    try:
        community_embedding_df = classify_with_community_embeddings(
            df,
            min_participation=2,
            api_data_path=api_data_path if api_data_path.exists() else None
        )
        
        # Save community embeddings
        comm_output_path = DATA_FEATURES / "community_embeddings.parquet"
        community_embedding_df.to_parquet(comm_output_path, index=False)
        logger.info(f"Saved community embeddings to {comm_output_path}")
        
    except Exception as e:
        logger.error(f"Error in community embedding classification: {e}", exc_info=True)
        community_embedding_df = None
    
    # Step 3: LLM classification (for users without self-declarations)
    logger.info("\n" + "-" * 70)
    logger.info("Step 3: LLM-based age classification")
    logger.info("-" * 70)
    
    users_with_decl = set(self_decl_df[self_decl_df['age_bucket_self_declared'].notna()]['author'])
    users_to_classify = df[~df['author'].isin(users_with_decl)]['author'].unique()
    
    logger.info(f"Classifying {len(users_to_classify)} users with LLM")
    
    if len(users_to_classify) > 0:
        max_llm_users = 5000
        if len(users_to_classify) > max_llm_users:
            logger.warning(f"Limiting LLM classification to {max_llm_users} users for cost control")
            import numpy as np
            np.random.seed(42)
            users_to_classify = np.random.choice(
                users_to_classify, 
                size=max_llm_users, 
                replace=False
            )
        
        df_subset = df[df['author'].isin(users_to_classify)]
        
        llm_df = classify_users_llm(
            df_subset,
            batch_size=100,
            max_comments_per_user=20,
            rate_limit=0.5
        )
        
        # Save LLM classifications
        llm_output_path = DATA_FEATURES / "llm_classifications.parquet"
        llm_df.to_parquet(llm_output_path, index=False)
        logger.info(f"Saved LLM classifications to {llm_output_path}")
    else:
        logger.info("All users have self-declarations, skipping LLM classification")
        llm_df = None
    
    # Step 4: Create ensemble classification
    logger.info("\n" + "-" * 70)
    logger.info("Step 4: Creating ensemble classification")
    logger.info("-" * 70)
    
    ensemble_df = create_ensemble_classification(
        self_decl_df=self_decl_df,
        community_embedding_df=community_embedding_df,
        llm_df=llm_df
    )
    
    # Save final demographics
    final_output_path = DATA_FEATURES / "demographics.parquet"
    ensemble_df.to_parquet(final_output_path, index=False)
    logger.info(f"Saved final demographics to {final_output_path}")
    
    # Generate statistics
    logger.info("\n" + "-" * 70)
    logger.info("Demographics Statistics")
    logger.info("-" * 70)
    
    total_users = len(ensemble_df)
    age_classified = ensemble_df['age_bucket'].notna().sum()
    gender_classified = ensemble_df['gender'].notna().sum()
    
    logger.info(f"Total users: {total_users}")
    logger.info(f"Age classified: {age_classified} ({100*age_classified/total_users:.1f}%)")
    logger.info(f"Gender classified: {gender_classified} ({100*gender_classified/total_users:.1f}%)")
    
    if age_classified > 0:
        logger.info("\nAge distribution:")
        age_dist = ensemble_df['age_bucket'].value_counts().sort_index()
        for bucket, count in age_dist.items():
            pct = 100 * count / age_classified
            logger.info(f"  {bucket}: {count} ({pct:.1f}%)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2 Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

