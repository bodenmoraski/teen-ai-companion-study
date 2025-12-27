"""
Targeted Phase 3 & 4: Only compute what we need!

Strategy:
1. Use existing AnthroScore data (15,978 users already have it)
2. Only compute AnthroScore for missing 2,553 users (5,756 comments)
3. Merge everything together
4. Run Phase 4 with full statistics
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.analysis.anthroscore_runner import (
    compute_anthroscores,
    aggregate_to_user_level as aggregate_anthro
)
from src.analysis.bertopic_clustering import (
    extract_topic_features,
    aggregate_topics_to_user_level
)
from src.analysis.emotion_analysis import (
    extract_emotion_features,
    aggregate_emotions_to_user_level
)
from src.statistical.descriptive_stats import (
    generate_descriptive_statistics,
    generate_correlation_table
)
from src.statistical.regression_models import (
    run_rq2_regression,
    generate_regression_tables
)
from src.statistical.visualization import (
    plot_age_distribution,
    plot_anthroscore_by_demographics,
    plot_topic_distribution,
    plot_emotion_distribution
)
from src.utils.config import (
    DATA_PROCESSED, 
    DATA_FEATURES,
    RESULTS_TABLES,
    RESULTS_FIGURES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('targeted_phase3_phase4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("TARGETED Phase 3 & 4: Only Compute What We Need!")
    logger.info("=" * 70)
    
    # Load demographics (what we need)
    demo_path = DATA_FEATURES / "demographics.parquet"
    if not demo_path.exists():
        logger.error(f"Demographics not found: {demo_path}")
        return
    
    demo = pd.read_parquet(demo_path)
    age_classified = demo[demo['age_bucket'].notna()]
    logger.info(f"\nAge-classified users: {len(age_classified):,}")
    
    # Check existing AnthroScore
    anthro_path = DATA_FEATURES / "user_anthroscores.parquet"
    existing_anthro = None
    if anthro_path.exists():
        existing_anthro = pd.read_parquet(anthro_path)
        logger.info(f"Existing AnthroScore data: {len(existing_anthro):,} users")
        
        # Check overlap
        age_users = set(age_classified['author'].unique())
        anthro_users = set(existing_anthro['author'].unique())
        overlap = age_users & anthro_users
        missing = age_users - anthro_users
        
        logger.info(f"  Overlap: {len(overlap):,} users ({100*len(overlap)/len(age_users):.1f}%)")
        logger.info(f"  Missing: {len(missing):,} users ({100*len(missing)/len(age_users):.1f}%)")
    
    # Load comments
    comments_path = DATA_PROCESSED / "all_comments.parquet"
    if not comments_path.exists():
        logger.error(f"Comments not found: {comments_path}")
        return
    
    comments = pd.read_parquet(comments_path)
    logger.info(f"\nTotal comments: {len(comments):,}")
    
    # ===================================================================
    # Step 1: AnthroScore (targeted - only missing users)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: AnthroScore (Targeted)")
    logger.info("=" * 70)
    
    if existing_anthro is not None and len(missing) > 0:
        # Only compute for missing users
        logger.info(f"Computing AnthroScore for {len(missing):,} missing users only")
        missing_comments = comments[comments['author'].isin(missing)]
        logger.info(f"  Comments to process: {len(missing_comments):,} (not 283k!)")
        
        if len(missing_comments) > 0:
            try:
                missing_with_scores = compute_anthroscores(missing_comments, batch_size=16)
                missing_anthro = aggregate_anthro(missing_with_scores)
                
                # Combine with existing
                all_anthro = pd.concat([existing_anthro, missing_anthro], ignore_index=True)
                all_anthro = all_anthro.drop_duplicates(subset=['author'], keep='first')
                
                # Save
                all_anthro.to_parquet(anthro_path, index=False)
                logger.info(f"✅ Combined AnthroScore: {len(all_anthro):,} users")
                
            except Exception as e:
                logger.error(f"Error computing AnthroScore: {e}", exc_info=True)
                all_anthro = existing_anthro
        else:
            all_anthro = existing_anthro
            logger.info("✅ Using existing AnthroScore (no missing users)")
    elif existing_anthro is not None:
        all_anthro = existing_anthro
        logger.info("✅ Using existing AnthroScore (all users covered)")
    else:
        # No existing data - compute for all age-classified users
        logger.info(f"Computing AnthroScore for {len(age_classified):,} age-classified users")
        age_comments = comments[comments['author'].isin(age_classified['author'].unique())]
        logger.info(f"  Comments to process: {len(age_comments):,}")
        
        try:
            comments_with_scores = compute_anthroscores(age_comments, batch_size=16)
            all_anthro = aggregate_anthro(comments_with_scores)
            all_anthro.to_parquet(anthro_path, index=False)
            logger.info(f"✅ Computed AnthroScore: {len(all_anthro):,} users")
        except Exception as e:
            logger.error(f"Error computing AnthroScore: {e}", exc_info=True)
            return
    
    # ===================================================================
    # Step 2: BERTopic (can use existing or compute for age-classified)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: BERTopic Clustering")
    logger.info("=" * 70)
    
    topics_path = DATA_FEATURES / "user_topics.parquet"
    if topics_path.exists():
        user_topics = pd.read_parquet(topics_path)
        logger.info(f"✅ Using existing topics: {len(user_topics):,} users")
    else:
        logger.info("Computing topics for age-classified users...")
        age_comments = comments[comments['author'].isin(age_classified['author'].unique())]
        try:
            comments_with_topics = extract_topic_features(age_comments, min_topic_size=50)
            user_topics = aggregate_topics_to_user_level(comments_with_topics)
            user_topics.to_parquet(topics_path, index=False)
            logger.info(f"✅ Computed topics: {len(user_topics):,} users")
        except Exception as e:
            logger.error(f"Error computing topics: {e}", exc_info=True)
            user_topics = None
    
    # ===================================================================
    # Step 3: Emotions (can use existing or compute for age-classified)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Emotion Classification")
    logger.info("=" * 70)
    
    emotions_path = DATA_FEATURES / "user_emotions.parquet"
    if emotions_path.exists():
        user_emotions = pd.read_parquet(emotions_path)
        logger.info(f"✅ Using existing emotions: {len(user_emotions):,} users")
    else:
        logger.info("Computing emotions for age-classified users...")
        age_comments = comments[comments['author'].isin(age_classified['author'].unique())]
        try:
            comments_with_emotions = extract_emotion_features(age_comments, batch_size=32)
            user_emotions = aggregate_emotions_to_user_level(comments_with_emotions)
            user_emotions.to_parquet(emotions_path, index=False)
            logger.info(f"✅ Computed emotions: {len(user_emotions):,} users")
        except Exception as e:
            logger.error(f"Error computing emotions: {e}", exc_info=True)
            user_emotions = None
    
    # ===================================================================
    # Step 4: Merge Everything
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Merging All Features")
    logger.info("=" * 70)
    
    merged_df = demo.copy()
    logger.info(f"Starting with {len(merged_df):,} users from demographics")
    
    # Merge AnthroScore
    merged_df = merged_df.merge(all_anthro, on='author', how='left')
    logger.info(f"  After AnthroScore: {merged_df['anthroscore_mean'].notna().sum():,} users")
    
    # Merge topics
    if user_topics is not None:
        merged_df = merged_df.merge(user_topics, on='author', how='left')
        logger.info(f"  After topics: {merged_df['dominant_topic'].notna().sum():,} users")
    
    # Merge emotions
    if user_emotions is not None:
        merged_df = merged_df.merge(user_emotions, on='author', how='left')
        logger.info(f"  After emotions: {merged_df['dominant_emotion'].notna().sum():,} users")
    
    # Save merged dataset
    merged_output = DATA_FEATURES / "full_merged_dataset.parquet"
    merged_df.to_parquet(merged_output, index=False)
    logger.info(f"\n✅ Saved merged dataset: {len(merged_df):,} users, {len(merged_df.columns)} features")
    
    # ===================================================================
    # Step 5: Phase 4 - Statistical Analysis
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Statistical Analysis (NeurIPS-Level)")
    logger.info("=" * 70)
    
    # Descriptive statistics
    stats_path = RESULTS_TABLES / "descriptive_statistics.txt"
    generate_descriptive_statistics(merged_df, stats_path)
    logger.info(f"✅ Descriptive statistics saved")
    
    corr_path = RESULTS_TABLES / "correlation_matrix.csv"
    generate_correlation_table(merged_df, corr_path)
    logger.info(f"✅ Correlation matrix saved")
    
    # Regression analysis
    regression_results = run_rq2_regression(merged_df)
    if "error" not in regression_results:
        regression_path = RESULTS_TABLES / "regression_results.txt"
        generate_regression_tables(regression_results, regression_path)
        logger.info(f"✅ Regression results saved (NeurIPS-level statistics)")
    
    # Figures
    plot_age_distribution(merged_df, RESULTS_FIGURES / "age_distribution.png")
    plot_anthroscore_by_demographics(merged_df, RESULTS_FIGURES / "anthroscore_by_demographics.png")
    plot_topic_distribution(merged_df, RESULTS_FIGURES / "topic_distribution.png")
    plot_emotion_distribution(merged_df, RESULTS_FIGURES / "emotion_distribution.png")
    logger.info(f"✅ All figures generated")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total users: {len(merged_df):,}")
    logger.info(f"Age classified: {merged_df['age_bucket'].notna().sum():,}")
    logger.info(f"AnthroScore: {merged_df['anthroscore_mean'].notna().sum():,}")
    logger.info(f"Users with BOTH (for analysis): {(merged_df['age_bucket'].notna() & merged_df['anthroscore_mean'].notna()).sum():,}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TARGETED Phase 3 & 4 Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

