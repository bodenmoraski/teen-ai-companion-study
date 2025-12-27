"""
Phase 1: Data Collection & Preprocessing Script

This script collects Reddit comments from multiple subreddits and preprocesses them.
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.arctic_shift import ArcticShiftClient, collect_all_subreddits
from src.data_collection.preprocess import preprocess_all_files
from src.utils.config import (
    TARGET_SUBREDDITS,
    COLLECTION_START_UTC,
    COLLECTION_END_UTC,
    DATA_RAW,
    DATA_PROCESSED,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def verify_existing_data():
    """Verify existing CharacterAI data."""
    charai_file = DATA_RAW / "characterai_comments.jsonl"
    
    if not charai_file.exists():
        logger.error(f"CharacterAI data file not found: {charai_file}")
        return False
    
    # Count lines
    count = 0
    try:
        with open(charai_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        logger.info(f"Verified CharacterAI data: {count} comments found")
        return True
    except Exception as e:
        logger.error(f"Error verifying CharacterAI data: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("Phase 1: Data Collection & Preprocessing")
    logger.info("=" * 70)
    
    # Step 1: Verify existing data
    logger.info("\nStep 1: Verifying existing CharacterAI data")
    if not verify_existing_data():
        logger.warning("CharacterAI data verification failed, but continuing...")
    
    # Step 2: Collect additional subreddits (capped at reasonable sample size)
    logger.info("\nStep 2: Collecting additional subreddits (capped samples)")
    
    # Reasonable sample size per subreddit (10k-20k comments is plenty for analysis)
    MAX_COMMENTS_PER_SUBREDDIT = 15000
    
    subreddits_to_collect = [
        s for s in TARGET_SUBREDDITS 
        if s.lower() not in ["characterai", "replika"]  # Skip if we already have data
    ]
    
    # Check for existing Replika data
    replika_files = list(DATA_RAW.glob("*replika*.jsonl"))
    if replika_files:
        logger.info(f"Found existing Replika data: {replika_files}")
        subreddits_to_collect = [s for s in subreddits_to_collect if "replika" not in s.lower()]
    
    if subreddits_to_collect:
        logger.info(f"Collecting capped samples from: {subreddits_to_collect}")
        logger.info(f"Target: ~{MAX_COMMENTS_PER_SUBREDDIT} comments per subreddit")
        
        client = ArcticShiftClient()
        results = {}
        
        for subreddit in subreddits_to_collect:
            output_path = DATA_RAW / f"{subreddit.lower()}_comments.jsonl"
            
            if output_path.exists():
                logger.info(f"Skipping r/{subreddit} - file already exists")
                with open(output_path, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                results[subreddit] = count
                continue
            
            try:
                # Collect capped sample
                logger.info(f"Collecting r/{subreddit} (capped at {MAX_COMMENTS_PER_SUBREDDIT} comments)...")
                count = client.collect_subreddit(
                    subreddit=subreddit,
                    start_utc=COLLECTION_START_UTC,
                    end_utc=COLLECTION_END_UTC,
                    output_path=output_path,
                    batch_size=100,
                    max_comments=MAX_COMMENTS_PER_SUBREDDIT  # Add this parameter
                )
                results[subreddit] = count
            except Exception as e:
                logger.error(f"Failed to collect r/{subreddit}: {e}")
                results[subreddit] = 0
        
        logger.info("\nCollection results:")
        for subreddit, count in results.items():
            logger.info(f"  r/{subreddit}: {count} comments")
    else:
        logger.info("No additional subreddits to collect (already have data or none specified)")
    
    # Step 3: Preprocess all files
    logger.info("\nStep 3: Preprocessing all data files")
    df = preprocess_all_files(
        input_dir=DATA_RAW,
        output_path=DATA_PROCESSED / "all_comments.parquet"
    )
    
    if not df.empty:
        logger.info(f"\nFinal dataset statistics:")
        logger.info(f"  Total comments: {len(df)}")
        logger.info(f"  Unique authors: {df['author'].nunique()}")
        logger.info(f"  Subreddits: {df['subreddit'].unique().tolist()}")
        logger.info(f"  Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
        
        # Generate collection statistics report
        stats_path = DATA_PROCESSED / "collection_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("Data Collection Statistics\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total comments: {len(df)}\n")
            f.write(f"Unique authors: {df['author'].nunique()}\n")
            f.write(f"Subreddits: {', '.join(sorted(df['subreddit'].unique()))}\n\n")
            f.write("Comments by subreddit:\n")
            for subreddit, count in df['subreddit'].value_counts().items():
                f.write(f"  r/{subreddit}: {count}\n")
            f.write("\nComments by author (top 10):\n")
            for author, count in df['author'].value_counts().head(10).items():
                f.write(f"  {author}: {count}\n")
        
        logger.info(f"Collection statistics saved to {stats_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1 Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

