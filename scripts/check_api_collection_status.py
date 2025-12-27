"""Check API collection status."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_status():
    """Check collection status."""
    api_data_path = Path("data/features/user_subreddit_interactions.parquet")
    log_file = Path("phase2_with_api_data.log")
    
    logger.info("=" * 70)
    logger.info("API Collection Status")
    logger.info("=" * 70)
    
    if api_data_path.exists():
        try:
            df = pd.read_parquet(api_data_path)
            logger.info(f"✓ Collection in progress or complete")
            logger.info(f"  Interactions collected: {len(df):,}")
            logger.info(f"  Unique users: {df['author'].nunique():,}")
            logger.info(f"  Unique subreddits: {df['subreddit'].nunique():,}")
            
            # Show top subreddits
            logger.info(f"\n  Top 20 subreddits:")
            top_subs = df['subreddit'].value_counts().head(20)
            for sub, count in top_subs.items():
                logger.info(f"    r/{sub}: {count:,} users")
            
            # Estimate progress
            total_users = 47062  # From log
            collected_users = df['author'].nunique()
            progress = 100 * collected_users / total_users if total_users > 0 else 0
            logger.info(f"\n  Progress: {collected_users:,}/{total_users:,} users ({progress:.1f}%)")
            
            # Estimate remaining time
            if collected_users > 0:
                # Assume 1 request/second, but some users might have no data
                remaining_users = total_users - collected_users
                remaining_hours = remaining_users / 3600
                logger.info(f"  Estimated remaining: ~{remaining_hours:.1f} hours")
        except Exception as e:
            logger.error(f"Error reading file: {e}")
    else:
        logger.info("Collection not started or file not created yet")
        
        # Check log for progress
        if log_file.exists():
            logger.info(f"\nChecking log file...")
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Look for progress messages
                for line in lines[-50:]:
                    if "Processed" in line and "users" in line:
                        logger.info(f"  {line.strip()}")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    check_status()

