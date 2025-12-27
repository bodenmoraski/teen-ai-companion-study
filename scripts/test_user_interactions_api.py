"""Test Arctic Shift user interactions API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.arctic_shift import collect_user_subreddit_interactions, ArcticShiftClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api():
    """Test the user interactions API."""
    logger.info("Testing Arctic Shift User Interactions API")
    logger.info("=" * 70)
    
    # Test with a known active user
    test_users = ["spez", "AutoModerator"]  # Known active users
    
    for user in test_users:
        logger.info(f"\nTesting with user: {user}")
        try:
            interactions = collect_user_subreddit_interactions(
                author=user,
                min_count=1,
                limit=10  # Just get 10 for testing
            )
            
            logger.info(f"✓ Successfully fetched interactions for {user}")
            logger.info(f"  Found {len(interactions)} subreddits")
            
            if interactions:
                logger.info("  Sample subreddits:")
                for item in interactions[:5]:
                    if isinstance(item, dict):
                        sub = item.get('subreddit', 'N/A')
                        count = item.get('count', item.get('interactions', 'N/A'))
                        logger.info(f"    r/{sub}: {count} interactions")
                    else:
                        logger.info(f"    {item}")
            
        except Exception as e:
            logger.error(f"✗ Failed for {user}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_api()

