"""
Test script for Arctic Shift API integration.

This script tests the API connection and data retrieval.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.arctic_shift import ArcticShiftClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api():
    """Test Arctic Shift API connection."""
    client = ArcticShiftClient()
    
    # Test with a small, known subreddit
    logger.info("Testing Arctic Shift API...")
    
    # Test 1: Simple comment fetch from CharacterAI (known to work)
    logger.info("\nTest 1: Fetching recent comments from r/CharacterAI")
    try:
        comments = client.fetch_comments(
            subreddit="CharacterAI",
            after_utc=1700000000,  # Around Dec 2023
            before_utc=1735689600,  # Dec 2024
            limit=10
        )
        logger.info(f"✓ Successfully fetched {len(comments)} comments")
        if comments:
            logger.info(f"Sample comment ID: {comments[0].get('id')}")
            logger.info(f"Sample comment author: {comments[0].get('author')}")
    except Exception as e:
        logger.error(f"✗ Test 1 failed: {e}")
        return False
    
    # Test 2: Try Replika subreddit
    logger.info("\nTest 2: Fetching recent comments from r/Replika")
    try:
        comments = client.fetch_comments(
            subreddit="Replika",
            after_utc=1700000000,
            before_utc=1735689600,
            limit=10
        )
        logger.info(f"✓ Successfully fetched {len(comments)} comments from r/Replika")
        if comments:
            logger.info(f"Sample comment ID: {comments[0].get('id')}")
    except Exception as e:
        logger.error(f"✗ Test 2 failed: {e}")
        logger.error("This might indicate the subreddit has no data in this time range")
    
    logger.info("\n" + "="*70)
    logger.info("API test complete!")
    return True

if __name__ == "__main__":
    test_api()

