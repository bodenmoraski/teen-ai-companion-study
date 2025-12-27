"""
Arctic Shift API wrapper for collecting Reddit comments.

This module provides functionality to collect Reddit comments from various
subreddits using the Arctic Shift API (https://arctic-shift.photon-reddit.com).
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from tqdm import tqdm
import pandas as pd

from ..utils.config import (
    ARCTIC_SHIFT_BASE_URL,
    ARCTIC_SHIFT_RATE_LIMIT_SECONDS,
    DATA_RAW,
)

logger = logging.getLogger(__name__)


class ArcticShiftClient:
    """Client for interacting with the Arctic Shift API."""
    
    def __init__(
        self,
        base_url: str = ARCTIC_SHIFT_BASE_URL,
        rate_limit: float = ARCTIC_SHIFT_RATE_LIMIT_SECONDS
    ):
        """
        Initialize the Arctic Shift API client.
        
        Args:
            base_url: Base URL for the Arctic Shift API
            rate_limit: Seconds to wait between requests
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Teen-AI-Companion-Research/1.0 (Research Project)'
        })
    
    def fetch_comments(
        self,
        subreddit: str,
        after_utc: Optional[int] = None,
        before_utc: Optional[int] = None,
        limit: int = 100,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments from a subreddit.
        
        Args:
            subreddit: Name of the subreddit (without r/)
            after_utc: Start timestamp (Unix time in seconds)
            before_utc: End timestamp (Unix time in seconds)
            limit: Maximum number of comments per request (1-100)
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of comment dictionaries
        """
        endpoint = f"{self.base_url}/api/comments/search"
        
        # Validate limit
        limit = max(1, min(100, limit))  # API allows 1-100
        
        params = {
            "subreddit": subreddit,
            "limit": limit,
            "sort": "asc"  # Sort by created_utc ascending
        }
        
        # Convert timestamps to epoch seconds (API accepts epoch seconds)
        if after_utc:
            params["after"] = int(after_utc)
        if before_utc:
            params["before"] = int(before_utc)
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Requesting: {endpoint} with params: {params}")
                response = self.session.get(endpoint, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                # API returns data directly as a list, or wrapped in a data field
                if isinstance(data, list):
                    comments = data
                else:
                    comments = data.get("data", [])
                
                logger.info(
                    f"Fetched {len(comments)} comments from r/{subreddit} "
                    f"(attempt {attempt + 1})"
                )
                
                time.sleep(self.rate_limit)
                return comments
                
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed for r/{subreddit} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch from r/{subreddit} after {max_retries} attempts")
                    raise
        
        return []
    
    def collect_subreddit(
        self,
        subreddit: str,
        start_utc: int,
        end_utc: int,
        output_path: Path,
        batch_size: int = 100,
        max_comments: Optional[int] = None
    ) -> int:
        """
        Collect comments from a subreddit within a time range.
        
        Args:
            subreddit: Name of the subreddit (without r/)
            start_utc: Start timestamp (Unix time)
            end_utc: End timestamp (Unix time)
            output_path: Path to save JSONL file
            batch_size: Number of comments per batch
            max_comments: Maximum number of comments to collect (None = no limit)
            
        Returns:
            Total number of comments collected
        """
        logger.info(f"Starting collection for r/{subreddit}")
        logger.info(f"Time range: {datetime.fromtimestamp(start_utc)} to {datetime.fromtimestamp(end_utc)}")
        
        all_comments = []
        current_utc = start_utc
        total_collected = 0
        
        # Use a set to track IDs and avoid duplicates
        seen_ids = set()
        
        # Track consecutive empty results to avoid infinite loops
        empty_count = 0
        max_empty = 3
        
        with tqdm(desc=f"Collecting r/{subreddit}") as pbar:
            while current_utc < end_utc and empty_count < max_empty:
                try:
                    comments = self.fetch_comments(
                        subreddit=subreddit,
                        after_utc=current_utc,
                        before_utc=end_utc,
                        limit=batch_size
                    )
                    
                    if not comments:
                        empty_count += 1
                        logger.info(f"No comments returned (empty count: {empty_count}/{max_empty})")
                        if empty_count >= max_empty:
                            logger.info(f"No more comments found for r/{subreddit}")
                            break
                        # Small increment to advance
                        current_utc += 86400  # Advance by 1 day
                        continue
                    
                    empty_count = 0  # Reset empty counter
                    
                    # Filter duplicates and update timestamp
                    new_comments = []
                    max_timestamp = current_utc
                    for comment in comments:
                        comment_id = comment.get("id")
                        if comment_id and comment_id not in seen_ids:
                            seen_ids.add(comment_id)
                            new_comments.append(comment)
                            # Track maximum timestamp for pagination
                            comment_time = comment.get("created_utc", 0)
                            if isinstance(comment_time, (int, float)):
                                max_timestamp = max(max_timestamp, int(comment_time))
                    
                    all_comments.extend(new_comments)
                    total_collected += len(new_comments)
                    pbar.update(len(new_comments))
                    
                    # Check if we've reached the max_comments limit
                    if max_comments and total_collected >= max_comments:
                        logger.info(f"Reached max_comments limit ({max_comments}) for r/{subreddit}")
                        # Trim to exact limit
                        all_comments = all_comments[:max_comments]
                        total_collected = len(all_comments)
                        break
                    
                    # Update current_utc to the max timestamp + 1 second for next query
                    if new_comments:
                        current_utc = max_timestamp + 1
                    else:
                        # If all were duplicates, advance timestamp
                        current_utc += 86400
                    
                    # If we got fewer than batch_size, we might be near the end
                    if len(comments) < batch_size:
                        logger.info(f"Received {len(comments)} comments (less than batch size), checking for more...")
                        # Don't break immediately - might be more data
                        
                except Exception as e:
                    logger.error(f"Error collecting from r/{subreddit}: {e}")
                    logger.exception("Full error details:")
                    empty_count += 1
                    if empty_count >= max_empty:
                        break
                    current_utc += 86400  # Advance on error
        
        # Save to JSONL
        if all_comments:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for comment in all_comments:
                    f.write(json.dumps(comment, ensure_ascii=False) + '\n')
            
            logger.info(
                f"Saved {total_collected} comments from r/{subreddit} to {output_path}"
            )
        
        return total_collected


def collect_all_subreddits(
    subreddits: List[str],
    start_utc: int,
    end_utc: int,
    output_dir: Path = DATA_RAW
) -> Dict[str, int]:
    """
    Collect comments from multiple subreddits.
    
    Args:
        subreddits: List of subreddit names (without r/)
        start_utc: Start timestamp (Unix time)
        end_utc: End timestamp (Unix time)
        output_dir: Directory to save JSONL files
        
    Returns:
        Dictionary mapping subreddit names to comment counts
    """
    client = ArcticShiftClient()
    results = {}
    
    for subreddit in subreddits:
        output_path = output_dir / f"{subreddit.lower()}_comments.jsonl"
        
        # Skip if file already exists
        if output_path.exists():
            logger.info(f"Skipping r/{subreddit} - file already exists: {output_path}")
            # Count existing comments
            with open(output_path, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            results[subreddit] = count
            continue
        
        try:
            count = client.collect_subreddit(
                subreddit=subreddit,
                start_utc=start_utc,
                end_utc=end_utc,
                output_path=output_path
            )
            results[subreddit] = count
        except Exception as e:
            logger.error(f"Failed to collect r/{subreddit}: {e}")
            results[subreddit] = 0
    
    return results


def collect_user_subreddit_interactions(
    author: str,
    after_utc: Optional[int] = None,
    before_utc: Optional[int] = None,
    min_count: int = 1,
    limit: Optional[int] = None,
    client: Optional[ArcticShiftClient] = None
) -> List[Dict[str, Any]]:
    """
    Collect subreddit participation data for a user.
    
    Uses Arctic Shift API: /api/users/interactions/subreddits
    
    Args:
        author: Reddit username
        after_utc: Start timestamp (optional)
        before_utc: End timestamp (optional)
        min_count: Minimum interactions per subreddit
        limit: Maximum number of subreddits to return (None = no limit)
        client: Optional ArcticShiftClient instance
        
    Returns:
        List of dicts with subreddit and count
    """
    if client is None:
        client = ArcticShiftClient()
    endpoint = f"{client.base_url}/api/users/interactions/subreddits"
    
    params = {
        "author": author,
        "min_count": min_count
    }
    
    if after_utc:
        params["after"] = int(after_utc)
    if before_utc:
        params["before"] = int(before_utc)
    if limit:
        params["limit"] = limit
    else:
        params["limit"] = ""  # Empty string means no limit per API docs
    
    try:
        response = client.session.get(endpoint, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # API returns list of {subreddit: count} or wrapped
        if isinstance(data, list):
            # Convert to list of dicts if needed
            result = []
            for item in data:
                if isinstance(item, dict):
                    result.append(item)
                elif isinstance(item, str):
                    result.append({"subreddit": item, "count": 1})
            return result
        elif isinstance(data, dict):
            if "data" in data:
                return data["data"]
            elif "subreddit" in data:  # Single result
                return [data]
        return []
            
    except Exception as e:
        logger.warning(f"Failed to get subreddit interactions for {author}: {e}")
        return []


def collect_user_subreddits_batch(
    authors: List[str],
    output_path: Path,
    batch_size: int = 100,
    rate_limit: float = 1.0,
    after_utc: Optional[int] = None,
    before_utc: Optional[int] = None
) -> pd.DataFrame:
    """
    Collect subreddit participation for multiple users.
    
    Args:
        authors: List of usernames
        output_path: Path to save results
        batch_size: Process in batches
        rate_limit: Seconds between requests
        
    Returns:
        DataFrame with user subreddit participation
    """
    logger.info(f"Collecting subreddit interactions for {len(authors)} users")
    
    client = ArcticShiftClient()
    all_results = []
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing data if file exists (for resuming)
    processed_authors = set()
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            processed_authors = set(existing_df['author'].unique())
            all_results = existing_df.to_dict('records')
            logger.info(f"Resuming: Found {len(processed_authors)} already processed users")
        except Exception as e:
            logger.warning(f"Could not load existing file: {e}")
    
    for i, author in enumerate(authors, 1):
        # Skip if already processed
        if author in processed_authors:
            continue
            
        try:
            interactions = collect_user_subreddit_interactions(
                author=author,
                after_utc=after_utc,
                before_utc=before_utc,
                min_count=1,
                limit=None,
                client=client
            )
            
            for item in interactions:
                if isinstance(item, dict):
                    subreddit = item.get('subreddit', '')
                    count = item.get('count', item.get('interactions', 1))
                    if subreddit:
                        all_results.append({
                            'author': author,
                            'subreddit': subreddit.lower(),  # Normalize
                            'count': int(count) if isinstance(count, (int, float)) else 1
                        })
            
            # Save incrementally every batch (MERGE with existing, don't overwrite!)
            if i % batch_size == 0:
                logger.info(f"Processed {i}/{len(authors)} users, collected {len(all_results)} interactions so far")
                # Save intermediate results - MERGE with existing data
                if all_results:
                    df_new = pd.DataFrame(all_results)
                    
                    # Load existing data if file exists
                    if output_path.exists():
                        try:
                            df_existing = pd.read_parquet(output_path)
                            # Merge, removing duplicates
                            df_merged = pd.concat([df_existing, df_new])
                            df_merged = df_merged.drop_duplicates(subset=['author', 'subreddit'], keep='first')
                            df_merged.to_parquet(output_path, index=False)
                            logger.info(f"  Saved intermediate results (merged): {len(df_merged)} total interactions, {df_merged['author'].nunique()} users")
                        except Exception as e:
                            logger.warning(f"  Could not merge with existing file: {e}, saving new data only")
                            df_new.to_parquet(output_path, index=False)
                            logger.info(f"  Saved intermediate results: {len(df_new)} interactions")
                    else:
                        df_new.to_parquet(output_path, index=False)
                        logger.info(f"  Saved intermediate results: {len(df_new)} interactions")
                time.sleep(rate_limit)
            else:
                time.sleep(rate_limit)
                
        except Exception as e:
            logger.warning(f"Error processing user {author}: {e}")
            continue
    
    # Final save - MERGE with existing data
    if all_results:
        df_new = pd.DataFrame(all_results)
        
        # Merge with existing data if file exists
        if output_path.exists():
            try:
                df_existing = pd.read_parquet(output_path)
                df_final = pd.concat([df_existing, df_new])
                df_final = df_final.drop_duplicates(subset=['author', 'subreddit'], keep='first')
                df_final.to_parquet(output_path, index=False)
                logger.info(f"Final save (merged): {len(df_final)} user-subreddit interactions to {output_path}")
                logger.info(f"  Unique users: {df_final['author'].nunique()}")
                logger.info(f"  Unique subreddits: {df_final['subreddit'].nunique()}")
                return df_final
            except Exception as e:
                logger.warning(f"Could not merge with existing file: {e}, saving new data only")
                df_new.to_parquet(output_path, index=False)
                logger.info(f"Final save: {len(df_new)} user-subreddit interactions to {output_path}")
                return df_new
        else:
            df_new.to_parquet(output_path, index=False)
            logger.info(f"Final save: {len(df_new)} user-subreddit interactions to {output_path}")
            logger.info(f"  Unique users: {df_new['author'].nunique()}")
            logger.info(f"  Unique subreddits: {df_new['subreddit'].nunique()}")
            return df_new
    else:
        logger.warning("No interactions collected")
        return pd.DataFrame(columns=['author', 'subreddit', 'count'])

