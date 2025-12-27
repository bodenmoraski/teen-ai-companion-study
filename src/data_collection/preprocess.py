"""
Data preprocessing pipeline for Reddit comments.

This module handles cleaning, filtering, and standardizing Reddit comment data
for downstream analysis.
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
import pandas as pd
from tqdm import tqdm

from ..utils.config import (
    DATA_RAW,
    DATA_PROCESSED,
    MIN_COMMENT_LENGTH,
    MAX_COMMENT_LENGTH,
    BOT_AUTHORS,
)

logger = logging.getLogger(__name__)


def standardize_comment_schema(comment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Standardize a comment to the common schema.
    
    Expected schema:
    {
        "id": str,
        "author": str,
        "body": str,
        "created_utc": int,
        "subreddit": str,
        "link_id": str,
        "parent_id": str,
        "score": int,
        "author_flair_text": Optional[str]
    }
    
    Args:
        comment: Raw comment dictionary from API
        
    Returns:
        Standardized comment dictionary or None if invalid
    """
    try:
        standardized = {
            "id": str(comment.get("id", "")),
            "author": str(comment.get("author", "")),
            "body": str(comment.get("body", "")),
            "created_utc": int(comment.get("created_utc", 0)),
            "subreddit": str(comment.get("subreddit", "")),
            "link_id": str(comment.get("link_id", "")),
            "parent_id": str(comment.get("parent_id", "")),
            "score": int(comment.get("score", 0)),
            "author_flair_text": comment.get("author_flair_text") or None
        }
        
        # Validate required fields
        if not standardized["id"] or not standardized["author"]:
            return None
            
        return standardized
        
    except (ValueError, KeyError, TypeError) as e:
        logger.debug(f"Failed to standardize comment: {e}")
        return None


def filter_bots(comment: Dict[str, Any], bot_authors: Set[str]) -> bool:
    """
    Check if a comment should be filtered (bot/deleted).
    
    Args:
        comment: Comment dictionary
        bot_authors: Set of bot author names to filter
        
    Returns:
        True if comment should be kept, False if filtered
    """
    author = comment.get("author", "").strip()
    
    if not author or author in bot_authors:
        return False
    
    # Filter deleted/removed
    if author in ["[deleted]", "[removed]"]:
        return False
    
    return True


def filter_text_quality(comment: Dict[str, Any]) -> bool:
    """
    Check if comment meets text quality requirements.
    
    Args:
        comment: Comment dictionary
        
    Returns:
        True if comment meets quality requirements
    """
    body = comment.get("body", "")
    
    if not body or not isinstance(body, str):
        return False
    
    # Check length
    body_length = len(body.strip())
    if body_length < MIN_COMMENT_LENGTH or body_length > MAX_COMMENT_LENGTH:
        return False
    
    # Filter out URLs only (no actual text content)
    url_pattern = r'https?://\S+'
    text_without_urls = re.sub(url_pattern, '', body).strip()
    if len(text_without_urls) < MIN_COMMENT_LENGTH:
        return False
    
    return True


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of comment dictionaries
    """
    comments = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.strip():
                    comments.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
    
    return comments


def preprocess_file(
    input_path: Path,
    output_path: Path,
    bot_authors: Set[str] = None,
    deduplicate: bool = True
) -> Dict[str, int]:
    """
    Preprocess a single JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output parquet file
        bot_authors: Set of bot author names
        deduplicate: Whether to remove duplicate IDs
        
    Returns:
        Dictionary with preprocessing statistics
    """
    if bot_authors is None:
        bot_authors = set(BOT_AUTHORS)
    
    logger.info(f"Preprocessing {input_path}")
    
    # Load comments
    comments = load_jsonl(input_path)
    initial_count = len(comments)
    logger.info(f"Loaded {initial_count} comments")
    
    stats = {
        "initial_count": initial_count,
        "after_standardization": 0,
        "after_bot_filter": 0,
        "after_text_filter": 0,
        "after_deduplication": 0,
        "final_count": 0
    }
    
    # Standardize schema
    standardized = []
    seen_ids = set()
    
    for comment in tqdm(comments, desc="Standardizing", leave=False):
        std_comment = standardize_comment_schema(comment)
        if std_comment:
            comment_id = std_comment["id"]
            
            # Deduplication
            if deduplicate and comment_id in seen_ids:
                continue
            
            seen_ids.add(comment_id)
            standardized.append(std_comment)
    
    stats["after_standardization"] = len(standardized)
    logger.info(f"After standardization: {len(standardized)} comments")
    
    # Filter bots
    filtered = [c for c in standardized if filter_bots(c, bot_authors)]
    stats["after_bot_filter"] = len(filtered)
    logger.info(f"After bot filter: {len(filtered)} comments")
    
    # Filter text quality
    quality_filtered = [c for c in filtered if filter_text_quality(c)]
    stats["after_text_filter"] = len(quality_filtered)
    logger.info(f"After text quality filter: {len(quality_filtered)} comments")
    
    # Final deduplication (by ID)
    if deduplicate:
        final_seen = set()
        final_comments = []
        for comment in quality_filtered:
            comment_id = comment["id"]
            if comment_id not in final_seen:
                final_seen.add(comment_id)
                final_comments.append(comment)
        quality_filtered = final_comments
    
    stats["after_deduplication"] = len(quality_filtered)
    stats["final_count"] = len(quality_filtered)
    
    # Convert to DataFrame and save
    if quality_filtered:
        df = pd.DataFrame(quality_filtered)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Saved {len(quality_filtered)} comments to {output_path}")
    
    return stats


def preprocess_all_files(
    input_dir: Path = DATA_RAW,
    output_path: Path = None,
    pattern: str = "*_comments.jsonl"
) -> pd.DataFrame:
    """
    Preprocess all JSONL files in a directory and combine them.
    
    Args:
        input_dir: Directory containing JSONL files
        output_path: Path to save combined parquet file
        pattern: Glob pattern for input files
        
    Returns:
        Combined DataFrame with all preprocessed comments
    """
    if output_path is None:
        output_path = DATA_PROCESSED / "all_comments.parquet"
    
    logger.info(f"Preprocessing all files in {input_dir}")
    
    jsonl_files = list(input_dir.glob(pattern))
    logger.info(f"Found {len(jsonl_files)} JSONL files")
    
    all_stats = []
    all_dfs = []
    
    for jsonl_file in jsonl_files:
        # Create temporary output path
        temp_output = DATA_PROCESSED / f"{jsonl_file.stem}.parquet"
        
        stats = preprocess_file(jsonl_file, temp_output)
        stats["file"] = jsonl_file.name
        all_stats.append(stats)
        
        # Load the preprocessed data
        if temp_output.exists():
            df = pd.read_parquet(temp_output)
            all_dfs.append(df)
    
    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Final deduplication across files
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["id"], keep="first")
        final_count = len(combined_df)
        
        logger.info(
            f"Combined {len(all_dfs)} files: {initial_count} -> {final_count} comments "
            f"(removed {initial_count - final_count} duplicates)"
        )
        
        # Save combined file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path, index=False, engine='pyarrow')
        logger.info(f"Saved combined dataset to {output_path}")
        
        # Save statistics
        stats_df = pd.DataFrame(all_stats)
        stats_path = output_path.parent / "preprocessing_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved preprocessing statistics to {stats_path}")
        
        return combined_df
    else:
        logger.warning("No data to combine")
        return pd.DataFrame()

