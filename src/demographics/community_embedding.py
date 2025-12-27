"""
Community embedding-based age and gender classification.

This module implements the Toronto CSS Lab methodology for inferring
demographics from user subreddit participation patterns.
"""
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.config import (
    AGE_SEED_PAIRS,
    GENDER_SEED_PAIRS,
    AGE_BUCKETS
)

logger = logging.getLogger(__name__)


def collect_user_subreddits(
    df: pd.DataFrame,
    author_column: str = "author",
    subreddit_column: str = "subreddit",
    min_participation: int = 3,
    use_api: bool = True,
    api_data_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Collect subreddit participation data for each user.
    
    Args:
        df: DataFrame with comments
        author_column: Name of author column
        subreddit_column: Name of subreddit column
        min_participation: Minimum comments in a subreddit to count
        use_api: Whether to use Arctic Shift API for broader Reddit data
        api_data_path: Path to pre-collected API data (if available)
        
    Returns:
        DataFrame with user subreddit participation vectors
    """
    logger.info("Collecting user subreddit participation")
    
    # Try to use API data first if available
    if use_api and api_data_path and api_data_path.exists():
        logger.info(f"Loading API-collected subreddit data from {api_data_path}")
        try:
            api_df = pd.read_parquet(api_data_path)
            # Filter by min_participation
            api_df = api_df[api_df['count'] >= min_participation]
            
            # Group by user
            user_subreddit_lists = api_df.groupby('author')['subreddit'].apply(list).reset_index()
            user_subreddit_lists.columns = ['author', 'subreddits']
            user_subreddit_counts = api_df.groupby('author').size().reset_index(name='num_subreddits')
            
            result = user_subreddit_lists.merge(user_subreddit_counts, on='author', how='left')
            
            logger.info(f"Loaded API data for {len(result)} users")
            logger.info(f"Average subreddits per user: {result['num_subreddits'].mean():.1f}")
            logger.info(f"Total unique subreddits: {api_df['subreddit'].nunique()}")
            
            return result
        except Exception as e:
            logger.warning(f"Failed to load API data: {e}, falling back to comment data")
    
    # Fallback: Count subreddit participation from comments
    logger.info("Using subreddit data from comments (limited to collected subreddits)")
    user_subreddits = df.groupby([author_column, subreddit_column]).size().reset_index(name='count')
    
    # Filter by minimum participation
    user_subreddits = user_subreddits[user_subreddits['count'] >= min_participation]
    
    # Create list of subreddits per user
    user_subreddit_lists = user_subreddits.groupby(author_column)[subreddit_column].apply(list).reset_index()
    user_subreddit_lists.columns = ['author', 'subreddits']
    
    # Count subreddits per user
    user_subreddit_counts = user_subreddits.groupby(author_column).size().reset_index(name='num_subreddits')
    
    result = user_subreddit_lists.merge(user_subreddit_counts, on='author', how='left')
    
    logger.info(f"Collected subreddit data for {len(result)} users")
    logger.info(f"Average subreddits per user: {result['num_subreddits'].mean():.1f}")
    logger.warning("Limited to subreddits in collected data - consider using API for broader coverage")
    
    return result


def build_subreddit_embeddings(
    user_subreddit_lists: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 10
) -> Word2Vec:
    """
    Build Word2Vec embeddings for subreddits based on co-occurrence.
    
    Args:
        user_subreddit_lists: List of subreddit lists (one per user)
        vector_size: Dimension of embedding vectors
        window: Context window size
        min_count: Minimum subreddit frequency
        epochs: Training epochs
        
    Returns:
        Trained Word2Vec model
    """
    logger.info("Building subreddit embeddings with Word2Vec")
    logger.info(f"Training on {len(user_subreddit_lists)} users")
    
    model = Word2Vec(
        sentences=user_subreddit_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1  # Skip-gram
    )
    
    logger.info(f"Trained model with {len(model.wv)} subreddits")
    
    return model


def build_dimension_vector(
    model: Word2Vec,
    seed_pairs: List[Tuple[str, str]],
    normalize: bool = True
) -> np.ndarray:
    """
    Build a dimension vector from seed pairs.
    
    For each seed pair (young, old) or (female, male), compute the
    difference vector and average them to create the dimension.
    
    Args:
        model: Word2Vec model
        seed_pairs: List of (first_term, second_term) tuples
        normalize: Whether to normalize the resulting vector
        
    Returns:
        Dimension vector (numpy array)
    """
    dimension_vectors = []
    
    for term1, term2 in seed_pairs:
        if term1 in model.wv and term2 in model.wv:
            # Compute difference: term2 - term1
            diff = model.wv[term2] - model.wv[term1]
            dimension_vectors.append(diff)
        else:
            logger.debug(f"Seed pair ({term1}, {term2}) not found in model")
    
    if not dimension_vectors:
        logger.warning("No valid seed pairs found, returning zero vector")
        return np.zeros(model.vector_size)
    
    # Average all difference vectors
    dimension = np.mean(dimension_vectors, axis=0)
    
    if normalize:
        norm = np.linalg.norm(dimension)
        if norm > 0:
            dimension = dimension / norm
    
    return dimension


def project_user_to_dimension(
    user_subreddits: List[str],
    model: Word2Vec,
    dimension: np.ndarray
) -> float:
    """
    Project a user's subreddit vector onto a dimension.
    
    Args:
        user_subreddits: List of subreddits the user participates in
        model: Word2Vec model
        dimension: Dimension vector
        
    Returns:
        Projection score (higher = more aligned with second term in seed pairs)
    """
    if not user_subreddits:
        return 0.0
    
    # Average embeddings of user's subreddits
    valid_subreddits = [s for s in user_subreddits if s in model.wv]
    
    if not valid_subreddits:
        return 0.0
    
    user_vector = np.mean([model.wv[s] for s in valid_subreddits], axis=0)
    
    # Project onto dimension
    projection = np.dot(user_vector, dimension)
    
    return float(projection)


def age_score_to_bucket(score: float, percentiles: Dict[str, Tuple[float, float]] = None) -> str:
    """
    Convert age dimension score to age bucket.
    
    Args:
        score: Projection score on age dimension
        percentiles: Percentile thresholds for each bucket (if None, uses defaults)
        
    Returns:
        Age bucket string
    """
    if percentiles is None:
        # Default thresholds (can be calibrated on validation data)
        # Negative scores = younger, positive = older
        if score < -0.3:
            return "13-18"
        elif score < -0.1:
            return "19-25"
        elif score < 0.1:
            return "26-40"
        elif score < 0.3:
            return "41-60"
        else:
            return "61-80"
    
    # Use provided percentiles
    for bucket, (low, high) in percentiles.items():
        if low <= score < high:
            return bucket
    
    return "unknown"


def classify_with_community_embeddings(
    df: pd.DataFrame,
    author_column: str = "author",
    subreddit_column: str = "subreddit",
    min_participation: int = 3,
    vector_size: int = 100,
    api_data_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Classify user age and gender using community embeddings.
    
    Args:
        df: DataFrame with comments
        author_column: Name of author column
        subreddit_column: Name of subreddit column
        min_participation: Minimum comments per subreddit
        vector_size: Embedding vector size
        api_data_path: Path to pre-collected API data (if available)
        
    Returns:
        DataFrame with community embedding classifications
    """
    logger.info("Starting community embedding classification")
    
    # Step 1: Collect user subreddit participation
    user_subreddits_df = collect_user_subreddits(
        df, author_column, subreddit_column, min_participation,
        use_api=True, api_data_path=api_data_path
    )
    
    # Step 2: Build embeddings
    user_subreddit_lists = user_subreddits_df['subreddits'].tolist()
    model = build_subreddit_embeddings(user_subreddit_lists, vector_size=vector_size)
    
    # Step 3: Build age dimension
    logger.info("Building age dimension from seed pairs")
    age_dimension = build_dimension_vector(model, AGE_SEED_PAIRS)
    
    # Step 4: Build gender dimension
    logger.info("Building gender dimension from seed pairs")
    gender_dimension = build_dimension_vector(model, GENDER_SEED_PAIRS)
    
    # Step 5: Project users onto dimensions
    logger.info("Projecting users onto dimensions")
    results = []
    
    for _, row in user_subreddits_df.iterrows():
        user_subs = row['subreddits']
        
        # Age projection
        age_score = project_user_to_dimension(user_subs, model, age_dimension)
        age_bucket = age_score_to_bucket(age_score)
        
        # Gender projection (negative = more aligned with first term, positive = second)
        gender_score = project_user_to_dimension(user_subs, model, gender_dimension)
        # For gender, we need to interpret the score
        # Assuming seed pairs are (female, male) oriented
        if gender_score < -0.2:
            gender = "female"
        elif gender_score > 0.2:
            gender = "male"
        else:
            gender = "unknown"
        
        results.append({
            'author': row['author'],
            'age_bucket_community': age_bucket,
            'age_community_score': age_score,
            'gender_community': gender,
            'gender_community_score': gender_score
        })
    
    result_df = pd.DataFrame(results)
    
    logger.info(f"Community embedding classification complete: {len(result_df)} users classified")
    
    return result_df

