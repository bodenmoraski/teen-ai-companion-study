"""
Fixed community embedding-based age and gender classification.

This version fixes multiple issues:
1. Case-sensitivity: Subreddit names are normalized to lowercase
2. Missing gender_community_score: Score is now saved to dataframe
3. Invalid seed pair (everyman): Replaced with working alternatives
4. Threshold calibration: Uses percentile-based thresholds for gender

FIX SUMMARY:
- All subreddits are lowercased before Word2Vec training and lookup
- Gender scores are saved to output dataframe
- Seed pair 'everyman' replaced with 'OneY' (men's issues subreddit)
- Gender thresholds use adaptive percentile-based approach
"""
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# FIXED seed pairs with lowercase names and working subreddits
AGE_SEED_PAIRS_FIXED = [
    ("teenagers", "redditforgrownups"),
    ("teenrelationships", "relationship_advice"),
    ("highschool", "college"),
    ("genz", "genx"),
]

# FIXED: Replaced 'everyman' (0 users) with 'oney' (men's issues subreddit)
# All lowercase for case-insensitive matching
GENDER_SEED_PAIRS_FIXED = [
    ("askwomen", "askmen"),
    ("twoxchromosomes", "mensrights"),
    ("thegirlsurvivalguide", "oney"),  # OneY is a men's issues subreddit
]

AGE_BUCKETS = ["13-18", "19-25", "26-40", "41-60", "61-80"]


def collect_user_subreddits_fixed(
    df: pd.DataFrame,
    author_column: str = "author",
    subreddit_column: str = "subreddit",
    min_participation: int = 3,
    use_api: bool = True,
    api_data_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Collect subreddit participation data for each user.
    
    FIX: Normalizes all subreddit names to lowercase for consistent matching.
    
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
    logger.info("Collecting user subreddit participation (FIXED - lowercase normalization)")
    
    # Try to use API data first if available
    if use_api and api_data_path and api_data_path.exists():
        logger.info(f"Loading API-collected subreddit data from {api_data_path}")
        try:
            api_df = pd.read_parquet(api_data_path)
            
            # FIX: Normalize subreddit names to lowercase
            api_df['subreddit'] = api_df['subreddit'].str.lower()
            
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
    
    # FIX: Normalize subreddit names to lowercase
    df_copy = df.copy()
    df_copy[subreddit_column] = df_copy[subreddit_column].str.lower()
    
    user_subreddits = df_copy.groupby([author_column, subreddit_column]).size().reset_index(name='count')
    
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
    
    return result


def build_subreddit_embeddings_fixed(
    user_subreddit_lists: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 3,  # Reduced from 5 to capture more subreddits
    epochs: int = 10
) -> Word2Vec:
    """
    Build Word2Vec embeddings for subreddits based on co-occurrence.
    
    FIX: Uses lowercase subreddit names consistently.
    
    Args:
        user_subreddit_lists: List of subreddit lists (one per user)
        vector_size: Dimension of embedding vectors
        window: Context window size
        min_count: Minimum subreddit frequency
        epochs: Training epochs
        
    Returns:
        Trained Word2Vec model
    """
    logger.info("Building subreddit embeddings with Word2Vec (FIXED)")
    logger.info(f"Training on {len(user_subreddit_lists)} users")
    
    # FIX: Ensure all subreddits are lowercase
    normalized_lists = [[s.lower() for s in subs] for subs in user_subreddit_lists]
    
    model = Word2Vec(
        sentences=normalized_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1  # Skip-gram
    )
    
    logger.info(f"Trained model with {len(model.wv)} subreddits in vocabulary")
    
    return model


def build_dimension_vector_fixed(
    model: Word2Vec,
    seed_pairs: List[Tuple[str, str]],
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Build a dimension vector from seed pairs.
    
    FIX: Uses lowercase seed pair names for matching.
    
    Args:
        model: Word2Vec model
        seed_pairs: List of (first_term, second_term) tuples
        normalize: Whether to normalize the resulting vector
        
    Returns:
        Tuple of (dimension vector, number of valid pairs found)
    """
    dimension_vectors = []
    valid_pairs = 0
    
    for term1, term2 in seed_pairs:
        # FIX: Normalize to lowercase for matching
        term1_lower = term1.lower()
        term2_lower = term2.lower()
        
        if term1_lower in model.wv and term2_lower in model.wv:
            # Compute difference: term2 - term1
            diff = model.wv[term2_lower] - model.wv[term1_lower]
            dimension_vectors.append(diff)
            valid_pairs += 1
            logger.debug(f"Valid seed pair: ({term1}, {term2})")
        else:
            missing = []
            if term1_lower not in model.wv:
                missing.append(term1_lower)
            if term2_lower not in model.wv:
                missing.append(term2_lower)
            logger.debug(f"Seed pair ({term1}, {term2}) - missing: {missing}")
    
    if not dimension_vectors:
        logger.warning(f"No valid seed pairs found out of {len(seed_pairs)} pairs")
        return np.zeros(model.vector_size), 0
    
    logger.info(f"Found {valid_pairs}/{len(seed_pairs)} valid seed pairs")
    
    # Average all difference vectors
    dimension = np.mean(dimension_vectors, axis=0)
    
    if normalize:
        norm = np.linalg.norm(dimension)
        if norm > 0:
            dimension = dimension / norm
    
    return dimension, valid_pairs


def project_user_to_dimension(
    user_subreddits: List[str],
    model: Word2Vec,
    dimension: np.ndarray
) -> float:
    """
    Project a user's subreddit vector onto a dimension.
    
    FIX: Normalizes subreddit names to lowercase.
    
    Args:
        user_subreddits: List of subreddits the user participates in
        model: Word2Vec model
        dimension: Dimension vector
        
    Returns:
        Projection score (higher = more aligned with second term in seed pairs)
    """
    if not user_subreddits:
        return 0.0
    
    # FIX: Normalize to lowercase
    valid_subreddits = [s.lower() for s in user_subreddits if s.lower() in model.wv]
    
    if not valid_subreddits:
        return 0.0
    
    # Average embeddings of user's subreddits
    user_vector = np.mean([model.wv[s] for s in valid_subreddits], axis=0)
    
    # Project onto dimension
    projection = np.dot(user_vector, dimension)
    
    return float(projection)


def age_score_to_bucket_fixed(score: float, all_scores: Optional[np.ndarray] = None) -> str:
    """
    Convert age dimension score to age bucket.
    
    FIX: Uses more calibrated thresholds based on score distribution.
    
    Args:
        score: Projection score on age dimension
        all_scores: Optional array of all scores for percentile-based thresholds
        
    Returns:
        Age bucket string
    """
    if all_scores is not None and len(all_scores) > 100:
        # Use percentile-based thresholds for better calibration
        p20, p40, p60, p80 = np.percentile(all_scores, [20, 40, 60, 80])
        
        if score < p20:
            return "13-18"
        elif score < p40:
            return "19-25"
        elif score < p60:
            return "26-40"
        elif score < p80:
            return "41-60"
        else:
            return "61-80"
    else:
        # Default thresholds when percentiles not available
        # Negative scores = younger (aligned with first term like "teenagers")
        # Positive scores = older (aligned with second term like "redditforgrownups")
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


def gender_score_to_category_fixed(
    score: float, 
    all_scores: Optional[np.ndarray] = None,
    threshold_percentile: float = 70
) -> str:
    """
    Convert gender dimension score to category.
    
    FIX: Uses adaptive percentile-based thresholds instead of fixed 0.2.
    
    Args:
        score: Projection score on gender dimension
        all_scores: Optional array of all scores for percentile-based thresholds
        threshold_percentile: Percentile for threshold (default 70 = top 30% get classified)
        
    Returns:
        Gender category string
    """
    if all_scores is not None and len(all_scores) > 100:
        # Use percentile-based threshold for adaptive classification
        # Users in top X% of absolute scores get classified
        abs_scores = np.abs(all_scores)
        threshold = np.percentile(abs_scores, threshold_percentile)
        
        if abs(score) < threshold:
            return "unknown"
        elif score < 0:
            return "female"  # Aligned with first term (askwomen, twoxchromosomes)
        else:
            return "male"    # Aligned with second term (askmen, mensrights)
    else:
        # Default: more lenient threshold than original 0.2
        if score < -0.1:
            return "female"
        elif score > 0.1:
            return "male"
        else:
            return "unknown"


def classify_with_community_embeddings_fixed(
    df: pd.DataFrame,
    author_column: str = "author",
    subreddit_column: str = "subreddit",
    min_participation: int = 3,
    vector_size: int = 100,
    api_data_path: Optional[Path] = None,
    gender_threshold_percentile: float = 70
) -> pd.DataFrame:
    """
    Classify user age and gender using community embeddings (FIXED VERSION).
    
    FIXES APPLIED:
    1. Case-sensitivity: All subreddit names normalized to lowercase
    2. Gender scores saved: gender_community_score column is included
    3. Seed pairs: Uses working alternatives (everyman -> oney)
    4. Adaptive thresholds: Percentile-based instead of fixed values
    
    Args:
        df: DataFrame with comments
        author_column: Name of author column
        subreddit_column: Name of subreddit column
        min_participation: Minimum comments per subreddit
        vector_size: Embedding vector size
        api_data_path: Path to pre-collected API data (if available)
        gender_threshold_percentile: Percentile for gender classification threshold
        
    Returns:
        DataFrame with community embedding classifications including scores
    """
    logger.info("Starting community embedding classification (FIXED version)")
    
    # Step 1: Collect user subreddit participation (with lowercase normalization)
    user_subreddits_df = collect_user_subreddits_fixed(
        df, author_column, subreddit_column, min_participation,
        use_api=True, api_data_path=api_data_path
    )
    
    if len(user_subreddits_df) == 0:
        logger.warning("No users with subreddit data")
        return pd.DataFrame(columns=[
            'author', 'age_bucket_community', 'age_community_score',
            'gender_community', 'gender_community_score'
        ])
    
    # Step 2: Build embeddings
    user_subreddit_lists = user_subreddits_df['subreddits'].tolist()
    model = build_subreddit_embeddings_fixed(
        user_subreddit_lists, 
        vector_size=vector_size,
        min_count=3  # Reduced to capture more subreddits
    )
    
    # Step 3: Build age dimension
    logger.info("Building age dimension from seed pairs")
    age_dimension, age_valid = build_dimension_vector_fixed(model, AGE_SEED_PAIRS_FIXED)
    
    # Step 4: Build gender dimension
    logger.info("Building gender dimension from seed pairs")
    gender_dimension, gender_valid = build_dimension_vector_fixed(model, GENDER_SEED_PAIRS_FIXED)
    
    # Log seed pair validity
    if age_valid == 0:
        logger.error("NO valid age seed pairs - age classification will not work!")
    if gender_valid == 0:
        logger.error("NO valid gender seed pairs - gender classification will not work!")
    
    # Step 5: Project users onto dimensions and collect scores
    logger.info("Projecting users onto dimensions")
    age_scores = []
    gender_scores = []
    authors = []
    
    for _, row in user_subreddits_df.iterrows():
        user_subs = row['subreddits']
        authors.append(row['author'])
        
        # Age projection
        age_score = project_user_to_dimension(user_subs, model, age_dimension)
        age_scores.append(age_score)
        
        # Gender projection
        gender_score = project_user_to_dimension(user_subs, model, gender_dimension)
        gender_scores.append(gender_score)
    
    age_scores = np.array(age_scores)
    gender_scores = np.array(gender_scores)
    
    # Log score distributions
    logger.info(f"Age scores: mean={age_scores.mean():.4f}, std={age_scores.std():.4f}")
    logger.info(f"Gender scores: mean={gender_scores.mean():.4f}, std={gender_scores.std():.4f}")
    
    # Step 6: Convert scores to categories (with adaptive thresholds)
    results = []
    
    for i, author in enumerate(authors):
        age_score = age_scores[i]
        gender_score = gender_scores[i]
        
        # Use percentile-based classification
        age_bucket = age_score_to_bucket_fixed(age_score, age_scores if age_valid > 0 else None)
        gender = gender_score_to_category_fixed(
            gender_score, 
            gender_scores if gender_valid > 0 else None,
            threshold_percentile=gender_threshold_percentile
        )
        
        results.append({
            'author': author,
            'age_bucket_community': age_bucket,
            'age_community_score': age_score,
            'gender_community': gender,
            'gender_community_score': gender_score  # FIX: Now included!
        })
    
    result_df = pd.DataFrame(results)
    
    # Log classification distribution
    logger.info(f"Community embedding classification complete: {len(result_df)} users classified")
    logger.info(f"Age distribution:\n{result_df['age_bucket_community'].value_counts()}")
    logger.info(f"Gender distribution:\n{result_df['gender_community'].value_counts()}")
    
    return result_df


# Diagnostic function to check seed pair availability
def diagnose_seed_pairs(api_data_path: Path) -> Dict[str, any]:
    """
    Diagnose seed pair availability in the dataset.
    
    Args:
        api_data_path: Path to user_subreddit_interactions.parquet
        
    Returns:
        Dictionary with diagnostic information
    """
    logger.info("Diagnosing seed pair availability...")
    
    if not api_data_path.exists():
        return {"error": f"Data file not found: {api_data_path}"}
    
    usi = pd.read_parquet(api_data_path)
    
    # Normalize to lowercase
    usi['subreddit_lower'] = usi['subreddit'].str.lower()
    
    all_subreddits = set(usi['subreddit_lower'].unique())
    
    results = {
        "total_subreddits": len(all_subreddits),
        "age_seed_pairs": {},
        "gender_seed_pairs": {},
    }
    
    for term1, term2 in AGE_SEED_PAIRS_FIXED:
        in_data_1 = term1.lower() in all_subreddits
        in_data_2 = term2.lower() in all_subreddits
        users_1 = usi[usi['subreddit_lower'] == term1.lower()]['author'].nunique() if in_data_1 else 0
        users_2 = usi[usi['subreddit_lower'] == term2.lower()]['author'].nunique() if in_data_2 else 0
        
        results["age_seed_pairs"][(term1, term2)] = {
            "term1_found": in_data_1,
            "term2_found": in_data_2,
            "term1_users": users_1,
            "term2_users": users_2,
            "valid": in_data_1 and in_data_2
        }
    
    for term1, term2 in GENDER_SEED_PAIRS_FIXED:
        in_data_1 = term1.lower() in all_subreddits
        in_data_2 = term2.lower() in all_subreddits
        users_1 = usi[usi['subreddit_lower'] == term1.lower()]['author'].nunique() if in_data_1 else 0
        users_2 = usi[usi['subreddit_lower'] == term2.lower()]['author'].nunique() if in_data_2 else 0
        
        results["gender_seed_pairs"][(term1, term2)] = {
            "term1_found": in_data_1,
            "term2_found": in_data_2,
            "term1_users": users_1,
            "term2_users": users_2,
            "valid": in_data_1 and in_data_2
        }
    
    return results


# Alias for drop-in replacement
classify_with_community_embeddings = classify_with_community_embeddings_fixed

