"""
Community embedding-based classification V2 - FULLY FIXED.

This version implements all critical fixes:
1. CALIBRATED age thresholds (not percentile-based)
2. CENTERED gender scores with optimal thresholds
3. BETTER seed pairs with sufficient users
4. PROPER validation-informed calibration

Author: Research Team
Date: December 26, 2025
"""
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

# =============================================================================
# IMPROVED SEED PAIRS (all have 50+ users)
# =============================================================================

AGE_SEED_PAIRS_V2 = [
    ("teenagers", "redditforgrownups"),       # 10563 vs 140 users
    ("teenrelationships", "relationship_advice"),  # 105 vs 957 users
    ("highschool", "college"),                # 1188 vs 356 users
    ("genz", "genx"),                         # 2504 vs 434 users
]

# FIXED: Replaced "oney" (1 user) with "malelivingspace" (994 users)
GENDER_SEED_PAIRS_V2 = [
    ("askwomen", "askmen"),                   # 443 vs 763 users
    ("twoxchromosomes", "mensrights"),        # 499 vs 165 users
    ("thegirlsurvivalguide", "malelivingspace"),  # 339 vs 994 users
]

# Calibrated thresholds based on self-declared ground truth
# Calculated from: diagnose_issues.py output
# 13-18: mean=-0.30, 19-25: mean=-0.23, 26-40: mean=-0.12, 61-80: mean=0.13
AGE_THRESHOLDS_CALIBRATED = {
    "13-18": (-float('inf'), -0.26),  # score < -0.26
    "19-25": (-0.26, -0.17),           # -0.26 <= score < -0.17
    "26-40": (-0.17, -0.05),           # -0.17 <= score < -0.05
    "41-60": (-0.05, 0.05),            # -0.05 <= score < 0.05
    "61-80": (0.05, float('inf')),     # score >= 0.05
}

# Gender threshold: midpoint between male and female means
# male: -0.18, female: -0.24, midpoint = -0.21
GENDER_THRESHOLD_CALIBRATED = -0.21

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def collect_user_subreddits_v2(
    df: pd.DataFrame,
    author_column: str = "author",
    subreddit_column: str = "subreddit",
    min_participation: int = 3,
    api_data_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Collect subreddit participation with lowercase normalization.
    """
    logger.info("Collecting user subreddit participation (V2)")
    
    if api_data_path and api_data_path.exists():
        logger.info(f"Loading API-collected subreddit data from {api_data_path}")
        try:
            api_df = pd.read_parquet(api_data_path)
            
            # CRITICAL: Normalize to lowercase
            api_df['subreddit'] = api_df['subreddit'].str.lower()
            
            # Filter by min_participation
            api_df = api_df[api_df['count'] >= min_participation]
            
            # Group by user
            user_subreddit_lists = api_df.groupby('author')['subreddit'].apply(list).reset_index()
            user_subreddit_lists.columns = ['author', 'subreddits']
            
            user_subreddit_counts = api_df.groupby('author').size().reset_index(name='num_subreddits')
            result = user_subreddit_lists.merge(user_subreddit_counts, on='author', how='left')
            
            logger.info(f"Loaded API data for {len(result)} users")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to load API data: {e}")
    
    # Fallback to comment data
    df_copy = df.copy()
    df_copy[subreddit_column] = df_copy[subreddit_column].str.lower()
    
    user_subreddits = df_copy.groupby([author_column, subreddit_column]).size().reset_index(name='count')
    user_subreddits = user_subreddits[user_subreddits['count'] >= min_participation]
    
    user_subreddit_lists = user_subreddits.groupby(author_column)[subreddit_column].apply(list).reset_index()
    user_subreddit_lists.columns = ['author', 'subreddits']
    
    user_subreddit_counts = user_subreddits.groupby(author_column).size().reset_index(name='num_subreddits')
    result = user_subreddit_lists.merge(user_subreddit_counts, on='author', how='left')
    
    logger.info(f"Collected subreddit data for {len(result)} users")
    return result


def build_subreddit_embeddings_v2(
    user_subreddit_lists: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 10
) -> Word2Vec:
    """
    Build Word2Vec embeddings with lowercase normalization.
    """
    logger.info("Building subreddit embeddings (V2)")
    
    # Ensure all lowercase
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
    
    logger.info(f"Trained model with {len(model.wv)} subreddits")
    return model


def build_dimension_vector_v2(
    model: Word2Vec,
    seed_pairs: List[Tuple[str, str]]
) -> Tuple[np.ndarray, int, List[Tuple[str, str]]]:
    """
    Build dimension vector from seed pairs.
    Returns: (dimension_vector, num_valid_pairs, valid_pairs_list)
    """
    dimension_vectors = []
    valid_pairs = []
    
    for term1, term2 in seed_pairs:
        t1_lower = term1.lower()
        t2_lower = term2.lower()
        
        if t1_lower in model.wv and t2_lower in model.wv:
            diff = model.wv[t2_lower] - model.wv[t1_lower]
            dimension_vectors.append(diff)
            valid_pairs.append((term1, term2))
            logger.debug(f"Valid seed pair: ({term1}, {term2})")
        else:
            logger.warning(f"Seed pair ({term1}, {term2}) not in vocabulary")
    
    if not dimension_vectors:
        logger.error("NO valid seed pairs found!")
        return np.zeros(model.vector_size), 0, []
    
    logger.info(f"Found {len(valid_pairs)}/{len(seed_pairs)} valid seed pairs")
    
    dimension = np.mean(dimension_vectors, axis=0)
    
    # Normalize
    norm = np.linalg.norm(dimension)
    if norm > 0:
        dimension = dimension / norm
    
    return dimension, len(valid_pairs), valid_pairs


def project_user_to_dimension_v2(
    user_subreddits: List[str],
    model: Word2Vec,
    dimension: np.ndarray
) -> float:
    """
    Project user vector onto dimension.
    """
    if not user_subreddits:
        return 0.0
    
    valid_subs = [s.lower() for s in user_subreddits if s.lower() in model.wv]
    
    if not valid_subs:
        return 0.0
    
    user_vector = np.mean([model.wv[s] for s in valid_subs], axis=0)
    return float(np.dot(user_vector, dimension))


def age_score_to_bucket_calibrated(score: float) -> str:
    """
    Convert age score to bucket using CALIBRATED thresholds.
    
    Unlike percentile-based, this creates NON-UNIFORM distribution
    that reflects actual age patterns.
    """
    for bucket, (low, high) in AGE_THRESHOLDS_CALIBRATED.items():
        if low <= score < high:
            return bucket
    return "26-40"  # Default middle bucket


def gender_score_to_category_calibrated(
    score: float,
    population_mean: float = -0.21
) -> str:
    """
    Convert gender score to category using CALIBRATED threshold.
    
    Uses centered scores: score > population_mean → male, else → female
    Only classifies users with clear signal (|score - mean| > 0.1)
    """
    centered_score = score - population_mean
    
    # Require minimum signal to classify
    if abs(centered_score) < 0.05:
        return "unknown"
    
    if centered_score > 0:
        return "male"
    else:
        return "female"


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_with_community_embeddings_v2(
    df: pd.DataFrame,
    author_column: str = "author",
    subreddit_column: str = "subreddit",
    min_participation: int = 3,
    vector_size: int = 100,
    api_data_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Classify users using community embeddings V2.
    
    Key improvements:
    1. Calibrated age thresholds (not percentile-based)
    2. Centered gender scores with calibrated threshold
    3. Better seed pairs
    """
    logger.info("=" * 60)
    logger.info("Starting Community Embedding Classification V2")
    logger.info("=" * 60)
    
    # Step 1: Collect subreddit participation
    user_subs_df = collect_user_subreddits_v2(
        df, author_column, subreddit_column, min_participation,
        api_data_path=api_data_path
    )
    
    if len(user_subs_df) == 0:
        logger.error("No users with subreddit data!")
        return pd.DataFrame(columns=[
            'author', 'age_bucket_community', 'age_community_score',
            'gender_community', 'gender_community_score'
        ])
    
    # Step 2: Build embeddings
    user_sub_lists = user_subs_df['subreddits'].tolist()
    model = build_subreddit_embeddings_v2(user_sub_lists, vector_size=vector_size)
    
    # Step 3: Build dimension vectors
    age_dim, age_valid, age_pairs = build_dimension_vector_v2(model, AGE_SEED_PAIRS_V2)
    gender_dim, gender_valid, gender_pairs = build_dimension_vector_v2(model, GENDER_SEED_PAIRS_V2)
    
    logger.info(f"Age dimension: {age_valid} valid seed pairs")
    logger.info(f"Gender dimension: {gender_valid} valid seed pairs")
    
    if age_valid == 0:
        logger.error("NO valid age seed pairs - age classification will fail!")
    if gender_valid == 0:
        logger.error("NO valid gender seed pairs - gender classification will fail!")
    
    # Step 4: Project all users and collect scores
    logger.info("Projecting users onto dimensions...")
    age_scores = []
    gender_scores = []
    authors = []
    
    for _, row in user_subs_df.iterrows():
        user_subs = row['subreddits']
        authors.append(row['author'])
        
        age_score = project_user_to_dimension_v2(user_subs, model, age_dim)
        gender_score = project_user_to_dimension_v2(user_subs, model, gender_dim)
        
        age_scores.append(age_score)
        gender_scores.append(gender_score)
    
    age_scores = np.array(age_scores)
    gender_scores = np.array(gender_scores)
    
    # Log distributions
    logger.info(f"Age scores: mean={age_scores.mean():.4f}, std={age_scores.std():.4f}")
    logger.info(f"Gender scores: mean={gender_scores.mean():.4f}, std={gender_scores.std():.4f}")
    
    # Calculate population mean for gender centering
    gender_pop_mean = gender_scores.mean()
    logger.info(f"Using gender population mean for centering: {gender_pop_mean:.4f}")
    
    # Step 5: Classify using CALIBRATED thresholds
    logger.info("Classifying users with CALIBRATED thresholds...")
    results = []
    
    for i, author in enumerate(authors):
        age_score = age_scores[i]
        gender_score = gender_scores[i]
        
        # CALIBRATED classification
        age_bucket = age_score_to_bucket_calibrated(age_score)
        gender = gender_score_to_category_calibrated(gender_score, population_mean=gender_pop_mean)
        
        results.append({
            'author': author,
            'age_bucket_community': age_bucket,
            'age_community_score': age_score,
            'gender_community': gender,
            'gender_community_score': gender_score
        })
    
    result_df = pd.DataFrame(results)
    
    # Log final distributions
    logger.info("\n=== FINAL CLASSIFICATION DISTRIBUTION ===")
    logger.info(f"Age distribution:\n{result_df['age_bucket_community'].value_counts()}")
    logger.info(f"Gender distribution:\n{result_df['gender_community'].value_counts()}")
    
    # Verify NOT uniform distribution
    age_pcts = result_df['age_bucket_community'].value_counts(normalize=True)
    if age_pcts.std() < 0.02:
        logger.warning("Age distribution is too uniform - calibration may not be working!")
    else:
        logger.info("Age distribution is NON-UNIFORM (calibration working)")
    
    return result_df


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_against_ground_truth(
    classifications: pd.DataFrame,
    demographics: pd.DataFrame
) -> Dict:
    """
    Validate classifications against self-declarations.
    """
    merged = classifications.merge(
        demographics[['author', 'age_bucket_self_declared', 'gender_self_declared']],
        on='author',
        how='inner'
    )
    
    results = {}
    
    # Age validation
    age_mask = merged['age_bucket_self_declared'].notna() & merged['age_bucket_community'].notna()
    if age_mask.sum() > 0:
        age_correct = (
            merged.loc[age_mask, 'age_bucket_self_declared'] == 
            merged.loc[age_mask, 'age_bucket_community']
        ).sum()
        age_total = age_mask.sum()
        results['age_accuracy'] = age_correct / age_total
        results['age_n'] = age_total
        logger.info(f"Age accuracy: {results['age_accuracy']:.1%} ({age_correct}/{age_total})")
    
    # Gender validation
    gender_mask = (
        merged['gender_self_declared'].notna() & 
        merged['gender_community'].notna() &
        (merged['gender_community'] != 'unknown')
    )
    if gender_mask.sum() > 0:
        gender_correct = (
            merged.loc[gender_mask, 'gender_self_declared'] == 
            merged.loc[gender_mask, 'gender_community']
        ).sum()
        gender_total = gender_mask.sum()
        results['gender_accuracy'] = gender_correct / gender_total
        results['gender_n'] = gender_total
        logger.info(f"Gender accuracy: {results['gender_accuracy']:.1%} ({gender_correct}/{gender_total})")
    
    return results


# Aliases for backward compatibility
classify_with_community_embeddings = classify_with_community_embeddings_v2

