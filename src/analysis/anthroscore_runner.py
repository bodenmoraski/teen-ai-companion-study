"""
AnthroScore V2 runner for computing anthropomorphization scores.

This module provides functionality to run AnthroScore V2 on all comments
and aggregate scores to user level.
"""
import logging
import torch
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm
import numpy as np

# Import AnthroScore V2
try:
    from ...anthroscore.anthroscore_v2 import AnthroScoreV2
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.anthroscore.anthroscore_v2 import AnthroScoreV2

logger = logging.getLogger(__name__)


def compute_anthroscores(
    df: pd.DataFrame,
    text_column: str = "body",
    batch_size: int = 32,
    device: str = None,
    use_twitter_model: bool = True
) -> pd.DataFrame:
    """
    Compute AnthroScore for all comments.
    
    Args:
        df: DataFrame with comments
        text_column: Name of column with comment text
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu'), auto-detect if None
        use_twitter_model: Whether to use Twitter-trained model
        
    Returns:
        DataFrame with additional 'anthroscore' column
    """
    logger.info("Initializing AnthroScore V2")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Initialize scorer
    try:
        scorer = AnthroScoreV2(
            use_twitter_model=use_twitter_model,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize AnthroScore V2: {e}")
        raise
    
    logger.info(f"Computing AnthroScore for {len(df)} comments")
    
    # Process in batches
    scores = []
    texts = df[text_column].fillna("").astype(str).tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing AnthroScores"):
        batch_texts = texts[i:i+batch_size]
        
        try:
            batch_scores = []
            for text in batch_texts:
                try:
                    result = scorer.compute_score(text)
                    # Extract mean score
                    if isinstance(result, dict):
                        score = result.get('mean_score', result.get('score', 0.0))
                    else:
                        score = float(result) if isinstance(result, (int, float)) else 0.0
                    batch_scores.append(score)
                except Exception as e:
                    logger.debug(f"Error computing score for text: {e}")
                    batch_scores.append(np.nan)
            
            scores.extend(batch_scores)
            
        except Exception as e:
            logger.warning(f"Error processing batch {i//batch_size}: {e}")
            scores.extend([np.nan] * len(batch_texts))
    
    # Add scores to dataframe
    result_df = df.copy()
    result_df['anthroscore'] = scores
    
    # Log statistics
    valid_scores = result_df['anthroscore'].dropna()
    logger.info(f"Computed scores for {len(valid_scores)} comments")
    logger.info(f"Score statistics: mean={valid_scores.mean():.3f}, "
                f"std={valid_scores.std():.3f}, "
                f"min={valid_scores.min():.3f}, "
                f"max={valid_scores.max():.3f}")
    
    return result_df


def aggregate_to_user_level(
    df: pd.DataFrame,
    author_column: str = "author",
    score_column: str = "anthroscore"
) -> pd.DataFrame:
    """
    Aggregate AnthroScores to user level.
    
    Args:
        df: DataFrame with comment-level scores
        author_column: Name of author column
        score_column: Name of score column
        
    Returns:
        DataFrame with user-level aggregations
    """
    logger.info("Aggregating AnthroScores to user level")
    
    user_agg = df.groupby(author_column)[score_column].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
        ('min', 'min'),
        ('max', 'max'),
        ('median', 'median')
    ]).reset_index()
    
    user_agg.columns = [
        'author',
        f'{score_column}_mean',
        f'{score_column}_std',
        f'{score_column}_count',
        f'{score_column}_min',
        f'{score_column}_max',
        f'{score_column}_median'
    ]
    
    logger.info(f"Aggregated scores for {len(user_agg)} users")
    
    return user_agg

