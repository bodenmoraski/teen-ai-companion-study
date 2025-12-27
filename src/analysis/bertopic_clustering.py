"""
BERTopic clustering for interaction pattern analysis.

This module implements topic modeling on Reddit comments to identify
dominant interaction patterns.
"""
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

logger = logging.getLogger(__name__)


def cluster_comments(
    texts: List[str],
    min_topic_size: int = 50,
    n_components: int = 5,
    random_state: int = 42
) -> tuple:
    """
    Cluster comments using BERTopic.
    
    Args:
        texts: List of comment texts
        min_topic_size: Minimum size for a topic
        n_components: Number of UMAP components
        random_state: Random seed
        
    Returns:
        Tuple of (topic_model, topics, probabilities)
    """
    logger.info(f"Clustering {len(texts)} comments with BERTopic")
    
    # Use domain-appropriate embeddings
    logger.info("Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Configure dimensionality reduction and clustering
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=random_state
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size,
        nr_topics='auto',
        calculate_probabilities=True,
        verbose=True
    )
    
    # Fit model
    logger.info("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(texts)
    
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info(f"Found {n_topics} topics ({sum(topics == -1)} comments in noise cluster)")
    
    return topic_model, topics, probs


def extract_topic_features(
    df: pd.DataFrame,
    text_column: str = "body",
    min_topic_size: int = 50
) -> pd.DataFrame:
    """
    Extract topic features from comments.
    
    Args:
        df: DataFrame with comments
        text_column: Name of column with comment text
        min_topic_size: Minimum size for a topic
        
    Returns:
        DataFrame with topic assignments and probabilities
    """
    # Filter out empty texts
    valid_df = df[df[text_column].notna() & (df[text_column].astype(str).str.len() > 0)].copy()
    texts = valid_df[text_column].astype(str).tolist()
    
    if len(texts) < min_topic_size:
        logger.warning(f"Not enough comments ({len(texts)}) for topic modeling")
        result_df = valid_df.copy()
        result_df['topic'] = -1
        result_df['topic_probability'] = 0.0
        return result_df
    
    # Cluster comments
    topic_model, topics, probs = cluster_comments(texts, min_topic_size=min_topic_size)
    
    # Add topic assignments
    result_df = valid_df.copy()
    result_df['topic'] = topics
    result_df['topic_probability'] = probs.max(axis=1) if probs is not None else [0.0] * len(topics)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    logger.info(f"Topic information:\n{topic_info.head(20)}")
    
    return result_df


def aggregate_topics_to_user_level(
    df: pd.DataFrame,
    author_column: str = "author",
    topic_column: str = "topic"
) -> pd.DataFrame:
    """
    Aggregate topic distributions to user level.
    
    Args:
        df: DataFrame with topic assignments
        author_column: Name of author column
        topic_column: Name of topic column
        
    Returns:
        DataFrame with user-level topic distributions
    """
    logger.info("Aggregating topics to user level")
    
    # Create topic distribution per user
    topic_dists = df.groupby([author_column, topic_column]).size().unstack(fill_value=0)
    topic_dists = topic_dists.div(topic_dists.sum(axis=1), axis=0)  # Normalize to proportions
    
    # Get dominant topic per user
    dominant_topics = topic_dists.idxmax(axis=1)
    dominant_topic_probs = topic_dists.max(axis=1)
    
    user_topics = pd.DataFrame({
        'author': topic_dists.index,
        'dominant_topic': dominant_topics.values,
        'dominant_topic_probability': dominant_topic_probs.values
    })
    
    # Add topic distribution columns
    for topic_id in topic_dists.columns:
        user_topics[f'topic_{topic_id}_proportion'] = topic_dists[topic_id].values
    
    logger.info(f"Aggregated topics for {len(user_topics)} users")
    
    return user_topics

