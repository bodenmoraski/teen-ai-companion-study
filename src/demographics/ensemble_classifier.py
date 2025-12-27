"""
Ensemble classifier for age and gender classification.

This module combines self-declaration, community embedding, and LLM predictions
using weighted voting.
"""
import logging
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Weights for ensemble voting
SELF_DECLARATION_WEIGHT = 1.0
COMMUNITY_EMBEDDING_WEIGHT = 0.7
LLM_WEIGHT = 0.6

AGE_BUCKETS = ["13-18", "19-25", "26-40", "41-60", "61-80"]


def weighted_age_vote(
    self_declaration: Optional[str],
    community_embedding: Optional[str],
    llm_prediction: Optional[str],
    llm_confidence: float = 0.0,
    community_score: Optional[float] = None
) -> Tuple[Optional[str], float]:
    """
    Combine age predictions using weighted voting.
    
    Args:
        self_declaration: Age bucket from self-declaration
        community_embedding: Age bucket from community embedding
        llm_prediction: Age bucket from LLM
        llm_confidence: LLM confidence score
        
    Returns:
        Tuple of (predicted_bucket, confidence_score)
    """
    votes = {}
    
    # Self-declaration (highest weight)
    if self_declaration and self_declaration in AGE_BUCKETS:
        votes[self_declaration] = votes.get(self_declaration, 0) + SELF_DECLARATION_WEIGHT
    
    # Community embedding (weighted by score magnitude if available)
    if community_embedding and community_embedding in AGE_BUCKETS:
        embedding_weight = COMMUNITY_EMBEDDING_WEIGHT
        if community_score is not None:
            # Weight by absolute score (stronger signal = higher weight)
            embedding_weight *= min(1.0, abs(community_score) * 2)
        votes[community_embedding] = votes.get(community_embedding, 0) + embedding_weight
    
    # LLM (weighted by confidence)
    if llm_prediction and llm_prediction in AGE_BUCKETS:
        llm_weight = LLM_WEIGHT * llm_confidence
        votes[llm_prediction] = votes.get(llm_prediction, 0) + llm_weight
    
    if not votes:
        return None, 0.0
    
    # Get prediction with highest vote count
    predicted_bucket = max(votes.items(), key=lambda x: x[1])[0]
    
    # Calculate confidence (normalized vote total)
    total_weight = SELF_DECLARATION_WEIGHT + COMMUNITY_EMBEDDING_WEIGHT + LLM_WEIGHT
    confidence = votes[predicted_bucket] / total_weight
    
    return predicted_bucket, min(confidence, 1.0)


def create_ensemble_classification(
    self_decl_df: pd.DataFrame,
    community_embedding_df: Optional[pd.DataFrame] = None,
    llm_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create ensemble classification combining all methods.
    
    Args:
        self_decl_df: DataFrame with self-declaration results
        community_embedding_df: DataFrame with community embedding results (optional)
        llm_df: DataFrame with LLM classification results (optional)
        
    Returns:
        DataFrame with ensemble predictions
    """
    logger.info("Creating ensemble classification")
    
    # Start with self-declarations
    result_df = self_decl_df[["author"]].copy()
    
    # Merge community embedding predictions
    if community_embedding_df is not None:
        comm_cols = ["author", "age_bucket_community", "age_community_score"]
        if "gender_community" in community_embedding_df.columns:
            comm_cols.append("gender_community")
        result_df = result_df.merge(
            community_embedding_df[comm_cols],
            on="author",
            how="left"
        )
    else:
        result_df["age_bucket_community"] = None
        result_df["age_community_score"] = None
    
    # Merge LLM predictions
    if llm_df is not None:
        result_df = result_df.merge(
            llm_df[["author", "age_bucket_llm", "confidence_llm"]],
            on="author",
            how="left"
        )
    else:
        result_df["age_bucket_llm"] = None
        result_df["confidence_llm"] = 0.0
    
    # Merge self-declarations
    result_df = result_df.merge(
        self_decl_df[["author", "age_bucket_self_declared"]],
        on="author",
        how="left"
    )
    
    # Apply weighted voting
    result_df[["age_bucket", "confidence"]] = result_df.apply(
        lambda row: pd.Series(weighted_age_vote(
            self_declaration=row.get("age_bucket_self_declared"),
            community_embedding=row.get("age_bucket_community"),
            llm_prediction=row.get("age_bucket_llm"),
            llm_confidence=row.get("confidence_llm", 0.0),
            community_score=row.get("age_community_score")
        )),
        axis=1
    )
    
    # Track which methods agreed
    result_df["methods_used"] = result_df.apply(
        lambda row: ",".join([
            "self_decl" if pd.notna(row.get("age_bucket_self_declared")) else "",
            "community" if pd.notna(row.get("age_bucket_community")) else "",
            "llm" if pd.notna(row.get("age_bucket_llm")) else ""
        ]).strip(","),
        axis=1
    )
    
    # Add gender (self-declaration takes precedence, then community embedding)
    if 'gender_self_declared' in self_decl_df.columns:
        result_df = result_df.merge(
            self_decl_df[['author', 'gender_self_declared']],
            on='author',
            how='left'
        )
    else:
        result_df['gender_self_declared'] = None
    
    # Combine gender from self-declaration and community embedding
    result_df['gender'] = result_df['gender_self_declared'].fillna(
        result_df.get('gender_community', None)
    )
    
    classified_count = result_df["age_bucket"].notna().sum()
    gender_count = result_df["gender"].notna().sum()
    logger.info(f"Ensemble classification complete: {classified_count} users classified (age), {gender_count} (gender)")
    
    return result_df

