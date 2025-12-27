"""
Emotion analysis for Reddit comments.

This module classifies emotions in comments using a pre-trained RoBERTa model.
"""
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

# Emotion labels from the distilroberta model
EMOTION_LABELS = [
    'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral'
]


def classify_emotions(
    texts: List[str],
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    batch_size: int = 32,
    device: int = None
) -> List[Dict[str, float]]:
    """
    Classify emotions in texts.
    
    Args:
        texts: List of texts to classify
        model_name: Name of emotion classification model
        batch_size: Batch size for processing
        device: Device to use (-1 for CPU, 0+ for GPU)
        
    Returns:
        List of dictionaries with emotion scores
    """
    logger.info(f"Classifying emotions for {len(texts)} texts")
    
    # Auto-detect device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    
    logger.info(f"Loading emotion classifier: {model_name}")
    try:
        emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,  # Return all emotions
            device=device,
            batch_size=batch_size
        )
    except Exception as e:
        logger.error(f"Failed to load emotion classifier: {e}")
        raise
    
    logger.info("Classifying emotions...")
    results = []
    
    for text in texts:
        try:
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                # Return neutral for empty text
                emotion_dict = {label: 0.0 for label in EMOTION_LABELS}
                emotion_dict['neutral'] = 1.0
                results.append(emotion_dict)
                continue
            
            # Truncate long texts
            text_truncated = text[:512] if len(text) > 512 else text
            
            # Classify
            emotions = emotion_classifier(text_truncated)[0]
            
            # Convert to dictionary
            emotion_dict = {label: 0.0 for label in EMOTION_LABELS}
            for item in emotions:
                label = item['label'].lower()
                score = item['score']
                if label in emotion_dict:
                    emotion_dict[label] = score
            
            results.append(emotion_dict)
            
        except Exception as e:
            logger.debug(f"Error classifying emotion: {e}")
            emotion_dict = {label: 0.0 for label in EMOTION_LABELS}
            emotion_dict['neutral'] = 1.0
            results.append(emotion_dict)
    
    logger.info("Emotion classification complete")
    return results


def extract_emotion_features(
    df: pd.DataFrame,
    text_column: str = "body"
) -> pd.DataFrame:
    """
    Extract emotion features from comments.
    
    Args:
        df: DataFrame with comments
        text_column: Name of column with comment text
        
    Returns:
        DataFrame with emotion scores
    """
    texts = df[text_column].fillna("").astype(str).tolist()
    
    # Classify emotions
    emotion_results = classify_emotions(texts)
    
    # Add emotion columns to dataframe
    result_df = df.copy()
    for label in EMOTION_LABELS:
        result_df[f'emotion_{label}'] = [r[label] for r in emotion_results]
    
    # Get dominant emotion
    emotion_columns = [f'emotion_{label}' for label in EMOTION_LABELS]
    result_df['dominant_emotion'] = result_df[emotion_columns].idxmax(axis=1).str.replace('emotion_', '')
    result_df['dominant_emotion_score'] = result_df[emotion_columns].max(axis=1)
    
    return result_df


def aggregate_emotions_to_user_level(
    df: pd.DataFrame,
    author_column: str = "author"
) -> pd.DataFrame:
    """
    Aggregate emotion features to user level.
    
    Args:
        df: DataFrame with emotion features
        author_column: Name of author column
        
    Returns:
        DataFrame with user-level emotion aggregations
    """
    logger.info("Aggregating emotions to user level")
    
    emotion_columns = [f'emotion_{label}' for label in EMOTION_LABELS]
    
    # Compute mean emotion scores per user
    user_emotions = df.groupby(author_column)[emotion_columns].mean().reset_index()
    
    # Add dominant emotion per user
    user_emotions['dominant_emotion'] = user_emotions[emotion_columns].idxmax(axis=1).str.replace('emotion_', '')
    user_emotions['dominant_emotion_score'] = user_emotions[emotion_columns].max(axis=1)
    
    logger.info(f"Aggregated emotions for {len(user_emotions)} users")
    
    return user_emotions

