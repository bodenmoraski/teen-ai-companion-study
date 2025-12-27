"""
LLM-based age classification for Reddit users.

This module uses GPT-4o-mini to classify user age buckets based on comment history.
"""
import json
import logging
import time
from typing import List, Dict, Optional, Any
import pandas as pd
from openai import OpenAI

from ..utils.config import OPENAI_API_KEY, LLM_AGE_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

AGE_BUCKETS = ["13-18", "19-25", "26-40", "41-60", "61-80"]


def classify_age_llm(
    comments: List[str],
    max_comments: int = 20,
    client: Optional[OpenAI] = None
) -> Dict[str, Any]:
    """
    Classify user age bucket using GPT-4o-mini.
    
    Args:
        comments: List of comment texts from the user
        max_comments: Maximum number of comments to include in context
        client: Optional OpenAI client (creates new one if not provided)
        
    Returns:
        Dictionary with age_bucket, confidence, and reasoning
    """
    if not comments:
        return {
            "age_bucket": None,
            "confidence": 0.0,
            "reasoning": "No comments provided"
        }
    
    # Limit comments for context
    sample_comments = comments[:max_comments]
    comments_text = "\n".join([f"- {c[:200]}" for c in sample_comments if c])
    
    if not comments_text:
        return {
            "age_bucket": None,
            "confidence": 0.0,
            "reasoning": "No valid comments provided"
        }
    
    prompt = f"""Analyze these Reddit comments from a single user and estimate their most likely age bucket.

Consider:
- Vocabulary and language complexity
- Topics and interests mentioned
- Life stage indicators (school, work, family references)
- Cultural references
- Writing style and maturity

Comments:
{comments_text}

Age buckets: 13-18, 19-25, 26-40, 41-60, 61-80

Respond with JSON only:
{{"age_bucket": "one of the buckets above", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    if client is None:
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not configured")
            return {
                "age_bucket": None,
                "confidence": 0.0,
                "reasoning": "API key not configured"
            }
        client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # GPT-5 models use max_completion_tokens instead of max_tokens
        is_gpt5 = "gpt-5" in LLM_AGE_MODEL.lower()
        
        params = {
            "model": LLM_AGE_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": LLM_TEMPERATURE,
            "response_format": {"type": "json_object"}
        }
        
        # Add token limit based on model type
        if is_gpt5:
            params["max_completion_tokens"] = 500
        else:
            params["max_tokens"] = 500
        
        response = client.chat.completions.create(**params)
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate bucket
        if result.get("age_bucket") not in AGE_BUCKETS:
            logger.warning(f"Invalid age bucket returned: {result.get('age_bucket')}")
            result["age_bucket"] = None
            result["confidence"] = 0.0
        
        return result
        
    except Exception as e:
        logger.error(f"Error in LLM age classification: {e}")
        return {
            "age_bucket": None,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def classify_users_llm(
    df: pd.DataFrame,
    author_column: str = "author",
    text_column: str = "body",
    batch_size: int = 100,
    max_comments_per_user: int = 20,
    rate_limit: float = 0.5
) -> pd.DataFrame:
    """
    Classify age buckets for multiple users using LLM.
    
    Args:
        df: DataFrame with comments
        author_column: Name of author column
        text_column: Name of text column
        batch_size: Number of users to process before saving
        max_comments_per_user: Maximum comments per user to analyze
        rate_limit: Seconds to wait between API calls
        
    Returns:
        DataFrame with user age classifications
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not configured, skipping LLM classification")
        return pd.DataFrame(columns=["author", "age_bucket_llm", "confidence_llm", "reasoning_llm"])
    
    logger.info("Starting LLM age classification")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Group comments by author
    user_comments = df.groupby(author_column)[text_column].apply(list).to_dict()
    
    results = []
    total_users = len(user_comments)
    
    logger.info(f"Classifying {total_users} users with LLM")
    
    for idx, (author, comments) in enumerate(user_comments.items(), 1):
        try:
            result = classify_age_llm(
                comments=comments,
                max_comments=max_comments_per_user,
                client=client
            )
            
            results.append({
                "author": author,
                "age_bucket_llm": result.get("age_bucket"),
                "confidence_llm": result.get("confidence", 0.0),
                "reasoning_llm": result.get("reasoning", "")
            })
            
            # Rate limiting
            if idx % 10 == 0:
                time.sleep(rate_limit)
                logger.info(f"Processed {idx}/{total_users} users")
                
        except Exception as e:
            logger.error(f"Error classifying user {author}: {e}")
            results.append({
                "author": author,
                "age_bucket_llm": None,
                "confidence_llm": 0.0,
                "reasoning_llm": f"Error: {str(e)}"
            })
    
    result_df = pd.DataFrame(results)
    logger.info(f"LLM classification complete: {result_df['age_bucket_llm'].notna().sum()} users classified")
    
    return result_df

