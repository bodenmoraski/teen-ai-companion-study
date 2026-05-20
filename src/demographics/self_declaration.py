"""
Self-declaration extraction for age and gender from Reddit comments.

This module implements regex-based extraction of explicit age and gender
declarations from comment text and author flair.
"""
import re
import logging
from typing import Dict, Optional, Tuple, List
import pandas as pd

logger = logging.getLogger(__name__)

# Age patterns: various ways users declare their age
AGE_PATTERNS = [
    r'\b(?:I am|I\'m|im|I\'m a|I am a)\s*(\d{1,2})\s*(?:years?\s*old|yo|y\.?o\.?|year|years)\b',
    r'\b(\d{1,2})\s*(?:years?\s*old|yo|y\.?o\.?)\b',
    r'\b(\d{1,2})\s*[MF]\b',  # "16F", "23M"
    r'\bas a\s*(\d{1,2})\b',
    r'\bage\s*(\d{1,2})\b',
    r'\bturned\s*(\d{1,2})\b',
    r'\b(?:when I was|back when I was)\s*(\d{1,2})\b',
    # Gender-letter-first Reddit tags: "(M38)", "I (M38)", bare "M38" / "F22"
    r'\(\s*[MF]\s*(\d{1,2})\s*\)',
    r'\b[MF]\s*(\d{1,2})\b',
]

# Gender patterns
GENDER_PATTERNS = {
    'male': [
        r'\b(\d{1,2})\s*M\b',  # "23M"
        r'\(\s*M\s*\d{1,2}\s*\)',  # "(M38)", "I (M38)"
        r'\bM\s*\d{1,2}\b',  # bare "M38" (word boundary avoids mid-token matches)
        r'\b(?:I am|I\'m|im)\s*(?:a\s*)?(?:guy|male|man|dude)\b',
        r'\bhe/him\b',
        r'\bmale\b',
        r'\bguy\b',
        r'\bman\b',
    ],
    'female': [
        r'\b(\d{1,2})\s*F\b',  # "23F"
        r'\(\s*F\s*\d{1,2}\s*\)',  # "(F19)"
        r'\bF\s*\d{1,2}\b',  # bare "F22"
        r'\b(?:I am|I\'m|im)\s*(?:a\s*)?(?:girl|female|woman|gal)\b',
        r'\bshe/her\b',
        r'\bfemale\b',
        r'\bgirl\b',
        r'\bwoman\b',
    ],
    'nonbinary': [
        r'\bthey/them\b',
        r'\benby\b',
        r'\bnon-?binary\b',
        r'\bnb\b',
        r'\bnonbinary\b',
    ]
}


def extract_age_from_text(text: str) -> Optional[int]:
    """
    Extract age from text using regex patterns.
    
    Args:
        text: Text to search
        
    Returns:
        Extracted age (13-80) or None if not found
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    for pattern in AGE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            try:
                age = int(matches[0])
                # Validate reasonable age range
                if 13 <= age <= 80:
                    return age
            except (ValueError, IndexError):
                continue
    
    return None


def extract_gender_from_text(text: str) -> Optional[str]:
    """
    Extract gender from text using regex patterns.
    
    Args:
        text: Text to search
        
    Returns:
        'male', 'female', 'nonbinary', or None
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Check each gender category
    for gender, patterns in GENDER_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return gender
    
    return None


def age_to_bucket(age: int) -> str:
    """
    Convert age to age bucket.
    
    Args:
        age: Age in years
        
    Returns:
        Age bucket string
    """
    if 13 <= age <= 18:
        return "13-18"
    elif 19 <= age <= 25:
        return "19-25"
    elif 26 <= age <= 40:
        return "26-40"
    elif 41 <= age <= 60:
        return "41-60"
    elif 61 <= age <= 80:
        return "61-80"
    else:
        return "unknown"


def extract_self_declarations(
    df: pd.DataFrame,
    text_column: str = "body",
    flair_column: str = "author_flair_text"
) -> pd.DataFrame:
    """
    Extract self-declared age and gender from comments.
    
    Args:
        df: DataFrame with comments
        text_column: Name of column with comment text
        flair_column: Name of column with author flair
        
    Returns:
        DataFrame with additional columns:
        - age_self_declared: Extracted age (int)
        - age_bucket_self_declared: Age bucket (str)
        - gender_self_declared: Gender (str)
    """
    logger.info("Extracting self-declarations from comments")
    
    result_df = df.copy()
    
    # Initialize columns
    result_df["age_self_declared"] = None
    result_df["age_bucket_self_declared"] = None
    result_df["gender_self_declared"] = None
    
    # Extract from body text
    logger.info("Extracting from comment text...")
    result_df["age_from_body"] = result_df[text_column].apply(extract_age_from_text)
    result_df["gender_from_body"] = result_df[text_column].apply(extract_gender_from_text)
    
    # Extract from flair if available
    if flair_column in result_df.columns:
        logger.info("Extracting from author flair...")
        result_df["age_from_flair"] = result_df[flair_column].apply(
            lambda x: extract_age_from_text(str(x)) if pd.notna(x) else None
        )
        result_df["gender_from_flair"] = result_df[flair_column].apply(
            lambda x: extract_gender_from_text(str(x)) if pd.notna(x) else None
        )
        
        # Combine body and flair (flair takes precedence if both present)
        result_df["age_self_declared"] = result_df.apply(
            lambda row: row["age_from_flair"] if pd.notna(row["age_from_flair"]) 
                       else row["age_from_body"],
            axis=1
        )
        result_df["gender_self_declared"] = result_df.apply(
            lambda row: row["gender_from_flair"] if pd.notna(row["gender_from_flair"])
                       else row["gender_from_body"],
            axis=1
        )
    else:
        result_df["age_self_declared"] = result_df["age_from_body"]
        result_df["gender_self_declared"] = result_df["gender_from_body"]
    
    # Convert age to bucket
    result_df["age_bucket_self_declared"] = result_df["age_self_declared"].apply(
        lambda x: age_to_bucket(x) if pd.notna(x) and isinstance(x, (int, float)) else None
    )
    
    # Aggregate to user level
    logger.info("Aggregating to user level...")
    user_declarations = result_df.groupby("author").agg({
        "age_self_declared": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None,
        "age_bucket_self_declared": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None,
        "gender_self_declared": lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None,
    }).reset_index()
    
    logger.info(
        f"Found self-declarations for {user_declarations['age_self_declared'].notna().sum()} users "
        f"({user_declarations['gender_self_declared'].notna().sum()} for gender)"
    )
    
    return user_declarations

