"""
Create a stratified sample of comments for AnthroScore V3 validation.

Samples comments across different AnthroScore strata to ensure we test
the full range of anthropomorphization levels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def create_test_set(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create a stratified sample of comments for validation.
    
    Args:
        n_samples: Total number of comments to sample
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled comments
    """
    np.random.seed(seed)
    
    # Load all comments with anthroscores
    logger.info("Loading comments data...")
    
    # Try to load from processed features
    anthroscores_path = PROJECT_ROOT / "Data/features/user_anthroscores.parquet"
    comments_path = PROJECT_ROOT / "Data/processed/all_comments.parquet"
    
    if not comments_path.exists():
        raise FileNotFoundError(f"Comments file not found: {comments_path}")
    
    comments_df = pd.read_parquet(comments_path)
    logger.info(f"Loaded {len(comments_df)} comments")
    
    # Filter for quality
    # - Minimum length for meaningful analysis
    # - Has actual text content
    comments_df = comments_df[
        (comments_df['body'].str.len() >= 50) &
        (comments_df['body'].str.len() <= 2000) &
        (comments_df['body'].notna()) &
        (~comments_df['body'].str.contains(r'^\[deleted\]$|^\[removed\]$', regex=True, na=False))
    ].copy()
    
    logger.info(f"After filtering: {len(comments_df)} comments")
    
    # If we have anthroscores, stratify by them
    if anthroscores_path.exists():
        logger.info("Loading anthroscores for stratification...")
        anthroscores = pd.read_parquet(anthroscores_path)
        
        # Merge anthroscores (user-level) to comments
        if 'author' in comments_df.columns and 'author' in anthroscores.columns:
            comments_df = comments_df.merge(
                anthroscores[['author', 'anthroscore_mean', 'anthroscore_max']],
                on='author',
                how='left'
            )
    
    # Create strata based on anthroscore or comment characteristics
    if 'anthroscore_mean' in comments_df.columns:
        # Stratify by anthroscore quartiles
        comments_df['stratum'] = pd.qcut(
            comments_df['anthroscore_mean'].fillna(0),
            q=4,
            labels=['Q1_Low', 'Q2_Mid_Low', 'Q3_Mid_High', 'Q4_High'],
            duplicates='drop'
        )
    else:
        # Fallback: stratify by subreddit
        comments_df['stratum'] = comments_df['subreddit'].fillna('unknown')
    
    # Sample from each stratum
    samples_per_stratum = n_samples // comments_df['stratum'].nunique()
    
    sampled = []
    for stratum in comments_df['stratum'].unique():
        stratum_df = comments_df[comments_df['stratum'] == stratum]
        n = min(samples_per_stratum, len(stratum_df))
        if n > 0:
            sampled.append(stratum_df.sample(n=n, random_state=seed))
    
    test_set = pd.concat(sampled, ignore_index=True)
    
    # If we need more samples, add randomly
    if len(test_set) < n_samples:
        remaining = comments_df[~comments_df.index.isin(test_set.index)]
        additional = remaining.sample(n=min(n_samples - len(test_set), len(remaining)), random_state=seed)
        test_set = pd.concat([test_set, additional], ignore_index=True)
    
    # Shuffle
    test_set = test_set.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Select columns for labeling
    columns_to_keep = ['id', 'author', 'body', 'subreddit']
    if 'anthroscore_mean' in test_set.columns:
        columns_to_keep.extend(['anthroscore_mean', 'anthroscore_max', 'stratum'])
    
    test_set = test_set[[c for c in columns_to_keep if c in test_set.columns]]
    
    # Add empty columns for labels
    test_set['expert_score'] = np.nan
    test_set['expert_reasoning'] = ''
    test_set['llm_score'] = np.nan
    test_set['llm_reasoning'] = ''
    
    logger.info(f"Created test set with {len(test_set)} comments")
    logger.info(f"Strata distribution:\n{test_set['stratum'].value_counts() if 'stratum' in test_set.columns else 'N/A'}")
    
    return test_set


def main():
    """Create and save test set."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test set
    test_set = create_test_set(n_samples=100)
    
    # Save
    output_path = output_dir / "test_set_unlabeled.parquet"
    test_set.to_parquet(output_path, index=False)
    logger.info(f"Saved test set to: {output_path}")
    
    # Also save as CSV for easy viewing
    csv_path = output_dir / "test_set_unlabeled.csv"
    test_set.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV version to: {csv_path}")
    
    # Print sample
    print("\n" + "="*80)
    print("SAMPLE COMMENTS FROM TEST SET")
    print("="*80)
    for i, row in test_set.head(5).iterrows():
        print(f"\n[{i+1}] Subreddit: r/{row.get('subreddit', 'unknown')}")
        print(f"    AnthroScore (V2): {row.get('anthroscore_mean', 'N/A')}")
        print(f"    Text: {row['body'][:200]}...")
        print("-"*40)


if __name__ == "__main__":
    main()
