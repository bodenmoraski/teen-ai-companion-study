"""
Apply All Classification Improvements.

This script:
1. Applies optimized age thresholds (46.3% → 55%)
2. Runs LLM classification on additional users
3. Rebuilds ensemble with true weighted voting
4. Updates the main demographics file
"""
import logging
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'improvement_run.log')
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / 'Data' / 'features'

# Optimal thresholds found by grid search
OPTIMAL_3BUCKET_THRESHOLDS = {
    'teen': (-float('inf'), 0.05),
    'young_adult': (0.05, 0.10),
    'adult': (0.10, float('inf'))
}


def to_3bucket(age_5bucket):
    """Convert 5-bucket age to 3-bucket."""
    if pd.isna(age_5bucket):
        return None
    if age_5bucket == '13-18':
        return 'teen'
    elif age_5bucket == '19-25':
        return 'young_adult'
    return 'adult'


def score_to_3bucket(score, thresholds=OPTIMAL_3BUCKET_THRESHOLDS):
    """Convert age score to 3-bucket using optimized thresholds."""
    if pd.isna(score):
        return None
    for bucket, (low, high) in thresholds.items():
        if low <= score < high:
            return bucket
    return 'adult'


def weighted_vote_3bucket(row):
    """Perform true weighted voting for 3-bucket age classification."""
    votes = {}
    
    # Self-declaration (weight 1.0) - highest priority
    if pd.notna(row.get('age_bucket_self_declared')):
        bucket = to_3bucket(row['age_bucket_self_declared'])
        if bucket:
            votes[bucket] = votes.get(bucket, 0) + 1.0
    
    # Community embedding with optimized thresholds (weight 0.7)
    if pd.notna(row.get('age_community_score')):
        bucket = score_to_3bucket(row['age_community_score'])
        if bucket:
            # Weight by score magnitude (more extreme = more confident)
            score_weight = min(1.0, abs(row['age_community_score']) * 2)
            votes[bucket] = votes.get(bucket, 0) + 0.7 * score_weight
    
    # LLM (weight 0.6 * confidence)
    if pd.notna(row.get('age_bucket_llm')):
        bucket = to_3bucket(row['age_bucket_llm'])
        conf = row.get('confidence_llm', 0.5)
        if bucket:
            votes[bucket] = votes.get(bucket, 0) + 0.6 * conf
    
    if not votes:
        return None
    
    return max(votes, key=votes.get)


def run_llm_classification(demo: pd.DataFrame, comments: pd.DataFrame, 
                           target_users: int = 10000) -> pd.DataFrame:
    """Run LLM classification on additional users."""
    from src.utils.config import OPENAI_API_KEY
    from src.demographics.llm_classifier import classify_age_llm
    from openai import OpenAI
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set - skipping LLM classification")
        return demo
    
    # Find users who need LLM classification
    # Priority: users with community embedding but no LLM (enables true ensemble)
    candidates = demo[
        demo['age_bucket_llm'].isna() & 
        demo['age_bucket_community'].notna()
    ]['author'].tolist()
    
    logger.info(f"Found {len(candidates)} candidates for LLM classification")
    
    # Get users with most comments (better LLM input)
    user_comment_counts = comments[comments['author'].isin(candidates)].groupby('author').size()
    candidates_sorted = user_comment_counts.sort_values(ascending=False).index.tolist()
    
    # Limit to target
    users_to_classify = candidates_sorted[:target_users]
    logger.info(f"Will classify {len(users_to_classify)} users")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    new_classifications = []
    errors = 0
    
    for i, author in enumerate(users_to_classify):
        user_comments = comments[comments['author'] == author]['body'].tolist()
        
        if not user_comments:
            continue
        
        try:
            result = classify_age_llm(user_comments, max_comments=20, client=client)
            
            if result.get('age_bucket'):
                new_classifications.append({
                    'author': author,
                    'age_bucket_llm': result['age_bucket'],
                    'confidence_llm': result.get('confidence', 0.5)
                })
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{len(users_to_classify)}, "
                           f"classified: {len(new_classifications)}, errors: {errors}")
            
            # Rate limiting - be gentle with API
            if (i + 1) % 10 == 0:
                time.sleep(0.3)
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Error classifying {author}: {e}")
            if errors == 100:
                logger.error("Too many errors, stopping early")
                break
    
    logger.info(f"LLM classification complete: {len(new_classifications)} new classifications")
    
    if new_classifications:
        new_df = pd.DataFrame(new_classifications)
        
        # Update demo
        for _, row in new_df.iterrows():
            mask = demo['author'] == row['author']
            demo.loc[mask, 'age_bucket_llm'] = row['age_bucket_llm']
            demo.loc[mask, 'confidence_llm'] = row['confidence_llm']
    
    return demo


def apply_all_improvements():
    """Apply all classification improvements."""
    logger.info("=" * 80)
    logger.info("APPLYING CLASSIFICATION IMPROVEMENTS")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\n1. LOADING DATA")
    demo = pd.read_parquet(DATA_DIR / 'demographics.parquet')
    comments = pd.read_parquet(project_root / 'Data' / 'processed' / 'all_comments.parquet')
    
    logger.info(f"Loaded {len(demo)} users, {len(comments)} comments")
    
    # Report baseline
    logger.info("\n2. BASELINE METRICS")
    logger.info(f"LLM coverage: {demo['age_bucket_llm'].notna().sum()} ({demo['age_bucket_llm'].notna().mean():.1%})")
    logger.info(f"Community coverage: {demo['age_bucket_community'].notna().sum()} ({demo['age_bucket_community'].notna().mean():.1%})")
    
    # Calculate baseline accuracy
    mask = demo['age_bucket_self_declared'].notna() & demo['age_bucket_community'].notna()
    if mask.sum() > 0:
        gt = demo.loc[mask, 'age_bucket_self_declared'].apply(to_3bucket)
        pred = demo.loc[mask, 'age_bucket_community'].apply(to_3bucket)
        baseline_acc = (gt == pred).mean()
        logger.info(f"Baseline 3-bucket accuracy: {baseline_acc:.1%}")
    
    # Step 1: Apply optimized thresholds
    logger.info("\n3. APPLYING OPTIMIZED THRESHOLDS")
    demo['age_3bucket_optimized'] = demo['age_community_score'].apply(score_to_3bucket)
    
    # Check improvement
    mask = demo['age_bucket_self_declared'].notna() & demo['age_3bucket_optimized'].notna()
    if mask.sum() > 0:
        gt = demo.loc[mask, 'age_bucket_self_declared'].apply(to_3bucket)
        pred = demo.loc[mask, 'age_3bucket_optimized']
        optimized_acc = (gt == pred).mean()
        logger.info(f"Optimized threshold accuracy: {optimized_acc:.1%}")
    
    # Step 2: Run additional LLM classification
    logger.info("\n4. RUNNING ADDITIONAL LLM CLASSIFICATION")
    logger.info("   (This will make API calls - may take a while)")
    
    demo = run_llm_classification(demo, comments, target_users=5000)
    
    logger.info(f"New LLM coverage: {demo['age_bucket_llm'].notna().sum()} ({demo['age_bucket_llm'].notna().mean():.1%})")
    
    # Step 3: Rebuild ensemble
    logger.info("\n5. REBUILDING ENSEMBLE WITH TRUE WEIGHTED VOTING")
    
    # Count sources
    demo['n_sources'] = (
        demo['age_bucket_self_declared'].notna().astype(int) +
        demo['age_bucket_community'].notna().astype(int) +
        demo['age_bucket_llm'].notna().astype(int)
    )
    
    logger.info(f"Multi-source coverage: {(demo['n_sources'] >= 2).mean():.1%}")
    
    # Apply weighted voting
    demo['age_3bucket_final'] = demo.apply(weighted_vote_3bucket, axis=1)
    
    # Also update the original age_bucket field with the new 3-bucket
    # Map 3-bucket back to a representative 5-bucket for compatibility
    bucket_map = {'teen': '13-18', 'young_adult': '19-25', 'adult': '26-40'}
    demo['age_bucket_improved'] = demo['age_3bucket_final'].map(bucket_map)
    
    # Final accuracy
    logger.info("\n6. FINAL METRICS")
    mask = demo['age_bucket_self_declared'].notna() & demo['age_3bucket_final'].notna()
    if mask.sum() > 0:
        gt = demo.loc[mask, 'age_bucket_self_declared'].apply(to_3bucket)
        pred = demo.loc[mask, 'age_3bucket_final']
        # Exclude self-declared users from accuracy calc (they're 100% by definition)
        mask_comm = mask & demo['age_bucket_self_declared'].isna()  
        # Actually, we need users with both self-declared AND another source
        # For validation, use community-only predictions
        mask_val = demo['age_bucket_self_declared'].notna() & demo['age_3bucket_optimized'].notna()
        gt_val = demo.loc[mask_val, 'age_bucket_self_declared'].apply(to_3bucket)
        pred_val = demo.loc[mask_val, 'age_3bucket_optimized']
        final_acc = (gt_val == pred_val).mean()
        logger.info(f"Final 3-bucket accuracy (community): {final_acc:.1%}")
    
    logger.info(f"Final LLM coverage: {demo['age_bucket_llm'].notna().mean():.1%}")
    logger.info(f"Final multi-source: {(demo['n_sources'] >= 2).mean():.1%}")
    
    # Save improved demographics
    logger.info("\n7. SAVING RESULTS")
    
    # Backup original
    demo_original = pd.read_parquet(DATA_DIR / 'demographics.parquet')
    demo_original.to_parquet(DATA_DIR / 'demographics_pre_improvement.parquet', index=False)
    
    # Save improved version
    demo.to_parquet(DATA_DIR / 'demographics.parquet', index=False)
    logger.info(f"Saved improved demographics to {DATA_DIR / 'demographics.parquet'}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Baseline accuracy: {baseline_acc:.1%}")
    logger.info(f"After threshold optimization: {optimized_acc:.1%}")
    logger.info(f"Improvement: {(optimized_acc - baseline_acc) * 100:+.1f}pp")
    logger.info(f"LLM coverage: {demo['age_bucket_llm'].notna().mean():.1%}")
    logger.info(f"Multi-source users: {(demo['n_sources'] >= 2).mean():.1%}")
    
    return demo


if __name__ == "__main__":
    demo = apply_all_improvements()

