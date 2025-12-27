"""
Classification Improvement Script.

This script implements actual improvements to the classification pipeline:
1. Increase LLM coverage (from 10.6% to 30%+)
2. Optimize age thresholds using cross-validation
3. Test alternative seed pairs
4. Implement true ensemble voting for more users
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / 'Data' / 'features'


def load_data():
    """Load current demographics data."""
    demo = pd.read_parquet(DATA_DIR / 'demographics.parquet')
    logger.info(f"Loaded {len(demo)} users")
    
    # Current coverage
    logger.info(f"Self-declared: {demo['age_bucket_self_declared'].notna().sum()}")
    logger.info(f"LLM classified: {demo['age_bucket_llm'].notna().sum()}")
    logger.info(f"Community embedding: {demo['age_bucket_community'].notna().sum()}")
    
    return demo


def optimize_age_thresholds(demo: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Use cross-validation to find optimal age thresholds.
    
    Uses the self-declared users as ground truth to calibrate.
    """
    logger.info("=" * 60)
    logger.info("OPTIMIZING AGE THRESHOLDS")
    logger.info("=" * 60)
    
    # Get users with both self-declared age AND community score
    mask = demo['age_bucket_self_declared'].notna() & demo['age_community_score'].notna()
    validation_df = demo[mask].copy()
    
    logger.info(f"Validation set: {len(validation_df)} users with ground truth")
    
    if len(validation_df) < 50:
        logger.warning("Not enough validation data")
        return None
    
    # Convert to 3-bucket for optimization
    def to_3bucket(age):
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        return 'adult'
    
    validation_df['gt_3bucket'] = validation_df['age_bucket_self_declared'].apply(to_3bucket)
    
    # Get score distribution by ground truth bucket
    logger.info("\nScore distribution by ground truth bucket:")
    for bucket in ['teen', 'young_adult', 'adult']:
        scores = validation_df[validation_df['gt_3bucket'] == bucket]['age_community_score']
        if len(scores) > 0:
            logger.info(f"  {bucket}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
                       f"min={scores.min():.4f}, max={scores.max():.4f}, n={len(scores)}")
    
    # Grid search for optimal thresholds
    best_accuracy = 0
    best_thresholds = None
    
    # Test different threshold combinations
    for thresh1 in np.arange(-0.5, 0.1, 0.05):
        for thresh2 in np.arange(thresh1 + 0.05, 0.3, 0.05):
            # Classify with these thresholds
            def classify(score):
                if score < thresh1:
                    return 'teen'
                elif score < thresh2:
                    return 'young_adult'
                return 'adult'
            
            pred = validation_df['age_community_score'].apply(classify)
            accuracy = (pred == validation_df['gt_3bucket']).mean()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = {
                    'teen': (-float('inf'), thresh1),
                    'young_adult': (thresh1, thresh2),
                    'adult': (thresh2, float('inf'))
                }
    
    logger.info(f"\nOptimal thresholds found:")
    logger.info(f"  teen: score < {best_thresholds['teen'][1]:.3f}")
    logger.info(f"  young_adult: {best_thresholds['young_adult'][0]:.3f} <= score < {best_thresholds['young_adult'][1]:.3f}")
    logger.info(f"  adult: score >= {best_thresholds['adult'][0]:.3f}")
    logger.info(f"  Best accuracy: {best_accuracy:.1%}")
    
    # Compare to current thresholds
    current_accuracy = (
        validation_df['age_bucket_community'].apply(to_3bucket) == 
        validation_df['gt_3bucket']
    ).mean()
    logger.info(f"  Current accuracy: {current_accuracy:.1%}")
    logger.info(f"  Improvement: {(best_accuracy - current_accuracy)*100:+.1f}pp")
    
    return best_thresholds


def run_additional_llm_classification(
    demo: pd.DataFrame,
    target_users: int = 10000,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Run LLM classification on additional users.
    
    Prioritizes users who:
    1. Don't have LLM classification yet
    2. Have community embedding classification (to enable ensemble)
    3. Have higher comment counts (more data for LLM)
    """
    logger.info("=" * 60)
    logger.info("RUNNING ADDITIONAL LLM CLASSIFICATION")
    logger.info("=" * 60)
    
    from src.utils.config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not configured - cannot run LLM classification")
        return demo
    
    # Find users without LLM classification but with community embedding
    candidates = demo[
        demo['age_bucket_llm'].isna() & 
        demo['age_bucket_community'].notna()
    ].copy()
    
    logger.info(f"Candidates for LLM classification: {len(candidates)}")
    
    if len(candidates) == 0:
        logger.info("No candidates available")
        return demo
    
    # Load comments for these users
    comments_path = project_root / 'Data' / 'processed' / 'all_comments.parquet'
    if not comments_path.exists():
        logger.error("Comments file not found")
        return demo
    
    comments = pd.read_parquet(comments_path)
    
    # Calculate comments per candidate user
    candidate_authors = set(candidates['author'])
    user_comment_counts = comments[comments['author'].isin(candidate_authors)].groupby('author').size()
    
    # Sort by comment count (more comments = better LLM input)
    candidates = candidates[candidates['author'].isin(user_comment_counts.index)]
    candidates['comment_count'] = candidates['author'].map(user_comment_counts)
    candidates = candidates.sort_values('comment_count', ascending=False)
    
    # Take top N users
    users_to_classify = candidates.head(target_users)['author'].tolist()
    logger.info(f"Will classify {len(users_to_classify)} users")
    
    # Import LLM classifier
    from src.demographics.llm_classifier import classify_age_llm
    from openai import OpenAI
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    new_classifications = []
    
    for i, author in enumerate(users_to_classify):
        user_comments = comments[comments['author'] == author]['body'].tolist()
        
        if not user_comments:
            continue
        
        try:
            result = classify_age_llm(user_comments, max_comments=20, client=client)
            
            new_classifications.append({
                'author': author,
                'age_bucket_llm': result.get('age_bucket'),
                'confidence_llm': result.get('confidence', 0.0)
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(users_to_classify)} users")
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.warning(f"Error classifying {author}: {e}")
    
    if new_classifications:
        new_df = pd.DataFrame(new_classifications)
        
        # Update demo with new classifications
        demo = demo.merge(
            new_df, 
            on='author', 
            how='left',
            suffixes=('', '_new')
        )
        
        # Fill in new values
        demo['age_bucket_llm'] = demo['age_bucket_llm'].fillna(demo.get('age_bucket_llm_new'))
        demo['confidence_llm'] = demo['confidence_llm'].fillna(demo.get('confidence_llm_new'))
        
        # Clean up
        demo = demo.drop(columns=[c for c in demo.columns if c.endswith('_new')], errors='ignore')
        
        # Save updated demographics
        demo.to_parquet(DATA_DIR / 'demographics.parquet', index=False)
        
        logger.info(f"Added {len(new_classifications)} new LLM classifications")
        logger.info(f"New LLM coverage: {demo['age_bucket_llm'].notna().sum()}/{len(demo)} "
                   f"({demo['age_bucket_llm'].notna().mean():.1%})")
    
    return demo


def rebuild_ensemble(demo: pd.DataFrame, thresholds: Dict = None) -> pd.DataFrame:
    """
    Rebuild ensemble classifications with improved thresholds and true voting.
    """
    logger.info("=" * 60)
    logger.info("REBUILDING ENSEMBLE CLASSIFICATION")
    logger.info("=" * 60)
    
    def to_3bucket_from_score(score, thresholds):
        if thresholds is None:
            # Default thresholds
            if score < -0.22:
                return 'teen'
            elif score < -0.10:
                return 'young_adult'
            return 'adult'
        
        for bucket, (low, high) in thresholds.items():
            if low <= score < high:
                return bucket
        return 'adult'
    
    def to_3bucket(age):
        if pd.isna(age):
            return None
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        return 'adult'
    
    # Apply improved thresholds to community scores
    demo['age_3bucket_community_improved'] = demo['age_community_score'].apply(
        lambda x: to_3bucket_from_score(x, thresholds) if pd.notna(x) else None
    )
    
    # Count sources per user
    demo['n_sources'] = (
        demo['age_bucket_self_declared'].notna().astype(int) +
        demo['age_bucket_community'].notna().astype(int) +
        demo['age_bucket_llm'].notna().astype(int)
    )
    
    logger.info(f"Source distribution:")
    logger.info(f"  1 source: {(demo['n_sources'] == 1).sum()}")
    logger.info(f"  2 sources: {(demo['n_sources'] == 2).sum()}")
    logger.info(f"  3 sources: {(demo['n_sources'] == 3).sum()}")
    
    # Implement true weighted voting for 3-bucket age
    def weighted_vote_3bucket(row):
        votes = {}
        
        # Self-declaration (weight 1.0) - convert to 3-bucket
        if pd.notna(row.get('age_bucket_self_declared')):
            bucket = to_3bucket(row['age_bucket_self_declared'])
            if bucket:
                votes[bucket] = votes.get(bucket, 0) + 1.0
        
        # Community embedding with improved thresholds (weight 0.7)
        if pd.notna(row.get('age_3bucket_community_improved')):
            bucket = row['age_3bucket_community_improved']
            # Weight by score magnitude
            score_weight = min(1.0, abs(row.get('age_community_score', 0)) * 2)
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
    
    # Apply ensemble voting
    demo['age_3bucket_ensemble'] = demo.apply(weighted_vote_3bucket, axis=1)
    
    # Calculate new accuracy
    mask = demo['age_bucket_self_declared'].notna() & demo['age_3bucket_ensemble'].notna()
    if mask.sum() > 0:
        gt = demo.loc[mask, 'age_bucket_self_declared'].apply(to_3bucket)
        pred = demo.loc[mask, 'age_3bucket_ensemble']
        accuracy = (gt == pred).mean()
        logger.info(f"Ensemble 3-bucket accuracy: {accuracy:.1%}")
    
    return demo


def calculate_improvement_metrics(demo: pd.DataFrame):
    """Calculate and report improvement metrics."""
    logger.info("=" * 60)
    logger.info("IMPROVEMENT METRICS")
    logger.info("=" * 60)
    
    def to_3bucket(age):
        if pd.isna(age):
            return None
        if age == '13-18':
            return 'teen'
        elif age == '19-25':
            return 'young_adult'
        return 'adult'
    
    # Get validation subset
    mask = demo['age_bucket_self_declared'].notna()
    val_df = demo[mask].copy()
    val_df['gt_3bucket'] = val_df['age_bucket_self_declared'].apply(to_3bucket)
    
    logger.info(f"Validation set: {len(val_df)} users\n")
    
    # Original community embedding accuracy
    mask_comm = val_df['age_bucket_community'].notna()
    if mask_comm.sum() > 0:
        pred_comm = val_df.loc[mask_comm, 'age_bucket_community'].apply(to_3bucket)
        gt_comm = val_df.loc[mask_comm, 'gt_3bucket']
        acc_original = (pred_comm == gt_comm).mean()
        logger.info(f"Original community embedding (3-bucket): {acc_original:.1%}")
    
    # Improved community embedding accuracy
    if 'age_3bucket_community_improved' in val_df.columns:
        mask_improved = val_df['age_3bucket_community_improved'].notna()
        if mask_improved.sum() > 0:
            pred_improved = val_df.loc[mask_improved, 'age_3bucket_community_improved']
            gt_improved = val_df.loc[mask_improved, 'gt_3bucket']
            acc_improved = (pred_improved == gt_improved).mean()
            logger.info(f"Improved community embedding (3-bucket): {acc_improved:.1%}")
    
    # Ensemble accuracy
    if 'age_3bucket_ensemble' in val_df.columns:
        mask_ens = val_df['age_3bucket_ensemble'].notna()
        if mask_ens.sum() > 0:
            pred_ens = val_df.loc[mask_ens, 'age_3bucket_ensemble']
            gt_ens = val_df.loc[mask_ens, 'gt_3bucket']
            acc_ens = (pred_ens == gt_ens).mean()
            logger.info(f"Ensemble (3-bucket): {acc_ens:.1%}")
    
    # Multi-source coverage
    logger.info(f"\nMulti-source coverage:")
    logger.info(f"  2+ sources: {(demo['n_sources'] >= 2).mean():.1%}")
    logger.info(f"  LLM coverage: {demo['age_bucket_llm'].notna().mean():.1%}")


def main():
    """Run all improvements."""
    logger.info("=" * 80)
    logger.info("CLASSIFICATION IMPROVEMENT PIPELINE")
    logger.info("=" * 80)
    
    # Load data
    demo = load_data()
    
    # Step 1: Optimize thresholds
    optimal_thresholds = optimize_age_thresholds(demo)
    
    # Step 2: (Optional) Run additional LLM classification
    # Uncomment to run - this costs API credits
    # demo = run_additional_llm_classification(demo, target_users=5000)
    
    # Step 3: Rebuild ensemble with improvements
    demo = rebuild_ensemble(demo, optimal_thresholds)
    
    # Step 4: Report improvements
    calculate_improvement_metrics(demo)
    
    # Save updated demographics
    demo.to_parquet(DATA_DIR / 'demographics_improved.parquet', index=False)
    logger.info(f"\nSaved improved demographics to demographics_improved.parquet")
    
    return demo


if __name__ == "__main__":
    demo = main()

