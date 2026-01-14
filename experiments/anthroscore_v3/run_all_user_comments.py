"""
Run LLM-based AnthroScore V3 on ALL comments from research users.

Processes all comments from users used in research analyses (~27k users, ~167k comments).
Cost: ~$10, Time: ~23 hours

Usage:
    python run_all_user_comments.py --dry-run  # Estimate only
    python run_all_user_comments.py            # Actually run (with confirmation)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
from typing import Set

from anthroscore_llm import AnthroScoreLLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / 'all_comments_run.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_research_users() -> Set[str]:
    """
    Get the set of users actually used in research analyses.
    
    Returns set of usernames from users with both age and gender predictions.
    Returns None to process all users.
    """
    # For now, process ALL users (user requested "all user comments")
    # If needed later, can filter to specific research users
    return None


def estimate_cost_for_users(n_comments: int, n_users: int, model: str = "gpt-4.1-nano") -> dict:
    """Estimate cost and time for processing."""
    
    avg_input_tokens = 400
    avg_output_tokens = 50
    
    costs = {
        'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
        'gpt-5-nano': {'input': 0.05, 'output': 0.20},
    }
    
    model_costs = costs.get(model, costs['gpt-4.1-nano'])
    
    total_input_tokens = n_comments * avg_input_tokens
    total_output_tokens = n_comments * avg_output_tokens
    
    input_cost = (total_input_tokens / 1_000_000) * model_costs['input']
    output_cost = (total_output_tokens / 1_000_000) * model_costs['output']
    total_cost = input_cost + output_cost
    
    estimated_hours = n_comments * 0.5 / 3600
    
    return {
        'n_comments': n_comments,
        'n_users': n_users,
        'model': model,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'estimated_hours': estimated_hours
    }


def run_all_user_comments(
    model: str = "gpt-4.1-nano",
    checkpoint_interval: int = 1000,
    dry_run: bool = False
):
    """
    Run LLM scorer on all comments from research users.
    
    Args:
        model: LLM model to use
        checkpoint_interval: Save progress every N comments
        dry_run: If True, just estimate cost without running
    """
    
    # Get research users
    research_users = get_research_users()
    
    # Load all comments
    comments_path = PROJECT_ROOT / "Data/processed/all_comments.parquet"
    if not comments_path.exists():
        logger.error(f"Comments file not found: {comments_path}")
        return
    
    logger.info(f"Loading comments from: {comments_path}")
    df = pd.read_parquet(comments_path)
    
    # Process all users (user requested "all user comments")
    logger.info("Processing ALL comments from ALL users")
    
    # Filter for quality (same as test set)
    df = df[
        (df['body'].str.len() >= 20) &
        (df['body'].str.len() <= 2000) &
        (df['body'].notna()) &
        (~df['body'].str.contains(r'^\[deleted\]$|^\[removed\]$', regex=True, na=False))
    ].copy()
    
    n_comments = len(df)
    n_users = df['author'].nunique()
    
    logger.info(f"Total comments to process: {n_comments:,}")
    logger.info(f"Total users: {n_users:,}")
    
    # Estimate cost
    estimate = estimate_cost_for_users(n_comments, n_users, model)
    
    print("\n" + "="*70)
    print("ALL USER COMMENTS PROCESSING ESTIMATE")
    print("="*70)
    print(f"Users: {estimate['n_users']:,}")
    print(f"Comments to process: {estimate['n_comments']:,}")
    print(f"Model: {estimate['model']}")
    print(f"Estimated cost: ${estimate['total_cost']:.2f}")
    print(f"  - Input: ${estimate['input_cost']:.2f}")
    print(f"  - Output: ${estimate['output_cost']:.2f}")
    print(f"Estimated time: {estimate['estimated_hours']:.1f} hours ({estimate['estimated_hours']/24:.1f} days)")
    print("="*70)
    
    if dry_run:
        print("\nDry run complete. Use --run to actually process.")
        return
    
    # Confirm
    print("\n⚠️  This will process ALL comments from research users.")
    print(f"   Estimated cost: ${estimate['total_cost']:.2f}")
    print(f"   Estimated time: {estimate['estimated_hours']:.1f} hours")
    confirm = input("\nProceed with processing? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return
    
    # Initialize scorer
    scorer = AnthroScoreLLM(model=model)
    
    # Output file
    output_path = PROJECT_ROOT / "Data/features/anthroscore_llm_all_comments.parquet"
    checkpoint_path = Path(__file__).parent / "checkpoint_all_comments.parquet"
    
    # Check for existing checkpoint
    start_idx = 0
    processed_comment_ids = set()
    if checkpoint_path.exists():
        checkpoint_df = pd.read_parquet(checkpoint_path)
        start_idx = len(checkpoint_df)
        processed_comment_ids = set(checkpoint_df['comment_id'].unique())
        logger.info(f"Resuming from checkpoint: {start_idx:,} comments already processed")
    
    # Process all comments
    results = []
    start_time = time.time()
    
    for i, row in df.iterrows():
        comment_id = row.get('id', i)
        
        # Skip if already processed
        if comment_id in processed_comment_ids:
            continue
        
        idx = len(results) + start_idx
        
        try:
            result = scorer.score_text(row['body'])
            results.append({
                'comment_id': comment_id,
                'author': row.get('author', 'unknown'),
                'subreddit': row.get('subreddit', 'unknown'),
                'created_utc': row.get('created_utc'),
                'anthroscore_llm': result.score,
                'llm_reasoning': result.reasoning,
                'llm_confidence': result.confidence,
                'llm_model': result.model,
                'llm_time_ms': result.processing_time_ms
            })
        except Exception as e:
            logger.error(f"Error on comment {idx}: {e}")
            results.append({
                'comment_id': comment_id,
                'author': row.get('author', 'unknown'),
                'subreddit': row.get('subreddit', 'unknown'),
                'created_utc': row.get('created_utc'),
                'anthroscore_llm': 0,
                'llm_reasoning': f"ERROR: {e}",
                'llm_confidence': 0,
                'llm_model': model,
                'llm_time_ms': 0
            })
        
        # Progress
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_comments - idx - 1) / rate / 3600 if rate > 0 else 0
            logger.info(f"Progress: {idx + 1:,}/{n_comments:,} ({100*(idx+1)/n_comments:.1f}%) | Rate: {rate:.1f}/s | ETA: {remaining:.1f}h")
        
        # Checkpoint
        if (idx + 1) % checkpoint_interval == 0:
            # Load existing checkpoint and merge
            if checkpoint_path.exists():
                existing = pd.read_parquet(checkpoint_path)
                new_batch = pd.DataFrame(results)
                checkpoint_df = pd.concat([existing, new_batch], ignore_index=True)
            else:
                checkpoint_df = pd.DataFrame(results)
            
            checkpoint_df.to_parquet(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved: {len(checkpoint_df):,} comments")
            results = []  # Clear buffer
    
    # Save final results
    if results:
        if checkpoint_path.exists():
            existing = pd.read_parquet(checkpoint_path)
            new_batch = pd.DataFrame(results)
            final_df = pd.concat([existing, new_batch], ignore_index=True)
        else:
            final_df = pd.DataFrame(results)
    else:
        final_df = pd.read_parquet(checkpoint_path)
    
    final_df.to_parquet(output_path, index=False)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    elapsed_hours = (time.time() - start_time) / 3600
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total comments processed: {len(final_df):,}")
    print(f"Total users: {final_df['author'].nunique():,}")
    print(f"Total time: {elapsed_hours:.2f} hours ({elapsed_hours/24:.2f} days)")
    print(f"Output saved to: {output_path}")
    print("="*70)
    
    # Summary statistics
    valid_scores = final_df[final_df['anthroscore_llm'] > 0]
    print(f"\nScore distribution:")
    print(valid_scores['anthroscore_llm'].value_counts().sort_index())
    print(f"\nMean: {valid_scores['anthroscore_llm'].mean():.2f}")
    print(f"Std: {valid_scores['anthroscore_llm'].std():.2f}")
    print(f"Valid scores: {len(valid_scores):,}/{len(final_df):,} ({100*len(valid_scores)/len(final_df):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run LLM AnthroScore on all comments from research users")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only")
    parser.add_argument("--run", action="store_true", help="Actually run (requires --run flag)")
    
    args = parser.parse_args()
    
    # Safety: require explicit --run flag
    if not args.dry_run and not args.run:
        print("⚠️  This will process many comments and cost money.")
        print("   Use --dry-run to estimate cost, or --run to actually process.")
        return
    
    run_all_user_comments(model=args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
