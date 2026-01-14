"""
Run smart routing scorer on all comments.

This script processes all comments using the smart routing system,
which adapts model selection to difficulty and importance.

Expected savings: ~70-80% vs using expert model everywhere
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
from typing import Dict, Set

from smart_routing_scorer import SmartRoutingScorer
from smart_routing_scorer_parallel import AsyncSmartRoutingScorer
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / 'smart_routing_run.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_research_users() -> Set[str]:
    """
    Get users in research sample (for importance weighting).
    
    Returns set of usernames, or None if not available.
    """
    # Try to load from demographics file
    demos_path = PROJECT_ROOT / "Data/features/ultimate_predictor/all_features.parquet"
    if not demos_path.exists():
        return None
    
    try:
        demos = pd.read_parquet(demos_path)
        if 'author' in demos.columns:
            research_users = set(demos['author'].unique())
            logger.info(f"Found {len(research_users):,} research users for importance weighting")
            return research_users
    except Exception as e:
        logger.warning(f"Could not load research users: {e}")
    
    return None


def build_user_importance_map(df: pd.DataFrame, research_users: Set[str] = None) -> Dict[str, float]:
    """
    Build importance map for users.
    
    Args:
        df: Comments dataframe
        research_users: Set of usernames in research sample
        
    Returns:
        Dict mapping username to importance score
    """
    importance_map = {}
    
    # Count comments per user
    user_counts = df.groupby('author').size()
    
    for username, count in user_counts.items():
        importance = 1.0
        
        # Research users are more important
        if research_users and username in research_users:
            importance *= 2.0
        
        # High engagement users are more important
        if count > 50:
            importance *= 1.5
        
        importance_map[username] = importance
    
    return importance_map


def estimate_smart_routing_cost(n_comments: int, escalation_rate: float = 0.15) -> dict:
    """
    Estimate cost for smart routing approach.
    
    Args:
        n_comments: Total number of comments
        escalation_rate: Fraction that get escalated (default 15%)
        
    Returns:
        Dict with cost breakdown
    """
    # Model costs (per 1M tokens) - GPT-4.1/GPT-5 series
    tier1_cost = 0.10  # GPT-4.1-nano input
    tier2_cost = 0.05  # GPT-5-nano input (cheaper!)
    tier3_cost = 0.25  # GPT-5-mini input (expert tier)
    
    avg_input = 400
    avg_output = 50
    
    # All get Tier 1
    tier1_comments = n_comments
    tier1_cost_total = (tier1_comments * avg_input / 1_000_000) * tier1_cost + \
                       (tier1_comments * avg_output / 1_000_000) * 0.40
    
    # Some get Tier 2 (assume 10% escalate to Tier 2)
    tier2_comments = int(n_comments * escalation_rate * 0.67)  # 2/3 of escalations
    tier2_cost_total = (tier2_comments * avg_input / 1_000_000) * tier2_cost + \
                       (tier2_comments * avg_output / 1_000_000) * 0.20
    
    # Some get Tier 3 (assume 5% escalate to Tier 3)
    tier3_comments = int(n_comments * escalation_rate * 0.33)  # 1/3 of escalations
    tier3_cost_total = (tier3_comments * 500 / 1_000_000) * tier3_cost + \
                       (tier3_comments * 100 / 1_000_000) * 1.00
    
    total_cost = tier1_cost_total + tier2_cost_total + tier3_cost_total
    
    # Compare to expert-only approach (GPT-5-mini)
    expert_cost_per_comment = (500 / 1_000_000) * tier3_cost + (100 / 1_000_000) * 1.00
    expert_total_cost = n_comments * expert_cost_per_comment
    
    savings = expert_total_cost - total_cost
    savings_pct = 100 * savings / expert_total_cost if expert_total_cost > 0 else 0
    
    return {
        'n_comments': n_comments,
        'tier1_count': tier1_comments,
        'tier2_count': tier2_comments,
        'tier3_count': tier3_comments,
        'tier1_cost': tier1_cost_total,
        'tier2_cost': tier2_cost_total,
        'tier3_cost': tier3_cost_total,
        'total_cost': total_cost,
        'expert_total_cost': expert_total_cost,
        'savings': savings,
        'savings_pct': savings_pct
    }


def run_smart_routing(
    checkpoint_interval: int = 1000,
    dry_run: bool = False
):
    """
    Run smart routing scorer on all comments.
    
    Args:
        checkpoint_interval: Save progress every N comments
        dry_run: If True, just estimate cost without running
    """
    # Load comments
    comments_path = PROJECT_ROOT / "Data/processed/all_comments.parquet"
    if not comments_path.exists():
        logger.error(f"Comments file not found: {comments_path}")
        return
    
    logger.info(f"Loading comments from: {comments_path}")
    df = pd.read_parquet(comments_path)
    
    # Filter for quality
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
    
    # Get research users for importance weighting
    research_users = get_research_users()
    
    # Build importance map
    user_importance_map = build_user_importance_map(df, research_users)
    
    # Estimate cost
    estimate = estimate_smart_routing_cost(n_comments, escalation_rate=0.15)
    
    print("\n" + "="*70)
    print("SMART ROUTING PROCESSING ESTIMATE")
    print("="*70)
    print(f"Users: {n_users:,}")
    print(f"Comments to process: {estimate['n_comments']:,}")
    print(f"\nRouting breakdown:")
    print(f"  Tier 1 (cheap): {estimate['tier1_count']:,} ({100*estimate['tier1_count']/n_comments:.1f}%)")
    print(f"  Tier 2 (medium): {estimate['tier2_count']:,} ({100*estimate['tier2_count']/n_comments:.1f}%)")
    print(f"  Tier 3 (expert): {estimate['tier3_count']:,} ({100*estimate['tier3_count']/n_comments:.1f}%)")
    print(f"\nCost breakdown:")
    print(f"  Tier 1: ${estimate['tier1_cost']:.2f}")
    print(f"  Tier 2: ${estimate['tier2_cost']:.2f}")
    print(f"  Tier 3: ${estimate['tier3_cost']:.2f}")
    print(f"  Total: ${estimate['total_cost']:.2f}")
    print(f"\nComparison:")
    print(f"  Expert-only cost: ${estimate['expert_total_cost']:.2f}")
    print(f"  Savings: ${estimate['savings']:.2f} ({estimate['savings_pct']:.1f}%)")
    print("="*70)
    
    if dry_run:
        print("\nDry run complete. Use --run to actually process.")
        return
    
    # Confirm
    print("\n⚠️  This will process ALL comments with smart routing.")
    print(f"   Estimated cost: ${estimate['total_cost']:.2f}")
    print(f"   Estimated savings vs expert-only: ${estimate['savings']:.2f} ({estimate['savings_pct']:.1f}%)")
    confirm = input("\nProceed with processing? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return
    
    # Initialize async scorer (much faster!)
    scorer = AsyncSmartRoutingScorer(
        user_importance_map=user_importance_map,
        max_concurrent=20  # Process 20 comments concurrently
    )
    
    # Output file
    output_path = PROJECT_ROOT / "Data/features/anthroscore_smart_routing.parquet"
    checkpoint_path = Path(__file__).parent / "checkpoint_smart_routing.parquet"
    
    # Check for existing checkpoint
    start_idx = 0
    processed_comment_ids = set()
    if checkpoint_path.exists():
        checkpoint_df = pd.read_parquet(checkpoint_path)
        start_idx = len(checkpoint_df)
        processed_comment_ids = set(checkpoint_df['comment_id'].unique())
        logger.info(f"Resuming from checkpoint: {start_idx:,} comments already processed")
    
    # Filter to unprocessed comments
    df_to_process = df[~df.get('id', df.index).isin(processed_comment_ids)].copy()
    logger.info(f"Processing {len(df_to_process):,} new comments")
    
    # Get user comment counts for importance
    user_counts = df.groupby('author').size().to_dict()
    
    # Process in batches (async)
    batch_size = 1000  # Process 1000 at a time
    all_results = []
    start_time = time.time()
    
    async def process_batch(batch_df):
        """Process a batch of comments asynchronously."""
        texts = batch_df['body'].tolist()
        usernames = batch_df.get('author', 'unknown').tolist()
        comment_counts = [user_counts.get(u, 1) for u in usernames]
        
        # Score batch
        results = await scorer.score_batch_async(
            texts, usernames, comment_counts,
            progress_interval=100
        )
        
        # Convert to dict format
        batch_results = []
        for i, (row, result) in enumerate(zip(batch_df.itertuples(), results)):
            batch_results.append({
                'comment_id': getattr(row, 'id', i),
                'author': getattr(row, 'author', 'unknown'),
                'subreddit': getattr(row, 'subreddit', 'unknown'),
                'created_utc': getattr(row, 'created_utc', None),
                'anthroscore_smart': result.score,
                'smart_reasoning': result.reasoning,
                'smart_tier': result.tier_used,
                'smart_model': result.model_used,
                'smart_routing_reason': result.routing_reason,
                'smart_confidence': result.confidence,
                'smart_cost_usd': result.total_cost_usd,
                'smart_time_ms': result.total_time_ms,
                'smart_escalated': result.escalation_occurred
            })
        
        return batch_results
    
    # Process all batches
    for batch_start in range(0, len(df_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(df_to_process))
        batch_df = df_to_process.iloc[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: comments {batch_start:,}-{batch_end:,}")
        
        try:
            batch_results = asyncio.run(process_batch(batch_df))
            all_results.extend(batch_results)
            
            # Progress
            processed = len(all_results) + start_idx
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (n_comments - processed) / rate / 3600 if rate > 0 else 0
            stats = scorer.get_stats()
            logger.info(
                f"Overall progress: {processed:,}/{n_comments:,} ({100*processed/n_comments:.1f}%) | "
                f"Rate: {rate:.1f}/s | ETA: {remaining:.1f}h | "
                f"Tier1: {stats['tier1_pct']:.1f}% | Tier2: {stats['tier2_pct']:.1f}% | Tier3: {stats['tier3_pct']:.1f}%"
            )
            
            # Checkpoint
            if processed % checkpoint_interval == 0 or batch_end >= len(df_to_process):
                if checkpoint_path.exists():
                    existing = pd.read_parquet(checkpoint_path)
                    new_batch = pd.DataFrame(all_results)
                    checkpoint_df = pd.concat([existing, new_batch], ignore_index=True)
                else:
                    checkpoint_df = pd.DataFrame(all_results)
                
                checkpoint_df.to_parquet(checkpoint_path, index=False)
                logger.info(f"Checkpoint saved: {len(checkpoint_df):,} comments")
                all_results = []  # Clear buffer
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_start//batch_size + 1}: {e}")
            # Continue with next batch
    
    # Save final results
    if all_results:
        if checkpoint_path.exists():
            existing = pd.read_parquet(checkpoint_path)
            new_batch = pd.DataFrame(all_results)
            final_df = pd.concat([existing, new_batch], ignore_index=True)
        else:
            final_df = pd.DataFrame(all_results)
    elif checkpoint_path.exists():
        final_df = pd.read_parquet(checkpoint_path)
    else:
        logger.error("No results to save!")
        return
    
    final_df.to_parquet(output_path, index=False)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    elapsed_hours = (time.time() - start_time) / 3600
    stats = scorer.get_stats()
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total comments processed: {len(final_df):,}")
    print(f"Total users: {final_df['author'].nunique():,}")
    print(f"Total time: {elapsed_hours:.2f} hours ({elapsed_hours/24:.2f} days)")
    print(f"\nRouting statistics:")
    print(f"  Tier 1: {stats['tier1_count']:,} ({stats['tier1_pct']:.1f}%)")
    print(f"  Tier 2: {stats['tier2_count']:,} ({stats['tier2_pct']:.1f}%)")
    print(f"  Tier 3: {stats['tier3_count']:,} ({stats['tier3_pct']:.1f}%)")
    print(f"\nCost:")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    print(f"  Avg cost/comment: ${stats['avg_cost_per_comment']:.6f}")
    print(f"\nOutput saved to: {output_path}")
    print("="*70)
    
    # Save statistics
    stats_path = Path(__file__).parent / "smart_routing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Run smart routing AnthroScore on all comments")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only")
    parser.add_argument("--run", action="store_true", help="Actually run (requires --run flag)")
    
    args = parser.parse_args()
    
    # Safety: require explicit --run flag
    if not args.dry_run and not args.run:
        print("⚠️  This will process many comments and cost money.")
        print("   Use --dry-run to estimate cost, or --run to actually process.")
        return
    
    run_smart_routing(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
