"""
Run LLM-based AnthroScore V3 on the full dataset.

IMPORTANT: This will make many API calls and cost approximately $6-10.
Run validation first to confirm the approach works!

Usage:
    python run_full_dataset.py --dry-run  # Estimate cost only
    python run_full_dataset.py            # Actually run (with confirmation)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json

from anthroscore_llm import AnthroScoreLLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / 'full_dataset_run.log')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def estimate_cost(n_comments: int, model: str = "gpt-4.1-nano") -> dict:
    """Estimate cost and time for processing."""
    
    # Estimated tokens per comment
    avg_input_tokens = 400  # prompt + text
    avg_output_tokens = 50  # score + reasoning
    
    # Costs per 1M tokens
    costs = {
        'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    }
    
    model_costs = costs.get(model, costs['gpt-4.1-nano'])
    
    total_input_tokens = n_comments * avg_input_tokens
    total_output_tokens = n_comments * avg_output_tokens
    
    input_cost = (total_input_tokens / 1_000_000) * model_costs['input']
    output_cost = (total_output_tokens / 1_000_000) * model_costs['output']
    total_cost = input_cost + output_cost
    
    # Time estimate (0.5s per comment)
    estimated_seconds = n_comments * 0.5
    estimated_hours = estimated_seconds / 3600
    
    return {
        'n_comments': n_comments,
        'model': model,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'estimated_hours': estimated_hours
    }


def run_full_dataset(
    model: str = "gpt-4.1-nano",
    batch_size: int = 1000,
    checkpoint_interval: int = 100,
    dry_run: bool = False
):
    """
    Run LLM scorer on the full comment dataset.
    
    Args:
        model: LLM model to use
        batch_size: Number of comments to load at once
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
    
    # Filter for quality (same as test set)
    df = df[
        (df['body'].str.len() >= 20) &
        (df['body'].str.len() <= 2000) &
        (df['body'].notna()) &
        (~df['body'].str.contains(r'^\[deleted\]$|^\[removed\]$', regex=True, na=False))
    ].copy()
    
    n_comments = len(df)
    logger.info(f"Found {n_comments} comments to score")
    
    # Estimate cost
    estimate = estimate_cost(n_comments, model)
    
    print("\n" + "="*60)
    print("FULL DATASET PROCESSING ESTIMATE")
    print("="*60)
    print(f"Comments to process: {estimate['n_comments']:,}")
    print(f"Model: {estimate['model']}")
    print(f"Estimated input tokens: {estimate['input_tokens']:,.0f}")
    print(f"Estimated output tokens: {estimate['output_tokens']:,.0f}")
    print(f"Estimated cost: ${estimate['total_cost']:.2f}")
    print(f"Estimated time: {estimate['estimated_hours']:.1f} hours")
    print("="*60)
    
    if dry_run:
        print("\nDry run complete. Use --run to actually process.")
        return
    
    # Confirm
    confirm = input("\nProceed with processing? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return
    
    # Initialize scorer
    scorer = AnthroScoreLLM(model=model)
    
    # Output file
    output_path = PROJECT_ROOT / "Data/features/anthroscore_llm.parquet"
    checkpoint_path = Path(__file__).parent / "checkpoint.parquet"
    
    # Check for existing checkpoint
    start_idx = 0
    if checkpoint_path.exists():
        checkpoint_df = pd.read_parquet(checkpoint_path)
        start_idx = len(checkpoint_df)
        logger.info(f"Resuming from checkpoint: {start_idx} comments already processed")
    
    # Process in batches
    results = []
    start_time = time.time()
    
    for i, row in df.iloc[start_idx:].iterrows():
        idx = len(results) + start_idx
        
        try:
            result = scorer.score_text(row['body'])
            results.append({
                'comment_id': row.get('id', idx),
                'author': row.get('author', 'unknown'),
                'anthroscore_llm': result.score,
                'llm_reasoning': result.reasoning,
                'llm_confidence': result.confidence,
                'llm_model': result.model,
                'llm_time_ms': result.processing_time_ms
            })
        except Exception as e:
            logger.error(f"Error on comment {idx}: {e}")
            results.append({
                'comment_id': row.get('id', idx),
                'author': row.get('author', 'unknown'),
                'anthroscore_llm': 0,
                'llm_reasoning': f"ERROR: {e}",
                'llm_confidence': 0,
                'llm_model': model,
                'llm_time_ms': 0
            })
        
        # Progress
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (len(results)) / elapsed
            remaining = (n_comments - idx - 1) / rate / 3600
            logger.info(f"Progress: {idx + 1}/{n_comments} ({100*(idx+1)/n_comments:.1f}%) | Rate: {rate:.1f}/s | ETA: {remaining:.1f}h")
        
        # Checkpoint
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_parquet(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved: {len(checkpoint_df)} comments")
        
        # Rate limiting (avoid hitting API limits)
        time.sleep(0.05)
    
    # Save final results
    final_df = pd.DataFrame(results)
    final_df.to_parquet(output_path, index=False)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    elapsed_hours = (time.time() - start_time) / 3600
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total comments processed: {len(final_df)}")
    print(f"Total time: {elapsed_hours:.2f} hours")
    print(f"Output saved to: {output_path}")
    print("="*60)
    
    # Summary statistics
    valid_scores = final_df[final_df['anthroscore_llm'] > 0]
    print(f"\nScore distribution:")
    print(valid_scores['anthroscore_llm'].value_counts().sort_index())
    print(f"\nMean: {valid_scores['anthroscore_llm'].mean():.2f}")
    print(f"Std: {valid_scores['anthroscore_llm'].std():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM AnthroScore on full dataset")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only")
    
    args = parser.parse_args()
    
    run_full_dataset(model=args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
