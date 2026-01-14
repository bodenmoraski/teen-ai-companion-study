"""
Compare LLM-based AnthroScore to existing MLM-based AnthroScore V2.

Analyzes correlation between the two approaches on the test set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_scores():
    """Compare LLM and MLM scores on the fully labeled test set."""
    
    exp_dir = Path(__file__).parent
    test_set_path = exp_dir / "test_set_fully_labeled.parquet"
    
    if not test_set_path.exists():
        logger.error("Fully labeled test set not found. Run validate_agreement.py first.")
        return
    
    df = pd.read_parquet(test_set_path)
    
    # Check for MLM anthroscore
    if 'anthroscore_mean' not in df.columns:
        logger.warning("No MLM anthroscore in test set. Cannot compare.")
        return
    
    # Filter valid rows
    df = df[
        (df['expert_score'] > 0) & 
        (df['llm_score'] > 0) & 
        df['anthroscore_mean'].notna()
    ].copy()
    
    logger.info(f"Comparing {len(df)} samples with all three scores")
    
    # Normalize MLM scores to 1-5 scale for comparison
    mlm_min = df['anthroscore_mean'].min()
    mlm_max = df['anthroscore_mean'].max()
    df['mlm_normalized'] = 1 + 4 * (df['anthroscore_mean'] - mlm_min) / (mlm_max - mlm_min + 1e-10)
    
    print("\n" + "="*70)
    print("LLM vs MLM ANTHROSCORE COMPARISON")
    print("="*70)
    
    # Correlations with expert
    llm_expert_r, llm_expert_p = pearsonr(df['llm_score'], df['expert_score'])
    mlm_expert_r, mlm_expert_p = pearsonr(df['mlm_normalized'], df['expert_score'])
    
    print("\n--- Correlation with Expert Labels ---")
    print(f"LLM vs Expert:  r = {llm_expert_r:.3f} (p = {llm_expert_p:.2e})")
    print(f"MLM vs Expert:  r = {mlm_expert_r:.3f} (p = {mlm_expert_p:.2e})")
    print(f"Winner: {'LLM' if llm_expert_r > mlm_expert_r else 'MLM'} (improvement: {llm_expert_r - mlm_expert_r:+.3f})")
    
    # Direct correlation between LLM and MLM
    llm_mlm_r, llm_mlm_p = pearsonr(df['llm_score'], df['anthroscore_mean'])
    llm_mlm_spearman, _ = spearmanr(df['llm_score'], df['anthroscore_mean'])
    
    print("\n--- LLM vs MLM Direct Correlation ---")
    print(f"Pearson r:  {llm_mlm_r:.3f} (p = {llm_mlm_p:.2e})")
    print(f"Spearman r: {llm_mlm_spearman:.3f}")
    
    # Score distributions
    print("\n--- Score Statistics ---")
    print(f"Expert:  mean={df['expert_score'].mean():.2f}, std={df['expert_score'].std():.2f}")
    print(f"LLM:     mean={df['llm_score'].mean():.2f}, std={df['llm_score'].std():.2f}")
    print(f"MLM:     mean={df['anthroscore_mean'].mean():.4f}, std={df['anthroscore_mean'].std():.4f}")
    print(f"MLM (normalized 1-5): mean={df['mlm_normalized'].mean():.2f}, std={df['mlm_normalized'].std():.2f}")
    
    # Cases where LLM and MLM disagree most
    df['llm_mlm_diff'] = abs(df['llm_score'] - df['mlm_normalized'])
    high_disagreement = df.nlargest(5, 'llm_mlm_diff')
    
    print("\n--- Highest LLM-MLM Disagreements ---")
    for i, row in high_disagreement.iterrows():
        print(f"\nExpert: {row['expert_score']}, LLM: {row['llm_score']}, MLM(norm): {row['mlm_normalized']:.1f}")
        print(f"LLM reasoning: {row['llm_reasoning'][:80]}...")
    
    # Compute which method is closer to expert for each sample
    df['llm_error'] = abs(df['llm_score'] - df['expert_score'])
    df['mlm_error'] = abs(df['mlm_normalized'] - df['expert_score'])
    
    llm_wins = (df['llm_error'] < df['mlm_error']).sum()
    mlm_wins = (df['mlm_error'] < df['llm_error']).sum()
    ties = (df['llm_error'] == df['mlm_error']).sum()
    
    print("\n--- Head-to-Head (closer to expert) ---")
    print(f"LLM wins: {llm_wins} ({100*llm_wins/len(df):.1f}%)")
    print(f"MLM wins: {mlm_wins} ({100*mlm_wins/len(df):.1f}%)")
    print(f"Ties:     {ties} ({100*ties/len(df):.1f}%)")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if llm_expert_r > mlm_expert_r:
        improvement = (llm_expert_r - mlm_expert_r) / mlm_expert_r * 100
        print(f"""
The LLM-based approach shows HIGHER correlation with expert labels!
- Improvement: {improvement:.1f}% over MLM
- LLM is also more interpretable (discrete 1-5 scale)
- Recommended: Use LLM for production scoring
        """)
    else:
        print(f"""
The MLM-based approach shows higher correlation with expert labels.
However, LLM may still be preferable for:
- Interpretability (discrete scale)
- Handling ambiguous cases
- Consistency with annotation guidelines
        """)
    
    # Save comparison results
    results = {
        'llm_expert_r': float(llm_expert_r),
        'mlm_expert_r': float(mlm_expert_r),
        'llm_mlm_r': float(llm_mlm_r),
        'llm_wins': int(llm_wins),
        'mlm_wins': int(mlm_wins),
        'ties': int(ties),
        'n_samples': len(df)
    }
    
    import json
    with open(exp_dir / 'mlm_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved comparison results to: {exp_dir / 'mlm_comparison_results.json'}")


if __name__ == "__main__":
    compare_scores()
