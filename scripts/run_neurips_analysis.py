"""
Run comprehensive NeurIPS-level analysis.

This script executes all analyses required for publication-quality research:
1. 3-bucket age classification
2. Method comparison and ablation
3. Hierarchical regression with controls
4. Robustness checks (bootstrap, sensitivity, temporal, subreddit-level)
5. Multiple comparison corrections
6. Model diagnostics
"""
import sys
from pathlib import Path
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurips_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("NEURIPS-LEVEL COMPREHENSIVE ANALYSIS")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\n=== LOADING DATA ===")
    demo = pd.read_parquet("data/features/demographics.parquet")
    features = pd.read_parquet("data/features/full_merged_dataset.parquet")
    comments = pd.read_parquet("data/processed/all_comments.parquet")
    
    logger.info(f"Demographics: {len(demo):,} users")
    logger.info(f"Features: {len(features):,} users")
    logger.info(f"Comments: {len(comments):,} comments")
    
    # Merge for full dataset
    # First get columns from demo that we need
    demo_cols = ['author', 'age_bucket', 'gender', 'age_bucket_self_declared', 
                 'gender_self_declared', 'age_bucket_community', 'age_bucket_llm',
                 'age_community_score']
    demo_subset = demo[[c for c in demo_cols if c in demo.columns]].copy()
    
    # Remove duplicate columns from features
    features_cols_to_drop = [c for c in demo_subset.columns if c in features.columns and c != 'author']
    features_clean = features.drop(columns=features_cols_to_drop, errors='ignore')
    
    # Merge
    merged = features_clean.merge(demo_subset, on='author', how='left')
    
    # Add subreddit and date info from comments
    user_subreddits = comments.groupby('author')['subreddit'].first().reset_index()
    merged = merged.merge(user_subreddits, on='author', how='left')
    
    # Add created_utc for temporal analysis
    if 'created_utc' in comments.columns:
        user_dates = comments.groupby('author')['created_utc'].first().reset_index()
        merged = merged.merge(user_dates, on='author', how='left')
    
    logger.info(f"Merged dataset: {len(merged):,} users")
    
    # Run full analysis
    from src.statistical.neurips_analysis import run_full_neurips_analysis
    
    output_dir = Path("results/neurips")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_full_neurips_analysis(
        merged,
        output_dir,
        run_bootstrap=True,
        n_bootstrap=500  # Reduced for speed, increase to 1000 for final
    )
    
    # Generate summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    # Method comparison
    if 'method_comparison' in results:
        logger.info("\nMETHOD COMPARISON:")
        mc = results['method_comparison']
        for _, row in mc.iterrows():
            logger.info(f"  {row['method']}: coverage={row['coverage']:.1%}, accuracy={row['accuracy']:.1%}")
    
    # 3-bucket accuracy
    if 'age_3bucket_accuracy' in results:
        logger.info(f"\n3-BUCKET AGE ACCURACY: {results['age_3bucket_accuracy']:.1%}")
    
    # Hierarchical regression R² progression
    if 'hierarchical_regression' in results:
        hier = results['hierarchical_regression']
        logger.info("\nHIERARCHICAL REGRESSION (R² progression):")
        for name, model in hier.get('models', {}).items():
            if model is not None and hasattr(model, 'rsquared'):
                logger.info(f"  {name}: R² = {model.rsquared:.6f}")
    
    # Robustness
    if 'temporal' in results and results['temporal']:
        logger.info("\nTEMPORAL STABILITY:")
        for year, data in results['temporal'].items():
            logger.info(f"  {year}: N={data['n']}, R²={data['r_squared']:.6f}")
    
    if 'subreddit' in results and results['subreddit']:
        logger.info("\nSUBREDDIT-LEVEL:")
        for sub, data in results['subreddit'].items():
            logger.info(f"  {sub}: N={data['n']}, R²={data['r_squared']:.6f}")
    
    # Multiple comparison
    if 'corrected_pvalues' in results:
        cp = results['corrected_pvalues']
        n_sig_before = (cp['p_uncorrected'] < 0.05).sum()
        n_sig_after = cp['significant_corrected'].sum()
        logger.info(f"\nMULTIPLE COMPARISON CORRECTION:")
        logger.info(f"  Before: {n_sig_before} significant")
        logger.info(f"  After FDR: {n_sig_after} significant")
    
    # Diagnostics
    if 'diagnostics' in results:
        diag = results['diagnostics']
        logger.info(f"\nMODEL DIAGNOSTICS:")
        logger.info(f"  Heteroscedasticity: {diag.get('heteroscedasticity', 'N/A')}")
        logger.info(f"  Residuals normal: {diag.get('residuals_normal', 'N/A')}")
        logger.info(f"  Influential observations: {diag.get('n_influential_obs', 'N/A')}")
    
    # Bootstrap
    if 'bootstrap' in results and len(results['bootstrap']) > 0:
        logger.info("\nBOOTSTRAP CIs available in output file")
    
    # Save complete results
    results_path = output_dir / 'complete_results.pkl'
    import pickle
    with open(results_path, 'wb') as f:
        # Can't pickle statsmodels objects easily, so save what we can
        serializable_results = {k: v for k, v in results.items() 
                                if k not in ['hierarchical_regression']}
        pickle.dump(serializable_results, f)
    
    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(f"Full report: {output_dir}/neurips_analysis_report.txt")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"ANALYSIS COMPLETE: {datetime.now()}")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

