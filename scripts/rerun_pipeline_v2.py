"""
Re-run the full pipeline with V2 community embeddings.

This script:
1. Loads existing demographics
2. Updates with V2 community embeddings
3. Updates the ensemble classification
4. Re-runs regression analysis
5. Generates updated validation report
"""
import sys
from pathlib import Path
import logging
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rerun_pipeline_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def update_demographics_with_v2(demo_df: pd.DataFrame, v2_df: pd.DataFrame) -> pd.DataFrame:
    """Update demographics with V2 community embeddings."""
    logger.info("Updating demographics with V2 community embeddings")
    
    # Drop old community columns if they exist
    cols_to_drop = [c for c in demo_df.columns if 'community' in c.lower()]
    if cols_to_drop:
        logger.info(f"Dropping old columns: {cols_to_drop}")
        demo_df = demo_df.drop(columns=cols_to_drop)
    
    # Merge with V2
    demo_df = demo_df.merge(v2_df, on='author', how='left')
    
    logger.info(f"Updated {demo_df['age_bucket_community'].notna().sum()} users with V2 age")
    logger.info(f"Updated {(demo_df['gender_community'] != 'unknown').sum()} users with V2 gender")
    
    return demo_df


def update_ensemble_classification(demo_df: pd.DataFrame) -> pd.DataFrame:
    """Update ensemble classification with self-declaration priority."""
    logger.info("Updating ensemble classification")
    
    # Final age bucket: self-declaration > LLM (high confidence) > community
    def get_final_age(row):
        # Self-declaration is ground truth
        if pd.notna(row.get('age_bucket_self_declared')):
            return row['age_bucket_self_declared']
        
        # LLM with high confidence
        if pd.notna(row.get('age_bucket_llm')) and row.get('confidence_llm', 0) > 0.7:
            return row['age_bucket_llm']
        
        # Community embedding
        if pd.notna(row.get('age_bucket_community')):
            return row['age_bucket_community']
        
        # LLM with lower confidence
        if pd.notna(row.get('age_bucket_llm')):
            return row['age_bucket_llm']
        
        return None
    
    # Final gender: self-declaration > community (if not unknown)
    def get_final_gender(row):
        # Self-declaration is ground truth
        if pd.notna(row.get('gender_self_declared')):
            return row['gender_self_declared']
        
        # Community embedding (if not unknown)
        if pd.notna(row.get('gender_community')) and row.get('gender_community') != 'unknown':
            return row['gender_community']
        
        return 'unknown'
    
    demo_df['age_bucket'] = demo_df.apply(get_final_age, axis=1)
    demo_df['gender'] = demo_df.apply(get_final_gender, axis=1)
    
    logger.info(f"Final age distribution:\n{demo_df['age_bucket'].value_counts(dropna=False)}")
    logger.info(f"Final gender distribution:\n{demo_df['gender'].value_counts(dropna=False)}")
    
    return demo_df


def run_regression_v2(features_df: pd.DataFrame, unused_df: pd.DataFrame, output_dir: Path):
    """Re-run regression with V2 demographics."""
    from src.statistical.regression_models_fixed import (
        run_rq2_regression,
        generate_regression_tables
    )
    
    logger.info("Preparing regression data")
    logger.info(f"Input columns: {list(features_df.columns)}")
    
    # Filter to users with age AND anthroscore
    reg_data = features_df[
        features_df['age_bucket'].notna() & 
        features_df['anthroscore_mean'].notna()
    ].copy()
    
    logger.info(f"Users with age AND anthroscore: {len(reg_data)}")
    
    if len(reg_data) < 100:
        logger.error("Not enough data for regression!")
        return
    
    # Run regression (it will prepare data internally)
    logger.info("Running RQ2 regression")
    results = run_rq2_regression(reg_data)
    
    # Generate tables
    output_file = output_dir / 'regression_results_v2.txt'
    generate_regression_tables(results, output_file)
    
    logger.info(f"Regression results saved to {output_file}")
    
    # Log key results
    for model_key in ['model1_age_only', 'model2_age_gender', 'model3_full']:
        model = results.get(model_key)
        if model is not None and hasattr(model, 'rsquared'):
            logger.info(f"{model_key}: R² = {model.rsquared:.6f}")
    
    return results


def generate_validation_report(demo_df: pd.DataFrame, output_dir: Path):
    """Generate validation report for V2."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("V2 VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Age validation
    age_mask = demo_df['age_bucket_self_declared'].notna() & demo_df['age_bucket_community'].notna()
    if age_mask.sum() > 0:
        age_data = demo_df[age_mask]
        age_accuracy = accuracy_score(
            age_data['age_bucket_self_declared'],
            age_data['age_bucket_community']
        )
        age_kappa = cohen_kappa_score(
            age_data['age_bucket_self_declared'],
            age_data['age_bucket_community']
        )
        
        report_lines.append("AGE CLASSIFICATION (V2)")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy: {age_accuracy:.1%}")
        report_lines.append(f"Cohen's Kappa: {age_kappa:.4f}")
        report_lines.append(f"N: {len(age_data)}")
        report_lines.append("")
    
    # Gender validation
    gender_mask = (
        demo_df['gender_self_declared'].notna() & 
        demo_df['gender_community'].notna() &
        (demo_df['gender_community'] != 'unknown')
    )
    if gender_mask.sum() > 0:
        gender_data = demo_df[gender_mask]
        gender_accuracy = accuracy_score(
            gender_data['gender_self_declared'],
            gender_data['gender_community']
        )
        
        report_lines.append("GENDER CLASSIFICATION (V2)")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy: {gender_accuracy:.1%}")
        report_lines.append(f"N: {len(gender_data)}")
        report_lines.append("")
        
        # Confusion matrix
        labels = ['male', 'female']
        subset = gender_data[gender_data['gender_self_declared'].isin(labels)]
        if len(subset) > 0:
            cm = confusion_matrix(
                subset['gender_self_declared'],
                subset['gender_community'],
                labels=labels
            )
            report_lines.append("Confusion Matrix (rows=true, cols=pred):")
            report_lines.append(f"         male  female")
            report_lines.append(f"male     {cm[0,0]:4d}    {cm[0,1]:4d}")
            report_lines.append(f"female   {cm[1,0]:4d}    {cm[1,1]:4d}")
            report_lines.append("")
    
    # Distribution summary
    report_lines.append("FINAL DISTRIBUTIONS")
    report_lines.append("-" * 40)
    report_lines.append("\nAge:")
    for bucket, count in demo_df['age_bucket'].value_counts(dropna=False).items():
        pct = count / len(demo_df) * 100
        report_lines.append(f"  {bucket}: {count:,} ({pct:.1f}%)")
    
    report_lines.append("\nGender:")
    for gender, count in demo_df['gender'].value_counts(dropna=False).items():
        pct = count / len(demo_df) * 100
        report_lines.append(f"  {gender}: {count:,} ({pct:.1f}%)")
    
    # Write report
    output_file = output_dir / 'validation_report_v2.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Validation report saved to {output_file}")


def main():
    logger.info("=" * 70)
    logger.info("RE-RUNNING PIPELINE WITH V2 COMMUNITY EMBEDDINGS")
    logger.info("=" * 70)
    
    # Load data
    v2_path = Path("data/features/community_embeddings_v2.parquet")
    if not v2_path.exists():
        logger.error("V2 embeddings not found! Run test_v2_fixes.py first.")
        return False
    
    demo_path = Path("data/features/demographics.parquet")
    comments_path = Path("data/processed/all_comments.parquet")
    
    v2_df = pd.read_parquet(v2_path)
    demo_df = pd.read_parquet(demo_path)
    comments_df = pd.read_parquet(comments_path)
    
    logger.info(f"Loaded V2 embeddings: {len(v2_df)} users")
    logger.info(f"Loaded demographics: {len(demo_df)} users")
    
    # Step 1: Update demographics with V2
    demo_df = update_demographics_with_v2(demo_df, v2_df)
    
    # Step 2: Update ensemble classification
    demo_df = update_ensemble_classification(demo_df)
    
    # Step 3: Save updated demographics
    output_path = Path("data/features/demographics_v2.parquet")
    demo_df.to_parquet(output_path, index=False)
    logger.info(f"Saved updated demographics to {output_path}")
    
    # Step 4: Run regression (need AnthroScore)
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features that have AnthroScore
    features_path = Path("data/features/full_merged_dataset.parquet")
    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        
        logger.info(f"Loaded features: {len(features_df)} users, columns with anthro: {[c for c in features_df.columns if 'anthro' in c.lower()]}")
        
        # Update with V2 demographics
        cols_to_drop = [c for c in ['age_bucket_community', 'gender_community', 'age_bucket', 'gender'] 
                        if c in features_df.columns]
        if cols_to_drop:
            features_df = features_df.drop(columns=cols_to_drop)
        
        features_df = features_df.merge(
            demo_df[['author', 'age_bucket_community', 'gender_community', 'age_bucket', 'gender']],
            on='author',
            how='left'
        )
        
        logger.info(f"After merge: {len(features_df)} users, anthroscore_mean non-null: {features_df['anthroscore_mean'].notna().sum()}")
        
        # Run regression
        results = run_regression_v2(features_df, features_df, output_dir)
    else:
        logger.warning("full_merged_dataset.parquet not found, skipping regression")
    
    # Step 5: Generate validation report
    validation_dir = Path("results/validation")
    validation_dir.mkdir(parents=True, exist_ok=True)
    generate_validation_report(demo_df, validation_dir)
    
    logger.info("=" * 70)
    logger.info("PIPELINE RE-RUN COMPLETE")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

