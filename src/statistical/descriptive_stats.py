"""
Descriptive statistics generation for the research project.

This module generates comprehensive descriptive statistics tables
for publication.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_descriptive_statistics(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate descriptive statistics tables.
    
    Args:
        df: Merged dataset with all features
        output_path: Path to save statistics file
    """
    logger.info("Generating descriptive statistics")
    
    stats_lines = []
    stats_lines.append("Descriptive Statistics")
    stats_lines.append("=" * 70)
    stats_lines.append("")
    
    # Overall statistics
    stats_lines.append("Overall Dataset Statistics")
    stats_lines.append("-" * 70)
    stats_lines.append(f"Total users: {len(df)}")
    stats_lines.append(f"Total features: {len(df.columns)}")
    stats_lines.append("")
    
    # Age distribution
    if 'age_bucket' in df.columns:
        stats_lines.append("Age Distribution")
        stats_lines.append("-" * 70)
        age_dist = df['age_bucket'].value_counts().sort_index()
        total_age = age_dist.sum()
        for bucket, count in age_dist.items():
            pct = 100 * count / total_age if total_age > 0 else 0
            stats_lines.append(f"  {bucket}: {count} ({pct:.1f}%)")
        stats_lines.append(f"  Unknown: {(df['age_bucket'].isna()).sum()}")
        stats_lines.append("")
    
    # Gender distribution
    if 'gender' in df.columns:
        stats_lines.append("Gender Distribution")
        stats_lines.append("-" * 70)
        gender_dist = df['gender'].value_counts()
        total_gender = gender_dist.sum()
        for gender, count in gender_dist.items():
            pct = 100 * count / total_gender if total_gender > 0 else 0
            stats_lines.append(f"  {gender}: {count} ({pct:.1f}%)")
        stats_lines.append(f"  Unknown: {(df['gender'].isna()).sum()}")
        stats_lines.append("")
    
    # AnthroScore statistics by demographics
    if 'anthroscore_mean' in df.columns:
        stats_lines.append("AnthroScore Statistics")
        stats_lines.append("-" * 70)
        valid_scores = df['anthroscore_mean'].dropna()
        if len(valid_scores) > 0:
            stats_lines.append(f"  Overall:")
            stats_lines.append(f"    Mean: {valid_scores.mean():.3f}")
            stats_lines.append(f"    Median: {valid_scores.median():.3f}")
            stats_lines.append(f"    Std: {valid_scores.std():.3f}")
            stats_lines.append(f"    Min: {valid_scores.min():.3f}")
            stats_lines.append(f"    Max: {valid_scores.max():.3f}")
            stats_lines.append(f"    N: {len(valid_scores)}")
            stats_lines.append("")
            
            # By age bucket
            if 'age_bucket' in df.columns:
                stats_lines.append("  By Age Bucket:")
                for bucket in sorted(df['age_bucket'].dropna().unique()):
                    bucket_scores = df[df['age_bucket'] == bucket]['anthroscore_mean'].dropna()
                    if len(bucket_scores) > 0:
                        stats_lines.append(f"    {bucket}:")
                        stats_lines.append(f"      Mean: {bucket_scores.mean():.3f}")
                        stats_lines.append(f"      Median: {bucket_scores.median():.3f}")
                        stats_lines.append(f"      Std: {bucket_scores.std():.3f}")
                        stats_lines.append(f"      N: {len(bucket_scores)}")
                stats_lines.append("")
            
            # By gender
            if 'gender' in df.columns:
                stats_lines.append("  By Gender:")
                for gender in sorted(df['gender'].dropna().unique()):
                    gender_scores = df[df['gender'] == gender]['anthroscore_mean'].dropna()
                    if len(gender_scores) > 0:
                        stats_lines.append(f"    {gender}:")
                        stats_lines.append(f"      Mean: {gender_scores.mean():.3f}")
                        stats_lines.append(f"      Median: {gender_scores.median():.3f}")
                        stats_lines.append(f"      Std: {gender_scores.std():.3f}")
                        stats_lines.append(f"      N: {len(gender_scores)}")
                stats_lines.append("")
    
    # Topic distribution
    if 'dominant_topic' in df.columns:
        stats_lines.append("Topic Distribution")
        stats_lines.append("-" * 70)
        topic_dist = df['dominant_topic'].value_counts().sort_index()
        total_topics = topic_dist.sum()
        for topic, count in topic_dist.head(10).items():
            pct = 100 * count / total_topics if total_topics > 0 else 0
            stats_lines.append(f"  Topic {topic}: {count} ({pct:.1f}%)")
        stats_lines.append("")
    
    # Emotion distribution
    if 'dominant_emotion' in df.columns:
        stats_lines.append("Dominant Emotion Distribution")
        stats_lines.append("-" * 70)
        emotion_dist = df['dominant_emotion'].value_counts()
        total_emotions = emotion_dist.sum()
        for emotion, count in emotion_dist.items():
            pct = 100 * count / total_emotions if total_emotions > 0 else 0
            stats_lines.append(f"  {emotion}: {count} ({pct:.1f}%)")
        stats_lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(stats_lines))
    
    logger.info(f"Descriptive statistics saved to {output_path}")


def generate_correlation_table(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate correlation table for key variables.
    
    Args:
        df: Merged dataset
        output_path: Path to save correlation table
    """
    logger.info("Generating correlation table")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to relevant columns
    relevant_cols = [
        col for col in numeric_cols 
        if any(keyword in col.lower() for keyword in [
            'anthroscore', 'emotion', 'topic', 'confidence'
        ])
    ]
    
    if len(relevant_cols) < 2:
        logger.warning("Insufficient numeric columns for correlation analysis")
        return
    
    corr_df = df[relevant_cols].corr()
    
    # Save as CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(output_path)
    
    logger.info(f"Correlation table saved to {output_path}")

