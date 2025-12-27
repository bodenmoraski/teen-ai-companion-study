"""
Visualization utilities for generating publication-ready figures.

This module provides functions to create all figures required for the research paper.
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

logger = logging.getLogger(__name__)


def plot_age_distribution(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot age distribution histogram.
    
    Args:
        df: DataFrame with age_bucket column
        output_path: Path to save figure
        figsize: Figure size
    """
    logger.info("Plotting age distribution")
    
    if 'age_bucket' not in df.columns:
        logger.warning("age_bucket column not found")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    age_counts = df['age_bucket'].value_counts().sort_index()
    
    age_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Age Bucket', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Users by Age Bucket', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Age distribution plot saved to {output_path}")


def plot_anthroscore_by_demographics(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot AnthroScore distribution by age and gender.
    
    Args:
        df: DataFrame with anthroscore_mean, age_bucket, gender columns
        output_path: Path to save figure
        figsize: Figure size
    """
    logger.info("Plotting AnthroScore by demographics")
    
    if 'anthroscore_mean' not in df.columns:
        logger.warning("anthroscore_mean column not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # By age
    if 'age_bucket' in df.columns:
        df_age = df[df['age_bucket'].notna() & df['anthroscore_mean'].notna()]
        if len(df_age) > 0:
            age_order = sorted(df_age['age_bucket'].unique())
            sns.boxplot(
                data=df_age,
                x='age_bucket',
                y='anthroscore_mean',
                order=age_order,
                ax=axes[0]
            )
            axes[0].set_xlabel('Age Bucket', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('AnthroScore (Mean)', fontsize=11, fontweight='bold')
            axes[0].set_title('AnthroScore by Age Bucket', fontsize=12, fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
    
    # By gender
    if 'gender' in df.columns:
        df_gender = df[df['gender'].notna() & df['anthroscore_mean'].notna()]
        if len(df_gender) > 0:
            sns.boxplot(
                data=df_gender,
                x='gender',
                y='anthroscore_mean',
                ax=axes[1]
            )
            axes[1].set_xlabel('Gender', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('AnthroScore (Mean)', fontsize=11, fontweight='bold')
            axes[1].set_title('AnthroScore by Gender', fontsize=12, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"AnthroScore by demographics plot saved to {output_path}")


def plot_topic_distribution(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot topic distribution.
    
    Args:
        df: DataFrame with dominant_topic column
        output_path: Path to save figure
        top_n: Number of top topics to show
        figsize: Figure size
    """
    logger.info("Plotting topic distribution")
    
    if 'dominant_topic' not in df.columns:
        logger.warning("dominant_topic column not found")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    topic_counts = df['dominant_topic'].value_counts().head(top_n)
    
    topic_counts.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
    ax.set_xlabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_ylabel('Topic ID', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of Top {top_n} Topics', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Topic distribution plot saved to {output_path}")


def plot_emotion_distribution(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot emotion distribution.
    
    Args:
        df: DataFrame with dominant_emotion column
        output_path: Path to save figure
        figsize: Figure size
    """
    logger.info("Plotting emotion distribution")
    
    if 'dominant_emotion' not in df.columns:
        logger.warning("dominant_emotion column not found")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    emotion_counts = df['dominant_emotion'].value_counts()
    
    emotion_counts.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='black')
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Dominant Emotions', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Emotion distribution plot saved to {output_path}")

