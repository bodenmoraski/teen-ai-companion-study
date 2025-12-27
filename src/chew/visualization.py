"""
Visualization Module for Reddit Age Classification

Creates publication-quality visualizations including feature importance,
confusion matrix, and ROC curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


class ResultsVisualizer:
    """
    Create visualizations for model evaluation results.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 15,
                               title: str = "Feature Importance (Permutation)") -> None:
        """
        Create horizontal bar chart of feature importance (Figure 1 style).
        
        Args:
            importance_df: DataFrame with feature, importance_mean, importance_std columns
            top_n: Number of top features to display
            title: Plot title
        """
        logger.info(f"Creating feature importance plot (top {top_n})...")
        
        # Get top N features
        plot_data = importance_df.head(top_n).copy()
        plot_data = plot_data.sort_values('importance_mean', ascending=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(plot_data))
        ax.barh(y_pos, plot_data['importance_mean'], 
               xerr=plot_data['importance_std'],
               align='center',
               alpha=0.8,
               color='steelblue',
               ecolor='gray',
               capsize=3)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data['feature'])
        ax.set_xlabel('Permutation Importance (Mean ± Std)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            ax.text(row['importance_mean'] + row['importance_std'] + 0.002,
                   i, f"{row['importance_mean']:.4f}",
                   va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved feature importance plot to {output_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray,
                            class_names: list = None,
                            title: str = "Confusion Matrix") -> None:
        """
        Create annotated confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            class_names: Names of classes (default: ['Adolescent', 'Adult'])
            title: Plot title
        """
        logger.info("Creating confusion matrix plot...")
        
        if class_names is None:
            class_names = ['Adolescent\n(13-20)', 'Adult\n(21-54)']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'},
                   ax=ax)
        
        # Customize plot
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
        
        # Add accuracy percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm_norm[i, j] * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                       ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved confusion matrix plot to {output_path}")
    
    def plot_roc_curve(self, roc_data: Dict[str, np.ndarray],
                      auroc: float,
                      title: str = "ROC Curve") -> None:
        """
        Create ROC curve with AUROC score.
        
        Args:
            roc_data: Dictionary with 'fpr', 'tpr', 'thresholds'
            auroc: AUROC score
            title: Plot title
        """
        logger.info("Creating ROC curve plot...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot ROC curve
        ax.plot(roc_data['fpr'], roc_data['tpr'],
               color='darkorange', lw=2,
               label=f'ROC curve (AUROC = {auroc:.3f})')
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random classifier')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved ROC curve plot to {output_path}")
    
    def plot_all(self, results: Dict[str, Any]) -> None:
        """
        Create all visualizations from evaluation results.
        
        Args:
            results: Dictionary of evaluation results
        """
        logger.info("Creating all visualizations...")
        
        # Plot feature importance
        if 'permutation_importance' in results:
            self.plot_feature_importance(results['permutation_importance'])
        
        # Plot confusion matrix
        if 'confusion_matrix' in results:
            self.plot_confusion_matrix(results['confusion_matrix'])
        
        # Plot ROC curve
        if 'roc_curve' in results and 'classification_metrics' in results:
            self.plot_roc_curve(
                results['roc_curve'],
                results['classification_metrics']['auroc']
            )
        
        logger.info("✓ All visualizations created")
    
    def create_summary_figure(self, results: Dict[str, Any],
                             title: str = "Model Performance Summary") -> None:
        """
        Create a combined figure with multiple subplots.
        
        Args:
            results: Dictionary of evaluation results
            title: Main title for the figure
        """
        logger.info("Creating summary figure...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Feature Importance (top 10)
        ax1 = fig.add_subplot(gs[0, :])
        if 'permutation_importance' in results:
            importance_df = results['permutation_importance'].head(10).copy()
            importance_df = importance_df.sort_values('importance_mean', ascending=True)
            
            y_pos = np.arange(len(importance_df))
            ax1.barh(y_pos, importance_df['importance_mean'],
                    xerr=importance_df['importance_std'],
                    align='center', alpha=0.8, color='steelblue',
                    ecolor='gray', capsize=3)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(importance_df['feature'])
            ax1.set_xlabel('Permutation Importance')
            ax1.set_title('Top 10 Features', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
        
        # Subplot 2: Confusion Matrix
        ax2 = fig.add_subplot(gs[1, 0])
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Adolescent', 'Adult'],
                       yticklabels=['Adolescent', 'Adult'],
                       cbar=False, ax=ax2)
            ax2.set_title('Confusion Matrix', fontweight='bold')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
        
        # Subplot 3: ROC Curve
        ax3 = fig.add_subplot(gs[1, 1])
        if 'roc_curve' in results and 'classification_metrics' in results:
            roc_data = results['roc_curve']
            auroc = results['classification_metrics']['auroc']
            
            ax3.plot(roc_data['fpr'], roc_data['tpr'],
                    color='darkorange', lw=2,
                    label=f'AUROC = {auroc:.3f}')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2,
                    linestyle='--', alpha=0.5)
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', fontweight='bold')
            ax3.legend(loc='lower right')
            ax3.grid(alpha=0.3)
        
        # Main title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        output_path = self.output_dir / "summary_figure.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved summary figure to {output_path}")


def visualize_results(results: Dict[str, Any], output_dir: str = "results") -> None:
    """
    Main function to create all visualizations.
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save visualizations
    """
    visualizer = ResultsVisualizer(output_dir)
    visualizer.plot_all(results)
    visualizer.create_summary_figure(results)


if __name__ == "__main__":
    # Test visualization
    from feature_engineering import prepare_dataset
    from model_training import train_age_classifier
    from evaluation import evaluate_age_classifier
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    # Train model
    classifier = train_age_classifier(X_train, y_train, use_grid_search=False)
    
    # Evaluate model
    results = evaluate_age_classifier(
        classifier.model, X_test, y_test,
        X_train, y_train
    )
    
    # Create visualizations
    visualize_results(results)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nVisualizations saved to: results/")

