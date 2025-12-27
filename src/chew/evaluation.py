"""
Evaluation Module for Reddit Age Classification

Implements comprehensive model evaluation including metrics, feature importance,
and statistical tests as described in Chew et al. (2021).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_ind
from typing import Dict, Tuple, Any
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation of age classification model.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.results = {}
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      X_train: pd.DataFrame = None, y_train: pd.Series = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional, for statistical tests)
            y_train: Training labels (optional, for statistical tests)
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate classification metrics
        self.results['classification_metrics'] = self._calculate_classification_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Calculate confusion matrix
        self.results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve data
        self.results['roc_curve'] = self._calculate_roc_curve(y_test, y_pred_proba)
        
        # Calculate permutation importance
        self.results['permutation_importance'] = self._calculate_permutation_importance(
            model, X_test, y_test
        )
        
        # Calculate feature statistics if training data provided
        if X_train is not None and y_train is not None:
            self.results['feature_statistics'] = self._calculate_feature_statistics(
                X_train, y_train
            )
        
        # Log summary
        self._log_evaluation_summary()
        
        return self.results
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Overall metrics
        metrics = {
            # Adolescent (class 0) metrics
            'adolescent_precision': precision_per_class[0],
            'adolescent_recall': recall_per_class[0],
            'adolescent_f1': f1_per_class[0],
            
            # Adult (class 1) metrics
            'adult_precision': precision_per_class[1],
            'adult_recall': recall_per_class[1],
            'adult_f1': f1_per_class[1],
            
            # Overall metrics
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'auroc': roc_auc_score(y_true, y_pred_proba[:, 1]),
            
            # Accuracy
            'accuracy': (y_true == y_pred).mean()
        }
        
        return metrics
    
    def _calculate_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate ROC curve data.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with FPR, TPR, and thresholds
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def _calculate_permutation_importance(self, model: Any, X_test: pd.DataFrame,
                                         y_test: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
        """
        Calculate permutation importance for all features.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            n_repeats: Number of times to permute each feature
            
        Returns:
            DataFrame with feature importance statistics
        """
        logger.info(f"Calculating permutation importance ({n_repeats} repeats)...")
        
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        logger.info("✓ Permutation importance calculated")
        
        return importance_df
    
    def _calculate_feature_statistics(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Calculate feature statistics by age group with t-tests (Table 5 style).
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            DataFrame with feature statistics
        """
        logger.info("Calculating feature statistics by age group...")
        
        # Separate by age group
        X_adolescent = X_train[y_train == 0]
        X_adult = X_train[y_train == 1]
        
        stats_list = []
        
        for feature in X_train.columns:
            # Calculate means and stds
            adolescent_mean = X_adolescent[feature].mean()
            adolescent_std = X_adolescent[feature].std()
            adult_mean = X_adult[feature].mean()
            adult_std = X_adult[feature].std()
            
            # Perform t-test
            t_stat, p_value = ttest_ind(X_adolescent[feature], X_adult[feature])
            
            # Calculate degrees of freedom
            n_adolescent = len(X_adolescent)
            n_adult = len(X_adult)
            df = n_adolescent + n_adult - 2
            
            stats_list.append({
                'feature': feature,
                'adolescent_mean': adolescent_mean,
                'adolescent_std': adolescent_std,
                'adult_mean': adult_mean,
                'adult_std': adult_std,
                't_statistic': t_stat,
                'df': df,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        stats_df = pd.DataFrame(stats_list)
        
        logger.info("✓ Feature statistics calculated")
        
        return stats_df
    
    def _log_evaluation_summary(self) -> None:
        """Log evaluation summary to console."""
        metrics = self.results['classification_metrics']
        
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info("\nOverall Metrics:")
        logger.info(f"  F1 Score (Macro): {metrics['macro_f1']:.3f}")
        logger.info(f"  AUROC: {metrics['auroc']:.3f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        
        logger.info("\nPer-Class Metrics:")
        logger.info(f"  Adolescent (13-20):")
        logger.info(f"    Precision: {metrics['adolescent_precision']:.3f}")
        logger.info(f"    Recall: {metrics['adolescent_recall']:.3f}")
        logger.info(f"    F1 Score: {metrics['adolescent_f1']:.3f}")
        
        logger.info(f"  Adult (21-54):")
        logger.info(f"    Precision: {metrics['adult_precision']:.3f}")
        logger.info(f"    Recall: {metrics['adult_recall']:.3f}")
        logger.info(f"    F1 Score: {metrics['adult_f1']:.3f}")
        
        if 'permutation_importance' in self.results:
            logger.info("\nFeature Importance (Top 5):")
            importance_df = self.results['permutation_importance']
            for idx, row in importance_df.head(5).iterrows():
                logger.info(f"  {idx+1}. {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        logger.info("="*60 + "\n")
    
    def save_results(self, output_dir: str) -> None:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save classification report
            if 'classification_metrics' in self.results:
                report_path = Path(output_dir) / "classification_report.txt"
                with open(report_path, 'w') as f:
                    f.write("Reddit Age Classification - Evaluation Report\n")
                    f.write("="*60 + "\n\n")
                    
                    metrics = self.results['classification_metrics']
                    
                    f.write("Overall Metrics:\n")
                    f.write(f"  F1 Score (Macro): {metrics['macro_f1']:.4f}\n")
                    f.write(f"  F1 Score (Weighted): {metrics['weighted_f1']:.4f}\n")
                    f.write(f"  AUROC: {metrics['auroc']:.4f}\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n\n")
                    
                    f.write("Per-Class Metrics:\n")
                    f.write(f"  Adolescent (13-20):\n")
                    f.write(f"    Precision: {metrics['adolescent_precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['adolescent_recall']:.4f}\n")
                    f.write(f"    F1 Score: {metrics['adolescent_f1']:.4f}\n\n")
                    
                    f.write(f"  Adult (21-54):\n")
                    f.write(f"    Precision: {metrics['adult_precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['adult_recall']:.4f}\n")
                    f.write(f"    F1 Score: {metrics['adult_f1']:.4f}\n\n")
                
                logger.info(f"✓ Saved classification report to {report_path}")
            
            # Save feature importance
            if 'permutation_importance' in self.results:
                importance_path = Path(output_dir) / "feature_importance.csv"
                self.results['permutation_importance'].to_csv(importance_path, index=False)
                logger.info(f"✓ Saved feature importance to {importance_path}")
            
            # Save feature statistics
            if 'feature_statistics' in self.results:
                stats_path = Path(output_dir) / "feature_statistics.csv"
                self.results['feature_statistics'].to_csv(stats_path, index=False)
                logger.info(f"✓ Saved feature statistics to {stats_path}")
            
            # Save confusion matrix data
            if 'confusion_matrix' in self.results:
                cm_path = Path(output_dir) / "confusion_matrix_data.csv"
                cm_df = pd.DataFrame(
                    self.results['confusion_matrix'],
                    index=['Adolescent', 'Adult'],
                    columns=['Predicted Adolescent', 'Predicted Adult']
                )
                cm_df.to_csv(cm_path)
                logger.info(f"✓ Saved confusion matrix data to {cm_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


def evaluate_age_classifier(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                           X_train: pd.DataFrame = None, y_train: pd.Series = None,
                           output_dir: str = "results") -> Dict[str, Any]:
    """
    Main function to evaluate the age classifier.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_train: Training features (optional)
        y_train: Training labels (optional)
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation results
    """
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate model
    results = evaluator.evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Save results
    evaluator.save_results(output_dir)
    
    return results


if __name__ == "__main__":
    # Test evaluation
    from feature_engineering import prepare_dataset
    from model_training import train_age_classifier
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    # Train model
    classifier = train_age_classifier(X_train, y_train, use_grid_search=False)
    
    # Evaluate model
    results = evaluate_age_classifier(
        classifier.model, X_test, y_test,
        X_train, y_train
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: results/")

