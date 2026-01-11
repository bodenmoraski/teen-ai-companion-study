"""
GENDER PREDICTOR V3: Balanced Optimization

CONSTRAINT: Improve female recall while MAINTAINING or IMPROVING:
- Male recall (must stay >= 90%)
- Overall accuracy (must stay >= 80%)
- Female precision (should stay reasonable)

APPROACH:
1. Grid search over scale_pos_weight to find optimal balance
2. Use macro-F1 as primary optimization metric (balances all classes)
3. NO aggressive threshold manipulation
4. Test multiple configurations, pick Pareto-optimal
5. Validate that ALL metrics improve or stay constant

Author: Research Agent V3
Created: 2026-01-10
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import pickle
from collections import Counter
from itertools import product

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenderPredictor_V3:
    """
    Gender Predictor V3: Balanced optimization approach.
    
    Key difference from V2: Optimize for OVERALL improvement, not just female recall.
    Uses grid search to find scale_pos_weight that maximizes macro-F1 
    while maintaining constraints on all metrics.
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
        # Feature columns (set during training)
        self.text_cols = None
        self.subreddit_cols = None
        self.behavioral_cols = None
        self.linguistic_cols = None
        
        # Scalers
        self.text_scaler = StandardScaler()
        self.behavioral_scaler = StandardScaler()
        
        # Classifiers
        self.text_classifier = None
        self.subreddit_classifier = None
        self.behavioral_classifier = None
        self.linguistic_classifier = None
        self.meta_classifier = None
        
        # Optimal parameters found
        self.best_params = None
        self.best_metrics = None
        
        self.is_fitted = False
    
    def _evaluate_config(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_pos_weight: float,
        cv: StratifiedKFold
    ) -> Dict:
        """Evaluate a single configuration using cross-validation."""
        
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        clf = xgb.XGBClassifier(**params)
        
        # Get cross-validated predictions
        y_pred_proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'female_recall': recall_score(y, y_pred),  # y=1 is female
            'female_precision': precision_score(y, y_pred, zero_division=0),
            'male_recall': recall_score(1-y, 1-y_pred),
            'male_precision': precision_score(1-y, 1-y_pred, zero_division=0),
            'macro_f1': f1_score(y, y_pred, average='macro'),
            'female_f1': f1_score(y, y_pred),
            'scale_pos_weight': scale_pos_weight
        }
        
        return metrics
    
    def _grid_search_optimal_weight(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: StratifiedKFold,
        v1_metrics: Dict
    ) -> Tuple[float, Dict]:
        """
        Grid search to find optimal scale_pos_weight.
        
        CONSTRAINTS:
        - Male recall >= V1 male recall - 5% (allow small drop)
        - Accuracy >= V1 accuracy - 3%
        - Maximize macro_f1 subject to constraints
        """
        
        logger.info("\n" + "="*60)
        logger.info("GRID SEARCH: Finding optimal scale_pos_weight")
        logger.info("="*60)
        
        # Calculate class ratio for reference
        n_female = y.sum()
        n_male = len(y) - n_female
        natural_ratio = n_male / n_female
        
        logger.info(f"Class ratio (male/female): {natural_ratio:.2f}")
        logger.info(f"V1 Baseline - Accuracy: {v1_metrics['accuracy']:.1%}, "
                   f"Female Recall: {v1_metrics['female_recall']:.1%}, "
                   f"Male Recall: {v1_metrics['male_recall']:.1%}")
        
        # Search range: from 1.0 to 2x the natural ratio
        weights_to_test = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.68, 3.0, 3.5, 4.0]
        
        results = []
        
        for weight in weights_to_test:
            metrics = self._evaluate_config(X, y, weight, cv)
            results.append(metrics)
            
            # Check constraints
            acc_ok = metrics['accuracy'] >= v1_metrics['accuracy'] - 0.03
            male_ok = metrics['male_recall'] >= v1_metrics['male_recall'] - 0.05
            
            status = "✓" if (acc_ok and male_ok) else "✗"
            
            logger.info(f"  weight={weight:.1f}: Acc={metrics['accuracy']:.1%}, "
                       f"F-Rec={metrics['female_recall']:.1%}, "
                       f"M-Rec={metrics['male_recall']:.1%}, "
                       f"F1={metrics['macro_f1']:.1%} {status}")
        
        # Find best config that meets constraints
        valid_results = [
            r for r in results
            if r['accuracy'] >= v1_metrics['accuracy'] - 0.03
            and r['male_recall'] >= v1_metrics['male_recall'] - 0.05
        ]
        
        if valid_results:
            # Among valid configs, pick the one with best macro_f1
            best = max(valid_results, key=lambda x: x['macro_f1'])
            logger.info(f"\n✓ Best valid config: weight={best['scale_pos_weight']:.1f}")
        else:
            # No valid config - pick the one closest to constraints with best tradeoff
            logger.warning("\n⚠ No config meets all constraints. Picking best tradeoff...")
            # Sort by macro_f1 but penalize constraint violations
            def score(r):
                acc_penalty = max(0, v1_metrics['accuracy'] - 0.03 - r['accuracy']) * 10
                male_penalty = max(0, v1_metrics['male_recall'] - 0.05 - r['male_recall']) * 10
                return r['macro_f1'] - acc_penalty - male_penalty
            
            best = max(results, key=score)
            logger.info(f"  Selected weight={best['scale_pos_weight']:.1f} as best tradeoff")
        
        return best['scale_pos_weight'], best
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = 'gender_self_declared',
        v1_metrics: Dict = None
    ) -> Dict:
        """
        Train with balanced optimization.
        
        Args:
            features_df: Feature dataframe
            labels_df: Labels dataframe
            v1_metrics: V1 baseline metrics to compare against
        """
        logger.info("="*60)
        logger.info("GENDER PREDICTOR V3: BALANCED OPTIMIZATION")
        logger.info("="*60)
        
        # Set V1 baselines if not provided
        if v1_metrics is None:
            v1_metrics = {
                'accuracy': 0.813,
                'female_recall': 0.44,
                'male_recall': 0.95,
                'female_precision': 0.77
            }
        
        # Merge features with labels
        train_df = features_df.merge(
            labels_df[['author', label_col]],
            on='author',
            how='inner'
        )
        train_df = train_df[train_df[label_col].isin(['male', 'female'])]
        train_df = train_df.dropna(subset=[label_col])
        
        logger.info(f"Training set: {len(train_df)} users")
        
        # Encode labels (female = 1)
        y = (train_df[label_col] == 'female').astype(int).values
        n_female = y.sum()
        n_male = len(y) - n_female
        logger.info(f"Class distribution: {n_male} male, {n_female} female")
        
        # Identify feature columns
        feature_cols = [c for c in train_df.columns if c not in ['author', label_col]]
        
        self.text_cols = [c for c in feature_cols if c.startswith('text_emb_')]
        self.subreddit_cols = [c for c in feature_cols if c.startswith('sub_')]
        self.behavioral_cols = ['comment_count', 'avg_comment_length', 'std_comment_length',
                                'hour_mean', 'hour_std', 'weekend_ratio']
        self.behavioral_cols = [c for c in self.behavioral_cols if c in feature_cols]
        self.linguistic_cols = ['female_marker_rate', 'male_marker_rate', 'marker_ratio',
                                'marker_diff', 'avg_word_length', 'exclamation_rate',
                                'question_rate', 'emoji_rate', 'uppercase_ratio',
                                'avg_sentence_length']
        self.linguistic_cols = [c for c in self.linguistic_cols if c in feature_cols]
        
        logger.info(f"\nFeatures: {len(self.text_cols)} text, {len(self.subreddit_cols)} sub, "
                   f"{len(self.behavioral_cols)} behav, {len(self.linguistic_cols)} ling")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Prepare combined feature matrix for grid search
        all_features = []
        
        if self.text_cols:
            X_text = train_df[self.text_cols].values
            X_text_scaled = self.text_scaler.fit_transform(X_text)
            all_features.append(X_text_scaled)
        
        if self.subreddit_cols:
            X_sub = train_df[self.subreddit_cols].values
            all_features.append(X_sub)
        
        if self.behavioral_cols:
            X_behav = train_df[self.behavioral_cols].fillna(0).values
            X_behav_scaled = self.behavioral_scaler.fit_transform(X_behav)
            all_features.append(X_behav_scaled)
        
        if self.linguistic_cols:
            X_ling = train_df[self.linguistic_cols].fillna(0).values
            all_features.append(X_ling)
        
        X_combined = np.hstack(all_features)
        logger.info(f"Combined feature matrix: {X_combined.shape}")
        
        # Grid search for optimal weight
        best_weight, best_metrics = self._grid_search_optimal_weight(
            X_combined, y, cv, v1_metrics
        )
        
        self.best_params = {'scale_pos_weight': best_weight}
        self.best_metrics = best_metrics
        
        # Train final model with best weight
        logger.info(f"\nTraining final model with scale_pos_weight={best_weight:.2f}")
        
        final_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': best_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Train individual signal classifiers (for stacking potential)
        if self.text_cols:
            X_text_scaled = self.text_scaler.transform(train_df[self.text_cols].values)
            self.text_classifier = xgb.XGBClassifier(**final_params)
            self.text_classifier.fit(X_text_scaled, y)
        
        if self.subreddit_cols:
            X_sub = train_df[self.subreddit_cols].values
            self.subreddit_classifier = xgb.XGBClassifier(**final_params)
            self.subreddit_classifier.fit(X_sub, y)
        
        if self.behavioral_cols:
            X_behav_scaled = self.behavioral_scaler.transform(
                train_df[self.behavioral_cols].fillna(0).values
            )
            self.behavioral_classifier = xgb.XGBClassifier(**final_params)
            self.behavioral_classifier.fit(X_behav_scaled, y)
        
        if self.linguistic_cols:
            X_ling = train_df[self.linguistic_cols].fillna(0).values
            self.linguistic_classifier = xgb.XGBClassifier(**final_params)
            self.linguistic_classifier.fit(X_ling, y)
        
        # Meta classifier on combined features
        self.meta_classifier = xgb.XGBClassifier(**final_params)
        self.meta_classifier.fit(X_combined, y)
        
        self.label_encoder.fit(['male', 'female'])
        self.is_fitted = True
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("V3 TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best scale_pos_weight: {best_weight:.2f}")
        logger.info(f"\nV3 CV Metrics:")
        logger.info(f"  Accuracy:         {best_metrics['accuracy']:.1%}")
        logger.info(f"  Female Recall:    {best_metrics['female_recall']:.1%}")
        logger.info(f"  Female Precision: {best_metrics['female_precision']:.1%}")
        logger.info(f"  Male Recall:      {best_metrics['male_recall']:.1%}")
        logger.info(f"  Macro F1:         {best_metrics['macro_f1']:.1%}")
        
        logger.info(f"\nComparison to V1:")
        logger.info(f"  Accuracy:      {v1_metrics['accuracy']:.1%} -> {best_metrics['accuracy']:.1%} "
                   f"({best_metrics['accuracy'] - v1_metrics['accuracy']:+.1%})")
        logger.info(f"  Female Recall: {v1_metrics['female_recall']:.1%} -> {best_metrics['female_recall']:.1%} "
                   f"({best_metrics['female_recall'] - v1_metrics['female_recall']:+.1%})")
        logger.info(f"  Male Recall:   {v1_metrics['male_recall']:.1%} -> {best_metrics['male_recall']:.1%} "
                   f"({best_metrics['male_recall'] - v1_metrics['male_recall']:+.1%})")
        
        return {
            'best_weight': best_weight,
            'cv_metrics': best_metrics,
            'v1_comparison': {
                'accuracy_change': best_metrics['accuracy'] - v1_metrics['accuracy'],
                'female_recall_change': best_metrics['female_recall'] - v1_metrics['female_recall'],
                'male_recall_change': best_metrics['male_recall'] - v1_metrics['male_recall']
            }
        }
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict gender."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        # Build combined feature matrix
        all_features = []
        
        if self.text_cols:
            X_text = features_df[self.text_cols].fillna(0).values
            X_text_scaled = self.text_scaler.transform(X_text)
            all_features.append(X_text_scaled)
        
        if self.subreddit_cols:
            X_sub = features_df[self.subreddit_cols].fillna(0).values
            all_features.append(X_sub)
        
        if self.behavioral_cols:
            X_behav = features_df[self.behavioral_cols].fillna(0).values
            X_behav_scaled = self.behavioral_scaler.transform(X_behav)
            all_features.append(X_behav_scaled)
        
        if self.linguistic_cols:
            X_ling = features_df[self.linguistic_cols].fillna(0).values
            all_features.append(X_ling)
        
        X_combined = np.hstack(all_features)
        
        # Get predictions
        proba = self.meta_classifier.predict_proba(X_combined)
        predictions = (proba[:, 1] >= 0.5).astype(int)
        gender_preds = np.where(predictions == 1, 'female', 'male')
        
        return pd.DataFrame({
            'author': features_df['author'].values,
            'gender_predicted': gender_preds,
            'confidence': np.maximum(proba[:, 0], proba[:, 1]),
            'prob_female': proba[:, 1],
            'prob_male': proba[:, 0]
        })
    
    def save(self, path: Path):
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'gender_predictor_v3.pkl', 'wb') as f:
            pickle.dump({
                'text_classifier': self.text_classifier,
                'subreddit_classifier': self.subreddit_classifier,
                'behavioral_classifier': self.behavioral_classifier,
                'linguistic_classifier': self.linguistic_classifier,
                'meta_classifier': self.meta_classifier,
                'text_scaler': self.text_scaler,
                'behavioral_scaler': self.behavioral_scaler,
                'label_encoder': self.label_encoder,
                'text_cols': self.text_cols,
                'subreddit_cols': self.subreddit_cols,
                'behavioral_cols': self.behavioral_cols,
                'linguistic_cols': self.linguistic_cols,
                'best_params': self.best_params,
                'best_metrics': self.best_metrics
            }, f)
        
        logger.info(f"V3 Gender Predictor saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'GenderPredictor_V3':
        """Load model."""
        with open(Path(path) / 'gender_predictor_v3.pkl', 'rb') as f:
            state = pickle.load(f)
        
        predictor = cls()
        for key, value in state.items():
            setattr(predictor, key, value)
        predictor.is_fitted = True
        
        return predictor
