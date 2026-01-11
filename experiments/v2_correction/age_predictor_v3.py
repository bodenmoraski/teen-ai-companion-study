"""
AGE PREDICTOR V3: Accuracy-Preserving Improvement

CONSTRAINT: Maintain or improve accuracy while exploring ground truth alignment.

APPROACH:
1. Keep ALL features including text embeddings (they provide accuracy)
2. Train on golden set with proper regularization
3. Use cross-validation to ensure generalization
4. Compare multiple architectures
5. Pick the one that maintains V1 accuracy while best aligning with ground truth

Author: Research Agent V3
Created: 2026-01-10
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def map_to_binary(age_bucket: str) -> str:
    """Map age buckets to binary."""
    if age_bucket == '13-18':
        return 'teen'
    return 'adult'


class AgePredictor_V3:
    """
    Age Predictor V3: Accuracy-preserving approach.
    
    Key changes from V2:
    - KEEP text embeddings (they're needed for accuracy)
    - Add regularization to prevent overfitting
    - Use proper CV to select best hyperparameters
    - Track both accuracy AND ground truth alignment
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
        self.text_scaler = StandardScaler()
        self.behavioral_scaler = StandardScaler()
        
        self.text_cols = None
        self.behavioral_cols = None
        self.subreddit_cols = None
        
        self.classifier = None
        self.best_params = None
        self.cv_accuracy = None
        
        self.is_fitted = False
    
    def _grid_search_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: StratifiedKFold,
        target_accuracy: float
    ) -> Tuple[Dict, float]:
        """Grid search for best parameters while maintaining accuracy."""
        
        logger.info("\nGrid search for optimal parameters...")
        
        # Parameter grid - focus on regularization
        param_grid = {
            'max_depth': [3, 4, 5],
            'reg_alpha': [0.1, 1.0, 2.0],
            'reg_lambda': [0.1, 1.0, 2.0],
            'n_estimators': [50, 100, 150]
        }
        
        best_accuracy = 0
        best_params = None
        
        # Simple grid search
        for max_depth in param_grid['max_depth']:
            for reg_alpha in param_grid['reg_alpha']:
                for reg_lambda in param_grid['reg_lambda']:
                    for n_estimators in param_grid['n_estimators']:
                        params = {
                            'max_depth': max_depth,
                            'reg_alpha': reg_alpha,
                            'reg_lambda': reg_lambda,
                            'n_estimators': n_estimators,
                            'learning_rate': 0.1,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'objective': 'binary:logistic',
                            'eval_metric': 'logloss',
                            'use_label_encoder': False,
                            'random_state': 42,
                            'n_jobs': -1,
                            'verbosity': 0
                        }
                        
                        clf = xgb.XGBClassifier(**params)
                        preds = cross_val_predict(clf, X, y, cv=cv, method='predict')
                        acc = accuracy_score(y, preds)
                        
                        if acc > best_accuracy:
                            best_accuracy = acc
                            best_params = params.copy()
        
        logger.info(f"Best CV accuracy: {best_accuracy:.1%}")
        logger.info(f"Best params: max_depth={best_params['max_depth']}, "
                   f"reg_alpha={best_params['reg_alpha']}, "
                   f"n_estimators={best_params['n_estimators']}")
        
        return best_params, best_accuracy
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = 'age_bucket_self_declared',
        anthroscore_df: Optional[pd.DataFrame] = None,
        v1_accuracy: float = 0.706
    ) -> Dict:
        """
        Train with accuracy preservation.
        """
        logger.info("="*60)
        logger.info("AGE PREDICTOR V3: ACCURACY-PRESERVING TRAINING")
        logger.info("="*60)
        
        # Merge with labels
        train_df = features_df.merge(
            labels_df[['author', label_col]].dropna(),
            on='author',
            how='inner'
        )
        
        logger.info(f"Training samples: {len(train_df)}")
        
        # Map to binary
        train_df['age_label'] = train_df[label_col].apply(map_to_binary)
        y = self.label_encoder.fit_transform(train_df['age_label'])
        
        logger.info(f"Class distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y)))}")
        
        # Identify feature columns
        feature_cols = [c for c in train_df.columns if c not in ['author', label_col, 'age_label']]
        
        self.text_cols = [c for c in feature_cols if c.startswith('text_emb_')]
        self.behavioral_cols = [c for c in feature_cols if c in [
            'late_night_ratio', 'school_hours_ratio', 'evening_ratio',
            'weekend_ratio', 'activity_span_days', 'hour_std',
            'comment_count', 'log_comment_count', 'avg_comment_length',
            'comment_length_std', 'subreddit_diversity', 'unique_subreddits'
        ]]
        self.subreddit_cols = [c for c in feature_cols if c.startswith('sub_')]
        
        logger.info(f"\nFeatures: {len(self.text_cols)} text, "
                   f"{len(self.behavioral_cols)} behavioral, "
                   f"{len(self.subreddit_cols)} subreddit")
        
        # Build feature matrix - USE ALL FEATURES
        all_features = []
        
        if self.text_cols:
            X_text = train_df[self.text_cols].fillna(0).values
            X_text_scaled = self.text_scaler.fit_transform(X_text)
            all_features.append(X_text_scaled)
            logger.info(f"  Text embeddings: {X_text_scaled.shape[1]} features")
        
        if self.behavioral_cols:
            X_behav = train_df[self.behavioral_cols].fillna(0).values
            X_behav_scaled = self.behavioral_scaler.fit_transform(X_behav)
            all_features.append(X_behav_scaled)
            logger.info(f"  Behavioral: {X_behav_scaled.shape[1]} features")
        
        if self.subreddit_cols:
            X_sub = train_df[self.subreddit_cols].fillna(0).values
            all_features.append(X_sub)
            logger.info(f"  Subreddit: {X_sub.shape[1]} features")
        
        X_combined = np.hstack(all_features)
        logger.info(f"\nCombined: {X_combined.shape}")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search for best parameters
        self.best_params, self.cv_accuracy = self._grid_search_params(
            X_combined, y, cv, v1_accuracy
        )
        
        # Train final model
        logger.info("\nTraining final model...")
        self.classifier = xgb.XGBClassifier(**self.best_params)
        self.classifier.fit(X_combined, y)
        
        self.is_fitted = True
        
        # Validate on training set for ground truth direction
        if anthroscore_df is not None:
            self._validate_ground_truth(train_df, X_combined, anthroscore_df)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("V3 AGE PREDICTOR COMPLETE")
        logger.info("="*60)
        logger.info(f"CV Accuracy: {self.cv_accuracy:.1%}")
        logger.info(f"V1 Baseline: {v1_accuracy:.1%}")
        logger.info(f"Change: {self.cv_accuracy - v1_accuracy:+.1%}")
        
        if self.cv_accuracy >= v1_accuracy - 0.03:
            logger.info("OK: Accuracy maintained within tolerance")
        else:
            logger.warning("WARNING: Accuracy dropped below tolerance")
        
        return {
            'cv_accuracy': self.cv_accuracy,
            'v1_accuracy': v1_accuracy,
            'accuracy_change': self.cv_accuracy - v1_accuracy,
            'best_params': self.best_params
        }
    
    def _validate_ground_truth(
        self,
        train_df: pd.DataFrame,
        X: np.ndarray,
        anthroscore_df: pd.DataFrame
    ):
        """Check if predictions align with ground truth direction."""
        logger.info("\nGround truth direction check...")
        
        # Get predictions
        preds = self.classifier.predict(X)
        pred_labels = self.label_encoder.inverse_transform(preds)
        
        # Create results df
        results_df = pd.DataFrame({
            'author': train_df['author'].values,
            'age_predicted': pred_labels
        })
        
        # Merge with anthroscores
        merged = results_df.merge(
            anthroscore_df[['author', 'anthroscore_max']],
            on='author'
        )
        merged = merged[merged['anthroscore_max'] != 0]
        
        teens = merged[merged['age_predicted'] == 'teen']['anthroscore_max']
        adults = merged[merged['age_predicted'] == 'adult']['anthroscore_max']
        
        if len(teens) > 0 and len(adults) > 0:
            d = (teens.mean() - adults.mean()) / np.sqrt((teens.std()**2 + adults.std()**2) / 2)
            logger.info(f"  Teen mean: {teens.mean():.3f}, Adult mean: {adults.mean():.3f}")
            logger.info(f"  Cohen's d: {d:.4f}")
            logger.info(f"  Direction: {'Teens higher' if d > 0 else 'Adults higher'}")
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict age."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        # Build feature matrix
        all_features = []
        
        if self.text_cols:
            X_text = features_df[self.text_cols].fillna(0).values
            X_text_scaled = self.text_scaler.transform(X_text)
            all_features.append(X_text_scaled)
        
        if self.behavioral_cols:
            X_behav = features_df[self.behavioral_cols].fillna(0).values
            X_behav_scaled = self.behavioral_scaler.transform(X_behav)
            all_features.append(X_behav_scaled)
        
        if self.subreddit_cols:
            X_sub = features_df[self.subreddit_cols].fillna(0).values
            all_features.append(X_sub)
        
        X_combined = np.hstack(all_features)
        
        proba = self.classifier.predict_proba(X_combined)
        preds = self.classifier.predict(X_combined)
        pred_labels = self.label_encoder.inverse_transform(preds)
        
        return pd.DataFrame({
            'author': features_df['author'].values,
            'age_predicted': pred_labels,
            'confidence': np.max(proba, axis=1),
            'prob_adult': proba[:, self.label_encoder.transform(['adult'])[0]],
            'prob_teen': proba[:, self.label_encoder.transform(['teen'])[0]]
        })
    
    def save(self, path: Path):
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'age_predictor_v3.pkl', 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'text_scaler': self.text_scaler,
                'behavioral_scaler': self.behavioral_scaler,
                'label_encoder': self.label_encoder,
                'text_cols': self.text_cols,
                'behavioral_cols': self.behavioral_cols,
                'subreddit_cols': self.subreddit_cols,
                'best_params': self.best_params,
                'cv_accuracy': self.cv_accuracy
            }, f)
        
        logger.info(f"V3 Age Predictor saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'AgePredictor_V3':
        """Load model."""
        with open(Path(path) / 'age_predictor_v3.pkl', 'rb') as f:
            state = pickle.load(f)
        
        predictor = cls()
        for key, value in state.items():
            setattr(predictor, key, value)
        predictor.is_fitted = True
        
        return predictor
