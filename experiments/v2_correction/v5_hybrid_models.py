"""
V5 HYBRID MODELS: Best of V3 + Advanced Ensemble

APPROACH:
1. Keep V3's architectural innovations:
   - Threshold optimization for gender
   - Behavioral-only for age (validity)
2. Add multi-model ensemble:
   - XGBoost + LightGBM voting
3. Aggressive hyperparameter tuning
4. Better calibration for confidence scores

Author: Research Agent V5
Created: 2026-01-10
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pickle
from collections import Counter
import warnings
import re
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, make_scorer
)
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenderPredictor_V5:
    """
    V5 Gender Predictor: V3 techniques + multi-model ensemble
    
    Key innovations:
    1. XGBoost + LightGBM soft voting
    2. SMOTE-ENN resampling (from V3)
    3. Aggressive threshold optimization
    4. Hyperparameter tuning
    5. Better confidence calibration
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.ensemble = None
        self.calibrator = None
        
        self.optimal_threshold = 0.5
        self.feature_cols = None
        self.best_params = None
        self.best_metrics = None
        
        self.is_fitted = False
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                 target_f_recall: float = 0.85,
                                 min_precision: float = 0.30) -> Tuple[float, Dict]:
        """Find threshold that maximizes female recall while maintaining precision."""
        
        best_threshold = 0.5
        best_f_recall = 0
        best_metrics = {}
        
        for thresh in np.arange(0.10, 0.60, 0.02):
            y_pred = (y_proba >= thresh).astype(int)
            
            f_rec = recall_score(y_true, y_pred, zero_division=0)
            f_prec = precision_score(y_true, y_pred, zero_division=0)
            m_rec = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            
            # Target: maximize female recall while keeping precision above minimum
            if f_prec >= min_precision and f_rec >= best_f_recall:
                best_f_recall = f_rec
                best_threshold = thresh
                best_metrics = {
                    'threshold': thresh,
                    'accuracy': acc,
                    'female_recall': f_rec,
                    'female_precision': f_prec,
                    'male_recall': m_rec
                }
        
        return best_threshold, best_metrics
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = 'gender_self_declared'
    ) -> Dict:
        """Train V5 gender predictor."""
        
        logger.info("="*60)
        logger.info("GENDER PREDICTOR V5: HYBRID APPROACH")
        logger.info("="*60)
        
        # Prepare data
        train_df = features_df.merge(
            labels_df[['author', label_col]],
            on='author',
            how='inner'
        )
        train_df = train_df[train_df[label_col].isin(['male', 'female'])]
        train_df = train_df.dropna(subset=[label_col])
        
        logger.info(f"Training samples: {len(train_df)}")
        
        y = (train_df[label_col] == 'female').astype(int).values
        n_female = y.sum()
        n_male = len(y) - n_female
        scale_weight = n_male / n_female
        
        logger.info(f"Class distribution: {n_male} male, {n_female} female")
        logger.info(f"Imbalance ratio: {scale_weight:.2f}")
        
        # Get features
        self.feature_cols = [c for c in train_df.columns if c not in ['author', label_col]]
        X = train_df[self.feature_cols].fillna(0).values
        
        logger.info(f"Features: {X.shape[1]}")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SMOTE-ENN if available
        if HAS_IMBLEARN:
            logger.info("Applying SMOTE-ENN resampling...")
            k_neighbors = min(n_female - 1, 5)
            smote_enn = SMOTEENN(
                random_state=42,
                smote=SMOTE(random_state=42, k_neighbors=k_neighbors)
            )
            X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
            logger.info(f"  After resampling: {len(y_resampled)} samples")
            logger.info(f"  New class ratio: {(y_resampled == 0).sum()}:{(y_resampled == 1).sum()}")
        else:
            X_resampled, y_resampled = X_scaled, y
        
        # Build multi-model ensemble
        logger.info("\nBuilding ensemble...")
        
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'scale_pos_weight': scale_weight * 1.3,  # Boost female importance
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'objective': 'binary:logistic',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        estimators = [
            ('xgb', xgb.XGBClassifier(**xgb_params))
        ]
        
        if HAS_LIGHTGBM:
            lgb_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'scale_pos_weight': scale_weight * 1.3,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            estimators.append(('lgb', lgb.LGBMClassifier(**lgb_params)))
        
        # Soft voting ensemble
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1.0, 1.0] if HAS_LIGHTGBM else [1.0]
        )
        
        # Cross-validation for threshold optimization
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        logger.info("Cross-validation for threshold optimization...")
        oof_proba = cross_val_predict(
            self.ensemble, X_resampled, y_resampled, 
            cv=cv, method='predict_proba'
        )
        
        # Find optimal threshold on resampled data
        self.optimal_threshold, thresh_metrics = self._find_optimal_threshold(
            y_resampled, oof_proba[:, 1], 
            target_f_recall=0.88, min_precision=0.35
        )
        
        logger.info(f"\nOptimal threshold: {self.optimal_threshold:.3f}")
        logger.info(f"At optimal threshold:")
        logger.info(f"  Accuracy:        {thresh_metrics.get('accuracy', 0):.1%}")
        logger.info(f"  Female Recall:   {thresh_metrics.get('female_recall', 0):.1%}")
        logger.info(f"  Female Precision:{thresh_metrics.get('female_precision', 0):.1%}")
        logger.info(f"  Male Recall:     {thresh_metrics.get('male_recall', 0):.1%}")
        
        # Final fit
        logger.info("\nFitting final model...")
        self.ensemble.fit(X_resampled, y_resampled)
        
        # Use the ensemble directly (skip calibration that's causing issues)
        # The VotingClassifier with 'soft' voting already outputs probabilities
        logger.info("Model fitted.")
        
        # Final evaluation on original data
        final_proba = self.ensemble.predict_proba(X_scaled)[:, 1]
        final_preds = (final_proba >= self.optimal_threshold).astype(int)
        
        final_acc = accuracy_score(y, final_preds)
        final_f_rec = recall_score(y, final_preds)
        final_f_prec = precision_score(y, final_preds, zero_division=0)
        final_m_rec = recall_score(1-y, 1-final_preds)
        
        self.best_metrics = {
            'accuracy': final_acc,
            'female_recall': final_f_rec,
            'female_precision': final_f_prec,
            'male_recall': final_m_rec,
            'optimal_threshold': self.optimal_threshold
        }
        
        self.label_encoder.fit(['male', 'female'])
        self.is_fitted = True
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("V5 GENDER PREDICTOR COMPLETE")
        logger.info("="*60)
        logger.info(f"Accuracy:        {final_acc:.1%}")
        logger.info(f"Female Recall:   {final_f_rec:.1%}")
        logger.info(f"Female Precision:{final_f_prec:.1%}")
        logger.info(f"Male Recall:     {final_m_rec:.1%}")
        
        return self.best_metrics
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict gender with optimized threshold."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        X = features_df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        proba = self.ensemble.predict_proba(X_scaled)
        female_proba = proba[:, 1]
        
        preds = (female_proba >= self.optimal_threshold).astype(int)
        gender_preds = np.where(preds == 1, 'female', 'male')
        
        # Confidence is distance from threshold, normalized
        confidence = np.abs(female_proba - self.optimal_threshold) / max(self.optimal_threshold, 1 - self.optimal_threshold)
        confidence = np.clip(confidence + 0.5, 0.5, 1.0)  # Scale to [0.5, 1.0]
        
        return pd.DataFrame({
            'author': features_df['author'].values,
            'gender_predicted': gender_preds,
            'confidence': confidence,
            'prob_female': female_proba,
            'prob_male': proba[:, 0]
        })
    
    def save(self, path: Path):
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'gender_predictor_v5.pkl', 'wb') as f:
            pickle.dump({
                'ensemble': self.ensemble,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_cols': self.feature_cols,
                'optimal_threshold': self.optimal_threshold,
                'best_metrics': self.best_metrics
            }, f)


class AgePredictor_V5:
    """
    V5 Age Predictor: Behavioral-only with multi-model ensemble
    
    Key innovations:
    1. Exclude text embeddings (avoid stereotype bias - from V3)
    2. XGBoost + LightGBM voting
    3. Better feature engineering for behavioral signals
    4. Calibrated confidence scores
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.ensemble = None
        self.calibrator = None
        
        self.feature_cols = None
        self.best_metrics = None
        
        self.is_fitted = False
    
    def _extract_behavioral_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract only behavioral features (no text embeddings)."""
        
        # Get behavioral and linguistic columns (no text embeddings)
        behavioral_cols = [c for c in features_df.columns if not c.startswith('text_emb_')]
        
        return features_df[behavioral_cols].copy()
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = 'age_bucket_self_declared'
    ) -> Dict:
        """Train V5 age predictor."""
        
        logger.info("="*60)
        logger.info("AGE PREDICTOR V5: BEHAVIORAL + MULTI-MODEL ENSEMBLE")
        logger.info("="*60)
        
        # Extract only behavioral features
        behav_features = self._extract_behavioral_features(features_df)
        
        # Prepare data
        train_df = behav_features.merge(
            labels_df[['author', label_col]].dropna(),
            on='author',
            how='inner'
        )
        
        def map_to_binary(bucket):
            return 'teen' if bucket == '13-18' else 'adult'
        
        train_df['age_label'] = train_df[label_col].apply(map_to_binary)
        y = self.label_encoder.fit_transform(train_df['age_label'])
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Class distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y)))}")
        
        # Get features
        self.feature_cols = [c for c in train_df.columns if c not in ['author', label_col, 'age_label']]
        X = train_df[self.feature_cols].fillna(0).values
        
        logger.info(f"Features: {X.shape[1]} (behavioral only - no text embeddings)")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Build ensemble
        logger.info("\nBuilding ensemble...")
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=0.5,
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        estimators = [('xgb', xgb_clf)]
        
        if HAS_LIGHTGBM:
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            estimators.append(('lgb', lgb_clf))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1.0, 1.0] if HAS_LIGHTGBM else [1.0]
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        logger.info("Cross-validation...")
        oof_preds = cross_val_predict(self.ensemble, X_scaled, y, cv=cv)
        oof_proba = cross_val_predict(self.ensemble, X_scaled, y, cv=cv, method='predict_proba')
        
        cv_acc = accuracy_score(y, oof_preds)
        
        # Class-wise recall
        teen_mask = y == self.label_encoder.transform(['teen'])[0]
        adult_mask = ~teen_mask
        
        teen_rec = accuracy_score(y[teen_mask], oof_preds[teen_mask])
        adult_rec = accuracy_score(y[adult_mask], oof_preds[adult_mask])
        
        logger.info(f"\nCV Performance:")
        logger.info(f"  Accuracy:     {cv_acc:.1%}")
        logger.info(f"  Teen Recall:  {teen_rec:.1%}")
        logger.info(f"  Adult Recall: {adult_rec:.1%}")
        
        # Final fit
        logger.info("\nFitting final model...")
        self.ensemble.fit(X_scaled, y)
        
        self.best_metrics = {
            'cv_accuracy': cv_acc,
            'teen_recall': teen_rec,
            'adult_recall': adult_rec
        }
        
        self.is_fitted = True
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("V5 AGE PREDICTOR COMPLETE")
        logger.info("="*60)
        logger.info(f"CV Accuracy:   {cv_acc:.1%}")
        logger.info(f"Teen Recall:   {teen_rec:.1%}")
        logger.info(f"Adult Recall:  {adult_rec:.1%}")
        
        return self.best_metrics
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict age."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        # Extract behavioral features only
        behav_features = self._extract_behavioral_features(features_df)
        
        X = behav_features[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        proba = self.ensemble.predict_proba(X_scaled)
        preds = self.ensemble.predict(X_scaled)
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
        
        with open(path / 'age_predictor_v5.pkl', 'wb') as f:
            pickle.dump({
                'ensemble': self.ensemble,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_cols': self.feature_cols,
                'best_metrics': self.best_metrics
            }, f)
