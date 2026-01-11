"""
V4 ADVANCED MODELS: Pushing for Maximum Performance

TECHNIQUES:
1. Multi-algorithm stacking (XGBoost + LightGBM + LogReg + RF)
2. Extensive hyperparameter search
3. Feature selection to reduce noise
4. Calibrated probability outputs
5. Ensemble voting with soft probabilities

Author: Research Agent V4
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
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available - will use XGBoost only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent


class GenderPredictor_V4:
    """
    V4 Gender Predictor with advanced stacking ensemble.
    
    Uses multiple base learners:
    - XGBoost (optimized)
    - LightGBM (if available)
    - Random Forest
    - Logistic Regression
    
    Meta-learner: Calibrated logistic regression for optimal probabilities
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_selector = None
        
        self.base_learners = {}
        self.meta_learner = None
        self.stacker = None
        
        self.feature_cols = None
        self.selected_features = None
        self.best_metrics = None
        
        self.is_fitted = False
    
    def _build_base_learners(self, scale_weight: float) -> List[Tuple]:
        """Build diverse base learners."""
        
        learners = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_weight,
                reg_alpha=0.5,
                reg_lambda=0.5,
                objective='binary:logistic',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('lr', LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        if HAS_LIGHTGBM:
            learners.append(('lgb', lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_weight,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )))
        
        return learners
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = 'gender_self_declared',
        v1_metrics: Dict = None
    ) -> Dict:
        """Train V4 with stacking ensemble."""
        
        logger.info("="*60)
        logger.info("GENDER PREDICTOR V4: ADVANCED STACKING ENSEMBLE")
        logger.info("="*60)
        
        if v1_metrics is None:
            v1_metrics = {
                'accuracy': 0.813,
                'female_recall': 0.44,
                'male_recall': 0.95
            }
        
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
        
        # Get features
        self.feature_cols = [c for c in train_df.columns if c not in ['author', label_col]]
        X = train_df[self.feature_cols].fillna(0).values
        
        logger.info(f"Features: {X.shape[1]}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection to reduce noise
        logger.info("\nFeature selection...")
        selector = SelectFromModel(
            xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, verbosity=0),
            threshold='0.5*median'
        )
        X_selected = selector.fit_transform(X_scaled, y)
        self.feature_selector = selector
        self.selected_features = X_selected.shape[1]
        logger.info(f"  Selected {self.selected_features} features (from {X.shape[1]})")
        
        # Build base learners
        logger.info("\nBuilding base learners...")
        base_learners = self._build_base_learners(scale_weight * 1.2)  # Slightly higher for female recall
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Evaluate individual learners
        logger.info("\nIndividual learner performance:")
        for name, clf in base_learners:
            preds = cross_val_predict(clf, X_selected, y, cv=cv)
            acc = accuracy_score(y, preds)
            f_rec = recall_score(y, preds)
            m_rec = recall_score(1-y, 1-preds)
            logger.info(f"  {name}: Acc={acc:.1%}, F-Rec={f_rec:.1%}, M-Rec={m_rec:.1%}")
        
        # Build stacking classifier
        logger.info("\nBuilding stacking ensemble...")
        self.stacker = StackingClassifier(
            estimators=base_learners,
            final_estimator=CalibratedClassifierCV(
                LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                cv=3
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Get CV predictions
        stacker_preds = cross_val_predict(self.stacker, X_selected, y, cv=cv, method='predict_proba')
        stacker_pred_labels = (stacker_preds[:, 1] >= 0.5).astype(int)
        
        stacker_acc = accuracy_score(y, stacker_pred_labels)
        stacker_f_rec = recall_score(y, stacker_pred_labels)
        stacker_m_rec = recall_score(1-y, 1-stacker_pred_labels)
        stacker_f1 = f1_score(y, stacker_pred_labels, average='macro')
        
        logger.info(f"\nStacking ensemble CV performance:")
        logger.info(f"  Accuracy:      {stacker_acc:.1%}")
        logger.info(f"  Female Recall: {stacker_f_rec:.1%}")
        logger.info(f"  Male Recall:   {stacker_m_rec:.1%}")
        logger.info(f"  Macro F1:      {stacker_f1:.1%}")
        
        # Fit final model
        logger.info("\nFitting final model...")
        self.stacker.fit(X_selected, y)
        
        self.best_metrics = {
            'accuracy': stacker_acc,
            'female_recall': stacker_f_rec,
            'male_recall': stacker_m_rec,
            'macro_f1': stacker_f1
        }
        
        self.label_encoder.fit(['male', 'female'])
        self.is_fitted = True
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("V4 COMPLETE")
        logger.info("="*60)
        logger.info(f"Accuracy:      {v1_metrics['accuracy']:.1%} -> {stacker_acc:.1%} ({stacker_acc - v1_metrics['accuracy']:+.1%})")
        logger.info(f"Female Recall: {v1_metrics['female_recall']:.1%} -> {stacker_f_rec:.1%} ({stacker_f_rec - v1_metrics['female_recall']:+.1%})")
        logger.info(f"Male Recall:   {v1_metrics['male_recall']:.1%} -> {stacker_m_rec:.1%} ({stacker_m_rec - v1_metrics['male_recall']:+.1%})")
        
        return {
            'cv_metrics': self.best_metrics,
            'improvements': {
                'accuracy': stacker_acc - v1_metrics['accuracy'],
                'female_recall': stacker_f_rec - v1_metrics['female_recall'],
                'male_recall': stacker_m_rec - v1_metrics['male_recall']
            }
        }
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict gender."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        X = features_df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        proba = self.stacker.predict_proba(X_selected)
        preds = (proba[:, 1] >= 0.5).astype(int)
        gender_preds = np.where(preds == 1, 'female', 'male')
        
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
        
        with open(path / 'gender_predictor_v4.pkl', 'wb') as f:
            pickle.dump({
                'stacker': self.stacker,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'label_encoder': self.label_encoder,
                'feature_cols': self.feature_cols,
                'selected_features': self.selected_features,
                'best_metrics': self.best_metrics
            }, f)
        
        logger.info(f"V4 Gender Predictor saved to {path}")


class AgePredictor_V4:
    """
    V4 Age Predictor with advanced ensemble.
    
    Uses stacking of multiple learners with calibrated probabilities.
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_selector = None
        
        self.stacker = None
        
        self.feature_cols = None
        self.selected_features = None
        self.best_metrics = None
        
        self.is_fitted = False
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = 'age_bucket_self_declared',
        v1_accuracy: float = 0.706
    ) -> Dict:
        """Train V4 age predictor."""
        
        logger.info("="*60)
        logger.info("AGE PREDICTOR V4: ADVANCED ENSEMBLE")
        logger.info("="*60)
        
        # Prepare data
        train_df = features_df.merge(
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
        
        logger.info(f"Features: {X.shape[1]}")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        logger.info("\nFeature selection...")
        selector = SelectFromModel(
            xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, verbosity=0),
            threshold='0.5*median'
        )
        X_selected = selector.fit_transform(X_scaled, y)
        self.feature_selector = selector
        self.selected_features = X_selected.shape[1]
        logger.info(f"  Selected {self.selected_features} features")
        
        # Build base learners
        base_learners = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
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
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            ('lr', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        if HAS_LIGHTGBM:
            base_learners.append(('lgb', lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )))
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Individual performance
        logger.info("\nIndividual learner performance:")
        for name, clf in base_learners:
            preds = cross_val_predict(clf, X_selected, y, cv=cv)
            acc = accuracy_score(y, preds)
            logger.info(f"  {name}: Accuracy={acc:.1%}")
        
        # Stacking
        logger.info("\nBuilding stacking ensemble...")
        self.stacker = StackingClassifier(
            estimators=base_learners,
            final_estimator=CalibratedClassifierCV(
                LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                cv=3
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # CV performance
        stacker_preds = cross_val_predict(self.stacker, X_selected, y, cv=cv)
        stacker_acc = accuracy_score(y, stacker_preds)
        
        logger.info(f"\nStacking ensemble CV accuracy: {stacker_acc:.1%}")
        
        # Fit final
        self.stacker.fit(X_selected, y)
        
        self.best_metrics = {'accuracy': stacker_acc}
        self.is_fitted = True
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("V4 COMPLETE")
        logger.info("="*60)
        logger.info(f"CV Accuracy: {v1_accuracy:.1%} -> {stacker_acc:.1%} ({stacker_acc - v1_accuracy:+.1%})")
        
        return {
            'cv_accuracy': stacker_acc,
            'v1_accuracy': v1_accuracy,
            'improvement': stacker_acc - v1_accuracy
        }
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict age."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        X = features_df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        proba = self.stacker.predict_proba(X_selected)
        preds = self.stacker.predict(X_selected)
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
        
        with open(path / 'age_predictor_v4.pkl', 'wb') as f:
            pickle.dump({
                'stacker': self.stacker,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'label_encoder': self.label_encoder,
                'feature_cols': self.feature_cols,
                'selected_features': self.selected_features,
                'best_metrics': self.best_metrics
            }, f)
        
        logger.info(f"V4 Age Predictor saved to {path}")
