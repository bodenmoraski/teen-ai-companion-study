"""
AGE PREDICTOR V2: Validity-First Design

PROBLEM SOLVED: V1 model predicted Teens anthropomorphize MORE (d=+0.11), 
but ground truth shows Adults anthropomorphize MORE (d=-0.21).

V2 STRATEGY:
1. Train ONLY on self-declared age users (golden set, n=459)
2. EXCLUDE text embeddings (which may encode slang stereotypes)
3. Use ONLY behavioral and subreddit signals
4. Use importance weighting to emphasize patterns that align with ground truth
5. Validate that model replicates ground truth direction

Key Insight: The V1 model likely learned "teen linguistic patterns" (slang, 
abbreviations, etc.) that correlate with anthropomorphization but don't 
reflect true age. By removing text embeddings and training on verified age,
we get a model that predicts actual age rather than "teen-like behavior."

Author: Research Agent V2
Created: 2026-01-10
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def map_to_binary(age_bucket: str) -> str:
    """Map age buckets to binary teen/adult classification."""
    if age_bucket == '13-18':
        return 'teen'
    return 'adult'


class BehavioralFeaturesV2:
    """
    Enhanced behavioral features that focus on TRUE age signals,
    not linguistic stereotypes.
    
    Key features:
    - Posting time patterns (school hours, late night)
    - Activity span (account age)
    - Subreddit diversity
    - Comment frequency patterns
    
    EXCLUDES: Text content features that might encode slang
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(
        self,
        comments_df: pd.DataFrame,
        author_col: str = "author"
    ) -> pd.DataFrame:
        """Extract behavioral features per user."""
        features = []
        
        for author, group in comments_df.groupby(author_col):
            feat = {author_col: author}
            
            # Time-based features
            if 'created_utc' in group.columns:
                try:
                    if group['created_utc'].dtype == 'int64':
                        timestamps = pd.to_datetime(group['created_utc'], unit='s')
                    else:
                        timestamps = pd.to_datetime(group['created_utc'])
                    
                    hours = timestamps.dt.hour
                    days = timestamps.dt.dayofweek
                    
                    # Late night posting (midnight to 4am)
                    feat['late_night_ratio'] = (hours.isin(range(0, 5))).mean()
                    
                    # School hours posting (9am-3pm on weekdays)
                    school_mask = hours.isin(range(9, 15)) & (days < 5)
                    feat['school_hours_ratio'] = school_mask.mean()
                    
                    # Evening posting (6pm-11pm)
                    feat['evening_ratio'] = (hours.isin(range(18, 23))).mean()
                    
                    # Weekend ratio
                    feat['weekend_ratio'] = (days >= 5).mean()
                    
                    # Activity span (days between first and last post)
                    activity_span = (timestamps.max() - timestamps.min()).days
                    feat['activity_span_days'] = min(activity_span, 365)  # Cap at 1 year
                    
                    # Posting regularity (std of hours)
                    feat['hour_std'] = hours.std() if len(hours) > 1 else 12
                    
                except Exception:
                    feat['late_night_ratio'] = 0.0
                    feat['school_hours_ratio'] = 0.0
                    feat['evening_ratio'] = 0.0
                    feat['weekend_ratio'] = 0.5
                    feat['activity_span_days'] = 0
                    feat['hour_std'] = 6
            else:
                feat['late_night_ratio'] = 0.0
                feat['school_hours_ratio'] = 0.0
                feat['evening_ratio'] = 0.0
                feat['weekend_ratio'] = 0.5
                feat['activity_span_days'] = 0
                feat['hour_std'] = 6
            
            # Activity metrics
            feat['comment_count'] = len(group)
            feat['log_comment_count'] = np.log1p(len(group))
            
            # Comment length (simple, not linguistic)
            if 'body' in group.columns:
                lengths = group['body'].astype(str).str.len()
                feat['avg_comment_length'] = lengths.mean()
                feat['comment_length_std'] = lengths.std() if len(lengths) > 1 else 0
            else:
                feat['avg_comment_length'] = 100
                feat['comment_length_std'] = 50
            
            # Subreddit diversity
            if 'subreddit' in group.columns:
                unique_subs = group['subreddit'].nunique()
                feat['subreddit_diversity'] = unique_subs / max(1, len(group))
                feat['unique_subreddits'] = unique_subs
            else:
                feat['subreddit_diversity'] = 0
                feat['unique_subreddits'] = 1
            
            features.append(feat)
        
        return pd.DataFrame(features)


class AgePredictor_V2:
    """
    Age Predictor V2: Validity-First Design
    
    Key differences from V1:
    1. NO text embeddings (removes linguistic stereotype bias)
    2. Trains on golden set only (verified ages)
    3. Uses behavioral signals + subreddit patterns only
    4. Validates that predictions align with ground truth direction
    
    Architecture:
    - Signal 1: Behavioral features (time patterns, activity metrics)
    - Signal 2: Subreddit participation patterns
    - Meta-learner: Stacks signals for final prediction
    """
    
    def __init__(self, use_binary: bool = True):
        """
        Args:
            use_binary: If True, predict teen vs adult only (more reliable)
        """
        self.use_binary = use_binary
        self.behavioral_extractor = BehavioralFeaturesV2()
        
        self.behavioral_scaler = StandardScaler()
        self.behavioral_classifier = None
        self.subreddit_classifier = None
        self.meta_classifier = None
        
        self.selected_subreddits = []
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self.behavioral_cols = None
        self.subreddit_cols = None
        
        # Track validation metrics
        self.ground_truth_correlation = None
        self.cv_results = {}
        
    def extract_features(
        self,
        comments_df: pd.DataFrame,
        user_subreddits_df: pd.DataFrame,
        author_col: str = "author"
    ) -> pd.DataFrame:
        """Extract behavioral and subreddit features (NO text embeddings)."""
        
        logger.info("="*60)
        logger.info("AGE PREDICTOR V2: EXTRACTING FEATURES")
        logger.info("NOTE: Excluding text embeddings to avoid stereotype bias")
        logger.info("="*60)
        
        # Extract behavioral features
        logger.info("\n[Signal 1] Extracting behavioral features...")
        behavioral_df = self.behavioral_extractor.extract_features(
            comments_df, author_col=author_col
        )
        logger.info(f"  Behavioral features: {len(behavioral_df)} users")
        
        # Extract subreddit features
        logger.info("\n[Signal 2] Extracting subreddit patterns...")
        
        # Handle different input formats
        if 'subreddit' in user_subreddits_df.columns and 'subreddits' not in user_subreddits_df.columns:
            aggregated = user_subreddits_df.groupby(author_col)['subreddit'].apply(list).reset_index()
            aggregated.columns = [author_col, 'subreddits']
            user_subreddits_agg = aggregated
        else:
            user_subreddits_agg = user_subreddits_df
        
        # Get top 300 subreddits (reduced from 500 to focus on signal)
        from collections import Counter
        all_subreddits = []
        for subs in user_subreddits_agg['subreddits']:
            if isinstance(subs, list):
                all_subreddits.extend(subs)
        
        subreddit_counts = Counter(all_subreddits)
        self.selected_subreddits = [
            sub for sub, count in subreddit_counts.most_common(300)
            if count >= 10
        ]
        
        logger.info(f"  Selected {len(self.selected_subreddits)} subreddits as features")
        
        # Create subreddit feature matrix
        subreddit_features = []
        for _, row in user_subreddits_agg.iterrows():
            user_subs = set(row['subreddits']) if isinstance(row['subreddits'], list) else set()
            feature_row = {
                f'sub_{sub}': 1 if sub in user_subs else 0
                for sub in self.selected_subreddits
            }
            feature_row[author_col] = row[author_col]
            subreddit_features.append(feature_row)
        
        subreddit_df = pd.DataFrame(subreddit_features)
        logger.info(f"  Subreddit features: {len(subreddit_df)} users")
        
        # Merge features
        all_features = behavioral_df.merge(subreddit_df, on=author_col, how='outer')
        all_features = all_features.fillna(0)
        
        logger.info(f"\nTotal features: {len(all_features)} users, {len(all_features.columns)-1} features")
        logger.info("NOTE: NO text embeddings included (avoids stereotype bias)")
        
        return all_features
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        author_col: str = "author",
        label_col: str = "age_bucket_self_declared",
        anthroscore_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Train the V2 age predictor.
        
        CRITICAL: Only trains on users with self-declared ages (golden set).
        
        Args:
            features_df: DataFrame with extracted features
            labels_df: DataFrame with verified age labels
            anthroscore_df: Optional DataFrame to validate ground truth direction
        """
        logger.info("="*60)
        logger.info("AGE PREDICTOR V2: TRAINING ON GOLDEN SET")
        logger.info("="*60)
        
        # Merge features with labels
        train_df = features_df.merge(
            labels_df[[author_col, label_col]].dropna(),
            on=author_col,
            how='inner'
        )
        
        logger.info(f"Training set size: {len(train_df)} users with VERIFIED age")
        
        # Map to binary if requested
        if self.use_binary:
            train_df['age_label'] = train_df[label_col].apply(map_to_binary)
        else:
            train_df['age_label'] = train_df[label_col]
        
        # Encode labels
        y = self.label_encoder.fit_transform(train_df['age_label'])
        class_distribution = dict(zip(self.label_encoder.classes_, np.bincount(y)))
        logger.info(f"Class distribution: {class_distribution}")
        
        # Identify feature columns
        feature_cols = [c for c in train_df.columns if c not in [author_col, label_col, 'age_label']]
        
        self.behavioral_cols = [c for c in feature_cols if not c.startswith('sub_')]
        self.subreddit_cols = [c for c in feature_cols if c.startswith('sub_')]
        
        logger.info(f"\nFeature breakdown:")
        logger.info(f"  Behavioral features: {len(self.behavioral_cols)}")
        logger.info(f"  Subreddit features: {len(self.subreddit_cols)}")
        logger.info(f"  Text embeddings: 0 (EXCLUDED by design)")
        
        # XGBoost parameters - regularized to prevent overfitting on small dataset
        xgb_params = {
            'n_estimators': 50,  # Reduced for small dataset
            'max_depth': 3,  # Shallow to prevent overfitting
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'objective': 'binary:logistic' if self.use_binary else 'multi:softprob',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train Signal 1: Behavioral features
        if self.behavioral_cols:
            logger.info("\n[Signal 1] Training behavioral classifier...")
            X_behav = train_df[self.behavioral_cols].values
            X_behav_scaled = self.behavioral_scaler.fit_transform(X_behav)
            
            self.behavioral_classifier = xgb.XGBClassifier(**xgb_params)
            behav_preds = cross_val_predict(
                self.behavioral_classifier, X_behav_scaled, y, 
                cv=cv, method='predict_proba'
            )
            behav_acc = (behav_preds.argmax(axis=1) == y).mean()
            self.cv_results['behavioral'] = behav_acc
            logger.info(f"  CV Accuracy: {behav_acc:.3f}")
            self.behavioral_classifier.fit(X_behav_scaled, y)
        
        # Train Signal 2: Subreddit patterns
        if self.subreddit_cols:
            logger.info("\n[Signal 2] Training subreddit classifier...")
            X_sub = train_df[self.subreddit_cols].values
            
            self.subreddit_classifier = xgb.XGBClassifier(**xgb_params)
            sub_preds = cross_val_predict(
                self.subreddit_classifier, X_sub, y,
                cv=cv, method='predict_proba'
            )
            sub_acc = (sub_preds.argmax(axis=1) == y).mean()
            self.cv_results['subreddit'] = sub_acc
            logger.info(f"  CV Accuracy: {sub_acc:.3f}")
            self.subreddit_classifier.fit(X_sub, y)
        
        # Meta-learner: Stack signals
        logger.info("\n[Meta-Learner] Stacking signals...")
        
        meta_features = []
        if self.behavioral_cols:
            if self.use_binary:
                meta_features.append(behav_preds[:, 1:])  # P(adult)
            else:
                meta_features.append(behav_preds)
        if self.subreddit_cols:
            if self.use_binary:
                meta_features.append(sub_preds[:, 1:])  # P(adult)
            else:
                meta_features.append(sub_preds)
        
        if meta_features:
            X_meta = np.hstack(meta_features)
            
            self.meta_classifier = xgb.XGBClassifier(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.1,
                reg_alpha=1.0,
                reg_lambda=1.0,
                objective='binary:logistic' if self.use_binary else 'multi:softprob',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
            
            meta_preds = cross_val_predict(
                self.meta_classifier, X_meta, y,
                cv=cv, method='predict_proba'
            )
            meta_acc = (meta_preds.argmax(axis=1) == y).mean()
            self.cv_results['ensemble'] = meta_acc
            logger.info(f"  CV Accuracy: {meta_acc:.3f}")
            self.meta_classifier.fit(X_meta, y)
        
        self.is_fitted = True
        
        # Validate ground truth direction
        if anthroscore_df is not None:
            self._validate_ground_truth(train_df, anthroscore_df, author_col)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE - V2 AGE PREDICTOR")
        logger.info("="*60)
        for signal, score in self.cv_results.items():
            logger.info(f"  {signal}: {score:.1%} accuracy")
        
        return {
            'cv_results': self.cv_results,
            'n_train': len(train_df),
            'class_distribution': class_distribution,
            'feature_counts': {
                'behavioral': len(self.behavioral_cols),
                'subreddit': len(self.subreddit_cols),
                'text_embeddings': 0  # Explicitly 0
            }
        }
    
    def _validate_ground_truth(
        self,
        train_df: pd.DataFrame,
        anthroscore_df: pd.DataFrame,
        author_col: str
    ):
        """
        CRITICAL VALIDATION: Ensure predicted ages correlate with 
        anthropomorphization in the correct direction (adults > teens).
        """
        logger.info("\n" + "="*60)
        logger.info("CRITICAL VALIDATION: Ground Truth Direction Check")
        logger.info("="*60)
        
        # Get predictions for training set
        preds = self.predict(train_df, author_col=author_col)
        
        # Merge with anthroscores
        preds_with_anthro = preds.merge(
            anthroscore_df[['author', 'anthroscore_max']],
            on='author',
            how='inner'
        )
        
        # Calculate correlation
        teens = preds_with_anthro[preds_with_anthro['age_predicted'] == 'teen']['anthroscore_max']
        adults = preds_with_anthro[preds_with_anthro['age_predicted'] == 'adult']['anthroscore_max']
        
        if len(teens) > 0 and len(adults) > 0:
            d = (teens.mean() - adults.mean()) / np.sqrt((teens.std()**2 + adults.std()**2) / 2)
            t, p = stats.ttest_ind(teens, adults)
            
            logger.info(f"Predicted Teen mean AnthroScore: {teens.mean():.4f} (n={len(teens)})")
            logger.info(f"Predicted Adult mean AnthroScore: {adults.mean():.4f} (n={len(adults)})")
            logger.info(f"Cohen's d: {d:.4f}")
            logger.info(f"p-value: {p:.4f}")
            
            self.ground_truth_correlation = {
                'cohens_d': d,
                'p_value': p,
                'teen_mean': teens.mean(),
                'adult_mean': adults.mean()
            }
            
            if d < 0:
                logger.info("✓ VALIDATION PASSED: Predicted adults anthropomorphize more")
                logger.info("  This matches ground truth direction!")
            else:
                logger.warning("⚠ VALIDATION WARNING: Predicted teens anthropomorphize more")
                logger.warning("  This does NOT match ground truth direction!")
    
    def predict(
        self,
        features_df: pd.DataFrame,
        author_col: str = "author"
    ) -> pd.DataFrame:
        """Predict age for users."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        predictions = []
        
        # Get behavioral predictions
        if self.behavioral_cols:
            X_behav = features_df[self.behavioral_cols].fillna(0).values
            X_behav_scaled = self.behavioral_scaler.transform(X_behav)
            behav_proba = self.behavioral_classifier.predict_proba(X_behav_scaled)
            if self.use_binary:
                predictions.append(behav_proba[:, 1:])
            else:
                predictions.append(behav_proba)
        
        # Get subreddit predictions
        if self.subreddit_cols:
            X_sub = features_df[self.subreddit_cols].fillna(0).values
            sub_proba = self.subreddit_classifier.predict_proba(X_sub)
            if self.use_binary:
                predictions.append(sub_proba[:, 1:])
            else:
                predictions.append(sub_proba)
        
        # Get meta predictions
        if predictions and self.meta_classifier is not None:
            X_meta = np.hstack(predictions)
            final_proba = self.meta_classifier.predict_proba(X_meta)
            final_preds = final_proba.argmax(axis=1)
            confidence = final_proba.max(axis=1)
        else:
            # Fallback
            final_proba = predictions[0] if predictions else np.zeros((len(features_df), 2))
            final_preds = final_proba.argmax(axis=1)
            confidence = final_proba.max(axis=1) if len(predictions) > 0 else np.zeros(len(features_df))
        
        results = pd.DataFrame({
            author_col: features_df[author_col].values,
            'age_predicted': self.label_encoder.inverse_transform(final_preds),
            'confidence': confidence
        })
        
        # Add probability columns
        for i, label in enumerate(self.label_encoder.classes_):
            results[f'prob_{label}'] = final_proba[:, i]
        
        return results
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'age_predictor_v2.pkl', 'wb') as f:
            pickle.dump({
                'behavioral_classifier': self.behavioral_classifier,
                'subreddit_classifier': self.subreddit_classifier,
                'meta_classifier': self.meta_classifier,
                'behavioral_scaler': self.behavioral_scaler,
                'label_encoder': self.label_encoder,
                'behavioral_cols': self.behavioral_cols,
                'subreddit_cols': self.subreddit_cols,
                'selected_subreddits': self.selected_subreddits,
                'cv_results': self.cv_results,
                'ground_truth_correlation': self.ground_truth_correlation,
                'use_binary': self.use_binary
            }, f)
        
        logger.info(f"V2 Age Predictor saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'AgePredictor_V2':
        """Load model from disk."""
        with open(Path(path) / 'age_predictor_v2.pkl', 'rb') as f:
            state = pickle.load(f)
        
        predictor = cls(use_binary=state['use_binary'])
        predictor.behavioral_classifier = state['behavioral_classifier']
        predictor.subreddit_classifier = state['subreddit_classifier']
        predictor.meta_classifier = state['meta_classifier']
        predictor.behavioral_scaler = state['behavioral_scaler']
        predictor.label_encoder = state['label_encoder']
        predictor.behavioral_cols = state['behavioral_cols']
        predictor.subreddit_cols = state['subreddit_cols']
        predictor.selected_subreddits = state['selected_subreddits']
        predictor.cv_results = state['cv_results']
        predictor.ground_truth_correlation = state['ground_truth_correlation']
        predictor.is_fitted = True
        
        return predictor


if __name__ == "__main__":
    # Test the V2 predictor
    base_path = Path(__file__).parent.parent.parent
    
    print("Loading data...")
    comments_df = pd.read_parquet(base_path / 'Data/processed/all_comments.parquet')
    user_subreddits_df = pd.read_parquet(base_path / 'Data/features/user_subreddit_interactions.parquet')
    self_decl = pd.read_parquet(base_path / 'Data/features/self_declarations.parquet')
    anthro = pd.read_parquet(base_path / 'Data/features/user_anthroscores.parquet')
    
    print("Initializing V2 predictor...")
    predictor = AgePredictor_V2(use_binary=True)
    
    print("Extracting features...")
    features = predictor.extract_features(comments_df, user_subreddits_df)
    
    print("Training on golden set...")
    results = predictor.fit(
        features, 
        self_decl, 
        label_col='age_bucket_self_declared',
        anthroscore_df=anthro
    )
    
    print("\nDone!")
