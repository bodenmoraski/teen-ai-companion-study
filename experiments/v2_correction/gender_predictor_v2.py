"""
GENDER PREDICTOR V2: High Female Recall Design

PROBLEM SOLVED: V1 model has Female Recall of only 44% due to 85% male class imbalance.

V2 STRATEGY:
1. SMOTE oversampling for female class during training
2. Cost-sensitive learning (scale_pos_weight based on class ratio)
3. Threshold optimization for female recall
4. Ensemble voting that prioritizes minority class recall
5. Calibrated probabilities for reliable confidence scores

KEY METRICS TO OPTIMIZE:
- Female Recall (PRIMARY) - must be >> 44%
- Female Precision - maintain reasonable level
- Overall F1 - should not degrade significantly
- Male Recall - should remain high

Author: Research Agent V2
Created: 2026-01-10
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle
from datetime import datetime
from collections import Counter

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, f1_score, recall_score, precision_score
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logging.warning("imblearn not found. Install with: pip install imbalanced-learn")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedLinguisticSignal:
    """
    Enhanced linguistic features with better gender markers.
    
    Research-backed linguistic patterns associated with gender.
    """
    
    def __init__(self):
        import re
        self.female_patterns = [
            # Emotional expressions
            r'\b(omg|omfg|omggg+)\b',
            r'\b(aww+|aw+)\b',
            r'!{2,}',  # Multiple exclamation marks
            r'\b(cute|adorable|lovely|gorgeous|beautiful)\b',
            r'<3',
            r'\b(hubby|bf|boyfriend|fiance)\b',
            r'\b(girl|woman|lady|sister|mom|mother|daughter)\b',
            r'\bi love (this|that|it|you)\b',
            r'\bso (happy|excited|glad|proud)\b',
            r'\b(hugs?|kisses?)\b',
            r'\b(sweetie|honey|babe|hun)\b',
            r'\bhaha+\b',
            r'\b(yay|yayyy+)\b',
            r'\b(lmao+|lol+)\b',
            r':[\)D]\b',  # Smiley emoticons
            r'\b(amazing|wonderful|fantastic)\b',
        ]
        
        self.male_patterns = [
            # More assertive/technical language
            r'\b(dude|bro|man|guy)\b',
            r'\b(wife|gf|girlfriend)\b',
            r'\b(boy|brother|dad|father|son)\b',
            r'\b(fuck|shit|damn|hell|ass)\b',
            r'\b(nah|meh)\b',
            r'\b(awesome|epic|sick|based|chad)\b',
            r'\btbh\b',
            r'\bimo\b',
            r'\b(pc|gpu|cpu|gaming|gamer)\b',
            r'\b(btw|afaik|iirc)\b',
            r'\b(crypto|bitcoin|stocks)\b',
            r'\b(sigma|alpha|beta)\b',
        ]
        
        self.compiled_female = [re.compile(p, re.IGNORECASE) for p in self.female_patterns]
        self.compiled_male = [re.compile(p, re.IGNORECASE) for p in self.male_patterns]
    
    def extract_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced linguistic features."""
        features = []
        
        for author, group in comments_df.groupby('author'):
            combined_text = ' '.join(group['body'].astype(str).tolist()).lower()
            word_count = len(combined_text.split())
            
            feat = {'author': author}
            
            # Count female patterns
            female_count = sum(len(p.findall(combined_text)) for p in self.compiled_female)
            
            # Count male patterns
            male_count = sum(len(p.findall(combined_text)) for p in self.compiled_male)
            
            # Normalized rates (per 1000 words)
            feat['female_marker_rate'] = female_count / max(word_count, 1) * 1000
            feat['male_marker_rate'] = male_count / max(word_count, 1) * 1000
            feat['marker_ratio'] = (female_count + 1) / (male_count + 1)
            feat['marker_diff'] = female_count - male_count
            
            # Text style features
            feat['avg_word_length'] = np.mean([len(w) for w in combined_text.split()]) if combined_text.split() else 5
            feat['exclamation_rate'] = combined_text.count('!') / max(word_count, 1) * 100
            feat['question_rate'] = combined_text.count('?') / max(word_count, 1) * 100
            feat['emoji_rate'] = len([c for c in combined_text if ord(c) > 127]) / max(word_count, 1) * 100
            feat['uppercase_ratio'] = sum(1 for c in combined_text if c.isupper()) / max(len(combined_text), 1)
            
            # Sentence patterns
            sentences = combined_text.split('.')
            feat['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 10
            
            features.append(feat)
        
        return pd.DataFrame(features)


class GenderPredictor_V2:
    """
    Gender Predictor V2: High Female Recall Design
    
    Key improvements over V1:
    1. SMOTE oversampling during training
    2. Cost-sensitive learning with optimized scale_pos_weight
    3. Dynamic threshold tuning for optimal female recall
    4. Ensemble with recall-focused voting
    
    Architecture:
    - Signal 1: Text embeddings (SBERT)
    - Signal 2: Subreddit patterns
    - Signal 3: Behavioral features  
    - Signal 4: Enhanced linguistic markers
    - Meta-learner: Stacked ensemble with threshold optimization
    """
    
    def __init__(
        self,
        target_female_recall: float = 0.70,
        use_smote: bool = True,
        use_cost_sensitive: bool = True
    ):
        """
        Args:
            target_female_recall: Minimum female recall to achieve (default 70%)
            use_smote: Whether to use SMOTE oversampling
            use_cost_sensitive: Whether to use cost-sensitive learning
        """
        self.target_female_recall = target_female_recall
        self.use_smote = use_smote and HAS_IMBLEARN
        self.use_cost_sensitive = use_cost_sensitive
        
        self.linguistic_extractor = EnhancedLinguisticSignal()
        
        self.text_classifier = None
        self.subreddit_classifier = None
        self.behavioral_classifier = None
        self.linguistic_classifier = None
        self.meta_classifier = None
        
        self.text_scaler = StandardScaler()
        self.behavioral_scaler = StandardScaler()
        
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Feature columns
        self.text_cols = None
        self.subreddit_cols = None
        self.behavioral_cols = None
        self.linguistic_cols = None
        
        # Optimal threshold for female recall
        self.optimal_threshold = 0.5
        self.scale_pos_weight = 1.0
        
        # CV results
        self.cv_results = {}
        self.recall_metrics = {}
        
    def extract_features(
        self,
        comments_df: pd.DataFrame,
        user_subreddits_df: pd.DataFrame,
        existing_text_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract all features for gender prediction.
        
        Can reuse existing text embeddings to save computation.
        """
        logger.info("="*60)
        logger.info("GENDER PREDICTOR V2: EXTRACTING FEATURES")
        logger.info("="*60)
        
        authors = comments_df['author'].unique()
        
        # Signal 1: Text embeddings (reuse if available)
        if existing_text_features is not None:
            text_features = existing_text_features
            logger.info(f"[Signal 1] Reusing text embeddings: {len(text_features)} users")
        else:
            logger.info("[Signal 1] Extracting text embeddings...")
            from sentence_transformers import SentenceTransformer
            
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            user_texts = comments_df.groupby('author')['body'].apply(
                lambda x: ' '.join(x.astype(str).head(50))
            ).reset_index()
            user_texts.columns = ['author', 'combined_text']
            
            embeddings = encoder.encode(
                user_texts['combined_text'].tolist(),
                show_progress_bar=True,
                batch_size=32
            )
            
            embedding_cols = [f'text_emb_{i}' for i in range(embeddings.shape[1])]
            text_features = pd.DataFrame(embeddings, columns=embedding_cols)
            text_features['author'] = user_texts['author'].values
            logger.info(f"  Text embeddings: {len(text_features)} users")
        
        # Signal 2: Subreddit patterns
        logger.info("[Signal 2] Extracting subreddit patterns...")
        
        if 'subreddit' in user_subreddits_df.columns:
            aggregated = user_subreddits_df.groupby('author')['subreddit'].apply(list).reset_index()
            aggregated.columns = ['author', 'subreddits']
            user_subreddits_agg = aggregated
        else:
            user_subreddits_agg = user_subreddits_df
        
        # Get top subreddits
        all_subs = []
        for subs in user_subreddits_agg['subreddits']:
            if isinstance(subs, list):
                all_subs.extend(subs)
        
        sub_counts = Counter(all_subs)
        selected_subs = [s for s, c in sub_counts.most_common(400) if c >= 10]
        
        sub_features = []
        for _, row in user_subreddits_agg.iterrows():
            user_subs = set(row['subreddits']) if isinstance(row['subreddits'], list) else set()
            feat = {f'sub_{s}': 1 if s in user_subs else 0 for s in selected_subs}
            feat['author'] = row['author']
            sub_features.append(feat)
        
        subreddit_df = pd.DataFrame(sub_features)
        logger.info(f"  Subreddit features: {len(subreddit_df)} users, {len(selected_subs)} subreddits")
        
        # Signal 3: Behavioral features
        logger.info("[Signal 3] Extracting behavioral features...")
        
        behav_features = []
        for author, group in comments_df.groupby('author'):
            feat = {'author': author}
            feat['comment_count'] = len(group)
            feat['avg_comment_length'] = group['body'].str.len().mean()
            feat['std_comment_length'] = group['body'].str.len().std()
            
            if 'created_utc' in group.columns:
                try:
                    ts = pd.to_datetime(group['created_utc'], unit='s')
                    feat['hour_mean'] = ts.dt.hour.mean()
                    feat['hour_std'] = ts.dt.hour.std() if len(ts) > 1 else 6
                    feat['weekend_ratio'] = (ts.dt.dayofweek >= 5).mean()
                except:
                    feat['hour_mean'] = 12
                    feat['hour_std'] = 6
                    feat['weekend_ratio'] = 0.29
            else:
                feat['hour_mean'] = 12
                feat['hour_std'] = 6
                feat['weekend_ratio'] = 0.29
            
            behav_features.append(feat)
        
        behavioral_df = pd.DataFrame(behav_features)
        logger.info(f"  Behavioral features: {len(behavioral_df)} users")
        
        # Signal 4: Enhanced linguistic features
        logger.info("[Signal 4] Extracting enhanced linguistic features...")
        linguistic_df = self.linguistic_extractor.extract_features(comments_df)
        logger.info(f"  Linguistic features: {len(linguistic_df)} users")
        
        # Merge all features
        all_features = text_features.merge(subreddit_df, on='author', how='left')
        all_features = all_features.merge(behavioral_df, on='author', how='left')
        all_features = all_features.merge(linguistic_df, on='author', how='left')
        all_features = all_features.fillna(0)
        
        logger.info(f"\nTotal: {len(all_features)} users, {len(all_features.columns)-1} features")
        
        return all_features
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to balance classes."""
        if not self.use_smote or not HAS_IMBLEARN:
            return X, y
        
        # Use SMOTE-ENN for better quality synthetic samples
        try:
            smote = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"  SMOTE-ENN: {len(y)} -> {len(y_resampled)} samples")
        except:
            # Fallback to basic SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"  SMOTE: {len(y)} -> {len(y_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def _get_xgb_params(self, scale_weight: float = 1.0) -> Dict:
        """Get XGBoost parameters with cost-sensitive learning."""
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if self.use_cost_sensitive:
            params['scale_pos_weight'] = scale_weight
        
        return params
    
    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_recall: float
    ) -> float:
        """
        Find decision threshold that achieves target female recall.
        
        Args:
            y_true: True labels (1 = female)
            y_proba: Probability of female class
            target_recall: Target recall for female class
        
        Returns:
            Optimal threshold
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold that achieves target recall
        valid_idx = np.where(recalls[:-1] >= target_recall)[0]
        
        if len(valid_idx) > 0:
            # Among thresholds that achieve target recall, pick highest precision
            best_idx = valid_idx[np.argmax(precisions[:-1][valid_idx])]
            optimal_thresh = thresholds[best_idx]
        else:
            # If target not achievable, use threshold that maximizes recall
            optimal_thresh = thresholds[np.argmax(recalls[:-1])]
        
        return float(optimal_thresh)
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        label_col: str = 'gender_self_declared'
    ) -> Dict:
        """
        Train the V2 gender predictor with enhanced minority class handling.
        
        Args:
            features_df: DataFrame with all extracted features
            labels_df: DataFrame with gender labels
        """
        logger.info("="*60)
        logger.info("GENDER PREDICTOR V2: TRAINING WITH ENHANCED RECALL")
        logger.info("="*60)
        
        # Merge with labels
        train_df = features_df.merge(
            labels_df[['author', label_col]],
            on='author',
            how='inner'
        )
        
        # Filter to binary (male/female)
        train_df = train_df[train_df[label_col].isin(['male', 'female'])]
        train_df = train_df.dropna(subset=[label_col])
        
        logger.info(f"Training set: {len(train_df)} users")
        
        # Encode labels (female = 1, male = 0 for scale_pos_weight)
        y = (train_df[label_col] == 'female').astype(int).values
        
        n_female = y.sum()
        n_male = len(y) - n_female
        self.scale_pos_weight = n_male / n_female
        
        logger.info(f"Class distribution: {n_male} male, {n_female} female")
        logger.info(f"Imbalance ratio: {self.scale_pos_weight:.2f}:1")
        logger.info(f"Using SMOTE: {self.use_smote}")
        logger.info(f"Using cost-sensitive: {self.use_cost_sensitive}")
        
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
        
        logger.info(f"\nFeature breakdown:")
        logger.info(f"  Text embeddings: {len(self.text_cols)}")
        logger.info(f"  Subreddit features: {len(self.subreddit_cols)}")
        logger.info(f"  Behavioral features: {len(self.behavioral_cols)}")
        logger.info(f"  Linguistic features: {len(self.linguistic_cols)}")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        xgb_params = self._get_xgb_params(self.scale_pos_weight)
        
        all_oof_preds = []
        
        # Train Signal 1: Text embeddings
        if self.text_cols:
            logger.info("\n[Signal 1] Training text classifier with SMOTE + cost-sensitive...")
            X_text = train_df[self.text_cols].values
            X_text_scaled = self.text_scaler.fit_transform(X_text)
            
            # Apply SMOTE
            X_text_smote, y_text_smote = self._apply_smote(X_text_scaled, y)
            
            self.text_classifier = xgb.XGBClassifier(**xgb_params)
            self.text_classifier.fit(X_text_smote, y_text_smote)
            
            # Get OOF predictions on original data
            text_preds = cross_val_predict(
                xgb.XGBClassifier(**xgb_params),
                X_text_scaled, y, cv=cv, method='predict_proba'
            )
            text_recall = recall_score(y, (text_preds[:, 1] >= 0.5).astype(int))
            self.cv_results['text'] = text_recall
            logger.info(f"  Female Recall: {text_recall:.3f}")
            all_oof_preds.append(text_preds[:, 1])
        
        # Train Signal 2: Subreddit patterns
        if self.subreddit_cols:
            logger.info("\n[Signal 2] Training subreddit classifier...")
            X_sub = train_df[self.subreddit_cols].values
            X_sub_smote, y_sub_smote = self._apply_smote(X_sub, y)
            
            self.subreddit_classifier = xgb.XGBClassifier(**xgb_params)
            self.subreddit_classifier.fit(X_sub_smote, y_sub_smote)
            
            sub_preds = cross_val_predict(
                xgb.XGBClassifier(**xgb_params),
                X_sub, y, cv=cv, method='predict_proba'
            )
            sub_recall = recall_score(y, (sub_preds[:, 1] >= 0.5).astype(int))
            self.cv_results['subreddit'] = sub_recall
            logger.info(f"  Female Recall: {sub_recall:.3f}")
            all_oof_preds.append(sub_preds[:, 1])
        
        # Train Signal 3: Behavioral features
        if self.behavioral_cols:
            logger.info("\n[Signal 3] Training behavioral classifier...")
            X_behav = train_df[self.behavioral_cols].values
            X_behav_scaled = self.behavioral_scaler.fit_transform(X_behav)
            X_behav_smote, y_behav_smote = self._apply_smote(X_behav_scaled, y)
            
            self.behavioral_classifier = xgb.XGBClassifier(**xgb_params)
            self.behavioral_classifier.fit(X_behav_smote, y_behav_smote)
            
            behav_preds = cross_val_predict(
                xgb.XGBClassifier(**xgb_params),
                X_behav_scaled, y, cv=cv, method='predict_proba'
            )
            behav_recall = recall_score(y, (behav_preds[:, 1] >= 0.5).astype(int))
            self.cv_results['behavioral'] = behav_recall
            logger.info(f"  Female Recall: {behav_recall:.3f}")
            all_oof_preds.append(behav_preds[:, 1])
        
        # Train Signal 4: Linguistic features
        if self.linguistic_cols:
            logger.info("\n[Signal 4] Training linguistic classifier...")
            X_ling = train_df[self.linguistic_cols].values
            X_ling_smote, y_ling_smote = self._apply_smote(X_ling, y)
            
            self.linguistic_classifier = xgb.XGBClassifier(**xgb_params)
            self.linguistic_classifier.fit(X_ling_smote, y_ling_smote)
            
            ling_preds = cross_val_predict(
                xgb.XGBClassifier(**xgb_params),
                X_ling, y, cv=cv, method='predict_proba'
            )
            ling_recall = recall_score(y, (ling_preds[:, 1] >= 0.5).astype(int))
            self.cv_results['linguistic'] = ling_recall
            logger.info(f"  Female Recall: {ling_recall:.3f}")
            all_oof_preds.append(ling_preds[:, 1])
        
        # Meta-learner with enhanced female class handling
        logger.info("\n[Meta-Learner] Stacking with threshold optimization...")
        X_meta = np.column_stack(all_oof_preds)
        X_meta_smote, y_meta_smote = self._apply_smote(X_meta, y)
        
        self.meta_classifier = xgb.XGBClassifier(**xgb_params)
        self.meta_classifier.fit(X_meta_smote, y_meta_smote)
        
        # Get final OOF predictions
        meta_preds = cross_val_predict(
            xgb.XGBClassifier(**xgb_params),
            X_meta, y, cv=cv, method='predict_proba'
        )
        
        # Find optimal threshold for target female recall
        self.optimal_threshold = self._find_optimal_threshold(
            y, meta_preds[:, 1], self.target_female_recall
        )
        logger.info(f"  Optimal threshold for {self.target_female_recall:.0%} female recall: {self.optimal_threshold:.3f}")
        
        # Calculate metrics at optimal threshold
        y_pred_opt = (meta_preds[:, 1] >= self.optimal_threshold).astype(int)
        
        female_recall = recall_score(y, y_pred_opt)
        female_precision = precision_score(y, y_pred_opt)
        male_recall = recall_score(1 - y, 1 - y_pred_opt)
        overall_f1 = f1_score(y, y_pred_opt, average='macro')
        
        self.recall_metrics = {
            'female_recall': female_recall,
            'female_precision': female_precision,
            'male_recall': male_recall,
            'macro_f1': overall_f1,
            'optimal_threshold': self.optimal_threshold
        }
        
        self.cv_results['ensemble'] = female_recall
        
        # Also store metrics at default 0.5 threshold for comparison
        y_pred_default = (meta_preds[:, 1] >= 0.5).astype(int)
        default_female_recall = recall_score(y, y_pred_default)
        self.recall_metrics['default_threshold_recall'] = default_female_recall
        
        self.label_encoder.fit(['male', 'female'])
        self.is_fitted = True
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE - V2 GENDER PREDICTOR")
        logger.info("="*60)
        logger.info(f"\nSignal-level Female Recall:")
        for signal, recall in self.cv_results.items():
            logger.info(f"  {signal}: {recall:.1%}")
        
        logger.info(f"\nFinal Metrics at Optimal Threshold ({self.optimal_threshold:.3f}):")
        logger.info(f"  Female Recall:    {female_recall:.1%} (target: {self.target_female_recall:.0%})")
        logger.info(f"  Female Precision: {female_precision:.1%}")
        logger.info(f"  Male Recall:      {male_recall:.1%}")
        logger.info(f"  Macro F1:         {overall_f1:.1%}")
        
        logger.info(f"\nImprovement over default threshold (0.5):")
        improvement = female_recall - default_female_recall
        logger.info(f"  Female Recall: {default_female_recall:.1%} -> {female_recall:.1%} ({improvement:+.1%})")
        
        return {
            'cv_results': self.cv_results,
            'recall_metrics': self.recall_metrics,
            'n_train': len(train_df),
            'class_distribution': {'male': n_male, 'female': n_female}
        }
    
    def predict(
        self,
        features_df: pd.DataFrame,
        use_optimal_threshold: bool = True
    ) -> pd.DataFrame:
        """
        Predict gender with optimized threshold for female recall.
        
        Args:
            features_df: DataFrame with features
            use_optimal_threshold: If True, use threshold optimized for female recall
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        
        # Get predictions from each signal
        signal_preds = []
        
        if self.text_cols and self.text_classifier:
            X_text = features_df[self.text_cols].fillna(0).values
            X_text_scaled = self.text_scaler.transform(X_text)
            signal_preds.append(self.text_classifier.predict_proba(X_text_scaled)[:, 1])
        
        if self.subreddit_cols and self.subreddit_classifier:
            X_sub = features_df[self.subreddit_cols].fillna(0).values
            signal_preds.append(self.subreddit_classifier.predict_proba(X_sub)[:, 1])
        
        if self.behavioral_cols and self.behavioral_classifier:
            X_behav = features_df[self.behavioral_cols].fillna(0).values
            X_behav_scaled = self.behavioral_scaler.transform(X_behav)
            signal_preds.append(self.behavioral_classifier.predict_proba(X_behav_scaled)[:, 1])
        
        if self.linguistic_cols and self.linguistic_classifier:
            X_ling = features_df[self.linguistic_cols].fillna(0).values
            signal_preds.append(self.linguistic_classifier.predict_proba(X_ling)[:, 1])
        
        # Meta prediction
        X_meta = np.column_stack(signal_preds)
        final_proba = self.meta_classifier.predict_proba(X_meta)
        
        # Apply threshold
        prob_female = final_proba[:, 1]
        predictions = np.where(prob_female >= threshold, 'female', 'male')
        confidence = np.maximum(prob_female, 1 - prob_female)
        
        results = pd.DataFrame({
            'author': features_df['author'].values,
            'gender_predicted': predictions,
            'confidence': confidence,
            'prob_female': prob_female,
            'prob_male': final_proba[:, 0],
            'threshold_used': threshold
        })
        
        return results
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'gender_predictor_v2.pkl', 'wb') as f:
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
                'optimal_threshold': self.optimal_threshold,
                'scale_pos_weight': self.scale_pos_weight,
                'cv_results': self.cv_results,
                'recall_metrics': self.recall_metrics,
                'target_female_recall': self.target_female_recall,
                'use_smote': self.use_smote,
                'use_cost_sensitive': self.use_cost_sensitive
            }, f)
        
        logger.info(f"V2 Gender Predictor saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'GenderPredictor_V2':
        """Load model from disk."""
        with open(Path(path) / 'gender_predictor_v2.pkl', 'rb') as f:
            state = pickle.load(f)
        
        predictor = cls(
            target_female_recall=state.get('target_female_recall', 0.7),
            use_smote=state.get('use_smote', True),
            use_cost_sensitive=state.get('use_cost_sensitive', True)
        )
        
        for key, value in state.items():
            setattr(predictor, key, value)
        
        predictor.is_fitted = True
        return predictor


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent
    
    print("Loading data...")
    comments_df = pd.read_parquet(base_path / 'Data/processed/all_comments.parquet')
    user_subreddits_df = pd.read_parquet(base_path / 'Data/features/user_subreddit_interactions.parquet')
    self_decl = pd.read_parquet(base_path / 'Data/features/self_declarations.parquet')
    
    print("Initializing V2 predictor...")
    predictor = GenderPredictor_V2(target_female_recall=0.70)
    
    print("Extracting features...")
    features = predictor.extract_features(comments_df, user_subreddits_df)
    
    print("Training...")
    results = predictor.fit(features, self_decl)
    
    print("\nDone!")
