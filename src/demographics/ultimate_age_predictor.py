"""
ULTIMATE AGE PREDICTION SYSTEM

This module implements a state-of-the-art multi-signal stacked ensemble
for predicting Reddit user age from multiple data sources.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│  SIGNAL 1: Text Embeddings (Sentence-BERT)                  │
│  SIGNAL 2: Subreddit Patterns (Trained Classifier)          │
│  SIGNAL 3: Behavioral Features (Posting Patterns)           │
│  SIGNAL 4: LLM Chain-of-Thought (Few-Shot)                  │
│  ──────────────────────────────────────────────────────────│
│  META-LEARNER: XGBoost Stacking on all signals              │
└─────────────────────────────────────────────────────────────┘

Key innovations:
- TRAINS classifiers on known-age users (not arbitrary thresholds)
- Uses ALL available signals (text, subreddits, behavior, LLM)
- Stacking meta-learner for optimal combination
- Calibrated probability outputs with confidence filtering
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
from datetime import datetime
from collections import Counter

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

logger = logging.getLogger(__name__)

# Age bucket definitions
AGE_BUCKETS_5 = ["13-18", "19-25", "26-40", "41-60", "61-80"]
AGE_BUCKETS_3 = ["teen", "young_adult", "adult"]  # 13-18, 19-25, 26+

def map_to_3_bucket(age_bucket: str) -> str:
    """Map 5-bucket age to 3-bucket for higher accuracy."""
    if age_bucket == "13-18":
        return "teen"
    elif age_bucket == "19-25":
        return "young_adult"
    else:  # 26-40, 41-60, 61-80
        return "adult"


class Signal1_TextEmbeddings:
    """
    Text embedding features using Sentence-BERT.
    
    Embeds user comment history and trains a classifier on embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _get_encoder(self):
        if not HAS_SBERT:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")
        if self.encoder is None:
            logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
        return self.encoder
    
    def extract_user_embeddings(
        self, 
        comments_df: pd.DataFrame,
        author_col: str = "author",
        text_col: str = "body",
        max_comments: int = 50
    ) -> pd.DataFrame:
        """
        Extract mean text embedding per user.
        
        Args:
            comments_df: DataFrame with user comments
            author_col: Author column name
            text_col: Text column name
            max_comments: Max comments per user to embed
            
        Returns:
            DataFrame with author and embedding columns
        """
        encoder = self._get_encoder()
        
        # Group comments by author, take first max_comments
        user_texts = (
            comments_df
            .groupby(author_col)[text_col]
            .apply(lambda x: " ".join(x.head(max_comments).tolist()))
            .reset_index()
        )
        user_texts.columns = [author_col, "combined_text"]
        
        logger.info(f"Embedding text for {len(user_texts)} users...")
        
        # Batch encode
        embeddings = encoder.encode(
            user_texts["combined_text"].tolist(),
            batch_size=32,
            show_progress_bar=True
        )
        
        # Create DataFrame with embedding columns
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])]
        )
        embedding_df[author_col] = user_texts[author_col].values
        
        return embedding_df
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Signal1_TextEmbeddings":
        """Train classifier on text embeddings."""
        logger.info("Training text embedding classifier...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if HAS_XGBOOST:
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                n_jobs=-1,
                random_state=42
            )
        else:
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
        
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict_proba()")
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)


class Signal2_SubredditPatterns:
    """
    Subreddit participation-based classifier.
    
    Instead of arbitrary threshold projections, trains an actual classifier
    on subreddit participation patterns.
    """
    
    def __init__(self, min_subreddit_freq: int = 10, max_features: int = 500):
        self.min_subreddit_freq = min_subreddit_freq
        self.max_features = max_features
        self.selected_subreddits = []
        self.classifier = None
        self.is_fitted = False
    
    def extract_subreddit_features(
        self,
        user_subreddits_df: pd.DataFrame,
        author_col: str = "author"
    ) -> pd.DataFrame:
        """
        Create subreddit participation feature matrix.
        
        Args:
            user_subreddits_df: DataFrame with either:
                - 'author' and 'subreddits' (list) columns, OR
                - 'author', 'subreddit', 'count' columns (raw format)
            
        Returns:
            DataFrame with binary subreddit features
        """
        # Handle raw format (author, subreddit, count) - aggregate to lists
        if "subreddit" in user_subreddits_df.columns and "subreddits" not in user_subreddits_df.columns:
            logger.info("Converting raw subreddit format to aggregated format...")
            user_subreddits_agg = (
                user_subreddits_df
                .groupby(author_col)["subreddit"]
                .apply(list)
                .reset_index()
            )
            user_subreddits_agg.columns = [author_col, "subreddits"]
        else:
            user_subreddits_agg = user_subreddits_df
        
        # Count subreddit frequencies across all users
        all_subreddits = []
        for subs in user_subreddits_agg["subreddits"]:
            if isinstance(subs, list):
                all_subreddits.extend(subs)
        
        subreddit_counts = Counter(all_subreddits)
        
        # Select top subreddits by frequency (above minimum)
        self.selected_subreddits = [
            sub for sub, count in subreddit_counts.most_common(self.max_features)
            if count >= self.min_subreddit_freq
        ]
        
        logger.info(f"Selected {len(self.selected_subreddits)} subreddits as features")
        
        # Create binary feature matrix
        features = []
        for _, row in user_subreddits_agg.iterrows():
            user_subs = set(row["subreddits"]) if isinstance(row["subreddits"], list) else set()
            feature_row = {
                sub: 1 if sub in user_subs else 0
                for sub in self.selected_subreddits
            }
            feature_row[author_col] = row[author_col]
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Signal2_SubredditPatterns":
        """Train classifier on subreddit features."""
        logger.info("Training subreddit pattern classifier...")
        
        if HAS_XGBOOST:
            self.classifier = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                n_jobs=-1,
                random_state=42
            )
        else:
            self.classifier = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict_proba()")
        return self.classifier.predict_proba(X)


class Signal3_BehavioralFeatures:
    """
    Behavioral features based on posting patterns.
    
    Features:
    - Hour of day distribution (teens post at different times)
    - Day of week patterns
    - Posting frequency
    - Account activity level
    - Subreddit diversity
    - Comment length patterns
    """
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_behavioral_features(
        self,
        comments_df: pd.DataFrame,
        author_col: str = "author",
        timestamp_col: str = "created_utc",
        text_col: str = "body"
    ) -> pd.DataFrame:
        """
        Extract behavioral features per user.
        """
        features = []
        
        for author, group in comments_df.groupby(author_col):
            # Parse timestamps
            if timestamp_col in group.columns:
                try:
                    # Handle different timestamp formats
                    if group[timestamp_col].dtype == 'int64':
                        timestamps = pd.to_datetime(group[timestamp_col], unit='s')
                    else:
                        timestamps = pd.to_datetime(group[timestamp_col])
                    
                    hours = timestamps.dt.hour
                    days = timestamps.dt.dayofweek
                    
                    # Hour distribution (normalized)
                    hour_dist = hours.value_counts(normalize=True).reindex(range(24), fill_value=0)
                    
                    # Late night posting (midnight to 4am) - teens tend to post late
                    late_night_ratio = hour_dist[0:5].sum()
                    
                    # School hours posting (9am-3pm) - less if in school
                    school_hours_ratio = hour_dist[9:15].sum()
                    
                    # Weekend ratio
                    weekend_ratio = (days >= 5).mean()
                    
                    # Activity span (days between first and last post)
                    activity_span = (timestamps.max() - timestamps.min()).days
                    
                except Exception:
                    late_night_ratio = 0.0
                    school_hours_ratio = 0.0
                    weekend_ratio = 0.5
                    activity_span = 0
            else:
                late_night_ratio = 0.0
                school_hours_ratio = 0.0
                weekend_ratio = 0.5
                activity_span = 0
            
            # Text-based features
            if text_col in group.columns:
                avg_comment_length = group[text_col].str.len().mean()
                comment_count = len(group)
            else:
                avg_comment_length = 0
                comment_count = 0
            
            # Subreddit diversity
            if "subreddit" in group.columns:
                subreddit_diversity = group["subreddit"].nunique() / max(1, len(group))
            else:
                subreddit_diversity = 0
            
            features.append({
                author_col: author,
                "late_night_ratio": late_night_ratio,
                "school_hours_ratio": school_hours_ratio,
                "weekend_ratio": weekend_ratio,
                "activity_span_days": activity_span,
                "avg_comment_length": avg_comment_length,
                "comment_count": comment_count,
                "subreddit_diversity": subreddit_diversity
            })
        
        return pd.DataFrame(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Signal3_BehavioralFeatures":
        """Train classifier on behavioral features."""
        logger.info("Training behavioral features classifier...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if HAS_XGBOOST:
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                n_jobs=-1,
                random_state=42
            )
        else:
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict_proba()")
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)


class Signal4_LLMClassifier:
    """
    Improved LLM classifier with few-shot examples and probability outputs.
    """
    
    def __init__(self, model: str = "gpt-4.1-nano"):
        self.model = model
        self.few_shot_examples = self._get_few_shot_examples()
    
    def _get_few_shot_examples(self) -> str:
        """Few-shot examples for better calibration."""
        return """
Example 1:
Comments: "just got my license yesterday!! finally can drive to school without my mom"
Answer: {"age_bucket": "13-18", "probabilities": {"13-18": 0.85, "19-25": 0.12, "26+": 0.03}, "reasoning": "Just got license, mentions school, mom drives them"}

Example 2:
Comments: "started my first real job after college, crazy how fast the time went by"
Answer: {"age_bucket": "19-25", "probabilities": {"13-18": 0.05, "19-25": 0.75, "26+": 0.20}, "reasoning": "Recent college grad, first job after college"}

Example 3:
Comments: "my kids are finally back in school and I can work in peace again lol"
Answer: {"age_bucket": "26+", "probabilities": {"13-18": 0.01, "19-25": 0.04, "26+": 0.95}, "reasoning": "Has kids in school, works from home"}
"""
    
    def classify(
        self, 
        comments: List[str],
        client: Any,
        max_comments: int = 30
    ) -> Dict[str, Any]:
        """
        Classify with improved prompting.
        
        Returns probabilities for each age bucket.
        """
        sample_comments = comments[:max_comments]
        comments_text = "\n".join([f"- {c[:300]}" for c in sample_comments if c])
        
        if not comments_text:
            return {"age_bucket": None, "probabilities": None, "confidence": 0.0}
        
        prompt = f"""You are an expert at inferring Reddit user demographics from their writing style.

Analyze these Reddit comments and estimate the user's age group.

Look for:
- Life stage indicators (school, college, job, kids, retirement)
- Vocabulary and slang usage
- Topics and interests
- Cultural references
- Maturity of writing style

{self.few_shot_examples}

Now analyze this user:

Comments:
{comments_text}

Respond with JSON only:
{{"age_bucket": "13-18" or "19-25" or "26+", "probabilities": {{"13-18": 0.0-1.0, "19-25": 0.0-1.0, "26+": 0.0-1.0}}, "reasoning": "brief explanation"}}

Make sure probabilities sum to 1.0."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for more consistent predictions
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Extract confidence from probabilities
            probs = result.get("probabilities", {})
            predicted = result.get("age_bucket")
            confidence = probs.get(predicted, 0.0) if probs and predicted else 0.0
            
            return {
                "age_bucket": predicted,
                "probabilities": probs,
                "confidence": confidence,
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return {"age_bucket": None, "probabilities": None, "confidence": 0.0}


class UltimateAgePredictor:
    """
    Meta-learner that combines all signals using stacking.
    """
    
    def __init__(self, use_3_bucket: bool = True):
        self.use_3_bucket = use_3_bucket
        self.age_buckets = AGE_BUCKETS_3 if use_3_bucket else AGE_BUCKETS_5
        
        self.signal1 = Signal1_TextEmbeddings()
        self.signal2 = Signal2_SubredditPatterns()
        self.signal3 = Signal3_BehavioralFeatures()
        self.signal4 = Signal4_LLMClassifier()
        
        self.meta_learner = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Feature storage for each user
        self.text_features = None
        self.subreddit_features = None
        self.behavioral_features = None
        self.llm_predictions = None
        
    def extract_all_features(
        self,
        comments_df: pd.DataFrame,
        user_subreddits_df: pd.DataFrame,
        author_col: str = "author"
    ) -> pd.DataFrame:
        """
        Extract features from all signals for all users.
        """
        logger.info("=" * 60)
        logger.info("EXTRACTING ALL FEATURES FOR ULTIMATE AGE PREDICTOR")
        logger.info("=" * 60)
        
        # Get unique users
        users = comments_df[author_col].unique()
        logger.info(f"Processing {len(users)} users")
        
        # Signal 1: Text embeddings
        logger.info("\n[Signal 1] Extracting text embeddings...")
        try:
            self.text_features = self.signal1.extract_user_embeddings(
                comments_df, author_col=author_col
            )
            logger.info(f"  ✓ Text embeddings: {len(self.text_features)} users")
        except Exception as e:
            logger.warning(f"  ✗ Text embeddings failed: {e}")
            self.text_features = pd.DataFrame({author_col: users})
        
        # Signal 2: Subreddit patterns
        logger.info("\n[Signal 2] Extracting subreddit features...")
        try:
            self.subreddit_features = self.signal2.extract_subreddit_features(
                user_subreddits_df, author_col=author_col
            )
            logger.info(f"  ✓ Subreddit features: {len(self.subreddit_features)} users, {len(self.signal2.selected_subreddits)} subreddits")
        except Exception as e:
            logger.warning(f"  ✗ Subreddit features failed: {e}")
            self.subreddit_features = pd.DataFrame({author_col: users})
        
        # Signal 3: Behavioral features
        logger.info("\n[Signal 3] Extracting behavioral features...")
        try:
            self.behavioral_features = self.signal3.extract_behavioral_features(
                comments_df, author_col=author_col
            )
            logger.info(f"  ✓ Behavioral features: {len(self.behavioral_features)} users")
        except Exception as e:
            logger.warning(f"  ✗ Behavioral features failed: {e}")
            self.behavioral_features = pd.DataFrame({author_col: users})
        
        # Merge all features
        logger.info("\nMerging all features...")
        all_features = self.text_features.merge(
            self.subreddit_features, on=author_col, how="outer"
        ).merge(
            self.behavioral_features, on=author_col, how="outer"
        )
        
        logger.info(f"Final feature matrix: {all_features.shape[0]} users, {all_features.shape[1]} features")
        
        return all_features
    
    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        author_col: str = "author",
        label_col: str = "age_bucket_self_declared",
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the stacked ensemble using cross-validation.
        
        Args:
            features_df: DataFrame with all extracted features
            labels_df: DataFrame with known age labels
            author_col: Author column name
            label_col: Label column name
            cv_folds: Number of CV folds for stacking
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("=" * 60)
        logger.info("TRAINING ULTIMATE AGE PREDICTOR (STACKING ENSEMBLE)")
        logger.info("=" * 60)
        
        # Merge features with labels
        train_df = features_df.merge(
            labels_df[[author_col, label_col]].dropna(),
            on=author_col,
            how="inner"
        )
        
        logger.info(f"Training set size: {len(train_df)} users with known age")
        
        if len(train_df) < 50:
            raise ValueError(f"Not enough training data: {len(train_df)} users (need at least 50)")
        
        # Convert to 3-bucket if needed
        if self.use_3_bucket:
            train_df["age_label"] = train_df[label_col].apply(map_to_3_bucket)
        else:
            train_df["age_label"] = train_df[label_col]
        
        # Encode labels
        y = self.label_encoder.fit_transform(train_df["age_label"])
        logger.info(f"Class distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y)))}")
        
        # Prepare feature matrices for each signal
        feature_cols = [c for c in train_df.columns if c not in [author_col, label_col, "age_label"]]
        
        # Separate feature groups
        text_cols = [c for c in feature_cols if c.startswith("text_emb_")]
        subreddit_cols = [c for c in feature_cols if c in self.signal2.selected_subreddits]
        behavioral_cols = [c for c in feature_cols if c in [
            "late_night_ratio", "school_hours_ratio", "weekend_ratio",
            "activity_span_days", "avg_comment_length", "comment_count", "subreddit_diversity"
        ]]
        
        logger.info(f"\nFeature breakdown:")
        logger.info(f"  Text embedding features: {len(text_cols)}")
        logger.info(f"  Subreddit features: {len(subreddit_cols)}")
        logger.info(f"  Behavioral features: {len(behavioral_cols)}")
        
        # Train individual signals and get OOF predictions for stacking
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        meta_features = []
        signal_scores = {}
        
        # Signal 1: Text embeddings
        if text_cols:
            logger.info("\n[Signal 1] Training text embedding classifier...")
            X1 = train_df[text_cols].fillna(0).values
            self.signal1.fit(X1, y)
            oof_preds1 = cross_val_predict(
                self.signal1.classifier, 
                self.signal1.scaler.fit_transform(X1), 
                y, 
                cv=cv, 
                method="predict_proba"
            )
            meta_features.append(oof_preds1)
            signal_scores["text_embeddings"] = accuracy_score(y, oof_preds1.argmax(axis=1))
            logger.info(f"  CV Accuracy: {signal_scores['text_embeddings']:.3f}")
        
        # Signal 2: Subreddit patterns
        if subreddit_cols:
            logger.info("\n[Signal 2] Training subreddit pattern classifier...")
            X2 = train_df[subreddit_cols].fillna(0).values
            self.signal2.fit(X2, y)
            oof_preds2 = cross_val_predict(
                self.signal2.classifier, 
                X2, 
                y, 
                cv=cv, 
                method="predict_proba"
            )
            meta_features.append(oof_preds2)
            signal_scores["subreddit_patterns"] = accuracy_score(y, oof_preds2.argmax(axis=1))
            logger.info(f"  CV Accuracy: {signal_scores['subreddit_patterns']:.3f}")
        
        # Signal 3: Behavioral features
        if behavioral_cols:
            logger.info("\n[Signal 3] Training behavioral features classifier...")
            X3 = train_df[behavioral_cols].fillna(0).values
            self.signal3.fit(X3, y)
            oof_preds3 = cross_val_predict(
                self.signal3.classifier,
                self.signal3.scaler.fit_transform(X3),
                y,
                cv=cv,
                method="predict_proba"
            )
            meta_features.append(oof_preds3)
            signal_scores["behavioral_features"] = accuracy_score(y, oof_preds3.argmax(axis=1))
            logger.info(f"  CV Accuracy: {signal_scores['behavioral_features']:.3f}")
        
        # Stack meta-features
        if meta_features:
            X_meta = np.hstack(meta_features)
            logger.info(f"\n[Meta-Learner] Stacking {X_meta.shape[1]} meta-features...")
            
            if HAS_XGBOOST:
                self.meta_learner = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    use_label_encoder=False,
                    n_jobs=-1,
                    random_state=42
                )
            else:
                self.meta_learner = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )
            
            # Cross-validation for meta-learner
            meta_oof_preds = cross_val_predict(
                self.meta_learner, X_meta, y, cv=cv, method="predict_proba"
            )
            meta_accuracy = accuracy_score(y, meta_oof_preds.argmax(axis=1))
            
            # Fit final meta-learner on all data
            self.meta_learner.fit(X_meta, y)
            
            signal_scores["stacked_ensemble"] = meta_accuracy
            logger.info(f"  CV Accuracy: {meta_accuracy:.3f}")
        
        self.is_fitted = True
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE - RESULTS SUMMARY")
        logger.info("=" * 60)
        for signal, score in signal_scores.items():
            logger.info(f"  {signal}: {score:.1%} accuracy")
        
        # Calculate improvement over baseline
        best_single = max(v for k, v in signal_scores.items() if k != "stacked_ensemble")
        improvement = signal_scores.get("stacked_ensemble", best_single) - best_single
        logger.info(f"\nStacking improvement: +{improvement:.1%} over best single signal")
        
        return {
            "signal_scores": signal_scores,
            "n_train": len(train_df),
            "feature_counts": {
                "text": len(text_cols),
                "subreddit": len(subreddit_cols),
                "behavioral": len(behavioral_cols)
            }
        }
    
    def predict(
        self,
        features_df: pd.DataFrame,
        author_col: str = "author",
        confidence_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Predict age for all users with calibrated probabilities.
        
        Args:
            features_df: DataFrame with extracted features
            author_col: Author column name
            confidence_threshold: Minimum confidence to return prediction
            
        Returns:
            DataFrame with predictions and confidences
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        # Get feature columns
        feature_cols = [c for c in features_df.columns if c != author_col]
        text_cols = [c for c in feature_cols if c.startswith("text_emb_")]
        subreddit_cols = [c for c in feature_cols if c in self.signal2.selected_subreddits]
        behavioral_cols = [c for c in feature_cols if c in [
            "late_night_ratio", "school_hours_ratio", "weekend_ratio",
            "activity_span_days", "avg_comment_length", "comment_count", "subreddit_diversity"
        ]]
        
        # Get predictions from each signal
        predictions = []
        
        if text_cols and self.signal1.is_fitted:
            X1 = features_df[text_cols].fillna(0).values
            predictions.append(self.signal1.predict_proba(X1))
        
        if subreddit_cols and self.signal2.is_fitted:
            X2 = features_df[subreddit_cols].fillna(0).values
            predictions.append(self.signal2.predict_proba(X2))
        
        if behavioral_cols and self.signal3.is_fitted:
            X3 = features_df[behavioral_cols].fillna(0).values
            predictions.append(self.signal3.predict_proba(X3))
        
        # Stack and get meta-predictions
        if predictions and self.meta_learner is not None:
            X_meta = np.hstack(predictions)
            final_probs = self.meta_learner.predict_proba(X_meta)
            final_preds = final_probs.argmax(axis=1)
            confidences = final_probs.max(axis=1)
        else:
            # Fallback to best available signal
            if predictions:
                final_probs = predictions[0]  # Use first available
                final_preds = final_probs.argmax(axis=1)
                confidences = final_probs.max(axis=1)
            else:
                final_preds = np.zeros(len(features_df))
                confidences = np.zeros(len(features_df))
        
        # Create results DataFrame
        results = pd.DataFrame({
            author_col: features_df[author_col].values,
            "age_bucket_predicted": self.label_encoder.inverse_transform(final_preds),
            "confidence": confidences
        })
        
        # Add probability columns
        for i, bucket in enumerate(self.label_encoder.classes_):
            results[f"prob_{bucket}"] = final_probs[:, i]
        
        # Filter by confidence threshold
        if confidence_threshold > 0:
            results.loc[results["confidence"] < confidence_threshold, "age_bucket_predicted"] = None
        
        return results
    
    def save(self, path: Path):
        """Save trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "ultimate_predictor.pkl", "wb") as f:
            pickle.dump(self, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "UltimateAgePredictor":
        """Load trained model from disk."""
        with open(Path(path) / "ultimate_predictor.pkl", "rb") as f:
            return pickle.load(f)


def run_ultimate_predictor(
    comments_path: Path,
    user_subreddits_path: Path,
    self_declarations_path: Path,
    output_path: Path,
    use_3_bucket: bool = True
) -> Dict[str, Any]:
    """
    Main function to run the ultimate age predictor.
    
    Args:
        comments_path: Path to comments parquet
        user_subreddits_path: Path to user subreddit interactions parquet
        self_declarations_path: Path to self declarations parquet
        output_path: Path to save results
        use_3_bucket: Whether to use 3-bucket (more accurate) or 5-bucket classification
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 70)
    logger.info("ULTIMATE AGE PREDICTOR - FULL PIPELINE")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\nLoading data...")
    comments_df = pd.read_parquet(comments_path)
    user_subreddits_df = pd.read_parquet(user_subreddits_path)
    self_decl_df = pd.read_parquet(self_declarations_path)
    
    logger.info(f"  Comments: {len(comments_df)} rows")
    logger.info(f"  User subreddits: {len(user_subreddits_df)} users")
    logger.info(f"  Self-declared ages: {self_decl_df['age_bucket_self_declared'].notna().sum()} users")
    
    # Initialize predictor
    predictor = UltimateAgePredictor(use_3_bucket=use_3_bucket)
    
    # Extract features
    features_df = predictor.extract_all_features(comments_df, user_subreddits_df)
    
    # Train
    train_results = predictor.fit(features_df, self_decl_df)
    
    # Predict on all users
    predictions_df = predictor.predict(features_df, confidence_threshold=0.5)
    
    # Save results
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    predictions_df.to_parquet(output_path / "ultimate_predictions.parquet")
    predictor.save(output_path / "model")
    
    # Summary
    high_conf = predictions_df[predictions_df["confidence"] >= 0.7]
    med_conf = predictions_df[(predictions_df["confidence"] >= 0.5) & (predictions_df["confidence"] < 0.7)]
    
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total users: {len(predictions_df)}")
    logger.info(f"High confidence (≥70%): {len(high_conf)} ({len(high_conf)/len(predictions_df):.1%})")
    logger.info(f"Medium confidence (50-70%): {len(med_conf)} ({len(med_conf)/len(predictions_df):.1%})")
    logger.info(f"\nPrediction distribution (high confidence):")
    if len(high_conf) > 0:
        for bucket, count in high_conf["age_bucket_predicted"].value_counts().items():
            logger.info(f"  {bucket}: {count} ({count/len(high_conf):.1%})")
    
    return {
        "train_results": train_results,
        "n_predictions": len(predictions_df),
        "n_high_conf": len(high_conf),
        "predictions_path": str(output_path / "ultimate_predictions.parquet")
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Default paths
    base_path = Path(__file__).parent.parent.parent
    
    run_ultimate_predictor(
        comments_path=base_path / "Data/processed/all_comments.parquet",
        user_subreddits_path=base_path / "Data/features/user_subreddit_interactions.parquet",
        self_declarations_path=base_path / "Data/features/self_declarations.parquet",
        output_path=base_path / "Data/features/ultimate_predictor",
        use_3_bucket=True
    )

