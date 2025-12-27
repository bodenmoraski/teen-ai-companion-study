"""
Model Training Module for Reddit Age Classification

Implements gradient boosted trees with hyperparameter tuning using grid search
and 5-fold cross-validation as described in Chew et al. (2021).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from typing import Dict, Tuple, Any
import logging
from pathlib import Path
import joblib
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgeClassifier:
    """
    Gradient Boosted Trees classifier for Reddit user age classification.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the age classifier.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              use_grid_search: bool = True, n_jobs: int = -1) -> GradientBoostingClassifier:
        """
        Train the gradient boosted trees classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_grid_search: Whether to perform hyperparameter tuning
            n_jobs: Number of parallel jobs for grid search (-1 = all cores)
            
        Returns:
            Trained model
        """
        logger.info("Training gradient boosted trees classifier...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        if use_grid_search:
            self.model = self._train_with_grid_search(X_train, y_train, n_jobs)
        else:
            self.model = self._train_default(X_train, y_train)
        
        logger.info("✓ Training complete")
        
        return self.model
    
    def _train_default(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        """
        Train model with default hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        logger.info("Using default hyperparameters...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_seed,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        cv_scores = self._cross_validate(model, X_train, y_train)
        logger.info(f"✓ CV F1 score: {cv_scores['mean']:.3f} ± {cv_scores['std']:.3f}")
        
        return model
    
    def _train_with_grid_search(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                n_jobs: int = -1) -> GradientBoostingClassifier:
        """
        Train model with hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_jobs: Number of parallel jobs
            
        Returns:
            Best model from grid search
        """
        logger.info("Performing hyperparameter tuning with grid search...")
        
        # Define hyperparameter search space
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Setup 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        
        # Initialize base estimator
        base_estimator = GradientBoostingClassifier(random_state=self.random_seed)
        
        # Perform grid search
        logger.info(f"Testing {np.prod([len(v) for v in param_grid.values()])} parameter combinations...")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Grid search completed in {elapsed_time:.1f} seconds")
        
        # Store results
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_
        
        # Log best parameters
        logger.info(f"✓ Best hyperparameters: {self.best_params}")
        logger.info(f"✓ Best CV F1 score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def _cross_validate(self, model: GradientBoostingClassifier, 
                       X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation to estimate model performance.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with mean and std of CV scores
        """
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predictions (0=adolescent, 1=adult)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, output_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'best_params': self.best_params
            }
            
            joblib.dump(model_data, output_path)
            logger.info(f"✓ Saved model to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, input_path: str) -> None:
        """
        Load a previously trained model from disk.
        
        Args:
            input_path: Path to load the model from
        """
        try:
            model_data = joblib.load(input_path)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.best_params = model_data.get('best_params', None)
            
            logger.info(f"✓ Loaded model from {input_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def train_age_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                        use_grid_search: bool = True,
                        output_path: str = "models/trained_model.pkl",
                        random_seed: int = 42) -> AgeClassifier:
    """
    Main function to train and save the age classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        use_grid_search: Whether to perform hyperparameter tuning
        output_path: Path to save the trained model
        random_seed: Random seed for reproducibility
        
    Returns:
        Trained AgeClassifier instance
    """
    # Initialize classifier
    classifier = AgeClassifier(random_seed=random_seed)
    
    # Train model
    classifier.train(X_train, y_train, use_grid_search=use_grid_search)
    
    # Save model
    classifier.save_model(output_path)
    
    # Display feature importance
    importance_df = classifier.get_feature_importance()
    logger.info("\nFeature Importance (Top 10):")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return classifier


if __name__ == "__main__":
    # Test model training
    from feature_engineering import prepare_dataset
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    # Train model
    classifier = train_age_classifier(X_train, y_train, use_grid_search=True)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"\nBest hyperparameters: {classifier.best_params}")
    print(f"\nPredictions on test set: {len(y_pred)} samples")
    print(f"Predicted adolescents: {(y_pred == 0).sum()}")
    print(f"Predicted adults: {(y_pred == 1).sum()}")

