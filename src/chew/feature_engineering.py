"""
Feature Engineering Module for Reddit Age Classification

Handles feature transformations including quantile transformation
and train/test splitting with stratification.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging
from pathlib import Path
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature preprocessing and transformation for the age classification model.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the feature engineer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.quantile_transformer = None
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels from raw data.
        
        Args:
            df: Raw dataframe with features and age_group column
            
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is labels Series
        """
        # Separate features and labels
        X = df.drop('age_group', axis=1)
        y = df['age_group']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Convert labels to binary (0=adolescent, 1=adult)
        y_binary = (y == 'adult').astype(int)
        
        logger.info(f"✓ Prepared {len(X)} samples with {len(X.columns)} features")
        
        return X, y_binary
    
    def apply_quantile_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply quantile transformation to make features approximately normal.
        
        The paper mentions that features were quantile-transformed, which is
        a standard preprocessing technique that maps features to a normal distribution.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of (X_train_transformed, X_test_transformed) or just X_train_transformed
        """
        logger.info("Applying quantile transformation...")
        
        # Initialize transformer
        self.quantile_transformer = QuantileTransformer(
            output_distribution='normal',
            random_state=self.random_seed,
            n_quantiles=min(1000, len(X_train))  # Adjust based on sample size
        )
        
        # Fit and transform training data
        X_train_transformed = self.quantile_transformer.fit_transform(X_train)
        X_train_transformed = pd.DataFrame(
            X_train_transformed,
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        if X_test is not None:
            X_test_transformed = self.quantile_transformer.transform(X_test)
            X_test_transformed = pd.DataFrame(
                X_test_transformed,
                columns=X_test.columns,
                index=X_test.index
            )
            logger.info("✓ Applied quantile transformation to train and test sets")
            return X_train_transformed, X_test_transformed
        
        logger.info("✓ Applied quantile transformation to training set")
        return X_train_transformed, None
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets with stratification.
        
        Args:
            X: Features DataFrame
            y: Labels Series
            test_size: Proportion of data for test set (default 0.2 = 80/20 split)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with {test_size*100:.0f}% test size...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_seed
        )
        
        # Log split statistics
        train_adolescent = (y_train == 0).sum()
        train_adult = (y_train == 1).sum()
        test_adolescent = (y_test == 0).sum()
        test_adult = (y_test == 1).sum()
        
        logger.info(f"✓ Train: {len(X_train)} users ({train_adolescent} adolescent, {train_adult} adult)")
        logger.info(f"✓ Test: {len(X_test)} users ({test_adolescent} adolescent, {test_adult} adult)")
        
        return X_train, X_test, y_train, y_test
    
    def save_transformer(self, output_path: str) -> None:
        """
        Save the fitted quantile transformer for later use.
        
        Args:
            output_path: Path to save the transformer
        """
        if self.quantile_transformer is None:
            logger.warning("No transformer to save - apply_quantile_transformation must be called first")
            return
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.quantile_transformer, output_path)
            logger.info(f"✓ Saved transformer to {output_path}")
        except Exception as e:
            logger.error(f"Error saving transformer: {e}")
            raise
    
    def load_transformer(self, input_path: str) -> None:
        """
        Load a previously saved quantile transformer.
        
        Args:
            input_path: Path to load the transformer from
        """
        try:
            self.quantile_transformer = joblib.load(input_path)
            logger.info(f"✓ Loaded transformer from {input_path}")
        except Exception as e:
            logger.error(f"Error loading transformer: {e}")
            raise
    
    def save_processed_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                output_dir: str) -> None:
        """
        Save processed features to CSV files.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            output_dir: Directory to save processed features
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Combine features and labels
            train_df = X_train.copy()
            train_df['age_group'] = y_train.map({0: 'adolescent', 1: 'adult'})
            
            test_df = X_test.copy()
            test_df['age_group'] = y_test.map({0: 'adolescent', 1: 'adult'})
            
            # Save to CSV
            train_path = Path(output_dir) / "train_features.csv"
            test_path = Path(output_dir) / "test_features.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"✓ Saved processed features to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving processed features: {e}")
            raise


def prepare_dataset(input_path: str = "data/synthetic_data.csv",
                   output_dir: str = "data",
                   test_size: float = 0.2,
                   apply_transformation: bool = True,
                   random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Main function to prepare dataset for modeling.
    
    Args:
        input_path: Path to raw synthetic data
        output_dir: Directory to save processed features
        test_size: Proportion for test set
        apply_transformation: Whether to apply quantile transformation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Load data
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(random_seed=random_seed)
    
    # Prepare features and labels
    X, y = engineer.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = engineer.split_data(X, y, test_size=test_size)
    
    # Apply quantile transformation if requested
    if apply_transformation:
        X_train, X_test = engineer.apply_quantile_transformation(X_train, X_test)
        
        # Save transformer
        transformer_path = Path(output_dir) / "quantile_transformer.pkl"
        engineer.save_transformer(str(transformer_path))
    
    # Save processed features
    engineer.save_processed_features(X_train, X_test, y_train, y_test, output_dir)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test feature engineering
    X_train, X_test, y_train, y_test = prepare_dataset()
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nTraining label distribution:")
    print(y_train.value_counts())
    print(f"\nTest label distribution:")
    print(y_test.value_counts())
    print(f"\nFeature statistics (training set):")
    print(X_train.describe())

