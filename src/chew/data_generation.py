"""
Data Generation Module for Reddit Age Classification

Generates synthetic data matching the statistical properties from Chew et al. (2021).
Creates 2,075 labeled users with 15 features based on Table 5 distributions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates synthetic Reddit user data matching Chew et al. (2021) distributions.
    
    All feature means and standard deviations are from Table 5 (quantile-transformed values).
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load WWBP terms
        self.wwbp_23_29_terms, self.wwbp_13_18_terms = self._load_wwbp_terms()
        
        # Dataset specifications from paper
        self.n_total = 2075
        self.train_ratio = 0.8
        self.adolescent_ratio = 0.611  # 1268/2075 from paper
        
        # Feature specifications (quantile-transformed means and stds from Table 5)
        self.feature_specs = self._define_feature_specs()
        
    def _load_wwbp_terms(self) -> Tuple[List[str], List[str]]:
        """
        Load WWBP word lists from WWBP.md file.
        
        Returns:
            Tuple of (adult_terms, adolescent_terms)
        """
        try:
            wwbp_path = Path(__file__).parent.parent / "WWBP.md"
            
            with open(wwbp_path, 'r') as f:
                content = f.read()
            
            # Parse the file to extract term lists
            import re
            
            # Extract wwbp_23_29_terms - find entire list including nested brackets
            wwbp_23_29_terms = []
            if 'wwbp_23_29_terms = [' in content:
                start_idx = content.find('wwbp_23_29_terms = [')
                bracket_count = 0
                end_idx = start_idx
                
                # Find matching closing bracket
                for i in range(start_idx, len(content)):
                    if content[i] == '[':
                        bracket_count += 1
                    elif content[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                
                terms_block = content[start_idx:end_idx]
                wwbp_23_29_terms = re.findall(r'"([^"]+)"', terms_block)
            
            # Extract wwbp_13_18_terms
            wwbp_13_18_terms = []
            if 'wwbp_13_18_terms = [' in content:
                start_idx = content.find('wwbp_13_18_terms = [')
                bracket_count = 0
                end_idx = start_idx
                
                # Find matching closing bracket
                for i in range(start_idx, len(content)):
                    if content[i] == '[':
                        bracket_count += 1
                    elif content[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                
                terms_block = content[start_idx:end_idx]
                wwbp_13_18_terms = re.findall(r'"([^"]+)"', terms_block)
            
            logger.info(f"✓ Loaded {len(wwbp_23_29_terms)} adult terms and {len(wwbp_13_18_terms)} adolescent terms from WWBP.md")
            
            return wwbp_23_29_terms, wwbp_13_18_terms
            
        except Exception as e:
            logger.error(f"Error loading WWBP.md: {e}")
            raise
    
    def _define_feature_specs(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Define feature specifications based on Table 5 from the paper.
        
        Returns:
            Dictionary mapping feature names to their adolescent/adult (mean, std) tuples
        """
        return {
            # Literary Characteristics (3 features)
            'sentences_per_comment': {
                'adolescent': (-0.43, 0.4),
                'adult': (0.37, 0.4)
            },
            'avg_coleman_liau_index': {
                'adolescent': (-0.25, 0.4),
                'adult': (0.08, 0.4)
            },
            'prop_comments_in_own_thread': {
                'adolescent': (-0.74, 0.8),
                'adult': (-1.41, 1.0)
            },
            
            # Account Metadata (2 features)
            'year_account_created': {
                'adolescent': (2.96, 0.8),  # newer accounts
                'adult': (1.35, 1.2)        # older accounts
            },
            'comment_karma': {
                'adolescent': (-0.19, 0.3),
                'adult': (0.30, 0.4)
            },
            
            # Subreddit Usage (3 features)
            'prop_posts_in_teenagers': {
                'adolescent': (-3.49, 0.9),  # higher proportion
                'adult': (-5.11, 0.3)        # very low
            },
            'prop_posts_in_news': {
                'adolescent': (-4.29, 0.9),
                'adult': (-5.03, 0.3)
            },
            'percentile_75_subscriber_count': {
                'adolescent': (-0.14, 0.5),  # post in larger subreddits
                'adult': (-0.33, 0.6)
            },
            
            # TF-IDF Text Features (5 features)
            'tfidf_school': {
                'adolescent': (-2.46, 1.0),
                'adult': (-2.63, 1.2)
            },
            'tfidf_need': {
                'adolescent': (-1.58, 0.9),
                'adult': (-0.31, 0.9)  # adults use more
            },
            'tfidf_look_like': {
                'adolescent': (-3.04, 0.9),
                'adult': (-1.84, 1.1)  # adults use more
            },
            'tfidf_home': {
                'adolescent': (-3.45, 0.9),
                'adult': (-1.89, 1.2)  # adults use more
            },
            'tfidf_totally': {
                'adolescent': (-4.23, 0.7),
                'adult': (-2.95, 1.2)  # adults use more
            },
            
            # WWBP Features (2 features)
            'freq_wwbp_23_29_word_set': {
                'adolescent': (-1.23, 0.8),  # low usage of adult words
                'adult': (-0.05, 1.0)        # high usage
            },
            'normalized_count_wwbp_23_29': {
                'adolescent': (-1.29, 0.8),
                'adult': (0.04, 0.8)
            }
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete synthetic dataset with realistic correlations.
        
        Returns:
            DataFrame with 2,075 rows and 16 columns (15 features + age_group label)
        """
        logger.info(f"Generating {self.n_total} synthetic users...")
        
        # Calculate sample sizes
        n_adolescent = int(self.n_total * self.adolescent_ratio)
        n_adult = self.n_total - n_adolescent
        
        logger.info(f"  Adolescents (13-20): {n_adolescent}")
        logger.info(f"  Adults (21-54): {n_adult}")
        
        # Generate features for each age group
        adolescent_data = self._generate_age_group_data(n_adolescent, 'adolescent')
        adult_data = self._generate_age_group_data(n_adult, 'adult')
        
        # Combine and shuffle
        df = pd.concat([adolescent_data, adult_data], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        logger.info(f"✓ Generated {len(df)} synthetic users")
        
        return df
    
    def _generate_age_group_data(self, n_samples: int, age_group: str) -> pd.DataFrame:
        """
        Generate feature data for a specific age group with realistic correlations.
        
        Args:
            n_samples: Number of samples to generate
            age_group: 'adolescent' or 'adult'
            
        Returns:
            DataFrame with generated features
        """
        data = {}
        
        # Generate base features independently
        for feature_name, specs in self.feature_specs.items():
            mean, std = specs[age_group]
            data[feature_name] = np.random.normal(mean, std, n_samples)
        
        # Add realistic correlations
        data = self._add_feature_correlations(data, age_group)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add age group label
        df['age_group'] = age_group
        
        return df
    
    def _add_feature_correlations(self, data: Dict[str, np.ndarray], age_group: str) -> Dict[str, np.ndarray]:
        """
        Add realistic correlations between features.
        
        Args:
            data: Dictionary of feature arrays
            age_group: 'adolescent' or 'adult'
            
        Returns:
            Modified data dictionary with correlations
        """
        n = len(data['sentences_per_comment'])
        
        # Correlation 1: Higher comment_karma → Older year_account_created
        # Add correlated noise (correlation ~0.3)
        karma_influence = 0.3 * (data['comment_karma'] / np.std(data['comment_karma']))
        data['year_account_created'] += karma_influence
        
        # Correlation 2: Higher prop_posts_in_teenagers → Lower prop_posts_in_news
        # Negative correlation (~-0.25)
        teenagers_influence = -0.25 * (data['prop_posts_in_teenagers'] / np.std(data['prop_posts_in_teenagers']))
        data['prop_posts_in_news'] += teenagers_influence
        
        # Correlation 3: Higher sentences_per_comment → Higher avg_coleman_liau_index
        # Positive correlation (~0.4)
        sentences_influence = 0.4 * (data['sentences_per_comment'] / np.std(data['sentences_per_comment']))
        data['avg_coleman_liau_index'] += sentences_influence
        
        # Correlation 4: WWBP features should be correlated
        # freq_wwbp and normalized_count should have high correlation (~0.8)
        wwbp_base = 0.8 * (data['freq_wwbp_23_29_word_set'] / np.std(data['freq_wwbp_23_29_word_set']))
        data['normalized_count_wwbp_23_29'] = (
            data['normalized_count_wwbp_23_29'] * 0.6 + wwbp_base * 0.4
        )
        
        # Correlation 5: TF-IDF school term higher for adolescents with more teenager posts
        if age_group == 'adolescent':
            school_influence = 0.3 * (data['prop_posts_in_teenagers'] / np.std(data['prop_posts_in_teenagers']))
            data['tfidf_school'] += school_influence
        
        return data
    
    def save_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save generated dataset to CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Path to output CSV file
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved dataset to {output_path}")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise


def generate_synthetic_data(output_path: str = "data/synthetic_data.csv", random_seed: int = 42) -> pd.DataFrame:
    """
    Main function to generate and save synthetic dataset.
    
    Args:
        output_path: Path to save the generated dataset
        random_seed: Random seed for reproducibility
        
    Returns:
        Generated DataFrame
    """
    generator = SyntheticDataGenerator(random_seed=random_seed)
    df = generator.generate_dataset()
    generator.save_dataset(df, output_path)
    return df


if __name__ == "__main__":
    # Test the data generation
    df = generate_synthetic_data()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SYNTHETIC DATA SUMMARY")
    print("="*60)
    print(f"\nTotal samples: {len(df)}")
    print(f"\nAge group distribution:")
    print(df['age_group'].value_counts())
    print(f"\nFeature names ({len(df.columns)-1} features):")
    for col in df.columns:
        if col != 'age_group':
            print(f"  - {col}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nBasic statistics by age group:")
    print(df.groupby('age_group')['sentences_per_comment'].describe())

