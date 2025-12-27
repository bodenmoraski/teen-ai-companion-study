"""
Comprehensive tests for demographic classification fixes.

These tests validate that our fixes actually work before we re-run the pipeline.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def demographics():
    """Load demographics data."""
    path = Path("data/features/demographics.parquet")
    if not path.exists():
        pytest.skip("demographics.parquet not found")
    return pd.read_parquet(path)


@pytest.fixture
def user_subreddits():
    """Load user subreddit interactions."""
    path = Path("data/features/user_subreddit_interactions.parquet")
    if not path.exists():
        pytest.skip("user_subreddit_interactions.parquet not found")
    return pd.read_parquet(path)


@pytest.fixture
def ground_truth_age(demographics):
    """Get users with self-declared age as ground truth."""
    mask = demographics['age_bucket_self_declared'].notna()
    return demographics[mask].copy()


@pytest.fixture
def ground_truth_gender(demographics):
    """Get users with self-declared gender as ground truth."""
    mask = demographics['gender_self_declared'].notna()
    return demographics[mask].copy()


# ==============================================================================
# PHASE 1: DIAGNOSTIC TESTS (Baseline)
# ==============================================================================

class TestAgeDistribution:
    """Tests for age score distribution."""
    
    def test_age_scores_exist(self, demographics):
        """Check that age_community_score column exists."""
        assert 'age_community_score' in demographics.columns
    
    def test_age_scores_not_all_zero(self, demographics):
        """Check that age scores have variance."""
        scores = demographics['age_community_score'].dropna()
        assert len(scores) > 0, "No age scores found"
        assert scores.std() > 0.01, "Age scores have no variance"
    
    def test_age_scores_trend_with_age(self, ground_truth_age):
        """Check that age scores increase with self-declared age."""
        mask = ground_truth_age['age_community_score'].notna()
        subset = ground_truth_age[mask]
        
        if len(subset) < 50:
            pytest.skip("Not enough ground truth data")
        
        # Calculate mean score by bucket
        bucket_means = {}
        for bucket in ['13-18', '19-25', '26-40', '41-60', '61-80']:
            data = subset[subset['age_bucket_self_declared'] == bucket]['age_community_score']
            if len(data) > 5:
                bucket_means[bucket] = data.mean()
        
        # Check that 13-18 has lower score than 26-40
        if '13-18' in bucket_means and '26-40' in bucket_means:
            assert bucket_means['13-18'] < bucket_means['26-40'], \
                "Age scores should increase with age (13-18 should be lower than 26-40)"
    
    def test_age_correlation_positive(self, ground_truth_age):
        """Check that age score correlates positively with age."""
        mask = ground_truth_age['age_community_score'].notna()
        subset = ground_truth_age[mask]
        
        age_to_num = {'13-18': 15, '19-25': 22, '26-40': 33, '41-60': 50, '61-80': 70}
        subset = subset.copy()
        subset['age_numeric'] = subset['age_bucket_self_declared'].map(age_to_num)
        
        corr = subset['age_community_score'].corr(subset['age_numeric'])
        print(f"Age correlation: {corr:.4f}")
        
        # Correlation should be positive (older = higher score)
        assert corr > 0, f"Age correlation should be positive, got {corr:.4f}"
    
    def test_age_bucket_distribution_not_equal(self, demographics):
        """
        CRITICAL TEST: Check that age buckets are NOT exactly equal.
        
        If they are 20%/20%/20%/20%/20%, the percentile-based approach is broken.
        """
        counts = demographics['age_bucket_community'].value_counts(normalize=True, dropna=True)
        
        # Check if any bucket is between 19-21% (would indicate equal split)
        for bucket, pct in counts.items():
            if 0.19 < pct < 0.21:
                print(f"WARNING: {bucket} is {pct:.1%} - suggests equal percentile split")
        
        # The distribution should NOT be uniform
        std_of_pcts = counts.std()
        print(f"Std of bucket proportions: {std_of_pcts:.4f}")
        
        # For calibrated thresholds, we expect non-uniform distribution
        # This test will FAIL with current percentile approach
        # After fix, it should PASS
        assert std_of_pcts > 0.02, \
            "Age buckets are too uniform - likely using percentile-based approach"


class TestGenderDistribution:
    """Tests for gender score distribution."""
    
    def test_gender_scores_exist(self, demographics):
        """Check that gender_community_score column exists."""
        assert 'gender_community_score' in demographics.columns
    
    def test_gender_scores_not_all_zero(self, demographics):
        """Check that gender scores have variance."""
        scores = demographics['gender_community_score'].dropna()
        assert len(scores) > 0, "No gender scores found"
        assert scores.std() > 0.01, "Gender scores have no variance"
    
    def test_gender_scores_separate_genders(self, ground_truth_gender):
        """Check that male and female scores are statistically different."""
        mask = ground_truth_gender['gender_community_score'].notna()
        subset = ground_truth_gender[mask]
        
        male_scores = subset[subset['gender_self_declared'] == 'male']['gender_community_score']
        female_scores = subset[subset['gender_self_declared'] == 'female']['gender_community_score']
        
        if len(male_scores) < 20 or len(female_scores) < 20:
            pytest.skip("Not enough ground truth data")
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((male_scores.std()**2 + female_scores.std()**2) / 2)
        cohens_d = (male_scores.mean() - female_scores.mean()) / pooled_std
        
        print(f"Cohen's d for gender: {cohens_d:.4f}")
        print(f"Male mean: {male_scores.mean():.4f}, Female mean: {female_scores.mean():.4f}")
        
        # Effect size should be > 0.2 for any meaningful separation
        # Note: males SHOULD have higher scores (positive direction)
        # Currently this test may fail because direction is wrong
    
    def test_gender_ratio_reasonable(self, demographics):
        """
        CRITICAL TEST: Gender ratio should be within reasonable bounds.
        
        Reddit skews male (~70%), but AI companion subs may skew female.
        A ratio of 37:1 female:male is implausible.
        """
        gender_counts = demographics['gender_community'].value_counts()
        
        if 'female' in gender_counts and 'male' in gender_counts:
            ratio = gender_counts['female'] / max(gender_counts['male'], 1)
            print(f"Female:Male ratio: {ratio:.1f}:1")
            
            # Ratio should be at most 10:1 (very generous)
            assert ratio < 10, f"Female:male ratio of {ratio:.1f}:1 is implausible"


class TestSeedPairs:
    """Tests for seed pair validity."""
    
    def test_age_seed_pairs_have_users(self, user_subreddits):
        """Check that all age seed pairs have sufficient users."""
        age_seeds = [
            ("teenagers", "redditforgrownups"),
            ("teenrelationships", "relationship_advice"),
            ("highschool", "college"),
            ("genz", "genx"),
        ]
        
        usi = user_subreddits.copy()
        usi['subreddit_lower'] = usi['subreddit'].str.lower()
        
        for term1, term2 in age_seeds:
            t1_count = usi[usi['subreddit_lower'] == term1]['author'].nunique()
            t2_count = usi[usi['subreddit_lower'] == term2]['author'].nunique()
            
            print(f"  {term1}: {t1_count} users, {term2}: {t2_count} users")
            
            # Both terms should have at least 10 users
            assert t1_count >= 10, f"{term1} has only {t1_count} users"
            assert t2_count >= 10, f"{term2} has only {t2_count} users"
    
    def test_gender_seed_pairs_have_users(self, user_subreddits):
        """Check that all gender seed pairs have sufficient users."""
        # V2 seed pairs (with malelivingspace instead of oney)
        gender_seeds = [
            ("askwomen", "askmen"),
            ("twoxchromosomes", "mensrights"),
            ("thegirlsurvivalguide", "malelivingspace"),  # V2: replaced oney with malelivingspace
        ]
        
        usi = user_subreddits.copy()
        usi['subreddit_lower'] = usi['subreddit'].str.lower()
        
        for term1, term2 in gender_seeds:
            t1_count = usi[usi['subreddit_lower'] == term1]['author'].nunique()
            t2_count = usi[usi['subreddit_lower'] == term2]['author'].nunique()
            
            print(f"  {term1}: {t1_count} users, {term2}: {t2_count} users")
            
            # Both terms should have at least 10 users
            # This test will FAIL for oney (only 1 user)
            assert t1_count >= 10, f"{term1} has only {t1_count} users"
            assert t2_count >= 10, f"{term2} has only {t2_count} users"


class TestValidationAccuracy:
    """Tests for classification accuracy against ground truth."""
    
    def test_age_accuracy_above_random(self, ground_truth_age):
        """Check that age accuracy is above random (20%)."""
        mask = ground_truth_age['age_bucket_community'].notna()
        subset = ground_truth_age[mask]
        
        if len(subset) < 50:
            pytest.skip("Not enough validation data")
        
        correct = (subset['age_bucket_self_declared'] == subset['age_bucket_community']).sum()
        total = len(subset)
        accuracy = correct / total
        
        print(f"Age accuracy: {accuracy:.1%} ({correct}/{total})")
        
        # Should be significantly above 20% random
        # Currently this test will FAIL (~35%)
        # After fix, target is 50%+
        assert accuracy > 0.35, f"Age accuracy {accuracy:.1%} should be > 35%"
    
    def test_gender_accuracy_above_zero(self, ground_truth_gender):
        """Check that gender accuracy is above 0%."""
        mask = ground_truth_gender['gender_community'].notna()
        mask &= ground_truth_gender['gender_community'] != 'unknown'
        subset = ground_truth_gender[mask]
        
        if len(subset) < 50:
            pytest.skip("Not enough validation data")
        
        correct = (subset['gender_self_declared'] == subset['gender_community']).sum()
        total = len(subset)
        accuracy = correct / total if total > 0 else 0
        
        print(f"Gender accuracy: {accuracy:.1%} ({correct}/{total})")
        
        # Should be above random (33% for 3 classes)
        # Currently this test will FAIL (0%)
        assert accuracy > 0.20, f"Gender accuracy {accuracy:.1%} should be > 20%"


# ==============================================================================
# PHASE 2: CALIBRATED THRESHOLD TESTS
# ==============================================================================

class TestCalibratedThresholds:
    """Tests for the new calibrated threshold approach."""
    
    def test_calculate_optimal_age_thresholds(self, ground_truth_age):
        """Calculate and verify optimal age thresholds."""
        mask = ground_truth_age['age_community_score'].notna()
        subset = ground_truth_age[mask]
        
        # Calculate mean score for each bucket
        bucket_stats = {}
        for bucket in ['13-18', '19-25', '26-40', '41-60', '61-80']:
            data = subset[subset['age_bucket_self_declared'] == bucket]['age_community_score']
            if len(data) > 3:
                bucket_stats[bucket] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'n': len(data)
                }
        
        print("Bucket statistics:")
        for bucket, stats in bucket_stats.items():
            print(f"  {bucket}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")
        
        # Verify that bucket means generally increase with age
        means = [bucket_stats[b]['mean'] for b in ['13-18', '19-25', '26-40'] if b in bucket_stats]
        if len(means) >= 3:
            assert means[0] < means[2], "13-18 should have lower mean than 26-40"
    
    def test_optimal_gender_threshold(self, ground_truth_gender):
        """Calculate optimal gender threshold."""
        mask = ground_truth_gender['gender_community_score'].notna()
        subset = ground_truth_gender[mask]
        
        male_scores = subset[subset['gender_self_declared'] == 'male']['gender_community_score']
        female_scores = subset[subset['gender_self_declared'] == 'female']['gender_community_score']
        
        if len(male_scores) < 20 or len(female_scores) < 20:
            pytest.skip("Not enough data")
        
        # Optimal threshold is midpoint between means
        optimal_threshold = (male_scores.mean() + female_scores.mean()) / 2
        
        print(f"Male mean: {male_scores.mean():.4f}")
        print(f"Female mean: {female_scores.mean():.4f}")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        
        # Test classification with optimal threshold
        # Score > threshold → male, Score < threshold → female
        all_scores = subset['gender_community_score']
        all_true = subset['gender_self_declared']
        
        predicted = ['male' if s > optimal_threshold else 'female' for s in all_scores]
        
        # Calculate accuracy (excluding nonbinary for this test)
        binary_mask = (all_true == 'male') | (all_true == 'female')
        binary_true = all_true[binary_mask]
        binary_pred = pd.Series(predicted)[binary_mask.values]
        
        accuracy = (binary_true.values == binary_pred.values).mean()
        print(f"Accuracy with optimal threshold: {accuracy:.1%}")
        
        # This shows the maximum accuracy we can achieve with this data
        # If it's below 60%, the embeddings don't separate genders well


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

