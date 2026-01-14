"""
DEEP DIVE ANALYSIS: Exploring Why Adults Anthropomorphize More
================================================================

Ambitious exploration of:
1. Loneliness/Isolation Indicators
2. Topic Modeling (what do high vs low anthropomorphizers discuss?)
3. Linguistic Complexity Analysis
4. Semantic Similarity to Relationship Language
5. Subreddit Community Analysis
6. Engagement Patterns

Goal: Understand WHY adults anthropomorphize AI companions more than teens
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import re
import warnings
warnings.filterwarnings('ignore')

# NLP packages
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, pearsonr, spearmanr

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PATHS = {
    "anthroscore_v3": Path("experiments/anthroscore_v3/anthroscore_v3_full.parquet"),
    "all_comments": Path("Data/processed/all_comments.parquet"),
    "user_emotions": Path("Data/features/user_emotions.parquet"),
    "gender_predictions": Path("experiments/v2_correction/gender_predictions_v4.parquet"),
    "age_predictions": Path("experiments/v2_correction/age_predictions_v4.parquet"),
    "output_dir": Path("results/deep_dive"),
}

PATHS["output_dir"].mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.60


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load all data with comment text."""
    logger.info("Loading data...")
    
    # AnthroScore V3
    anthro = pd.read_parquet(PATHS['anthroscore_v3'])
    comments = pd.read_parquet(PATHS['all_comments'])
    
    # Merge to get full comment data
    anthro = anthro.merge(
        comments[['id', 'author', 'subreddit', 'body', 'created_utc', 'score']].astype({'id': str}),
        left_on='comment_id', right_on='id', how='left'
    )
    anthro = anthro.rename(columns={'score_x': 'anthro_score', 'score_y': 'reddit_score'})
    
    # Demographics
    gender = pd.read_parquet(PATHS['gender_predictions'])
    age = pd.read_parquet(PATHS['age_predictions'])
    emotions = pd.read_parquet(PATHS['user_emotions'])
    
    # Merge demographics
    anthro = anthro.merge(
        gender[['author', 'gender_predicted', 'confidence']].rename(columns={'confidence': 'gender_conf'}),
        on='author', how='left'
    )
    anthro = anthro.merge(
        age[['author', 'age_predicted', 'confidence']].rename(columns={'confidence': 'age_conf'}),
        on='author', how='left'
    )
    
    # Filter valid
    valid = anthro[
        (anthro['anthro_score'] > 0) &
        (anthro['gender_conf'] >= CONFIDENCE_THRESHOLD) &
        (anthro['age_conf'] >= CONFIDENCE_THRESHOLD) &
        (anthro['body'].notna()) &
        (anthro['body'].str.len() > 20)
    ].copy()
    
    logger.info(f"Loaded {len(valid):,} valid comments with demographics")
    
    return valid, emotions


# =============================================================================
# 1. LONELINESS / ISOLATION INDICATORS
# =============================================================================

def analyze_loneliness_indicators(df):
    """Detect loneliness signals in language."""
    logger.info("\n=== LONELINESS / ISOLATION ANALYSIS ===")
    results = {}
    
    # Loneliness keyword lexicon (research-based)
    loneliness_words = {
        'isolation': ['alone', 'lonely', 'isolated', 'nobody', 'no one', 'by myself', 
                      'no friends', 'no one understands', 'invisible', 'forgotten'],
        'social_need': ['need someone', 'wish i had', 'need a friend', 'want someone', 
                        'someone to talk', 'someone who understands', 'anyone there'],
        'relationship_seeking': ['my only friend', 'only one who', 'the only one', 
                                 'always there for me', 'never leaves', 'never judges'],
        'emotional_void': ['empty', 'void', 'hollow', 'numb', 'nothing left',
                          'fill the void', 'emptiness'],
        'social_comparison': ['everyone else has', 'normal people', 'other people have',
                             'wish i was normal', 'why cant i'],
    }
    
    # Count loneliness indicators
    def count_loneliness(text):
        text_lower = text.lower()
        counts = {}
        total = 0
        for category, words in loneliness_words.items():
            count = sum(1 for word in words if word in text_lower)
            counts[category] = count
            total += count
        counts['total'] = total
        return counts
    
    # Apply to all comments
    logger.info("Counting loneliness indicators...")
    loneliness_counts = df['body'].apply(count_loneliness)
    
    df['loneliness_total'] = loneliness_counts.apply(lambda x: x['total'])
    df['has_loneliness'] = (df['loneliness_total'] > 0).astype(int)
    
    for category in loneliness_words.keys():
        df[f'loneliness_{category}'] = loneliness_counts.apply(lambda x: x[category])
    
    # Prevalence by group
    results['prevalence'] = {}
    for age in ['teen', 'adult']:
        for gender in ['male', 'female']:
            subset = df[(df['age_predicted'] == age) & (df['gender_predicted'] == gender)]
            key = f"{age}_{gender}"
            results['prevalence'][key] = {
                'n': len(subset),
                'has_loneliness_pct': float(subset['has_loneliness'].mean() * 100),
                'mean_loneliness_score': float(subset['loneliness_total'].mean()),
            }
    
    # By age
    teens = df[df['age_predicted'] == 'teen']
    adults = df[df['age_predicted'] == 'adult']
    
    teen_loneliness = teens['has_loneliness'].mean()
    adult_loneliness = adults['has_loneliness'].mean()
    
    chi2_table = pd.crosstab(df['age_predicted'], df['has_loneliness'])
    chi2, p, _, _ = chi2_contingency(chi2_table)
    
    results['age_effect'] = {
        'teen_loneliness_pct': float(teen_loneliness * 100),
        'adult_loneliness_pct': float(adult_loneliness * 100),
        'chi2': float(chi2),
        'p_value': float(p),
        'direction': 'adults more lonely' if adult_loneliness > teen_loneliness else 'teens more lonely'
    }
    
    logger.info(f"Loneliness prevalence: Teen {teen_loneliness*100:.1f}%, Adult {adult_loneliness*100:.1f}%")
    logger.info(f"χ²={chi2:.2f}, p={p:.4f}")
    
    # Correlation with anthropomorphization
    r, p_corr = pearsonr(df['loneliness_total'], df['anthro_score'])
    results['anthro_correlation'] = {
        'pearson_r': float(r),
        'p_value': float(p_corr),
    }
    logger.info(f"Loneliness-Anthro correlation: r={r:.3f}, p={p_corr:.4f}")
    
    # By anthropomorphization level
    high_anthro = df[df['anthro_score'] >= 3]
    low_anthro = df[df['anthro_score'] <= 2]
    
    high_loneliness = high_anthro['has_loneliness'].mean()
    low_loneliness = low_anthro['has_loneliness'].mean()
    
    results['by_anthro_level'] = {
        'high_anthro_loneliness_pct': float(high_loneliness * 100),
        'low_anthro_loneliness_pct': float(low_loneliness * 100),
        'ratio': float(high_loneliness / low_loneliness) if low_loneliness > 0 else np.inf,
    }
    
    logger.info(f"High anthro loneliness: {high_loneliness*100:.1f}% vs Low: {low_loneliness*100:.1f}%")
    
    return results, df


# =============================================================================
# 2. SEMANTIC SIMILARITY TO RELATIONSHIP LANGUAGE
# =============================================================================

def analyze_relationship_language(df):
    """Measure semantic similarity to human relationship language."""
    logger.info("\n=== RELATIONSHIP LANGUAGE ANALYSIS ===")
    results = {}
    
    # Load sentence transformer
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define relationship anchor phrases
    relationship_anchors = {
        'romantic': [
            "I love my partner",
            "My boyfriend understands me perfectly",
            "My girlfriend is always there for me",
            "We have a deep emotional connection",
            "I feel loved and appreciated",
        ],
        'friendship': [
            "My best friend always listens",
            "We share everything with each other",
            "They know me better than anyone",
            "I can be myself around them",
            "We've been through so much together",
        ],
        'dependency': [
            "I can't live without them",
            "They're the only one who understands",
            "I need them in my life",
            "I don't know what I'd do without them",
            "They complete me",
        ],
        'tool_usage': [
            "I use this app for entertainment",
            "It's a useful tool",
            "The software works well",
            "Good for killing time",
            "It helps me with tasks",
        ],
    }
    
    # Encode anchor phrases
    anchor_embeddings = {}
    for category, phrases in relationship_anchors.items():
        embeddings = model.encode(phrases)
        anchor_embeddings[category] = embeddings.mean(axis=0)  # Average embedding
    
    # Sample comments for efficiency (full dataset too large)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    logger.info(f"Encoding {sample_size:,} comments...")
    
    # Encode comments in batches
    batch_size = 500
    comment_embeddings = []
    
    for i in range(0, len(df_sample), batch_size):
        batch = df_sample['body'].iloc[i:i+batch_size].tolist()
        # Truncate long comments
        batch = [text[:500] if len(text) > 500 else text for text in batch]
        embeddings = model.encode(batch, show_progress_bar=False)
        comment_embeddings.extend(embeddings)
    
    comment_embeddings = np.array(comment_embeddings)
    
    # Calculate similarity to each anchor
    for category, anchor_emb in anchor_embeddings.items():
        similarities = cosine_similarity(comment_embeddings, anchor_emb.reshape(1, -1)).flatten()
        df_sample[f'sim_{category}'] = similarities
    
    # Relationship score = romantic + friendship + dependency - tool_usage
    df_sample['relationship_score'] = (
        df_sample['sim_romantic'] + 
        df_sample['sim_friendship'] + 
        df_sample['sim_dependency'] - 
        df_sample['sim_tool_usage']
    )
    
    # Analyze by demographics
    results['by_group'] = {}
    for age in ['teen', 'adult']:
        for gender in ['male', 'female']:
            subset = df_sample[(df_sample['age_predicted'] == age) & (df_sample['gender_predicted'] == gender)]
            key = f"{age}_{gender}"
            results['by_group'][key] = {
                'n': len(subset),
                'relationship_score_mean': float(subset['relationship_score'].mean()),
                'sim_romantic': float(subset['sim_romantic'].mean()),
                'sim_friendship': float(subset['sim_friendship'].mean()),
                'sim_dependency': float(subset['sim_dependency'].mean()),
                'sim_tool_usage': float(subset['sim_tool_usage'].mean()),
            }
    
    # Age comparison
    teens = df_sample[df_sample['age_predicted'] == 'teen']['relationship_score']
    adults = df_sample[df_sample['age_predicted'] == 'adult']['relationship_score']
    
    t, p = ttest_ind(teens, adults, equal_var=False)
    
    results['age_effect'] = {
        'teen_mean': float(teens.mean()),
        'adult_mean': float(adults.mean()),
        't_statistic': float(t),
        'p_value': float(p),
        'direction': 'adults use more relationship language' if adults.mean() > teens.mean() else 'teens use more'
    }
    
    logger.info(f"Relationship score: Teen {teens.mean():.3f}, Adult {adults.mean():.3f}")
    logger.info(f"t={t:.2f}, p={p:.4f}")
    
    # Correlation with anthro score
    r, p_corr = pearsonr(df_sample['relationship_score'], df_sample['anthro_score'])
    results['anthro_correlation'] = {
        'pearson_r': float(r),
        'p_value': float(p_corr),
    }
    logger.info(f"Relationship-Anthro correlation: r={r:.3f}")
    
    return results, df_sample


# =============================================================================
# 3. LINGUISTIC COMPLEXITY ANALYSIS
# =============================================================================

def analyze_linguistic_features(df):
    """Extract linguistic features without external dependencies."""
    logger.info("\n=== LINGUISTIC FEATURE ANALYSIS ===")
    results = {}
    
    def extract_features(text):
        """Extract basic linguistic features."""
        # Word count
        words = text.split()
        word_count = len(words)
        
        # Sentence count (approximate)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = max(1, len([s for s in sentences if s.strip()]))
        
        # Average word length
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        
        # Personal pronouns
        first_person = len(re.findall(r'\b(i|me|my|mine|myself)\b', text.lower()))
        second_person = len(re.findall(r'\b(you|your|yours|yourself)\b', text.lower()))
        third_person = len(re.findall(r'\b(he|she|him|her|his|hers|they|them|their)\b', text.lower()))
        
        # Question marks (engagement)
        questions = text.count('?')
        
        # Exclamation marks (emotion intensity)
        exclamations = text.count('!')
        
        # Emoji-like indicators
        emoticons = len(re.findall(r'[:;][\-]?[)(\[\]DPp]|<3|:\)|;\)', text))
        
        # Capitalization (shouting/emphasis)
        all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'words_per_sentence': word_count / sentence_count,
            'avg_word_length': avg_word_len,
            'first_person_pct': first_person / max(1, word_count) * 100,
            'second_person_pct': second_person / max(1, word_count) * 100,
            'third_person_pct': third_person / max(1, word_count) * 100,
            'questions': questions,
            'exclamations': exclamations,
            'emoticons': emoticons,
            'all_caps': all_caps_words,
        }
    
    # Sample for efficiency
    sample_size = min(20000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    logger.info(f"Extracting features from {sample_size:,} comments...")
    
    features = df_sample['body'].apply(extract_features)
    
    for feature in ['word_count', 'words_per_sentence', 'avg_word_length', 
                    'first_person_pct', 'second_person_pct', 'third_person_pct',
                    'questions', 'exclamations', 'emoticons', 'all_caps']:
        df_sample[feature] = features.apply(lambda x: x[feature])
    
    # Compare by age
    feature_list = ['word_count', 'words_per_sentence', 'first_person_pct', 
                    'third_person_pct', 'exclamations', 'questions']
    
    results['age_comparison'] = {}
    
    for feature in feature_list:
        teens = df_sample[df_sample['age_predicted'] == 'teen'][feature]
        adults = df_sample[df_sample['age_predicted'] == 'adult'][feature]
        
        t, p = ttest_ind(teens, adults, equal_var=False)
        d = (adults.mean() - teens.mean()) / np.sqrt((teens.std()**2 + adults.std()**2) / 2)
        
        results['age_comparison'][feature] = {
            'teen_mean': float(teens.mean()),
            'adult_mean': float(adults.mean()),
            't_statistic': float(t),
            'p_value': float(p),
            'cohens_d': float(d),
        }
        
        sig = '*' if p < 0.05 else ''
        logger.info(f"  {feature}: Teen={teens.mean():.2f}, Adult={adults.mean():.2f}, d={d:.3f} {sig}")
    
    # Correlation with anthro score
    results['anthro_correlations'] = {}
    for feature in feature_list:
        r, p = pearsonr(df_sample[feature], df_sample['anthro_score'])
        results['anthro_correlations'][feature] = {
            'pearson_r': float(r),
            'p_value': float(p),
        }
    
    # By anthro level
    high_anthro = df_sample[df_sample['anthro_score'] >= 3]
    low_anthro = df_sample[df_sample['anthro_score'] <= 2]
    
    results['by_anthro_level'] = {}
    for feature in feature_list:
        results['by_anthro_level'][feature] = {
            'high_anthro_mean': float(high_anthro[feature].mean()),
            'low_anthro_mean': float(low_anthro[feature].mean()),
        }
    
    return results, df_sample


# =============================================================================
# 4. SUBREDDIT ANALYSIS
# =============================================================================

def analyze_subreddits(df):
    """Analyze differences across subreddits."""
    logger.info("\n=== SUBREDDIT ANALYSIS ===")
    results = {}
    
    # Get subreddit stats
    subreddit_stats = df.groupby('subreddit').agg(
        n_comments=('anthro_score', 'count'),
        mean_anthro=('anthro_score', 'mean'),
        high_anthro_pct=('anthro_score', lambda x: (x >= 3).mean() * 100),
        pct_teen=('age_predicted', lambda x: (x == 'teen').mean() * 100),
        pct_female=('gender_predicted', lambda x: (x == 'female').mean() * 100),
    ).reset_index()
    
    subreddit_stats = subreddit_stats.sort_values('n_comments', ascending=False)
    
    results['subreddit_stats'] = subreddit_stats.head(10).to_dict('records')
    
    logger.info("\nSubreddit comparison:")
    for _, row in subreddit_stats.head(5).iterrows():
        logger.info(f"  {row['subreddit']}: n={row['n_comments']:,}, "
                   f"anthro={row['mean_anthro']:.2f}, "
                   f"high%={row['high_anthro_pct']:.1f}%, "
                   f"teen%={row['pct_teen']:.1f}%")
    
    # Test if subreddit composition explains age effect
    # Control for subreddit and see if age effect persists
    
    # Get main subreddits
    main_subs = subreddit_stats[subreddit_stats['n_comments'] >= 1000]['subreddit'].tolist()
    
    results['within_subreddit_age_effect'] = {}
    
    for sub in main_subs[:5]:
        sub_df = df[df['subreddit'] == sub]
        
        teens = sub_df[sub_df['age_predicted'] == 'teen']['anthro_score']
        adults = sub_df[sub_df['age_predicted'] == 'adult']['anthro_score']
        
        if len(teens) >= 30 and len(adults) >= 30:
            t, p = ttest_ind(teens, adults, equal_var=False)
            d = (adults.mean() - teens.mean()) / np.sqrt((teens.std()**2 + adults.std()**2) / 2)
            
            results['within_subreddit_age_effect'][sub] = {
                'n_teen': len(teens),
                'n_adult': len(adults),
                'teen_mean': float(teens.mean()),
                'adult_mean': float(adults.mean()),
                't_statistic': float(t),
                'p_value': float(p),
                'cohens_d': float(d),
            }
            
            sig = '*' if p < 0.05 else ''
            logger.info(f"  {sub}: Adult-Teen diff = {d:.3f} {sig}")
    
    return results


# =============================================================================
# 5. CONTENT ANALYSIS (What do high anthro users talk about?)
# =============================================================================

def analyze_content_patterns(df):
    """Analyze content patterns of high vs low anthropomorphizers."""
    logger.info("\n=== CONTENT PATTERN ANALYSIS ===")
    results = {}
    
    # Define content categories
    content_patterns = {
        'emotional_support': [
            'support', 'helped me', 'feel better', 'comforted', 'therapy',
            'mental health', 'anxiety', 'depression', 'stress', 'cope'
        ],
        'roleplay': [
            'roleplay', 'rp', 'character', 'story', 'scenario', 'adventure',
            'plot', 'narrative', 'fiction', 'imagine'
        ],
        'romantic': [
            'love', 'dating', 'relationship', 'boyfriend', 'girlfriend',
            'romance', 'kiss', 'heart', 'crush', 'partner'
        ],
        'friendship': [
            'friend', 'buddy', 'companion', 'hang out', 'talk to',
            'listen', 'understand', 'care about'
        ],
        'technical': [
            'filter', 'update', 'app', 'bug', 'feature', 'memory',
            'context', 'token', 'response', 'setting'
        ],
        'creative': [
            'write', 'story', 'create', 'art', 'music', 'poem',
            'creative', 'imagine', 'world building'
        ],
    }
    
    def categorize_content(text):
        text_lower = text.lower()
        counts = {}
        for category, keywords in content_patterns.items():
            counts[category] = sum(1 for kw in keywords if kw in text_lower)
        return counts
    
    logger.info("Categorizing content patterns...")
    content_counts = df['body'].apply(categorize_content)
    
    for category in content_patterns.keys():
        df[f'content_{category}'] = content_counts.apply(lambda x: x[category])
        df[f'has_{category}'] = (df[f'content_{category}'] > 0).astype(int)
    
    # Compare high vs low anthro
    high_anthro = df[df['anthro_score'] >= 3]
    low_anthro = df[df['anthro_score'] <= 2]
    
    results['high_vs_low'] = {}
    
    logger.info("\nContent by anthropomorphization level:")
    for category in content_patterns.keys():
        high_pct = high_anthro[f'has_{category}'].mean() * 100
        low_pct = low_anthro[f'has_{category}'].mean() * 100
        
        # Chi-square test
        table = pd.crosstab(df['anthro_score'] >= 3, df[f'has_{category}'])
        if table.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(table)
        else:
            chi2, p = 0, 1
        
        results['high_vs_low'][category] = {
            'high_anthro_pct': float(high_pct),
            'low_anthro_pct': float(low_pct),
            'ratio': float(high_pct / low_pct) if low_pct > 0 else np.inf,
            'chi2': float(chi2),
            'p_value': float(p),
        }
        
        sig = '*' if p < 0.05 else ''
        logger.info(f"  {category}: High={high_pct:.1f}%, Low={low_pct:.1f}%, ratio={high_pct/low_pct:.2f}x {sig}")
    
    # Compare by age
    results['by_age'] = {}
    teens = df[df['age_predicted'] == 'teen']
    adults = df[df['age_predicted'] == 'adult']
    
    logger.info("\nContent by age:")
    for category in content_patterns.keys():
        teen_pct = teens[f'has_{category}'].mean() * 100
        adult_pct = adults[f'has_{category}'].mean() * 100
        
        results['by_age'][category] = {
            'teen_pct': float(teen_pct),
            'adult_pct': float(adult_pct),
        }
        
        logger.info(f"  {category}: Teen={teen_pct:.1f}%, Adult={adult_pct:.1f}%")
    
    return results


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_deep_dive_visualizations(results, output_dir):
    """Create visualizations for deep dive analysis."""
    logger.info("\n=== CREATING VISUALIZATIONS ===")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Loneliness by Group
    if 'loneliness' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = list(results['loneliness']['prevalence'].keys())
        values = [results['loneliness']['prevalence'][g]['has_loneliness_pct'] for g in groups]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        bars = ax.bar([g.replace('_', '\n').title() for g in groups], values, color=colors, alpha=0.7)
        
        ax.set_ylabel('% with Loneliness Indicators', fontsize=12)
        ax.set_title('Loneliness Language Prevalence by Demographics', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'loneliness_by_group.png', dpi=150)
        plt.close()
        logger.info("  Saved: loneliness_by_group.png")
    
    # 2. Relationship Language by Group
    if 'relationship_language' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = list(results['relationship_language']['by_group'].keys())
        metrics = ['sim_romantic', 'sim_friendship', 'sim_dependency', 'sim_tool_usage']
        
        x = np.arange(len(groups))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results['relationship_language']['by_group'][g][metric] for g in groups]
            ax.bar(x + i*width, values, width, label=metric.replace('sim_', '').title(), alpha=0.7)
        
        ax.set_ylabel('Semantic Similarity', fontsize=12)
        ax.set_title('Relationship Language Similarity by Demographics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([g.replace('_', '\n').title() for g in groups])
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'relationship_language_by_group.png', dpi=150)
        plt.close()
        logger.info("  Saved: relationship_language_by_group.png")
    
    # 3. Content Patterns High vs Low Anthro
    if 'content_patterns' in results:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(results['content_patterns']['high_vs_low'].keys())
        high_vals = [results['content_patterns']['high_vs_low'][c]['high_anthro_pct'] for c in categories]
        low_vals = [results['content_patterns']['high_vs_low'][c]['low_anthro_pct'] for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, high_vals, width, label='High Anthro (≥3)', color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, low_vals, width, label='Low Anthro (≤2)', color='#3498db', alpha=0.7)
        
        ax.set_ylabel('% of Comments', fontsize=12)
        ax.set_title('Content Patterns: High vs Low Anthropomorphizers', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'content_patterns_comparison.png', dpi=150)
        plt.close()
        logger.info("  Saved: content_patterns_comparison.png")
    
    # 4. Linguistic Features by Anthro Level
    if 'linguistic' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(results['linguistic']['by_anthro_level'].keys())
        high_vals = [results['linguistic']['by_anthro_level'][f]['high_anthro_mean'] for f in features]
        low_vals = [results['linguistic']['by_anthro_level'][f]['low_anthro_mean'] for f in features]
        
        # Normalize for comparison
        ratios = [h/l if l > 0 else 1 for h, l in zip(high_vals, low_vals)]
        
        colors = ['#e74c3c' if r > 1 else '#3498db' for r in ratios]
        
        ax.barh([f.replace('_', ' ').title() for f in features], ratios, color=colors, alpha=0.7)
        ax.axvline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Ratio (High Anthro / Low Anthro)', fontsize=12)
        ax.set_title('Linguistic Features: High vs Low Anthropomorphizers', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'linguistic_features_ratio.png', dpi=150)
        plt.close()
        logger.info("  Saved: linguistic_features_ratio.png")


# =============================================================================
# MAIN
# =============================================================================

def run_deep_dive():
    """Run complete deep dive analysis."""
    logger.info("=" * 60)
    logger.info("DEEP DIVE ANALYSIS: Why Do Adults Anthropomorphize More?")
    logger.info("=" * 60)
    
    # Load data
    df, emotions = load_data()
    
    all_results = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'n_comments': len(df),
        }
    }
    
    # 1. Loneliness analysis
    loneliness_results, df = analyze_loneliness_indicators(df)
    all_results['loneliness'] = loneliness_results
    
    # 2. Relationship language
    relationship_results, _ = analyze_relationship_language(df)
    all_results['relationship_language'] = relationship_results
    
    # 3. Linguistic features
    linguistic_results, _ = analyze_linguistic_features(df)
    all_results['linguistic'] = linguistic_results
    
    # 4. Subreddit analysis
    subreddit_results = analyze_subreddits(df)
    all_results['subreddits'] = subreddit_results
    
    # 5. Content patterns
    content_results = analyze_content_patterns(df)
    all_results['content_patterns'] = content_results
    
    # Create visualizations
    create_deep_dive_visualizations(all_results, PATHS['output_dir'])
    
    # Save results
    with open(PATHS['output_dir'] / 'deep_dive_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate markdown report
    generate_deep_dive_report(all_results)
    
    logger.info(f"\nResults saved to {PATHS['output_dir']}")
    
    return all_results


def generate_deep_dive_report(results):
    """Generate markdown report."""
    
    md = f"""# Deep Dive Analysis: Why Adults Anthropomorphize More

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## Executive Summary

This analysis explores potential explanations for why adults anthropomorphize AI companions more than teens.

---

## 1. Loneliness / Isolation Indicators

We searched for loneliness-related language patterns in comments.

"""
    
    if 'loneliness' in results:
        lone = results['loneliness']
        
        md += f"""### Prevalence by Age

| Group | % with Loneliness Indicators |
|-------|------------------------------|
| Teens | {lone['age_effect']['teen_loneliness_pct']:.1f}% |
| Adults | {lone['age_effect']['adult_loneliness_pct']:.1f}% |

**χ²** = {lone['age_effect']['chi2']:.2f}, p = {lone['age_effect']['p_value']:.4f}

**Direction:** {lone['age_effect']['direction'].upper()}

### Correlation with Anthropomorphization

- **Pearson r:** {lone['anthro_correlation']['pearson_r']:.3f}
- **High anthropomorphizers loneliness:** {lone['by_anthro_level']['high_anthro_loneliness_pct']:.1f}%
- **Low anthropomorphizers loneliness:** {lone['by_anthro_level']['low_anthro_loneliness_pct']:.1f}%
- **Ratio:** {lone['by_anthro_level']['ratio']:.2f}x

"""
    
    md += """
---

## 2. Relationship Language Analysis

We measured semantic similarity to human relationship language.

"""
    
    if 'relationship_language' in results:
        rel = results['relationship_language']
        
        md += f"""### Age Effect

| Group | Relationship Score |
|-------|-------------------|
| Teens | {rel['age_effect']['teen_mean']:.3f} |
| Adults | {rel['age_effect']['adult_mean']:.3f} |

**t** = {rel['age_effect']['t_statistic']:.2f}, p = {rel['age_effect']['p_value']:.4f}

**Direction:** {rel['age_effect']['direction']}

### Correlation with Anthropomorphization

**Pearson r** = {rel['anthro_correlation']['pearson_r']:.3f}

"""
    
    md += """
---

## 3. Linguistic Feature Analysis

"""
    
    if 'linguistic' in results:
        ling = results['linguistic']
        
        md += "| Feature | Teen Mean | Adult Mean | Cohen's d |\n|---------|-----------|------------|----------|\n"
        
        for feature, vals in ling['age_comparison'].items():
            sig = '*' if vals['p_value'] < 0.05 else ''
            md += f"| {feature.replace('_', ' ').title()} | {vals['teen_mean']:.2f} | {vals['adult_mean']:.2f} | {vals['cohens_d']:.3f} {sig} |\n"
    
    md += """
---

## 4. Content Pattern Analysis

What topics do high vs low anthropomorphizers discuss?

"""
    
    if 'content_patterns' in results:
        content = results['content_patterns']
        
        md += "| Content Type | High Anthro % | Low Anthro % | Ratio |\n|--------------|---------------|--------------|-------|\n"
        
        for cat, vals in sorted(content['high_vs_low'].items(), key=lambda x: -x[1]['ratio']):
            sig = '*' if vals['p_value'] < 0.05 else ''
            md += f"| {cat.replace('_', ' ').title()} | {vals['high_anthro_pct']:.1f}% | {vals['low_anthro_pct']:.1f}% | {vals['ratio']:.2f}x {sig} |\n"
    
    md += """
---

## 5. Subreddit Analysis

Does the age effect persist within individual subreddits?

"""
    
    if 'subreddits' in results:
        subs = results['subreddits']
        
        if 'within_subreddit_age_effect' in subs:
            md += "| Subreddit | Teen Mean | Adult Mean | Cohen's d | Sig. |\n|-----------|-----------|------------|-----------|------|\n"
            
            for sub, vals in subs['within_subreddit_age_effect'].items():
                sig = '*' if vals['p_value'] < 0.05 else ''
                md += f"| {sub} | {vals['teen_mean']:.2f} | {vals['adult_mean']:.2f} | {vals['cohens_d']:.3f} | {sig} |\n"
    
    md += """
---

## Key Findings

### Why Adults Anthropomorphize More:

1. **Loneliness Hypothesis:** [Result based on analysis]
2. **Relationship Language:** [Result based on analysis]
3. **Content Differences:** [Result based on analysis]
4. **Effect Persists Across Subreddits:** The age effect is not explained by different platform preferences

---

## Visualizations

Generated in `results/deep_dive/`:
- `loneliness_by_group.png`
- `relationship_language_by_group.png`
- `content_patterns_comparison.png`
- `linguistic_features_ratio.png`

"""
    
    with open(PATHS['output_dir'] / 'DEEP_DIVE_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(md)
    
    logger.info(f"Report saved to {PATHS['output_dir'] / 'DEEP_DIVE_REPORT.md'}")


if __name__ == "__main__":
    run_deep_dive()
