#!/usr/bin/env python3
"""
COMPREHENSIVE INTENT/PURPOSE ANALYSIS

This script:
1. Runs BERTopic to identify user intent/purpose topics
2. Labels topics semantically (emotional support, roleplay, curiosity, etc.)
3. Correlates intent with demographics (age, gender)
4. Correlates intent with anthropomorphization
5. Runs full statistical analysis with all variables
6. Generates comprehensive report

This completes RQ1's "intent and purpose" requirement.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("intent_analysis.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def run_bertopic_analysis(comments_df: pd.DataFrame, sample_size: int = 50000):
    """Run BERTopic on comments to extract topics."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    
    logger.info("=" * 60)
    logger.info("RUNNING BERTOPIC FOR INTENT/PURPOSE EXTRACTION")
    logger.info("=" * 60)
    
    # Sample if too many comments (for speed)
    if len(comments_df) > sample_size:
        logger.info(f"Sampling {sample_size} comments from {len(comments_df)}")
        sample_df = comments_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = comments_df
    
    # Filter valid texts
    texts = sample_df['body'].dropna().astype(str).tolist()
    texts = [t for t in texts if len(t) > 20]  # Minimum length
    logger.info(f"Processing {len(texts)} valid comments")
    
    # Load embedding model (reuse the one we've been using)
    logger.info("Loading sentence transformer...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        low_memory=True
    )
    
    # Configure HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=100,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Initialize BERTopic
    logger.info("Initializing BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=100,
        nr_topics='auto',
        calculate_probabilities=True,
        verbose=True
    )
    
    # Fit model
    logger.info("Fitting BERTopic model (this may take a while)...")
    topics, probs = topic_model.fit_transform(texts)
    
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info(f"Found {n_topics} topics")
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    logger.info("\nTop Topics:")
    for _, row in topic_info.head(15).iterrows():
        logger.info(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    
    return topic_model, topics, probs, sample_df, texts


def label_topics_semantically(topic_model):
    """
    Label topics with semantic intent categories.
    
    Categories based on AI companion research:
    - Emotional Support: loneliness, depression, therapy, help, sad
    - Roleplay/Fantasy: character, roleplay, story, adventure, scenario
    - Romantic/Sexual: love, relationship, girlfriend, boyfriend, dating
    - Curiosity/Experimentation: trying, test, curious, experiment, new
    - Technical/Issues: bug, broken, not working, update, fix
    - Character Creation: create, make, design, custom, persona
    - Community/Social: share, post, anyone, people, community
    """
    
    intent_keywords = {
        'emotional_support': ['lonely', 'loneliness', 'depressed', 'depression', 'sad', 'help', 
                              'therapy', 'anxious', 'anxiety', 'mental', 'feeling', 'comfort',
                              'supportive', 'understand', 'listen', 'care', 'friend', 'alone'],
        'roleplay_fantasy': ['roleplay', 'rp', 'character', 'story', 'adventure', 'scenario',
                             'play', 'pretend', 'fantasy', 'imagine', 'world', 'setting'],
        'romantic_attachment': ['love', 'girlfriend', 'boyfriend', 'relationship', 'dating',
                                'romance', 'romantic', 'crush', 'marry', 'wife', 'husband',
                                'heart', 'feelings', 'attracted', 'kiss'],
        'curiosity_experimentation': ['try', 'trying', 'test', 'curious', 'experiment', 'new',
                                       'first', 'wonder', 'interesting', 'explore', 'see'],
        'technical_issues': ['bug', 'broken', 'work', 'working', 'update', 'fix', 'error',
                             'issue', 'problem', 'glitch', 'server', 'loading'],
        'character_creation': ['create', 'make', 'design', 'custom', 'persona', 'bot',
                               'definition', 'personality', 'traits', 'build'],
        'community_sharing': ['share', 'post', 'anyone', 'people', 'community', 'think',
                              'opinion', 'thoughts', 'discuss', 'conversation']
    }
    
    topic_labels = {}
    topic_info = topic_model.get_topic_info()
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            topic_labels[topic_id] = 'noise'
            continue
            
        # Get topic keywords
        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            topic_labels[topic_id] = 'unknown'
            continue
            
        topic_word_list = [word for word, _ in topic_words[:20]]
        topic_text = ' '.join(topic_word_list).lower()
        
        # Score each intent category
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for kw in keywords if kw in topic_text)
            intent_scores[intent] = score
        
        # Assign to highest scoring intent (or 'other' if no matches)
        if max(intent_scores.values()) > 0:
            best_intent = max(intent_scores, key=intent_scores.get)
            topic_labels[topic_id] = best_intent
        else:
            topic_labels[topic_id] = 'other'
    
    return topic_labels


def assign_user_intents(comments_df, topics, texts, topic_labels, topic_model):
    """Assign intent labels to users based on their dominant topics."""
    
    # Create mapping from text to topic
    text_to_topic = dict(zip(texts, topics))
    
    # Assign topics to original comments
    comments_df = comments_df.copy()
    comments_df['topic'] = comments_df['body'].map(
        lambda x: text_to_topic.get(str(x)[:500], -1) if pd.notna(x) else -1
    )
    comments_df['intent'] = comments_df['topic'].map(topic_labels)
    
    # Aggregate to user level - get dominant intent per user
    user_intents = []
    for author, group in comments_df.groupby('author'):
        intent_counts = group['intent'].value_counts()
        
        # Exclude noise
        intent_counts = intent_counts[intent_counts.index != 'noise']
        
        if len(intent_counts) > 0:
            dominant_intent = intent_counts.index[0]
            intent_proportion = intent_counts.iloc[0] / len(group)
        else:
            dominant_intent = 'unknown'
            intent_proportion = 0
        
        # Get proportions for each intent
        intent_props = {}
        for intent in ['emotional_support', 'roleplay_fantasy', 'romantic_attachment',
                       'curiosity_experimentation', 'technical_issues', 
                       'character_creation', 'community_sharing', 'other']:
            count = group[group['intent'] == intent].shape[0]
            intent_props[f'intent_{intent}_prop'] = count / len(group)
        
        user_intents.append({
            'author': author,
            'dominant_intent': dominant_intent,
            'dominant_intent_proportion': intent_proportion,
            **intent_props
        })
    
    return pd.DataFrame(user_intents)


def run_intent_correlations(user_intents_df, age_preds, gender_preds, anthro_df):
    """Run comprehensive correlations between intent and all variables."""
    
    logger.info("=" * 60)
    logger.info("RUNNING INTENT CORRELATIONS")
    logger.info("=" * 60)
    
    # Merge all data
    age_preds = age_preds.rename(columns={'confidence': 'age_confidence'})
    gender_preds = gender_preds.rename(columns={'confidence': 'gender_confidence'})
    
    merged = user_intents_df.merge(age_preds, on='author', how='inner')
    merged = merged.merge(gender_preds[['author', 'gender_predicted', 'gender_confidence']], 
                          on='author', how='inner')
    merged = merged.merge(anthro_df, on='author', how='inner')
    
    # Filter to high confidence
    data = merged[(merged['age_confidence'] >= 0.6) & 
                  (merged['gender_confidence'] >= 0.6) &
                  (merged['anthroscore_mean'] != 0)]
    
    logger.info(f"Sample size for correlations: {len(data)}")
    
    results = {}
    
    # 1. Intent distribution
    logger.info("\n=== INTENT DISTRIBUTION ===")
    intent_dist = data['dominant_intent'].value_counts()
    logger.info(f"\n{intent_dist}")
    results['intent_distribution'] = intent_dist.to_dict()
    
    # 2. Intent by Age
    logger.info("\n=== INTENT BY AGE ===")
    data['is_teen'] = data['age_bucket_predicted'] == 'teen'
    
    intent_by_age = pd.crosstab(data['dominant_intent'], data['is_teen'], normalize='columns')
    intent_by_age.columns = ['Adult', 'Teen']
    logger.info(f"\nIntent proportions by age:\n{intent_by_age}")
    
    # Chi-square test
    contingency = pd.crosstab(data['dominant_intent'], data['is_teen'])
    chi2, p_age, dof, expected = stats.chi2_contingency(contingency)
    logger.info(f"\nChi-square test (Intent vs Age): chi2={chi2:.2f}, p={p_age:.4f}")
    results['intent_age_chi2'] = {'chi2': chi2, 'p': p_age}
    
    # 3. Intent by Gender
    logger.info("\n=== INTENT BY GENDER ===")
    intent_by_gender = pd.crosstab(data['dominant_intent'], data['gender_predicted'], normalize='columns')
    logger.info(f"\nIntent proportions by gender:\n{intent_by_gender}")
    
    contingency = pd.crosstab(data['dominant_intent'], data['gender_predicted'])
    chi2, p_gender, dof, expected = stats.chi2_contingency(contingency)
    logger.info(f"\nChi-square test (Intent vs Gender): chi2={chi2:.2f}, p={p_gender:.4f}")
    results['intent_gender_chi2'] = {'chi2': chi2, 'p': p_gender}
    
    # 4. AnthroScore by Intent
    logger.info("\n=== ANTHROSCORE BY INTENT ===")
    anthro_by_intent = data.groupby('dominant_intent')['anthroscore_max'].agg(['mean', 'std', 'count'])
    anthro_by_intent = anthro_by_intent.sort_values('mean', ascending=False)
    logger.info(f"\nMax AnthroScore by intent:\n{anthro_by_intent}")
    
    # ANOVA
    groups = [group['anthroscore_max'].values for name, group in data.groupby('dominant_intent')]
    f_stat, p_anova = stats.f_oneway(*groups)
    logger.info(f"\nANOVA (AnthroScore vs Intent): F={f_stat:.2f}, p={p_anova:.4f}")
    results['anthro_intent_anova'] = {'F': f_stat, 'p': p_anova}
    
    # 5. Key finding: which intent has highest anthropomorphization?
    logger.info("\n=== KEY FINDINGS ===")
    top_intent = anthro_by_intent.index[0]
    top_mean = anthro_by_intent.loc[top_intent, 'mean']
    logger.info(f"Highest anthropomorphizing intent: {top_intent} (mean={top_mean:.3f})")
    
    # Compare romantic_attachment vs others
    if 'romantic_attachment' in data['dominant_intent'].values:
        romantic = data[data['dominant_intent'] == 'romantic_attachment']['anthroscore_max']
        other = data[data['dominant_intent'] != 'romantic_attachment']['anthroscore_max']
        t, p = stats.ttest_ind(romantic, other)
        logger.info(f"Romantic vs Other intents: t={t:.3f}, p={p:.4f}")
        results['romantic_vs_other'] = {'t': t, 'p': p, 
                                        'romantic_mean': romantic.mean(),
                                        'other_mean': other.mean()}
    
    # Compare emotional_support vs others
    if 'emotional_support' in data['dominant_intent'].values:
        emotional = data[data['dominant_intent'] == 'emotional_support']['anthroscore_max']
        other = data[data['dominant_intent'] != 'emotional_support']['anthroscore_max']
        t, p = stats.ttest_ind(emotional, other)
        logger.info(f"Emotional Support vs Other: t={t:.3f}, p={p:.4f}")
        results['emotional_vs_other'] = {'t': t, 'p': p,
                                         'emotional_mean': emotional.mean(),
                                         'other_mean': other.mean()}
    
    # 6. Full regression with all predictors
    logger.info("\n=== FULL REGRESSION MODEL ===")
    
    # Create dummy variables for intent
    intent_dummies = pd.get_dummies(data['dominant_intent'], prefix='intent')
    data = pd.concat([data, intent_dummies], axis=1)
    
    # Binary variables
    data['is_teen_int'] = data['is_teen'].astype(int)
    data['is_female'] = (data['gender_predicted'] == 'female').astype(int)
    
    # Get intent columns
    intent_cols = [c for c in data.columns if c.startswith('intent_') and not c.endswith('_prop')]
    
    # Full model formula
    formula = 'anthroscore_max ~ is_teen_int + is_female'
    for col in intent_cols[:5]:  # Top 5 intents
        if col in data.columns:
            formula += f' + {col}'
    
    try:
        model = smf.ols(formula, data=data).fit()
        logger.info(f"\nRegression Results:")
        logger.info(f"R-squared: {model.rsquared:.4f}")
        logger.info("\nSignificant predictors (p < 0.05):")
        for var, p in model.pvalues.items():
            if p < 0.05 and var != 'Intercept':
                coef = model.params[var]
                logger.info(f"  {var}: coef={coef:.4f}, p={p:.4f}")
        results['regression'] = {
            'r_squared': model.rsquared,
            'significant_vars': {k: {'coef': model.params[k], 'p': v} 
                                for k, v in model.pvalues.items() if v < 0.05}
        }
    except Exception as e:
        logger.error(f"Regression failed: {e}")
    
    # 7. Interaction: Intent × Age
    logger.info("\n=== INTENT × AGE INTERACTION ===")
    for intent in ['emotional_support', 'romantic_attachment', 'roleplay_fantasy']:
        if intent not in data['dominant_intent'].values:
            continue
        subset = data[data['dominant_intent'] == intent]
        teen = subset[subset['is_teen']]['anthroscore_max']
        adult = subset[~subset['is_teen']]['anthroscore_max']
        if len(teen) >= 10 and len(adult) >= 10:
            t, p = stats.ttest_ind(teen, adult)
            logger.info(f"{intent}: Teen({len(teen)})={teen.mean():.3f} vs Adult({len(adult)})={adult.mean():.3f}, p={p:.4f}")
    
    return results, data


def generate_comprehensive_report(results, output_path):
    """Generate comprehensive analysis report."""
    
    lines = [
        "=" * 70,
        "INTENT/PURPOSE ANALYSIS - COMPREHENSIVE REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This analysis completes RQ1's 'intent and purpose' requirement.",
        "",
        "=" * 70,
        "METHODOLOGY",
        "=" * 70,
        "",
        "1. BERTopic clustering on Reddit comments",
        "2. Semantic labeling of topics into intent categories:",
        "   - emotional_support: Users seeking comfort, help with loneliness",
        "   - romantic_attachment: Users forming romantic bonds with AI",
        "   - roleplay_fantasy: Users engaged in creative roleplay",
        "   - curiosity_experimentation: Users testing/exploring AI",
        "   - technical_issues: Users discussing bugs/problems",
        "   - character_creation: Users creating/customizing AI personas",
        "   - community_sharing: Users sharing experiences",
        "",
        "=" * 70,
        "KEY FINDINGS",
        "=" * 70,
        "",
    ]
    
    # Add intent distribution
    if 'intent_distribution' in results:
        lines.append("INTENT DISTRIBUTION:")
        lines.append("-" * 40)
        for intent, count in results['intent_distribution'].items():
            lines.append(f"  {intent}: {count}")
        lines.append("")
    
    # Add statistical tests
    if 'intent_age_chi2' in results:
        r = results['intent_age_chi2']
        sig = "SIGNIFICANT" if r['p'] < 0.05 else "not significant"
        lines.append(f"Intent vs Age: chi2={r['chi2']:.2f}, p={r['p']:.4f} ({sig})")
    
    if 'intent_gender_chi2' in results:
        r = results['intent_gender_chi2']
        sig = "SIGNIFICANT" if r['p'] < 0.05 else "not significant"
        lines.append(f"Intent vs Gender: chi2={r['chi2']:.2f}, p={r['p']:.4f} ({sig})")
    
    if 'anthro_intent_anova' in results:
        r = results['anthro_intent_anova']
        sig = "SIGNIFICANT" if r['p'] < 0.05 else "not significant"
        lines.append(f"AnthroScore vs Intent: F={r['F']:.2f}, p={r['p']:.4f} ({sig})")
    
    lines.append("")
    
    # Add key comparisons
    if 'romantic_vs_other' in results:
        r = results['romantic_vs_other']
        lines.append(f"Romantic Attachment vs Other Intents:")
        lines.append(f"  Romantic mean: {r['romantic_mean']:.3f}")
        lines.append(f"  Other mean: {r['other_mean']:.3f}")
        lines.append(f"  t={r['t']:.3f}, p={r['p']:.4f}")
        lines.append("")
    
    if 'emotional_vs_other' in results:
        r = results['emotional_vs_other']
        lines.append(f"Emotional Support vs Other Intents:")
        lines.append(f"  Emotional mean: {r['emotional_mean']:.3f}")
        lines.append(f"  Other mean: {r['other_mean']:.3f}")
        lines.append(f"  t={r['t']:.3f}, p={r['p']:.4f}")
        lines.append("")
    
    # Regression results
    if 'regression' in results:
        lines.append("MULTIVARIATE REGRESSION:")
        lines.append("-" * 40)
        lines.append(f"R-squared: {results['regression']['r_squared']:.4f}")
        lines.append("Significant predictors:")
        for var, vals in results['regression']['significant_vars'].items():
            if var != 'Intercept':
                lines.append(f"  {var}: coef={vals['coef']:.4f}, p={vals['p']:.4f}")
    
    report = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def main():
    logger.info("=" * 70)
    logger.info("STARTING COMPREHENSIVE INTENT/PURPOSE ANALYSIS")
    logger.info("=" * 70)
    
    # Paths
    data_path = Path("Data")
    output_path = data_path / "features"
    results_path = Path("results")
    
    # Load data
    logger.info("\n[Step 1] Loading data...")
    comments_df = pd.read_parquet(data_path / "processed/all_comments.parquet")
    logger.info(f"  Loaded {len(comments_df)} comments")
    
    age_preds = pd.read_parquet(data_path / "features/ultimate_predictor/ultimate_predictions.parquet")
    gender_preds = pd.read_parquet(data_path / "features/ultimate_predictor/gender_predictions.parquet")
    anthro_df = pd.read_parquet(data_path / "features/user_anthroscores.parquet")
    
    # Check if BERTopic results already exist
    bertopic_cache = output_path / "intent_topics.parquet"
    
    if bertopic_cache.exists():
        logger.info("\n[Step 2] Loading cached BERTopic results...")
        user_intents_df = pd.read_parquet(bertopic_cache)
    else:
        # Run BERTopic
        logger.info("\n[Step 2] Running BERTopic analysis...")
        topic_model, topics, probs, sample_df, texts = run_bertopic_analysis(
            comments_df, sample_size=50000
        )
        
        # Label topics
        logger.info("\n[Step 3] Labeling topics semantically...")
        topic_labels = label_topics_semantically(topic_model)
        logger.info(f"Topic labels: {topic_labels}")
        
        # Assign to users
        logger.info("\n[Step 4] Assigning intents to users...")
        user_intents_df = assign_user_intents(sample_df, topics, texts, topic_labels, topic_model)
        
        # Save
        user_intents_df.to_parquet(bertopic_cache)
        logger.info(f"  Saved intent data to {bertopic_cache}")
    
    logger.info(f"  Users with intent data: {len(user_intents_df)}")
    logger.info(f"  Intent distribution:\n{user_intents_df['dominant_intent'].value_counts()}")
    
    # Run correlations
    logger.info("\n[Step 5] Running correlations...")
    results, merged_data = run_intent_correlations(
        user_intents_df, age_preds, gender_preds, anthro_df
    )
    
    # Save merged data
    merged_data.to_parquet(output_path / "full_analysis_data.parquet")
    
    # Generate report
    logger.info("\n[Step 6] Generating report...")
    report = generate_comprehensive_report(results, results_path / "intent_analysis_report.txt")
    print("\n" + report)
    
    logger.info("\n" + "=" * 70)
    logger.info("INTENT ANALYSIS COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

