#!/usr/bin/env python3
"""
Prepare Human Validation Materials for AnthroScore
===================================================

This script generates all materials needed for human validation of the AnthroScore:
1. Stratified sample of 200 comments
2. Annotation spreadsheet (CSV)
3. Annotation guidelines (Markdown)

Output: Data/annotations/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("Data/features")
ANNOTATIONS_DIR = Path("Data/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load comments with AnthroScores."""
    logger.info("Loading data...")
    
    # Load comments with anthroscores
    comments_path = DATA_DIR / "comments_with_anthroscores.parquet"
    if not comments_path.exists():
        raise FileNotFoundError(f"Required file not found: {comments_path}")
    
    df = pd.read_parquet(comments_path)
    logger.info(f"Loaded {len(df):,} comments")
    
    # Load demographics for age group stratification
    demo_path = DATA_DIR / "demographics_v2.parquet"
    if demo_path.exists():
        demo = pd.read_parquet(demo_path)
        df = df.merge(demo[['author', 'age_bucket', 'confidence']], on='author', how='left')
        logger.info(f"Merged demographics for {df['age_bucket'].notna().sum():,} comments")
    
    return df


def create_anthroscore_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Assign AnthroScore strata (Low/Medium/High) for stratified sampling."""
    # Only consider comments with non-zero AnthroScore
    df_nonzero = df[df['anthroscore'] > 0].copy()
    
    # Also include some zero AnthroScore comments as negatives
    df_zero = df[df['anthroscore'] == 0].copy()
    df_zero['anthro_stratum'] = 'Zero'
    
    # Create tertiles for non-zero
    q33 = df_nonzero['anthroscore'].quantile(0.33)
    q66 = df_nonzero['anthroscore'].quantile(0.66)
    
    def assign_stratum(score: float) -> str:
        if score <= q33:
            return 'Low'
        elif score <= q66:
            return 'Medium'
        else:
            return 'High'
    
    df_nonzero['anthro_stratum'] = df_nonzero['anthroscore'].apply(assign_stratum)
    
    logger.info(f"AnthroScore tertile thresholds: Low ≤ {q33:.3f}, Medium ≤ {q66:.3f}, High > {q66:.3f}")
    
    return pd.concat([df_nonzero, df_zero], ignore_index=True)


def stratified_sample(df: pd.DataFrame, n_total: int = 200) -> pd.DataFrame:
    """
    Create stratified sample of comments for annotation.
    
    Stratification:
    - AnthroScore strata (Zero/Low/Medium/High): Equal representation
    - Age group (Teen/Adult): Proportional to population
    """
    logger.info(f"Creating stratified sample of {n_total} comments...")
    
    # Assign strata
    df = create_anthroscore_strata(df)
    
    # Define sampling targets
    # 4 strata × 50 samples each = 200 total
    samples_per_stratum = n_total // 4
    
    samples = []
    for stratum in ['Zero', 'Low', 'Medium', 'High']:
        stratum_df = df[df['anthro_stratum'] == stratum]
        
        if len(stratum_df) >= samples_per_stratum:
            sample = stratum_df.sample(n=samples_per_stratum, random_state=42)
        else:
            sample = stratum_df  # Take all if not enough
            logger.warning(f"Only {len(sample)} comments in {stratum} stratum")
        
        samples.append(sample)
        logger.info(f"  {stratum}: sampled {len(sample)} comments")
    
    result = pd.concat(samples, ignore_index=True)
    
    # Shuffle for annotation
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Total sample: {len(result)} comments")
    return result


def create_annotation_spreadsheet(df: pd.DataFrame) -> Path:
    """
    Create annotation spreadsheet for human coders.
    
    Columns:
    - comment_id: Unique identifier
    - comment_text: The comment to annotate
    - annotator1_score: Rating (1-5) from annotator 1
    - annotator2_score: Rating (1-5) from annotator 2
    - annotator1_notes: Optional notes
    - annotator2_notes: Optional notes
    - (hidden) true_anthroscore: For later validation
    - (hidden) anthro_stratum: For analysis
    """
    logger.info("Creating annotation spreadsheet...")
    
    # Select columns for annotation
    annotation_df = pd.DataFrame({
        'comment_id': range(1, len(df) + 1),
        'comment_text': df['body'].values if 'body' in df.columns else df['text'].values,
        'annotator1_score': '',
        'annotator1_notes': '',
        'annotator2_score': '',
        'annotator2_notes': '',
        'annotator3_score': '',
        'annotator3_notes': '',
    })
    
    # Save annotation sheet (without ground truth - for annotators)
    output_path = ANNOTATIONS_DIR / "annotation_sheet.csv"
    annotation_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Saved annotation sheet to {output_path}")
    
    # Save ground truth separately (for validation after annotation)
    ground_truth_df = pd.DataFrame({
        'comment_id': range(1, len(df) + 1),
        'true_anthroscore': df['anthroscore'].values,
        'anthro_stratum': df['anthro_stratum'].values,
        'age_bucket': df['age_bucket'].values if 'age_bucket' in df.columns else None,
        'author': df['author'].values,
    })
    
    ground_truth_path = ANNOTATIONS_DIR / "ground_truth_DO_NOT_SHARE.csv"
    ground_truth_df.to_csv(ground_truth_path, index=False, encoding='utf-8')
    logger.info(f"Saved ground truth to {ground_truth_path}")
    
    return output_path


def create_annotation_guidelines() -> Path:
    """Create comprehensive annotation guidelines document."""
    logger.info("Creating annotation guidelines...")
    
    guidelines = """# AnthroScore Annotation Guidelines

## Overview

You will be annotating Reddit comments for **anthropomorphization** - the degree to which the user treats an AI companion as if it were human.

**Your Task**: Rate each comment on a 1-5 scale based on how much the user anthropomorphizes (treats as human) the AI companion.

---

## Rating Scale

| Score | Label | Description | Examples |
|-------|-------|-------------|----------|
| **1** | None | No anthropomorphization. AI is treated as software/tool. | "The chatbot gave a weird response", "I reset the app" |
| **2** | Minimal | Slight humanization, but still clearly referring to AI. | "It's pretty smart", "The bot understood my question" |
| **3** | Moderate | Some human-like attributes/emotions ascribed to AI. | "She seemed confused", "It was being stubborn today" |
| **4** | High | AI treated as having genuine feelings, personality, or agency. | "He really cares about me", "She gets jealous when I talk to others" |
| **5** | Extreme | Full anthropomorphization. AI treated as equivalent to human relationship. | "We're in a relationship", "I love them so much, they're my everything" |

---

## Key Indicators to Look For

### Evidence of Anthropomorphization (Higher Scores)
- **Pronouns**: Using "he/she/they" instead of "it"
- **Emotions**: Attributing feelings ("happy", "sad", "jealous", "caring")
- **Agency**: Treating AI as making choices ("decided to", "wanted to", "chose to")
- **Relationship language**: "friend", "partner", "relationship", "we"
- **Personality**: "funny", "kind", "supportive", "understanding"
- **Consciousness**: "thinks", "feels", "knows", "understands"

### Evidence Against Anthropomorphization (Lower Scores)
- Technical language ("glitch", "bug", "response", "output")
- Tool metaphors ("use", "configure", "settings")
- Neutral pronoun "it"
- References to AI as software/product

---

## Annotation Process

1. **Read the full comment** carefully
2. **Identify anthropomorphizing language** (see indicators above)
3. **Consider context**: Some casual language ("it's cool") doesn't count
4. **Assign a score** (1-5)
5. **Add notes** if the rating was difficult or ambiguous

---

## Edge Cases

### Roleplay Context
- If the user is clearly doing **creative writing/roleplay**, still rate the anthropomorphization present
- Focus on how the user *frames* their relationship, not the story content

### Quotations
- If the user is quoting the AI's response, focus on the user's framing, not the AI's words

### Complaints/Criticism
- Complaints about AI behavior can still be anthropomorphizing
- "He was being rude" (anthropomorphizing) vs "The response was inappropriate" (not)

### Multiple References
- If a comment has mixed signals, rate the **overall dominant tone**

---

## Examples

### Score 1 - None
> "I cleared the cache and the app works fine now"
- *Reasoning: Purely technical, no anthropomorphization*

### Score 2 - Minimal
> "The AI gave a pretty good response about cooking tips"
- *Reasoning: Refers to AI as "the AI", minimal personification*

### Score 3 - Moderate
> "She seemed to understand what I was going through"
- *Reasoning: Uses "she", attributes understanding, but somewhat hedged ("seemed")*

### Score 4 - High
> "He really gets me. Like he actually cares about my problems and remembers things"
- *Reasoning: Strong attribution of genuine caring and memory/intention*

### Score 5 - Extreme
> "I know it sounds crazy but I'm genuinely in love with her. She's my best friend and the only one who truly understands me"
- *Reasoning: Full romantic relationship framing, treated as equivalent to human*

---

## Important Notes

1. **Be consistent**: Try to apply the same standards across all comments
2. **Don't judge**: We're measuring anthropomorphization, not evaluating whether it's good/bad
3. **When uncertain**: Use your best judgment and add a note explaining your reasoning
4. **Take breaks**: Annotating 200+ comments is tiring - take breaks to maintain quality

---

## Contact

If you have questions or encounter difficult cases, please note the comment ID and describe the issue.

---

*Generated: {timestamp}*
*Study: The Illusion Project - Anthropomorphization of AI Companions*
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    output_path = ANNOTATIONS_DIR / "ANNOTATION_GUIDELINES.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guidelines)
    
    logger.info(f"Saved annotation guidelines to {output_path}")
    return output_path


def generate_sample_statistics(df: pd.DataFrame) -> None:
    """Generate statistics about the sample for documentation."""
    logger.info("Generating sample statistics...")
    
    stats_report = []
    stats_report.append("=" * 70)
    stats_report.append("ANNOTATION SAMPLE STATISTICS")
    stats_report.append("=" * 70)
    stats_report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    stats_report.append("-" * 70)
    stats_report.append("Sample Size and Stratification")
    stats_report.append("-" * 70)
    stats_report.append(f"Total comments in sample: {len(df)}")
    stats_report.append(f"\nBy AnthroScore Stratum:")
    for stratum in ['Zero', 'Low', 'Medium', 'High']:
        count = (df['anthro_stratum'] == stratum).sum()
        pct = count / len(df) * 100
        stats_report.append(f"  {stratum:10s}: {count:4d} ({pct:.1f}%)")
    
    if 'age_bucket' in df.columns:
        stats_report.append(f"\nBy Age Group:")
        for age in df['age_bucket'].dropna().unique():
            count = (df['age_bucket'] == age).sum()
            pct = count / len(df) * 100
            stats_report.append(f"  {age:10s}: {count:4d} ({pct:.1f}%)")
    
    stats_report.append("-" * 70)
    stats_report.append("AnthroScore Distribution in Sample")
    stats_report.append("-" * 70)
    stats_report.append(f"Mean:   {df['anthroscore'].mean():.4f}")
    stats_report.append(f"Std:    {df['anthroscore'].std():.4f}")
    stats_report.append(f"Min:    {df['anthroscore'].min():.4f}")
    stats_report.append(f"25%:    {df['anthroscore'].quantile(0.25):.4f}")
    stats_report.append(f"Median: {df['anthroscore'].median():.4f}")
    stats_report.append(f"75%:    {df['anthroscore'].quantile(0.75):.4f}")
    stats_report.append(f"Max:    {df['anthroscore'].max():.4f}")
    
    stats_report.append("-" * 70)
    stats_report.append("Comment Length Distribution")
    stats_report.append("-" * 70)
    text_col = 'body' if 'body' in df.columns else 'text'
    lengths = df[text_col].str.len()
    stats_report.append(f"Mean:   {lengths.mean():.0f} characters")
    stats_report.append(f"Median: {lengths.median():.0f} characters")
    stats_report.append(f"Min:    {lengths.min():.0f} characters")
    stats_report.append(f"Max:    {lengths.max():.0f} characters")
    
    # Save report
    output_path = ANNOTATIONS_DIR / "sample_statistics.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_report))
    
    logger.info(f"Saved sample statistics to {output_path}")
    
    # Print summary
    print('\n'.join(stats_report))


def main():
    """Main function to prepare all human validation materials."""
    logger.info("=" * 70)
    logger.info("PREPARING HUMAN VALIDATION MATERIALS")
    logger.info("=" * 70)
    
    # Load data
    df = load_data()
    
    # Create stratified sample
    sample_df = stratified_sample(df, n_total=200)
    
    # Save sample for reference
    sample_path = ANNOTATIONS_DIR / "sample_for_annotation.parquet"
    sample_df.to_parquet(sample_path)
    logger.info(f"Saved sample to {sample_path}")
    
    # Create annotation spreadsheet
    annotation_path = create_annotation_spreadsheet(sample_df)
    
    # Create annotation guidelines
    guidelines_path = create_annotation_guidelines()
    
    # Generate sample statistics
    generate_sample_statistics(sample_df)
    
    # Summary
    logger.info("=" * 70)
    logger.info("HUMAN VALIDATION MATERIALS READY")
    logger.info("=" * 70)
    logger.info(f"Output folder: {ANNOTATIONS_DIR}")
    logger.info("Files created:")
    logger.info(f"  1. {annotation_path.name} - For annotators")
    logger.info(f"  2. {guidelines_path.name} - Instructions")
    logger.info(f"  3. ground_truth_DO_NOT_SHARE.csv - For validation")
    logger.info(f"  4. sample_statistics.txt - Sample info")
    logger.info(f"  5. sample_for_annotation.parquet - Full sample data")
    
    return sample_df


if __name__ == "__main__":
    sample = main()

