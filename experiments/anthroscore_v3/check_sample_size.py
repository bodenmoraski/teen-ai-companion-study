"""Check how many users we actually need to score for research."""
import pandas as pd
from pathlib import Path

root = Path(__file__).parent.parent.parent

# Check current research sample sizes
print("="*70)
print("SAMPLE SIZE ANALYSIS")
print("="*70)

# Load comments
comments = pd.read_parquet(root / "Data/processed/all_comments.parquet")
print(f"\nTotal comments: {len(comments):,}")
print(f"Total unique users: {comments['author'].nunique():,}")

# Check what the research actually uses
# From FINAL_MASTER_RESEARCH_DOCUMENT.md: 27,846 users with both age and gender
research_users = 27_846
print(f"\nUsers used in research analyses: ~{research_users:,}")

# Estimate comments per user
avg_comments_per_user = len(comments) / comments['author'].nunique()
print(f"Average comments per user: {avg_comments_per_user:.1f}")

# Estimate total comments needed for research users
estimated_comments_needed = research_users * avg_comments_per_user
print(f"\nEstimated comments for {research_users:,} users: ~{estimated_comments_needed:,.0f}")
print(f"This is {100*estimated_comments_needed/len(comments):.1f}% of total comments")

# Cost estimate
from anthroscore_llm import AnthroScoreLLM
model = AnthroScoreLLM.MODELS.get('gpt-4.1-nano', {})
avg_input_tokens = 400
avg_output_tokens = 50
cost_per_comment = ((avg_input_tokens / 1_000_000) * model.get('input', 0.10)) + \
                   ((avg_output_tokens / 1_000_000) * model.get('output', 0.40))
total_cost = estimated_comments_needed * cost_per_comment

print(f"\nCost estimate for {research_users:,} users (~{estimated_comments_needed:,.0f} comments):")
print(f"  Per comment: ~${cost_per_comment:.6f}")
print(f"  Total: ~${total_cost:.2f}")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"""
For research purposes, you need scores for ~{research_users:,} users.

Options:
1. Score all comments from these {research_users:,} users (~{estimated_comments_needed:,.0f} comments, ~${total_cost:.2f})
2. Score 1 comment per user ({research_users:,} comments, ~${research_users * cost_per_comment:.2f})
3. Sample for validation (e.g., 5,000-10,000 comments, ~${5000 * cost_per_comment:.2f} - ${10000 * cost_per_comment:.2f})

RECOMMENDATION: Option 2 (1 comment per user)
- Research analyses are at USER level, not comment level
- One representative comment per user should be sufficient
- Much cheaper (~${research_users * cost_per_comment:.2f} vs ~${total_cost:.2f})
- Faster (~{research_users * 0.5 / 3600:.1f} hours vs ~{estimated_comments_needed * 0.5 / 3600:.1f} hours)
""")
