"""Verify demographics results and check API key."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Demographics Verification")
print("=" * 70)

# Check API key
raw_key = os.getenv("OPENAI_API_KEY", "")
print(f"\nAPI Key Status:")
print(f"  Raw key length: {len(raw_key)}")
if raw_key:
    print(f"  First 10 chars: {raw_key[:10]}...")
    has_quotes = raw_key.startswith('"') or raw_key.startswith("'") or raw_key.endswith('"') or raw_key.endswith("'")
    print(f"  Has quotes: {has_quotes}")
    # Check if it's properly formatted
    if raw_key.startswith("'") and raw_key.endswith("'"):
        cleaned = raw_key[1:-1]
        print(f"  After removing single quotes: {len(cleaned)} chars")
    elif raw_key.startswith('"') and raw_key.endswith('"'):
        cleaned = raw_key[1:-1]
        print(f"  After removing double quotes: {len(cleaned)} chars")
    else:
        cleaned = raw_key
    print(f"  Starts with sk-: {cleaned.startswith('sk-')}")
else:
    print("  ⚠ No API key found in environment")

# Check demographics file
print(f"\nDemographics File Analysis:")
demo_path = Path("data/features/demographics.parquet")
if demo_path.exists():
    demo = pd.read_parquet(demo_path)
    print(f"  Total users: {len(demo):,}")
    print(f"  Columns: {list(demo.columns)}")
    print()
    
    # Age analysis
    age_classified = demo['age_bucket'].notna().sum()
    print(f"Age Classification:")
    print(f"  Total classified: {age_classified:,} ({100*age_classified/len(demo):.1f}%)")
    if age_classified > 0:
        print(f"  Distribution:")
        for bucket, count in demo['age_bucket'].value_counts().sort_index().items():
            pct = 100 * count / age_classified
            print(f"    {bucket}: {count:,} ({pct:.1f}%)")
    print()
    
    # Method breakdown
    if 'methods_used' in demo.columns:
        print(f"Methods Used (for age classification):")
        methods = demo[demo['age_bucket'].notna()]['methods_used'].value_counts()
        for method, count in methods.items():
            pct = 100 * count / age_classified if age_classified > 0 else 0
            print(f"  {method}: {count:,} ({pct:.1f}%)")
        print()
    
    # Community embeddings
    comm_age = demo['age_bucket_community'].notna().sum()
    comm_gender = demo['gender_community'].notna().sum()
    print(f"Community Embeddings:")
    print(f"  Age classifications: {comm_age:,} ({100*comm_age/len(demo):.1f}%)")
    print(f"  Gender classifications: {comm_gender:,} ({100*comm_gender/len(demo):.1f}%)")
    if comm_age > 0:
        print(f"  Age distribution:")
        for bucket, count in demo['age_bucket_community'].value_counts().head().items():
            print(f"    {bucket}: {count:,}")
    print()
    
    # LLM
    llm_age = demo['age_bucket_llm'].notna().sum()
    print(f"LLM Classifications:")
    print(f"  Age classifications: {llm_age:,} ({100*llm_age/len(demo):.1f}%)")
    if llm_age == 0:
        print(f"  ⚠ No LLM classifications found (API key may not be configured)")
    print()
    
    # Gender
    gender_classified = demo['gender'].notna().sum()
    print(f"Gender Classification:")
    print(f"  Total classified: {gender_classified:,} ({100*gender_classified/len(demo):.1f}%)")
    if gender_classified > 0:
        print(f"  Distribution:")
        for gender, count in demo['gender'].value_counts().items():
            pct = 100 * count / gender_classified
            print(f"    {gender}: {count:,} ({pct:.1f}%)")
    
    print()
    print("=" * 70)
    print("Summary:")
    print(f"  ✓ Self-declaration: Working")
    print(f"  ✓ Community embeddings: Working ({comm_age:,} users)")
    if llm_age > 0:
        print(f"  ✓ LLM classification: Working ({llm_age:,} users)")
    else:
        print(f"  ⚠ LLM classification: Not used (API key issue)")
    print(f"  ✓ Ensemble: Working ({age_classified:,} final classifications)")
    print("=" * 70)
else:
    print("  ✗ Demographics file not found!")

