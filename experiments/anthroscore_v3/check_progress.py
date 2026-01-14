"""Quick script to check processing progress."""
import pandas as pd
from pathlib import Path

checkpoint = Path(__file__).parent / "anthroscore_v3_checkpoint.parquet"
full = Path(__file__).parent / "anthroscore_v3_full.parquet"

for path in [full, checkpoint]:
    if path.exists():
        df = pd.read_parquet(path)
        print(f"File: {path.name}")
        print(f"  Total: {len(df):,}")
        print(f"  Valid (score 1-5): {sum(df['score'] > 0):,}")
        print(f"  Errors (score 0): {sum(df['score'] == 0):,}")
        print(f"  Pre-filtered: {sum(df['source'] == 'prefilter'):,}")
        print(f"\n  Score distribution:")
        print(df['score'].value_counts().sort_index())
        break
else:
    print("No checkpoint or results file found yet")
