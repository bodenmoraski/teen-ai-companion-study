"""Check current state of expert labels."""
import pandas as pd
from pathlib import Path

exp_dir = Path(__file__).parent
file_path = exp_dir / "test_set_expert_labeled.parquet"

if not file_path.exists():
    print(f"File not found: {file_path}")
else:
    df = pd.read_parquet(file_path)
    print(f"Total rows: {len(df)}")
    valid = sum(df['expert_score'] > 0)
    failed = sum(df['expert_score'] == 0)
    print(f"Valid expert scores (1-5): {valid}")
    print(f"Failed (score=0): {failed}")
    print()
    print("Score distribution:")
    print(df['expert_score'].value_counts().sort_index())
    
    # Check last modified time
    import os
    mtime = os.path.getmtime(file_path)
    from datetime import datetime
    print(f"\nFile last modified: {datetime.fromtimestamp(mtime)}")
