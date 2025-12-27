"""Check progress of Phase 3 & 4 re-run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

print("=" * 70)
print("PHASE 3 & 4 PROGRESS CHECK")
print("=" * 70)
print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check log file
log_path = Path("targeted_phase3_phase4.log")
if log_path.exists():
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        if lines:
            print("Last log entries:")
            print("-" * 70)
            for line in lines[-10:]:
                print(line.rstrip())
            print("-" * 70)
            
            # Check for completion
            if any("Complete" in line or "complete" in line for line in lines[-20:]):
                print("\n[STATUS: COMPLETE!]")
            elif any("Error" in line or "error" in line for line in lines[-20:]):
                print("\n[STATUS: ERROR DETECTED - Check log above]")
            else:
                print("\n[STATUS: IN PROGRESS]")
        else:
            print("Log file is empty")
else:
    print("⚠️  Log file not found - script may not have started")

# Check output files
print("\n" + "=" * 70)
print("OUTPUT FILES CHECK")
print("=" * 70)

files_to_check = {
    "Merged Dataset": Path("data/features/full_merged_dataset.parquet"),
    "Regression Results": Path("results/tables/regression_results.txt"),
    "Descriptive Stats": Path("results/tables/descriptive_statistics.txt"),
    "Age Distribution Plot": Path("results/figures/age_distribution.png"),
    "AnthroScore Plot": Path("results/figures/anthroscore_by_demographics.png"),
}

all_complete = True
for name, path in files_to_check.items():
    if path.exists():
        if path.suffix == '.parquet':
            try:
                df = pd.read_parquet(path)
                size = path.stat().st_size / (1024*1024)  # MB
                print(f"[OK] {name}: {len(df):,} rows ({size:.1f} MB)")
            except:
                print(f"[WARNING] {name}: File exists but can't read")
        else:
            size = path.stat().st_size / 1024  # KB
            print(f"[OK] {name}: {size:.1f} KB")
    else:
        print(f"[MISSING] {name}: NOT FOUND")
        all_complete = False

# Check merged dataset details
merged_path = Path("data/features/full_merged_dataset.parquet")
if merged_path.exists():
    try:
        df = pd.read_parquet(merged_path)
        print("\n" + "=" * 70)
        print("MERGED DATASET DETAILS")
        print("=" * 70)
        print(f"Total users: {len(df):,}")
        print(f"Age classified: {df['age_bucket'].notna().sum():,}")
        print(f"AnthroScore: {df['anthroscore_mean'].notna().sum():,}")
        print(f"Users with BOTH (for analysis): {(df['age_bucket'].notna() & df['anthroscore_mean'].notna()).sum():,}")
        
        if 'dominant_topic' in df.columns:
            print(f"Topics: {df['dominant_topic'].notna().sum():,}")
        if 'dominant_emotion' in df.columns:
            print(f"Emotions: {df['dominant_emotion'].notna().sum():,}")
    except Exception as e:
        print(f"\n⚠️  Could not read merged dataset: {e}")

# Final status
print("\n" + "=" * 70)
if all_complete:
    print("[ALL OUTPUTS GENERATED - READY FOR ANALYSIS!]")
else:
    print("[STILL IN PROGRESS - Some outputs missing]")
print("=" * 70)

print("\nNEXT STEPS (when complete):")
print("1. Run validation: python scripts/validate_demographics.py")
print("2. Review results in results/tables/")
print("3. Check figures in results/figures/")
print("4. Run method comparison analysis")
