"""Quick script to check current state for criticism update."""
import pandas as pd

demo = pd.read_parquet('data/features/demographics.parquet')

print("=== CURRENT STATE CHECK ===")
print(f"\nTotal users: {len(demo):,}")

print("\n--- Age Classification ---")
print("Age bucket distribution (final):")
print(demo['age_bucket'].value_counts())
print(f"\nAge classified: {demo['age_bucket'].notna().sum():,} ({demo['age_bucket'].notna().sum()/len(demo)*100:.1f}%)")

print("\n--- Gender Classification ---")
print("Gender distribution (final):")
print(demo['gender'].value_counts())
print(f"\nGender classified (non-unknown): {(demo['gender'] != 'unknown').sum():,}")

if 'gender_community' in demo.columns:
    print("\nGender community classifications:")
    print(demo['gender_community'].value_counts())
    print(f"Overlap with self-declared: {(demo['gender_community'].notna() & demo['gender_self_declared'].notna()).sum():,}")

print("\n--- 3-Bucket Age ---")
if 'age_3bucket' in demo.columns:
    print("3-bucket age available in demographics")
else:
    print("3-bucket age NOT in demographics (created on-the-fly in neurips_analysis)")

