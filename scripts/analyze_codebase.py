"""Analyze codebase to identify what's current vs outdated."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from collections import defaultdict

print("=" * 70)
print("CODEBASE ANALYSIS FOR REFACTORING")
print("=" * 70)

# Analyze scripts
scripts_dir = Path("scripts")
scripts = list(scripts_dir.glob("*.py"))

print(f"\nSCRIPTS ANALYSIS ({len(scripts)} files):")
print("-" * 70)

# Categorize scripts
categories = {
    "MAIN PIPELINE": [],
    "RE-RUN/UPDATE": [],
    "TESTING": [],
    "UTILITY/CHECK": [],
    "OUTDATED": []
}

for script in scripts:
    name = script.name
    if name.startswith("phase") and name.endswith(".py"):
        if "rerun" in name or "targeted" in name:
            categories["RE-RUN/UPDATE"].append(name)
        else:
            categories["MAIN PIPELINE"].append(name)
    elif name.startswith("test_"):
        categories["TESTING"].append(name)
    elif name.startswith("check_") or name.startswith("verify_") or name.startswith("monitor_"):
        categories["UTILITY/CHECK"].append(name)
    elif name in ["run_all_phases.py", "complete_pipeline.py", "validate_demographics.py"]:
        categories["MAIN PIPELINE"].append(name)
    elif name in ["restart_with_sample.py", "restore_and_merge_data.py", "diagnose_issue.py", 
                  "explain_numbers.py", "check_what_we_have.py", "final_data_assessment.py",
                  "comprehensive_status_check.py", "check_demographics_issue.py", "check_file_issue.py"]:
        categories["UTILITY/CHECK"].append(name)
    elif name in ["rerun_phase2_full.py", "rerun_phase3_phase4.py", "targeted_phase3_phase4.py"]:
        categories["RE-RUN/UPDATE"].append(name)
    else:
        categories["OUTDATED"].append(name)

for cat, files in categories.items():
    if files:
        print(f"\n{cat}:")
        for f in sorted(files):
            print(f"  - {f}")

# Analyze markdown files
md_files = list(Path(".").glob("*.md"))
print(f"\n\nMARKDOWN FILES ({len(md_files)} files):")
print("-" * 70)

md_categories = {
    "CORE DOCS": ["README.md", "COMPREHENSIVE_RESEARCH_PLAN.md", "PLAN.md", "TODO.md"],
    "STATUS/REPORTS": ["FINAL_STATUS_REPORT.md", "PROJECT_STATUS.md", "CURRENT_WORK_STATUS.md", 
                      "NEURIPS_READINESS_ASSESSMENT.md", "RESEARCH_NUMBERS_BREAKDOWN.md", "CLEAR_BREAKDOWN.md"],
    "SETUP/GUIDES": ["SETUP.md", "EXECUTION_GUIDE.md", "QUICK_START.md", "WHAT_TO_CHECK.md"],
    "FIXES/NOTES": ["FIXES_APPLIED.md", "COMMUNITY_EMBEDDINGS_FIX.md", "METHODOLOGY_COMPLETE.md",
                   "IMPLEMENTATION_SUMMARY.md", "VERIFICATION_SUMMARY.md", "COMPONENT_STATUS.md",
                   "MODEL_VERIFICATION.md", "MODEL_RECOMMENDATION.md", "API_KEY_STATUS.md",
                   "ENV_SETUP.md", "OVERNIGHT_STATUS.md"],
    "OUTDATED": []
}

for md in md_files:
    name = md.name
    found = False
    for cat, files in md_categories.items():
        if name in files:
            found = True
            break
    if not found:
        md_categories["OUTDATED"].append(name)

for cat, files in md_categories.items():
    if files:
        print(f"\n{cat}:")
        for f in sorted(files):
            print(f"  - {f}")

# Check data files
print(f"\n\nDATA FILES:")
print("-" * 70)
data_features = Path("data/features")
if data_features.exists():
    parquet_files = list(data_features.glob("*.parquet"))
    print(f"Feature files: {len(parquet_files)}")
    for f in sorted(parquet_files):
        size = f.stat().st_size / (1024*1024)
        print(f"  - {f.name}: {size:.1f} MB")

print("\n" + "=" * 70)
print("RECOMMENDATIONS FOR CLEANUP:")
print("=" * 70)
print("\n1. KEEP (Main Pipeline):")
print("   - phase1_data_collection.py")
print("   - phase2_with_api_data.py (current version)")
print("   - targeted_phase3_phase4.py (current version)")
print("   - validate_demographics.py")
print("   - check_progress.py")
print("\n2. ARCHIVE/DELETE (Outdated):")
print("   - rerun_phase2_full.py (already done)")
print("   - rerun_phase3_phase4.py (replaced by targeted)")
print("   - phase2_demographics.py (old version)")
print("   - phase3_core_analysis.py (old version)")
print("   - phase4_statistical_analysis.py (old version)")
print("   - Many utility scripts that were one-time fixes")
print("\n3. CONSOLIDATE:")
print("   - Merge status reports into one comprehensive doc")
print("   - Keep only current methodology docs")

