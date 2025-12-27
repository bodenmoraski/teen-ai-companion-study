"""
Cleanup and organize codebase for future AI/researchers.

This script:
1. Moves outdated scripts to archive/
2. Consolidates documentation
3. Organizes files properly
"""
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Create archive directory
archive_dir = Path("archive")
archive_dir.mkdir(exist_ok=True)
archive_scripts = archive_dir / "scripts"
archive_docs = archive_dir / "docs"
archive_scripts.mkdir(exist_ok=True)
archive_docs.mkdir(exist_ok=True)

# Files to archive (outdated/replaced)
scripts_to_archive = [
    "scripts/rerun_phase2_full.py",  # Already done
    "scripts/rerun_phase3_phase4.py",  # Replaced by targeted
    "scripts/phase2_demographics.py",  # Old version
    "scripts/phase3_core_analysis.py",  # Old version
    "scripts/phase4_statistical_analysis.py",  # Old version
    "scripts/phase1_quick_collection.py",  # One-time utility
    "scripts/restart_with_sample.py",  # One-time fix
    "scripts/restore_and_merge_data.py",  # One-time fix
    "scripts/diagnose_issue.py",  # One-time diagnostic
    "scripts/check_demographics_issue.py",  # One-time check
    "scripts/check_file_issue.py",  # One-time check
    "scripts/explain_numbers.py",  # One-time utility
    "scripts/check_what_we_have.py",  # One-time utility
    "scripts/final_data_assessment.py",  # One-time utility
    "scripts/comprehensive_status_check.py",  # Replaced by check_progress
    "scripts/verify_all_saved.py",  # One-time utility
    "scripts/verify_and_run.py",  # One-time utility
    "scripts/verify_data_safe.py",  # One-time utility
    "scripts/check_tomorrow.py",  # One-time utility
    "scripts/monitor_and_continue.py",  # One-time utility
    "scripts/monitor_api_collection.py",  # One-time utility
    "scripts/collect_user_subreddits.py",  # Integrated into arctic_shift
    "scripts/fix_and_test_api.py",  # One-time fix
    "scripts/analyze_results.py",  # Can be regenerated
]

docs_to_archive = [
    "FIXES_APPLIED.md",
    "COMMUNITY_EMBEDDINGS_FIX.md",
    "COMPONENT_STATUS.md",
    "VERIFICATION_SUMMARY.md",
    "MODEL_VERIFICATION.md",
    "MODEL_RECOMMENDATION.md",
    "API_KEY_STATUS.md",
    "ENV_SETUP.md",
    "OVERNIGHT_STATUS.md",
    "INSTALLATION_UPDATE.md",
    "CURRENT_WORK_STATUS.md",
    "RESEARCH_NUMBERS_BREAKDOWN.md",
    "CLEAR_BREAKDOWN.md",
    "WHAT_TO_CHECK.md",
    "PROMPT.md",
    "README-ANTRHOSCOREV2.md",
    "README-CHEWV2.md",
    "Artic_shift_API_docs.md",  # Reference, but can be in archive
]

def archive_file(file_path: str, target_dir: Path):
    """Archive a file by moving it."""
    source = Path(file_path)
    if source.exists():
        target = target_dir / source.name
        if target.exists():
            logger.warning(f"  Target exists, skipping: {target}")
        else:
            shutil.move(str(source), str(target))
            logger.info(f"  Archived: {source.name}")
    else:
        logger.debug(f"  Not found (may already be deleted): {source}")

def main():
    """Main cleanup function."""
    logger.info("=" * 70)
    logger.info("CLEANING UP CODEBASE")
    logger.info("=" * 70)
    
    logger.info("\nArchiving outdated scripts...")
    for script in scripts_to_archive:
        archive_file(script, archive_scripts)
    
    logger.info("\nArchiving outdated documentation...")
    for doc in docs_to_archive:
        archive_file(doc, archive_docs)
    
    logger.info("\n" + "=" * 70)
    logger.info("CLEANUP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nArchived files moved to:")
    logger.info(f"  - {archive_scripts}")
    logger.info(f"  - {archive_docs}")
    logger.info("\nMain codebase is now cleaner and more organized!")

if __name__ == "__main__":
    main()

