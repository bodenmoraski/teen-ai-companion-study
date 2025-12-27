"""
Complete pipeline execution with automatic progression.

This script runs all remaining phases and generates final analysis.
"""
import sys
import time
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_phase_complete(phase_num, check_paths):
    """Check if a phase is complete."""
    for path in check_paths:
        if Path(path).exists():
            return True
    return False


def wait_for_phase(phase_num, check_paths, max_wait_hours=6):
    """Wait for a phase to complete."""
    logger.info(f"Waiting for Phase {phase_num} to complete...")
    logger.info(f"Checking for: {check_paths}")
    
    max_wait_seconds = max_wait_hours * 3600
    check_interval = 60  # Check every minute
    waited = 0
    
    while waited < max_wait_seconds:
        if check_phase_complete(phase_num, check_paths):
            logger.info(f"✓ Phase {phase_num} complete!")
            return True
        
        if waited % 300 == 0:  # Log every 5 minutes
            logger.info(f"Phase {phase_num} still running... (waited {waited//60} minutes)")
        
        time.sleep(check_interval)
        waited += check_interval
    
    logger.warning(f"Phase {phase_num} not complete after {max_wait_hours} hours")
    return False


def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("Complete Pipeline Execution")
    logger.info("=" * 70)
    
    # Check current status
    logger.info("\nChecking current status...")
    
    # Phase 3
    phase3_paths = ["data/features/full_merged_dataset.parquet"]
    if not check_phase_complete(3, phase3_paths):
        logger.info("Phase 3 is running or not started. Waiting for completion...")
        if wait_for_phase(3, phase3_paths, max_wait_hours=6):
            logger.info("Phase 3 completed!")
        else:
            logger.error("Phase 3 did not complete. Please check manually.")
            return
    
    # Phase 4
    logger.info("\n" + "=" * 70)
    logger.info("Starting Phase 4: Statistical Analysis")
    logger.info("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/phase4_statistical_analysis.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Phase 4 completed successfully")
        logger.info(result.stdout[-500:])  # Last 500 chars
    except subprocess.CalledProcessError as e:
        logger.error(f"Phase 4 failed: {e}")
        logger.error(e.stderr)
        return
    
    # Generate comprehensive analysis
    logger.info("\n" + "=" * 70)
    logger.info("Generating Comprehensive Analysis Report")
    logger.info("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/analyze_results.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Analysis report generated")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Analysis generation failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info("=" * 70)
    logger.info("Check results/ directory for outputs")


if __name__ == "__main__":
    main()

