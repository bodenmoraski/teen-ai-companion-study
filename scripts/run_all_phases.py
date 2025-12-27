"""
Master script to run all phases of the research pipeline.

This script orchestrates the complete research pipeline from data collection
through final statistical analysis.
"""
import logging
import sys
from pathlib import Path

# Add scripts to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_phase(script_name: str, phase_name: str) -> bool:
    """
    Run a phase script.
    
    Args:
        script_name: Name of the script file
        phase_name: Human-readable phase name
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    
    script_path = scripts_dir / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    logger.info("=" * 70)
    logger.info(f"Starting {phase_name}")
    logger.info("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"{phase_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{phase_name} failed with error:")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"Error running {phase_name}: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("Master Research Pipeline")
    logger.info("=" * 70)
    logger.info("This will run all phases of the research pipeline")
    logger.info("")
    
    phases = [
        ("phase1_data_collection.py", "Phase 1: Data Collection & Preprocessing"),
        ("phase2_demographics.py", "Phase 2: Demographics Extraction"),
        ("phase3_core_analysis.py", "Phase 3: Core Analysis"),
        ("phase4_statistical_analysis.py", "Phase 4: Statistical Analysis"),
    ]
    
    results = {}
    
    for script_name, phase_name in phases:
        success = run_phase(script_name, phase_name)
        results[phase_name] = success
        
        if not success:
            logger.warning(f"Phase failed: {phase_name}")
            logger.warning("You may continue manually or fix errors and re-run")
            response = input(f"Continue to next phase? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 70)
    
    for phase_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status}: {phase_name}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("\nAll phases completed successfully!")
    else:
        logger.warning("\nSome phases failed. Please review logs and fix errors.")


if __name__ == "__main__":
    main()

