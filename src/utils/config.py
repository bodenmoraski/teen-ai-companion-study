"""Configuration and constants for the research project."""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES = PROJECT_ROOT / "data" / "features"
DATA_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations"

# Results directories
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"

# Ensure directories exist
for directory in [DATA_RAW, DATA_PROCESSED, DATA_FEATURES, DATA_ANNOTATIONS,
                  RESULTS_FIGURES, RESULTS_TABLES, RESULTS_MODELS]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys (handle quotes in .env file)
# Try multiple ways to get the key
_openai_key = os.getenv("OPENAI_API_KEY", "")

# If empty, try reading .env file directly (handles cases where key is on line 2)
if not _openai_key:
    try:
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY") and "=" in line:
                        _openai_key = line.split("=", 1)[1].strip()
                        break
    except Exception:
        pass

# Remove quotes if present (handles both single and double quotes)
if _openai_key:
    _openai_key = _openai_key.strip()
    if (_openai_key.startswith("'") and _openai_key.endswith("'")) or \
       (_openai_key.startswith('"') and _openai_key.endswith('"')):
        _openai_key = _openai_key[1:-1].strip()

OPENAI_API_KEY = _openai_key

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Arctic Shift API configuration
ARCTIC_SHIFT_BASE_URL = "https://arctic-shift.photon-reddit.com"
ARCTIC_SHIFT_RATE_LIMIT_SECONDS = 1.0  # Delay between requests

# Data collection parameters
TARGET_SUBREDDITS = [
    "CharacterAI",
    "Replika",
    "replika_ai",
    "AICompanions",
    "SocialChatbots"
]

# Time range for data collection (Unix timestamps)
# Jan 1, 2024 to Dec 31, 2025
COLLECTION_START_UTC = 1704067200  # 2024-01-01 00:00:00 UTC
COLLECTION_END_UTC = 1767225600    # 2025-12-31 23:59:59 UTC

# Preprocessing parameters
MIN_COMMENT_LENGTH = 20
MAX_COMMENT_LENGTH = 10000
BOT_AUTHORS = ["AutoModerator", "[deleted]", "[removed]", "None"]

# Demographics classification
AGE_BUCKETS = ["13-18", "19-25", "26-40", "41-60", "61-80"]
GENDER_CATEGORIES = ["male", "female", "nonbinary", "unknown"]

# Seed pairs for community embeddings
AGE_SEED_PAIRS = [
    ("teenagers", "RedditForGrownups"),
    ("teenrelationships", "relationship_advice"),
    ("highschool", "college"),
    ("GenZ", "GenX"),
]

GENDER_SEED_PAIRS = [
    ("AskWomen", "AskMen"),
    ("TwoXChromosomes", "MensRights"),
    ("TheGirlSurvivalGuide", "everyman"),
]

# Model configurations
# Available options (tested and working):
# - gpt-4.1-nano: Cheapest, excellent for classification (RECOMMENDED)
# - gpt-5-nano: Newer, better accuracy, slightly more expensive
# - gpt-5-mini: Balance of cost and performance
# - gpt-4o-mini: Older standard, still works
# 
# Recommendation: gpt-4.1-nano for best cost/performance ratio
LLM_AGE_MODEL = "gpt-4.1-nano"  # Updated to newer, cheaper model
LLM_TEMPERATURE = 0.3
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

