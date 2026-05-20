"""Analysis modules for AnthroScore, BERTopic, and emotion analysis."""

# Lightweight emotion analysis functions (regex-only, no GPU deps)
from .emotion_analysis import (
    extract_emotion_features,
    aggregate_emotions_to_user_level,
    classify_bot_attributed_emotions,
    detect_bot_attribution,
)

# Heavy modules (torch, transformers, spacy) — import only when available
try:
    from .anthroscore_runner import compute_anthroscores, aggregate_to_user_level
except ImportError:
    compute_anthroscores = None
    aggregate_to_user_level = None

try:
    from .bertopic_clustering import extract_topic_features, aggregate_topics_to_user_level
except ImportError:
    extract_topic_features = None
    aggregate_topics_to_user_level = None

__all__ = [
    "compute_anthroscores",
    "aggregate_to_user_level",
    "extract_topic_features",
    "aggregate_topics_to_user_level",
    "extract_emotion_features",
    "aggregate_emotions_to_user_level",
    "classify_bot_attributed_emotions",
    "detect_bot_attribution",
]

