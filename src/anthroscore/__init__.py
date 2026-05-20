"""
AnthroScore: Anthropomorphism scoring for informal social media text.

This package implements:
- AnthroScore V2: MLM-based pronoun log-ratio (Cheng et al. 2024 methodology)
- Human calibration: loads validation data to improve V3 LLM scoring
- N-gram features: bigram/trigram phrase detection for anthropomorphization signals
"""

# Lightweight modules — always available
from .human_calibration import (
    load_validation_data,
    run_calibration,
    generate_calibrated_prompt,
    print_calibration_report,
)
from .ngram_features import (
    analyze_ngrams,
    get_ngram_features,
    enrich_prompt_with_ngrams,
)

# Heavy modules (torch, spacy, transformers) — import only when available
try:
    from .anthroscore_v2 import AnthroScoreV2
    from .preprocessors import RedditPreprocessor
    from .entity_normalizer import EntityNormalizer
except ImportError:
    AnthroScoreV2 = None
    RedditPreprocessor = None
    EntityNormalizer = None

__version__ = "3.1.0"
__all__ = [
    "AnthroScoreV2",
    "RedditPreprocessor",
    "EntityNormalizer",
    "load_validation_data",
    "run_calibration",
    "generate_calibrated_prompt",
    "print_calibration_report",
    "analyze_ngrams",
    "get_ngram_features",
    "enrich_prompt_with_ngrams",
]

