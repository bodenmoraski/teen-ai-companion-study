"""
AnthroScore V2: Enhanced anthropomorphism scoring for informal social media text.

This package implements AnthroScore V2, an enhanced version of the computational
linguistic measure from Cheng et al. (2024) designed for Reddit posts about
AI companions.
"""

from .anthroscore_v2 import AnthroScoreV2
from .preprocessors import RedditPreprocessor
from .entity_normalizer import EntityNormalizer

__version__ = "2.0.0"
__all__ = ["AnthroScoreV2", "RedditPreprocessor", "EntityNormalizer"]

