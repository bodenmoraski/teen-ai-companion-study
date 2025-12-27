"""Demographics classification modules."""
from .self_declaration import extract_self_declarations
from .llm_classifier import classify_users_llm, classify_age_llm
from .community_embedding import classify_with_community_embeddings
from .ensemble_classifier import create_ensemble_classification, weighted_age_vote

__all__ = [
    "extract_self_declarations",
    "classify_users_llm",
    "classify_age_llm",
    "classify_with_community_embeddings",
    "create_ensemble_classification",
    "weighted_age_vote",
]

