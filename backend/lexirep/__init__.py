"""
backend.lexirep package
Public interface for LexiRep syllable-level lexical stress inference in ProsodyLab.
"""

from .service import (
    LexiRepService,
    lexirep_service,
    SUPPORTED_LANGUAGES,
    InferenceResult,
    WordPrediction,
    SyllablePrediction,
    FeatureValidationError,
    ModelLoadError,
)

__all__ = [
    "LexiRepService",
    "lexirep_service",
    "SUPPORTED_LANGUAGES",
    "InferenceResult",
    "WordPrediction",
    "SyllablePrediction",
    "FeatureValidationError",
    "ModelLoadError",
]
