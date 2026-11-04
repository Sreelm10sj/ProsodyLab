"""
lexirep_inference package
Production-grade inference interface for collaborator LexiRep lexical stress prediction.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from .schemas import (
    SyllablePrediction,
    WordPrediction,
    InferenceResult
)
from .feature_interface import (
    validate_features,
    validate_word_groupings,
    FeatureValidationError
)
from .model_loader import (
    LexiRepEncoder,
    load_lexirep_bundle,
    ModelLoadError
)
from .stress_predictor import (
    LexiRepPredictor
)

# Global model cache for service reuse
_PREDICTOR_CACHE: Dict[str, LexiRepPredictor] = {}


def load_model(
    language: str,
    bundle_root: Optional[str] = None,
    device: Optional[str] = None,
    use_cache: bool = True
) -> LexiRepPredictor:
    """
    Loads and caches a LexiRepPredictor for German or Italian.

    Args:
        language: 'GER' or 'ITA'.
        bundle_root: Optional path to the language model bundle.
        device: Optional 'cpu' or 'cuda'.
        use_cache: If True, caches loaded model instance for subsequent calls.

    Returns:
        LexiRepPredictor instance ready for inference.
    """
    lang = language.upper()
    if use_cache and lang in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[lang]

    predictor = LexiRepPredictor.from_bundle(lang, bundle_root, device)
    if use_cache:
        _PREDICTOR_CACHE[lang] = predictor
    return predictor


def predict_stress(
    language: str,
    features: np.ndarray,
    word_boundaries: List[List[int]],
    word_texts: Optional[List[str]] = None,
    bundle_root: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level functional entrypoint for LexiRep inference.

    Args:
        language: 'GER' or 'ITA'.
        features: (N, 768) float32 array of wav2vec representations.
        word_boundaries: List of syllable index lists covering all N syllables.
        word_texts: Optional list of word strings.
        bundle_root: Optional custom bundle path.
        device: Optional target device.

    Returns:
        Structured dictionary conforming to the ProsodyLab inference schema.
    """
    predictor = load_model(language, bundle_root, device, use_cache=True)
    result = predictor.predict_stress(features, word_boundaries, word_texts)
    return result.to_dict()


__all__ = [
    "LexiRepPredictor",
    "LexiRepEncoder",
    "SyllablePrediction",
    "WordPrediction",
    "InferenceResult",
    "FeatureValidationError",
    "ModelLoadError",
    "load_model",
    "predict_stress",
    "validate_features",
    "validate_word_groupings",
]
