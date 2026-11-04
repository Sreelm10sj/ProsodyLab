"""
service.py
Thin, production-ready adapter service wrapping the verified LexiRep inference engine.
Delegates all forward inference, prototype contrast, and linguistic constraint
enforcement directly to backend.prosodylab_model_export.lexirep_inference without
duplicating model architectures or inference algorithms.
"""

from typing import List, Optional, Dict, Any
import os
import sys
import numpy as np

# Ensure prosodylab_model_export is importable
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
_EXPORT_DIR = os.path.join(_BACKEND_DIR, "prosodylab_model_export")

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _EXPORT_DIR not in sys.path:
    sys.path.insert(0, _EXPORT_DIR)

try:
    from prosodylab_model_export.lexirep_inference import (
        LexiRepPredictor,
        InferenceResult,
        WordPrediction,
        SyllablePrediction,
        FeatureValidationError,
        ModelLoadError,
        validate_features,
        validate_word_groupings,
    )
except ImportError:
    from lexirep_inference import (
        LexiRepPredictor,
        InferenceResult,
        WordPrediction,
        SyllablePrediction,
        FeatureValidationError,
        ModelLoadError,
        validate_features,
        validate_word_groupings,
    )


SUPPORTED_LANGUAGES = ("GER", "ITA")


class LexiRepService:
    """
    Isolated backend service managing frozen LexiRep model lifecycles and inference.
    Acts as a thin adapter delegating to the verified collaborator export package.
    """

    def __init__(
        self,
        bundle_root: Optional[str] = None,
        default_device: Optional[str] = None
    ):
        """
        Initializes the service adapter.

        Args:
            bundle_root: Optional root directory containing language model bundles.
                         Defaults to backend/prosodylab_model_export.
            default_device: Optional target device ('cpu' or 'cuda').
        """
        self.bundle_root = bundle_root or _EXPORT_DIR
        self.default_device = default_device
        self._predictors: Dict[str, Optional[LexiRepPredictor]] = {
            "GER": None,
            "ITA": None,
        }

    def supported_languages(self) -> List[str]:
        """Returns list of supported language / nativity identifiers."""
        return list(SUPPORTED_LANGUAGES)

    def is_loaded(self, language: str) -> bool:
        """Checks if the predictor for the specified language is loaded into memory."""
        lang = self._normalize_language(language)
        return self._predictors.get(lang) is not None

    def load(
        self,
        language: str,
        device: Optional[str] = None
    ) -> LexiRepPredictor:
        """
        Explicitly loads and caches a LexiRep predictor into memory.

        Args:
            language: 'GER' or 'ITA'.
            device: Optional device ('cpu', 'cuda'). Defaults to self.default_device.

        Returns:
            Loaded LexiRepPredictor instance.

        Raises:
            ValueError: If language is unsupported.
            ModelLoadError: If checkpoint or prototype assets are missing/corrupted.
        """
        lang = self._normalize_language(language)
        target_device = device or self.default_device
        bundle_dir = os.path.join(self.bundle_root, lang)

        if not os.path.isdir(bundle_dir):
            raise ModelLoadError(f"LexiRep bundle directory not found: {bundle_dir}")

        predictor = LexiRepPredictor.from_bundle(
            language=lang,
            bundle_root=bundle_dir,
            device=target_device
        )
        self._predictors[lang] = predictor
        return predictor

    def get_predictor(
        self,
        language: str,
        device: Optional[str] = None
    ) -> LexiRepPredictor:
        """
        Retrieves the cached predictor, loading it on demand if not already in memory.
        """
        lang = self._normalize_language(language)
        if self._predictors[lang] is None:
            return self.load(lang, device=device)
        return self._predictors[lang]

    def predict(
        self,
        language: str,
        features: np.ndarray,
        word_boundaries: List[List[int]],
        word_texts: Optional[List[str]] = None,
        device: Optional[str] = None
    ) -> InferenceResult:
        """
        Validates syllable inputs and executes lexical stress inference.

        Args:
            language: 'GER' or 'ITA'.
            features: 2D float32 array of shape (N, 768).
            word_boundaries: Syllable indices partitioned by word, covering all 0..N-1.
            word_texts: Optional list of word strings matching word_boundaries length.
            device: Optional target device override.

        Returns:
            InferenceResult object with structured word and syllable stress predictions.

        Raises:
            ValueError: If language is unsupported or inputs are invalid.
            FeatureValidationError: If feature shape/dtype/values violate inference contract.
        """
        # 1. Validate language
        lang = self._normalize_language(language)

        # 2. Validate input features
        clean_features = self._validate_input_features(features)

        # 3. Validate word boundaries
        clean_boundaries = validate_word_groupings(len(clean_features), word_boundaries)

        # 4. Delegate to verified predictor
        predictor = self.get_predictor(lang, device=device)
        return predictor.predict_stress(
            features=clean_features,
            word_boundaries=clean_boundaries,
            word_texts=word_texts
        )

    def _normalize_language(self, language: str) -> str:
        """Validates and normalizes language string to uppercase."""
        if not language or not isinstance(language, str):
            raise ValueError(f"Language must be a non-empty string, got: {language}")

        lang_upper = language.strip().upper()
        if lang_upper not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Supported languages: {list(SUPPORTED_LANGUAGES)}"
            )
        return lang_upper

    def _validate_input_features(self, features: Any) -> np.ndarray:
        """
        Strictly validates the input feature tensor before passing to inference.
        Rejects None, non-arrays, non-2D arrays, non-768 dimensions, incompatible dtypes,
        and NaN/Inf values.
        """
        if features is None:
            raise FeatureValidationError("Input feature matrix cannot be None.")

        if not isinstance(features, np.ndarray):
            raise FeatureValidationError(f"Expected numpy.ndarray, got {type(features).__name__}.")

        if features.ndim != 2:
            raise FeatureValidationError(
                f"Expected 2D feature matrix (N, 768), got ndim={features.ndim} with shape {features.shape}."
            )

        if features.shape[1] != 768:
            raise FeatureValidationError(
                f"Expected feature dimensionality 768, got {features.shape[1]} (shape: {features.shape})."
            )

        if features.shape[0] == 0:
            raise FeatureValidationError("Input feature matrix has 0 rows (no syllables provided).")

        # Check dtype compatibility
        if not np.issubdtype(features.dtype, np.floating):
            raise FeatureValidationError(
                f"Expected float32 feature array, got incompatible dtype: {features.dtype}."
            )

        if not np.all(np.isfinite(features)):
            raise FeatureValidationError("Input feature matrix contains NaN or Infinite values.")

        return features.astype(np.float32, copy=False)


# Default singleton service instance for backend reuse
lexirep_service = LexiRepService()
