"""
feature_interface.py
Input feature validation, shape inspection, and syllable-to-word grouping interfaces.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class FeatureValidationError(ValueError):
    """Raised when input feature tensor or word metadata violates inference contracts."""
    pass


def validate_features(
    features: np.ndarray,
    expected_dim: int = 768
) -> np.ndarray:
    """
    Validates that the input representation tensor is a 2D float32 array
    of shape (N, expected_dim) with no NaNs or Infs.
    """
    if not isinstance(features, np.ndarray):
        features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise FeatureValidationError(f"Expected 2D feature matrix [num_syllables, {expected_dim}], got ndim={features.ndim} (shape: {features.shape})")

    if features.shape[1] != expected_dim:
        raise FeatureValidationError(f"Expected feature dimension {expected_dim}, got {features.shape[1]}")

    if not np.all(np.isfinite(features)):
        raise FeatureValidationError("Input feature matrix contains NaN or Infinite values.")

    return features.astype(np.float32, copy=False)


def validate_word_groupings(
    num_syllables: int,
    word_boundaries: List[List[int]]
) -> List[List[int]]:
    """
    Validates that word_boundaries forms a valid, non-overlapping partition
    covering all syllable indices [0, num_syllables - 1].
    """
    if not word_boundaries or len(word_boundaries) == 0:
        raise FeatureValidationError("word_boundaries cannot be empty.")

    all_indices = []
    for w_idx, syl_list in enumerate(word_boundaries):
        if not syl_list or len(syl_list) == 0:
            raise FeatureValidationError(f"Word at index {w_idx} has 0 syllables.")
        for s in syl_list:
            if not isinstance(s, (int, np.integer)):
                raise FeatureValidationError(f"Syllable index {s} in word {w_idx} is not an integer.")
            if s < 0 or s >= num_syllables:
                raise FeatureValidationError(f"Syllable index {s} out of bounds [0, {num_syllables - 1}].")
            all_indices.append(s)

    if len(all_indices) != len(set(all_indices)):
        raise FeatureValidationError("Duplicate syllable indices detected across word boundaries.")

    if len(all_indices) != num_syllables:
        missing = set(range(num_syllables)) - set(all_indices)
        raise FeatureValidationError(f"Syllables missing from word boundaries: {sorted(list(missing))[:10]}")

    return word_boundaries
