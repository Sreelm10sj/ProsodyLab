"""
backend.training.validator

Strict validation service for LexiRep training datasets.
Verifies:
- Representation shape: 2-D matrix (N, 768)
- Label and grouping alignment: Y (N,), W (N,)
- Numeric integrity: float32-compatible, no NaN, no Inf, non-zero norms
- Binary label space: strictly {0, 1} with both classes represented
- Word structure: arbitrary integer IDs, >= 2 syllables per word, >= 1 stressed syllable per word
- Anchor viability: >= min_anchor_words bisyllabic words with exactly (1 stressed, 1 unstressed)
- Metadata: optional word_texts validation
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any, Sequence, Union
import numpy as np


@dataclass
class DatasetValidationResult:
    """Structured result of dataset validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    syllable_count: int = 0
    word_count: int = 0

    stressed_count: int = 0
    unstressed_count: int = 0
    stress_ratio: float = 0.0

    bisyllabic_anchor_candidate_count: int = 0
    rejected_word_count: int = 0

    feature_shape: Tuple[int, ...] = ()
    feature_dtype: str = ""

    word_size_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to JSON-serializable dictionary."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "syllable_count": self.syllable_count,
            "word_count": self.word_count,
            "stressed_count": self.stressed_count,
            "unstressed_count": self.unstressed_count,
            "stress_ratio": round(self.stress_ratio, 4),
            "bisyllabic_anchor_candidate_count": self.bisyllabic_anchor_candidate_count,
            "rejected_word_count": self.rejected_word_count,
            "feature_shape": list(self.feature_shape),
            "feature_dtype": self.feature_dtype,
            "word_size_distribution": {str(k): v for k, v in sorted(self.word_size_distribution.items())},
        }


class DatasetValidationError(ValueError):
    """Raised when dataset validation fails with structured diagnostics."""
    def __init__(self, message: str, result: DatasetValidationResult):
        super().__init__(message)
        self.result = result


def validate_dataset(
    X: Any,
    Y: Any,
    W: Any,
    word_texts: Optional[Union[Dict[int, str], Sequence[str]]] = None,
    *,
    min_syllables: int = 100,
    min_anchor_words: int = 5,
) -> DatasetValidationResult:
    """
    Validate a dataset for LexiRep training and anchor search.

    Args:
        X: Syllable feature matrix, expected shape (N, 768), float-like.
        Y: Syllable binary stress labels, expected shape (N,), {0, 1}.
        W: Word grouping IDs, expected shape (N,), arbitrary integer IDs.
        word_texts: Optional word orthography (dict mapping word_id -> str, or syllable-aligned sequence).
        min_syllables: Minimum required total syllables (default: 100).
        min_anchor_words: Minimum required candidate bisyllabic anchor words (default: 5).

    Returns:
        DatasetValidationResult with valid=True/False and full diagnostic statistics.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ------------------------------------------------------------------
    # 1. Feature matrix X shape and type inspection
    # ------------------------------------------------------------------
    x_arr: Optional[np.ndarray] = None
    feature_shape: Tuple[int, ...] = ()
    feature_dtype: str = ""

    try:
        x_arr = np.asarray(X)
        feature_shape = tuple(x_arr.shape)
        feature_dtype = str(x_arr.dtype)
    except Exception as exc:
        errors.append(f"X cannot be converted to a NumPy array: {exc}")

    if x_arr is not None:
        if x_arr.ndim != 2:
            errors.append(f"X must be 2-D with shape (N, 768), got {x_arr.ndim}-D with shape {x_arr.shape}.")
        else:
            n_samples, dim = x_arr.shape
            if dim != 768:
                errors.append(f"X feature dimension must be 768, got {dim}.")
            if n_samples == 0:
                errors.append("X is empty (0 syllables).")
            elif n_samples < min_syllables:
                errors.append(
                    f"Dataset contains {n_samples} syllables, which is below the minimum required {min_syllables}."
                )

    # ------------------------------------------------------------------
    # 2. Labels Y and Word IDs W shape inspection
    # ------------------------------------------------------------------
    y_arr: Optional[np.ndarray] = None
    w_arr: Optional[np.ndarray] = None

    try:
        y_arr = np.asarray(Y)
    except Exception as exc:
        errors.append(f"Y cannot be converted to a NumPy array: {exc}")

    try:
        w_arr = np.asarray(W)
    except Exception as exc:
        errors.append(f"W cannot be converted to a NumPy array: {exc}")

    n_syllables = len(x_arr) if (x_arr is not None and x_arr.ndim >= 1) else 0

    if y_arr is not None:
        if y_arr.ndim != 1:
            errors.append(f"Y must be 1-D with shape (N,), got {y_arr.ndim}-D with shape {y_arr.shape}.")
        elif x_arr is not None and x_arr.ndim == 2 and len(y_arr) != n_syllables:
            errors.append(f"Length mismatch: len(X)={n_syllables} != len(Y)={len(y_arr)}.")

    if w_arr is not None:
        if w_arr.ndim != 1:
            errors.append(f"W must be 1-D with shape (N,), got {w_arr.ndim}-D with shape {w_arr.shape}.")
        elif x_arr is not None and x_arr.ndim == 2 and len(w_arr) != n_syllables:
            errors.append(f"Length mismatch: len(X)={n_syllables} != len(W)={len(w_arr)}.")

    # ------------------------------------------------------------------
    # 3. Numeric Sanity of X (NaN, Inf, Zero Norms)
    # ------------------------------------------------------------------
    if x_arr is not None and x_arr.ndim == 2 and x_arr.size > 0:
        if not np.issubdtype(x_arr.dtype, np.number):
            errors.append(f"X must contain numeric values, got non-numeric dtype {x_arr.dtype}.")
        else:
            has_nan = bool(np.isnan(x_arr).any())
            has_inf = bool(np.isinf(x_arr).any())
            if has_nan:
                errors.append("X contains NaN (Not a Number) values.")
            if has_inf:
                errors.append("X contains infinite (Inf) values.")

            # Check for zero or near-zero norms
            if not has_nan and not has_inf:
                norms = np.linalg.norm(x_arr, axis=1)
                zero_norm_count = int(np.sum(norms < 1e-12))
                if zero_norm_count > 0:
                    errors.append(
                        f"X contains {zero_norm_count} syllable embedding(s) with zero or near-zero norm (<1e-12)."
                    )

    # ------------------------------------------------------------------
    # 4. Binary Labels Y validation
    # ------------------------------------------------------------------
    stressed_count = 0
    unstressed_count = 0
    stress_ratio = 0.0

    if y_arr is not None and y_arr.ndim == 1 and len(y_arr) > 0:
        if not np.issubdtype(y_arr.dtype, np.integer) and not np.issubdtype(y_arr.dtype, np.floating):
            errors.append(f"Y must contain numeric labels, got dtype {y_arr.dtype}.")
        else:
            unique_y = set(np.unique(y_arr))
            invalid_labels = unique_y - {0, 1, 0.0, 1.0}
            if invalid_labels:
                errors.append(f"Y must contain only binary labels (0 and 1), found invalid values: {sorted(invalid_labels)}.")

            stressed_count = int(np.sum(y_arr == 1))
            unstressed_count = int(np.sum(y_arr == 0))
            total_labeled = stressed_count + unstressed_count

            if stressed_count == 0:
                errors.append("Y contains no stressed syllables (class 1).")
            if unstressed_count == 0:
                errors.append("Y contains no unstressed syllables (class 0).")

            if total_labeled > 0:
                stress_ratio = float(stressed_count / total_labeled)
                if stress_ratio < 0.15 or stress_ratio > 0.60:
                    warnings.append(
                        f"Unusual lexical stress ratio: {stress_ratio:.2%}. Typically lexical stress ratio is between 20% and 45%."
                    )

    # ------------------------------------------------------------------
    # 5. Word Grouping (W) and Linguistic Constraints
    # ------------------------------------------------------------------
    word_count = 0
    bisyllabic_anchor_candidate_count = 0
    rejected_word_count = 0
    word_size_dist: Dict[int, int] = defaultdict(int)

    if (
        w_arr is not None
        and y_arr is not None
        and w_arr.ndim == 1
        and y_arr.ndim == 1
        and len(w_arr) == len(y_arr)
        and len(w_arr) > 0
    ):
        wmap: Dict[Any, List[int]] = defaultdict(list)
        for idx, w_id in enumerate(w_arr):
            wmap[w_id].append(idx)

        word_count = len(wmap)
        single_syl_words: List[Any] = []
        no_stress_words: List[Any] = []

        for w_id, idxs in wmap.items():
            k = len(idxs)
            word_size_dist[k] += 1

            if k < 2:
                single_syl_words.append(w_id)

            s_in_word = sum(1 for i in idxs if y_arr[i] == 1)
            u_in_word = sum(1 for i in idxs if y_arr[i] == 0)

            if s_in_word == 0:
                no_stress_words.append(w_id)

            # Anchor candidacy check: exactly 2 syllables, exactly 1 stressed, 1 unstressed
            if k == 2 and s_in_word == 1 and u_in_word == 1:
                bisyllabic_anchor_candidate_count += 1

        rejected_word_count = word_count - bisyllabic_anchor_candidate_count

        if single_syl_words:
            sample = [str(w) for w in single_syl_words[:5]]
            errors.append(
                f"Found {len(single_syl_words)} word(s) with only 1 syllable (e.g. {sample}). "
                f"LexiRep requires every word to have at least 2 syllables."
            )

        if no_stress_words:
            sample = [str(w) for w in no_stress_words[:5]]
            errors.append(
                f"Found {len(no_stress_words)} word(s) with no stressed syllables (e.g. {sample}). "
                f"LexiRep requires every word to have at least one stressed syllable."
            )

        # --------------------------------------------------------------
        # 6. Anchor Viability Threshold
        # --------------------------------------------------------------
        if bisyllabic_anchor_candidate_count < min_anchor_words:
            errors.append(
                f"Insufficient bisyllabic anchor candidate words: found {bisyllabic_anchor_candidate_count}, "
                f"required at least {min_anchor_words}."
            )

    # ------------------------------------------------------------------
    # 7. word_texts validation (optional metadata)
    # ------------------------------------------------------------------
    if word_texts is not None:
        if isinstance(word_texts, dict):
            if w_arr is not None:
                unique_w_ids = set(w_arr)
                missing = unique_w_ids - set(word_texts.keys())
                if missing:
                    sample = [str(w) for w in list(missing)[:5]]
                    errors.append(
                        f"word_texts dictionary is missing {len(missing)} word IDs from W (e.g. {sample})."
                    )
                # Ensure all values are strings
                non_str_vals = [k for k, v in word_texts.items() if not isinstance(v, str)]
                if non_str_vals:
                    sample = [str(k) for k in non_str_vals[:5]]
                    errors.append(f"word_texts contains non-string values for keys: {sample}.")
        elif isinstance(word_texts, (list, tuple, np.ndarray)):
            seq_len = len(word_texts)
            if w_arr is not None and seq_len != len(w_arr):
                errors.append(
                    f"word_texts sequence length ({seq_len}) does not match syllable count ({len(w_arr)}). "
                    f"Pass a syllable-aligned sequence of strings or a dict mapping word_id -> str."
                )
            elif w_arr is not None and seq_len == len(w_arr):
                # Verify internal consistency: all syllables belonging to the same word ID must share text
                wmap_text: Dict[Any, set] = defaultdict(set)
                for idx, w_id in enumerate(w_arr):
                    wmap_text[w_id].add(str(word_texts[idx]))
                inconsistent = {w_id: texts for w_id, texts in wmap_text.items() if len(texts) > 1}
                if inconsistent:
                    sample = list(inconsistent.items())[:3]
                    errors.append(
                        f"Conflicting word_texts within the same word ID for {len(inconsistent)} word(s) (e.g. {sample})."
                    )
        else:
            errors.append(
                f"word_texts must be a dict[int, str] or a syllable-aligned Sequence[str], got {type(word_texts).__name__}."
            )

    # ------------------------------------------------------------------
    # Final Result
    # ------------------------------------------------------------------
    valid = len(errors) == 0

    return DatasetValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        syllable_count=n_syllables,
        word_count=word_count,
        stressed_count=stressed_count,
        unstressed_count=unstressed_count,
        stress_ratio=stress_ratio,
        bisyllabic_anchor_candidate_count=bisyllabic_anchor_candidate_count,
        rejected_word_count=rejected_word_count,
        feature_shape=feature_shape,
        feature_dtype=feature_dtype,
        word_size_distribution=dict(word_size_dist),
    )
