"""
backend.lexirep.pooling

Syllable-level frame-center assignment and mean-pooling for LexiRep.
Maps continuous time intervals [start, end) to discrete Wav2Vec 2.0 frames
using exact convolutional receptive field centers.

Frame Geometry (at 16 kHz):
- Convolutional stride: 320 samples (20.0 ms)
- Receptive field: 400 samples (25.0 ms)
- Frame center formula:
    t_center(i) = (i * 320 + 200) / 16000 = i * 0.020 + 0.0125 seconds

Interval Slicing Rule:
- Frame i belongs to syllable [start, end) if and only if:
    start <= t_center(i) < end
- Adjacent intervals [t0, t1) and [t1, t2) produce strictly disjoint frame sets.

Short-Syllable Fallback:
- If a valid syllable interval captures 0 frame centers:
    i* = argmin_i |t_center(i) - midpoint|
  where midpoint = (start + end) / 2.0.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


FRAME_STRIDE_SAMPLES: int = 320
FRAME_STRIDE_SEC: float = 0.020
RECEPTIVE_FIELD_SAMPLES: int = 400
RECEPTIVE_FIELD_SEC: float = 0.025
FRAME_CENTER_OFFSET_SEC: float = 0.0125  # 200 samples / 16000 Hz


def compute_frame_centers(num_frames: int, sample_rate: int = 16000) -> np.ndarray:
    """
    Computes exact continuous center timestamps for all Wav2Vec 2.0 frames.

    Formula:
        t_center(i) = (i * 320 + 200) / sample_rate = i * 0.020 + 0.0125 seconds
    """
    if num_frames <= 0:
        return np.empty((0,), dtype=np.float64)

    indices = np.arange(num_frames, dtype=np.float64)
    stride_sec = FRAME_STRIDE_SAMPLES / float(sample_rate)
    center_offset_sec = (RECEPTIVE_FIELD_SAMPLES / 2.0) / float(sample_rate)
    return indices * stride_sec + center_offset_sec


def validate_syllable_intervals(
    syllable_intervals: Sequence[Tuple[float, float]],
    max_duration: Optional[float] = None
) -> None:
    """
    Validates syllable boundary intervals for temporal validity, ordering, and non-overlap.

    Parameters
    ----------
    syllable_intervals : Sequence[Tuple[float, float]]
        List or tuple of (start, end) timestamps in seconds.
    max_duration : Optional[float]
        Optional total utterance duration in seconds to bound intervals.

    Raises
    ------
    ValueError
        If intervals are empty, not ordered, overlapping, non-finite, or inverted.
    """
    if len(syllable_intervals) == 0:
        raise ValueError("Syllable intervals list is empty")

    prev_end: Optional[float] = None
    for idx, interval in enumerate(syllable_intervals):
        if len(interval) != 2:
            raise ValueError(
                f"Interval at index {idx} must be a 2-tuple (start, end), got {interval}"
            )

        start, end = float(interval[0]), float(interval[1])

        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(
                f"Interval at index {idx} contains non-finite values: [{start}, {end}]"
            )

        if start < 0.0:
            raise ValueError(
                f"Interval at index {idx} has negative start time: {start:.4f}s"
            )

        if start >= end:
            raise ValueError(
                f"Interval at index {idx} has invalid ordering (start >= end): [{start:.4f}s, {end:.4f}s]"
            )

        if prev_end is not None:
            if start < prev_end:
                raise ValueError(
                    f"Interval at index {idx} [{start:.4f}s, {end:.4f}s] overlaps or is out of order "
                    f"with previous interval ending at {prev_end:.4f}s"
                )

        if max_duration is not None and start >= max_duration:
            raise ValueError(
                f"Interval at index {idx} starts at {start:.4f}s, which exceeds utterance duration {max_duration:.4f}s"
            )

        prev_end = end


def assign_frames_to_interval(
    start: float,
    end: float,
    frame_centers: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """
    Assigns frame indices to a syllable interval [start, end).

    Rule:
        start <= t_center(i) < end

    Fallback:
        If 0 frame centers fall within [start, end), selects the single frame center
        closest to the midpoint (start + end) / 2.0.

    Returns
    -------
    Tuple[np.ndarray, bool]
        (frame_indices, used_fallback)
    """
    if len(frame_centers) == 0:
        raise ValueError("Cannot assign frames from empty frame_centers array")

    assigned = np.where((frame_centers >= start) & (frame_centers < end))[0]
    if len(assigned) > 0:
        return assigned, False

    # Short-syllable fallback: nearest frame center to interval midpoint
    midpoint = 0.5 * (start + end)
    nearest_idx = int(np.argmin(np.abs(frame_centers - midpoint)))
    return np.array([nearest_idx], dtype=np.int64), True


def pool_syllables(
    hidden_states: np.ndarray,
    syllable_intervals: Sequence[Tuple[float, float]],
    sample_rate: int = 16000
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Mean-pools Wav2Vec 2.0 frame representations into syllable feature vectors.

    Parameters
    ----------
    hidden_states : np.ndarray
        Contextual hidden states of shape (T, 768), dtype float32.
    syllable_intervals : Sequence[Tuple[float, float]]
        List of non-overlapping, ordered (start, end) time intervals in seconds.
    sample_rate : int
        Audio sampling rate, default 16000 Hz.

    Returns
    -------
    Tuple[np.ndarray, List[Dict[str, Any]]]
        - pooled_features: (N_syllables, 768) float32 matrix.
        - diagnostics: List of dicts with per-syllable assignment metadata:
            {
                "syllable_index": int,
                "start": float,
                "end": float,
                "frame_indices": List[int],
                "frame_count": int,
                "used_fallback": bool
            }
    """
    if not isinstance(hidden_states, np.ndarray):
        raise TypeError(
            f"Expected hidden_states as np.ndarray, got {type(hidden_states).__name__}"
        )

    if hidden_states.ndim != 2:
        raise ValueError(
            f"Expected 2-D hidden_states array (T, 768), got ndim={hidden_states.ndim} (shape {hidden_states.shape})"
        )

    if hidden_states.shape[1] != 768:
        raise ValueError(
            f"Expected hidden dimension 768, got {hidden_states.shape[1]}"
        )

    if not np.all(np.isfinite(hidden_states)):
        raise ValueError("hidden_states matrix contains non-finite values (NaN or Inf)")

    num_frames = hidden_states.shape[0]
    if num_frames == 0:
        raise ValueError("hidden_states matrix has 0 frames")

    frame_centers = compute_frame_centers(num_frames, sample_rate=sample_rate)
    max_duration = float(frame_centers[-1] + FRAME_STRIDE_SEC / 2.0)

    validate_syllable_intervals(syllable_intervals, max_duration=max_duration)

    pooled_vectors: List[np.ndarray] = []
    diagnostics: List[Dict[str, Any]] = []

    for idx, (start, end) in enumerate(syllable_intervals):
        frame_indices, used_fallback = assign_frames_to_interval(
            float(start), float(end), frame_centers
        )

        selected_frames = hidden_states[frame_indices, :]
        syllable_vec = np.mean(selected_frames, axis=0).astype(np.float32)

        pooled_vectors.append(syllable_vec)
        diagnostics.append({
            "syllable_index": idx,
            "start": float(start),
            "end": float(end),
            "frame_indices": frame_indices.tolist(),
            "frame_count": int(len(frame_indices)),
            "used_fallback": bool(used_fallback)
        })

    pooled_matrix = np.stack(pooled_vectors, axis=0).astype(np.float32)

    return pooled_matrix, diagnostics
