"""
backend.syllable.partitioner

Pure temporal partitioning: converts a word timestamp interval [word_start, word_end]
and syllable sequence into equal-duration syllable timestamp intervals.
Independent of audio, Wav2Vec2, and model representations.
"""

from typing import List
from .types import SyllableInterval


def partition_syllable_intervals(
    word_start: float,
    word_end: float,
    syllables: List[str]
) -> List[SyllableInterval]:
    """
    Partitions a word temporal interval into K equal-duration syllable intervals.

    Parameters
    ----------
    word_start : float
        Start timestamp of the parent word in seconds. Must be >= 0.
    word_end : float
        End timestamp of the parent word in seconds. Must be > word_start.
    syllables : List[str]
        List of orthographic syllable strings. Length K must be >= 1.

    Returns
    -------
    List[SyllableInterval]
        List of K contiguous, non-overlapping SyllableInterval objects.

    Raises
    ------
    ValueError
        If inputs are invalid or violating temporal monotonicity.
    """
    if not syllables:
        raise ValueError("Syllables list cannot be empty")

    if word_start < 0.0:
        raise ValueError(f"Word start cannot be negative: {word_start:.4f}s")

    if word_start >= word_end:
        raise ValueError(
            f"Invalid word interval (start >= end): [{word_start:.4f}s, {word_end:.4f}s]"
        )

    K = len(syllables)
    total_duration = word_end - word_start
    dt = total_duration / float(K)

    intervals: List[SyllableInterval] = []

    for k in range(K):
        s_start = word_start + k * dt
        # Snap the final syllable endpoint to exactly word_end to prevent float drift
        s_end = word_end if k == K - 1 else (word_start + (k + 1) * dt)

        intervals.append(
            SyllableInterval(
                syllable_index=k,
                syllable_text=syllables[k],
                start=float(s_start),
                end=float(s_end)
            )
        )

    # Sanity checks on constructed intervals
    for idx, interval in enumerate(intervals):
        if interval.start >= interval.end:
            raise ValueError(
                f"Generated syllable {idx} has invalid bounds: [{interval.start:.4f}s, {interval.end:.4f}s]"
            )
        if idx > 0:
            prev = intervals[idx - 1]
            if abs(interval.start - prev.end) > 1e-9:
                raise ValueError(
                    f"Syllables {idx-1} and {idx} are not contiguous: {prev.end:.6f}s vs {interval.start:.6f}s"
                )

    return intervals
