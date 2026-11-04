"""
backend.syllable.types

Typed data models for the ProsodyLab syllable count and segmentation engine.
Independent of Whisper, Wav2Vec2, LexiRep, and the annotation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass(frozen=True)
class SyllableResult:
    """
    Canonical representation of a word's syllable count and orthographic segmentation.

    Attributes
    ----------
    word : str
        The raw input token.
    normalized_word : str
        The cleaned, normalized token used for dictionary and rule evaluation.
    syllables : List[str]
        Ordered list of orthographic syllable substrings.
    count : int
        Syllable count K (must equal len(syllables)).
    source : Literal["dictionary", "rule", "fallback"]
        Origin of the syllable decomposition.
    is_polysyllabic : bool
        True if count >= 2, False if count == 1.
    confidence : Optional[float]
        Confidence score between 0.0 and 1.0 (1.0 for dictionary hit).
    """

    word: str
    normalized_word: str
    syllables: List[str]
    count: int
    source: Literal["dictionary", "rule", "fallback"]
    is_polysyllabic: bool
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.count != len(self.syllables):
            raise ValueError(
                f"Syllable count ({self.count}) does not match syllables list length ({len(self.syllables)})"
            )
        if self.count <= 0:
            raise ValueError(f"Syllable count must be positive, got {self.count}")
        if self.is_polysyllabic != (self.count >= 2):
            raise ValueError(
                f"is_polysyllabic ({self.is_polysyllabic}) inconsistent with count ({self.count})"
            )


@dataclass(frozen=True)
class SyllableInterval:
    """
    Pure temporal interval representing a syllable's span within a parent word.

    Attributes
    ----------
    syllable_index : int
        0-indexed position within the parent word (0 .. K-1).
    syllable_text : str
        Orthographic syllable text.
    start : float
        Start timestamp in seconds.
    end : float
        End timestamp in seconds.
    duration : float
        Duration in seconds (end - start).
    """

    syllable_index: int
    syllable_text: str
    start: float
    end: float
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(
                f"Invalid syllable interval: start ({self.start:.4f}s) >= end ({self.end:.4f}s)"
            )
        object.__setattr__(self, "duration", self.end - self.start)
