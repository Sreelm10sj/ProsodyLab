"""
schemas.py
Pydantic and dataclass schemas defining the standard input/output contracts
for ProsodyLab lexical stress inference.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class SyllablePrediction:
    syllable_index: int
    stress: int  # 1 for primary stress, 0 for unstressed
    contrast_score: float  # Signed acoustic contrast Δ = cos(z, z_s) - cos(z, z_u)
    confidence: float  # Calibrated probability in [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WordPrediction:
    word_id: int
    word_text: str = ""
    stressed_syllable_index: int = -1
    syllables: List[SyllablePrediction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_id": self.word_id,
            "word_text": self.word_text,
            "stressed_syllable_index": self.stressed_syllable_index,
            "syllables": [s.to_dict() for s in self.syllables]
        }


@dataclass
class InferenceResult:
    language: str
    total_words: int
    total_syllables: int
    linguistic_compliance: bool  # True if 100% of words have exactly 1 primary stress
    zero_stress_words: int
    multi_stress_words: int
    words: List[WordPrediction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "total_words": self.total_words,
            "total_syllables": self.total_syllables,
            "linguistic_compliance": self.linguistic_compliance,
            "zero_stress_words": self.zero_stress_words,
            "multi_stress_words": self.multi_stress_words,
            "words": [w.to_dict() for w in self.words]
        }
