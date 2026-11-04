"""
backend.syllable package

Public API for syllable count and orthographic segmentation in ProsodyLab.
Provides:
- SyllableEngine: Two-tier engine (Dictionary lookup -> Rule fallback)
- SyllableDictionary: Canonical offline dictionary loader and ISLE train-builder
- SyllableResult, SyllableInterval: Typed data structures
- normalize_word: Conservative token normalizer
- count_syllables_rule_based, segment_syllables_rule_based: Dependency-free rule engine
- partition_syllable_intervals: Pure temporal equal-duration interval partitioning
"""

from .types import SyllableResult, SyllableInterval
from .normalizer import normalize_word, is_numeric, is_acronym, split_hyphenated
from .rule_engine import (
    count_syllables_rule_based,
    segment_syllables_rule_based,
    analyze_with_rules,
)
from .dictionary import SyllableDictionary, SyllableEngine
from .partitioner import partition_syllable_intervals

__all__ = [
    "SyllableResult",
    "SyllableInterval",
    "SyllableDictionary",
    "SyllableEngine",
    "normalize_word",
    "is_numeric",
    "is_acronym",
    "split_hyphenated",
    "count_syllables_rule_based",
    "segment_syllables_rule_based",
    "analyze_with_rules",
    "partition_syllable_intervals",
]
