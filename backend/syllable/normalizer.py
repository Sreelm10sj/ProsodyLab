"""
backend.syllable.normalizer

Conservative token normalization for the ProsodyLab syllable engine.
Handles punctuation, casing, apostrophes, contractions, hyphens, numbers, and acronyms
without aggressively rewriting or altering word identities.
"""

import re
from typing import List


# Unicode character normalization maps
_APOSTROPHE_MAP = str.maketrans({
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
})

_DASH_MAP = str.maketrans({
    "–": "-",  # en-dash
    "—": "-",  # em-dash
    "−": "-",  # minus sign
    "―": "-",  # horizontal bar
})

# Punctuation to strip from word boundaries
_BOUNDARY_PUNCT_REGEX = re.compile(
    r"^[\s\"'“”«»()[\]{}<>.,;:!?—–\-_/\\*~`^#@$%&+=|]+|"
    r"[\s\"'“”«»()[\]{}<>.,;:!?—–\-_/\\*~`^#@$%&+=|]+$"
)

# Common contraction expansions / checks
COMMON_CONTRACTIONS = {
    "wasn't": ("was", "n't"),
    "weren't": ("were", "n't"),
    "can't": ("can't",),
    "couldn't": ("could", "n't"),
    "don't": ("don't",),
    "doesn't": ("does", "n't"),
    "didn't": ("did", "n't"),
    "won't": ("won't",),
    "wouldn't": ("would", "n't"),
    "shouldn't": ("should", "n't"),
    "isn't": ("is", "n't"),
    "aren't": ("are", "n't"),
    "hasn't": ("has", "n't"),
    "haven't": ("have", "n't"),
    "hadn't": ("had", "n't"),
    "they'll": ("they'll",),
    "we'll": ("we'll",),
    "you'll": ("you'll",),
    "i'll": ("i'll",),
    "he'll": ("he'll",),
    "she'll": ("she'll",),
    "it'll": ("it'll",),
    "i'm": ("i'm",),
    "you're": ("you're",),
    "we're": ("we're",),
    "they're": ("they're",),
    "it's": ("it's",),
    "that's": ("that's",),
    "there's": ("there's",),
    "what's": ("what's",),
    "who's": ("who's",),
}


def normalize_word(token: str) -> str:
    """
    Conservatively normalizes a speech/transcript token.

    Rules applied:
    1. Standardize unicode apostrophes and dashes to ASCII.
    2. Strip leading and trailing punctuation and whitespace.
    3. Collapse multiple periods/spaces in acronyms (e.g. 'U.S.A.' -> 'usa').
    4. Strip trailing possessive apostrophe-s ("john's" -> "john") if root word remains.
    5. Convert to lowercase.

    Examples:
    ---------
    "Communication," -> "communication"
    "'really'"       -> "really"
    "state-of-the-art" -> "state-of-the-art"
    "don't"          -> "don't"
    "U.S.A."         -> "usa"
    """
    if not token or not isinstance(token, str):
        return ""

    # 1. Normalize unicode quotes, apostrophes, and dashes
    t = token.translate(_APOSTROPHE_MAP).translate(_DASH_MAP).strip()

    # Handle acronyms with periods (e.g. U.S.A. or Ph.D.) before boundary stripping
    if re.fullmatch(r"(?:[a-zA-Z]\.)+[a-zA-Z]?", t):
        t = t.replace(".", "")

    # 2. Strip leading and trailing boundary punctuation
    t = _BOUNDARY_PUNCT_REGEX.sub("", t)

    # 4. Handle possessive 's (e.g. "student's" -> "student", "students'" -> "students")
    # Preserve contractions like "it's", "that's", "there's", "what's", "who's"
    t_lower = t.lower()
    if t_lower not in COMMON_CONTRACTIONS:
        if t_lower.endswith("'s") and len(t) > 2:
            t = t[:-2]
        elif t.endswith("'") and len(t) > 1:
            t = t[:-1]

    # Re-strip any edge punctuation revealed after possessive removal
    t = _BOUNDARY_PUNCT_REGEX.sub("", t)

    return t.lower()


def is_numeric(token: str) -> bool:
    """Checks if a normalized token is composed entirely of digits or decimals."""
    t = token.strip()
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", t))


def is_acronym(token: str) -> bool:
    """
    Detects short all-caps abbreviations/initialisms (e.g. GPU, API, AI, NLP, F0).
    """
    raw = token.strip().translate(_APOSTROPHE_MAP).translate(_DASH_MAP)
    raw = _BOUNDARY_PUNCT_REGEX.sub("", raw)
    if re.fullmatch(r"[A-Z0-9]{2,5}", raw) and any(c.isalpha() for c in raw):
        return True
    if re.fullmatch(r"[A-Z]\d+", raw):  # e.g. F0
        return True
    return False


def split_hyphenated(token: str) -> List[str]:
    """
    Splits a hyphenated compound into its component word tokens.
    Returns [token] if no internal hyphens are present.
    """
    cleaned = normalize_word(token)
    if "-" in cleaned:
        parts = [p for p in cleaned.split("-") if p]
        return parts if parts else [cleaned]
    return [cleaned]
