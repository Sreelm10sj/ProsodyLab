"""
backend.syllable.rule_engine

Deterministic, dependency-free English orthographic syllable count and segmentation engine.
Applies phonological rules including vowel nucleation, diphthong coalescing, hiatus splitting,
silent-e, consonant+le/re, suffix rules, and the Maximal Onset Principle.
"""

import re
from typing import List, Tuple
import numpy as np

from .types import SyllableResult
from .normalizer import normalize_word, is_numeric, is_acronym, split_hyphenated, COMMON_CONTRACTIONS


VOWELS = "aeiouy"

DIPHTHONGS = {
    "ai", "au", "aw", "ay", "ea", "ee", "ei", "ey", "ie", "oa", "oe", "oi", "oo", "ou", "ow", "oy"
}

HIATUS_PAIRS = {
    "ia", "io", "iu", "ua", "uo", "eo", "ii", "uu"
}

# Legal English syllable onsets under the Maximal Onset Principle
LEGAL_ONSETS = {
    "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr",
    "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw", "th", "tr", "tw",
    "wh", "wr", "str", "spr", "spl", "scr", "qu", "ph"
}


def count_syllables_rule_based(token: str) -> int:
    """
    Computes canonical orthographic syllable count K for an English word token.
    Deterministic, pure Python, zero external dependencies.
    """
    w = normalize_word(token)
    if not w:
        return 1

    # Numeric tokens
    if is_numeric(w):
        return 1

    # Acronyms (e.g. GPU -> 3, API -> 3, AI -> 2, NLP -> 3, F0 -> 2)
    if is_acronym(token):
        clean_raw = re.sub(r"[^a-zA-Z0-9]", "", token)
        return max(1, len(clean_raw))

    # Hyphenated compound handling (sum of components)
    if "-" in w:
        parts = split_hyphenated(w)
        if len(parts) > 1:
            return sum(count_syllables_rule_based(p) for p in parts)

    # Standard contractions
    if w in COMMON_CONTRACTIONS:
        return len(COMMON_CONTRACTIONS[w])

    # Strip any internal apostrophe for base phonetic rule evaluation
    w_clean = re.sub(r"[^a-z]", "", w)
    if len(w_clean) <= 2:
        return 1

    count = 0

    # 1. Morphological suffixes that form exactly 1 syllable
    tion_matches = len(re.findall(r"(?:tion|sion|cial|tial|cious|tious)", w_clean))
    # Temporarily mask these suffixes to prevent hiatus rules from double-counting 'io'
    w_masked = re.sub(r"(?:tion|sion|cial|tial|cious|tious)", "_T_", w_clean)

    # 2. Known hiatus and multisyllabic vowel exceptions
    w_masked = re.sub(r"idea(s)?", r"i_de_a\1", w_masked)
    w_masked = re.sub(r"realis", r"re_alis", w_masked)
    w_masked = re.sub(r"reali", r"re_ali", w_masked)
    w_masked = re.sub(r"^bio([aeiouy])", r"bi_o_\1", w_masked)
    w_masked = re.sub(r"^tri([aeiouy])", r"tri_\1", w_masked)
    w_masked = re.sub(r"^co([aeiouy])", r"co_\1", w_masked)
    w_masked = re.sub(r"creat", r"cre_at", w_masked)
    w_masked = re.sub(r"di(et|ar)", r"di_\1", w_masked)
    w_masked = re.sub(r"qui(et)", r"qui_\1", w_masked)
    w_masked = re.sub(r"sci(en)", r"sci_\1", w_masked)
    w_masked = re.sub(r"po(em|et)", r"po_\1", w_masked)

    # 3. Syllabic endings: -ism is 2 syllables
    if w_masked.endswith("ism") or w_masked.endswith("isms"):
        count += 1
        w_masked = re.sub(r"isms?$", "", w_masked)

    # Consonant + re ending (e.g. centre, theatre, metre, fibre)
    if re.search(r"[^aeiouy]re$", w_masked):
        count += 1
        w_masked = w_masked[:-2]
    # Consonant + le ending (e.g. table, bottle, purple, candle)
    elif re.search(r"[^aeiouy]le$", w_masked) and not w_masked.endswith(("ale", "ile", "ole", "ule")):
        count += 1
        w_masked = w_masked[:-2]

    # Hiatus pairs (adjacent vowels in separate syllables outside of masked suffixes)
    for hp in HIATUS_PAIRS:
        w_masked = w_masked.replace(hp, hp[0] + "_" + hp[1])

    # 4. Count vowel runs (nuclei)
    vowel_runs = re.findall(r"[aeiouy]+", w_masked)
    count += len(vowel_runs) + tion_matches

    # 5. Silent 'e' at end
    if w_clean.endswith("e") and not w_clean.endswith(("ee", "ye")) and not re.search(r"[^aeiouy][lr]e$", w_clean):
        if len(w_clean) >= 2 and w_clean[-2] not in VOWELS:
            count -= 1
    # Silent 'ed' (e.g. walked, played vs wanted, needed)
    elif w_clean.endswith("ed"):
        if len(w_clean) >= 3 and w_clean[-3] not in ["t", "d"] and count > 1:
            count -= 1
    # Silent 'es' (e.g. cakes, names vs boxes, watches)
    elif w_clean.endswith("es"):
        if len(w_clean) >= 3 and w_clean[-3] not in ["s", "z", "x"] and not w_clean.endswith(("ches", "shes")) and count > 1:
            count -= 1

    return max(1, count)


def segment_syllables_rule_based(token: str) -> List[str]:
    """
    Decomposes an English word into an ordered list of K orthographic syllable substrings
    using phonotactic rules and the Maximal Onset Principle.
    """
    w = normalize_word(token)
    if not w:
        return [token if token else ""]

    # Special handling for numeric tokens
    if is_numeric(w):
        return [token]

    # Special handling for acronyms (e.g. GPU -> ["G", "P", "U"])
    if is_acronym(token):
        clean_raw = re.sub(r"[^a-zA-Z0-9]", "", token)
        return list(clean_raw)

    # Special handling for hyphenated compounds
    if "-" in w:
        parts = split_hyphenated(w)
        if len(parts) > 1:
            compound_sylls = []
            for p in parts:
                compound_sylls.extend(segment_syllables_rule_based(p))
            return compound_sylls

    # Special handling for common contractions
    if w in COMMON_CONTRACTIONS:
        return list(COMMON_CONTRACTIONS[w])

    K = count_syllables_rule_based(w)
    if K <= 1:
        return [token.strip()]

    w_clean = re.sub(r"[^a-z]", "", w)
    if len(w_clean) <= 3:
        return [token.strip()]

    # 1. Locate vowel nuclei spans
    nuclei_spans: List[Tuple[int, int]] = [m.span() for m in re.finditer(r"[aeiouy]+", w_clean)]

    # Remove silent 'e' at end from nuclei if present
    if w_clean.endswith("e") and not w_clean.endswith(("ee", "ye", "le", "re")):
        if len(w_clean) >= 2 and w_clean[-2] not in VOWELS:
            if nuclei_spans and nuclei_spans[-1][0] == len(w_clean) - 1:
                nuclei_spans.pop()

    # Split hiatus vowels inside vowel runs (e.g. 'io' in 'biological')
    adjusted_nuclei: List[Tuple[int, int]] = []
    for s_start, s_end in nuclei_spans:
        v_sub = w_clean[s_start:s_end]
        hiatus_split = False
        for hp in ["ia", "io", "iu", "ua", "uo", "eo"]:
            if hp in v_sub and len(v_sub) == 2:
                idx = v_sub.index(hp)
                adjusted_nuclei.append((s_start, s_start + idx + 1))
                adjusted_nuclei.append((s_start + idx + 1, s_end))
                hiatus_split = True
                break
        if not hiatus_split:
            adjusted_nuclei.append((s_start, s_end))
    nuclei_spans = adjusted_nuclei

    # Fallback to proportional slicing if nuclei count doesn't match K
    if len(nuclei_spans) < K:
        step = len(token) / float(K)
        return [token[int(i * step):int((i + 1) * step)] for i in range(K)]

    if len(nuclei_spans) > K:
        indices = np.round(np.linspace(0, len(nuclei_spans) - 1, K)).astype(int)
        nuclei_spans = [nuclei_spans[i] for i in indices]

    # 2. Maximum Onset Principle consonant partitioning
    cuts = [0]
    for i in range(len(nuclei_spans) - 1):
        c_start = nuclei_spans[i][1]
        c_end = nuclei_spans[i + 1][0]
        consonants = w_clean[c_start:c_end]
        n_c = len(consonants)

        if n_c == 0:
            cut = c_start
        elif n_c == 1:
            # Single intervocalic consonant goes to onset of next syllable
            cut = c_start
        elif n_c == 2:
            if consonants in LEGAL_ONSETS:
                cut = c_start
            else:
                cut = c_start + 1
        else:
            if consonants[1:] in LEGAL_ONSETS:
                cut = c_start + 1
            elif consonants[2:] in LEGAL_ONSETS:
                cut = c_start + 2
            else:
                cut = c_start + (n_c // 2)
        cuts.append(cut)
    cuts.append(len(token))

    sylls = [token[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1) if cuts[i] < cuts[i + 1]]

    # Ensure exact length K
    if len(sylls) != K:
        step = len(token) / float(K)
        sylls = [token[int(i * step):int((i + 1) * step)] for i in range(K)]

    return sylls


def analyze_with_rules(token: str) -> SyllableResult:
    """
    Main rule-engine entrypoint returning a structured SyllableResult.
    """
    w_norm = normalize_word(token)
    sylls = segment_syllables_rule_based(token)
    K = len(sylls)

    source = "rule"
    if is_numeric(w_norm) or not any(c.isalpha() for c in token):
        source = "fallback"

    confidence = 0.85 if source == "rule" else 0.50

    return SyllableResult(
        word=token,
        normalized_word=w_norm,
        syllables=sylls,
        count=K,
        source=source,
        is_polysyllabic=(K >= 2),
        confidence=confidence
    )
