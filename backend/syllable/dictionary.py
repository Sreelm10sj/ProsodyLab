"""
backend.syllable.dictionary

Offline dictionary lookup and two-tier SyllableEngine.
Tier 1: Offline static dictionary (exact canonical K + orthographic syllables).
Tier 2: Rule-based linguistic fallback for out-of-vocabulary (OOV) tokens.
Both tiers return identical SyllableResult structures.
"""

import os
import json
import scipy.io
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import SyllableResult
from .normalizer import normalize_word
from .rule_engine import analyze_with_rules, segment_syllables_rule_based


class SyllableDictionary:
    """
    In-memory canonical syllable dictionary mapping normalized words to:
    (canonical_count, canonical_syllable_list).
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[int, List[str]]] = {}
        self.provenance: str = "empty"

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, normalized_word: str) -> bool:
        return normalized_word.lower() in self._entries

    def lookup(self, normalized_word: str) -> Optional[Tuple[int, List[str]]]:
        """
        Retrieves the canonical (K, syllables) tuple for a normalized word.
        Returns None if OOV.
        """
        return self._entries.get(normalized_word.lower(), None)

    def add_entry(self, word: str, count: int, syllables: Sequence[str]) -> None:
        """Adds or updates a single dictionary entry."""
        if count <= 0:
            raise ValueError(f"Syllable count must be >= 1, got {count}")
        self._entries[word.lower().strip()] = (count, list(syllables))

    def load_from_dict(self, data: Dict[str, Any], provenance: str = "custom_dict") -> None:
        """
        Populates the dictionary from a Python dict mapping:
        word -> {"count": int, "syllables": list[str]} or word -> (count, list[str]).
        """
        for w, val in data.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                self.add_entry(w, int(val[0]), val[1])
            elif isinstance(val, dict):
                k = int(val.get("count", val.get("K", 1)))
                syls = val.get("syllables", val.get("ortho_sequence", []))
                if not syls:
                    syls = segment_syllables_rule_based(w)
                self.add_entry(w, k, syls)
        self.provenance = provenance

    def load_from_json(self, json_path: str) -> None:
        """Loads entries from a JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dictionary file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_from_dict(data, provenance=f"file:{os.path.basename(json_path)}")

    def build_from_isle_train(
        self,
        isle_root: str,
        languages: Sequence[str] = ("GER", "ITA")
    ) -> int:
        """
        Constructs a canonical lookup dictionary strictly from the ISLE TRAINING split.
        Guarantees zero data leakage: NEVER reads from 'test' audio or annotation directories.

        Parameters
        ----------
        isle_root : str
            Root directory containing 'data' and 'htkCorrectedLab'.
        languages : Sequence[str]
            Languages to process from the train split (default GER and ITA).

        Returns
        -------
        int
            Number of unique vocabulary entries added.
        """
        mat_word_dir = os.path.join(isle_root, r"htkCorrectedLab\lab\mat\word")
        mat_stress_dir = os.path.join(isle_root, r"htkCorrectedLab\lab\mat\sylStress")

        train_dict: Dict[str, Dict[str, Any]] = {}

        for lang in languages:
            # STRICT GUARD: Exclusively 'train' split directory
            audio_dir = os.path.join(isle_root, "data", lang, "train")
            if not os.path.exists(audio_dir):
                continue

            wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

            for wf in wav_files:
                utt_id = wf[:-4]
                w_mat_p = os.path.join(mat_word_dir, utt_id + ".mat")
                s_mat_p = os.path.join(mat_stress_dir, utt_id + ".mat")

                if not os.path.exists(w_mat_p) or not os.path.exists(s_mat_p):
                    continue

                try:
                    w_mat = scipy.io.loadmat(w_mat_p)
                    w_times = w_mat["spurtWordTimes"]
                    words = w_mat.get("words", None)

                    s_mat = scipy.io.loadmat(s_mat_p)
                    s_times = s_mat["spurtSylTimes"]
                except Exception:
                    continue

                n_w = len(w_times)
                n_s = len(s_times)
                if n_w == 0 or n_s == 0:
                    continue

                # Map syllables to words via maximum temporal overlap
                word_syl_map: List[List[int]] = [[] for _ in range(n_w)]
                for s_i in range(n_s):
                    s0, s1 = s_times[s_i, 0], s_times[s_i, 1]
                    best_w = -1
                    best_ov = 0.0
                    for w_i in range(n_w):
                        w0, w1 = w_times[w_i, 0], w_times[w_i, 1]
                        ov = max(0.0, min(s1, w1) - max(s0, w0))
                        if ov > best_ov:
                            best_ov = ov
                            best_w = w_i
                    if best_w >= 0 and best_ov > 0:
                        word_syl_map[best_w].append(s_i)

                for w_i, s_idxs in enumerate(word_syl_map):
                    if not s_idxs or words is None or w_i >= len(words):
                        continue
                    raw_w = words[w_i][0]
                    w_text = str(raw_w[0] if hasattr(raw_w, '__len__') and len(raw_w) > 0 else raw_w).strip()
                    norm_w = normalize_word(w_text)
                    if not norm_w:
                        continue

                    K = len(s_idxs)
                    if norm_w not in train_dict:
                        train_dict[norm_w] = {"counts": Counter(), "total": 0}
                    train_dict[norm_w]["counts"][K] += 1
                    train_dict[norm_w]["total"] += 1

        # Commit majority-vote canonical entries with orthographic syllable decomposition
        for norm_w, data in train_dict.items():
            canonical_K = data["counts"].most_common(1)[0][0]
            ortho_sylls = segment_syllables_rule_based(norm_w)
            if len(ortho_sylls) != canonical_K:
                # Proportional character slicing if rule segmentation diverges from canonical K
                step = len(norm_w) / float(canonical_K)
                ortho_sylls = [norm_w[int(i * step):int((i + 1) * step)] for i in range(canonical_K)]
            self.add_entry(norm_w, canonical_K, ortho_sylls)

        self.provenance = f"ISLE_TRAIN_STRICT ({len(self._entries)} types from GER/train + ITA/train)"
        return len(self._entries)


class SyllableEngine:
    """
    Two-tier Syllable Count and Segmentation Engine.
    Tier 1: SyllableDictionary (canonical dictionary lookup)
    Tier 2: Rule Engine (deterministic linguistic fallback)
    """

    def __init__(self, dictionary: Optional[SyllableDictionary] = None) -> None:
        self.dictionary = dictionary or SyllableDictionary()

    def analyze(self, token: str) -> SyllableResult:
        """
        Analyzes a word token to produce canonical syllable count K and orthographic syllables.

        Parameters
        ----------
        token : str
            Raw word token (e.g. from Whisper or user input).

        Returns
        -------
        SyllableResult
            Structured result containing word, normalized_word, syllables, count,
            source ("dictionary" or "rule"), is_polysyllabic, and confidence.
        """
        norm = normalize_word(token)
        if not norm:
            return SyllableResult(
                word=token,
                normalized_word="",
                syllables=[""],
                count=1,
                source="fallback",
                is_polysyllabic=False,
                confidence=0.0
            )

        # Tier 1: Static Dictionary Lookup
        hit = self.dictionary.lookup(norm)
        if hit is not None:
            count, syllables = hit
            return SyllableResult(
                word=token,
                normalized_word=norm,
                syllables=list(syllables),
                count=count,
                source="dictionary",
                is_polysyllabic=(count >= 2),
                confidence=1.0
            )

        # Tier 2: Rule-Based Fallback
        return analyze_with_rules(token)
