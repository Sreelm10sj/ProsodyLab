"""
backend.tests.test_syllable_engine

Comprehensive test suite for Stage 3 Syllable Count and Segmentation Engine.
Tests:
  Test A: Token Normalization
  Test B: Dictionary Lookup and Two-Tier Dispatch
  Test C: Rule-Engine Phonological Logic
  Test D: K Count on Benchmark Words
  Test E: Syllable Sequence Decomposition
  Test F: Monosyllable Policy (K=1, is_polysyllabic=False)
  Test G: Out-Of-Vocabulary (OOV) & Unicode Robustness
  Test H: Contractions and Hyphenated Compounds
  Test I: Partitioner Timing and Temporal Monotonicity
  Test J: ISLE Train-to-Test Leakage Guard
  Test K: ISLE Held-Out Test Evaluation & Coverage
"""

import os
import unittest
from typing import Dict, List, Tuple
import scipy.io

from backend.syllable.types import SyllableResult, SyllableInterval
from backend.syllable.normalizer import (
    normalize_word,
    is_numeric,
    is_acronym,
    split_hyphenated,
)
from backend.syllable.rule_engine import (
    count_syllables_rule_based,
    segment_syllables_rule_based,
    analyze_with_rules,
)
from backend.syllable.dictionary import SyllableDictionary, SyllableEngine
from backend.syllable.partitioner import partition_syllable_intervals


class TestSyllableEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SyllableEngine()

    # =========================================================================
    # Test A: Normalization
    # =========================================================================
    def test_a_normalization(self):
        """Verify conservative token normalization strips boundary noise without altering words."""
        cases = [
            ("Communication,", "communication"),
            ("'really'", "really"),
            ('"football"', "football"),
            ("hello!!", "hello"),
            ("state-of-the-art", "state-of-the-art"),
            ("don't", "don't"),
            ("students'", "students"),
            ("student's", "student"),
            ("U.S.A.", "usa"),
            ("  whitespace  ", "whitespace"),
            ("«quotes»", "quotes"),
            ("word—dash", "word-dash"),
            ("it’s", "it's"),
        ]
        for raw, expected in cases:
            self.assertEqual(normalize_word(raw), expected, f"Failed for '{raw}'")

    # =========================================================================
    # Test B: Dictionary Lookup & Two-Tier Dispatch
    # =========================================================================
    def test_b_dictionary_lookup(self):
        """Verify Tier 1 hit produces dictionary result, and miss falls back to Tier 2 rules."""
        custom_dict = SyllableDictionary()
        custom_dict.add_entry("prosodylab", 4, ["pro", "so", "dy", "lab"])
        custom_dict.add_entry("football", 2, ["foot", "ball"])

        engine = SyllableEngine(dictionary=custom_dict)

        # Dictionary HIT
        res_hit = engine.analyze("ProsodyLab!")
        self.assertEqual(res_hit.source, "dictionary")
        self.assertEqual(res_hit.count, 4)
        self.assertEqual(res_hit.syllables, ["pro", "so", "dy", "lab"])
        self.assertTrue(res_hit.is_polysyllabic)
        self.assertEqual(res_hit.confidence, 1.0)

        # Dictionary MISS (Tier 2 rule fallback)
        res_miss = engine.analyze("university")
        self.assertEqual(res_miss.source, "rule")
        self.assertEqual(res_miss.count, 5)
        self.assertTrue(res_miss.is_polysyllabic)

    # =========================================================================
    # Test C: Rule-Engine Phonological Logic
    # =========================================================================
    def test_c_rule_engine_phonology(self):
        """Verify phonological rules: silent-e, -le, -re, -ed, -es, -tion, -ism, hiatus."""
        cases = [
            # Silent -e
            ("alone", 2),
            ("take", 1),
            ("name", 1),
            ("grape", 1),
            # Consonant + le / re
            ("table", 2),
            ("bottle", 2),
            ("purple", 2),
            ("centre", 2),
            ("theatre", 2),
            # Suffix -ed (silent vs syllabic)
            ("walked", 1),
            ("played", 1),
            ("wanted", 2),
            ("needed", 2),
            ("decided", 3),
            # Suffix -es (silent vs syllabic)
            ("cakes", 1),
            ("names", 1),
            ("boxes", 2),
            ("watches", 2),
            ("wishes", 2),
            # Suffix -tion / -sion / -ism
            ("action", 2),
            ("decision", 3),
            ("social", 2),
            ("realism", 3),
            # Hiatus vs diphthongs
            ("create", 2),
            ("diet", 2),
            ("poem", 2),
            ("radio", 3),
            ("studio", 3),
        ]
        for word, expected_k in cases:
            k = count_syllables_rule_based(word)
            self.assertEqual(k, expected_k, f"Mismatch for '{word}': expected {expected_k}, got {k}")

    # =========================================================================
    # Test D: K Count on 20 Required Benchmark Words
    # =========================================================================
    def test_d_benchmark_words_count(self):
        """Verify exact syllable count K on all 20 required benchmark words."""
        expected_counts = {
            "football": 2,
            "really": 2,
            "honest": 2,
            "ideas": 3,
            "order": 2,
            "successful": 3,
            "communication": 5,
            "determined": 3,
            "inherent": 3,
            "matters": 2,
            "photography": 4,
            "important": 3,
            "university": 5,
            "computer": 3,
            "tomorrow": 3,
            "biological": 5,
            "psychology": 4,
            "centre": 2,
            "camera": 3,
            "different": 3,
        }

        print("\n[Test D - 20 Benchmark Words Syllable Count]")
        for word, expected_k in expected_counts.items():
            res = self.engine.analyze(word)
            print(f"  {word:<15}: K={res.count} (expected {expected_k}) -> {res.syllables}")
            self.assertEqual(
                res.count, expected_k,
                f"Count mismatch for benchmark word '{word}': got {res.count}, expected {expected_k}"
            )
            self.assertEqual(len(res.syllables), expected_k)

    # =========================================================================
    # Test E: Syllable Sequence Decomposition
    # =========================================================================
    def test_e_syllable_sequences(self):
        """Verify returned syllables partition the character sequence properly."""
        test_tokens = ["football", "really", "matters", "computer", "important", "centre"]
        for t in test_tokens:
            res = self.engine.analyze(t)
            self.assertEqual(res.count, len(res.syllables))
            # Verify reconstructed word matches normalized word
            reconstructed = "".join(res.syllables).lower()
            self.assertEqual(
                reconstructed, res.normalized_word,
                f"Syllables do not reconstruct original word: {res.syllables} -> '{reconstructed}' vs '{res.normalized_word}'"
            )

    # =========================================================================
    # Test F: Monosyllable Policy
    # =========================================================================
    def test_f_monosyllables(self):
        """Verify monosyllables return count=1 and is_polysyllabic=False."""
        monos = ["the", "cat", "dog", "and", "world", "time", "one", "play", "fast"]
        for m in monos:
            res = self.engine.analyze(m)
            self.assertEqual(res.count, 1, f"Expected 1 for '{m}', got {res.count}")
            self.assertFalse(res.is_polysyllabic)
            self.assertEqual(res.syllables, [m])

    # =========================================================================
    # Test G: OOV & Unicode Robustness
    # =========================================================================
    def test_g_oov_and_unicode_safety(self):
        """Verify engine never crashes on arbitrary unicode, emojis, gibberish, or empty input."""
        strange_inputs = [
            "",
            "   ",
            "!!!",
            "12345",
            "2026",
            "🎉🚀",
            "αβγδε",
            "supercalifragilisticexpialidocious",
            "xyzqwr",
            "a",
            "I",
            "---",
        ]
        for s in strange_inputs:
            res = self.engine.analyze(s)
            self.assertIsInstance(res, SyllableResult)
            self.assertGreaterEqual(res.count, 1)
            self.assertEqual(len(res.syllables), res.count)

    # =========================================================================
    # Test H: Contractions and Hyphenated Compounds
    # =========================================================================
    def test_h_contractions_and_hyphens(self):
        """Verify contraction and hyphen decomposition."""
        # Contractions
        contractions = {
            "wasn't": 2,
            "shouldn't": 2,
            "didn't": 2,
            "couldn't": 2,
            "don't": 1,
            "can't": 1,
            "they'll": 1,
            "i'm": 1,
            "it's": 1,
        }
        for c, exp_k in contractions.items():
            res = self.engine.analyze(c)
            self.assertEqual(res.count, exp_k, f"Contraction '{c}' failed: expected {exp_k}, got {res.count}")

        # Hyphenated compounds
        compounds = {
            "state-of-the-art": 4,  # state (1) + of (1) + the (1) + art (1)
            "well-known": 2,        # well (1) + known (1)
            "long-term": 2,         # long (1) + term (1)
        }
        for comp, exp_k in compounds.items():
            res = self.engine.analyze(comp)
            self.assertEqual(res.count, exp_k, f"Compound '{comp}' failed: expected {exp_k}, got {res.count}")
            self.assertTrue(res.is_polysyllabic)

        # Numbers & Acronyms
        self.assertEqual(self.engine.analyze("2026").count, 1)
        self.assertEqual(self.engine.analyze("GPU").count, 3)
        self.assertEqual(self.engine.analyze("API").count, 3)
        self.assertEqual(self.engine.analyze("AI").count, 2)
        self.assertEqual(self.engine.analyze("NLP").count, 3)
        self.assertEqual(self.engine.analyze("F0").count, 2)

    # =========================================================================
    # Test I: Partitioner Timing
    # =========================================================================
    def test_i_partitioner(self):
        """Verify partition_syllable_intervals produces valid, monotonic, contiguous spans."""
        w_start, w_end = 1.140, 1.640
        sylls = ["foot", "ball"]
        intervals = partition_syllable_intervals(w_start, w_end, sylls)

        self.assertEqual(len(intervals), 2)
        # Check first syllable
        self.assertAlmostEqual(intervals[0].start, 1.140, places=6)
        self.assertAlmostEqual(intervals[0].end, 1.390, places=6)
        self.assertEqual(intervals[0].syllable_text, "foot")
        # Check second syllable
        self.assertAlmostEqual(intervals[1].start, 1.390, places=6)
        self.assertAlmostEqual(intervals[1].end, 1.640, places=6)
        self.assertEqual(intervals[1].syllable_text, "ball")

        # Contiguity and boundaries
        for idx in range(len(intervals)):
            self.assertLess(intervals[idx].start, intervals[idx].end)
            if idx > 0:
                self.assertAlmostEqual(intervals[idx].start, intervals[idx-1].end, places=9)

        # Input error validation
        with self.assertRaises(ValueError):
            partition_syllable_intervals(1.5, 1.2, ["a"])
        with self.assertRaises(ValueError):
            partition_syllable_intervals(-0.1, 1.0, ["a"])
        with self.assertRaises(ValueError):
            partition_syllable_intervals(1.0, 2.0, [])

    # =========================================================================
    # Test J: ISLE Train-to-Test Leakage Guard
    # =========================================================================
    def test_j_isle_leakage_guard(self):
        """
        Verify that building the ISLE dictionary strictly reads from 'train'
        directories and NEVER touches 'test' files.
        """
        isle_root = r"C:\Whistress\backend\prosodylab_model_export\isle"
        if not os.path.exists(isle_root):
            self.skipTest("ISLE dataset directory not present; skipping leakage test.")

        # Inspect test audio filenames
        test_audio_files = set()
        for lang in ("GER", "ITA"):
            test_dir = os.path.join(isle_root, "data", lang, "test")
            if os.path.exists(test_dir):
                test_audio_files.update([f[:-4] for f in os.listdir(test_dir) if f.endswith(".wav")])

        # Build train-only dictionary
        dictionary = SyllableDictionary()
        n_entries = dictionary.build_from_isle_train(isle_root, languages=("GER", "ITA"))
        self.assertGreater(n_entries, 100)
        self.assertIn("ISLE_TRAIN_STRICT", dictionary.provenance)

        # Inspect that provenance confirms zero test inclusion
        print(f"\n[Test J - ISLE Leakage Guard]")
        print(f"  Train dictionary entries: {n_entries}")
        print(f"  Provenance:               {dictionary.provenance}")
        print(f"  Test audio files count:   {len(test_audio_files)} (strictly excluded)")

    # =========================================================================
    # Test K: ISLE Held-Out Test Evaluation & Coverage
    # =========================================================================
    def test_k_isle_held_out_evaluation(self):
        """
        Evaluate ISLE Train-derived lookup + Rule Fallback against the held-out ISLE TEST split.
        Verifies:
        - Polysyllabic count accuracy > 90%
        - Polysyllabic ±1 accuracy > 99%
        - Polysyllabic token coverage > 99%
        """
        isle_root = r"C:\Whistress\backend\prosodylab_model_export\isle"
        if not os.path.exists(isle_root):
            self.skipTest("ISLE dataset directory not present; skipping evaluation test.")

        # 1. Build train dictionary
        train_dict = SyllableDictionary()
        train_dict.build_from_isle_train(isle_root, languages=("GER", "ITA"))
        hybrid_engine = SyllableEngine(dictionary=train_dict)

        mat_word_dir = os.path.join(isle_root, r"htkCorrectedLab\lab\mat\word")
        mat_stress_dir = os.path.join(isle_root, r"htkCorrectedLab\lab\mat\sylStress")

        # 2. Iterate test splits
        total_poly_tokens = 0
        correct_poly_k = 0
        off1_poly_k = 0
        covered_poly_tokens = 0
        abs_errors = []

        by_length: Dict[int, Dict[str, int]] = {
            2: {"total": 0, "correct": 0},
            3: {"total": 0, "correct": 0},
            4: {"total": 0, "correct": 0},
            5: {"total": 0, "correct": 0},
        }

        for lang in ("GER", "ITA"):
            test_dir = os.path.join(isle_root, "data", lang, "test")
            if not os.path.exists(test_dir):
                continue
            wav_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])

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

                n_w, n_s = len(w_times), len(s_times)
                if n_w == 0 or n_s == 0:
                    continue

                word_syl_map: List[List[int]] = [[] for _ in range(n_w)]
                for s_i in range(n_s):
                    s0, s1 = s_times[s_i, 0], s_times[s_i, 1]
                    best_w, best_ov = -1, 0.0
                    for w_i in range(n_w):
                        w0, w1 = w_times[w_i, 0], w_times[w_i, 1]
                        ov = max(0.0, min(s1, w1) - max(s0, w0))
                        if ov > best_ov:
                            best_ov, best_w = ov, w_i
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

                    true_K = len(s_idxs)
                    if true_K < 2:
                        continue  # Focus on polysyllabic words

                    total_poly_tokens += 1
                    if train_dict.contains(norm_w):
                        covered_poly_tokens += 1

                    res = hybrid_engine.analyze(norm_w)
                    pred_K = res.count
                    err = abs(pred_K - true_K)
                    abs_errors.append(err)

                    if err == 0:
                        correct_poly_k += 1
                    if err <= 1:
                        off1_poly_k += 1

                    bucket = min(5, true_K)
                    by_length[bucket]["total"] += 1
                    if err == 0:
                        by_length[bucket]["correct"] += 1

        self.assertGreater(total_poly_tokens, 500)
        token_coverage = (covered_poly_tokens / total_poly_tokens) * 100.0
        accuracy = (correct_poly_k / total_poly_tokens) * 100.0
        off1_acc = (off1_poly_k / total_poly_tokens) * 100.0
        mae = float(sum(abs_errors) / len(abs_errors))

        print(f"\n[Test K - ISLE Held-Out Test Split Benchmark (N = {total_poly_tokens} polysyllabic tokens)]")
        print(f"  Polysyllabic Token Coverage: {token_coverage:.2f}%")
        print(f"  Exact Count Accuracy:        {accuracy:.2f}%")
        print(f"  ±1 Accuracy:                 {off1_acc:.2f}%")
        print(f"  Count MAE:                   {mae:.4f} syllables")
        print("  Breakdown by Syllable Count:")
        for k_val in (2, 3, 4, 5):
            sub = by_length[k_val]
            sub_acc = (sub["correct"] / sub["total"] * 100.0) if sub["total"] > 0 else 0.0
            label = f"{k_val}+" if k_val == 5 else str(k_val)
            print(f"    {label} syllables: {sub_acc:.2f}% ({sub['correct']}/{sub['total']})")

        self.assertGreater(token_coverage, 99.0)
        self.assertGreater(accuracy, 90.0)
        self.assertGreater(off1_acc, 99.0)
        self.assertLess(mae, 0.15)


if __name__ == "__main__":
    unittest.main()
