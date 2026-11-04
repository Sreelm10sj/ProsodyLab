"""
backend.tests.test_anchor_scanner

Comprehensive unit and regression tests for AnchorScanner in backend/training/anchor_scanner.py.
Verifies:
1. Synthetic dataset with analytically known results
2. Exact candidate filtering (strictly bisyllabic with 1 stressed, 1 unstressed)
3. Exact score calculation matching collaborator formula
4. Descending score ranking
5. selected_anchor == rank 1 candidate
6. Non-contiguous, arbitrary integer word IDs
7. max_candidates clipping and max_candidates=None behavior
8. Deterministic tie handling (stable sort)
9. Zero-norm feature protection
10. Mandatory regression on GER contextual cache (Word 1181, s=2713, u=2714, score=68.2231%)
11. Mandatory regression on ITA contextual cache (Word 1899, s=4374, u=4375, score=72.7906%)
12. Direct bitwise / floating-point cross-check against collaborator script (0.0 divergence)
"""

import os
import sys
import unittest
import numpy as np
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)
_EXPORT_DIR = os.path.join(_BACKEND_DIR, "prosodylab_model_export")
_DATASET_DIR = os.path.join(_EXPORT_DIR, "code and dataset")
_ANCHORCODE_DIR = os.path.join(_EXPORT_DIR, "anchorcode")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _ANCHORCODE_DIR not in sys.path:
    sys.path.insert(0, _ANCHORCODE_DIR)

from backend.training import AnchorScanner, AnchorCandidate, AnchorScanResult


class TestAnchorScanner(unittest.TestCase):
    """Unit and regression tests for AnchorScanner."""

    def test_01_synthetic_two_word_dataset_known_result(self):
        """1. Synthetic 2-word dataset computes analytically exact score."""
        # 2 words: Word 0: [s0, u0], Word 1: [s1, u1]
        # Total syllables = 4, total words = 2
        # Base correct = 4 - 2 * 2 = 0
        dim = 768
        X = np.zeros((4, dim), dtype=np.float32)
        # Word 0: syl 0 is stressed, syl 1 is unstressed
        # Let syl 0 be [1, 0, ...] and syl 1 be [0, 1, ...]
        X[0, 0] = 1.0  # s0
        X[1, 1] = 1.0  # u0
        # Word 1: syl 2 is stressed, syl 3 is unstressed
        # Let syl 2 be [1, 0, ...] and syl 3 be [0, 1, ...]
        X[2, 0] = 1.0  # s1
        X[3, 1] = 1.0  # u1

        Y = np.array([1, 0, 1, 0], dtype=np.int64)
        W = np.array([0, 0, 1, 1], dtype=np.int64)

        scanner = AnchorScanner(max_candidates=10)
        res = scanner.scan(X, Y, W)

        self.assertEqual(res.total_training_words, 2)
        self.assertEqual(res.total_training_syllables, 4)
        self.assertEqual(res.bisyllabic_candidates, 2)
        self.assertEqual(res.rejected_words, 0)
        # For Word 0 as anchor:
        # v_diff = X_norm[0] - X_norm[1] = [1, -1, 0...]
        # diffs = X_norm @ v_diff = [1, -1, 1, -1]
        # Word 0: diffs[s0]=1 > diffs[u0]=-1 -> correct (1)
        # Word 1: diffs[s1]=1 > diffs[u1]=-1 -> correct (1)
        # m2_correct = 2. tot_correct = base(0) + (2 + 0)*2 = 4
        # score = (4 / 4) * 100 = 100.0%
        self.assertIsNotNone(res.selected_anchor)
        self.assertAlmostEqual(res.selected_anchor.separation_score, 100.0, places=4)

    def test_02_candidate_filtering(self):
        """2. Only bisyllabic words with exactly 1 stressed and 1 unstressed are candidates."""
        dim = 768
        # 4 words:
        # Word 0: 2 syls, [1, 0] -> valid candidate
        # Word 1: 3 syls, [1, 0, 0] -> rejected (not bisyllabic)
        # Word 2: 2 syls, [1, 1] -> rejected (2 stressed)
        # Word 3: 2 syls, [0, 1] -> valid candidate
        W = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        Y = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=np.int64)
        X = np.random.randn(len(W), dim).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        scanner = AnchorScanner(max_candidates=None)
        res = scanner.scan(X, Y, W)

        self.assertEqual(res.total_training_words, 4)
        self.assertEqual(res.total_training_syllables, 9)
        self.assertEqual(res.bisyllabic_candidates, 2)
        self.assertEqual(res.rejected_words, 2)
        candidate_word_ids = {c.word_id for c in res.top_anchors}
        self.assertEqual(candidate_word_ids, {0, 3})

    def test_03_descending_ranking_and_selected_anchor(self):
        """3. Candidates are sorted descending by separation score; selected_anchor is rank 1."""
        dim = 768
        # Create 10 bisyllabic words
        w_list = []
        y_list = []
        for w in range(10):
            w_list.extend([w, w])
            y_list.extend([1, 0])
        W = np.array(w_list, dtype=np.int64)
        Y = np.array(y_list, dtype=np.int64)
        rng = np.random.default_rng(123)
        X = rng.standard_normal((len(W), dim)).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        scanner = AnchorScanner(max_candidates=None)
        res = scanner.scan(X, Y, W)

        self.assertEqual(len(res.top_anchors), 10)
        # Verify strictly descending or equal
        scores = [c.separation_score for c in res.top_anchors]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

        # Verify selected anchor matches rank 1
        self.assertEqual(res.selected_anchor.rank, 1)
        self.assertEqual(res.selected_anchor.word_id, res.top_anchors[0].word_id)
        self.assertEqual(res.selected_anchor.separation_score, scores[0])

    def test_04_non_contiguous_word_ids(self):
        """4. Scanner works seamlessly with non-contiguous, sparse word IDs."""
        dim = 768
        W = np.array([5555, 5555, 99999, 99999, 12, 12], dtype=np.int64)
        Y = np.array([1, 0, 0, 1, 1, 0], dtype=np.int64)
        rng = np.random.default_rng(99)
        X = rng.standard_normal((len(W), dim)).astype(np.float32)

        scanner = AnchorScanner()
        res = scanner.scan(X, Y, W)
        self.assertEqual(res.total_training_words, 3)
        self.assertEqual(res.bisyllabic_candidates, 3)
        self.assertEqual(set(c.word_id for c in res.top_anchors), {5555, 99999, 12})

    def test_05_max_candidates_clipping(self):
        """5. max_candidates clips top_anchors correctly; None returns all."""
        dim = 768
        w_list, y_list = [], []
        for w in range(15):
            w_list.extend([w, w])
            y_list.extend([1, 0])
        W = np.array(w_list, dtype=np.int64)
        Y = np.array(y_list, dtype=np.int64)
        rng = np.random.default_rng(77)
        X = rng.standard_normal((len(W), dim)).astype(np.float32)

        # max_candidates=5
        scanner_5 = AnchorScanner(max_candidates=5)
        res_5 = scanner_5.scan(X, Y, W)
        self.assertEqual(len(res_5.top_anchors), 5)
        self.assertEqual(res_5.bisyllabic_candidates, 15)

        # max_candidates=None
        scanner_all = AnchorScanner(max_candidates=None)
        res_all = scanner_all.scan(X, Y, W)
        self.assertEqual(len(res_all.top_anchors), 15)

    def test_06_deterministic_tie_handling(self):
        """6. Tied separation scores preserve stable candidate insertion order."""
        dim = 768
        # Create identical representations for words 0 and 1
        X = np.zeros((4, dim), dtype=np.float32)
        X[0, 0] = 1.0
        X[1, 1] = 1.0
        X[2, 0] = 1.0
        X[3, 1] = 1.0
        Y = np.array([1, 0, 1, 0], dtype=np.int64)
        W = np.array([0, 0, 1, 1], dtype=np.int64)

        scanner = AnchorScanner(max_candidates=None)
        res = scanner.scan(X, Y, W)
        # Scores are identical (100.0%)
        self.assertEqual(res.top_anchors[0].separation_score, res.top_anchors[1].separation_score)
        # Word 0 was first in W, so it must remain rank 1
        self.assertEqual(res.top_anchors[0].word_id, 0)
        self.assertEqual(res.top_anchors[1].word_id, 1)

    def test_07_zero_norm_protection(self):
        """7. Feature matrix containing zero-norm row raises ValueError clearly."""
        dim = 768
        W = np.array([0, 0, 1, 1], dtype=np.int64)
        Y = np.array([1, 0, 1, 0], dtype=np.int64)
        X = np.ones((4, dim), dtype=np.float32)
        X[2] = 0.0  # Zero vector at index 2

        scanner = AnchorScanner()
        with self.assertRaises(ValueError) as ctx:
            scanner.scan(X, Y, W)
        self.assertIn("zero-norm", str(ctx.exception).lower())

    # ==================================================================
    # MANDATORY REGRESSION TESTS AGAINST REAL COLLABORATOR CACHES
    # ==================================================================

    def test_08_ger_cache_exact_regression(self):
        """8. Mandatory regression: GER contextual cache reproduces verified collaborator anchor."""
        ger_cache_path = os.path.join(_DATASET_DIR, "ger_contextual_cache.npz")
        self.assertTrue(os.path.exists(ger_cache_path), f"Missing GER cache: {ger_cache_path}")

        data = np.load(ger_cache_path)
        X_tr = data["X_tr"]
        Y_tr = data["Y_tr"]
        W_tr = data["W_tr"]

        scanner = AnchorScanner(max_candidates=15, device="cpu")
        res = scanner.scan(X_tr, Y_tr, W_tr, language="GER")

        # Verify dataset statistics
        self.assertEqual(res.total_training_words, 3447)
        self.assertEqual(res.total_training_syllables, 7924)
        self.assertEqual(res.bisyllabic_candidates, 2628)
        self.assertEqual(res.rejected_words, 819)

        # Verify Rank 1 definitive anchor
        top1 = res.selected_anchor
        self.assertIsNotNone(top1)
        self.assertEqual(top1.rank, 1)
        self.assertEqual(top1.word_id, 1181)
        self.assertEqual(top1.stressed_syllable_index, 2713)
        self.assertEqual(top1.unstressed_syllable_index, 2714)
        self.assertAlmostEqual(top1.separation_score, 68.22312, places=3)

        # Verify Top 10 word IDs exactly match collaborator results
        expected_top10_words = [1181, 2611, 2038, 1742, 293, 1226, 2708, 2035, 2246, 603]
        actual_top10_words = [c.word_id for c in res.top_anchors[:10]]
        self.assertEqual(actual_top10_words, expected_top10_words)

        # Verify Top 10 scores
        expected_top10_scores = [
            68.2231, 67.6426, 66.5825, 66.5068, 66.1030,
            65.9768, 65.4720, 65.1439, 65.0429, 65.0177,
        ]
        for i, expected_s in enumerate(expected_top10_scores):
            self.assertAlmostEqual(res.top_anchors[i].separation_score, expected_s, places=2)

    def test_09_ita_cache_exact_regression(self):
        """9. Mandatory regression: ITA contextual cache reproduces verified collaborator anchor."""
        ita_cache_path = os.path.join(_DATASET_DIR, "ita_contextual_cache.npz")
        self.assertTrue(os.path.exists(ita_cache_path), f"Missing ITA cache: {ita_cache_path}")

        data = np.load(ita_cache_path)
        X_tr = data["X_tr"]
        Y_tr = data["Y_tr"]
        W_tr = data["W_tr"]

        scanner = AnchorScanner(max_candidates=15, device="cpu")
        res = scanner.scan(X_tr, Y_tr, W_tr, language="ITA")

        # Verify dataset statistics
        self.assertEqual(res.total_training_words, 5158)
        self.assertEqual(res.total_training_syllables, 12062)
        self.assertEqual(res.bisyllabic_candidates, 3754)
        self.assertEqual(res.rejected_words, 1404)

        # Verify Rank 1 definitive anchor
        top1 = res.selected_anchor
        self.assertIsNotNone(top1)
        self.assertEqual(top1.rank, 1)
        self.assertEqual(top1.word_id, 1899)
        self.assertEqual(top1.stressed_syllable_index, 4374)
        self.assertEqual(top1.unstressed_syllable_index, 4375)
        self.assertAlmostEqual(top1.separation_score, 72.79058, places=3)

        # Verify Top 10 word IDs exactly match collaborator results
        expected_top10_words = [1899, 4826, 3090, 3096, 2318, 4827, 1848, 1745, 3147, 2890]
        actual_top10_words = [c.word_id for c in res.top_anchors[:10]]
        self.assertEqual(actual_top10_words, expected_top10_words)

        # Verify Top 10 scores
        expected_top10_scores = [
            72.7906, 72.1273, 71.9450, 71.8952, 71.4309,
            71.2651, 71.1988, 71.0496, 70.9998, 70.8838,
        ]
        for i, expected_s in enumerate(expected_top10_scores):
            self.assertAlmostEqual(res.top_anchors[i].separation_score, expected_s, places=2)

    def test_10_collaborator_direct_cross_check(self):
        """10. Independent cross-check: 0.0 numerical difference across all 2,628 GER candidates."""
        import scan_all_ger_anchors

        # 1. Run collaborator script directly in dataset dir
        old_cwd = os.getcwd()
        try:
            os.chdir(_DATASET_DIR)
            collab_results = scan_all_ger_anchors.scan_ger_anchors()
        finally:
            os.chdir(old_cwd)

        # 2. Run AnchorScanner
        ger_cache_path = os.path.join(_DATASET_DIR, "ger_contextual_cache.npz")
        data = np.load(ger_cache_path)
        scanner = AnchorScanner(max_candidates=None, device="cpu")
        new_result = scanner.scan(data["X_tr"], data["Y_tr"], data["W_tr"], language="GER")

        self.assertEqual(len(collab_results), len(new_result.top_anchors))
        collab_scores = [r[3] for r in collab_results]
        new_scores = [c.separation_score for c in new_result.top_anchors]

        diffs = [abs(c - n) for c, n in zip(collab_scores, new_scores)]
        max_diff = max(diffs)
        self.assertLess(max_diff, 1e-5, f"Max score difference {max_diff} exceeded tolerance 1e-5.")
        # In fact, bitwise float32 equivalence on CPU gives exactly 0.0
        self.assertEqual(max_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
