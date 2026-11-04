"""
test_lexirep_service.py
Isolated unit and equivalence test suite for LexiRepService in backend/lexirep/service.py.
Verifies:
1. GER & ITA model asset loading and caching lifecycle
2. Strict input feature and word boundary contract validation
3. Rejection of invalid languages, shapes, dtypes, and non-finite numbers
4. Bitwise / numerical equivalence between direct exported package and LexiRepService wrapper
"""

import os
import sys
import unittest
from collections import defaultdict
import numpy as np
import torch

# Ensure backend is on sys.path
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
_EXPORT_DIR = os.path.join(_BACKEND_DIR, "prosodylab_model_export")

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _EXPORT_DIR not in sys.path:
    sys.path.insert(0, _EXPORT_DIR)

from lexirep import (
    LexiRepService,
    lexirep_service,
    InferenceResult,
    FeatureValidationError,
    ModelLoadError,
)
from prosodylab_model_export.lexirep_inference import LexiRepPredictor


class TestLexiRepService(unittest.TestCase):
    """Test suite verifying LexiRepService adapter behavior and contract conformance."""

    @classmethod
    def setUpClass(cls):
        cls.service = LexiRepService(bundle_root=_EXPORT_DIR)
        cls.ger_cache_dir = os.path.join(_EXPORT_DIR, "code and dataset", "ger_contextual_cache")
        cls.ita_cache_dir = os.path.join(_EXPORT_DIR, "code and dataset", "ita_contextual_cache")

    def test_a_ger_loading(self):
        """TEST A: Verify GER model loads successfully and caches in memory."""
        self.assertFalse(self.service.is_loaded("GER"))
        predictor = self.service.load("GER")
        self.assertIsNotNone(predictor)
        self.assertIsInstance(predictor, LexiRepPredictor)
        self.assertTrue(self.service.is_loaded("GER"))
        # Verify subsequent get_predictor returns cached instance
        cached = self.service.get_predictor("GER")
        self.assertIs(predictor, cached)

    def test_b_ita_loading(self):
        """TEST B: Verify ITA model loads successfully and caches in memory."""
        self.assertFalse(self.service.is_loaded("ITA"))
        predictor = self.service.load("ITA")
        self.assertIsNotNone(predictor)
        self.assertIsInstance(predictor, LexiRepPredictor)
        self.assertTrue(self.service.is_loaded("ITA"))
        # Verify subsequent get_predictor returns cached instance
        cached = self.service.get_predictor("ITA")
        self.assertIs(predictor, cached)

    def test_c_valid_input_acceptance(self):
        """TEST C: Verify valid (N, 768) float32 features and word boundaries are accepted."""
        rng = np.random.RandomState(42)
        # 10 syllables across 3 words: lengths 3, 2, 5
        features = rng.randn(10, 768).astype(np.float32)
        word_boundaries = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
        word_texts = ["successful", "football", "communication"]

        result = self.service.predict(
            language="GER",
            features=features,
            word_boundaries=word_boundaries,
            word_texts=word_texts
        )

        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(result.language, "GER")
        self.assertEqual(result.total_words, 3)
        self.assertEqual(result.total_syllables, 10)
        self.assertTrue(result.linguistic_compliance)
        self.assertEqual(result.zero_stress_words, 0)
        self.assertEqual(result.multi_stress_words, 0)

        # Verify each word has exactly 1 stressed syllable
        for w_idx, w in enumerate(result.words):
            self.assertEqual(w.word_text, word_texts[w_idx])
            stresses = [s.stress for s in w.syllables]
            self.assertEqual(sum(stresses), 1)
            stressed_syl = [s for s in w.syllables if s.stress == 1][0]
            self.assertEqual(w.stressed_syllable_index, stressed_syl.syllable_index)
            for s in w.syllables:
                self.assertIsInstance(s.contrast_score, float)
                self.assertIsInstance(s.confidence, float)
                self.assertTrue(0.0 <= s.confidence <= 1.0)

    def test_d_invalid_input_rejection(self):
        """TEST D: Verify invalid languages, feature shapes, dtypes, values, and boundaries fail clearly."""
        rng = np.random.RandomState(42)
        valid_features = rng.randn(4, 768).astype(np.float32)
        valid_boundaries = [[0, 1], [2, 3]]

        # 1. Invalid language
        with self.assertRaises(ValueError):
            self.service.predict("FRA", valid_features, valid_boundaries)
        with self.assertRaises(ValueError):
            self.service.predict("", valid_features, valid_boundaries)
        with self.assertRaises(ValueError):
            self.service.predict(None, valid_features, valid_boundaries)

        # 2. None or non-array features
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", None, valid_boundaries)
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", [[1.0]*768], [[0]])

        # 3. Wrong feature dimensionality (not 2D)
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features.flatten(), [[0, 1, 2, 3]])
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features[np.newaxis, :, :], valid_boundaries)

        # 4. Wrong feature dimension (767, 769)
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", rng.randn(4, 767).astype(np.float32), valid_boundaries)
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", rng.randn(4, 769).astype(np.float32), valid_boundaries)

        # 5. Empty features (0 rows)
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", np.empty((0, 768), dtype=np.float32), [])

        # 6. Incompatible dtype (integers)
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", np.ones((4, 768), dtype=np.int32), valid_boundaries)

        # 7. Non-finite features (NaN, Inf)
        nan_feats = valid_features.copy()
        nan_feats[1, 10] = np.nan
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", nan_feats, valid_boundaries)

        inf_feats = valid_features.copy()
        inf_feats[2, 20] = np.inf
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", inf_feats, valid_boundaries)

        # 8. Invalid word boundaries
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features, [])  # empty
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features, [[0, 1], []])  # empty sublist
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features, [[0, 1], [2, 4]])  # index 4 out of bounds
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features, [[0, 1], [2, -1]])  # negative index
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features, [[0, 1], [1, 2, 3]])  # duplicate index 1
        with self.assertRaises(FeatureValidationError):
            self.service.predict("GER", valid_features, [[0, 1], [2]])  # missing index 3

    def test_e_ger_direct_equivalence(self):
        """TEST E: Verify exact equivalence between direct exported package and LexiRepService on GER."""
        x_path = os.path.join(self.ger_cache_dir, "X_te.npy")
        w_path = os.path.join(self.ger_cache_dir, "W_te.npy")
        self.assertTrue(os.path.exists(x_path), f"GER test cache missing: {x_path}")
        self.assertTrue(os.path.exists(w_path), f"GER test cache missing: {w_path}")

        X_full = np.load(x_path)
        W_full = np.load(w_path)

        # Extract first 15 words
        wmap = defaultdict(list)
        for idx, w in enumerate(W_full):
            wmap[w].append(idx)
        selected_words = list(wmap.keys())[:15]
        selected_boundaries = [wmap[w] for w in selected_words]
        total_syls = sum(len(b) for b in selected_boundaries)
        features_subset = X_full[:total_syls]

        # 1. Run Direct Exported Predictor
        bundle_dir = os.path.join(_EXPORT_DIR, "GER")
        direct_predictor = LexiRepPredictor.from_bundle("GER", bundle_dir)
        direct_result = direct_predictor.predict_stress(features_subset, selected_boundaries)

        # 2. Run LexiRepService Wrapper
        service_result = self.service.predict(
            language="GER",
            features=features_subset,
            word_boundaries=selected_boundaries
        )

        # 3. Assert Equivalence
        self.assertEqual(direct_result.total_words, service_result.total_words)
        self.assertEqual(direct_result.total_syllables, service_result.total_syllables)
        self.assertEqual(direct_result.linguistic_compliance, service_result.linguistic_compliance)

        max_contrast_diff = 0.0
        max_conf_diff = 0.0

        for w_d, w_s in zip(direct_result.words, service_result.words):
            # Categorical checks: exact equality
            self.assertEqual(w_d.stressed_syllable_index, w_s.stressed_syllable_index)
            self.assertEqual(len(w_d.syllables), len(w_s.syllables))

            for s_d, s_s in zip(w_d.syllables, w_s.syllables):
                self.assertEqual(s_d.syllable_index, s_s.syllable_index)
                self.assertEqual(s_d.stress, s_s.stress)

                c_diff = abs(s_d.contrast_score - s_s.contrast_score)
                conf_diff = abs(s_d.confidence - s_s.confidence)
                max_contrast_diff = max(max_contrast_diff, c_diff)
                max_conf_diff = max(max_conf_diff, conf_diff)

        print(f"\n[TEST E - GER Equivalence] Evaluated {total_syls} syllables across 15 words:")
        print(f"  Max Absolute Contrast Diff: {max_contrast_diff:.10e}")
        print(f"  Max Absolute Confidence Diff: {max_conf_diff:.10e}")
        self.assertLessEqual(max_contrast_diff, 1e-6)
        self.assertLessEqual(max_conf_diff, 1e-6)

    def test_f_ita_direct_equivalence(self):
        """TEST F: Verify exact equivalence between direct exported package and LexiRepService on ITA."""
        x_path = os.path.join(self.ita_cache_dir, "X_te.npy")
        w_path = os.path.join(self.ita_cache_dir, "W_te.npy")
        self.assertTrue(os.path.exists(x_path), f"ITA test cache missing: {x_path}")
        self.assertTrue(os.path.exists(w_path), f"ITA test cache missing: {w_path}")

        X_full = np.load(x_path)
        W_full = np.load(w_path)

        # Extract first 15 words
        wmap = defaultdict(list)
        for idx, w in enumerate(W_full):
            wmap[w].append(idx)
        selected_words = list(wmap.keys())[:15]
        selected_boundaries = [wmap[w] for w in selected_words]
        total_syls = sum(len(b) for b in selected_boundaries)
        features_subset = X_full[:total_syls]

        # 1. Run Direct Exported Predictor
        bundle_dir = os.path.join(_EXPORT_DIR, "ITA")
        direct_predictor = LexiRepPredictor.from_bundle("ITA", bundle_dir)
        direct_result = direct_predictor.predict_stress(features_subset, selected_boundaries)

        # 2. Run LexiRepService Wrapper
        service_result = self.service.predict(
            language="ITA",
            features=features_subset,
            word_boundaries=selected_boundaries
        )

        # 3. Assert Equivalence
        self.assertEqual(direct_result.total_words, service_result.total_words)
        self.assertEqual(direct_result.total_syllables, service_result.total_syllables)
        self.assertEqual(direct_result.linguistic_compliance, service_result.linguistic_compliance)

        max_contrast_diff = 0.0
        max_conf_diff = 0.0

        for w_d, w_s in zip(direct_result.words, service_result.words):
            self.assertEqual(w_d.stressed_syllable_index, w_s.stressed_syllable_index)
            self.assertEqual(len(w_d.syllables), len(w_s.syllables))

            for s_d, s_s in zip(w_d.syllables, w_s.syllables):
                self.assertEqual(s_d.syllable_index, s_s.syllable_index)
                self.assertEqual(s_d.stress, s_s.stress)

                c_diff = abs(s_d.contrast_score - s_s.contrast_score)
                conf_diff = abs(s_d.confidence - s_s.confidence)
                max_contrast_diff = max(max_contrast_diff, c_diff)
                max_conf_diff = max(max_conf_diff, conf_diff)

        print(f"\n[TEST F - ITA Equivalence] Evaluated {total_syls} syllables across 15 words:")
        print(f"  Max Absolute Contrast Diff: {max_contrast_diff:.10e}")
        print(f"  Max Absolute Confidence Diff: {max_conf_diff:.10e}")
        self.assertLessEqual(max_contrast_diff, 1e-6)
        self.assertLessEqual(max_conf_diff, 1e-6)


if __name__ == "__main__":
    unittest.main()
