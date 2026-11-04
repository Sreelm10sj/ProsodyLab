"""
backend.tests.test_evaluator

Unit tests for the reconstructed LexiRep benchmark evaluator:
1. Exact reproduction of the production GER benchmark:
   - Bi-Syllabic (B): 79.91%
   - Bi + Tri (BT): 80.20%
   - Bi + Tri + Quad (BTQ): 79.96%
   - Balanced Accuracy: 79.70%
   - Total Syllable Accuracy: 79.83%
   - Total Word Accuracy: 76.84%
2. BenchmarkResult serialization to dict and JSON.
3. Edge cases and empty/mismatched input validation.
"""

import os
import unittest
import numpy as np

from backend.training.evaluator import (
    evaluate_bundle_on_dataset,
    evaluate_lexirep_benchmark,
    BenchmarkResult,
)
from prosodylab_model_export.lexirep_inference.stress_predictor import LexiRepPredictor

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TEST_DIR)
_GER_BUNDLE_DIR = os.path.join(_BACKEND_DIR, "prosodylab_model_export", "GER")
_GER_DATASET_PATH = "C:/LexiRep/new_code/ger_contextual_cache.npz"


class TestLexiRepBenchmarkEvaluator(unittest.TestCase):

    def setUp(self):
        self.dataset_available = os.path.isfile(_GER_DATASET_PATH)
        self.bundle_available = os.path.isdir(_GER_BUNDLE_DIR)

    def test_production_ger_reproduction(self):
        """Verify the production GER bundle reproduces historical benchmarks exactly."""
        if not self.dataset_available or not self.bundle_available:
            self.skipTest("Dataset or production bundle not available locally.")

        res = evaluate_bundle_on_dataset(
            bundle_path=_GER_BUNDLE_DIR,
            dataset_path=_GER_DATASET_PATH,
            language="GER",
            device="cpu",
        )

        # Expected historical figures
        self.assertEqual(res.total_syllables, 7000)
        self.assertEqual(res.total_words, 3048)

        self.assertAlmostEqual(res.b_accuracy, 79.91, places=1)
        self.assertAlmostEqual(res.bt_accuracy, 80.20, places=1)
        self.assertAlmostEqual(res.btq_accuracy, 79.96, places=1)
        self.assertAlmostEqual(res.balanced_accuracy, 79.70, places=1)
        self.assertAlmostEqual(res.total_syllable_accuracy, 79.83, places=1)
        self.assertAlmostEqual(res.word_accuracy, 76.84, places=1)

        # Exact underlying counts
        self.assertEqual(res.b_syllables, 4670)
        self.assertEqual(res.b_correct_syllables, 3732)
        self.assertEqual(res.bt_syllables, 6383)
        self.assertEqual(res.bt_correct_syllables, 5119)
        self.assertEqual(res.btq_syllables, 6775)
        self.assertEqual(res.btq_correct_syllables, 5417)
        self.assertEqual(res.correct_syllables, 5588)
        self.assertEqual(res.correct_words, 2342)

    def test_benchmark_result_serialization(self):
        """Verify BenchmarkResult dataclass correctly serializes to dict."""
        res = BenchmarkResult(
            b_accuracy=79.91,
            bt_accuracy=80.20,
            btq_accuracy=79.96,
            balanced_accuracy=79.70,
            total_syllable_accuracy=79.83,
            word_accuracy=76.84,
            total_syllables=7000,
            total_words=3048,
            correct_syllables=5588,
            correct_words=2342,
            b_words=2335,
            b_syllables=4670,
            b_correct_syllables=3732,
            bt_words=2906,
            bt_syllables=6383,
            bt_correct_syllables=5119,
            btq_words=3004,
            btq_syllables=6775,
            btq_correct_syllables=5417,
            word_length_distribution={2: 2335, 3: 571, 4: 98},
            word_length_accuracies={2: 78.5, 3: 75.2},
        )
        d = res.to_dict()
        self.assertIn("b_accuracy", d)
        self.assertIn("balanced_accuracy", d)
        self.assertIn("word_length_distribution", d)
        self.assertIsInstance(d["word_length_distribution"], dict)
        self.assertEqual(d["word_accuracy"], 76.84)

    def test_dimension_mismatch_validation(self):
        """Ensure evaluator raises ValueError when input dimensions mismatch."""
        if not self.bundle_available:
            self.skipTest("Production bundle not available locally.")

        predictor = LexiRepPredictor.from_bundle("GER", _GER_BUNDLE_DIR, device="cpu")
        with self.assertRaises(ValueError):
            evaluate_lexirep_benchmark(
                predictor,
                X_test=np.zeros((10, 768), dtype=np.float32),
                Y_test=np.zeros((8,), dtype=np.int64),
                W_test=np.zeros((10,), dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
