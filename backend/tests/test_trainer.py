"""
backend.tests.test_trainer

Unit and integration tests for LexiRepTrainer in backend/training/trainer.py.
Verifies:
1. LexiRepTrainer execution with small reduced configuration
2. Exported bundle structure (model, prototypes, config, anchors, metrics, history)
3. Direct bundle reload and smoke inference via LexiRepPredictor.from_bundle()
4. Progress callback event stream
5. Cooperative cancellation raising TrainingCancelled
6. Critical real-data integration test using ger_contextual_cache.npz:
   AnchorScanner -> LexiRepTrainer -> Export -> LexiRepPredictor smoke test
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
_EXPORT_DIR = os.path.join(_BACKEND_DIR, "prosodylab_model_export")
_DATASET_DIR = os.path.join(_EXPORT_DIR, "code and dataset")

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _EXPORT_DIR not in sys.path:
    sys.path.insert(0, _EXPORT_DIR)

from backend.training import (
    LexiRepTrainer,
    TrainingConfig,
    TrainingProgress,
    TrainingResult,
    TrainingCancelled,
    AnchorScanner,
)
from prosodylab_model_export.lexirep_inference.stress_predictor import LexiRepPredictor


def _generate_synthetic_training_data(num_words=30, dim=768, seed=42):
    """Generates synthetic dataset with bisyllabic and trisyllabic words."""
    rng = np.random.default_rng(seed)
    w_list = []
    y_list = []

    # 25 bisyllabic words
    for w in range(25):
        w_list.extend([w, w])
        y_list.extend([1, 0])

    # 5 trisyllabic words
    for w in range(25, num_words):
        w_list.extend([w, w, w])
        y_list.extend([1, 0, 0])

    N = len(w_list)
    X = rng.standard_normal((N, dim)).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * 5.0
    Y = np.array(y_list, dtype=np.int64)
    W = np.array(w_list, dtype=np.int64)

    # Pick word 0 as anchor: syl 0 is stressed, syl 1 is unstressed
    anchor = {
        "word_id": 0,
        "stressed_syllable_index": 0,
        "unstressed_syllable_index": 1,
        "separation_score": 85.0,
    }
    return X, Y, W, anchor


class TestLexiRepTrainer(unittest.TestCase):
    """Test suite verifying LexiRepTrainer functionality and bundle compatibility."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="lexirep_train_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_trainer_reduced_config_synthetic(self):
        """1. Trainer executes successfully with tiny reduced config on synthetic data."""
        X, Y, W, anchor = _generate_synthetic_training_data()

        config = TrainingConfig(
            language="GER",
            seed=42,
            device="cpu",
            warmup_epochs=1,
            idec_pretrain_epochs=1,
            outer_loops=1,
            idec_steps_per_loop=2,
            refinement_epochs=1,
            batch_words=8,
            idec_batch_size=16,
        )

        events: list[TrainingProgress] = []
        def on_progress(p: TrainingProgress):
            events.append(p)

        trainer = LexiRepTrainer(config=config, callback=on_progress)
        res = trainer.train(X, Y, W, anchor, output_dir=self.temp_dir)

        self.assertTrue(res.success)
        self.assertEqual(res.output_dir, self.temp_dir)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[-1].status, "completed")
        self.assertEqual(events[-1].progress_percent, 100.0)

        # Verify bundle files exist
        expected_files = [
            os.path.join(self.temp_dir, "config.json"),
            os.path.join(self.temp_dir, "anchors.json"),
            os.path.join(self.temp_dir, "metrics.json"),
            os.path.join(self.temp_dir, "training_history.csv"),
            os.path.join(self.temp_dir, "model", "final_lexirep_model.pt"),
            os.path.join(self.temp_dir, "anchors", "prototypes_10d.npy"),
        ]
        for ef in expected_files:
            self.assertTrue(os.path.isfile(ef), f"Expected artifact missing: {ef}")

        # Verify prototype shape
        proto = np.load(os.path.join(self.temp_dir, "anchors", "prototypes_10d.npy"))
        self.assertEqual(proto.shape, (2, 10))
        self.assertEqual(proto.dtype, np.float32)

        # Verify real IDEC loss and training accuracy metrics
        self.assertIn("training_accuracy", res.final_metrics)
        self.assertGreater(res.final_metrics["training_accuracy"], 0.0)
        self.assertLessEqual(res.final_metrics["training_accuracy"], 1.0)
        self.assertIn("final_idec_loss", res.final_metrics)
        self.assertGreater(res.final_metrics["final_idec_loss"], 0.0)

        # Verify progress callback emitted idec_loss
        idec_events = [e for e in events if e.metrics and "idec_loss" in e.metrics]
        self.assertGreater(len(idec_events), 0)
        for ie in idec_events:
            self.assertIsInstance(ie.metrics["idec_loss"], float)
            self.assertGreater(ie.metrics["idec_loss"], 0.0)

    def test_02_bundle_reload_and_smoke_prediction(self):
        """2. Exported model bundle can be loaded and executed by LexiRepPredictor."""
        X, Y, W, anchor = _generate_synthetic_training_data()

        config = TrainingConfig(
            language="GER",
            seed=42,
            device="cpu",
            warmup_epochs=1,
            idec_pretrain_epochs=1,
            outer_loops=1,
            idec_steps_per_loop=2,
            refinement_epochs=1,
            batch_words=8,
            idec_batch_size=16,
        )

        trainer = LexiRepTrainer(config=config)
        trainer.train(X, Y, W, anchor, output_dir=self.temp_dir)

        # Load through official inference loader
        predictor = LexiRepPredictor.from_bundle(
            language="GER",
            bundle_root=self.temp_dir,
            device="cpu",
        )
        self.assertIsNotNone(predictor)

        # Test smoke inference on 3 words
        test_boundaries = [[0, 1], [2, 3], [4, 5]]
        test_features = X[:6]
        res = predictor.predict_stress(test_features, test_boundaries)

        self.assertEqual(len(res.words), 3)
        for wp in res.words:
            syl_indices = [s.syllable_index for s in wp.syllables]
            self.assertIn(wp.stressed_syllable_index, syl_indices)
            self.assertEqual(len(wp.syllables), 2)
            for syl in wp.syllables:
                self.assertTrue(np.isfinite(syl.contrast_score))
                self.assertTrue(np.isfinite(syl.confidence))

    def test_03_cooperative_cancellation(self):
        """3. Cooperative cancellation check safely halts training and raises TrainingCancelled."""
        X, Y, W, anchor = _generate_synthetic_training_data()

        config = TrainingConfig(
            language="GER",
            warmup_epochs=5,
            device="cpu",
        )

        call_count = [0]
        def should_cancel():
            call_count[0] += 1
            # Cancel after 2 checks
            return call_count[0] >= 2

        trainer = LexiRepTrainer(config=config, cancellation_check=should_cancel)
        with self.assertRaises(TrainingCancelled):
            trainer.train(X, Y, W, anchor, output_dir=self.temp_dir)

    def test_04_critical_real_data_ger_integration(self):
        """4. Critical integration: Real GER cache -> AnchorScanner -> LexiRepTrainer -> Predictor."""
        ger_cache_path = os.path.join(_DATASET_DIR, "ger_contextual_cache.npz")
        self.assertTrue(os.path.exists(ger_cache_path), f"Missing GER cache: {ger_cache_path}")

        data = np.load(ger_cache_path)
        X_tr = data["X_tr"]
        Y_tr = data["Y_tr"]
        W_tr = data["W_tr"]

        # Strict regression check on dataset split: 7924 syllables across 3447 unique words
        self.assertEqual(len(X_tr), 7924)
        self.assertEqual(len(np.unique(W_tr)), 3447)

        # 1. Anchor Search
        scanner = AnchorScanner(max_candidates=5, device="cpu")
        scan_res = scanner.scan(X_tr, Y_tr, W_tr, language="GER")

        self.assertIsNotNone(scan_res.selected_anchor)
        self.assertEqual(scan_res.selected_anchor.word_id, 1181)
        self.assertEqual(scan_res.selected_anchor.stressed_syllable_index, 2713)
        self.assertEqual(scan_res.selected_anchor.unstressed_syllable_index, 2714)

        # 2. Train with minimal loops
        config = TrainingConfig(
            language="GER",
            seed=42,
            device="cpu",
            warmup_epochs=1,
            idec_pretrain_epochs=1,
            outer_loops=1,
            idec_steps_per_loop=5,
            refinement_epochs=1,
            idec_batch_size=128,
        )
        self.assertEqual(config.top_k_rep, 1)
        self.assertEqual(config.batch_words, 32)

        trainer = LexiRepTrainer(config=config)
        res = trainer.train(X_tr, Y_tr, W_tr, scan_res.selected_anchor, output_dir=self.temp_dir)

        self.assertTrue(res.success)
        self.assertTrue(res.final_metrics["verification_passed"])

        # 3. Reload and test with LexiRepPredictor
        predictor = LexiRepPredictor.from_bundle(
            language="GER",
            bundle_root=self.temp_dir,
            device="cpu",
        )

        # Evaluate on the known 'football'-style 2-syllable word
        smoke_boundaries = [[0, 1], [2, 3]]
        smoke_features = X_tr[:4]
        smoke_res = predictor.predict_stress(smoke_features, smoke_boundaries)

        self.assertEqual(len(smoke_res.words), 2)
        for wp in smoke_res.words:
            syl_indices = [s.syllable_index for s in wp.syllables]
            self.assertIn(wp.stressed_syllable_index, syl_indices)

    def test_05_language_protocol_defaults(self):
        """5. Verify TrainingConfig automatically chooses verified protocol (GER top_k=1, ITA top_k=10)."""
        ger_cfg = TrainingConfig(language="GER")
        self.assertEqual(ger_cfg.top_k_rep, 1)
        self.assertEqual(ger_cfg.batch_words, 32)

        ita_cfg = TrainingConfig(language="ITA")
        self.assertEqual(ita_cfg.top_k_rep, 10)
        self.assertEqual(ita_cfg.batch_words, 48)

        # Explicit override takes precedence
        custom_cfg = TrainingConfig(language="GER", top_k_rep=7, batch_words=64)
        self.assertEqual(custom_cfg.top_k_rep, 7)
        self.assertEqual(custom_cfg.batch_words, 64)


if __name__ == "__main__":
    unittest.main()
