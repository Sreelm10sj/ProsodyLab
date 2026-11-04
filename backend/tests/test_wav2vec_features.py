"""
Unit tests for Stage 2: Wav2Vec2 Feature Extraction & Syllable Pooling.
Tests:
  Test A: Model Load (facebook/wav2vec2-base, hidden dim 768, eval mode)
  Test B: Full-Utterance Extraction on example1_football.wav
  Test C: Frame-Center Geometry Calculation (t_center(i) = i*0.020 + 0.0125)
  Test D: Known Phase-0 Interval Pooling Partition Recovery
  Test E: Frame Set Disjointness across adjacent syllables
  Test F: Pooling Output Shape and dtype (K, 768) float32
  Test G: Short-Syllable Fallback (< 20ms interval captures nearest center)
  Test H: Exact Collaborator Reference Feature Compatibility
  Test I: Stage 1 + Stage 2 End-to-End Local Chain (Audio -> Wav2Vec -> Pooling -> LexiRepService)
  Test J: Input Validation and Boundary Guards
"""

import os
import time
import unittest
import numpy as np
import soundfile as sf
import torch

from backend.lexirep.feature_extractor import Wav2VecFeatureExtractor
from backend.lexirep.pooling import (
    compute_frame_centers,
    assign_frames_to_interval,
    validate_syllable_intervals,
    pool_syllables,
    FRAME_STRIDE_SEC,
    FRAME_CENTER_OFFSET_SEC,
)
from backend.lexirep.service import LexiRepService


class TestWav2VecFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_audio_path = os.path.join(r"C:\Whistress\backend", "example1_football.wav")
        cls.extractor = Wav2VecFeatureExtractor(model_name="facebook/wav2vec2-base")
        t0 = time.perf_counter()
        cls.extractor.load()
        cls.model_load_time = time.perf_counter() - t0

    def test_a_model_load(self):
        """Verify model loads, runs in eval mode, and has hidden dimension 768."""
        self.assertIsNotNone(self.extractor.model)
        self.assertIsNotNone(self.extractor.processor)
        self.assertFalse(self.extractor.model.training, "Model should be in eval() mode")
        self.assertEqual(self.extractor.HIDDEN_DIMENSION, 768)
        self.assertEqual(self.extractor.model.config.hidden_size, 768)

    def test_b_full_utterance_extraction(self):
        """Extract full continuous utterance and verify dimensions and properties."""
        audio, sr = sf.read(self.example_audio_path, dtype="float32")
        duration = len(audio) / float(sr)
        self.assertEqual(sr, 16000, "Audio sample rate must be 16 kHz")

        t0 = time.perf_counter()
        hidden = self.extractor.extract(audio, sample_rate=sr)
        extraction_time = time.perf_counter() - t0

        self.assertEqual(hidden.ndim, 2)
        self.assertEqual(hidden.shape[1], 768)
        self.assertEqual(hidden.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(hidden)))

        num_frames = hidden.shape[0]
        # Verify stride relationship: T approx duration / 0.020
        expected_frames = int((len(audio) - 400) // 320) + 1
        self.assertEqual(num_frames, expected_frames)

        # Print diagnostics for report
        print(f"\n[Test B Diagnostics]")
        print(f"  Audio duration:    {duration:.3f} s ({len(audio)} samples)")
        print(f"  Extracted frames:  {num_frames} frames")
        print(f"  Extraction time:   {extraction_time * 1000:.1f} ms")
        print(f"  Hidden shape:      {hidden.shape}")

    def test_c_frame_center_calculation(self):
        """Unit test frame center formula: t_center(i) = i * 0.020 + 0.0125."""
        test_indices = [0, 1, 10, 50, 100, 250]
        max_idx = max(test_indices) + 1
        centers = compute_frame_centers(max_idx, sample_rate=16000)

        for i in test_indices:
            expected = i * 0.020 + 0.0125
            self.assertAlmostEqual(
                centers[i], expected, places=7,
                msg=f"Frame center mismatch at index {i}: got {centers[i]}, expected {expected}"
            )

        # Verify stride difference between consecutive frames
        diffs = np.diff(centers)
        self.assertTrue(np.allclose(diffs, 0.020, atol=1e-7))

    def test_d_known_interval_pooling(self):
        """Verify Phase-0 known interval partitions are reproduced exactly."""
        # Audio duration for example1_football.wav is ~5.82s (290 frames)
        num_frames = 290
        frame_centers = compute_frame_centers(num_frames, sample_rate=16000)

        # Known Phase-0 intervals:
        # really: [0.320, 0.470) -> 16..22, [0.470, 0.620) -> 23..30
        # football: [1.140, 1.390) -> 57..68, [1.390, 1.640) -> 69..81
        test_cases = [
            ("really_syl1", 0.320, 0.470, 16, 22),
            ("really_syl2", 0.470, 0.620, 23, 30),
            ("football_syl1", 1.140, 1.390, 57, 68),
            ("football_syl2", 1.390, 1.640, 69, 81),
        ]

        for name, start, end, exp_first, exp_last in test_cases:
            assigned, fallback = assign_frames_to_interval(start, end, frame_centers)
            self.assertFalse(fallback, f"Fallback should not activate for {name}")
            self.assertEqual(
                assigned[0], exp_first,
                f"{name} start frame mismatch: expected {exp_first}, got {assigned[0]}"
            )
            self.assertEqual(
                assigned[-1], exp_last,
                f"{name} end frame mismatch: expected {exp_last}, got {assigned[-1]}"
            )
            expected_count = exp_last - exp_first + 1
            self.assertEqual(
                len(assigned), expected_count,
                f"{name} count mismatch: expected {expected_count}, got {len(assigned)}"
            )

    def test_e_disjointness(self):
        """Verify adjacent syllable intervals produce strictly disjoint frame sets."""
        audio, sr = sf.read(self.example_audio_path, dtype="float32")
        hidden = self.extractor.extract(audio, sample_rate=sr)

        intervals = [
            (0.320, 0.470),
            (0.470, 0.620),
            (1.140, 1.390),
            (1.390, 1.640),
            (2.000, 2.250),
            (2.250, 2.500),
        ]

        _, diagnostics = pool_syllables(hidden, intervals, sample_rate=sr)

        for i in range(len(diagnostics) - 1):
            s1 = set(diagnostics[i]["frame_indices"])
            s2 = set(diagnostics[i + 1]["frame_indices"])
            intersection = s1.intersection(s2)
            self.assertEqual(
                len(intersection), 0,
                f"Adjacent intervals {i} and {i+1} have overlapping frames: {intersection}"
            )

    def test_f_pooling_shape_and_dtype(self):
        """Verify pooling output matrix shape (K, 768) and float32 dtype."""
        audio, sr = sf.read(self.example_audio_path, dtype="float32")
        hidden = self.extractor.extract(audio, sample_rate=sr)

        intervals = [
            (0.320, 0.470),
            (0.470, 0.620),
            (1.140, 1.390),
            (1.390, 1.640),
        ]

        X, diag = pool_syllables(hidden, intervals, sample_rate=sr)
        self.assertEqual(X.shape, (4, 768))
        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertEqual(len(diag), 4)

    def test_g_short_interval_fallback(self):
        """Verify short syllable (<20ms) with 0 frame centers uses nearest frame center."""
        num_frames = 100
        frame_centers = compute_frame_centers(num_frames, sample_rate=16000)
        # Frame 2 is at 2 * 0.020 + 0.0125 = 0.0525s
        # Frame 3 is at 3 * 0.020 + 0.0125 = 0.0725s
        # Interval [0.050, 0.051) contains zero frame centers, midpoint is 0.0505s.
        # Closest center is frame 2 (0.0525s, distance 0.0020s vs frame 1 0.0325s distance 0.0180s)
        start, end = 0.050, 0.051
        assigned, used_fallback = assign_frames_to_interval(start, end, frame_centers)

        self.assertTrue(used_fallback)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0], 2)

        # Test pooling with dummy hidden states
        dummy_hidden = np.random.randn(num_frames, 768).astype(np.float32)
        X, diag = pool_syllables(dummy_hidden, [(start, end)])
        self.assertEqual(X.shape, (1, 768))
        self.assertTrue(diag[0]["used_fallback"])
        self.assertEqual(diag[0]["frame_count"], 1)
        self.assertEqual(diag[0]["frame_indices"], [2])
        np.testing.assert_allclose(X[0], dummy_hidden[2], rtol=1e-5)

    def test_h_exact_reference_feature_compatibility(self):
        """
        Compare newly extracted features against collaborator reference output
        from C:\LexiRep\artifacts\ISLE_SESS0003_BLOCKD01_01_sprt1_features.npy
        on C:\Whistress\backend\prosodylab_model_export\isle\data\ITA\train\ISLE_SESS0003_BLOCKD01_01_sprt1.wav.
        The collaborator reference was generated on CPU.
        """
        ref_wav_path = r"C:\Whistress\backend\prosodylab_model_export\isle\data\ITA\train\ISLE_SESS0003_BLOCKD01_01_sprt1.wav"
        ref_feat_path = r"C:\LexiRep\artifacts\ISLE_SESS0003_BLOCKD01_01_sprt1_features.npy"

        if not os.path.exists(ref_wav_path) or not os.path.exists(ref_feat_path):
            self.skipTest("Reference audio or feature file not found; skipping direct bitwise check.")

        audio, sr = sf.read(ref_wav_path, dtype="float32")

        # Syllable intervals from ISLE_SESS0003_BLOCKD01_01_sprt1_syllables.json:
        intervals = [
            (0.14, 0.20),
            (0.20, 0.47),
            (0.47, 0.99),
            (0.99, 1.19),
            (1.19, 1.69),
        ]

        X_ref = np.load(ref_feat_path)

        # 1. CPU Extraction (Direct bit-level equivalence to collaborator CPU reference)
        extractor_cpu = Wav2VecFeatureExtractor(device="cpu")
        hidden_cpu = extractor_cpu.extract(audio, sample_rate=sr)
        X_cpu, _ = pool_syllables(hidden_cpu, intervals, sample_rate=sr)

        self.assertEqual(X_cpu.shape, X_ref.shape)
        max_abs_diff_cpu = float(np.max(np.abs(X_cpu - X_ref)))
        mean_abs_diff_cpu = float(np.mean(np.abs(X_cpu - X_ref)))

        print(f"\n[Test H Diagnostics - Collaborator Reference Feature Comparison]")
        print(f"  Reference shape:        {X_ref.shape}")
        print(f"  CPU extracted shape:    {X_cpu.shape}")
        print(f"  CPU Max absolute diff:  {max_abs_diff_cpu:.6e}")
        print(f"  CPU Mean absolute diff: {mean_abs_diff_cpu:.6e}")

        # Exact CPU bit-level equivalence test (machine epsilon for float32)
        self.assertLess(
            max_abs_diff_cpu, 2e-6,
            f"CPU difference from collaborator reference features too high: {max_abs_diff_cpu:.6e}"
        )

        # 2. CUDA Extraction (if available, verify cross-device numerical tolerance)
        if torch.cuda.is_available():
            extractor_cuda = Wav2VecFeatureExtractor(device="cuda")
            hidden_cuda = extractor_cuda.extract(audio, sample_rate=sr)
            X_cuda, _ = pool_syllables(hidden_cuda, intervals, sample_rate=sr)
            max_abs_diff_cuda = float(np.max(np.abs(X_cuda - X_ref)))
            mean_abs_diff_cuda = float(np.mean(np.abs(X_cuda - X_ref)))
            print(f"  CUDA Max absolute diff: {max_abs_diff_cuda:.6e}")
            print(f"  CUDA Mean abs diff:     {mean_abs_diff_cuda:.6e}")
            self.assertLess(
                max_abs_diff_cuda, 2e-3,
                f"CUDA cross-device numerical divergence too high: {max_abs_diff_cuda:.6e}"
            )

    def test_i_end_to_end_local_chain(self):
        """
        Verify complete Stage 1 + Stage 2 chain:
        Audio -> Wav2Vec Feature Extraction -> Syllable Pooling -> Stage 1 LexiRepService -> Stress Prediction.
        """
        audio, sr = sf.read(self.example_audio_path, dtype="float32")
        hidden = self.extractor.extract(audio, sample_rate=sr)

        # 'football': [1.140, 1.390), [1.390, 1.640)
        syl_intervals = [
            (1.140, 1.390),  # foot (stressed)
            (1.390, 1.640),  # ball (unstressed)
        ]

        X_football, _ = pool_syllables(hidden, syl_intervals, sample_rate=sr)
        self.assertEqual(X_football.shape, (2, 768))

        service = LexiRepService()

        for lang in ("GER", "ITA"):
            res = service.predict(
                language=lang,
                features=X_football,
                word_boundaries=[[0, 1]],
                word_texts=["football"]
            )

            self.assertTrue(res.linguistic_compliance)
            self.assertEqual(len(res.words), 1)
            word = res.words[0]
            self.assertEqual(word.word_text, "football")
            # Verify one primary stress assigned per word
            self.assertEqual(sum(s.stress for s in word.syllables), 1)
            self.assertIn(word.stressed_syllable_index, [0, 1])
            self.assertEqual(word.stressed_syllable_index, 1)
            self.assertEqual(word.syllables[1].stress, 1)
            self.assertEqual(word.syllables[0].stress, 0)
            self.assertGreater(word.syllables[1].contrast_score, word.syllables[0].contrast_score)
            self.assertGreater(word.syllables[1].confidence, 0.5)

            print(f"\n[Test I Diagnostics - {lang} End-to-End Prediction on 'football']")
            print(f"  Linguistic compliance: {res.linguistic_compliance}")
            print(f"  Stressed syl index:    {word.stressed_syllable_index}")
            print(f"  Syl 0 (foot): stress={word.syllables[0].stress}, contrast={word.syllables[0].contrast_score:+.4f}, conf={word.syllables[0].confidence:.3f}")
            print(f"  Syl 1 (ball): stress={word.syllables[1].stress}, contrast={word.syllables[1].contrast_score:+.4f}, conf={word.syllables[1].confidence:.3f}")

    def test_j_input_validation(self):
        """Verify strict error handling for malformed inputs."""
        # 1. Non-2D hidden states
        with self.assertRaises(ValueError):
            pool_syllables(np.zeros((10, 768, 2), dtype=np.float32), [(0.1, 0.2)])

        # 2. Hidden dimension not 768
        with self.assertRaises(ValueError):
            pool_syllables(np.zeros((10, 512), dtype=np.float32), [(0.1, 0.2)])

        # 3. Empty intervals
        with self.assertRaises(ValueError):
            pool_syllables(np.zeros((10, 768), dtype=np.float32), [])

        # 4. Inverted interval (start >= end)
        with self.assertRaises(ValueError):
            validate_syllable_intervals([(0.5, 0.2)])

        # 5. Overlapping intervals
        with self.assertRaises(ValueError):
            validate_syllable_intervals([(0.1, 0.4), (0.3, 0.6)])

        # 6. Negative start time
        with self.assertRaises(ValueError):
            validate_syllable_intervals([(-0.1, 0.2)])

        # 7. Non-finite interval
        with self.assertRaises(ValueError):
            validate_syllable_intervals([(0.1, float("nan"))])

        # 8. Non-finite audio
        with self.assertRaises(ValueError):
            self.extractor.extract(np.array([0.0, float("nan"), 0.1], dtype=np.float32))

        # 9. Invalid sample rate
        with self.assertRaises(ValueError):
            self.extractor.extract(np.zeros(1600, dtype=np.float32), sample_rate=22050)


if __name__ == "__main__":
    unittest.main()
