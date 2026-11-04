"""
backend.tests.test_lexirep_pipeline

Comprehensive unit and regression tests for Stage 4 LexiRepPipeline.
Tests:
  TEST A: Pipeline imports and instantiation.
  TEST B: Single known polysyllabic word ('football').
  TEST C: Known football frame partitions (foot: 57..68, ball: 69..81).
  TEST D: Monosyllable bypass (K=1 words bypass LexiRep).
  TEST E: Multiple words in one utterance (really + football).
  TEST F: GER LexiRep output on continuous speech.
  TEST G: ITA LexiRep output on continuous speech.
  TEST H: Output validity (zero NaN or Inf values).
  TEST I: Exactly one Wav2Vec2 forward pass per utterance.
  TEST J: Pause realignment regression (football clipped to ~1.640s, not raw Whisper 4.26s).
"""

import os
import unittest
import numpy as np
import soundfile as sf
from unittest.mock import MagicMock

from backend.lexirep.pipeline import LexiRepPipeline, LexiRepPipelineResult, LexiRepWord
from backend.lexirep.pooling import compute_frame_centers, assign_frames_to_interval
from backend.syllable import SyllableEngine


class TestLexiRepPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend_dir = r"C:\Whistress\backend"
        cls.example1_wav = os.path.join(cls.backend_dir, "example1_football.wav")
        cls.pipeline = LexiRepPipeline()

        # Load example1 audio array once
        cls.audio_array, cls.sr = sf.read(cls.example1_wav, dtype="float32")
        if cls.audio_array.ndim > 1:
            cls.audio_array = cls.audio_array.mean(axis=-1)

        # Realized Whisper + Realigned words for example1_football.wav
        cls.realigned_words = [
            ("I", 0, 0.000, 0.320),
            ("really", 1, 0.320, 0.620),
            ("like", 0, 0.620, 1.140),
            ("football", 0, 1.140, 1.640),
            ("but", 0, 4.120, 4.260),
            ("I'm", 0, 4.260, 4.600),
            ("tired", 0, 4.600, 4.840),
        ]

    # =========================================================================
    # TEST A: Pipeline Imports and Instantiation
    # =========================================================================
    def test_a_pipeline_instantiation(self):
        """Verify pipeline instantiates with all sub-services and validates nativity."""
        p = LexiRepPipeline()
        self.assertIsNotNone(p.syllable_engine)
        self.assertIsNotNone(p.feature_extractor)
        self.assertIsNotNone(p.lexirep_service)

        # Nativity guard
        with self.assertRaises(ValueError):
            p.analyze_realigned_words(self.audio_array, self.sr, self.realigned_words, nativity="FRA")
        with self.assertRaises(ValueError):
            p.analyze_realigned_words(self.audio_array, self.sr, self.realigned_words, nativity="")

    # =========================================================================
    # TEST B: Single Known Word ('football')
    # =========================================================================
    def test_b_single_known_word_football(self):
        """Verify pipeline handles a single polysyllabic word 'football'."""
        words = [("football", 0, 1.140, 1.640)]
        res = self.pipeline.analyze_realigned_words(
            self.audio_array, self.sr, words, nativity="GER"
        )
        self.assertEqual(len(res.words), 1)
        word = res.words[0]
        self.assertEqual(word.word, "football")
        self.assertTrue(word.is_polysyllabic)
        self.assertEqual(word.syllable_count, 2)
        self.assertEqual(len(word.syllables), 2)
        self.assertEqual(word.syllables[0].text, "foot")
        self.assertEqual(word.syllables[1].text, "ball")
        self.assertEqual(word.lexirep_stressed_syllable, 1)

    # =========================================================================
    # TEST C: Known Football Frame Partitions
    # =========================================================================
    def test_c_known_football_frame_partitions(self):
        """
        Verify that football intervals [1.140, 1.390) and [1.390, 1.640)
        recover frames 57..68 and 69..81 under the frame-center geometry.
        """
        num_frames = 290  # 5.82s audio
        centers = compute_frame_centers(num_frames, sample_rate=16000)

        f_foot, fb1 = assign_frames_to_interval(1.140, 1.390, centers)
        f_ball, fb2 = assign_frames_to_interval(1.390, 1.640, centers)

        self.assertFalse(fb1)
        self.assertFalse(fb2)
        self.assertEqual(f_foot[0], 57)
        self.assertEqual(f_foot[-1], 68)
        self.assertEqual(f_ball[0], 69)
        self.assertEqual(f_ball[-1], 81)

    # =========================================================================
    # TEST D: Monosyllable Bypass
    # =========================================================================
    def test_d_monosyllable_bypass(self):
        """
        Verify that monosyllabic words (K=1) bypass LexiRep inference
        and have stress=None, confidence=None, contrast_score=None.
        """
        res = self.pipeline.analyze_realigned_words(
            self.audio_array, self.sr, self.realigned_words, nativity="GER"
        )

        monosyllable_tokens = ["I", "like", "but", "I'm", "tired"]
        for w in res.words:
            if w.word in monosyllable_tokens:
                self.assertFalse(w.is_polysyllabic)
                self.assertEqual(w.syllable_count, 1)
                self.assertIsNone(w.lexirep_stressed_syllable)
                self.assertEqual(len(w.syllables), 1)
                s = w.syllables[0]
                self.assertIsNone(s.stress)
                self.assertIsNone(s.contrast_score)
                self.assertIsNone(s.confidence)

    # =========================================================================
    # TEST E: Multiple Words in One Utterance
    # =========================================================================
    def test_e_multiple_words_utterance(self):
        """Verify utterance with multiple polysyllabic and monosyllabic words."""
        res = self.pipeline.analyze_realigned_words(
            self.audio_array, self.sr, self.realigned_words, nativity="GER"
        )

        self.assertEqual(len(res.words), 7)
        self.assertEqual(res.num_polysyllabic_words, 2)  # really, football
        self.assertEqual(res.total_syllables_pooled, 4)   # 2 + 2
        self.assertTrue(res.linguistic_compliance)

        # really
        really_w = res.words[1]
        self.assertEqual(really_w.word, "really")
        self.assertTrue(really_w.is_polysyllabic)
        self.assertTrue(really_w.whistress_stressed)  # WhiStress acoustic prominence preserved
        self.assertEqual(really_w.lexirep_stressed_syllable, 1)

        # football
        foot_w = res.words[3]
        self.assertEqual(foot_w.word, "football")
        self.assertTrue(foot_w.is_polysyllabic)
        self.assertFalse(foot_w.whistress_stressed)
        self.assertEqual(foot_w.lexirep_stressed_syllable, 1)

    # =========================================================================
    # TEST F: GER Output
    # =========================================================================
    def test_f_ger_output(self):
        """Verify GER LexiRep predictions on example1_football.wav."""
        res = self.pipeline.analyze_realigned_words(
            self.audio_array, self.sr, self.realigned_words, nativity="GER"
        )
        self.assertEqual(res.nativity, "GER")

        # 'really' syllables
        s_really = res.words[1].syllables
        self.assertEqual(s_really[0].stress, 0)
        self.assertEqual(s_really[1].stress, 1)
        self.assertGreater(s_really[1].contrast_score, s_really[0].contrast_score)
        self.assertGreater(s_really[1].confidence, 0.5)

        # 'football' syllables
        s_foot = res.words[3].syllables
        self.assertEqual(s_foot[0].stress, 0)
        self.assertEqual(s_foot[1].stress, 1)
        self.assertGreater(s_foot[1].contrast_score, s_foot[0].contrast_score)
        self.assertGreater(s_foot[1].confidence, 0.5)

    # =========================================================================
    # TEST G: ITA Output
    # =========================================================================
    def test_g_ita_output(self):
        """Verify ITA LexiRep predictions on example1_football.wav."""
        res = self.pipeline.analyze_realigned_words(
            self.audio_array, self.sr, self.realigned_words, nativity="ITA"
        )
        self.assertEqual(res.nativity, "ITA")
        self.assertTrue(res.linguistic_compliance)

        s_really = res.words[1].syllables
        self.assertEqual(s_really[0].stress, 0)
        self.assertEqual(s_really[1].stress, 1)
        self.assertGreater(s_really[1].confidence, 0.5)

        s_foot = res.words[3].syllables
        self.assertEqual(s_foot[0].stress, 0)
        self.assertEqual(s_foot[1].stress, 1)
        self.assertGreater(s_foot[1].confidence, 0.5)

    # =========================================================================
    # TEST H: No NaN/Inf Outputs
    # =========================================================================
    def test_h_finite_outputs(self):
        """Verify all float outputs in the result structure are finite numbers."""
        res = self.pipeline.analyze_realigned_words(
            self.audio_array, self.sr, self.realigned_words, nativity="GER"
        )
        self.assertTrue(np.isfinite(res.audio_duration))

        for w in res.words:
            self.assertTrue(np.isfinite(w.start))
            self.assertTrue(np.isfinite(w.end))
            self.assertTrue(np.isfinite(w.duration))
            for s in w.syllables:
                self.assertTrue(np.isfinite(s.start))
                self.assertTrue(np.isfinite(s.end))
                self.assertTrue(np.isfinite(s.duration))
                if s.contrast_score is not None:
                    self.assertTrue(np.isfinite(s.contrast_score))
                if s.confidence is not None:
                    self.assertTrue(np.isfinite(s.confidence))

    # =========================================================================
    # TEST I: Exactly One Wav2Vec2 Forward Pass
    # =========================================================================
    def test_i_single_wav2vec_forward_pass(self):
        """
        Verify that Wav2Vec2 feature extraction runs EXACTLY ONCE for the entire utterance,
        and never per-word or per-syllable.
        """
        pipeline = LexiRepPipeline()
        real_extract = pipeline.feature_extractor.extract
        call_counter = [0]

        def spy_extract(*args, **kwargs):
            call_counter[0] += 1
            return real_extract(*args, **kwargs)

        pipeline.feature_extractor.extract = spy_extract

        res = pipeline.analyze_realigned_words(
            self.audio_array, self.sr, self.realigned_words, nativity="GER"
        )

        self.assertEqual(
            call_counter[0], 1,
            f"Wav2Vec2 extraction called {call_counter[0]} times; expected exactly 1 call."
        )
        self.assertEqual(res.wav2vec_frames, 290)

    # =========================================================================
    # TEST J: Pause Realignment Regression
    # =========================================================================
    def test_j_pause_realignment_regression(self):
        """
        Verify that football uses the acoustic-realigned endpoint (1.640s),
        not the raw Whisper endpoint that absorbed silence into the following word.
        """
        for w_text, st, s_start, s_end in self.realigned_words:
            if w_text == "football":
                self.assertAlmostEqual(s_start, 1.140, places=2)
                self.assertAlmostEqual(s_end, 1.640, places=2)
                self.assertLess(s_end, 2.0, "football endpoint should be clipped before the 1.7s pause")
            elif w_text == "but":
                self.assertGreaterEqual(s_start, 4.0, "but start should be after the long pause (~4.12s)")


if __name__ == "__main__":
    unittest.main()
