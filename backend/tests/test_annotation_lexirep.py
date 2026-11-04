"""
backend/tests/test_annotation_lexirep.py

Stage 8 Unified Annotation Integration Test Suite.
Verifies:
- Backward-compatible annotation when speaker_nativity is None (LexiRep absent)
- Unified annotation with GER and ITA
- Validation error on invalid speaker_nativity
- Monosyllables bypass LexiRep (lexical_stress_index is None)
- Polysyllabic word 'football': K=2, lexical_stress_index=1, boundaries [1.140, 1.640]
- Pauses, intonation, WhiStress word prominence unchanged between without/with LexiRep
- Full regression comparison (without LexiRep vs with LexiRep)
- Single Wav2Vec extraction invocation per annotation
- FastAPI /api/annotate integration test via TestClient
"""

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent.parent)
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
for _p in [_root_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient
from backend.main import app
from backend.annotation.pipeline import run_annotation_pipeline, get_lexirep_pipeline
from whistress_service import service


class TestUnifiedAnnotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio_path = os.path.join(_backend_dir, "example1_football.wav")
        if not os.path.exists(cls.audio_path):
            raise FileNotFoundError(f"Test audio not found: {cls.audio_path}")
        with open(cls.audio_path, "rb") as f:
            cls.audio_bytes = f.read()

        # Warm up services
        service.load_model()
        cls.client = TestClient(app)

    def test_a_without_nativity_lexirep_absent(self):
        """Test A: Annotation without speaker_nativity preserves existing schema with LexiRep absent."""
        res = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity=None)

        self.assertIn("metadata", res)
        self.assertIn("transcription", res)
        self.assertIn("words", res)
        self.assertIn("pauses", res)
        self.assertIn("intonation", res)
        self.assertIn("timeline", res)

        self.assertNotIn("speaker_nativity", res["metadata"])
        self.assertNotIn("lexirep_error", res["metadata"])

        for w in res["words"]:
            self.assertIn("id", w)
            self.assertIn("text", w)
            self.assertIn("start", w)
            self.assertIn("end", w)
            self.assertIn("stress", w)
            self.assertIn(w["stress"], ("stressed", "unstressed"))
            # LexiRep additive fields must NOT be present
            self.assertNotIn("is_polysyllabic", w)
            self.assertNotIn("syllable_count", w)
            self.assertNotIn("lexical_stress_index", w)
            self.assertNotIn("syllables", w)
            self.assertNotIn("lexirep", w)

    def test_b_annotation_with_ger(self):
        """Test B: Annotation with GER includes LexiRep syllable stress and metadata."""
        res = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")

        self.assertEqual(res["metadata"].get("speaker_nativity"), "GER")
        has_poly = False
        for w in res["words"]:
            self.assertIn("is_polysyllabic", w)
            self.assertIn("syllable_count", w)
            self.assertIn("lexical_stress_index", w)
            self.assertIn("syllables", w)
            if w["is_polysyllabic"]:
                has_poly = True
                self.assertGreaterEqual(w["syllable_count"], 2)
                self.assertIsNotNone(w["lexical_stress_index"])
                self.assertIn("lexirep", w)
                self.assertEqual(w["lexirep"]["model"], "GER")
        self.assertTrue(has_poly, "Expected at least one polysyllabic word in test audio")

    def test_c_annotation_with_ita(self):
        """Test C: Annotation with ITA includes LexiRep syllable stress and metadata."""
        res = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="ITA")

        self.assertEqual(res["metadata"].get("speaker_nativity"), "ITA")
        has_poly = False
        for w in res["words"]:
            self.assertIn("is_polysyllabic", w)
            self.assertIn("syllable_count", w)
            self.assertIn("lexical_stress_index", w)
            self.assertIn("syllables", w)
            if w["is_polysyllabic"]:
                has_poly = True
                self.assertGreaterEqual(w["syllable_count"], 2)
                self.assertIsNotNone(w["lexical_stress_index"])
                self.assertIn("lexirep", w)
                self.assertEqual(w["lexirep"]["model"], "ITA")
        self.assertTrue(has_poly, "Expected at least one polysyllabic word in test audio")

    def test_d_invalid_nativity_rejection(self):
        """Test D: Invalid speaker nativity raises clear ValueError / 422 error."""
        with self.assertRaises(ValueError):
            run_annotation_pipeline(self.audio_bytes, "test.wav", speaker_nativity="SPANISH")

        with self.assertRaises(ValueError):
            run_annotation_pipeline(self.audio_bytes, "test.wav", speaker_nativity="XYZ")

        # Via API endpoint
        response = self.client.post(
            "/api/annotate",
            files={"file": ("example1_football.wav", self.audio_bytes, "audio/wav")},
            data={"speaker_nativity": "SPANISH"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unsupported speaker_nativity", response.json()["detail"])

    def test_e_monosyllables_handling(self):
        """Test E: Monosyllables bypass LexiRep (is_polysyllabic=False, lexical_stress_index=None)."""
        res = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")

        monosyllable_found = False
        for w in res["words"]:
            if not w["is_polysyllabic"]:
                monosyllable_found = True
                self.assertEqual(w["syllable_count"], 1)
                self.assertIsNone(w["lexical_stress_index"])
                self.assertNotIn("lexirep", w)
                self.assertEqual(len(w["syllables"]), 1)
                s = w["syllables"][0]
                self.assertEqual(s["index"], 0)
                self.assertIsNone(s["stress"])
                self.assertNotIn("contrast_score", s)
                self.assertNotIn("confidence", s)
        self.assertTrue(monosyllable_found, "Expected at least one monosyllable in utterance")

    def test_f_football_syllable_count_and_stress(self):
        """Test F: Word 'football' produces K=2 syllables and lexical_stress_index=1 (ball)."""
        res_ger = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")
        football_ger = next((w for w in res_ger["words"] if "football" in w["text"].lower()), None)
        self.assertIsNotNone(football_ger, "'football' must be present in transcription")
        self.assertTrue(football_ger["is_polysyllabic"])
        self.assertEqual(football_ger["syllable_count"], 2)
        self.assertEqual(football_ger["lexical_stress_index"], 1)
        self.assertEqual(len(football_ger["syllables"]), 2)
        self.assertEqual(football_ger["syllables"][0]["stress"], 0)
        self.assertEqual(football_ger["syllables"][1]["stress"], 1)

        res_ita = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="ITA")
        football_ita = next((w for w in res_ita["words"] if "football" in w["text"].lower()), None)
        self.assertIsNotNone(football_ita)
        self.assertEqual(football_ita["lexical_stress_index"], 1)
        self.assertEqual(football_ita["syllables"][0]["stress"], 0)
        self.assertEqual(football_ita["syllables"][1]["stress"], 1)

    def test_g_realigned_football_timestamps(self):
        """Test G: 'football' uses realigned acoustic timestamps [1.140, 1.640]."""
        res = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")
        football = next((w for w in res["words"] if "football" in w["text"].lower()), None)
        self.assertIsNotNone(football)
        self.assertAlmostEqual(football["start"], 1.140, delta=0.03)
        self.assertAlmostEqual(football["end"], 1.640, delta=0.03)

        # Syllable boundaries partition the realigned interval
        syl0 = football["syllables"][0]
        syl1 = football["syllables"][1]
        self.assertAlmostEqual(syl0["start"], football["start"], delta=0.01)
        self.assertAlmostEqual(syl0["end"], syl1["start"], delta=0.01)
        self.assertAlmostEqual(syl1["end"], football["end"], delta=0.01)

    def test_h_pauses_remain_identical(self):
        """Test H: Existing pauses remain identical with or without LexiRep."""
        res_none = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity=None)
        res_ger = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")
        self.assertEqual(res_none["pauses"], res_ger["pauses"])

    def test_i_intonation_remains_identical(self):
        """Test I: Existing intonation remains identical with or without LexiRep."""
        res_none = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity=None)
        res_ger = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")
        self.assertEqual(res_none["intonation"], res_ger["intonation"])

    def test_j_whistress_stress_prominence_preserved(self):
        """Test J: Existing WhiStress word stress ('stress' field) remains preserved and distinct."""
        res_none = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity=None)
        res_ger = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")

        self.assertEqual(len(res_none["words"]), len(res_ger["words"]))
        for w_none, w_ger in zip(res_none["words"], res_ger["words"]):
            self.assertEqual(w_none["text"], w_ger["text"])
            self.assertEqual(w_none["start"], w_ger["start"])
            self.assertEqual(w_none["end"], w_ger["end"])
            # Word prominence must match exactly
            self.assertEqual(w_none["stress"], w_ger["stress"])
            self.assertIn(w_ger["stress"], ("stressed", "unstressed"))

    def test_k_wav2vec_extracted_exactly_once(self):
        """Test K: LexiRep Wav2Vec2 extraction occurs strictly once per annotation pass."""
        pipeline = get_lexirep_pipeline()
        with patch.object(pipeline.feature_extractor, "extract", wraps=pipeline.feature_extractor.extract) as mock_extract:
            res = run_annotation_pipeline(
                self.audio_bytes,
                "example1_football.wav",
                speaker_nativity="GER",
                lexirep_pipeline=pipeline,
            )
            self.assertEqual(mock_extract.call_count, 1)

    def test_regression_comparison_before_and_after(self):
        """Section 16: Full regression comparison: all non-LexiRep fields are strictly identical."""
        res_before = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity=None)
        res_after = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")

        # 1. Transcription text and language
        self.assertEqual(res_before["transcription"], res_after["transcription"])

        # 2. Metadata core fields
        self.assertEqual(res_before["metadata"]["audio_file"], res_after["metadata"]["audio_file"])
        self.assertEqual(res_before["metadata"]["duration"], res_after["metadata"]["duration"])

        # 3. Pauses
        self.assertEqual(res_before["pauses"], res_after["pauses"])

        # 4. Intonation
        self.assertEqual(res_before["intonation"], res_after["intonation"])

        # 5. Timeline
        self.assertEqual(res_before["timeline"], res_after["timeline"])

        # 6. Word base fields
        self.assertEqual(len(res_before["words"]), len(res_after["words"]))
        for wb, wa in zip(res_before["words"], res_after["words"]):
            for base_field in ["id", "text", "start", "end", "stress"]:
                self.assertEqual(wb[base_field], wa[base_field])

        # 7. Pitch contour timeline fields must be absent
        self.assertNotIn("pitch_contour", res_before)
        self.assertNotIn("pitch_contour", res_after)
        self.assertNotIn("pitch_stylized", res_before)
        self.assertNotIn("pitch_stylized", res_after)
        self.assertNotIn("utterance_median_hz", res_before)
        self.assertNotIn("utterance_median_hz", res_after)

    def test_annotation_output_pitch_timeline_removed_and_canonical_fields_preserved(self):
        """
        Verify Requirement 9:
        A. Annotation still contains words, syllables, stress, confidence, intonation labels, phrase boundaries, pauses, timeline.
        B. Annotation JSON no longer contains pitch contour timeline point arrays (pitch_contour, pitch_stylized).
        C. Intonation classification still works and provides labels and slopes.
        D. Internal pitch processing executed properly.
        """
        res = run_annotation_pipeline(self.audio_bytes, "example1_football.wav", speaker_nativity="GER")

        # Core sections present
        for section in ["metadata", "transcription", "words", "pauses", "intonation", "timeline"]:
            self.assertIn(section, res)

        # Pitch contour timeline point-arrays removed
        self.assertNotIn("pitch_contour", res)
        self.assertNotIn("pitch_stylized", res)
        self.assertNotIn("utterance_median_hz", res)

        # Words have syllables, stress, confidence
        words = res["words"]
        self.assertGreater(len(words), 0)
        football_word = next((w for w in words if "football" in w["text"].lower()), None)
        self.assertIsNotNone(football_word)
        self.assertIn("syllables", football_word)
        self.assertGreaterEqual(len(football_word["syllables"]), 2)
        for syl in football_word["syllables"]:
            self.assertIn("index", syl)
            self.assertIn("text", syl)
            self.assertIn("start", syl)
            self.assertIn("end", syl)
            self.assertIn("stress", syl)
            self.assertIn("confidence", syl)

        # Intonation phrases have labels and slopes
        intonation = res["intonation"]
        self.assertGreater(len(intonation), 0)
        for phrase in intonation:
            self.assertIn("phrase_id", phrase)
            self.assertIn("start", phrase)
            self.assertIn("end", phrase)
            self.assertIn("label", phrase)
            self.assertIn(phrase["label"], ["rising", "falling", "level", "insufficient_voicing"])
            self.assertIn("slope", phrase)

        # Pauses and timeline intact
        self.assertIsInstance(res["pauses"], list)
        self.assertIsInstance(res["timeline"], list)


    def test_api_endpoint_annotate(self):
        """Test FastAPI POST /api/annotate with and without speaker_nativity."""
        # Call without speaker_nativity (backward compatible)
        resp_base = self.client.post(
            "/api/annotate",
            files={"file": ("example1_football.wav", self.audio_bytes, "audio/wav")},
        )
        self.assertEqual(resp_base.status_code, 200)
        data_base = resp_base.json()
        self.assertNotIn("speaker_nativity", data_base["metadata"])
        self.assertNotIn("is_polysyllabic", data_base["words"][0])

        # Call with speaker_nativity = "GER"
        resp_ger = self.client.post(
            "/api/annotate",
            files={"file": ("example1_football.wav", self.audio_bytes, "audio/wav")},
            data={"speaker_nativity": "GER"},
        )
        self.assertEqual(resp_ger.status_code, 200)
        data_ger = resp_ger.json()
        self.assertEqual(data_ger["metadata"]["speaker_nativity"], "GER")
        fb = next((w for w in data_ger["words"] if "football" in w["text"].lower()), None)
        self.assertIsNotNone(fb)
        self.assertTrue(fb["is_polysyllabic"])
        self.assertEqual(fb["lexical_stress_index"], 1)

        # Call with speaker_nativity = "ITA"
        resp_ita = self.client.post(
            "/api/annotate",
            files={"file": ("example1_football.wav", self.audio_bytes, "audio/wav")},
            data={"speaker_nativity": "ITA"},
        )
        self.assertEqual(resp_ita.status_code, 200)
        data_ita = resp_ita.json()
        self.assertEqual(data_ita["metadata"]["speaker_nativity"], "ITA")


if __name__ == "__main__":
    unittest.main()
