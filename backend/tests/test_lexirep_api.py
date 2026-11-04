"""
backend.tests.test_lexirep_api

Unit and integration tests for the isolated LexiRep API endpoint:
    POST /api/lexirep/analyze

Tests:
    TEST A: Route exists in FastAPI router.
    TEST B: GER request succeeds.
    TEST C: ITA request succeeds.
    TEST D: Missing nativity rejected (HTTP 422).
    TEST E: Invalid nativity rejected (HTTP 422).
    TEST F: Missing file rejected (HTTP 400/422).
    TEST G: example1_football.wav succeeds end-to-end.
    TEST H: 'football' contains exactly 2 syllables.
    TEST I: 'football' syllable intervals remain [1.140, 1.390] and [1.390, 1.640].
    TEST J: Expected frame allocation preserved indirectly.
    TEST K: Monosyllables have null LexiRep stress.
    TEST L: GER and ITA produce structurally valid responses.
"""

import os
import io
import sys
sys.path.insert(0, r"C:\Whistress\backend")
import unittest
from fastapi.testclient import TestClient

from backend.main import app
from backend.lexirep.pooling import compute_frame_centers, assign_frames_to_interval


class TestLexiRepAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.backend_dir = r"C:\Whistress\backend"
        cls.example1_wav = os.path.join(cls.backend_dir, "example1_football.wav")
        with open(cls.example1_wav, "rb") as f:
            cls.example1_bytes = f.read()

    # =========================================================================
    # TEST A: Route Exists
    # =========================================================================
    def test_a_route_exists(self):
        """Verify POST /api/lexirep/analyze is registered in FastAPI routes."""
        routes = [route.path for route in app.routes]
        self.assertIn("/api/lexirep/analyze", routes)

    # =========================================================================
    # TEST D: Missing Nativity Rejected
    # =========================================================================
    def test_d_missing_nativity_rejected(self):
        """Verify request without nativity field fails with HTTP 422."""
        files = {"file": ("test.wav", io.BytesIO(self.example1_bytes), "audio/wav")}
        response = self.client.post("/api/lexirep/analyze", files=files)
        self.assertEqual(response.status_code, 422)

    # =========================================================================
    # TEST E: Invalid Nativity Rejected
    # =========================================================================
    def test_e_invalid_nativity_rejected(self):
        """Verify request with invalid nativity (e.g. 'FRA' or '') fails with HTTP 422."""
        files = {"file": ("test.wav", io.BytesIO(self.example1_bytes), "audio/wav")}
        data = {"nativity": "FRA"}
        response = self.client.post("/api/lexirep/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unsupported nativity", response.json()["detail"])

        # Empty nativity
        files = {"file": ("test.wav", io.BytesIO(self.example1_bytes), "audio/wav")}
        data = {"nativity": "   "}
        response = self.client.post("/api/lexirep/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 422)

    # =========================================================================
    # TEST F: Missing File Rejected
    # =========================================================================
    def test_f_missing_file_rejected(self):
        """Verify request without audio file or with empty file is rejected."""
        # Missing file field entirely
        response = self.client.post("/api/lexirep/analyze", data={"nativity": "GER"})
        self.assertEqual(response.status_code, 422)

        # Empty file content
        files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
        response = self.client.post("/api/lexirep/analyze", files=files, data={"nativity": "GER"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Empty audio file", response.json()["detail"])

    # =========================================================================
    # TEST B & G & H & I & J & K: GER End-to-End on example1_football.wav
    # =========================================================================
    def test_b_g_ger_end_to_end(self):
        """
        Verify GER request succeeds on example1_football.wav and validates:
        - Route success (Test B, Test G)
        - 'football' contains 2 syllables (Test H)
        - 'football' intervals [1.140, 1.390] and [1.390, 1.640] (Test I)
        - Expected frame allocation preserved indirectly (Test J)
        - Monosyllables have null LexiRep stress (Test K)
        """
        files = {"file": ("example1_football.wav", io.BytesIO(self.example1_bytes), "audio/wav")}
        data = {"nativity": "GER"}
        response = self.client.post("/api/lexirep/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)

        json_data = response.json()
        self.assertEqual(json_data["status"], "ok")
        self.assertEqual(json_data["nativity"], "GER")
        self.assertIn("transcription", json_data)
        self.assertIn("audio", json_data)
        self.assertAlmostEqual(json_data["audio"]["duration"], 5.82, places=1)
        self.assertIn("diagnostics", json_data)
        self.assertIn("whisper_ms", json_data["diagnostics"])
        self.assertIn("wav2vec_ms", json_data["diagnostics"])

        words = json_data["words"]
        self.assertGreater(len(words), 0)

        # Find football
        football_words = [w for w in words if w["text"] == "football"]
        self.assertEqual(len(football_words), 1)
        fb_word = football_words[0]

        # TEST H: football contains 2 syllables
        self.assertTrue(fb_word["is_polysyllabic"])
        self.assertEqual(fb_word["syllable_count"], 2)
        self.assertEqual(len(fb_word["syllables"]), 2)
        self.assertEqual(fb_word["lexirep_stressed_syllable_index"], 1)

        # TEST I: syllable intervals
        s0, s1 = fb_word["syllables"][0], fb_word["syllables"][1]
        self.assertEqual(s0["text"], "foot")
        self.assertAlmostEqual(s0["start"], 1.140, places=2)
        self.assertAlmostEqual(s0["end"], 1.390, places=2)
        self.assertEqual(s1["text"], "ball")
        self.assertAlmostEqual(s1["start"], 1.390, places=2)
        self.assertAlmostEqual(s1["end"], 1.640, places=2)

        # TEST J: frame allocation preserved indirectly
        centers = compute_frame_centers(290, sample_rate=16000)
        f_foot, _ = assign_frames_to_interval(s0["start"], s0["end"], centers)
        f_ball, _ = assign_frames_to_interval(s1["start"], s1["end"], centers)
        self.assertEqual(f_foot[0], 57)
        self.assertEqual(f_foot[-1], 68)
        self.assertEqual(f_ball[0], 69)
        self.assertEqual(f_ball[-1], 81)

        # Syllable metrics
        self.assertEqual(s0["stress"], 0)
        self.assertEqual(s1["stress"], 1)
        self.assertGreater(s1["confidence"], 0.5)

        # TEST K: Monosyllable policy
        monos = [w for w in words if not w["is_polysyllabic"]]
        self.assertGreater(len(monos), 0)
        for m in monos:
            self.assertEqual(m["syllable_count"], 1)
            self.assertIsNone(m["lexirep_stressed_syllable_index"])
            s = m["syllables"][0]
            self.assertIsNone(s["stress"])
            self.assertIsNone(s["contrast_score"])
            self.assertIsNone(s["confidence"])

    # =========================================================================
    # TEST C & L: ITA End-to-End on example1_football.wav
    # =========================================================================
    def test_c_l_ita_end_to_end(self):
        """
        Verify ITA request succeeds on example1_football.wav (Test C) and
        produces a structurally valid response with correct types (Test L).
        """
        files = {"file": ("example1_football.wav", io.BytesIO(self.example1_bytes), "audio/wav")}
        data = {"nativity": "ITA"}
        response = self.client.post("/api/lexirep/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)

        json_data = response.json()
        self.assertEqual(json_data["status"], "ok")
        self.assertEqual(json_data["nativity"], "ITA")

        # Schema structural validity
        self.assertIsInstance(json_data["words"], list)
        self.assertIsInstance(json_data["diagnostics"], dict)
        self.assertIsInstance(json_data["audio"]["duration"], (int, float))

        for w in json_data["words"]:
            self.assertIsInstance(w["word_index"], int)
            self.assertIsInstance(w["text"], str)
            self.assertIsInstance(w["start"], (int, float))
            self.assertIsInstance(w["end"], (int, float))
            self.assertIsInstance(w["whistress_stressed"], bool)
            self.assertIsInstance(w["is_polysyllabic"], bool)
            self.assertIsInstance(w["syllable_count"], int)
            self.assertIsInstance(w["syllables"], list)

            for s in w["syllables"]:
                self.assertIsInstance(s["index"], int)
                self.assertIsInstance(s["text"], str)
                self.assertIsInstance(s["start"], (int, float))
                self.assertIsInstance(s["end"], (int, float))
                self.assertIsInstance(s["duration"], (int, float))
                if s["stress"] is not None:
                    self.assertIsInstance(s["stress"], int)
                    self.assertIsInstance(s["contrast_score"], (int, float))
                    self.assertIsInstance(s["confidence"], (int, float))


if __name__ == "__main__":
    unittest.main()
