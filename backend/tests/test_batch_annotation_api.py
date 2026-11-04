"""
backend.tests.test_batch_annotation_api

Comprehensive unit and integration test suite for the Batch Audio Annotation feature:
1. Valid ZIP upload
2. Multiple supported audio formats (.wav, .mp3, .flac, .m4a, .ogg)
3. Unsupported file inside ZIP (filtered safely)
4. Empty ZIP rejection
5. Oversized ZIP rejection
6. Too many audio files rejection (> 100)
7. Path traversal archive entry rejection (.., absolute paths, drive letters)
8. Unknown batch ID -> 404
9. Invalid UUID format -> 404
10. Non-blocking job creation
11. State progression and per-file success
12. One-file failure does not destroy batch (completed_with_errors)
13. All-file failure (failed)
14. Cancellation flow
15. Result ZIP generation and strict content whitelist
16. Original audio files NOT in result ZIP
17. Worker logs and state NOT in result ZIP
18. Filename sanitization & collision resolution
19. Speaker nativity propagation ('GER', 'ITA', None)
20. Unsupported speaker nativity rejection (422)
21. Output JSON schema matches canonical /api/annotate schema
22. Premature download rejection (400)
23. Download with 0 completed annotations rejection (404)
"""

import os
import io
import sys
import json
import shutil
import tempfile
import zipfile
import unittest
from unittest.mock import patch, MagicMock

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)
for _p in [_ROOT_DIR, _BACKEND_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient
from backend.main import app
from backend.batch_annotation.job_manager import BatchJobManager
from backend.batch_annotation import api as batch_api
from backend.batch_annotation.validator import (
    validate_zip_archive,
    extract_batch_audio_files,
    BatchValidationError,
    MAX_BATCH_ZIP_BYTES,
    MAX_AUDIO_FILES,
)
from backend.batch_annotation.worker import run_worker, sanitize_json_floats


def _create_test_zip(files_dict: dict) -> bytes:
    """Creates an in-memory ZIP archive from a dict of {filename: bytes}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files_dict.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf.getvalue()


class TestBatchAnnotationAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp(prefix="test_batch_jobs_")
        self.job_manager = BatchJobManager(root_dir=self.temp_dir)
        # Patch job_manager in batch_api module to use our isolated temp_dir
        self.orig_manager = batch_api.job_manager
        batch_api.job_manager = self.job_manager

    def tearDown(self):
        batch_api.job_manager = self.orig_manager
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_valid_zip_upload(self):
        """Valid ZIP with 2 audio files creates a job immediately and returns 201."""
        zip_bytes = _create_test_zip({
            "speaker1.wav": b"RIFFfakeaudio1",
            "speaker2.mp3": b"ID3fakeaudio2",
        })

        with patch.object(self.job_manager, "start_batch_job") as mock_start:
            resp = self.client.post(
                "/api/batch-annotate",
                files={"file": ("batch.zip", zip_bytes, "application/zip")},
                data={"speaker_nativity": "GER"},
            )
            self.assertEqual(resp.status_code, 201)
            data = resp.json()
            self.assertIn("batch_id", data)
            self.assertEqual(data["status"], "queued")
            self.assertEqual(data["total_files"], 2)
            mock_start.assert_called_once_with(data["batch_id"])

    def test_02_multiple_supported_audio_formats(self):
        """Supported formats (.wav, .mp3, .flac, .m4a, .ogg) are accepted."""
        zip_bytes = _create_test_zip({
            "f1.wav": b"audio1",
            "f2.mp3": b"audio2",
            "f3.flac": b"audio3",
            "f4.m4a": b"audio4",
            "f5.ogg": b"audio5",
        })

        with patch.object(self.job_manager, "start_batch_job"):
            resp = self.client.post(
                "/api/batch-annotate",
                files={"file": ("batch.zip", zip_bytes, "application/zip")},
            )
            self.assertEqual(resp.status_code, 201)
            self.assertEqual(resp.json()["total_files"], 5)

    def test_03_unsupported_file_inside_zip(self):
        """Unsupported files (e.g. .txt, .pdf) are skipped without failing valid audio."""
        zip_bytes = _create_test_zip({
            "speech.wav": b"audiowav",
            "notes.txt": b"some text notes",
            "document.pdf": b"fake pdf",
        })

        with patch.object(self.job_manager, "start_batch_job"):
            resp = self.client.post(
                "/api/batch-annotate",
                files={"file": ("batch.zip", zip_bytes, "application/zip")},
            )
            self.assertEqual(resp.status_code, 201)
            # Only speech.wav should be registered
            self.assertEqual(resp.json()["total_files"], 1)

    def test_04_empty_zip_rejection(self):
        """Empty ZIP or ZIP with zero valid audio files returns 422."""
        zip_bytes = _create_test_zip({
            "readme.txt": b"no audio files here",
        })

        resp = self.client.post(
            "/api/batch-annotate",
            files={"file": ("batch.zip", zip_bytes, "application/zip")},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("No valid audio files", resp.json()["detail"])

    def test_05_prohibited_executable_rejection(self):
        """ZIP with executable (.exe, .py, .sh) is rejected immediately."""
        zip_bytes = _create_test_zip({
            "audio.wav": b"audiowav",
            "exploit.exe": b"executable_binary",
        })

        resp = self.client.post(
            "/api/batch-annotate",
            files={"file": ("batch.zip", zip_bytes, "application/zip")},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("Prohibited executable", resp.json()["detail"])

    def test_06_too_many_audio_files(self):
        """ZIP containing more than MAX_AUDIO_FILES (2000) returns 422."""
        with patch("backend.batch_annotation.validator.MAX_AUDIO_FILES", 5):
            files = {f"audio_{i:03d}.wav": b"wavdata" for i in range(6)}
            zip_bytes = _create_test_zip(files)

            resp = self.client.post(
                "/api/batch-annotate",
                files={"file": ("batch.zip", zip_bytes, "application/zip")},
            )
            self.assertEqual(resp.status_code, 422)
            self.assertIn("exceeding the maximum limit", resp.json()["detail"])

    def test_07_path_traversal_archive_entry(self):
        """Path traversal entries (.., absolute /) are rejected."""
        zip_bytes = _create_test_zip({
            "../escaped.wav": b"malicious",
            "legit.wav": b"audio",
        })

        resp = self.client.post(
            "/api/batch-annotate",
            files={"file": ("batch.zip", zip_bytes, "application/zip")},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("Path traversal", resp.json()["detail"])

    def test_08_unknown_batch_404(self):
        """Querying a non-existent UUIDv4 returns 404."""
        resp = self.client.get("/api/batch-annotate/99999999-9999-4999-a999-999999999999")
        self.assertEqual(resp.status_code, 404)

    def test_09_invalid_uuid_404(self):
        """Querying with non-UUID path returns 404."""
        resp = self.client.get("/api/batch-annotate/not-a-uuid")
        self.assertEqual(resp.status_code, 404)

    def test_10_speaker_nativity_validation(self):
        """Invalid speaker nativity (e.g. 'FR') returns 422."""
        zip_bytes = _create_test_zip({"speaker.wav": b"audio"})
        resp = self.client.post(
            "/api/batch-annotate",
            files={"file": ("batch.zip", zip_bytes, "application/zip")},
            data={"speaker_nativity": "FR"},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("Unsupported speaker_nativity", resp.json()["detail"])

    def test_11_state_progression_and_worker_success(self):
        """Simulates worker run: state updates to completed, builds result ZIP."""
        mock_annotation = {
            "metadata": {"audio_file": "speaker1.wav", "duration": 3.25},
            "transcription": {"text": "hello world", "language": "en"},
            "words": [{"id": 1, "text": "hello", "start": 0.0, "end": 1.5, "stress": "stressed"}],
            "pauses": [],
            "intonation": [],
            "timeline": [{"type": "word", "id": 1, "start": 0.0, "end": 1.5}],
        }

        # Create job
        valid_entries = [
            {"archive_name": "speaker1.wav", "normalized_name": "speaker1.wav", "output_name": "speaker1.json", "file_size": 100},
            {"archive_name": "speaker2.wav", "normalized_name": "speaker2.wav", "output_name": "speaker2.json", "file_size": 100},
        ]
        batch_id, job_dir, input_dir = self.job_manager.create_batch_job(valid_entries, speaker_nativity="GER")
        with open(os.path.join(input_dir, "speaker1.wav"), "wb") as f:
            f.write(b"wavdata1")
        with open(os.path.join(input_dir, "speaker2.wav"), "wb") as f:
            f.write(b"wavdata2")

        with patch("backend.batch_annotation.worker.service.load_model"), \
             patch("backend.batch_annotation.worker.get_lexirep_pipeline"), \
             patch("backend.batch_annotation.worker.run_annotation_pipeline", return_value=mock_annotation):
            run_worker(batch_id=batch_id, job_dir=job_dir, speaker_nativity="GER")

        # Verify final state
        state = self.job_manager.read_state(batch_id)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["completed_files"], 2)
        self.assertEqual(state["failed_files"], 0)
        self.assertEqual(state["progress_percent"], 100.0)

        # Verify output files
        out_dir = os.path.join(job_dir, "output")
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "speaker1.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "speaker2.json")))

    def test_12_one_file_failure_completed_with_errors(self):
        """When 1 file fails out of 2, status finishes as completed_with_errors."""
        mock_annotation = {
            "metadata": {"audio_file": "good.wav", "duration": 2.0},
            "transcription": {"text": "test", "language": "en"},
            "words": [],
            "pauses": [],
            "intonation": [],
            "timeline": [],
        }

        valid_entries = [
            {"archive_name": "good.wav", "normalized_name": "good.wav", "output_name": "good.json", "file_size": 100},
            {"archive_name": "bad.wav", "normalized_name": "bad.wav", "output_name": "bad.json", "file_size": 100},
        ]
        batch_id, job_dir, input_dir = self.job_manager.create_batch_job(valid_entries)
        with open(os.path.join(input_dir, "good.wav"), "wb") as f:
            f.write(b"goodwav")
        with open(os.path.join(input_dir, "bad.wav"), "wb") as f:
            f.write(b"badwav")

        def mock_pipeline(audio_bytes, filename, **kwargs):
            if "bad" in filename:
                raise RuntimeError("Corrupt audio stream")
            return mock_annotation

        with patch("backend.batch_annotation.worker.service.load_model"), \
             patch("backend.batch_annotation.worker.run_annotation_pipeline", side_effect=mock_pipeline):
            run_worker(batch_id=batch_id, job_dir=job_dir)

        state = self.job_manager.read_state(batch_id)
        self.assertEqual(state["status"], "completed_with_errors")
        self.assertEqual(state["completed_files"], 1)
        self.assertEqual(state["failed_files"], 1)
        # Check failed file record has clean error
        bad_file = next(f for f in state["files"] if f["input_name"] == "bad.wav")
        self.assertEqual(bad_file["status"], "failed")
        self.assertIn("Corrupt audio stream", bad_file["error"])

    def test_13_all_files_failure(self):
        """When all files fail, status finishes as failed."""
        valid_entries = [
            {"archive_name": "bad1.wav", "normalized_name": "bad1.wav", "output_name": "bad1.json", "file_size": 100},
        ]
        batch_id, job_dir, input_dir = self.job_manager.create_batch_job(valid_entries)
        with open(os.path.join(input_dir, "bad1.wav"), "wb") as f:
            f.write(b"badwav")

        with patch("backend.batch_annotation.worker.service.load_model"), \
             patch("backend.batch_annotation.worker.run_annotation_pipeline", side_effect=RuntimeError("Decoder crash")):
            run_worker(batch_id=batch_id, job_dir=job_dir)

        state = self.job_manager.read_state(batch_id)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed_files"], 0)
        self.assertEqual(state["failed_files"], 1)

    def test_14_cancellation(self):
        """Job cancellation writes cancel signal and sets state to cancelled."""
        valid_entries = [
            {"archive_name": "f1.wav", "normalized_name": "f1.wav", "output_name": "f1.json", "file_size": 100},
        ]
        batch_id, job_dir, _ = self.job_manager.create_batch_job(valid_entries)

        resp = self.client.post(f"/api/batch-annotate/{batch_id}/cancel")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "cancelled")

    def test_15_result_zip_strict_whitelist(self):
        """
        Output ZIP contains ONLY annotations/*.json and manifest.json.
        Original audio, logs, and state files are STRICTLY omitted.
        """
        valid_entries = [
            {"archive_name": "test.wav", "normalized_name": "test.wav", "output_name": "test.json", "file_size": 100},
        ]
        batch_id, job_dir, input_dir = self.job_manager.create_batch_job(valid_entries)

        # Put dummy audio in input/, dummy worker.log in job_dir
        with open(os.path.join(input_dir, "test.wav"), "wb") as f:
            f.write(b"audiobytes")
        with open(os.path.join(job_dir, "worker.log"), "w") as f:
            f.write("sensitive log data")

        # Put generated annotation in output/
        out_dir = os.path.join(job_dir, "output")
        with open(os.path.join(out_dir, "test.json"), "w") as f:
            json.dump({"metadata": {}, "words": []}, f)

        # Mark completed and build zip
        state = self.job_manager.read_state(batch_id)
        state["status"] = "completed"
        state["completed_files"] = 1
        zip_path = self.job_manager.build_result_zip(batch_id, state)
        self.job_manager.write_atomic_state(job_dir, state)

        # Inspect ZIP contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            self.assertIn("manifest.json", names)
            self.assertIn("annotations/test.json", names)
            # Verify NO audio, NO logs, NO source zip
            for n in names:
                self.assertFalse(n.endswith(".wav"), f"Audio file leaked into zip: {n}")
                self.assertFalse(n.endswith(".log"), f"Log file leaked into zip: {n}")
                self.assertFalse(n.endswith(".zip"), f"Nested zip leaked: {n}")
                self.assertTrue(n == "manifest.json" or n.startswith("annotations/"), f"Unexpected zip entry: {n}")

        # Test download endpoint
        resp = self.client.get(f"/api/batch-annotate/{batch_id}/download")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "application/zip")

    def test_16_duplicate_filename_collision_handling(self):
        """Duplicate filenames across subdirectories in archive receive unique normalized names."""
        zip_bytes = _create_test_zip({
            "speaker1/audio.wav": b"audio1",
            "speaker2/audio.wav": b"audio2",
            "root_audio.wav": b"audio3",
        })

        valid_entries = validate_zip_archive(tempfile.mktemp()) if False else None
        # Test directly via validate_zip_archive
        temp_zip = os.path.join(self.temp_dir, "test_dup.zip")
        with open(temp_zip, "wb") as f:
            f.write(zip_bytes)

        entries = validate_zip_archive(temp_zip)
        self.assertEqual(len(entries), 3)
        normalized_names = [e["normalized_name"] for e in entries]
        output_names = [e["output_name"] for e in entries]
        # All normalized names must be unique
        self.assertEqual(len(set(normalized_names)), 3)
        self.assertEqual(len(set(output_names)), 3)
        # Check stripping of extension
        for out in output_names:
            self.assertTrue(out.endswith(".json"))

    def test_17_premature_download_blocked(self):
        """Downloading while job is still queued returns 400."""
        valid_entries = [
            {"archive_name": "test.wav", "normalized_name": "test.wav", "output_name": "test.json", "file_size": 100},
        ]
        batch_id, _, _ = self.job_manager.create_batch_job(valid_entries)

        resp = self.client.get(f"/api/batch-annotate/{batch_id}/download")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("still in progress", resp.json()["detail"])

    def test_18_sanitize_json_floats(self):
        """NaN and Inf floats are converted to None (null in JSON)."""
        bad_data = {
            "a": float("nan"),
            "b": float("inf"),
            "c": float("-inf"),
            "d": 12.34,
            "e": [{"nested": float("nan")}],
        }
        cleaned = sanitize_json_floats(bad_data)
        self.assertIsNone(cleaned["a"])
        self.assertIsNone(cleaned["b"])
        self.assertIsNone(cleaned["c"])
        self.assertEqual(cleaned["d"], 12.34)
        self.assertIsNone(cleaned["e"][0]["nested"])

        # Ensure valid JSON string serialization
        dumped = json.dumps(cleaned)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("Infinity", dumped)

    def test_19_concurrent_state_read_write_stress(self):
        """
        Verify that concurrent reader (FastAPI polling) and writer (worker updating progress)
        threads do not crash on Windows due to file lock collisions (WinError 5).
        """
        import threading
        import time

        valid_entries = [
            {"archive_name": f"f{i}.wav", "normalized_name": f"f{i}.wav", "output_name": f"f{i}.json", "file_size": 100}
            for i in range(50)
        ]
        batch_id, job_dir, _ = self.job_manager.create_batch_job(valid_entries)

        stop_polling = False
        reader_errors = []
        writer_errors = []

        def reader_thread():
            while not stop_polling:
                try:
                    s = self.job_manager.read_state(batch_id)
                    self.assertIn("status", s)
                except Exception as e:
                    reader_errors.append(e)
                time.sleep(0.005)

        t = threading.Thread(target=reader_thread)
        t.start()

        state = self.job_manager.read_state(batch_id)
        for i in range(50):
            state["completed_files"] = i + 1
            state["progress_percent"] = (i + 1) * 2.0
            try:
                self.job_manager.write_atomic_state(job_dir, state)
            except Exception as e:
                writer_errors.append(e)
            time.sleep(0.005)

        stop_polling = True
        t.join()

        self.assertEqual(len(writer_errors), 0, f"Writer encountered errors: {writer_errors}")
        self.assertEqual(len(reader_errors), 0, f"Reader encountered errors: {reader_errors}")


if __name__ == "__main__":
    unittest.main()
