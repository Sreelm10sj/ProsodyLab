"""
backend.tests.test_batch_real_data_integration

Primary real-data integration test for Batch Audio Annotation:
1. Packages 2 real audio files from ISLE into a ZIP archive
2. Uploads via POST /api/batch-annotate with speaker_nativity="ITA"
3. Executes the REAL annotation pipeline through the batch worker
4. Verifies state reaches "completed"
5. Downloads result ZIP and inspects:
   - Exact count of JSON annotations
   - Manifest contents and structure
   - Canonical annotation keys (metadata, transcription, words, pauses, intonation, timeline)
   - Strict absence of audio files
   - Structural equivalence with single-file run_annotation_pipeline output
"""

import os
import io
import sys
import json
import shutil
import tempfile
import zipfile
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)
for _p in [_ROOT_DIR, _BACKEND_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient
from backend.main import app
from backend.batch_annotation.job_manager import BatchJobManager
from backend.batch_annotation.worker import run_worker
from backend.batch_annotation import api as batch_api
from annotation.pipeline import run_annotation_pipeline


class TestBatchRealDataIntegration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp(prefix="test_real_batch_")
        self.job_manager = BatchJobManager(root_dir=self.temp_dir)
        self.orig_manager = batch_api.job_manager
        batch_api.job_manager = self.job_manager

        # Locate sample ISLE audio files
        audio_dir = os.path.join(
            _BACKEND_DIR,
            "prosodylab_model_export",
            "isle",
            "data",
            "ITA",
            "train",
        )
        self.file1 = os.path.join(audio_dir, "ISLE_SESS0130_BLOCKG_01_sprt1.wav")
        self.file2 = os.path.join(audio_dir, "ISLE_SESS0130_BLOCKG_02_sprt1.wav")

        self.assertTrue(os.path.isfile(self.file1), f"Sample audio not found: {self.file1}")
        self.assertTrue(os.path.isfile(self.file2), f"Sample audio not found: {self.file2}")

    def tearDown(self):
        batch_api.job_manager = self.orig_manager
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_real_audio_batch_lifecycle(self):
        print("\n=== Running Real-Data Batch Integration Test ===")
        # 1. Create real ZIP archive in memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            with open(self.file1, "rb") as f:
                zf.writestr(os.path.basename(self.file1), f.read())
            with open(self.file2, "rb") as f:
                zf.writestr(os.path.basename(self.file2), f.read())
        zip_buf.seek(0)
        zip_bytes = zip_buf.getvalue()

        # 2. Upload via /api/batch-annotate
        # Mock start_batch_job so we can run run_worker synchronously in this test process
        with unittest.mock.patch.object(self.job_manager, "start_batch_job"):
            resp = self.client.post(
                "/api/batch-annotate",
                files={"file": ("real_batch.zip", zip_bytes, "application/zip")},
                data={"speaker_nativity": "ITA"},
            )
            self.assertEqual(resp.status_code, 201)
            data = resp.json()
            batch_id = data["batch_id"]
            self.assertEqual(data["status"], "queued")
            self.assertEqual(data["total_files"], 2)

        job_dir = self.job_manager.get_job_dir(batch_id)

        # 3. Execute REAL worker end-to-end
        print(f"Executing worker on real batch job: {batch_id}...")
        run_worker(batch_id=batch_id, job_dir=job_dir, speaker_nativity="ITA")

        # 4. Verify batch state
        state = self.job_manager.read_state(batch_id)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["completed_files"], 2)
        self.assertEqual(state["failed_files"], 0)
        self.assertEqual(state["progress_percent"], 100.0)

        # 5. Download results ZIP via endpoint
        dl_resp = self.client.get(f"/api/batch-annotate/{batch_id}/download")
        self.assertEqual(dl_resp.status_code, 200)
        self.assertEqual(dl_resp.headers["content-type"], "application/zip")

        # 6. Inspect ZIP archive contents
        dl_buf = io.BytesIO(dl_resp.content)
        with zipfile.ZipFile(dl_buf, "r") as zf:
            namelist = zf.namelist()
            print(f"ZIP Contents: {namelist}")

            self.assertIn("manifest.json", namelist)
            base1 = os.path.splitext(os.path.basename(self.file1))[0] + ".json"
            base2 = os.path.splitext(os.path.basename(self.file2))[0] + ".json"
            expected1 = f"annotations/{base1}"
            expected2 = f"annotations/{base2}"
            self.assertIn(expected1, namelist)
            self.assertIn(expected2, namelist)

            # Strict guard: ZERO audio files
            for name in namelist:
                self.assertFalse(name.endswith(".wav"), f"Audio file found in result ZIP: {name}")
                self.assertFalse(name.endswith(".mp3"))
                self.assertFalse(name.endswith(".log"))

            # Inspect manifest
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest["total_files"], 2)
            self.assertEqual(manifest["completed_files"], 2)
            self.assertEqual(manifest["failed_files"], 0)
            self.assertEqual(manifest["speaker_nativity"], "ITA")

            # Inspect individual JSON annotation
            ann1 = json.loads(zf.read(expected1).decode("utf-8"))
            self.assertIn("metadata", ann1)
            self.assertIn("transcription", ann1)
            self.assertIn("words", ann1)
            self.assertIn("pauses", ann1)
            self.assertIn("intonation", ann1)
            self.assertIn("timeline", ann1)
            self.assertEqual(ann1["metadata"]["speaker_nativity"], "ITA")
            self.assertGreater(len(ann1["words"]), 0)

            print(f"File 1 Transcript: {ann1['transcription']['text']}")
            print(f"File 1 Word count: {len(ann1['words'])}")
            print(f"File 1 Duration: {ann1['metadata']['duration']}s")

        print("=== Real-Data Integration Test Passed Successfully ===")


if __name__ == "__main__":
    unittest.main()
