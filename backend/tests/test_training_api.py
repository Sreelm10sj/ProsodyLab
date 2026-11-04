"""
backend.tests.test_training_api

Comprehensive unit and integration test suite for the FastAPI Training API.
Tests:
1. Valid GER dataset upload
2. Malformed NPZ rejection
3. Missing X_tr rejection
4. Missing Y_tr rejection
5. Missing W_tr rejection
6. Unsupported file extension rejection
7. Oversized upload rejection (250 MB guard)
8. Validation response serialization and schema fidelity
9. Anchor search on registered dataset
10. Anchor search on unknown dataset (HTTP 404)
11. GER definitive anchor regression (1181 / 2713 / 2714)
12. Stale / invalid anchor selection rejection
13. Create training job success and initial queued state
14. Language protocol resolution (GER -> top_k=1, batch=32; ITA -> top_k=10, batch=48)
15. Explicit protocol overrides preserved
16. Invalid configuration parameter rejection
17. Concurrency conflict (HTTP 409 when active job exists)
18. Unknown dataset on job creation (HTTP 404)
19. Unknown job state retrieval (HTTP 404)
20. Job status retrieval without server path / PID leaks
21. Job list pagination and sorting
22. Job cancellation and invalid state transition guard (HTTP 409 for completed)
23. Completed bundle download with strict zip content whitelist
24. Non-completed bundle download rejection (HTTP 409)
25. Artifact download with strict whitelist enforcement
26. Artifact directory traversal attack rejection
27. Critical Real-Data GER API lifecycle integration:
    Upload -> Validate -> Anchor Search -> Job Creation -> Worker Execution ->
    Completed State -> Bundle Download -> LexiRepPredictor Round-Trip Verification
"""

import os
import io
import sys
import json
import shutil
import tempfile
import zipfile
import unittest
from typing import Dict, Any
import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
_EXPORT_DIR = os.path.join(_BACKEND_DIR, "prosodylab_model_export")
_DATASET_DIR = os.path.join(_EXPORT_DIR, "code and dataset")

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _EXPORT_DIR not in sys.path:
    sys.path.insert(0, _EXPORT_DIR)

from fastapi.testclient import TestClient
from backend.main import app
from backend.training.job_manager import TrainingJobManager
from backend.training.trainer import TrainingConfig
from backend.training import api as training_api
from backend.training.dataset_loader import (
    DatasetRegistry,
    MAX_DATASET_UPLOAD_BYTES,
)
from backend.training.api import set_job_manager, set_dataset_registry
from prosodylab_model_export.lexirep_inference.stress_predictor import LexiRepPredictor


def _generate_synthetic_npz_bytes(num_words: int = 50, seed: int = 42) -> bytes:
    """Generate in-memory .npz bytes containing valid bisyllabic words."""
    rng = np.random.default_rng(seed)
    w_list = []
    y_list = []
    for wid in range(num_words):
        w_list.extend([wid, wid])
        y_list.extend([1, 0])

    N = len(w_list)
    X = rng.standard_normal((N, 768)).astype(np.float32)
    Y = np.array(y_list, dtype=np.int64)
    W = np.array(w_list, dtype=np.int64)

    buf = io.BytesIO()
    np.savez(buf, X_tr=X, Y_tr=Y, W_tr=W)
    buf.seek(0)
    return buf.getvalue()


class TestTrainingAPI(unittest.TestCase):
    def setUp(self):
        self.temp_root = tempfile.mkdtemp(prefix="test_training_api_")
        self.jobs_dir = os.path.join(self.temp_root, "jobs")
        self.datasets_dir = os.path.join(self.temp_root, "datasets")
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)

        self.manager = TrainingJobManager(self.jobs_dir)
        self.registry = DatasetRegistry(self.datasets_dir)

        set_job_manager(self.manager)
        set_dataset_registry(self.registry)

        self.client = TestClient(app)

    def tearDown(self):
        # Gracefully terminate any running worker processes spawned during tests
        for proc in list(self.manager._processes.values()):
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        set_job_manager(None)
        set_dataset_registry(None)
        if os.path.isdir(self.temp_root):
            try:
                shutil.rmtree(self.temp_root)
            except Exception:
                pass

    # =========================================================================
    # 1. DATASET UPLOAD & VALIDATION TESTS
    # =========================================================================

    def test_01_valid_dataset_upload(self):
        """1. Valid .npz upload returns HTTP 200 and validation diagnostics."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        files = {"file": ("test_data.npz", io.BytesIO(content), "application/octet-stream")}

        res = self.client.post("/api/training/dataset/validate", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["valid"])
        self.assertEqual(data["filename"], "test_data.npz")
        self.assertEqual(data["syllable_count"], 100)
        self.assertEqual(data["word_count"], 50)
        self.assertEqual(data["feature_shape"], [100, 768])
        self.assertIn("dataset_id", data)

    def test_02_malformed_npz_rejected(self):
        """2. Corrupted file content rejected with HTTP 400."""
        bad_content = b"PK\x03\x04corrupted_invalid_data"
        files = {"file": ("corrupt.npz", io.BytesIO(bad_content), "application/octet-stream")}

        res = self.client.post("/api/training/dataset/validate", files=files)
        self.assertEqual(res.status_code, 400)

    def test_03_missing_x_tr_rejected(self):
        """3. Archive missing X_tr rejected with HTTP 400."""
        buf = io.BytesIO()
        np.savez(buf, Y_tr=np.zeros(10), W_tr=np.zeros(10))
        buf.seek(0)

        files = {"file": ("missing_x.npz", buf, "application/octet-stream")}
        res = self.client.post("/api/training/dataset/validate", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("X_tr", res.json()["detail"])

    def test_04_missing_y_tr_rejected(self):
        """4. Archive missing Y_tr rejected with HTTP 400."""
        buf = io.BytesIO()
        np.savez(buf, X_tr=np.zeros((10, 768)), W_tr=np.zeros(10))
        buf.seek(0)

        files = {"file": ("missing_y.npz", buf, "application/octet-stream")}
        res = self.client.post("/api/training/dataset/validate", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Y_tr", res.json()["detail"])

    def test_05_missing_w_tr_rejected(self):
        """5. Archive missing W_tr rejected with HTTP 400."""
        buf = io.BytesIO()
        np.savez(buf, X_tr=np.zeros((10, 768)), Y_tr=np.zeros(10))
        buf.seek(0)

        files = {"file": ("missing_w.npz", buf, "application/octet-stream")}
        res = self.client.post("/api/training/dataset/validate", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("W_tr", res.json()["detail"])

    def test_06_unsupported_file_extension_rejected(self):
        """6. Non-npz upload (e.g. .pt, .pkl) rejected with HTTP 400."""
        files = {"file": ("model.pt", io.BytesIO(b"dummy_pt_content"), "application/octet-stream")}
        res = self.client.post("/api/training/dataset/validate", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn(".npz", res.json()["detail"])

    def test_07_oversized_upload_rejected(self):
        """7. Upload exceeding 250 MB rejected with HTTP 413."""
        class MockOversizedFile(io.BytesIO):
            def __len__(self):
                return MAX_DATASET_UPLOAD_BYTES + 1024

        fake_content = b"0" * 100
        files = {"file": ("huge.npz", io.BytesIO(fake_content), "application/octet-stream")}

        # Patch size limit to test 413 response
        original_max = training_api.MAX_DATASET_UPLOAD_BYTES
        try:
            training_api.MAX_DATASET_UPLOAD_BYTES = 50
            res = self.client.post("/api/training/dataset/validate", files=files)
            self.assertEqual(res.status_code, 413)
        finally:
            training_api.MAX_DATASET_UPLOAD_BYTES = original_max

    def test_08_validation_response_serialization(self):
        """8. Validation diagnostics serialize cleanly with all required fields."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        files = {"file": ("test.npz", io.BytesIO(content), "application/octet-stream")}
        res = self.client.post("/api/training/dataset/validate", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        expected_fields = [
            "dataset_id", "valid", "filename", "syllable_count", "word_count",
            "stressed_count", "unstressed_count", "stress_ratio",
            "bisyllabic_anchor_candidate_count", "rejected_word_count",
            "feature_shape", "feature_dtype", "warnings", "errors"
        ]
        for f in expected_fields:
            self.assertIn(f, data)

    # =========================================================================
    # 2. ANCHOR SEARCH TESTS
    # =========================================================================

    def test_09_anchor_search(self):
        """9. Anchor search on valid dataset returns ranked candidate list."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        res = self.client.post(
            "/api/training/anchors/search",
            json={"dataset_id": dataset_id, "max_candidates": 10},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["dataset_id"], dataset_id)
        self.assertEqual(data["total_training_syllables"], 100)
        self.assertEqual(data["total_training_words"], 50)
        self.assertEqual(len(data["top_anchors"]), 10)
        self.assertIsNotNone(data["selected_anchor"])
        self.assertEqual(data["top_anchors"][0]["rank"], 1)

    def test_10_unknown_dataset_anchor_search_404(self):
        """10. Anchor search on non-existent dataset returns HTTP 404."""
        random_id = "00000000-0000-4000-8000-000000000000"
        res = self.client.post(
            "/api/training/anchors/search",
            json={"dataset_id": random_id, "max_candidates": 5},
        )
        self.assertEqual(res.status_code, 404)

    def test_11_ger_cache_anchor_search_exact(self):
        """11. Anchor search on canonical GER cache returns exact Word 1181 (s=2713, u=2714)."""
        ger_cache_path = os.path.join(_DATASET_DIR, "ger_contextual_cache.npz")
        self.assertTrue(os.path.exists(ger_cache_path), f"Missing GER cache: {ger_cache_path}")

        with open(ger_cache_path, "rb") as f:
            content = f.read()

        dataset_id, _, _ = self.registry.save_dataset(content, "ger_contextual_cache.npz")

        res = self.client.post(
            "/api/training/anchors/search",
            json={"dataset_id": dataset_id, "max_candidates": 5},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["total_training_syllables"], 7924)
        self.assertEqual(data["total_training_words"], 3447)
        self.assertEqual(data["bisyllabic_candidates"], 2628)

        sel = data["selected_anchor"]
        self.assertIsNotNone(sel)
        self.assertEqual(sel["word_id"], 1181)
        self.assertEqual(sel["stressed_syllable_index"], 2713)
        self.assertEqual(sel["unstressed_syllable_index"], 2714)
        self.assertAlmostEqual(sel["separation_score"], 68.2231, places=2)

    # =========================================================================
    # 3. ANCHOR SELECTION SAFETY TESTS
    # =========================================================================

    def test_12_stale_invalid_anchor_rejected(self):
        """12. Server-side anchor verification rejects out-of-bounds, non-bisyllabic, or inverted pairs."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        # Case A: Out of bounds indices
        bad_req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 999, "unstressed_syllable_index": 1},
        }
        res = self.client.post("/api/training/jobs", json=bad_req)
        self.assertEqual(res.status_code, 400)
        self.assertIn("out of bounds", res.json()["detail"].lower())

        # Case B: Inverted labels (Y[0]=1 is stressed, Y[1]=0 is unstressed; passing 1 as stressed fails)
        bad_req_inv = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 1, "unstressed_syllable_index": 0},
        }
        res = self.client.post("/api/training/jobs", json=bad_req_inv)
        self.assertEqual(res.status_code, 400)
        self.assertIn("invalid anchor labels", res.json()["detail"].lower())

        # Case C: Indices belong to different words
        bad_req_diff = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 2},
        }
        res = self.client.post("/api/training/jobs", json=bad_req_diff)
        self.assertEqual(res.status_code, 400)
        self.assertIn("word ids", res.json()["detail"].lower())

    # =========================================================================
    # 4. TRAINING JOB CREATION & PROTOCOL RESOLUTION
    # =========================================================================

    def test_13_create_training_job_success(self):
        """13. Creating a valid training job returns HTTP 201 with UUID and queued status."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
            "config": {"warmup_epochs": 1, "outer_loops": 1},
        }
        res = self.client.post("/api/training/jobs", json=job_req)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "queued")

    def test_14_language_protocol_defaults(self):
        """14. Language protocol resolves defaults automatically (GER: top_k=1, batch=32; ITA: top_k=10, batch=48)."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        # GER Job
        ger_req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
        }
        res_ger = self.client.post("/api/training/jobs", json=ger_req)
        self.assertEqual(res_ger.status_code, 201)
        ger_job_id = res_ger.json()["job_id"]

        with open(os.path.join(self.manager.get_job_dir(ger_job_id), "config.json"), "r") as f:
            ger_cfg = json.load(f)
        self.assertEqual(ger_cfg["top_k_rep"], 1)
        self.assertEqual(ger_cfg["batch_words"], 32)

        # Cancel GER job to free single-job concurrency lock
        self.manager.cancel_job(ger_job_id)

        # ITA Job
        ita_req = {
            "dataset_id": dataset_id,
            "language": "ITA",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
        }
        res_ita = self.client.post("/api/training/jobs", json=ita_req)
        self.assertEqual(res_ita.status_code, 201)
        ita_job_id = res_ita.json()["job_id"]

        with open(os.path.join(self.manager.get_job_dir(ita_job_id), "config.json"), "r") as f:
            ita_cfg = json.load(f)
        self.assertEqual(ita_cfg["top_k_rep"], 10)
        self.assertEqual(ita_cfg["batch_words"], 48)

    def test_15_explicit_protocol_overrides_preserved(self):
        """15. Explicit config overrides (e.g. top_k_rep=7, batch_words=64) take precedence."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
            "config": {"top_k_rep": 7, "batch_words": 64},
        }
        res = self.client.post("/api/training/jobs", json=req)
        self.assertEqual(res.status_code, 201)
        job_id = res.json()["job_id"]

        with open(os.path.join(self.manager.get_job_dir(job_id), "config.json"), "r") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["top_k_rep"], 7)
        self.assertEqual(cfg["batch_words"], 64)

    def test_16_invalid_config_rejected(self):
        """16. Out-of-bounds config parameters (e.g. negative epochs, 0 temp) rejected."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        bad_cfg_req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
            "config": {"warmup_epochs": -5},
        }
        res = self.client.post("/api/training/jobs", json=bad_cfg_req)
        self.assertEqual(res.status_code, 422)

    def test_17_concurrency_conflict_409(self):
        """17. Requesting second job while first is running returns HTTP 409 Conflict."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
        }
        res1 = self.client.post("/api/training/jobs", json=job_req)
        self.assertEqual(res1.status_code, 201)

        # Second job attempt must be rejected with HTTP 409
        res2 = self.client.post("/api/training/jobs", json=job_req)
        self.assertEqual(res2.status_code, 409)
        self.assertIn("already active", res2.json()["detail"].lower())

    def test_18_unknown_dataset_create_job_404(self):
        """18. Creating job with unknown dataset_id returns HTTP 404."""
        random_id = "00000000-0000-4000-8000-000000000000"
        job_req = {
            "dataset_id": random_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
        }
        res = self.client.post("/api/training/jobs", json=job_req)
        self.assertEqual(res.status_code, 404)

    # =========================================================================
    # 5. JOB QUERY, LIST & CANCELLATION
    # =========================================================================

    def test_19_unknown_job_state_404(self):
        """19. Querying non-existent job returns HTTP 404."""
        random_id = "00000000-0000-4000-8000-000000000000"
        res = self.client.get(f"/api/training/jobs/{random_id}")
        self.assertEqual(res.status_code, 404)

    def test_20_job_status_retrieval(self):
        """20. Job status retrieval returns clean JSON without server path or PID leaks."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
        }
        create_res = self.client.post("/api/training/jobs", json=job_req)
        job_id = create_res.json()["job_id"]

        res = self.client.get(f"/api/training/jobs/{job_id}")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["job_id"], job_id)
        self.assertIn(data["status"], ["queued", "validating", "warming_up", "training", "completed"])
        # Ensure no private paths or PIDs leaked
        self.assertNotIn("pid", data)
        self.assertNotIn("path", data)
        self.assertNotIn("command", data)

    def test_21_job_list(self):
        """21. Job list returns array of recent job summaries."""
        res = self.client.get("/api/training/jobs?limit=10")
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), list)

    def test_22_cancellation(self):
        """22. Cancelling an active job transitions state to 'cancelled'; completed job gives 409."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_req = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
        }
        create_res = self.client.post("/api/training/jobs", json=job_req)
        job_id = create_res.json()["job_id"]

        cancel_res = self.client.post(f"/api/training/jobs/{job_id}/cancel")
        self.assertEqual(cancel_res.status_code, 200)
        self.assertEqual(cancel_res.json()["status"], "cancelled")

        # Mock completed status to verify completed cancellation rejection (HTTP 409)
        if job_id in self.manager._processes:
            del self.manager._processes[job_id]

        state_file = os.path.join(self.manager.get_job_dir(job_id), "job_state.json")
        with open(state_file, "r") as f:
            st = json.load(f)
        st["status"] = "completed"
        with open(state_file, "w") as f:
            json.dump(st, f)

        cancel_completed = self.client.post(f"/api/training/jobs/{job_id}/cancel")
        self.assertEqual(cancel_completed.status_code, 409)

    # =========================================================================
    # 6. BUNDLE & ARTIFACT DOWNLOAD TESTS
    # =========================================================================

    def test_23_completed_bundle_download_and_content_whitelist(self):
        """23. Completed job bundle download yields clean zip with strictly whitelisted contents."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_id = self.manager.create_job(
            dataset_path=self.registry.get_dataset_path(dataset_id),
            config=TrainingConfig(language="GER"),
            language="GER",
            anchor={"word_id": 0, "stressed_syllable_index": 0, "unstressed_syllable_index": 1},
        )
        job_dir = self.manager.get_job_dir(job_id)

        # Mock completed state and model bundle files
        os.makedirs(os.path.join(job_dir, "model"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "anchors"), exist_ok=True)

        with open(os.path.join(job_dir, "config.json"), "w") as f:
            json.dump({"language": "GER"}, f)
        with open(os.path.join(job_dir, "anchors.json"), "w") as f:
            json.dump({"word_id": 0}, f)
        with open(os.path.join(job_dir, "model", "final_lexirep_model.pt"), "wb") as f:
            f.write(b"dummy_pt")
        with open(os.path.join(job_dir, "anchors", "prototypes_10d.npy"), "wb") as f:
            np.save(f, np.zeros((2, 10), dtype=np.float32))

        # Also add unwhitelisted files that MUST NOT be in the bundle zip
        with open(os.path.join(job_dir, "worker.log"), "w") as f:
            f.write("private server log")
        with open(os.path.join(job_dir, "dataset.npz"), "wb") as f:
            f.write(b"raw dataset bytes")

        # Mark completed in job_state.json
        with open(os.path.join(job_dir, "job_state.json"), "w") as f:
            json.dump({"job_id": job_id, "status": "completed"}, f)

        res = self.client.get(f"/api/training/jobs/{job_id}/bundle")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "application/zip")

        # Inspect zip members
        zip_buf = io.BytesIO(res.content)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("config.json", namelist)
            self.assertIn("anchors.json", namelist)
            self.assertIn("model/final_lexirep_model.pt", namelist)
            self.assertIn("anchors/prototypes_10d.npy", namelist)

            # Strict exclusion check
            self.assertNotIn("worker.log", namelist)
            self.assertNotIn("dataset.npz", namelist)

    def test_24_non_completed_bundle_rejected_409(self):
        """24. Requesting bundle for a job that is not completed returns HTTP 409."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_id = self.manager.create_job(
            dataset_path=self.registry.get_dataset_path(dataset_id),
            config=TrainingConfig(language="GER"),
            language="GER",
        )
        res = self.client.get(f"/api/training/jobs/{job_id}/bundle")
        self.assertEqual(res.status_code, 409)

    def test_25_artifact_download_and_whitelist(self):
        """25. Allowed artifact files (metrics.json) download successfully; unlisted files return HTTP 400."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_id = self.manager.create_job(
            dataset_path=self.registry.get_dataset_path(dataset_id),
            config=TrainingConfig(language="GER"),
            language="GER",
        )
        job_dir = self.manager.get_job_dir(job_id)

        # Create whitelisted metrics.json
        with open(os.path.join(job_dir, "metrics.json"), "w") as f:
            json.dump({"loss": 0.123}, f)

        # Create unwhitelisted private.txt
        with open(os.path.join(job_dir, "private.txt"), "w") as f:
            f.write("secret")

        # Whitelisted artifact
        res_ok = self.client.get(f"/api/training/jobs/{job_id}/artifacts/metrics.json")
        self.assertEqual(res_ok.status_code, 200)
        self.assertEqual(res_ok.json(), {"loss": 0.123})

        # Unwhitelisted artifact
        res_blocked = self.client.get(f"/api/training/jobs/{job_id}/artifacts/private.txt")
        self.assertEqual(res_blocked.status_code, 400)
        self.assertIn("not in the allowed artifact list", res_blocked.json()["detail"])

    def test_26_artifact_traversal_attack_rejected(self):
        """26. Directory traversal attempts in artifact route return HTTP 400."""
        content = _generate_synthetic_npz_bytes(num_words=50)
        dataset_id, _, _ = self.registry.save_dataset(content, "test.npz")

        job_id = self.manager.create_job(
            dataset_path=self.registry.get_dataset_path(dataset_id),
            config=TrainingConfig(language="GER"),
            language="GER",
        )
        res = self.client.get(f"/api/training/jobs/{job_id}/artifacts/..%2F..%2Fetc%2Fpasswd")
        self.assertEqual(res.status_code, 400)

    # =========================================================================
    # 7. CRITICAL REAL GER LIFECYCLE API INTEGRATION TEST
    # =========================================================================

    def test_27_critical_real_ger_api_lifecycle(self):
        """
        27. Real-Data GER API lifecycle:
        Upload -> Validate -> Anchor Search -> Job Creation -> Worker Lifecycle ->
        Completed State -> Bundle Download -> LexiRepPredictor Verification.
        """
        ger_cache_path = os.path.join(_DATASET_DIR, "ger_contextual_cache.npz")
        self.assertTrue(os.path.exists(ger_cache_path), f"Missing GER cache: {ger_cache_path}")

        # A. Upload & Validate
        with open(ger_cache_path, "rb") as f:
            files = {"file": ("ger_contextual_cache.npz", f, "application/octet-stream")}
            val_res = self.client.post("/api/training/dataset/validate", files=files)

        self.assertEqual(val_res.status_code, 200)
        val_data = val_res.json()
        self.assertTrue(val_data["valid"])

        # Assert exact canonical GER training split
        self.assertEqual(val_data["syllable_count"], 7924)
        self.assertEqual(val_data["word_count"], 3447)
        self.assertEqual(val_data["bisyllabic_anchor_candidate_count"], 2628)
        dataset_id = val_data["dataset_id"]

        # B. Anchor Search
        search_res = self.client.post(
            "/api/training/anchors/search",
            json={"dataset_id": dataset_id, "max_candidates": 5},
        )
        self.assertEqual(search_res.status_code, 200)
        search_data = search_res.json()
        selected_anchor = search_data["selected_anchor"]
        self.assertIsNotNone(selected_anchor)
        self.assertEqual(selected_anchor["word_id"], 1181)
        self.assertEqual(selected_anchor["stressed_syllable_index"], 2713)
        self.assertEqual(selected_anchor["unstressed_syllable_index"], 2714)

        # C. Create Training Job with fast reduced configuration
        job_payload = {
            "dataset_id": dataset_id,
            "language": "GER",
            "anchor": {
                "word_id": 1181,
                "stressed_syllable_index": 2713,
                "unstressed_syllable_index": 2714,
            },
            "config": {
                "seed": 42,
                "device": "cpu",
                "warmup_epochs": 1,
                "idec_pretrain_epochs": 1,
                "outer_loops": 1,
                "idec_steps_per_loop": 2,
                "refinement_epochs": 1,
                "batch_words": 32,
                "idec_batch_size": 128,
            },
        }
        create_res = self.client.post("/api/training/jobs", json=job_payload)
        self.assertEqual(create_res.status_code, 201)
        job_id = create_res.json()["job_id"]

        # D. Wait for background worker completion (fast reduced config takes ~2-3s)
        proc = self.manager._processes.get(job_id)
        if proc:
            proc.wait(timeout=30.0)

        # Query state via API
        state_res = self.client.get(f"/api/training/jobs/{job_id}")
        self.assertEqual(state_res.status_code, 200)
        state_data = state_res.json()
        self.assertEqual(state_data["status"], "completed")
        self.assertIsNotNone(state_data["idec_loss"])
        self.assertIsInstance(state_data["idec_loss"], float)
        self.assertGreater(state_data["idec_loss"], 0.0)
        self.assertIsNotNone(state_data["training_accuracy"])
        self.assertIsInstance(state_data["training_accuracy"], float)
        self.assertGreater(state_data["training_accuracy"], 0.0)
        self.assertLessEqual(state_data["training_accuracy"], 1.0)

        # E. Download Bundle via API
        bundle_res = self.client.get(f"/api/training/jobs/{job_id}/bundle")
        self.assertEqual(bundle_res.status_code, 200)
        self.assertEqual(bundle_res.headers["content-type"], "application/zip")

        # F. Extract bundle and verify with LexiRepPredictor.from_bundle()
        extract_dir = tempfile.mkdtemp(prefix="test_bundle_verify_")
        try:
            with zipfile.ZipFile(io.BytesIO(bundle_res.content), "r") as zf:
                zf.extractall(extract_dir)

            predictor = LexiRepPredictor.from_bundle(
                language="GER",
                bundle_root=extract_dir,
                device="cpu",
            )
            self.assertIsNotNone(predictor)

            # Test inference pass on canonical bisyllabic boundaries
            d = np.load(ger_cache_path)
            test_x = d["X_tr"][:4]
            test_bounds = [[0, 1], [2, 3]]
            pred_res = predictor.predict_stress(test_x, test_bounds)

            self.assertTrue(pred_res.linguistic_compliance)
            self.assertEqual(len(pred_res.words), 2)
            for wp in pred_res.words:
                syl_idxs = [s.syllable_index for s in wp.syllables]
                self.assertIn(wp.stressed_syllable_index, syl_idxs)
                for s in wp.syllables:
                    self.assertTrue(np.isfinite(s.contrast_score))
                    self.assertTrue(np.isfinite(s.confidence))
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
