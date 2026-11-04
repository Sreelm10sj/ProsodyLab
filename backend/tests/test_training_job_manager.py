"""
backend.tests.test_training_job_manager

Unit and integration tests for TrainingJobManager and worker subprocess in backend/training/.
Verifies:
1. Job creation, sandboxing, and UUIDv4 path traversal protection
2. Subprocess worker execution and successful completion with state persistence
3. Failed subprocess error capture and status="failed" recording
4. Subprocess cancellation (process termination and status="cancelled")
5. Job cleanup and directory removal
6. Job listing across sandboxes
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_DIR)
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from backend.training.job_manager import TrainingJobManager, JobManagerError
from backend.training.trainer import TrainingConfig


def _create_test_npz(filepath: str, dim: int = 768, num_words: int = 60, corrupt: bool = False):
    """Creates a temporary .npz dataset file for worker integration tests."""
    rng = np.random.default_rng(42)
    w_list = []
    y_list = []
    for w in range(num_words):
        w_list.extend([w, w])
        y_list.extend([1, 0])

    N = len(w_list)
    if corrupt:
        X = np.ones((N, 512), dtype=np.float32)  # Wrong dimension (512 != 768)
    else:
        X = rng.standard_normal((N, dim)).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True) * 5.0

    Y = np.array(y_list, dtype=np.int64)
    W = np.array(w_list, dtype=np.int64)

    np.savez_compressed(filepath, X_tr=X, Y_tr=Y, W_tr=W)


class TestTrainingJobManager(unittest.TestCase):
    """Test suite verifying process management and worker isolation."""

    def setUp(self):
        self.temp_root = tempfile.mkdtemp(prefix="lexirep_mgr_test_")
        self.manager = TrainingJobManager(root_dir=self.temp_root)
        self.dataset_file = os.path.join(self.temp_root, "source_dataset.npz")
        _create_test_npz(self.dataset_file, corrupt=False)

    def tearDown(self):
        # Terminate any remaining running processes
        for job_id in list(self.manager._processes.keys()):
            try:
                self.manager.cancel_job(job_id)
            except Exception:
                pass
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_01_job_creation_and_traversal_guard(self):
        """1. Job creation stages dataset/config and rejects malicious path traversal IDs."""
        config = TrainingConfig(outer_loops=2)
        job_id = self.manager.create_job(self.dataset_file, config, language="GER")

        self.assertTrue(os.path.isdir(self.manager.get_job_dir(job_id)))
        self.assertTrue(os.path.isfile(os.path.join(self.manager.get_job_dir(job_id), "dataset.npz")))
        self.assertTrue(os.path.isfile(os.path.join(self.manager.get_job_dir(job_id), "config.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.manager.get_job_dir(job_id), "job_state.json")))

        state = self.manager.get_state(job_id)
        self.assertEqual(state["status"], "queued")
        self.assertEqual(state["job_id"], job_id)

        # Path traversal rejection
        with self.assertRaises(ValueError):
            self.manager.get_job_dir("../../etc/passwd")

        with self.assertRaises(ValueError):
            self.manager.get_job_dir("not-a-uuid")

    def test_02_worker_successful_execution(self):
        """2. Subprocess worker executes job to completion, producing valid artifacts and completed state."""
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

        job_id = self.manager.create_job(self.dataset_file, config, language="GER")
        proc = self.manager.start_job(job_id)
        self.assertIsNotNone(proc)

        # Wait for worker process to finish (timeout 30s)
        t_start = time.time()
        completed = False
        while time.time() - t_start < 30.0:
            state = self.manager.get_state(job_id)
            if state["status"] == "completed":
                completed = True
                break
            elif state["status"] == "failed":
                self.fail(f"Worker failed unexpectedly: {state.get('error')}")
            time.sleep(0.3)

        self.assertTrue(completed, "Worker did not complete within 30s timeout.")
        final_state = self.manager.get_state(job_id)
        self.assertEqual(final_state["status"], "completed")
        self.assertEqual(final_state["progress_percent"], 100.0)
        self.assertIsNone(final_state["error"])

        # Check that artifacts exist in job directory
        job_dir = self.manager.get_job_dir(job_id)
        self.assertTrue(os.path.isfile(os.path.join(job_dir, "model", "final_lexirep_model.pt")))
        self.assertTrue(os.path.isfile(os.path.join(job_dir, "anchors", "prototypes_10d.npy")))
        self.assertTrue(os.path.isfile(os.path.join(job_dir, "metrics.json")))

    def test_03_worker_failed_execution_reporting(self):
        """3. Worker with invalid dataset records status='failed' with structured error message."""
        corrupt_dataset = os.path.join(self.temp_root, "corrupt_dataset.npz")
        _create_test_npz(corrupt_dataset, corrupt=True)

        config = TrainingConfig(device="cpu")
        job_id = self.manager.create_job(corrupt_dataset, config)
        self.manager.start_job(job_id)

        # Wait for worker process to fail (validation should fail in < 5s)
        t_start = time.time()
        failed = False
        while time.time() - t_start < 15.0:
            state = self.manager.get_state(job_id)
            if state["status"] == "failed":
                failed = True
                break
            time.sleep(0.2)

        self.assertTrue(failed, "Worker should have reported status='failed'.")
        final_state = self.manager.get_state(job_id)
        self.assertEqual(final_state["status"], "failed")
        self.assertIsNotNone(final_state["error"])
        self.assertIn("768", final_state["error"])

    def test_04_subprocess_cancellation(self):
        """4. Subprocess can be cancelled mid-training, terminating process and updating state to cancelled."""
        # Config with many epochs so it runs long enough to cancel
        config = TrainingConfig(
            language="GER",
            device="cpu",
            warmup_epochs=50,
            batch_words=4,
        )

        job_id = self.manager.create_job(self.dataset_file, config)
        proc = self.manager.start_job(job_id)

        # Wait until process is active
        time.sleep(0.8)
        self.assertIsNone(proc.poll(), "Process should be actively running before cancel.")

        # Cancel job
        cancel_state = self.manager.cancel_job(job_id)
        self.assertEqual(cancel_state["status"], "cancelled")
        self.assertIsNotNone(proc.poll(), "Process should be terminated after cancel.")

    def test_05_cleanup_and_list_jobs(self):
        """5. Manager lists jobs and cleans up completed/cancelled job directories."""
        config = TrainingConfig()
        job1 = self.manager.create_job(self.dataset_file, config)
        job2 = self.manager.create_job(self.dataset_file, config)

        jobs = self.manager.list_jobs()
        job_ids = [j["job_id"] for j in jobs]
        self.assertIn(job1, job_ids)
        self.assertIn(job2, job_ids)

        # Cleanup job1
        removed = self.manager.cleanup_job(job1)
        self.assertTrue(removed)
        self.assertFalse(os.path.exists(self.manager.get_job_dir(job1)))

        jobs_after = self.manager.list_jobs()
        job_ids_after = [j["job_id"] for j in jobs_after]
        self.assertNotIn(job1, job_ids_after)
        self.assertIn(job2, job_ids_after)


if __name__ == "__main__":
    unittest.main()
