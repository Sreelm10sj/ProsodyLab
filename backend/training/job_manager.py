"""
backend.training.job_manager

Process management and lifecycle orchestration for LexiRep training jobs.
Responsibilities:
- Job directory sandboxing under UUIDv4 namespace
- Subprocess isolation executing `backend.training.worker` via sys.executable
- Atomic state reading and crash detection
- Cooperative and forced process cancellation (Windows-compatible)
- Job cleanup and artifact discovery
- Zero coupling to HTTP or FastAPI
"""

import os
import sys
import json
import uuid
import re
import shutil
import subprocess
import time
from typing import Dict, Any, Optional, List, Union

from backend.training.trainer import TrainingConfig

UUID4_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)


class JobManagerError(Exception):
    """Raised when job management operations fail."""
    pass


class TrainingJobManager:
    """Manages training subprocesses and state sandboxes."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self._processes: Dict[str, subprocess.Popen] = {}

    def _validate_job_id(self, job_id: str) -> str:
        """Strictly validate UUIDv4 structure to prevent directory traversal."""
        if not UUID4_PATTERN.match(job_id):
            raise ValueError(f"Invalid job ID '{job_id}'. Must be a valid UUIDv4.")
        return job_id

    def get_job_dir(self, job_id: str) -> str:
        self._validate_job_id(job_id)
        return os.path.join(self.root_dir, job_id)

    def create_job(
        self,
        dataset_path: str,
        config: Union[TrainingConfig, Dict[str, Any]],
        language: str = "GER",
        anchor: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new job sandbox with staged dataset and configuration.

        Args:
            dataset_path: Path to valid .npz dataset.
            config: TrainingConfig object or dictionary of hyperparameters.
            language: Dataset language ('GER', 'ITA', etc.).
            anchor: Optional pre-selected anchor dictionary.

        Returns:
            job_id: Generated UUIDv4 identifier.
        """
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Input dataset not found: {dataset_path}")

        job_id = str(uuid.uuid4())
        job_dir = self.get_job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Stage dataset into job sandbox
        staged_dataset = os.path.join(job_dir, "dataset.npz")
        shutil.copyfile(dataset_path, staged_dataset)

        # Prepare config
        if isinstance(config, TrainingConfig):
            cfg_dict = config.to_dict()
        else:
            cfg_dict = dict(config)

        cfg_dict["language"] = language
        if anchor is not None:
            cfg_dict["anchor"] = anchor

        config_path = os.path.join(job_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

        # Write initial state
        state_path = os.path.join(job_dir, "job_state.json")
        initial_state = {
            "job_id": job_id,
            "status": "queued",
            "current_step": "Job registered and queued",
            "progress_percent": 0.0,
            "loop": 0,
            "total_loops": cfg_dict.get("outer_loops", 13),
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "pseudo_label_changes": 0,
            "loss_history": [],
            "error": None,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, indent=2)

        return job_id

    def start_job(self, job_id: str) -> subprocess.Popen:
        """
        Launch the worker subprocess for the specified job.

        Args:
            job_id: UUIDv4 job identifier.

        Returns:
            subprocess.Popen handle.
        """
        job_dir = self.get_job_dir(job_id)
        if not os.path.isdir(job_dir):
            raise FileNotFoundError(f"Job directory not found: {job_dir}")

        if job_id in self._processes and self._processes[job_id].poll() is None:
            raise JobManagerError(f"Job {job_id} is already actively running.")

        dataset_path = os.path.join(job_dir, "dataset.npz")
        config_path = os.path.join(job_dir, "config.json")

        cmd = [
            sys.executable,
            "-m",
            "backend.training.worker",
            "--job-id",
            job_id,
            "--dataset",
            dataset_path,
            "--output-dir",
            job_dir,
            "--config",
            config_path,
        ]

        # Ensure PYTHONPATH includes repository root so backend package is found
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(backend_dir)
        env = dict(os.environ)
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{current_pp}" if current_pp else repo_root

        # Direct stdout and stderr to worker.log to avoid 64KB OS pipe buffer deadlock
        log_path = os.path.join(job_dir, "worker.log")
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=job_dir,
                env=env,
            )
        self._processes[job_id] = proc
        return proc

    def get_state(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieve current job state from disk with process liveness detection.

        Args:
            job_id: UUIDv4 job identifier.

        Returns:
            Dict containing job state.
        """
        job_dir = self.get_job_dir(job_id)
        state_path = os.path.join(job_dir, "job_state.json")
        log_path = os.path.join(job_dir, "worker.log")

        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"State file not found for job: {job_id}")

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            time.sleep(0.05)
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

        # Detect process death / crash
        proc = self._processes.get(job_id)
        if proc is not None:
            retcode = proc.poll()
            if retcode is not None:
                current_status = state.get("status")
                if retcode != 0 and current_status not in ["failed", "cancelled"]:
                    log_tail = ""
                    if os.path.isfile(log_path):
                        try:
                            with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                                log_tail = "".join(lf.readlines()[-20:]).strip()
                        except Exception:
                            pass
                    state["status"] = "failed"
                    state["error"] = f"Worker process died with exit code {retcode}. {log_tail}".strip()
                    try:
                        with open(state_path, "w", encoding="utf-8") as f:
                            json.dump(state, f, indent=2)
                    except Exception:
                        pass

        return state

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running training job cleanly.

        Args:
            job_id: UUIDv4 job identifier.

        Returns:
            Updated job state.
        """
        job_dir = self.get_job_dir(job_id)
        state_path = os.path.join(job_dir, "job_state.json")

        state = {}
        if os.path.isfile(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                pass

        proc = self._processes.get(job_id)
        if proc is not None and proc.poll() is None:
            # Terminate gracefully
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Update state file to cancelled
        if state.get("status") not in ["completed", "failed"]:
            state["status"] = "cancelled"
            state["current_step"] = "Job cancelled by user"
            state["estimated_remaining_seconds"] = None
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
            except Exception:
                pass

        return state

    def cleanup_job(self, job_id: str) -> bool:
        """
        Remove job directory from disk if job is not running.

        Args:
            job_id: UUIDv4 job identifier.

        Returns:
            True if removed, False otherwise.
        """
        job_dir = self.get_job_dir(job_id)
        proc = self._processes.get(job_id)

        if proc is not None and proc.poll() is None:
            raise JobManagerError(f"Cannot cleanup job {job_id} while worker is actively running.")

        if os.path.isdir(job_dir):
            shutil.rmtree(job_dir)
            if job_id in self._processes:
                del self._processes[job_id]
            return True
        return False

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all known jobs in the root directory."""
        jobs = []
        if not os.path.isdir(self.root_dir):
            return jobs

        for entry in os.listdir(self.root_dir):
            entry_path = os.path.join(self.root_dir, entry)
            if os.path.isdir(entry_path) and UUID4_PATTERN.match(entry):
                try:
                    state = self.get_state(entry)
                    jobs.append(state)
                except Exception:
                    pass

        return jobs

    def has_active_job(self) -> Optional[str]:
        """
        Check if there is currently an active or running training job.
        Returns job_id of the active job if present, None otherwise.
        """
        active_statuses = {"queued", "validating", "anchor_search", "warming_up", "idec_pretrain", "training"}
        for job_id, proc in list(self._processes.items()):
            if proc.poll() is None:
                return job_id

        # Also inspect on-disk states
        for job in self.list_jobs():
            jid = job.get("job_id")
            if job.get("status") in active_statuses and jid in self._processes:
                if self._processes[jid].poll() is None:
                    return jid
        return None
