"""
backend.batch_annotation.job_manager

Lifecycle orchestration and process management for batch audio annotation jobs.
Sandboxes each batch under a dedicated UUIDv4 directory, spawns worker subprocesses,
tracks atomic JSON state, and builds clean annotation-only downloadable ZIP archives.
"""

import os
import sys
import json
import uuid
import re
import shutil
import zipfile
import subprocess
import time
from typing import Dict, Any, Optional, List, Tuple

UUID4_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)


class BatchJobManagerError(Exception):
    """Raised when batch job management operations fail."""
    pass


class BatchJobManager:
    """Manages batch annotation subprocesses, sandboxes, and state."""

    def __init__(self, root_dir: Optional[str] = None):
        if root_dir is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            root_dir = os.path.join(backend_dir, "batch_annotation_jobs")
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self._processes: Dict[str, subprocess.Popen] = {}

    def validate_batch_id(self, batch_id: str) -> str:
        """Strictly validate UUIDv4 structure to prevent path traversal."""
        if not batch_id or not UUID4_PATTERN.match(batch_id):
            raise ValueError(f"Invalid batch ID '{batch_id}'. Must be a valid UUIDv4.")
        return batch_id

    def get_job_dir(self, batch_id: str) -> str:
        self.validate_batch_id(batch_id)
        return os.path.join(self.root_dir, batch_id)

    def write_atomic_state(self, job_dir: str, state: Dict[str, Any], max_retries: int = 15) -> None:
        """
        Write job state JSON atomically via a temp file and os.replace.
        Includes exponential backoff retry for Windows WinError 5 / WinError 32
        when destination file is momentarily open for reading by the polling endpoint.
        """
        state_path = os.path.join(job_dir, "batch_state.json")
        temp_path = os.path.join(job_dir, f"batch_state_{uuid.uuid4().hex[:8]}.tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
        except Exception:
            if os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise

        for attempt in range(max_retries):
            try:
                os.replace(temp_path, state_path)
                return
            except (PermissionError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(0.015 * (1.4 ** attempt))
                else:
                    # Fallback on final attempt: write directly
                    try:
                        with open(state_path, "w", encoding="utf-8") as f:
                            json.dump(state, f, indent=2)
                    except Exception:
                        pass
                    try:
                        if os.path.isfile(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass
                    return

        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    def read_state(self, batch_id: str, max_retries: int = 10) -> Dict[str, Any]:
        """
        Read state JSON from disk with crash detection.
        Uses fast buffered binary reading to minimize file handle open duration on Windows.
        """
        job_dir = self.get_job_dir(batch_id)
        state_path = os.path.join(job_dir, "batch_state.json")
        log_path = os.path.join(job_dir, "worker.log")

        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"State file not found for batch job: {batch_id}")

        state = None
        last_err = None
        for attempt in range(max_retries):
            try:
                # Read quickly as bytes and parse in memory so file handle is closed in microseconds
                with open(state_path, "rb") as f:
                    content = f.read()
                if content:
                    state = json.loads(content.decode("utf-8"))
                    break
            except (PermissionError, OSError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(0.02 * (attempt + 1))

        if state is None:
            if last_err:
                raise last_err
            raise FileNotFoundError(f"Failed to read state file for batch job: {batch_id}")

        # Detect unexpected worker crash
        proc = self._processes.get(batch_id)
        if proc is not None:
            retcode = proc.poll()
            if retcode is not None:
                current_status = state.get("status")
                if retcode != 0 and current_status not in ["completed", "completed_with_errors", "failed", "cancelled"]:
                    log_tail = ""
                    if os.path.isfile(log_path):
                        try:
                            with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                                log_tail = "".join(lf.readlines()[-20:]).strip()
                        except Exception:
                            pass
                    state["status"] = "failed"
                    state["error"] = f"Worker process terminated unexpectedly with code {retcode}. {log_tail}".strip()
                    try:
                        self.write_atomic_state(job_dir, state)
                    except Exception:
                        pass

        return state

    def create_batch_job(
        self,
        valid_entries: List[Dict[str, Any]],
        speaker_nativity: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Create a new sandbox for a batch job.

        Args:
            valid_entries: List of validated audio entries.
            speaker_nativity: Optional 'GER' or 'ITA'.

        Returns:
            (batch_id, job_dir, input_dir)
        """
        batch_id = str(uuid.uuid4())
        job_dir = self.get_job_dir(batch_id)
        input_dir = os.path.join(job_dir, "input")
        output_dir = os.path.join(job_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        files_state = []
        for entry in valid_entries:
            files_state.append({
                "archive_name": entry["archive_name"],
                "input_name": entry["normalized_name"],
                "output_name": entry["output_name"],
                "file_size": entry.get("file_size", 0),
                "status": "queued",
                "duration": None,
                "words_count": None,
                "error": None,
            })

        initial_state = {
            "batch_id": batch_id,
            "status": "queued",
            "total_files": len(files_state),
            "completed_files": 0,
            "failed_files": 0,
            "current_file": None,
            "current_file_index": 0,
            "progress_percent": 0.0,
            "speaker_nativity": speaker_nativity,
            "files": files_state,
            "elapsed_seconds": 0.0,
            "error": None,
            "created_at": time.time(),
        }

        self.write_atomic_state(job_dir, initial_state)
        return batch_id, job_dir, input_dir

    def start_batch_job(self, batch_id: str) -> subprocess.Popen:
        """Launch the worker subprocess for the batch job."""
        job_dir = self.get_job_dir(batch_id)
        if not os.path.isdir(job_dir):
            raise FileNotFoundError(f"Batch directory not found: {job_dir}")

        if batch_id in self._processes and self._processes[batch_id].poll() is None:
            raise BatchJobManagerError(f"Batch job {batch_id} is already actively running.")

        # Read state to get speaker_nativity
        state = self.read_state(batch_id)
        speaker_nativity = state.get("speaker_nativity")

        cmd = [
            sys.executable,
            "-m",
            "backend.batch_annotation.worker",
            "--batch-id",
            batch_id,
            "--job-dir",
            job_dir,
        ]
        if speaker_nativity:
            cmd.extend(["--speaker-nativity", speaker_nativity])

        # Prepare PYTHONPATH
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(backend_dir)
        env = dict(os.environ)
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{backend_dir}{os.pathsep}{current_pp}" if current_pp else f"{repo_root}{os.pathsep}{backend_dir}"

        log_path = os.path.join(job_dir, "worker.log")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=job_dir,
                env=env,
                creationflags=creationflags,
            )
        self._processes[batch_id] = proc
        return proc

    def cancel_batch_job(self, batch_id: str) -> Dict[str, Any]:
        """Cancel a running batch annotation job gracefully."""
        job_dir = self.get_job_dir(batch_id)
        state = self.read_state(batch_id)

        # Write cooperative cancel signal file
        cancel_signal_file = os.path.join(job_dir, "cancel.signal")
        try:
            with open(cancel_signal_file, "w", encoding="utf-8") as f:
                f.write("cancel")
        except Exception:
            pass

        proc = self._processes.get(batch_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        if state.get("status") not in ["completed", "completed_with_errors", "failed"]:
            state["status"] = "cancelled"
            state["error"] = "Job cancelled by user"
            # If any files were completed before cancellation, build partial result ZIP
            output_dir = os.path.join(job_dir, "output")
            json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "manifest.json"] if os.path.isdir(output_dir) else []
            if json_files:
                self.build_result_zip(batch_id, state)
            self.write_atomic_state(job_dir, state)

        return state

    def build_result_zip(self, batch_id: str, state: Optional[Dict[str, Any]] = None) -> str:
        """
        Creates prosodylab_annotations_<batch_id>.zip containing ONLY:
          annotations/
            audio1.json
            audio2.json
            ...
          manifest.json

        Strictly guarantees:
        - NO audio files (.wav, .mp3, .flac, .m4a, .ogg)
        - NO worker logs
        - NO state files
        - NO source zip archives
        """
        job_dir = self.get_job_dir(batch_id)
        output_dir = os.path.join(job_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        if state is None:
            state = self.read_state(batch_id)

        # Build clean manifest
        manifest = {
            "batch_id": batch_id,
            "status": state.get("status"),
            "total_files": state.get("total_files", 0),
            "completed_files": state.get("completed_files", 0),
            "failed_files": state.get("failed_files", 0),
            "speaker_nativity": state.get("speaker_nativity"),
            "created_at": state.get("created_at"),
            "completed_at": time.time(),
            "files": [
                {
                    "input_name": f.get("input_name"),
                    "output_name": f.get("output_name"),
                    "status": f.get("status"),
                    "duration": f.get("duration"),
                    "words_count": f.get("words_count"),
                    "error": f.get("error"),
                }
                for f in state.get("files", [])
            ]
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        zip_filename = f"prosodylab_annotations_{batch_id}.zip"
        zip_path = os.path.join(job_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            zf.write(manifest_path, arcname="manifest.json")

            # Add all generated json annotations under annotations/
            for f in os.listdir(output_dir):
                if f.endswith(".json") and f != "manifest.json":
                    file_path = os.path.join(output_dir, f)
                    if os.path.isfile(file_path):
                        zf.write(file_path, arcname=f"annotations/{f}")

        return zip_path

    def get_result_zip_path(self, batch_id: str) -> Optional[str]:
        """Returns the path to the result ZIP if it exists."""
        job_dir = self.get_job_dir(batch_id)
        zip_filename = f"prosodylab_annotations_{batch_id}.zip"
        zip_path = os.path.join(job_dir, zip_filename)
        if os.path.isfile(zip_path):
            return zip_path
        return None
