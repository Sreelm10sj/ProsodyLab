"""
backend.batch_annotation.worker

Background worker process for batch audio annotation.
Sequentially processes audio files through the canonical annotation pipeline,
records progress atomically after each file, and builds the final annotations ZIP archive.
"""

import os
import sys
import json
import math
import time
import argparse
import traceback
import gc
import shutil
import uuid
from typing import Any, Dict, Optional

# Ensure backend and root are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_backend_dir)
for _p in [_root_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from whistress_service import service
from annotation.pipeline import run_annotation_pipeline, get_lexirep_pipeline
from backend.batch_annotation.job_manager import BatchJobManager


def sanitize_json_floats(obj: Any) -> Any:
    """
    Recursively replaces NaN and Infinity floats with None (null in JSON)
    to guarantee strictly compliant JSON serialization.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_json_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_floats(v) for v in obj]
    return obj


def is_cancel_requested(job_dir: str) -> bool:
    """Checks if a cancellation signal file exists."""
    return os.path.isfile(os.path.join(job_dir, "cancel.signal"))


def safe_write_state(mgr: BatchJobManager, job_dir: str, state: Dict[str, Any]) -> None:
    """Safely updates batch_state.json, ensuring transient write errors never crash the worker."""
    try:
        mgr.write_atomic_state(job_dir, state)
    except Exception as e:
        print(f"[WARN] Non-fatal batch_state write error: {e}")


def run_worker(batch_id: str, job_dir: str, speaker_nativity: Optional[str] = None):
    print(f"=== Starting Batch Annotation Worker ===")
    print(f"Batch ID: {batch_id}")
    print(f"Job Directory: {job_dir}")
    print(f"Speaker Nativity: {speaker_nativity}")

    mgr = BatchJobManager(os.path.dirname(job_dir))
    state = mgr.read_state(batch_id)

    input_dir = os.path.join(job_dir, "input")
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Preload models
    print("Loading WhiStress model...")
    service.load_model()
    lexirep_pipe = None
    if speaker_nativity in ["GER", "ITA"]:
        print(f"Preloading LexiRep pipeline for {speaker_nativity}...")
        lexirep_pipe = get_lexirep_pipeline()

    start_time = time.time()
    state["status"] = "processing"
    safe_write_state(mgr, job_dir, state)

    files = state.get("files", [])
    total_files = len(files)
    completed_count = 0
    failed_count = 0

    for idx, file_entry in enumerate(files):
        # 1. Check for cancellation
        if is_cancel_requested(job_dir):
            print(f"Cancellation requested. Stopping batch at file {idx + 1}/{total_files}.")
            state["status"] = "cancelled"
            state["error"] = "Job cancelled by user"
            break

        input_name = file_entry["input_name"]
        output_name = file_entry["output_name"]
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, output_name)

        print(f"\n--- [{idx + 1}/{total_files}] Processing: {input_name} ---")

        # Update per-file state to processing
        file_entry["status"] = "processing"
        state["current_file"] = input_name
        state["current_file_index"] = idx + 1
        safe_write_state(mgr, job_dir, state)

        try:
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Input audio file not found on disk: {input_name}")

            with open(input_path, "rb") as f:
                audio_bytes = f.read()

            if not audio_bytes:
                raise ValueError(f"Audio file is empty: {input_name}")

            # Execute canonical annotation pipeline
            t_file_start = time.time()
            annotation = run_annotation_pipeline(
                audio_bytes=audio_bytes,
                filename=input_name,
                speaker_nativity=speaker_nativity,
                lexirep_pipeline=lexirep_pipe,
            )
            t_file_elapsed = time.time() - t_file_start

            # Validate required top-level schema keys
            required_keys = ["metadata", "transcription", "words", "pauses", "intonation", "timeline"]
            missing_keys = [k for k in required_keys if k not in annotation]
            if missing_keys:
                raise ValueError(f"Annotation output missing canonical keys: {missing_keys}")

            # Sanitize floats (NaN/Inf -> null)
            clean_annotation = sanitize_json_floats(annotation)

            # Write JSON output
            temp_output = f"{output_path}.tmp"
            with open(temp_output, "w", encoding="utf-8") as out_f:
                json.dump(clean_annotation, out_f, indent=2)
            try:
                os.replace(temp_output, output_path)
            except (PermissionError, OSError):
                time.sleep(0.02)
                try:
                    os.replace(temp_output, output_path)
                except Exception:
                    shutil.copyfile(temp_output, output_path)
                    if os.path.isfile(temp_output):
                        os.remove(temp_output)

            # Record success metadata
            duration = clean_annotation.get("metadata", {}).get("duration", 0.0)
            words_count = len(clean_annotation.get("words", []))

            file_entry["status"] = "completed"
            file_entry["duration"] = round(float(duration), 3) if duration else None
            file_entry["words_count"] = words_count
            file_entry["error"] = None
            completed_count += 1
            print(f"[OK] Completed {input_name} in {t_file_elapsed:.2f}s (duration: {duration}s, words: {words_count})")

            # Explicit memory reclamation
            del audio_bytes
            del annotation
            del clean_annotation

        except Exception as e:
            print(f"[FAIL] Failed processing {input_name}: {str(e)}")
            traceback.print_exc()

            file_entry["status"] = "failed"
            # User-friendly sanitized error message (no stack traces in state)
            err_msg = str(e)
            if "Unsupported speaker_nativity" in err_msg or "Invalid speaker_nativity" in err_msg:
                user_err = f"Nativity model error: {err_msg}"
            elif "Empty audio" in err_msg:
                user_err = "Audio file is empty or unreadable"
            else:
                user_err = f"Processing error: {err_msg[:120]}"
            file_entry["error"] = user_err
            failed_count += 1

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update overall batch progress
        state["completed_files"] = completed_count
        state["failed_files"] = failed_count
        processed = completed_count + failed_count
        state["progress_percent"] = round((processed / total_files) * 100.0, 1) if total_files > 0 else 100.0
        state["elapsed_seconds"] = round(time.time() - start_time, 1)
        safe_write_state(mgr, job_dir, state)

    # Final state calculation
    state["current_file"] = None
    state["elapsed_seconds"] = round(time.time() - start_time, 1)

    if is_cancel_requested(job_dir) or state.get("status") == "cancelled":
        state["status"] = "cancelled"
        if completed_count > 0:
            print(f"Building partial result ZIP for cancelled batch ({completed_count} files completed)...")
            mgr.build_result_zip(batch_id, state)
    elif completed_count == total_files and total_files > 0:
        state["status"] = "completed"
        print(f"All {total_files} files completed successfully. Building result ZIP...")
        mgr.build_result_zip(batch_id, state)
    elif completed_count > 0 and failed_count > 0:
        state["status"] = "completed_with_errors"
        print(f"Batch completed with errors ({completed_count} succeeded, {failed_count} failed). Building result ZIP...")
        mgr.build_result_zip(batch_id, state)
    else:
        state["status"] = "failed"
        state["error"] = "All audio files in batch failed to process."
        print("All audio files in batch failed. No result ZIP generated.")

    safe_write_state(mgr, job_dir, state)
    print(f"=== Worker Finished: Final Status = {state['status']} ===")


def main():
    parser = argparse.ArgumentParser(description="ProsodyLab Batch Annotation Worker")
    parser.add_argument("--batch-id", required=True, help="UUIDv4 identifier of the batch")
    parser.add_argument("--job-dir", required=True, help="Directory of the batch sandbox")
    parser.add_argument("--speaker-nativity", default=None, help="Optional LexiRep speaker nativity ('GER' or 'ITA')")
    args = parser.parse_args()

    try:
        run_worker(
            batch_id=args.batch_id,
            job_dir=args.job_dir,
            speaker_nativity=args.speaker_nativity,
        )
    except Exception as e:
        print(f"Fatal worker exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
