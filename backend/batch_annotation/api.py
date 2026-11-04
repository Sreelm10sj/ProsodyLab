"""
backend.batch_annotation.api

FastAPI router for batch audio annotation endpoints:
- POST /api/batch-annotate
- GET  /api/batch-annotate/{batch_id}
- POST /api/batch-annotate/{batch_id}/cancel
- GET  /api/batch-annotate/{batch_id}/download
"""

import os
import shutil
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse

from backend.lexirep.service import SUPPORTED_LANGUAGES
from backend.batch_annotation.validator import (
    validate_zip_archive,
    extract_batch_audio_files,
    BatchValidationError,
    MAX_BATCH_ZIP_BYTES,
)
from backend.batch_annotation.job_manager import BatchJobManager, BatchJobManagerError

router = APIRouter(prefix="/api/batch-annotate", tags=["batch-annotation"])
job_manager = BatchJobManager()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_batch_annotation(
    file: UploadFile = File(...),
    speaker_nativity: Optional[str] = Form(None),
):
    """
    Accepts a ZIP file containing multiple audio files, safely validates
    the archive, stages the files, and launches the background annotation worker.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Validate speaker_nativity if supplied
    nativity_clean = None
    if speaker_nativity is not None and speaker_nativity.strip():
        nativity_clean = speaker_nativity.strip().upper()
        if nativity_clean not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported speaker_nativity: '{speaker_nativity}'. Must be one of {sorted(list(SUPPORTED_LANGUAGES))}."
            )

    # Stage the uploaded archive to a temporary file
    temp_dir = tempfile.mkdtemp(prefix="batch_upload_")
    temp_zip_path = os.path.join(temp_dir, "upload.zip")

    try:
        total_bytes = 0
        with open(temp_zip_path, "wb") as f_out:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_BATCH_ZIP_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum allowable archive size of {MAX_BATCH_ZIP_BYTES // (1024 * 1024)} MB."
                    )
                f_out.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Inspect and validate ZIP contents safely
        try:
            valid_entries = validate_zip_archive(temp_zip_path)
        except BatchValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Create batch sandbox
        batch_id, job_dir, input_dir = job_manager.create_batch_job(
            valid_entries=valid_entries,
            speaker_nativity=nativity_clean,
        )

        # Extract only validated audio files into the batch sandbox input directory
        try:
            extract_batch_audio_files(
                zip_path=temp_zip_path,
                valid_entries=valid_entries,
                destination_dir=input_dir,
            )
        except BatchValidationError as e:
            # Clean up sandbox on extraction failure
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(e))

        # Launch background worker subprocess
        try:
            job_manager.start_batch_job(batch_id)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start batch annotation worker: {str(e)}"
            )

        return {
            "batch_id": batch_id,
            "status": "queued",
            "total_files": len(valid_entries),
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Polls the current status and progress of a batch annotation job.
    """
    try:
        job_manager.validate_batch_id(batch_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Batch job not found: '{batch_id}'")

    try:
        state = job_manager.read_state(batch_id)
        return state
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Batch job not found: '{batch_id}'")


@router.post("/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    """
    Cancels an actively running batch annotation job.
    """
    try:
        job_manager.validate_batch_id(batch_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Batch job not found: '{batch_id}'")

    try:
        state = job_manager.cancel_batch_job(batch_id)
        return state
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Batch job not found: '{batch_id}'")


@router.get("/{batch_id}/download")
async def download_batch_results(batch_id: str):
    """
    Downloads the completed batch annotations ZIP archive.
    Only permitted once processing has reached a terminal state and at least one
    annotation has been successfully generated.
    """
    try:
        job_manager.validate_batch_id(batch_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Batch job not found: '{batch_id}'")

    try:
        state = job_manager.read_state(batch_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Batch job not found: '{batch_id}'")

    status_val = state.get("status")
    terminal_statuses = ["completed", "completed_with_errors", "cancelled"]
    if status_val not in terminal_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Batch job is still in progress (status: '{status_val}'). Results cannot be downloaded yet."
        )

    completed_files = state.get("completed_files", 0)
    if completed_files == 0:
        raise HTTPException(
            status_code=404,
            detail="No annotations were generated for this batch job."
        )

    zip_path = job_manager.get_result_zip_path(batch_id)
    if not zip_path or not os.path.isfile(zip_path):
        # Generate if missing
        zip_path = job_manager.build_result_zip(batch_id, state)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"prosodylab_annotations_{batch_id[:8]}.zip",
    )
