"""
backend.training.api

FastAPI router for the LexiRep Training Workbench.
Endpoints:
- POST /api/training/dataset/validate
- POST /api/training/anchors/search
- POST /api/training/jobs
- GET  /api/training/jobs
- GET  /api/training/jobs/{job_id}
- POST /api/training/jobs/{job_id}/cancel
- GET  /api/training/jobs/{job_id}/bundle
- GET  /api/training/jobs/{job_id}/artifacts/{name}

Enforces:
- Strict Pydantic models
- Upload size limit (250 MB)
- UUIDv4 dataset & job sandboxing
- Server-side anchor verification
- Language protocol resolution (GER=1/32, ITA=10/48, CUSTOM)
- Single-active-job concurrency protection
- Directory traversal defenses
- Safe zip bundle and artifact download
"""

import os
import io
import json
import zipfile
import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from backend.training.validator import validate_dataset, DatasetValidationResult
from backend.training.anchor_scanner import AnchorScanner, AnchorScanResult
from backend.training.trainer import TrainingConfig
from backend.training.job_manager import TrainingJobManager, UUID4_PATTERN
from backend.training.dataset_loader import (
    load_dataset_npz,
    DatasetRegistry,
    DatasetLoaderError,
    DatasetTooLargeError,
    MAX_DATASET_UPLOAD_BYTES,
)

router = APIRouter(prefix="/api/training", tags=["training"])

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_job_manager: Optional[TrainingJobManager] = None
_dataset_registry: Optional[DatasetRegistry] = None

ALLOWED_ARTIFACTS = {
    "metrics.json",
    "training_history.csv",
    "anchors.json",
    "config.json",
    "evaluation_summary.txt",
    "training.log",
}


def get_job_manager() -> TrainingJobManager:
    global _job_manager
    if _job_manager is None:
        root = os.environ.get("TRAINING_JOBS_DIR") or os.path.abspath(os.path.join(_BACKEND_DIR, "training_jobs"))
        _job_manager = TrainingJobManager(root)
    return _job_manager


def get_dataset_registry() -> DatasetRegistry:
    global _dataset_registry
    if _dataset_registry is None:
        mgr = get_job_manager()
        datasets_dir = os.path.join(mgr.root_dir, "datasets")
        _dataset_registry = DatasetRegistry(datasets_dir)
    return _dataset_registry


def set_job_manager(manager: Optional[TrainingJobManager]) -> None:
    global _job_manager
    _job_manager = manager


def set_dataset_registry(registry: Optional[DatasetRegistry]) -> None:
    global _dataset_registry
    _dataset_registry = registry


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class DatasetValidationResponse(BaseModel):
    dataset_id: str
    valid: bool
    filename: str
    syllable_count: int
    word_count: int
    stressed_count: int
    unstressed_count: int
    stress_ratio: float
    bisyllabic_anchor_candidate_count: int
    rejected_word_count: int
    feature_shape: List[int]
    feature_dtype: str
    warnings: List[str] = []
    errors: List[str] = []


class AnchorCandidateModel(BaseModel):
    rank: int
    word_id: int
    stressed_syllable_index: int
    unstressed_syllable_index: int
    separation_score: float
    word_text: Optional[str] = None


class AnchorSearchRequest(BaseModel):
    dataset_id: str
    max_candidates: int = Field(default=20, ge=1, le=100)


class AnchorSearchResponse(BaseModel):
    dataset_id: str
    total_training_syllables: int
    total_training_words: int
    bisyllabic_candidates: int
    rejected_words: int
    scan_time_seconds: float
    top_anchors: List[AnchorCandidateModel]
    selected_anchor: Optional[AnchorCandidateModel] = None


class AnchorSelection(BaseModel):
    word_id: int
    stressed_syllable_index: int
    unstressed_syllable_index: int


class TrainingConfigRequest(BaseModel):
    seed: int = 42
    device: str = "auto"
    warmup_epochs: Optional[int] = Field(default=None, ge=1, le=100)
    idec_pretrain_epochs: Optional[int] = Field(default=None, ge=1, le=100)
    outer_loops: Optional[int] = Field(default=None, ge=1, le=50)
    idec_steps_per_loop: Optional[int] = Field(default=None, ge=1, le=1000)
    refinement_epochs: Optional[int] = Field(default=None, ge=1, le=50)
    supcon_temperature: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    batch_words: Optional[int] = Field(default=None, ge=1, le=1024)
    idec_batch_size: Optional[int] = Field(default=None, ge=1, le=2048)
    top_k_rep: Optional[int] = Field(default=None, ge=1, le=100)
    learning_rate: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class CreateTrainingJobRequest(BaseModel):
    dataset_id: str
    language: str = "GER"
    anchor: AnchorSelection
    config: Optional[TrainingConfigRequest] = None


class CreateTrainingJobResponse(BaseModel):
    job_id: str
    status: str


class JobSummaryResponse(BaseModel):
    job_id: str
    language: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    progress_percent: float = 0.0
    current_step: str = ""


class JobStateResponse(BaseModel):
    job_id: str
    status: str
    current_step: str
    progress_percent: float
    loop: int
    total_loops: int
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float] = None
    pseudo_label_changes: int = 0
    idec_loss: Optional[float] = None
    training_accuracy: Optional[float] = None
    current_metrics: Optional[Dict[str, Any]] = None
    final_metrics: Optional[Dict[str, Any]] = None
    loss_history: List[Dict[str, Any]] = []
    error: Optional[str] = None
    language: Optional[str] = None
    selected_anchor: Optional[Dict[str, Any]] = None
    benchmark: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ==============================================================================
# ROUTE HANDLERS
# ==============================================================================

@router.post("/dataset/validate", response_model=DatasetValidationResponse)
async def validate_dataset_upload(
    file: UploadFile = File(...),
    registry: DatasetRegistry = Depends(get_dataset_registry),
):
    """
    Upload and validate a candidate LexiRep dataset archive (.npz).
    Returns structural validation diagnostics and registers dataset for subsequent jobs.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not file.filename.lower().endswith(".npz"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file.filename}'. LexiRep datasets must have a .npz extension."
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    if len(content) > MAX_DATASET_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded dataset exceeds maximum size of 250 MB ({len(content)} bytes uploaded)."
        )

    try:
        dataset_id, npz_path, val_res = registry.save_dataset(content, file.filename)
    except DatasetTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except (DatasetLoaderError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error processing dataset: {str(e)}")

    return DatasetValidationResponse(
        dataset_id=dataset_id,
        valid=val_res.valid,
        filename=os.path.basename(file.filename),
        syllable_count=val_res.syllable_count,
        word_count=val_res.word_count,
        stressed_count=val_res.stressed_count,
        unstressed_count=val_res.unstressed_count,
        stress_ratio=round(val_res.stress_ratio, 4),
        bisyllabic_anchor_candidate_count=val_res.bisyllabic_anchor_candidate_count,
        rejected_word_count=val_res.rejected_word_count,
        feature_shape=list(val_res.feature_shape),
        feature_dtype=val_res.feature_dtype,
        warnings=list(val_res.warnings),
        errors=list(val_res.errors),
    )


@router.post("/anchors/search", response_model=AnchorSearchResponse)
def search_candidate_anchors(
    req: AnchorSearchRequest,
    registry: DatasetRegistry = Depends(get_dataset_registry),
):
    """
    Search and rank candidate bisyllabic anchor pairs for a registered dataset.
    """
    try:
        npz_path = registry.get_dataset_path(req.dataset_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    try:
        data = load_dataset_npz(npz_path)
    except DatasetLoaderError as e:
        raise HTTPException(status_code=400, detail=f"Dataset failed to load: {str(e)}")

    # Run linguistic validation
    val_res = validate_dataset(data["X_tr"], data["Y_tr"], data["W_tr"])
    if not val_res.valid:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot scan invalid dataset: {'; '.join(val_res.errors)}"
        )

    # Run exact Phase 1 AnchorScanner
    scanner = AnchorScanner(max_candidates=req.max_candidates, device="cpu")
    scan_res = scanner.scan(
        data["X_tr"],
        data["Y_tr"],
        data["W_tr"],
        word_texts=data.get("word_text"),
    )

    top_candidates = [
        AnchorCandidateModel(
            rank=c.rank,
            word_id=c.word_id,
            stressed_syllable_index=c.stressed_syllable_index,
            unstressed_syllable_index=c.unstressed_syllable_index,
            separation_score=round(c.separation_score, 4),
            word_text=c.word_text,
        )
        for c in scan_res.top_anchors
    ]

    selected = None
    if scan_res.selected_anchor:
        sa = scan_res.selected_anchor
        selected = AnchorCandidateModel(
            rank=sa.rank,
            word_id=sa.word_id,
            stressed_syllable_index=sa.stressed_syllable_index,
            unstressed_syllable_index=sa.unstressed_syllable_index,
            separation_score=round(sa.separation_score, 4),
            word_text=sa.word_text,
        )

    return AnchorSearchResponse(
        dataset_id=req.dataset_id,
        total_training_syllables=scan_res.total_training_syllables,
        total_training_words=scan_res.total_training_words,
        bisyllabic_candidates=scan_res.bisyllabic_candidates,
        rejected_words=scan_res.rejected_words,
        scan_time_seconds=round(scan_res.scan_time_seconds, 3),
        top_anchors=top_candidates,
        selected_anchor=selected,
    )


@router.post("/jobs", response_model=CreateTrainingJobResponse, status_code=201)
def create_training_job(
    req: CreateTrainingJobRequest,
    registry: DatasetRegistry = Depends(get_dataset_registry),
    manager: TrainingJobManager = Depends(get_job_manager),
):
    """
    Register and launch a new LexiRep training job in an isolated subprocess.
    Enforces concurrency limits, server-side anchor validation, and language protocol resolution.
    """
    # 1. Enforce single-active-job concurrency policy
    active_job_id = manager.has_active_job()
    if active_job_id:
        raise HTTPException(
            status_code=409,
            detail=f"A training job is already active ('{active_job_id}'). Only one active training job is permitted."
        )

    # 2. Validate language
    lang_upper = req.language.strip().upper()
    if lang_upper not in ["GER", "ITA", "CUSTOM"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{req.language}'. Supported values: 'GER', 'ITA', 'CUSTOM'."
        )

    # 3. Locate dataset
    try:
        npz_path = registry.get_dataset_path(req.dataset_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    try:
        data = load_dataset_npz(npz_path)
    except DatasetLoaderError as e:
        raise HTTPException(status_code=400, detail=f"Dataset loading failed: {str(e)}")

    X_tr, Y_tr, W_tr = data["X_tr"], data["Y_tr"], data["W_tr"]
    n_samples = len(X_tr)

    # 4. Strict Anchor Validation Safety
    s_idx = req.anchor.stressed_syllable_index
    u_idx = req.anchor.unstressed_syllable_index
    wid = req.anchor.word_id

    if not (0 <= s_idx < n_samples and 0 <= u_idx < n_samples):
        raise HTTPException(
            status_code=400,
            detail=f"Anchor syllable indices [{s_idx}, {u_idx}] are out of bounds for dataset with {n_samples} syllables."
        )

    if s_idx == u_idx:
        raise HTTPException(status_code=400, detail="Stressed and unstressed anchor indices must be distinct.")

    if W_tr[s_idx] != wid or W_tr[u_idx] != wid:
        raise HTTPException(
            status_code=400,
            detail=f"Anchor syllables [{s_idx}, {u_idx}] belong to word IDs [{W_tr[s_idx]}, {W_tr[u_idx]}], not requested word_id {wid}."
        )

    word_syls = np.where(W_tr == wid)[0]
    if len(word_syls) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Anchor word {wid} must be strictly bisyllabic (found {len(word_syls)} syllables)."
        )

    if int(Y_tr[s_idx]) != 1 or int(Y_tr[u_idx]) != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid anchor labels: index {s_idx} has Y={Y_tr[s_idx]} (expected 1), index {u_idx} has Y={Y_tr[u_idx]} (expected 0)."
        )

    # Recompute separation score server-side
    try:
        computed_sep_score = AnchorScanner.compute_single_anchor_score(X_tr, Y_tr, W_tr, s_idx, u_idx, device="cpu")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to verify anchor separation score: {str(e)}")

    verified_anchor = {
        "word_id": int(wid),
        "stressed_syllable_index": int(s_idx),
        "unstressed_syllable_index": int(u_idx),
        "separation_score": float(computed_sep_score),
    }

    # 5. Construct TrainingConfig with language protocol resolution
    cfg_args: Dict[str, Any] = {"language": lang_upper}
    if req.config:
        cfg_dict = req.config.model_dump(exclude_unset=True)
        for k, v in cfg_dict.items():
            if v is not None:
                cfg_args[k] = v

    try:
        training_config = TrainingConfig(**cfg_args)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid training configuration: {str(e)}")

    # 6. Create and launch job
    try:
        job_id = manager.create_job(
            dataset_path=npz_path,
            config=training_config,
            language=lang_upper,
            anchor=verified_anchor,
        )
        manager.start_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch training job: {str(e)}")

    return CreateTrainingJobResponse(job_id=job_id, status="queued")


@router.get("/jobs", response_model=List[JobSummaryResponse])
def list_training_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    manager: TrainingJobManager = Depends(get_job_manager),
):
    """
    List recent training jobs, sorted newest first.
    """
    all_jobs = manager.list_jobs()

    # Enrich and sort
    summaries: List[JobSummaryResponse] = []
    for j in all_jobs:
        jid = j.get("job_id", "")
        job_dir = manager.get_job_dir(jid)
        cfg_path = os.path.join(job_dir, "config.json")
        lang = "GER"
        created_at = None
        updated_at = None

        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    lang = cfg.get("language", "GER")
            except Exception:
                pass

        if os.path.isdir(job_dir):
            try:
                mtime = os.path.getmtime(job_dir)
                created_at = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()
                state_file = os.path.join(job_dir, "job_state.json")
                if os.path.isfile(state_file):
                    s_mtime = os.path.getmtime(state_file)
                    updated_at = datetime.datetime.fromtimestamp(s_mtime, datetime.timezone.utc).isoformat()
            except Exception:
                pass

        summaries.append(
            JobSummaryResponse(
                job_id=jid,
                language=lang,
                status=j.get("status", "unknown"),
                created_at=created_at,
                updated_at=updated_at,
                progress_percent=float(j.get("progress_percent", 0.0)),
                current_step=str(j.get("current_step", "")),
            )
        )

    # Sort newest first by updated_at / created_at
    summaries.sort(key=lambda s: s.updated_at or s.created_at or "", reverse=True)
    return summaries[:limit]


@router.get("/jobs/{job_id}", response_model=JobStateResponse)
def get_training_job_state(
    job_id: str,
    manager: TrainingJobManager = Depends(get_job_manager),
):
    """
    Retrieve live status and progress of a training job.
    """
    try:
        manager._validate_job_id(job_id)
        state = manager.get_state(job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job_dir = manager.get_job_dir(job_id)
    cfg_path = os.path.join(job_dir, "config.json")
    lang = None
    anchor = None
    created_at = None
    updated_at = None

    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                lang = cfg.get("language")
                anchor = cfg.get("anchor")
        except Exception:
            pass

    if os.path.isdir(job_dir):
        try:
            mtime = os.path.getmtime(job_dir)
            created_at = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()
            state_file = os.path.join(job_dir, "job_state.json")
            if os.path.isfile(state_file):
                s_mtime = os.path.getmtime(state_file)
                updated_at = datetime.datetime.fromtimestamp(s_mtime, datetime.timezone.utc).isoformat()
        except Exception:
            pass

    # Resolve current/final IDEC loss with robust fallbacks
    idec_loss = state.get("idec_loss")
    if idec_loss is None:
        curr_m = state.get("current_metrics") or {}
        idec_loss = curr_m.get("idec_loss")
    if idec_loss is None:
        fin_m = state.get("final_metrics") or {}
        idec_loss = fin_m.get("final_idec_loss") or fin_m.get("idec_loss")
    if idec_loss is None:
        for entry in reversed(state.get("loss_history", [])):
            val = entry.get("idec_loss")
            if val is not None:
                idec_loss = val
                break

    # Resolve training accuracy
    fin_m = state.get("final_metrics") or {}
    training_accuracy = state.get("training_accuracy")
    if training_accuracy is None:
        training_accuracy = fin_m.get("training_accuracy")

    return JobStateResponse(
        job_id=job_id,
        status=state.get("status", "unknown"),
        current_step=state.get("current_step", ""),
        progress_percent=float(state.get("progress_percent", 0.0)),
        loop=int(state.get("loop", 0)),
        total_loops=int(state.get("total_loops", 13)),
        elapsed_seconds=float(state.get("elapsed_seconds", 0.0)),
        estimated_remaining_seconds=state.get("estimated_remaining_seconds"),
        pseudo_label_changes=int(state.get("pseudo_label_changes", 0)),
        idec_loss=float(idec_loss) if idec_loss is not None else None,
        training_accuracy=float(training_accuracy) if training_accuracy is not None else None,
        current_metrics=state.get("current_metrics"),
        final_metrics=state.get("final_metrics"),
        loss_history=state.get("loss_history", []),
        error=state.get("error"),
        language=lang,
        selected_anchor=anchor,
        benchmark=fin_m.get("benchmark") or state.get("benchmark"),
        created_at=created_at,
        updated_at=updated_at,
    )


@router.post("/jobs/{job_id}/cancel", response_model=JobStateResponse)
def cancel_training_job(
    job_id: str,
    manager: TrainingJobManager = Depends(get_job_manager),
):
    """
    Cancel an active or queued training job.
    """
    try:
        manager._validate_job_id(job_id)
        current_state = manager.get_state(job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    status = current_state.get("status")
    if status == "completed":
        raise HTTPException(status_code=409, detail="Cannot cancel a job that has already completed.")

    if status == "cancelled":
        return get_training_job_state(job_id, manager=manager)

    try:
        manager.cancel_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

    return get_training_job_state(job_id, manager=manager)


@router.get("/jobs/{job_id}/bundle")
def download_model_bundle(
    job_id: str,
    manager: TrainingJobManager = Depends(get_job_manager),
):
    """
    Download production model bundle for a successfully completed training job.
    """
    try:
        manager._validate_job_id(job_id)
        state = manager.get_state(job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if state.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Model bundle is only available for completed jobs. Current status: '{state.get('status')}'."
        )

    job_dir = manager.get_job_dir(job_id)

    bundle_items = [
        ("config.json", "config.json"),
        ("anchors.json", "anchors.json"),
        (os.path.join("model", "final_lexirep_model.pt"), "model/final_lexirep_model.pt"),
        (os.path.join("anchors", "prototypes_10d.npy"), "anchors/prototypes_10d.npy"),
    ]

    for rel_path, _ in bundle_items:
        full_path = os.path.join(job_dir, rel_path)
        if not os.path.isfile(full_path):
            raise HTTPException(
                status_code=404,
                detail=f"Bundle artifact '{rel_path}' is missing from job directory."
            )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path, arcname in bundle_items:
            full_path = os.path.join(job_dir, rel_path)
            zf.write(full_path, arcname=arcname)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="lexirep_model_bundle.zip"'},
    )


@router.get("/jobs/{job_id}/artifacts/{name:path}")
def download_job_artifact(
    job_id: str,
    name: str,
    manager: TrainingJobManager = Depends(get_job_manager),
):
    """
    Download a specific whitelisted artifact file from a training job.
    """
    try:
        manager._validate_job_id(job_id)
        job_dir = manager.get_job_dir(job_id)
        if not os.path.isdir(job_dir):
            raise FileNotFoundError()
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Guard against traversal
    if ".." in name or "/" in name or "\\" in name or ":" in name:
        raise HTTPException(status_code=400, detail="Directory traversal characters are not permitted.")

    if name not in ALLOWED_ARTIFACTS:
        raise HTTPException(
            status_code=400,
            detail=f"Artifact '{name}' is not in the allowed artifact list."
        )

    # Search in job root or artifacts/ subfolder
    candidate_paths = [
        os.path.join(job_dir, name),
        os.path.join(job_dir, "artifacts", name),
    ]
    if name == "training.log":
        candidate_paths.append(os.path.join(job_dir, "worker.log"))

    target_path = None
    for cp in candidate_paths:
        if os.path.isfile(cp):
            target_path = cp
            break

    if target_path is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{name}' not found for job '{job_id}'.")

    # Extra containment verification
    if os.path.commonpath([os.path.abspath(target_path), os.path.abspath(job_dir)]) != os.path.abspath(job_dir):
        raise HTTPException(status_code=400, detail="Invalid path.")

    media_type = "application/json" if name.endswith(".json") else "text/plain"
    return FileResponse(target_path, filename=name, media_type=media_type)
