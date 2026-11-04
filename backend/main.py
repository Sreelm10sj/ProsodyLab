import sys
from pathlib import Path
_backend_dir = str(Path(__file__).resolve().parent)
_root_dir = str(Path(__file__).resolve().parent.parent)
for _p in [_root_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from whistress_service import service
import traceback
import math
import numpy as np

from pitch_analysis import PITCH_TIME_STEP, compute_pause_before, PAUSE_ANNOTATION_THRESHOLD_S

app = FastAPI(title="WhiStress API")

# Setup CORS to allow Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pitch_contour(
    pitch_times: np.ndarray,
    pitch_f0: np.ndarray,
    utterance_median_hz: float | None,
    target_step_ms: float = 25.0,
    k_array: np.ndarray | None = None,
    P_val: int | None = None,
) -> list[dict]:
    if len(pitch_times) == 0:
        return []

    frame_step_ms = PITCH_TIME_STEP * 1000.0          # 10 ms
    stride = max(1, round(target_step_ms / frame_step_ms))   # ≈ 2–3

    contour = []
    for i in range(0, len(pitch_times), stride):
        t = float(pitch_times[i])
        f = float(pitch_f0[i])

        if utterance_median_hz is None or np.isnan(f) or f <= 0:
            semitone = None
        else:
            semitone = round(12.0 * math.log2(f / utterance_median_hz), 3)

        point = {"time": round(t, 3), "semitone": semitone}
        
        if k_array is not None and P_val is not None:
            k = float(k_array[i])
            if not np.isnan(k):
                point["K"] = int(k)
                point["P"] = P_val
                
        contour.append(point)

    return contour


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    service.load_model()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "device": service.device,
        "model_loaded": service.client is not None,
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # ------------------------------------------------------------------
        # Run WhiStress inference + phrase-level intonation analysis.
        # predict_with_intonation() processes audio once and returns:
        #   pairs         — (word, stress, start, end) tuples
        #   phrases       — phrase-level intonation objects (existing field)
        #   pitch_times   - raw frame timestamps (for pitch_contour)
        #   pitch_f0      - raw F0 values in Hz  (for pitch_contour)
        #   audio_array   - raw acoustic signal  (for energy analysis)
        #   sr            - sample rate
        # ------------------------------------------------------------------
        pairs, pauses, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)

        # Build existing response fields — words array.
        # pause_before is added as an OPTIONAL field on each word object.
        # The first word always has pause_before=None (nothing precedes it).
        words = []
        for idx, (word, stress, start, end) in enumerate(pairs):
            pause_before = pauses[idx - 1] if idx > 0 else None
            # Filter output based on PAUSE_ANNOTATION_THRESHOLD_S
            if pause_before is not None and pause_before["duration"] <= PAUSE_ANNOTATION_THRESHOLD_S:
                pause_before = None
                
            words.append({
                "word":         word,
                "stressed":     bool(stress),
                "start":        float(start),
                "end":          float(end),
                "pause_before": pause_before,
            })

        transcription = " ".join(w for w, *_ in pairs)

        # ------------------------------------------------------------------
        # New: pitch_contour — downsampled, utterance-median-normalised
        # New: utterance_median_hz — reference Hz for the semitone conversion
        #      (frontend uses it to reconstruct approximate absolute Hz)
        # ------------------------------------------------------------------
        voiced = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
        utterance_median_hz = float(np.median(voiced)) if len(voiced) > 0 else None

        pitch_contour = _build_pitch_contour(pitch_times, pitch_f0, utterance_median_hz)
        pitch_stylized_contour = _build_pitch_contour(pitch_times, pitch_stylized, utterance_median_hz, k_array=k_array, P_val=P_val)

        return {
            "transcription":      transcription,
            "words":              words,
            "phrases":            phrases,
            # ── Phase 4A additions ─────────────────────────────────────
            "pitch_contour":      pitch_contour,
            "pitch_stylized":     pitch_stylized_contour,
            "utterance_median_hz": utterance_median_hz,
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

from typing import Optional
from backend.lexirep.service import SUPPORTED_LANGUAGES

@app.post("/api/annotate")
async def annotate(
    file: UploadFile = File(...),
    speaker_nativity: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    nativity_clean = None
    if speaker_nativity is not None and speaker_nativity.strip():
        nativity_clean = speaker_nativity.strip().upper()
        if nativity_clean not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported speaker_nativity: '{speaker_nativity}'. Must be one of {sorted(list(SUPPORTED_LANGUAGES))}."
            )

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        from annotation.pipeline import run_annotation_pipeline
        annotation = run_annotation_pipeline(
            audio_bytes,
            file.filename,
            speaker_nativity=nativity_clean,
        )
        return annotation

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")


# ---------------------------------------------------------------------------
# LexiRep Isolated Verification Endpoint (Stage 5)
# ---------------------------------------------------------------------------
# NOTE: This endpoint is an isolated LexiRep verification interface and is not
# yet the primary ProsodyLab annotation endpoint. It provides syllable-level
# lexical stress analysis via frozen GER/ITA LexiRep models without modifying
# /api/transcribe or /api/annotate.
# ---------------------------------------------------------------------------

from backend.lexirep.pipeline import LexiRepPipeline
from backend.lexirep.service import SUPPORTED_LANGUAGES

# Singleton instance of LexiRepPipeline for isolated LexiRep endpoint
lexirep_pipeline = LexiRepPipeline()


@app.post("/api/lexirep/analyze")
async def analyze_lexirep(
    file: UploadFile = File(...),
    nativity: str = Form(...),
):
    """
    Isolated verification endpoint for syllable-level lexical stress analysis.
    Requires explicit nativity ("GER" or "ITA") and audio file upload.
    """
    if not nativity or not nativity.strip():
        raise HTTPException(
            status_code=422,
            detail="Missing required 'nativity' field. Must be 'GER' or 'ITA'."
        )

    nativity_clean = nativity.strip().upper()
    if nativity_clean not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported nativity: '{nativity}'. Must be one of {sorted(list(SUPPORTED_LANGUAGES))}."
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Execute Stage 4 end-to-end pipeline
        result = lexirep_pipeline.process(audio=audio_bytes, nativity=nativity_clean)

        # Build dedicated JSON-safe response schema
        words_response = []
        for w_idx, w in enumerate(result.words):
            syllables_response = []
            for s in w.syllables:
                syllables_response.append({
                    "index": int(s.index),
                    "text": str(s.text),
                    "start": round(float(s.start), 4),
                    "end": round(float(s.end), 4),
                    "duration": round(float(s.duration), 4),
                    "stress": int(s.stress) if s.stress is not None else None,
                    "contrast_score": round(float(s.contrast_score), 4) if s.contrast_score is not None else None,
                    "confidence": round(float(s.confidence), 4) if s.confidence is not None else None,
                })

            words_response.append({
                "word_index": int(w_idx),
                "text": str(w.word),
                "start": round(float(w.start), 4),
                "end": round(float(w.end), 4),
                "whistress_stressed": bool(w.whistress_stressed),
                "is_polysyllabic": bool(w.is_polysyllabic),
                "syllable_count": int(w.syllable_count),
                "syllable_source": str(w.syllable_source) if w.syllable_source else None,
                "lexirep_stressed_syllable_index": int(w.lexirep_stressed_syllable) if w.lexirep_stressed_syllable is not None else None,
                "syllables": syllables_response,
            })

        stats = result.execution_stats
        diagnostics = {
            "whisper_ms": round(float(stats.get("whisper_time_s", 0.0)) * 1000.0, 2),
            "realignment_ms": round(float(stats.get("realign_time_s", 0.0)) * 1000.0, 2),
            "syllable_ms": round(float(stats.get("syllable_engine_time_s", 0.0)) * 1000.0, 2),
            "wav2vec_ms": round(float(stats.get("wav2vec_extraction_time_s", 0.0)) * 1000.0, 2),
            "pooling_ms": round(float(stats.get("pooling_time_s", 0.0)) * 1000.0, 2),
            "lexirep_ms": round(float(stats.get("lexirep_inference_time_s", 0.0)) * 1000.0, 2),
            "total_ms": round(float(stats.get("total_pipeline_time_s", 0.0)) * 1000.0, 2),
        }

        return {
            "status": "ok",
            "nativity": str(result.nativity),
            "transcription": str(result.transcription),
            "audio": {
                "duration": round(float(result.audio_duration), 4),
                "sample_rate": 16000,
            },
            "words": words_response,
            "diagnostics": diagnostics,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LexiRep analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# Training Workbench Router
# ---------------------------------------------------------------------------
try:
    from fastapi.routing import _IncludedRouter
    if not hasattr(_IncludedRouter, "path"):
        _IncludedRouter.path = property(lambda self: getattr(self.include_context, "prefix", ""))
except ImportError:
    pass

from backend.training.api import router as training_router
app.include_router(training_router)

from backend.batch_annotation.api import router as batch_router
app.include_router(batch_router)




