from fastapi import FastAPI, UploadFile, File, HTTPException
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

@app.post("/api/annotate")
async def annotate(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        from annotation.pipeline import run_annotation_pipeline
        annotation = run_annotation_pipeline(audio_bytes, file.filename)
        return annotation

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")

