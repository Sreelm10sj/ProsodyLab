# ProsodyLab — REST API Reference Specification

**Version:** 2.0  
**Base URL:** `http://localhost:8000`  
**Protocol:** HTTP / REST  
**Data Interchange:** JSON / Multipart Form-Data

---

## 1. System Health & Diagnostics

### `GET /api/health`
Verifies backend service operational readiness, execution device (CUDA GPU vs. CPU), and loaded deep learning models.

#### Request
```http
GET /api/health HTTP/1.1
Host: localhost:8000
Accept: application/json
```

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "lexirep_models": {
    "GER": true,
    "ITA": true
  }
}
```

---

## 2. Real-Time Inference Endpoints

### `POST /api/transcribe`
Transcribes audio input using WhiStress, generating word tokens with timestamp boundaries, sentence-level lexical prominence predictions, and raw/stylized pitch contours.

#### Request
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` *(binary audio file, required)*: Supported formats include WAV, MP3, WebM, OGG, FLAC.
  - `stylize` *(boolean, optional, default: `true`)*: Whether to compute DP-MAE pitch stylization.
  - `return_pitch` *(boolean, optional, default: `true`)*: Whether to return frame-by-frame pitch contours.

#### Response (`200 OK`)
```json
{
  "transcription": "The quick brown fox jumps over the lazy dog.",
  "utterance_median_hz": 182.4,
  "words": [
    {
      "word": "The",
      "start": 0.12,
      "end": 0.32,
      "confidence": 0.98,
      "whistress_stressed": false
    },
    {
      "word": "quick",
      "start": 0.34,
      "end": 0.65,
      "confidence": 0.95,
      "whistress_stressed": true
    }
  ],
  "phrases": [
    {
      "text": "The quick brown fox jumps over the lazy dog.",
      "start": 0.12,
      "end": 2.45,
      "intonation": "Falling",
      "slope": -2.31
    }
  ],
  "pitch_contour": [
    { "time": 0.12, "semitone": 0.15 },
    { "time": 0.15, "semitone": 0.28 }
  ],
  "pitch_stylized": [
    { "time": 0.12, "semitone": 0.18, "is_knot": true },
    { "time": 0.65, "semitone": 1.45, "is_knot": true }
  ]
}
```

---

### `POST /api/annotate`
Generates comprehensive multi-layer phonetic annotations combining sentence prominence (WhiStress), syllable-level lexical stress (LexiRep), pause detection, and pitch stylization.

#### Request
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` *(binary audio file, required)*: Uploaded audio.
  - `speaker_nativity` *(string, optional, values: `"GER"`, `"ITA"`, default: `null`)*: Selects pre-trained LexiRep acoustic metric learning bundle.

#### Response (`200 OK`)
```json
{
  "metadata": {
    "audio_duration_seconds": 2.45,
    "sample_rate_hz": 16000,
    "utterance_median_pitch_hz": 182.4,
    "speaker_nativity": "GER",
    "lexirep_model_active": true
  },
  "transcription": "Communication is essential.",
  "words": [
    {
      "word": "Communication",
      "start": 0.24,
      "end": 1.15,
      "whistress_stressed": true,
      "is_polysyllabic": true,
      "syllable_count": 5,
      "syllable_source": "ISLE_TRAIN_STRICT",
      "lexirep_stressed_syllable_index": 3,
      "syllables": [
        { "index": 0, "text": "com", "start": 0.24, "end": 0.42, "stress": 0, "contrast": -0.42, "confidence": 0.08 },
        { "index": 1, "text": "mu", "start": 0.42, "end": 0.60, "stress": 0, "contrast": -0.31, "confidence": 0.11 },
        { "index": 2, "text": "ni", "start": 0.60, "end": 0.78, "stress": 0, "contrast": -0.15, "confidence": 0.14 },
        { "index": 3, "text": "ca", "start": 0.78, "end": 0.96, "stress": 1, "contrast": 0.68, "confidence": 0.58 },
        { "index": 4, "text": "tion", "start": 0.96, "end": 1.15, "stress": 0, "contrast": -0.35, "confidence": 0.09 }
      ]
    }
  ],
  "pauses": [
    {
      "pause_id": 1,
      "start": 1.15,
      "end": 1.48,
      "duration": 0.33,
      "preceding_word": "Communication",
      "following_word": "is"
    }
  ],
  "intonation": [
    {
      "phrase_id": 1,
      "text": "Communication is essential.",
      "start": 0.24,
      "end": 2.10,
      "nuclear_direction": "Falling",
      "nuclear_semitone_delta": -3.12
    }
  ],
  "timeline": [
    { "type": "word", "time": 0.24, "label": "Communication", "stressed": true },
    { "type": "pause", "time": 1.15, "duration": 0.33 },
    { "type": "word", "time": 1.48, "label": "is", "stressed": false }
  ]
}
```

---

### `POST /api/lexirep/analyze`
Direct diagnostic endpoint for isolated syllable-level LexiRep inference.

#### Request
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` *(binary audio file, required)*: Audio file.
  - `speaker_nativity` *(string, required, values: `"GER"`, `"ITA"`)*: Language model bundle.

#### Response (`200 OK`)
Returns word-by-word syllable segmentation with Wav2Vec 2.0 extraction timing and LexiRep projection diagnostics.

---

## 3. Asynchronous Batch Annotation API

### `POST /api/batch/create`
Submits multiple audio files for asynchronous background annotation processing.

#### Request
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `files` *(multiple binary audio files, required)*
  - `speaker_nativity` *(string, optional, values: `"GER"`, `"ITA"`, default: `null`)*

#### Response (`202 Accepted`)
```json
{
  "batch_id": "b9821812-9c12-4eb5-bcf4-329a1b02194c",
  "status": "pending",
  "total_files": 25,
  "created_at": "2026-11-04T13:20:00Z"
}
```

---

### `GET /api/batch/status/{batch_id}`
Polls the processing progress of an ongoing batch job.

#### Response (`200 OK`)
```json
{
  "batch_id": "b9821812-9c12-4eb5-bcf4-329a1b02194c",
  "status": "processing",
  "progress_percent": 68.0,
  "completed_files": 17,
  "total_files": 25,
  "failed_files": 0
}
```

---

### `GET /api/batch/download/{batch_id}`
Downloads the final structured ZIP archive containing individual JSON annotations and `manifest.json`.

#### Response (`200 OK`)
- **Content-Type:** `application/zip`
- **Content-Disposition:** `attachment; filename="prosodylab_batch_{batch_id}.zip"`

---

### `POST /api/batch/cancel/{batch_id}`
Requests cooperative termination of an ongoing batch execution.

#### Response (`200 OK`)
```json
{
  "batch_id": "b9821812-9c12-4eb5-bcf4-329a1b02194c",
  "status": "cancelled"
}
```

---

## 4. Training Workbench API

### `POST /api/train/validate`
Validates uploaded training audio files or an archive for sample rate, duration, and channel format.

#### Request (`multipart/form-data`)
- `file`: ZIP archive or audio files.

#### Response (`200 OK`)
```json
{
  "valid": true,
  "file_count": 120,
  "total_duration_minutes": 14.2,
  "issues": []
}
```

---

### `POST /api/train/scan-anchors`
Scans candidate dataset words to discover defensible acoustic anchor pairs maximizing unsupervised train-set separation ($S_{\text{train}}$).

#### Request (`application/json`)
```json
{
  "language": "GER",
  "candidate_count": 15
}
```

#### Response (`200 OK`)
```json
{
  "optimal_anchor": {
    "word_id": 1181,
    "stressed_syllable_index": 0,
    "unstressed_syllable_index": 1,
    "separation_score": 68.22
  },
  "candidates": [
    { "word_id": 1181, "separation_score": 68.22 },
    { "word_id": 2611, "separation_score": 67.64 },
    { "word_id": 2038, "separation_score": 66.58 }
  ]
}
```

---

### `POST /api/train/start`
Launches an asynchronous self-training job with specified hyperparameters.

#### Request (`application/json`)
```json
{
  "language": "GER",
  "warmup_epochs": 15,
  "idec_pretrain_epochs": 10,
  "outer_loops": 5,
  "batch_words": 32,
  "top_k_rep": 1,
  "seed": 42
}
```

#### Response (`202 Accepted`)
```json
{
  "job_id": "train_8731adbc",
  "status": "running",
  "message": "LexiRep self-training job initialized."
}
```

---

### `GET /api/train/status/{job_id}`
Returns live epoch, loss, and linguistic compliance statistics for a running job.

#### Response (`200 OK`)
```json
{
  "job_id": "train_8731adbc",
  "status": "running",
  "epoch": 8,
  "total_epochs": 15,
  "loss": 0.284,
  "compliance_rate": 1.0,
  "progress_percent": 53.3
}
```

---

### `GET /api/train/logs/{job_id}`
Streams or fetches raw standard output execution logs from the training worker.

---

### `POST /api/train/export/{job_id}`
Packages the trained PyTorch model weights, prototype vectors, and training history into a deployable model bundle.

#### Response (`200 OK`)
- **Content-Type:** `application/zip`
- **Archive Contents:**
  - `config.json`
  - `anchors.json`
  - `metrics.json`
  - `training_history.csv`
  - `model/final_lexirep_model.pt`
  - `anchors/prototypes_10d.npy`

---

### `POST /api/train/cancel/{job_id}`
Gracefully terminates an ongoing self-training job.

---

## 5. Error Codes & Handling

| HTTP Code | Exception Name | Description |
| :--- | :--- | :--- |
| **`400 Bad Request`** | `InvalidAudioFormat` | Unreadable, corrupt, or zero-byte audio payload. |
| **`404 Not Found`** | `JobNotFound` | Requested `batch_id` or `job_id` does not exist in job registry. |
| **`422 Unprocessable`**| `ValidationError` | Missing required form field or unsupported `speaker_nativity` option. |
| **`500 Internal Error`**| `PipelineError` | Unhandled model forward failure; includes stack diagnostics. |

