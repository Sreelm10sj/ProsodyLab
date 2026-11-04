/**
 * api.js — fetch wrappers for the ProsodyLab backend.
 * Matches the exact response shapes documented in docs/frontend-audit.md §5.1.
 * Base URL is proxied to http://127.0.0.1:8000 by vite.config.js.
 *
 * GET  /api/health   → { status, device, model_loaded }
 * POST /api/transcribe → { transcription, words, phrases?, pitch_contour?,
 *                          pitch_stylized?, utterance_median_hz? }
 * POST /api/annotate  → { metadata, transcription, words, pauses,
 *                          intonation, timeline }
 */

const BASE = '/api';

export async function getHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function postTranscribe(blob, filename) {
  const form = new FormData()
  // Field name MUST be 'file' — per docs/frontend-audit.md §5.1 and backend contract.
  form.append('file', blob, filename ?? blob.name ?? 'recording.webm')
  const res = await fetch(`${BASE}/transcribe`, { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    // FastAPI returns { detail } on validation errors; custom errors use { error }
    throw new Error(body.detail ?? body.error ?? `Transcription failed: ${res.status}`)
  }
  return res.json()
}

export async function postAnnotate(blob, filename, speakerNativity = null) {
  const form = new FormData()
  // Field name MUST be 'file' — per docs/frontend-audit.md §5.1 and backend contract.
  form.append('file', blob, filename ?? blob.name ?? 'recording.webm')
  if (speakerNativity) {
    form.append('speaker_nativity', speakerNativity)
  }
  const res = await fetch(`${BASE}/annotate`, { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? body.error ?? `Annotation failed: ${res.status}`)
  }
  return res.json()
}

/**
 * POST /api/lexirep/analyze — isolated syllable-level lexical stress endpoint.
 * Accepts: audio blob, nativity ("GER" | "ITA"), optional filename.
 * Returns: { status, nativity, transcription, audio, words, diagnostics }
 */
export async function postLexiRepAnalyze(blob, nativity = 'GER', filename) {
  const form = new FormData()
  form.append('file', blob, filename ?? blob.name ?? 'recording.wav')
  form.append('nativity', nativity)
  const res = await fetch(`${BASE}/lexirep/analyze`, { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg = typeof body.detail === 'string'
      ? body.detail
      : (Array.isArray(body.detail) && body.detail[0]?.msg)
        ? body.detail[0].msg
        : `LexiRep analysis failed: ${res.status}`
    throw new Error(msg)
  }
  return res.json()
}

// =============================================================================
// LEXIREP TRAINING WORKBENCH API
// =============================================================================

/**
 * POST /api/training/dataset/validate — uploads and validates .npz feature cache.
 */
export async function validateTrainingDataset(file) {
  const form = new FormData()
  form.append('file', file, file.name)
  const res = await fetch(`${BASE}/training/dataset/validate`, { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg = typeof body.detail === 'string' ? body.detail : `Dataset validation failed (${res.status})`
    throw new Error(msg)
  }
  return res.json()
}

/**
 * POST /api/training/anchors/search — searches candidate bisyllabic anchors.
 */
export async function searchTrainingAnchors(datasetId, maxCandidates = 20) {
  const res = await fetch(`${BASE}/training/anchors/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, max_candidates: maxCandidates }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg = typeof body.detail === 'string' ? body.detail : `Anchor search failed (${res.status})`
    throw new Error(msg)
  }
  return res.json()
}

/**
 * POST /api/training/jobs — creates and launches background training job.
 */
export async function createTrainingJob(payload) {
  const res = await fetch(`${BASE}/training/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg = typeof body.detail === 'string' ? body.detail : `Failed to create training job (${res.status})`
    const err = new Error(msg)
    err.status = res.status
    throw err
  }
  return res.json()
}

/**
 * GET /api/training/jobs — lists recent training jobs.
 */
export async function listTrainingJobs(limit = 20) {
  const res = await fetch(`${BASE}/training/jobs?limit=${limit}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Failed to list training jobs (${res.status})`)
  }
  return res.json()
}

/**
 * GET /api/training/jobs/{job_id} — polls state of a specific training job.
 */
export async function getTrainingJob(jobId) {
  const res = await fetch(`${BASE}/training/jobs/${jobId}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Failed to fetch job state (${res.status})`)
  }
  return res.json()
}

/**
 * POST /api/training/jobs/{job_id}/cancel — cancels an active training job.
 */
export async function cancelTrainingJob(jobId) {
  const res = await fetch(`${BASE}/training/jobs/${jobId}/cancel`, { method: 'POST' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Failed to cancel job (${res.status})`)
  }
  return res.json()
}

/**
 * GET /api/training/jobs/{job_id}/bundle — downloads production zip bundle.
 */
export async function downloadTrainingBundle(jobId) {
  const res = await fetch(`${BASE}/training/jobs/${jobId}/bundle`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Bundle download failed (${res.status})`)
  }
  const blob = await res.blob()
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `lexirep_model_bundle_${jobId.slice(0, 8)}.zip`
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

/**
 * GET /api/training/jobs/{job_id}/artifacts/{name} — downloads a specific whitelisted artifact.
 */
export async function downloadTrainingArtifact(jobId, name) {
  const res = await fetch(`${BASE}/training/jobs/${jobId}/artifacts/${encodeURIComponent(name)}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Artifact download failed (${res.status})`)
  }
  const blob = await res.blob()
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

// =============================================================================
// BATCH AUDIO ANNOTATION API
// =============================================================================

/**
 * POST /api/batch-annotate — upload ZIP archive containing multiple audio files.
 * Returns: { batch_id, status: 'queued', total_files }
 */
export async function createBatchAnnotationJob(file, speakerNativity = null) {
  const form = new FormData()
  form.append('file', file, file.name)
  if (speakerNativity) {
    form.append('speaker_nativity', speakerNativity)
  }
  const res = await fetch(`${BASE}/batch-annotate`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg = typeof body.detail === 'string' ? body.detail : `Batch upload failed (${res.status})`
    const err = new Error(msg)
    err.status = res.status
    throw err
  }
  return res.json()
}

/**
 * GET /api/batch-annotate/{batch_id} — poll state of a batch annotation job.
 */
export async function getBatchAnnotationJob(batchId) {
  const res = await fetch(`${BASE}/batch-annotate/${batchId}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Failed to fetch batch status (${res.status})`)
  }
  return res.json()
}

/**
 * POST /api/batch-annotate/{batch_id}/cancel — cancel an actively running batch job.
 */
export async function cancelBatchAnnotationJob(batchId) {
  const res = await fetch(`${BASE}/batch-annotate/${batchId}/cancel`, {
    method: 'POST',
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Failed to cancel batch (${res.status})`)
  }
  return res.json()
}

/**
 * GET /api/batch-annotate/{batch_id}/download — triggers download of result ZIP.
 */
export async function downloadBatchAnnotations(batchId) {
  const res = await fetch(`${BASE}/batch-annotate/${batchId}/download`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Download failed (${res.status})`)
  }
  const blob = await res.blob()
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `prosodylab_annotations_${batchId.slice(0, 8)}.zip`
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}


