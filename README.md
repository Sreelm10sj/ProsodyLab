# ProsodyLab

A full-stack web application for speech prosody analysis. Built on the [WhiStress](https://github.com/slp-rl/WhiStress) model (Yosha, Shteyman & Adi, Interspeech 2025), ProsodyLab extends word-level stress detection with phrase-level intonation analysis, acoustic pause detection, pitch contour visualization, and a structured annotation pipeline.

## Features

- **Stress detection** — word-level stressed/unstressed labels from the WhiStress model
- **Acoustic pause detection** — inter-word pauses measured directly from the pitch contour
- **Phrase-level intonation** — rising / falling / level classification per intonation phrase
- **Pitch contour visualization** — raw F0 and DP-MAE stylized contour displayed together
- **Annotation pipeline** — exports a structured JSON annotation (transcription, words, pauses, intonation, timeline)
- **Audio input** — file upload (WAV / MP3 / M4A) or in-browser microphone recording

## Prerequisites

- Python 3.11
- Node.js & npm
- FFmpeg (installed and on your PATH)

## Setup

### 1. Install backend dependencies

```bash
cd backend
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
pip install -r WhiStress/requirements.txt
```

### 2. Download model weights

The WhiStress model weights are distributed via Hugging Face and are **not included in this repository**.

```bash
cd backend
python WhiStress/download_weights.py
```

This downloads `additional_decoder_block.pt` and `classifier.pt` into `backend/WhiStress/whistress/weights/`.

### 3. Start the backend

```bash
cd backend
# Activate the virtual environment (if not already active)
uvicorn main:app --reload
```

The API server starts at `http://127.0.0.1:8000`. WhiStress model loading happens on startup (allow a few seconds on first run).

### 4. Install and start the frontend

```bash
cd frontend
npm install
npm run dev
```

The app is available at `http://localhost:5173`.

## Usage

1. **Upload Audio** — select a `.wav`, `.mp3`, or `.m4a` file, or use **Record Audio** to capture directly from your microphone.
2. Click **Transcribe** to run the full prosody analysis pipeline.
3. Results are displayed across tabs: transcript with stress highlights, phrase intonation labels, pitch contour chart, and structured annotation export.

## Project Structure

```
backend/
├── main.py                      # FastAPI application and API endpoints
├── whistress_service.py         # WhiStress model service wrapper
├── pitch_analysis.py            # Pitch extraction, pause detection, intonation analysis
├── pitch_stylization_fast.py    # Optimized hybrid DP-MAE pitch stylization (production)
├── pitch_stylization_reference.py  # Reference DP-MAE implementation (reproducibility)
├── annotation/pipeline.py       # Structured annotation pipeline
├── WhiStress/                   # Upstream WhiStress model library
├── FINAL_REPORT.md              # Technical report: pitch stylization optimization
├── OPTIMIZED_DP_MAE.md         # Deep-dive on the hybrid L2/L1 algorithm
└── stylization_validation/      # Benchmark results and evaluation reports

frontend/
├── src/
│   ├── App.jsx                  # Root application component
│   ├── api.js                   # Backend API client
│   └── components/
│       ├── AudioInput.jsx        # File upload with drag-and-drop
│       ├── Recorder.jsx          # In-browser microphone recording
│       ├── TranscriptView.jsx    # Transcript with stress highlighting
│       ├── ProsodyPanel.jsx      # Main results panel
│       ├── PhraseIntonation.jsx  # Phrase-level intonation labels
│       ├── PitchContour.jsx      # SVG pitch contour visualization
│       └── AnnotationViewer.jsx  # Annotation export and timeline view
└── index.html
```

## Model Weights

Model weights are hosted on Hugging Face: [slprl/WhiStress](https://huggingface.co/slprl/WhiStress).
Run `python WhiStress/download_weights.py` from the `backend/` directory to fetch them.

## Citation

If you use the WhiStress model, please cite the original paper:

```bibtex
@misc{yosha2025whistress,
    title={WHISTRESS: Enriching Transcriptions with Sentence Stress Detection},
    author={Iddo Yosha and Dorin Shteyman and Yossi Adi},
    year={2025},
    eprint={2505.19103},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.19103},
}
```
