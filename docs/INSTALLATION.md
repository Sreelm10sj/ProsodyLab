# ProsodyLab — Cross-Platform Installation & Setup Guide

This guide provides end-to-end instructions for installing and running ProsodyLab on **Linux**, **macOS**, and **Windows**.

---

## 1. System Requirements

### Hardware
- **CPU**: Modern quad-core processor (x86_64 or ARM64 / Apple Silicon).
- **RAM**: Minimum 8 GB (16 GB recommended for batch training).
- **GPU (Recommended)**: NVIDIA GPU with $\ge 6\text{ GB}$ VRAM supporting CUDA 11.8 or 12.x. *(CPU inference is fully supported).*
- **Disk Space**: $\approx 3.5\text{ GB}$ free space for Python dependencies, Node packages, and pre-trained HuggingFace models.

### Software Prerequisites
- **Python**: Version `3.10` or `3.11`.
- **Node.js**: Version `18.x` or `20.x` (LTS) with `npm` $\ge 9.0$.
- **Git**: Installed and accessible in system path.
- **FFmpeg**: Required by Whisper and Librosa for audio stream decoding.
  - *Ubuntu/Debian:* `sudo apt-get install ffmpeg`
  - *macOS:* `brew install ffmpeg`
  - *Windows:* `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org).

---

## 2. Repository Setup

Clone the repository to your local machine:
```bash
git clone https://github.com/Sreelm10sj/ProsodyLab.git
cd ProsodyLab
```

---

## 3. Backend Setup (FastAPI & PyTorch)

### Step 3.1: Create and Activate Virtual Environment

#### On Linux / macOS:
```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
```

#### On Windows (PowerShell):
```powershell
python -m venv backend\venv
.\backend\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### Step 3.2: Install PyTorch with GPU Support (Optional)
If you have an NVIDIA GPU, install PyTorch with CUDA support first:
```bash
# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only systems:
pip install torch torchaudio
```

### Step 3.3: Install Backend Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
```

### Step 3.4: Download Pre-Trained WhiStress Weights
The WhiStress prominence classifier weights must be placed in `backend/WhiStress/whistress/weights/`:
```bash
python backend/WhiStress/download_weights.py
```
Alternatively, download `additional_decoder_block.pt` and `classifier.pt` manually from the official HuggingFace repository:
[https://huggingface.co/slprl/WhiStress/tree/main](https://huggingface.co/slprl/WhiStress/tree/main) and place them inside `backend/WhiStress/whistress/weights/`.

*Note*: The underlying foundation models (`openai/whisper-small.en` and `facebook/wav2vec2-base-960h`) will automatically download and cache to `~/.cache/huggingface/` upon first inference.

---

## 4. Frontend Setup (React 19 & Vite)

Navigate into the `frontend/` directory and install JavaScript dependencies:

```bash
cd frontend
npm install
```

To verify the production build generates without syntax or bundling errors:
```bash
npm run build
```
*(Build artifacts will be written to `frontend/dist/` in under 1 second).*

---

## 5. Verification & First-Run Test Execution

Before launching the servers, verify both backend and frontend test suites pass:

### 5.1 Run Backend Unit & Integration Tests (144 Tests)
Ensure your virtual environment is active:
```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```
*Expected Result:* All 14 test modules should pass with `OK` (0 failures, 0 errors).

### 5.2 Run Frontend Unit & Component Tests (72 Tests)
From the project root:
```bash
npm --prefix frontend test -- --run
```
*Expected Result:* All 8 test suites should pass with `72 passed`.

---

## 6. Running the Application

### 6.1 Start the Backend API Server
In terminal 1 (with virtual environment active):
```bash
# From the repository root:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify the API is operational by checking the health endpoint:
```bash
curl http://localhost:8000/api/health
```
Response should return:
```json
{"status":"healthy","device":"cuda","model_loaded":true,"lexirep_models":{"GER":true,"ITA":true}}
```

### 6.2 Start the Frontend Development Server
In terminal 2:
```bash
cd frontend
npm run dev
```
The Vite development server will start at `http://localhost:5173` (or `5174` if port 5173 is occupied).

Open your browser and navigate to:
[http://localhost:5173](http://localhost:5173)

---

## 7. Common Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ImportError: cannot import name 'audioop'` | Python 3.13 deprecation | Use Python `3.10` or `3.11`. |
| `RuntimeError: WhiStress weights missing` | Weights not downloaded | Run `python backend/WhiStress/download_weights.py`. |
| `Microphone access denied` | Browser permissions | Allow microphone access in browser address bar settings. |
| `CORS Error in Browser Console` | Port mismatch | Ensure backend runs on port 8000 and frontend runs on 5173/5174. |

