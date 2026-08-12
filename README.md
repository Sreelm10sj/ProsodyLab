# WhiStress Web App

A full-stack web application for transcribing speech and visually highlighting stressed words using the [WhiStress](https://github.com/slp-rl/WhiStress) model.

## Prerequisites

- Python 3.11
- Node.js & npm
- FFmpeg (installed and on your PATH)

## Setup and Running

### 1. Backend

The backend uses FastAPI and PyTorch to serve the WhiStress model.

```bash
cd backend
# Activate the Python 3.11 virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Start the FastAPI server
uvicorn main:app --reload
```
The server will start on `http://127.0.0.1:8000`. The model will be loaded on startup, which may take a few seconds.

### 2. Frontend

The frontend is built with React, Vite, and TailwindCSS v4.

```bash
cd frontend

# Start the development server
npm run dev
```
The app will be available at `http://localhost:5173`.

## Usage

1. **Upload Audio**: Click the "Upload Audio" button to select a `.wav`, `.mp3`, or `.m4a` file.
2. **Record Audio**: Click the "Record Audio" button to record directly from your microphone.
3. Once audio is selected, click **Transcribe** to send it to the backend.
4. The transcription will appear with stressed words highlighted!
