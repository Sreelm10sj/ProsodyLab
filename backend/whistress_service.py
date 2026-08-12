import os
import sys
import torch
import io
import traceback

# Add the cloned repo to sys.path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "WhiStress"))

try:
    from whistress import WhiStressInferenceClient
except ImportError:
    print("Warning: whistress not found. Make sure it is cloned into backend/WhiStress")
    WhiStressInferenceClient = None

class WhiStressService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client = None

    def load_model(self):
        if self.client is None and WhiStressInferenceClient is not None:
            print(f"Loading WhiStress model on {self.device}...")
            self.client = WhiStressInferenceClient(device=self.device)
            print("WhiStress model loaded successfully.")
        elif WhiStressInferenceClient is None:
            print("WhiStressInferenceClient is not available. Check installation.")

    def process_audio(self, audio_bytes: bytes):
        """
        Process raw audio bytes into the format expected by WhiStress.
        (mono, 16kHz, float32 numpy array)
        """
        from pydub import AudioSegment
        import soundfile as sf
        
        # Load audio using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Convert to mono, 16kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Export as wav float32
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Read back as numpy float32
        array, sr = sf.read(wav_io, dtype='float32')
        return {"sampling_rate": sr, "array": array}

    def predict(self, audio_bytes: bytes):
        if not self.client:
            raise RuntimeError("Model is not loaded")
            
        audio_dict = self.process_audio(audio_bytes)
        
        # Predict
        result = self.client.predict(
            audio=audio_dict,
            transcription=None,
            return_pairs=True
        )
        return result

    def predict_with_intonation(self, audio_bytes: bytes):
        """
        Full prediction pipeline including phrase-level intonation analysis.

        Processes audio bytes exactly once and passes the result to both
        the WhiStress inference client (stress + word timestamps) and the
        pitch analysis stage (F0 extraction + phrase segmentation).

        The original predict() method is left completely unchanged so that
        any existing callers continue to work without modification.

        Returns
        -------
        words : list of (word, stress, start, end) tuples
            Word-level output from WhiStress — identical to predict().
        phrases : list of dict
            Phrase-level intonation output from pitch_analysis.build_phrase_response().
        audio_array : np.ndarray
            Decoded acoustic signal.
        sr : int
            Sample rate of the acoustic signal.
        """
        if not self.client:
            raise RuntimeError("Model is not loaded")

        from pitch_analysis import extract_pitch, build_phrase_response
        import time
        import numpy as np

        STYLIZATION_MODE = "fast"
        # STYLIZATION_MODE = "reference"

        if STYLIZATION_MODE == "fast":
            from pitch_stylization_fast import stylize_pitch_contour
        else:
            from pitch_stylization_reference import stylize_pitch_contour

        # Decode and resample audio once; reuse for both pipelines.
        audio_dict = self.process_audio(audio_bytes)
        audio_array = audio_dict["array"]
        sr = audio_dict["sampling_rate"]

        # --- Existing WhiStress inference (unchanged) ---
        words = self.client.predict(
            audio=audio_dict,
            transcription=None,
            return_pairs=True,
        )

        # --- Independent pitch extraction stage ---
        # Does not touch Whisper, the stress classifier, or word timestamps.
        pitch_times, pitch_f0 = extract_pitch(audio_array, sr=sr)
        
        # --- DP-MAE Stylization stage ---
        t0 = time.time()
        pitch_stylized, k_array, P_val = stylize_pitch_contour(pitch_f0)
        t1 = time.time()
        
        valid_pitch = pitch_stylized[~np.isnan(pitch_stylized)]
        num_segments = np.sum(np.diff(valid_pitch) != 0) + 1 if len(valid_pitch) > 0 else 0
        
        print(f"Stylization Mode : {STYLIZATION_MODE.capitalize()}")
        print(f"Stylization Runtime : {t1 - t0:.2f} s")
        print(f"Segments : {num_segments}")

        # --- NEW: Word Timestamp Realignment ---
        from pitch_analysis import realign_word_timestamps
        corrected_words, pauses = realign_word_timestamps(words, pitch_times, pitch_f0, audio_array, sr)

        # --- Phrase segmentation and intonation analysis ---
        # Passes the corrected words, but phrase segmenter uses the original Whisper word-boundary semantics
        phrases = build_phrase_response(corrected_words, pitch_times, pitch_f0, pitch_stylized, audio_array, sr, pauses)

        return corrected_words, pauses, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr

service = WhiStressService()
