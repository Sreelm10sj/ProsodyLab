"""
backend.lexirep.feature_extractor

Full-utterance Wav2Vec 2.0 feature extraction for LexiRep.
Extracts Layer-12 contextual representations from raw audio.

Design constraints:
- Input audio: 16 kHz mono float32.
- Utterance scope: full continuous audio passed in a single forward pass.
  No independent syllable audio slicing before Wav2Vec2.
- Output: (T, 768) float32 array of last_hidden_state representations.
- Inference only: eval mode, torch.no_grad().
"""

from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class Wav2VecFeatureExtractor:
    """
    Extracts full-utterance Layer-12 contextual hidden states from 16 kHz audio
    using facebook/wav2vec2-base.
    """

    EXPECTED_SAMPLE_RATE: int = 16000
    HIDDEN_DIMENSION: int = 768

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: Optional[str] = None
    ):
        """
        Initialize the feature extractor.
        Model loading is deferred until load() is called or first extraction.
        """
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2Model] = None

    def load(self) -> None:
        """
        Loads the Wav2Vec2 processor and model onto the configured device in eval mode.
        """
        if self.model is not None and self.processor is not None:
            return

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        try:
            self.model = Wav2Vec2Model.from_pretrained(
                self.model_name, use_safetensors=True
            ).to(self.device)
        except Exception:
            self.model = Wav2Vec2Model.from_pretrained(
                self.model_name
            ).to(self.device)

        self.model.eval()

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts full-utterance last_hidden_state representations for the continuous audio.

        Parameters
        ----------
        audio : np.ndarray
            1D float32 numpy array representing mono audio at 16 kHz.
            If 2D (stereo), multi-channel audio is averaged across channels.
        sample_rate : int
            Sampling rate of the audio. Must be exactly 16000 Hz.

        Returns
        -------
        np.ndarray
            Contextual frame hidden states of shape (T, 768), dtype float32.
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Expected audio as np.ndarray, got {type(audio).__name__}")

        if sample_rate != self.EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Expected sample rate {self.EXPECTED_SAMPLE_RATE} Hz, but got {sample_rate} Hz"
            )

        if audio.size == 0:
            raise ValueError("Audio buffer is empty")

        # Handle stereo or multi-channel audio by averaging to mono
        if audio.ndim > 1:
            if audio.shape[0] <= 4 and audio.shape[1] > audio.shape[0]:
                audio = audio.mean(axis=0)
            else:
                audio = audio.mean(axis=-1)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio buffer contains non-finite values (NaN or Inf)")

        # Ensure model and processor are loaded
        if self.model is None or self.processor is None:
            self.load()

        # Utterance volume normalization is applied by the processor
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        # Extract final transformer layer (Layer 12): [1, T, 768] -> [T, 768]
        last_hidden_state = outputs.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)

        if last_hidden_state.ndim != 2:
            raise ValueError(
                f"Expected 2D hidden states [T, 768], got shape {last_hidden_state.shape}"
            )

        if last_hidden_state.shape[1] != self.HIDDEN_DIMENSION:
            raise ValueError(
                f"Expected hidden dimension {self.HIDDEN_DIMENSION}, got {last_hidden_state.shape[1]}"
            )

        return last_hidden_state

    def extract_from_file(self, wav_path: str) -> Tuple[np.ndarray, float]:
        """
        Convenience helper to read a WAV file and extract its full-utterance hidden states.

        Returns
        -------
        Tuple[np.ndarray, float]
            (hidden_states [T, 768], audio_duration_seconds)
        """
        audio, sr = sf.read(wav_path, dtype="float32")
        hidden = self.extract(audio, sample_rate=sr)
        duration = len(audio) / float(sr)
        return hidden, duration
