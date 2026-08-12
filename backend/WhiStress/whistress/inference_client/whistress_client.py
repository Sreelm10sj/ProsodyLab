import numpy as np
from .utils import get_loaded_model, scored_transcription
from typing import Union, Dict


class WhiStressInferenceClient:
    def __init__(self, device="cuda"):
        self.device = device
        self.whistress = get_loaded_model(self.device)

    def predict(
        self, audio: Dict[str, Union[np.ndarray, int]], transcription=None, return_pairs=True
    ):
        word_emphasis_pairs = scored_transcription(
            audio=audio, 
            model=self.whistress, 
            device=self.device, 
            strip_words=True, 
            transcription=transcription
        )
        if return_pairs:
            return word_emphasis_pairs
        # returs transcription str and list of emphasized words
        return " ".join([x[0] for x in word_emphasis_pairs]), [
            x[0] for x in word_emphasis_pairs if x[1] == 1
        ]
