import os
import sys

# Mock WhiStressInferenceClient to avoid loading actual heavy models just for a syntax check
class MockWhiStressInferenceClient:
    def __init__(self, device):
        self.device = device
    def predict(self, audio, transcription, return_pairs):
        return [("hello", 0, 0.0, 0.5), ("world", 1, 0.5, 1.0)]

import whistress_service
whistress_service.WhiStressInferenceClient = MockWhiStressInferenceClient
service = whistress_service.WhiStressService()
service.load_model()

with open("example3.wav", "rb") as f:
    audio_bytes = f.read()

try:
    print("Testing predict_with_intonation...")
    res = service.predict_with_intonation(audio_bytes)
    print(f"Success! Returned {len(res)} elements.")
    print(f"Stylized contour length: {len(res[4])}")
    print(f"Raw contour length: {len(res[3])}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
