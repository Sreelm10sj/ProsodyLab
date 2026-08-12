import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "WhiStress"))
from main import _build_pitch_contour
from whistress_service import service

def main():
    service.load_model()
    
    with open("example3.wav", "rb") as f:
        audio_bytes = f.read()
        
    pairs, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)
    
    import numpy as np
    voiced = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    utterance_median_hz = float(np.median(voiced)) if len(voiced) > 0 else None

    pitch_stylized_contour = _build_pitch_contour(pitch_times, pitch_stylized, utterance_median_hz, k_array=k_array, P_val=P_val)
    
    print(json.dumps(pitch_stylized_contour[:20], indent=2))
    
    voiced_pts = [p for p in pitch_stylized_contour if p["semitone"] is not None]
    print(f"\nFound {len(voiced_pts)} voiced points.")
    if voiced_pts:
        print("First voiced point:")
        print(json.dumps(voiced_pts[0], indent=2))

if __name__ == "__main__":
    main()
