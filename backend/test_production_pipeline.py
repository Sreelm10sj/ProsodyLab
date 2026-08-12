import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "WhiStress"))
from whistress_service import service

def main():
    service.load_model()
    
    with open("example3.wav", "rb") as f:
        audio_bytes = f.read()
        
    pairs, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)
    
    for idx, phrase in enumerate(phrases):
        print(f"\n--- Phrase {idx} ---")
        raw_slope = phrase["pitch_raw"]["overall_slope"]
        stylized_slope = phrase["pitch_stylized"]["overall_slope"]
        prod_slope = phrase["pitch"]["overall_slope"]
        prod_class = phrase["pitch"]["pattern"]
        
        print(f"Raw overall slope:       {raw_slope}")
        print(f"Stylized overall slope:  {stylized_slope}")
        print(f"Production slope:        {prod_slope}")
        print(f"Production class:        {prod_class}")
        
        # Verify requirement
        assert prod_slope == stylized_slope, "Production slope does not equal stylized slope!"
        
        # Verify classification matches slope logic
        if prod_slope is not None:
            expected_class = "rising" if prod_slope >= 1.5 else "falling" if prod_slope <= -1.5 else "level"
            assert prod_class == expected_class, f"Mismatch: expected {expected_class}, got {prod_class}"

if __name__ == "__main__":
    main()
