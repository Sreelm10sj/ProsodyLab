import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from whistress_service import service
from annotation.pipeline import run_annotation_pipeline

def main():
    test_file = "example3.wav"
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found.")
        return

    print("Loading models...")
    service.load_model()
    
    print(f"\nProcessing {test_file} via predict_with_intonation()...")
    with open(test_file, "rb") as f:
        audio_bytes = f.read()
        
    pairs, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)
    
    print(f"\nProcessing {test_file} via annotation pipeline...")
    annotation = run_annotation_pipeline(audio_bytes, test_file)
    
    print("\n--- Numerical Validation ---")
    
    if not phrases:
        print("No phrases found to validate.")
        return
        
    p_idx = 0
    test_phrase = phrases[p_idx]
    
    raw_slope = test_phrase["pitch_raw"]["overall_slope"]
    stylized_slope = test_phrase["pitch_stylized"]["overall_slope"]
    
    anno_phrase = annotation["intonation"][p_idx]
    saved_slope = anno_phrase["slope"]
    saved_label = anno_phrase["label"]
    
    print(f"Raw overall slope:       {raw_slope}")
    print(f"Stylized overall slope:  {stylized_slope}")
    print(f"Saved intonation slope:  {saved_slope}")
    
    assert saved_slope == stylized_slope, "ERROR: Saved slope does NOT match stylized slope!"
    
    # Calculate expected label based on saved slope
    if saved_slope is not None:
        expected = "rising" if saved_slope >= 1.5 else "falling" if saved_slope <= -1.5 else "level"
        assert saved_label == expected, f"ERROR: Saved label '{saved_label}' does not match expected '{expected}'"
        print(f"Saved label:             {saved_label} (Matches expected '{expected}')")
    
    print("\nSUCCESS: The annotation pipeline uses the production stylized-pitch intonation system.")

if __name__ == "__main__":
    main()
