import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiStress"))
from whistress_service import service
from annotation.pipeline import run_annotation_pipeline

def test_annotation(filepath):
    print(f"Running full annotation on {filepath}...")
    with open(filepath, "rb") as f:
        audio_bytes = f.read()
    
    # Run the pipeline
    annotation_dict = run_annotation_pipeline(audio_bytes, filepath)
    
    # Save the output to JSON
    out_name = os.path.basename(filepath) + "_annotation.json"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation_dict, f, indent=2)
    
    print(f"Saved annotation to {out_name}")

if __name__ == "__main__":
    service.load_model()
    test_annotation("example1_football.wav")
    test_annotation("example4.wav")
    test_annotation("test_two_phrases.wav")
