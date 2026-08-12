import os
import sys
import json
import argparse
import time

# Ensure we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whistress_service import service
from pitch_analysis import compute_pause_before

def run_annotation_pipeline(audio_bytes: bytes, filename: str) -> dict:
    """
    Orchestrates the existing ProsodyLab models into a structured JSON annotation.
    """
    t_start = time.time()
    
    from pitch_analysis import PAUSE_ANNOTATION_THRESHOLD_S

    # Run the core ML models and extraction
    pairs, pauses, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)
    
    # Calculate duration
    duration = len(audio_array) / sr if sr else 0.0
    
    words_out = []
    pauses_out = []
    timeline_out = []
    
    transcription_text = " ".join(w for w, *_ in pairs)
    
    for idx, (word, stress, start, end) in enumerate(pairs):
        start = float(start)
        end = float(end)
        
        # 1. Check for pause before this word
        pause_before = pauses[idx - 1] if idx > 0 else None
        
        if pause_before is not None and pause_before["duration"] > PAUSE_ANNOTATION_THRESHOLD_S:
            pause_start = pause_before["start"]
            pause_end = pause_before["end"]
            pause_duration = pause_before["duration"]
            
            pauses_out.append({
                "start": pause_start,
                "end": pause_end,
                "duration": pause_duration
            })
            
            # Add to timeline
            timeline_out.append({
                "type": "pause",
                "start": pause_start,
                "end": pause_end,
                "duration": pause_duration
            })
            
        # 2. Add the word itself
        words_out.append({
            "id": idx + 1,
            "text": word,
            "start": start,
            "end": end,
            "stress": "stressed" if stress else "unstressed"
        })
        
        # Add to timeline
        timeline_out.append({
            "type": "word",
            "id": idx + 1,
            "start": start,
            "end": end
        })
        
    # Sort timeline by start time to ensure chronological ordering
    timeline_out.sort(key=lambda x: x["start"])
    
    # Build Intonation
    intonation_out = []
    for p_idx, phrase in enumerate(phrases):
        # Handle cases where voicing was insufficient
        label = phrase["pitch"]["pattern"]
        slope = phrase["pitch"]["overall_slope"]
        
        intonation_out.append({
            "phrase_id": p_idx + 1,
            "start": phrase["start"],
            "end": phrase["end"],
            "slope": slope,
            "label": label
        })
        
    # Sort pauses and intonation by start time as well
    pauses_out.sort(key=lambda x: x["start"])
    intonation_out.sort(key=lambda x: x["start"])
    
    # Compile final JSON
    annotation = {
        "metadata": {
            "audio_file": os.path.basename(filename),
            "duration": round(duration, 3)
        },
        "transcription": {
            "text": transcription_text,
            "language": "en"
        },
        "words": words_out,
        "pauses": pauses_out,
        "intonation": intonation_out,
        "timeline": timeline_out
    }
    
    return annotation

def main():
    parser = argparse.ArgumentParser(description="ProsodyLab Automated Annotation Pipeline")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: Input file '{args.audio_path}' not found.")
        sys.exit(1)
        
    os.makedirs(args.out_dir, exist_ok=True)
    filename = os.path.basename(args.audio_path)
    base_name, _ = os.path.splitext(filename)
    
    json_path = os.path.join(args.out_dir, f"{base_name}_annotation.json")
    txt_path = os.path.join(args.out_dir, f"{base_name}_transcript.txt")
    
    print(f"Loading models...")
    service.load_model()
    
    with open(args.audio_path, "rb") as f:
        audio_bytes = f.read()
        
    print(f"Processing '{args.audio_path}'...")
    try:
        annotation = run_annotation_pipeline(audio_bytes, filename)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Annotation failed: {e}")
        sys.exit(1)
        
    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=4)
        
    # Write TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(annotation["transcription"]["text"])
        
    print("\nAnnotation completed successfully.")
    print(f"Audio: {filename}")
    print(f"Duration: {annotation['metadata']['duration']} s")
    print(f"Words: {len(annotation['words'])}")
    print(f"Pauses: {len(annotation['pauses'])}")
    print(f"Intonation phrases: {len(annotation['intonation'])}")
    print(f"Output saved to {args.out_dir}")

if __name__ == "__main__":
    main()
