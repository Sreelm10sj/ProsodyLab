import os
import json
import math

def validate_annotation_json(json_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("I. JSON valid and loaded successfully.")
    
    words = data.get("words", [])
    pauses = data.get("pauses", [])
    intonation = data.get("intonation", [])
    timeline = data.get("timeline", [])
    
    print(f"A/B. Phrases: {len(intonation)}, Pauses: {len(pauses)}")
    
    errors = 0
    
    # E. Verify all word timestamps start < end
    for w in words:
        if not (w["start"] < w["end"]):
            print(f"ERROR: Word {w['id']} has invalid timestamps: start={w['start']}, end={w['end']}")
            errors += 1
            
    # F. Verify all pause timestamps start <= end
    for p in pauses:
        if not (p["start"] <= p["end"]):
            print(f"ERROR: Pause has invalid timestamps: start={p['start']}, end={p['end']}")
            errors += 1
            
    # G. Verify all intonation phrase timestamps start < end
    for p in intonation:
        if not (p["start"] < p["end"]):
            print(f"ERROR: Phrase {p['phrase_id']} has invalid timestamps: start={p['start']}, end={p['end']}")
            errors += 1
            
    # C & D checks
    # Pauses should end exactly at the start of the following word
    word_starts = {w["start"]: w for w in words}
    for p in pauses:
        # C. verify pause.end is a word start (or close to it due to rounding)
        # In our implementation we set pause.end = word.start, so it should be exact.
        if p["end"] not in word_starts:
            # Let's check floating point issues
            found = any(math.isclose(p["end"], ws, abs_tol=1e-3) for ws in word_starts)
            if not found:
                print(f"ERROR: Pause ending at {p['end']} does not match any word start.")
                errors += 1
                
        # D. verify pause.duration == pause.end - pause.start
        expected_dur = p["end"] - p["start"]
        if not math.isclose(p["duration"], expected_dur, abs_tol=1e-3):
            print(f"ERROR: Pause duration {p['duration']} does not match end - start ({expected_dur})")
            errors += 1
            
    # H. Verify timeline is sorted chronologically
    for i in range(len(timeline) - 1):
        if timeline[i]["start"] > timeline[i+1]["start"]:
            print(f"ERROR: Timeline is not sorted chronologically at index {i} and {i+1}")
            errors += 1
            
    if errors == 0:
        print("All validation checks passed successfully!")
    else:
        print(f"Finished with {errors} errors.")

if __name__ == "__main__":
    validate_annotation_json("output/example3_annotation.json")
