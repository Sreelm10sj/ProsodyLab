import os
import re

def parse_sbcsae_trn(filepath):
    """
    Parses a SBCSAE .trn file into a list of Intonation Unit dictionaries.
    Handles continuing speakers and extracts terminal contours.
    """
    ius = []
    current_speaker = "UNKNOWN"
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line_idx, line in enumerate(f):
            line = line.strip('\n')
            if not line:
                continue
                
            parts = line.split('\t')
            
            # Usually format is: "start end \t SPEAKER: \t text"
            # Or: "start end \t \t text" (continuing speaker)
            if len(parts) >= 2:
                time_str = parts[0].strip()
                time_tokens = time_str.split(' ')
                
                # Check if it actually starts with timestamps
                if len(time_tokens) >= 2:
                    try:
                        start = float(time_tokens[0])
                        end = float(time_tokens[1])
                    except ValueError:
                        continue # Skip invalid time lines
                else:
                    continue
                    
                speaker_col = parts[1].strip()
                if speaker_col:
                    current_speaker = speaker_col.replace(':', '').strip()
                    
                text_col = parts[2].strip() if len(parts) > 2 else ""
                if not text_col:
                    continue
                    
                # Clean text of non-terminal noise
                # Remove brackets [ ], numbers inside brackets [2 ... 2], and other transcription tokens
                clean_text = text_col
                # strip trailing whitespace, trailing overlaps e.g. ] or ]* or 2] or >
                clean_text = clean_text.rstrip(' \t\n\r]*1234567890<@')
                
                if not clean_text:
                    continue
                    
                last_char = clean_text[-1]
                
                # Handle double dash -- (truncation)
                if clean_text.endswith('--') or clean_text.endswith('-'):
                    last_char = '-'
                    
                human_class = "Excluded"
                exclusion_reason = None
                
                if last_char == '.':
                    human_class = "Falling"
                elif last_char == '?':
                    human_class = "Rising"
                elif last_char == ',':
                    human_class = "Level"
                elif last_char == '-':
                    exclusion_reason = "Truncated IU (dash)"
                else:
                    exclusion_reason = f"No standard terminal contour found (ends with '{last_char}')"
                
                ius.append({
                    "iu_id": line_idx + 1, # 1-indexed
                    "start": start,
                    "end": end,
                    "speaker": current_speaker,
                    "annotation_symbol": last_char if human_class != "Excluded" or last_char == '-' else None,
                    "human_reference_class": human_class,
                    "exclusion_reason": exclusion_reason,
                    "text": text_col
                })
                
    return ius

if __name__ == "__main__":
    test_files = [
        "sbc_transcripts_stream/SBC001.trn",
        "sbc_transcripts_stream/SBC002.trn"
    ]
    
    for tf in test_files:
        if not os.path.exists(tf):
            print(f"File not found: {tf}")
            continue
            
        print(f"--- Parsing {os.path.basename(tf)} ---")
        ius = parse_sbcsae_trn(tf)
        print(f"Total IUs found: {len(ius)}")
        
        counts = {"Falling": 0, "Rising": 0, "Level": 0, "Excluded": 0}
        for iu in ius:
            counts[iu["human_reference_class"]] += 1
            
        print(f"Distribution: {counts}")
        
        # Print a few samples of each to verify mapping
        print("\nSamples:")
        samples_shown = 0
        for iu in ius:
            if samples_shown >= 10:
                break
            ann = iu['annotation_symbol'] or "None"
            print(f"{iu['iu_id']:<4} | {iu['start']:<6.2f} {iu['end']:<6.2f} | {iu['speaker'][:6]:<6} | {ann:<4} | {iu['human_reference_class']:<8} | {iu['text']}")
            samples_shown += 1
            
        print("\n")
