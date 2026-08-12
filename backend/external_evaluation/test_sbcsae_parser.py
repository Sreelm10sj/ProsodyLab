import re

def parse_sbcsae_iu(line):
    # Mock parser for SBCSAE .trn style format
    # Format typically: start_time end_time \t SPEAKER \t Text
    
    parts = line.strip().split('\t')
    if len(parts) < 3:
        # Fallback if speaker is omitted: start end \t Text
        parts = line.strip().split(' ', 2)
        if len(parts) < 3:
            return None
            
        start = float(parts[0])
        end = float(parts[1])
        text = parts[2].strip()
        speaker = "UNKNOWN"
    else:
        time_part = parts[0].strip().split(' ')
        if len(time_part) == 2:
            start, end = float(time_part[0]), float(time_part[1])
        else:
            return None
        speaker = parts[1].strip()
        text = parts[2].strip()
        
    # Extract final punctuation/continuation marker
    # In SBCSAE, punctuation marks terminal contours.
    
    # Strip trailing whitespace and non-prosodic brackets (e.g. overlap brackets like ] )
    clean_text = text.rstrip(' ]1234567890*')
    
    if not clean_text:
        return None
        
    last_char = clean_text[-1]
    
    # Optional: some transcriptions use -- or -
    if clean_text.endswith('--'):
        last_char = '-'
        
    if last_char == '.':
        human_class = "Falling"
    elif last_char == '?':
        human_class = "Rising"
    elif last_char == ',':
        human_class = "Level"
    elif last_char == '-':
        human_class = "Excluded"
    else:
        human_class = "Excluded"
        
    # Determine exclusion reason if any
    exclusion_reason = None
    if human_class == "Excluded":
        if last_char == '-':
            exclusion_reason = "Truncated IU (dash)"
        else:
            exclusion_reason = f"No standard terminal contour found (ends with '{last_char}')"
            
    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "annotation": last_char if human_class != "Excluded" or last_char == '-' else None,
        "human_class": human_class,
        "exclusion_reason": exclusion_reason,
        "text": text
    }

if __name__ == "__main__":
    test_lines = [
        "10.00 12.50\tLYNNE\tgonna do the feet today,",
        "12.60 14.20\tLYNNE\tI mean you get so tired.",
        "14.30 15.00\tLYNNE\tDid they train you? [2",
        "15.10 15.80\tLENORE\tDid they train you that --",
        "16.00 16.50\tLYNNE\tyeah, [3",
        "16.60 17.00\tLENORE\tyeah."
    ]
    
    print(f"{'Start':<6} | {'End':<6} | {'Annotation':<10} | {'Mapped Class':<10} | {'Text'}")
    print("-" * 70)
    for line in test_lines:
        res = parse_sbcsae_iu(line)
        if res:
            ann = res['annotation'] or "None"
            print(f"{res['start']:<6.2f} | {res['end']:<6.2f} | {ann:<10} | {res['human_class']:<10} | {res['text']}")
