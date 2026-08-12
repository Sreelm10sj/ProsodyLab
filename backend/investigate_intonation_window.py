import sys
import os
import json
import math
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiStress"))
from whistress_service import service
from pitch_analysis import PITCH_TIME_STEP, VOICED_MIN_FRAMES_FOR_SLOPE

def get_slopes(phrase, pitch_times, pitch_f0, utterance_median_hz):
    phrase_start = phrase["start"]
    phrase_end = phrase["end"]
    mask = (pitch_times >= phrase_start) & (pitch_times <= phrase_end)
    phrase_times = pitch_times[mask]
    phrase_f0 = pitch_f0[mask]

    voiced_mask = ~np.isnan(phrase_f0) & (phrase_f0 > 0.0)
    voiced_times = phrase_times[voiced_mask]
    voiced_f0 = phrase_f0[voiced_mask]
    voiced_count = len(voiced_f0)
    total_frames = len(phrase_f0)
    
    if voiced_count < VOICED_MIN_FRAMES_FOR_SLOPE:
        return None

    semitones = 12.0 * np.log2(voiced_f0 / utterance_median_hz)

    # 1. Current Slope
    t = voiced_times - voiced_times[0]
    A = np.column_stack([t, np.ones(voiced_count)])
    slope_coeff, _, _, _ = np.linalg.lstsq(A, semitones, rcond=None)
    current_slope = float(slope_coeff[0])

    # 2. Trimmed Slope
    first_voiced = float(voiced_times[0])
    last_voiced = float(voiced_times[-1])
    leading_silence = max(0.0, first_voiced - phrase_start)
    trailing_silence = max(0.0, phrase_end - last_voiced)
    
    t_trim = voiced_times - first_voiced
    A_trim = np.column_stack([t_trim, np.ones(voiced_count)])
    slope_coeff_trim, _, _, _ = np.linalg.lstsq(A_trim, semitones, rcond=None)
    trimmed_slope = float(slope_coeff_trim[0])
    
    # 3. Pause-Compressed Slope
    t_diff = np.diff(voiced_times)
    
    pause_compressed_times = [0.0]
    total_internal_unvoiced = 0.0
    num_short_pauses = 0
    short_pauses_list = []
    
    accumulated_time = 0.0
    for i in range(len(t_diff)):
        gap = float(t_diff[i])
        is_gap = gap > (PITCH_TIME_STEP * 1.5)
        if is_gap:
            # We measure unvoiced gap. A standard frame step is 0.01.
            # So the unvoiced silence duration is gap - 0.01.
            silence_duration = gap - PITCH_TIME_STEP
            total_internal_unvoiced += silence_duration
            if silence_duration < 0.45:
                accumulated_time += PITCH_TIME_STEP
                num_short_pauses += 1
                short_pauses_list.append({
                    "start": float(voiced_times[i] + PITCH_TIME_STEP),
                    "end": float(voiced_times[i+1]),
                    "duration": float(silence_duration)
                })
            else:
                accumulated_time += gap
        else:
            accumulated_time += gap
        pause_compressed_times.append(accumulated_time)
        
    pause_compressed_times = np.array(pause_compressed_times)
    A_comp = np.column_stack([pause_compressed_times, np.ones(voiced_count)])
    slope_coeff_comp, _, _, _ = np.linalg.lstsq(A_comp, semitones, rcond=None)
    pause_compressed_slope = float(slope_coeff_comp[0])

    def get_label(slope):
        if slope >= 1.5: return "Rising"
        if slope <= -1.5: return "Falling"
        return "Level"

    return {
        "phrase_id": phrase.get("phrase_id", "unknown"),
        "start": phrase_start,
        "end": phrase_end,
        "voiced_count": voiced_count,
        "voiced_fraction": voiced_count / total_frames if total_frames else 0,
        "first_voiced": first_voiced,
        "last_voiced": last_voiced,
        "leading_silence": leading_silence,
        "trailing_silence": trailing_silence,
        "total_internal_unvoiced": total_internal_unvoiced,
        "num_short_pauses": num_short_pauses,
        "short_pauses_list": short_pauses_list,
        "current_slope": current_slope,
        "trimmed_slope": trimmed_slope,
        "pause_compressed_slope": pause_compressed_slope,
        "current_label": get_label(current_slope),
        "trimmed_label": get_label(trimmed_slope),
        "pause_compressed_label": get_label(pause_compressed_slope),
    }

def process_file(filepath):
    print(f"\nProcessing {filepath}...")
    with open(filepath, "rb") as f:
        audio_bytes = f.read()
    
    pairs, pauses, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)
    
    # Calculate utterance median hz
    valid_f0 = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0.0)]
    utterance_median_hz = float(np.median(valid_f0)) if len(valid_f0) > 0 else None
    
    if utterance_median_hz is None:
        print("No voiced frames in file.")
        return []
    
    results = []
    for i, phrase in enumerate(phrases):
        res = get_slopes(phrase, pitch_times, pitch_stylized, utterance_median_hz)
        if res:
            res["phrase_id"] = i + 1
            results.append(res)
    return results

if __name__ == "__main__":
    service.load_model()
    
    files = ["example1_football.wav", "example4.wav", "test_two_phrases.wav", "test_pause.mp3"]
    all_results = {}
    for f in files:
        if os.path.exists(f):
            all_results[f] = process_file(f)
        else:
            print(f"File not found: {f}")
            
    with open("investigate_results.json", "w") as out:
        json.dump(all_results, out, indent=2)
    print("Done. Saved to investigate_results.json")
