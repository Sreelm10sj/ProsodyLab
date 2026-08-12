import os
import sys
import glob
import csv
import numpy as np
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "WhiStress"))
from whistress_service import service

def analyze_all_files():
    wav_files = glob.glob("*.wav")
    if not wav_files:
        print("No .wav files found in the current directory.")
        return

    os.makedirs("stylization_validation", exist_ok=True)
    intonation_csv_path = os.path.join("stylization_validation", "intonation_comparison.csv")
    
    all_data = []
    
    # Stats
    total_phrases = 0
    identical_classifications = 0
    diff_less_equal_025 = 0
    diff_above_025 = 0
    diff_above_05 = 0
    
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    slope_diffs = []
    
    borderline_phrases = []
    large_diff_phrases = []

    print(f"Found {len(wav_files)} WAV files. Beginning evaluation...")
    service.load_model()
    
    with open(intonation_csv_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "Audio File", "Phrase ID", "Start Time", "End Time", 
            "Raw Slope", "Raw Class", "Stylized Slope", "Stylized Class", 
            "Slope Difference", "Absolute Slope Difference"
        ])
        
        for wav_file in wav_files:
            print(f"Processing {wav_file}...")
            try:
                with open(wav_file, "rb") as f_wav:
                    audio_bytes = f_wav.read()
                
                # predict_with_intonation runs the full pipeline and returns phrase objects
                words, phrases, pitch_times, pitch_f0, pitch_stylized, audio_array, sr = service.predict_with_intonation(audio_bytes)
                
                for i, phrase in enumerate(phrases):
                    start = phrase["start"]
                    end = phrase["end"]
                    raw = phrase["pitch_raw"]
                    stylized = phrase["pitch_stylized"]

                    r_slope = raw.get("overall_slope")
                    r_class = raw.get("pattern", "unknown")
                    
                    s_slope = stylized.get("overall_slope")
                    s_class = stylized.get("pattern", "unknown")

                    if r_slope is None or s_slope is None:
                        continue
                        
                    slope_diff = s_slope - r_slope
                    abs_diff = abs(slope_diff)
                    
                    writer.writerow([
                        wav_file, i + 1, f"{start:.3f}", f"{end:.3f}",
                        f"{r_slope:.3f}", r_class, f"{s_slope:.3f}", s_class,
                        f"{slope_diff:.3f}", f"{abs_diff:.3f}"
                    ])
                    
                    total_phrases += 1
                    confusion_matrix[r_class][s_class] += 1
                    slope_diffs.append(slope_diff)
                    
                    if r_class == s_class:
                        identical_classifications += 1
                    
                    if abs_diff <= 0.25:
                        diff_less_equal_025 += 1
                    else:
                        diff_above_025 += 1
                        if abs_diff > 0.5:
                            diff_above_05 += 1
                            
                    # Identify Borderline (margin < 0.25 to thresholds +1.5 or -1.5)
                    margin_pos = abs(r_slope - 1.5)
                    margin_neg = abs(r_slope - (-1.5))
                    if margin_pos < 0.25 or margin_neg < 0.25:
                        borderline_phrases.append({
                            "file": wav_file, "phrase_id": i+1,
                            "r_slope": r_slope, "s_slope": s_slope,
                            "r_class": r_class, "s_class": s_class
                        })
                        
                    # Identify large diffs
                    if abs_diff > 0.5:
                        large_diff_phrases.append({
                            "file": wav_file, "phrase_id": i+1,
                            "r_slope": r_slope, "s_slope": s_slope,
                            "diff": slope_diff, "abs_diff": abs_diff,
                            "r_class": r_class, "s_class": s_class
                        })

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                
    # Generate slope_comparison.csv (summary stats)
    slope_csv_path = os.path.join("stylization_validation", "slope_comparison.csv")
    with open(slope_csv_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Metric", "Value"])
        if slope_diffs:
            mean_diff = np.mean(slope_diffs)
            median_diff = np.median(slope_diffs)
            std_diff = np.std(slope_diffs)
            max_abs_diff = np.max(np.abs(slope_diffs))
            writer.writerow(["Mean Slope Difference", f"{mean_diff:.4f}"])
            writer.writerow(["Median Slope Difference", f"{median_diff:.4f}"])
            writer.writerow(["Standard Deviation", f"{std_diff:.4f}"])
            writer.writerow(["Maximum Absolute Difference", f"{max_abs_diff:.4f}"])
            writer.writerow(["Total Phrases", total_phrases])
            writer.writerow(["Identical Classifications", identical_classifications])
            writer.writerow(["Within 0.25 st/s", diff_less_equal_025])
            writer.writerow(["Above 0.25 st/s", diff_above_025])
            writer.writerow(["Above 0.5 st/s", diff_above_05])

    print("\n" + "="*60)
    print("PHASE 1 & 2 SUMMARY: BASELINE COMPARISON")
    print("="*60)
    print(f"Total phrases analyzed: {total_phrases}")
    if total_phrases > 0:
        agreement = (identical_classifications / total_phrases) * 100
        disagreement = 100 - agreement
        print(f"Agreement: {agreement:.1f}% ({identical_classifications} phrases)")
        print(f"Disagreement: {disagreement:.1f}% ({total_phrases - identical_classifications} phrases)")
        print(f"Diff <= 0.25 st/s: {diff_less_equal_025}")
        print(f"Diff > 0.25 st/s: {diff_above_025}")
        print(f"Diff > 0.5 st/s: {diff_above_05}")
        
        print("\n--- Confusion Matrix ---")
        classes = ["rising", "level", "falling", "insufficient_voicing"]
        print("Raw \\ Sty".ljust(25) + " " + " ".join([f"{c:>12}" for c in classes]))
        for r_c in classes:
            row = [f"{confusion_matrix[r_c][s_c]:>12}" for s_c in classes]
            print(f"{r_c:<25} " + " ".join(row))
            
        print("\n--- Slope Difference Stats ---")
        print(f"Mean Difference:   {mean_diff:+.4f} st/s")
        print(f"Median Difference: {median_diff:+.4f} st/s")
        print(f"Std Deviation:     {std_diff:.4f} st/s")
        print(f"Max Abs Diff:      {max_abs_diff:.4f} st/s")
        
        print("\n" + "="*60)
        print("PHASE 3: THRESHOLD-BORDERLINE ANALYSIS")
        print("="*60)
        print(f"Found {len(borderline_phrases)} phrases near thresholds (+/- 1.5 st/s margin 0.25):")
        for p in borderline_phrases:
            print(f"[{p['file']} Phrase {p['phrase_id']}]")
            print(f"  Raw:      {p['r_slope']:+.3f} st/s ({p['r_class']})")
            print(f"  Stylized: {p['s_slope']:+.3f} st/s ({p['s_class']})")
            
        print("\n" + "="*60)
        print("LARGE SLOPE DIFFERENCES (> 0.5 st/s)")
        print("="*60)
        print(f"Found {len(large_diff_phrases)} phrases:")
        for p in large_diff_phrases:
            print(f"[{p['file']} Phrase {p['phrase_id']}]")
            print(f"  Diff: {p['diff']:+.3f} st/s | Raw: {p['r_slope']:+.3f} ({p['r_class']}) | Stylized: {p['s_slope']:+.3f} ({p['s_class']})")
            
if __name__ == "__main__":
    analyze_all_files()
