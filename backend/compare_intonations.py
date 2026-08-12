import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "WhiStress"))
from whistress_service import service

def analyze_file(filepath: str):
    print(f"\nAnalyzing {filepath} ...")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, "rb") as f:
        audio_bytes = f.read()

    service.load_model()
    
    words, phrases, pitch_times, pitch_f0, pitch_stylized, audio_array, sr = service.predict_with_intonation(audio_bytes)

    print("-" * 85)
    print(f"{'Phrase':<8} | {'Start':>7} | {'End':>7} | {'Raw Slope':>10} | {'Raw Class':<10} | {'Sty Slope':>10} | {'Sty Class':<10}")
    print("-" * 85)

    same_count = 0
    total_count = len(phrases)

    for i, phrase in enumerate(phrases):
        start = phrase["start"]
        end = phrase["end"]
        raw = phrase["pitch_raw"]
        stylized = phrase["pitch_stylized"]

        r_slope = raw.get("overall_slope")
        r_class = raw.get("pattern", "unknown")
        
        s_slope = stylized.get("overall_slope")
        s_class = stylized.get("pattern", "unknown")

        r_slope_str = f"{r_slope:+.2f}" if r_slope is not None else "N/A"
        s_slope_str = f"{s_slope:+.2f}" if s_slope is not None else "N/A"
        
        print(f"{i+1:<8} | {start:7.2f} | {end:7.2f} | {r_slope_str:>10} | {r_class:<10} | {s_slope_str:>10} | {s_class:<10}")

        if r_class == s_class:
            same_count += 1
            
    print("-" * 85)
    print("\nSummary:")
    print(f"Total phrases: {total_count}")
    print(f"Same classifications: {same_count}")
    changed = total_count - same_count
    print(f"Changed classifications: {changed}")
    if total_count > 0:
        print(f"Agreement %: {(same_count / total_count) * 100:.1f}%")
    else:
        print("Agreement %: N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare raw vs stylized phrase intonation classifications.")
    parser.add_argument("files", nargs="*", default=["example1_football.wav", "example3.wav"], help="Audio files to analyze")
    args = parser.parse_args()

    for f in args.files:
        analyze_file(f)
