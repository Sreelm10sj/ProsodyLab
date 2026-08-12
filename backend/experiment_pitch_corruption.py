import os
import sys
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "WhiStress"))
from whistress_service import service
from pitch_analysis import segment_phrases, extract_pitch_features, classify_contour
from pitch_stylization_fast import stylize_pitch_contour

# Configuration for Experiment
RANDOM_SEEDS = [42, 123, 456, 789, 999]
NOISE_LEVELS = [("Low", 2.0), ("Medium", 5.0), ("High", 10.0)]
OCTAVE_ERROR_RATES = [0.05, 0.10, 0.20]
OCTAVE_SEGMENT_LENGTHS = [5, 10, 15]

def apply_corruption(f0, condition_type, severity, seed):
    np.random.seed(seed)
    cf0 = f0.copy()
    valid_idx = np.where(~np.isnan(cf0) & (cf0 > 0))[0]
    
    if len(valid_idx) == 0:
        return cf0
        
    if condition_type == "gaussian":
        cf0[valid_idx] += np.random.normal(0, severity, size=len(valid_idx))
        cf0[valid_idx] = np.maximum(cf0[valid_idx], 1.0)
        
    elif condition_type == "octave_doubling":
        n_corrupt = int(len(valid_idx) * severity)
        if n_corrupt > 0:
            choices = np.random.choice(valid_idx, size=n_corrupt, replace=False)
            cf0[choices] *= 2.0
            
    elif condition_type == "octave_halving":
        n_corrupt = int(len(valid_idx) * severity)
        if n_corrupt > 0:
            choices = np.random.choice(valid_idx, size=n_corrupt, replace=False)
            cf0[choices] *= 0.5
            
    elif condition_type == "octave_jumps":
        num_segments = max(1, int(len(valid_idx) * 0.1 / severity)) # roughly 10% frames corrupted
        for _ in range(num_segments):
            if len(valid_idx) > severity:
                start_i = np.random.choice(len(valid_idx) - severity)
                actual_indices = valid_idx[start_i : start_i + severity]
                factor = np.random.choice([0.5, 2.0])
                cf0[actual_indices] *= factor
                
    elif condition_type == "combined":
        # Severity is a multiplier 1=low, 2=med, 3=high
        # Gaussian
        cf0[valid_idx] += np.random.normal(0, 2.0 * severity, size=len(valid_idx))
        cf0[valid_idx] = np.maximum(cf0[valid_idx], 1.0)
        # Random doubling/halving
        n_corrupt = int(len(valid_idx) * 0.05 * severity)
        if n_corrupt > 0:
            choices = np.random.choice(valid_idx, size=n_corrupt, replace=False)
            factors = np.random.choice([0.5, 2.0], size=n_corrupt)
            cf0[choices] *= factors
        # Short jumps
        num_segments = int(2 * severity)
        seg_len = 10
        for _ in range(num_segments):
            if len(valid_idx) > seg_len:
                start_i = np.random.choice(len(valid_idx) - seg_len)
                actual_indices = valid_idx[start_i : start_i + seg_len]
                factor = np.random.choice([0.5, 2.0])
                cf0[actual_indices] *= factor

    return cf0

def evaluate_contour(f0_contour, phrase, utterance_median_hz, pitch_times):
    features = extract_pitch_features(
        phrase_start=phrase["start"],
        phrase_end=phrase["end"],
        pitch_times=pitch_times,
        pitch_f0=f0_contour,
        utterance_median_hz=utterance_median_hz,
    )
    pattern = classify_contour(features)
    return features.get("overall_slope"), pattern

def run_experiment(files):
    os.makedirs("stylization_validation", exist_ok=True)
    os.makedirs("stylization_validation/plots", exist_ok=True)
    csv_path = os.path.join("stylization_validation", "robustness_results.csv")
    
    with open(csv_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "Audio File", "Phrase ID", "Condition", "Severity", "Seed",
            "Clean Slope", "Clean Class",
            "Raw Slope", "Raw Class", "Sty Slope", "Sty Class",
            "Raw Slope Error", "Sty Slope Error",
            "Raw RMSE", "Sty RMSE"
        ])
        
        service.load_model()
        
        for wav_file in files:
            print(f"Processing {wav_file} for robustness...")
            if not os.path.exists(wav_file):
                continue
                
            with open(wav_file, "rb") as f_wav:
                audio_bytes = f_wav.read()
                
            words, phrases, pitch_times, clean_f0, clean_stylized, audio_array, sr = service.predict_with_intonation(audio_bytes)
            
            phrases_raw = segment_phrases(words)
            
            # Ground truth is Clean F0 processing
            voiced_f0_all = clean_f0[~np.isnan(clean_f0) & (clean_f0 > 0.0)]
            if len(voiced_f0_all) == 0:
                continue
            clean_median_hz = float(np.median(voiced_f0_all))
            
            clean_results = {}
            for i, phrase in enumerate(phrases_raw):
                slope, cls = evaluate_contour(clean_f0, phrase, clean_median_hz, pitch_times)
                clean_results[i] = {"slope": slope, "cls": cls}
                
            conditions = []
            for name, level in NOISE_LEVELS:
                conditions.append(("gaussian", name, level))
            for rate in OCTAVE_ERROR_RATES:
                conditions.append(("octave_doubling", f"{rate*100}%", rate))
                conditions.append(("octave_halving", f"{rate*100}%", rate))
            for slen in OCTAVE_SEGMENT_LENGTHS:
                conditions.append(("octave_jumps", f"{slen}_frames", slen))
            for level in [1, 2, 3]:
                conditions.append(("combined", f"Level_{level}", level))
                
            for c_type, c_name, severity in conditions:
                for seed in RANDOM_SEEDS:
                    corrupted_f0 = apply_corruption(clean_f0, c_type, severity, seed)
                    corrupted_stylized_f0 = stylize_pitch_contour(corrupted_f0)
                    
                    voiced_c_f0 = corrupted_f0[~np.isnan(corrupted_f0) & (corrupted_f0 > 0.0)]
                    c_median_hz = float(np.median(voiced_c_f0)) if len(voiced_c_f0) > 0 else clean_median_hz
                    
                    # Compute RMSE on valid frames
                    valid = ~np.isnan(clean_f0) & (clean_f0 > 0) & ~np.isnan(corrupted_f0) & (corrupted_f0 > 0)
                    raw_rmse = np.sqrt(np.mean((corrupted_f0[valid] - clean_f0[valid])**2)) if np.any(valid) else 0.0
                    valid_s = ~np.isnan(clean_f0) & (clean_f0 > 0) & ~np.isnan(corrupted_stylized_f0)
                    sty_rmse = np.sqrt(np.mean((corrupted_stylized_f0[valid_s] - clean_f0[valid_s])**2)) if np.any(valid_s) else 0.0
                    
                    for i, phrase in enumerate(phrases_raw):
                        c_slope = clean_results[i]["slope"]
                        c_cls = clean_results[i]["cls"]
                        if c_slope is None:
                            continue
                            
                        r_slope, r_cls = evaluate_contour(corrupted_f0, phrase, c_median_hz, pitch_times)
                        s_slope, s_cls = evaluate_contour(corrupted_stylized_f0, phrase, c_median_hz, pitch_times)
                        
                        r_err = abs(r_slope - c_slope) if r_slope is not None else -1
                        s_err = abs(s_slope - c_slope) if s_slope is not None else -1
                        
                        writer.writerow([
                            wav_file, i+1, c_type, c_name, seed,
                            f"{c_slope:.3f}", c_cls,
                            f"{r_slope:.3f}" if r_slope else "N/A", r_cls,
                            f"{s_slope:.3f}" if s_slope else "N/A", s_cls,
                            f"{r_err:.3f}", f"{s_err:.3f}",
                            f"{raw_rmse:.2f}", f"{sty_rmse:.2f}"
                        ])
                        
                    # Save a representative plot for seed 42 only
                    if seed == 42 and i == 0 and ("example1" in wav_file or "example3" in wav_file):
                        plt.figure(figsize=(10, 8))
                        plt.subplot(4, 1, 1)
                        plt.plot(pitch_times, clean_f0, 'k.', label='Clean baseline F0')
                        plt.title(f"{wav_file} - {c_type} {c_name}")
                        plt.legend()
                        
                        plt.subplot(4, 1, 2)
                        plt.plot(pitch_times, corrupted_f0, 'r.', label='Corrupted raw F0')
                        plt.legend()
                        
                        plt.subplot(4, 1, 3)
                        plt.plot(pitch_times, corrupted_stylized_f0, 'g-', linewidth=2, label='Stylized after DP-MAE')
                        plt.legend()
                        
                        plt.subplot(4, 1, 4)
                        plt.plot(pitch_times[valid], corrupted_f0[valid] - clean_f0[valid], 'r.', alpha=0.5, label='Raw residual')
                        plt.plot(pitch_times[valid_s], corrupted_stylized_f0[valid_s] - clean_f0[valid_s], 'g-', alpha=0.8, label='Stylized residual')
                        plt.axhline(0, color='k', linestyle='--')
                        plt.legend()
                        
                        plt.tight_layout()
                        plt.savefig(f"stylization_validation/plots/{wav_file.replace('.wav','')}_{c_type}_{c_name}.png")
                        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pitch stylization robustness")
    parser.add_argument("files", nargs="*", default=["example1_football.wav", "example3.wav"], help="Audio files to analyze")
    args = parser.parse_args()
    
    run_experiment(args.files)
