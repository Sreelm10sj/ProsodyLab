import os
import glob
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import whistress_service
from pitch_analysis import extract_pitch, build_phrase_response
import pitch_stylization_reference as pitch_stylization_reference
import pitch_stylization_fast

OUTPUT_DIR = "stylization_validation"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

def extract_boundaries(stylized_f0):
    """
    Infers segment boundary indices from a piecewise linear contour.
    A boundary is considered the index where the slope changes.
    """
    boundaries = []
    in_region = False
    region_start = 0
    
    for i in range(len(stylized_f0)):
        if not np.isnan(stylized_f0[i]):
            if not in_region:
                in_region = True
                region_start = i
        else:
            if in_region:
                # Process region
                region_idx = np.arange(region_start, i)
                region_vals = stylized_f0[region_start:i]
                
                # Check for slope changes
                for j in range(1, len(region_vals) - 1):
                    slope_prev = region_vals[j] - region_vals[j-1]
                    slope_next = region_vals[j+1] - region_vals[j]
                    if abs(slope_next - slope_prev) > 1e-4:
                        boundaries.append(region_start + j)
                in_region = False
                
    if in_region:
        region_vals = stylized_f0[region_start:]
        for j in range(1, len(region_vals) - 1):
            slope_prev = region_vals[j] - region_vals[j-1]
            slope_next = region_vals[j+1] - region_vals[j]
            if abs(slope_next - slope_prev) > 1e-4:
                boundaries.append(region_start + j)
                
    return boundaries

def compute_boundary_differences(ref_bounds, fast_bounds):
    diffs = []
    # Nearest neighbor matching
    for b_f in fast_bounds:
        if len(ref_bounds) > 0:
            nearest_ref = min(ref_bounds, key=lambda x: abs(x - b_f))
            diffs.append(abs(b_f - nearest_ref))
    return diffs

def process_file(filepath, service):
    filename = os.path.basename(filepath)
    print(f"\nProcessing {filename}...")
    
    with open(filepath, "rb") as f:
        audio_bytes = f.read()
        
    audio_dict = service.process_audio(audio_bytes)
    audio_array = audio_dict["array"]
    sr = audio_dict["sampling_rate"]
    
    pitch_times, pitch_f0 = extract_pitch(audio_array, sr=sr)
    
    # Get words for intonation pipeline
    try:
        words = service.client.predict(audio=audio_dict, transcription=None, return_pairs=True)
    except Exception as e:
        print(f"Skipping {filename} due to WhiStress client error: {e}")
        return None
        
    duration = len(audio_array) / sr
    voiced_mask = ~np.isnan(pitch_f0) & (pitch_f0 > 0)
    voiced_duration = np.sum(voiced_mask) * (pitch_times[1] - pitch_times[0]) if len(pitch_times) > 1 else 0
    
    # Count voiced regions
    regions = 0
    in_region = False
    for v in voiced_mask:
        if v and not in_region:
            regions += 1
            in_region = True
        elif not v:
            in_region = False
            
    # Reference Stylization
    t0 = time.perf_counter()
    ref_stylized = pitch_stylization_reference.stylize_pitch_contour(pitch_f0, P=1)
    ref_time = time.perf_counter() - t0
    
    # Fast Stylization
    t0 = time.perf_counter()
    fast_stylized = pitch_stylization_fast.stylize_pitch_contour(pitch_f0, P=1)
    fast_time = time.perf_counter() - t0
    
    speedup = ref_time / fast_time if fast_time > 0 else 0
    
    valid_mask = ~np.isnan(ref_stylized) & ~np.isnan(fast_stylized)
    ref_valid = ref_stylized[valid_mask]
    fast_valid = fast_stylized[valid_mask]
    
    n_frames = len(ref_valid)
    if n_frames == 0:
        return None
        
    diff = ref_valid - fast_valid
    abs_diff = np.abs(diff)
    
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff)
    pearson_r, _ = pearsonr(ref_valid, fast_valid) if len(ref_valid) > 1 else (0, 0)
    
    ref_bounds = extract_boundaries(ref_stylized)
    fast_bounds = extract_boundaries(fast_stylized)
    bound_diffs = compute_boundary_differences(ref_bounds, fast_bounds)
    
    # Intonation Pipeline Test
    # Pipeline A: Raw
    phrases_raw = build_phrase_response(words, pitch_times, pitch_f0)
    # Pipeline B: Ref
    phrases_ref = build_phrase_response(words, pitch_times, ref_stylized)
    # Pipeline C: Fast
    phrases_fast = build_phrase_response(words, pitch_times, fast_stylized)
    
    intonation_results = []
    
    # Compare B and C (Ref vs Fast)
    total_phrases = min(len(phrases_ref), len(phrases_fast))
    match_count_ref_fast = 0
    
    for i in range(total_phrases):
        r_p = phrases_ref[i]
        f_p = phrases_fast[i]
        
        r_pitch = r_p.get("pitch")
        f_pitch = f_p.get("pitch")
        
        r_label = r_pitch.get("pattern") if r_pitch else "None"
        f_label = f_pitch.get("pattern") if f_pitch else "None"
        
        if r_label == f_label:
            match_count_ref_fast += 1
            
        intonation_results.append({
            'filename': filename,
            'phrase_text': r_p.get("text", ""),
            'ref_slope': r_pitch.get("slope", 0.0) if r_pitch else 0.0,
            'fast_slope': f_pitch.get("slope", 0.0) if f_pitch else 0.0,
            'ref_label': r_label,
            'fast_label': f_label
        })
            
    # Generate Plot
    plt.figure(figsize=(15, 8))
    
    # Top: Contours
    plt.subplot(2, 1, 1)
    plt.title(f"Pitch Stylization Overlay ({filename})")
    plt.plot(pitch_times, pitch_f0, color='lightgray', alpha=0.5, label='Raw F0', linewidth=1)
    plt.plot(pitch_times, ref_stylized, color='blue', alpha=0.7, label='Reference', linewidth=2.5)
    plt.plot(pitch_times, fast_stylized, color='red', alpha=0.7, label='Fast', linewidth=1.5, linestyle='--')
    
    # Plot fast boundaries
    for b in fast_bounds:
        plt.axvline(pitch_times[b], color='green', alpha=0.3, linestyle=':')
        
    plt.ylabel('Pitch (Semitones)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom: Residuals
    plt.subplot(2, 1, 2)
    plt.title("Residual Error (Reference - Fast)")
    diff_full = np.full_like(pitch_times, np.nan)
    diff_full[valid_mask] = ref_stylized[valid_mask] - fast_stylized[valid_mask]
    plt.plot(pitch_times, diff_full, color='black', linewidth=1)
    
    # Highlight errors > 0.5
    high_err_idx = np.where(np.abs(diff_full) > 0.5)[0]
    plt.scatter(pitch_times[high_err_idx], diff_full[high_err_idx], color='red', s=10, zorder=5)
    
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.ylabel('Difference (st)')
    plt.xlabel('Time (s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{os.path.splitext(filename)[0]}.png"), dpi=150)
    plt.close()
    
    return {
        'filename': filename,
        'audio_duration': duration,
        'voiced_duration': voiced_duration,
        'voiced_regions': regions,
        'ref_time': ref_time,
        'fast_time': fast_time,
        'speedup': speedup,
        'rmse': rmse,
        'mae': mae,
        'max_err': max_err,
        'pearson': pearson_r,
        'ref_segments': len(ref_bounds) + regions,
        'fast_segments': len(fast_bounds) + regions,
        'bound_diffs': bound_diffs,
        'intonation_results': intonation_results,
        'match_count_ref_fast': match_count_ref_fast,
        'total_phrases': total_phrases
    }

def generate_reports(results):
    print("\nGenerating final reports...")
    
    # 1. CSV
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Duration (s)', 'Voiced (s)', 'Regions', 'Ref Time (s)', 'Fast Time (s)', 
                         'Speedup', 'RMSE', 'MAE', 'Max Error', 'Pearson R', 'Ref Segments', 'Fast Segments', 'Intonation Match %'])
        
        for r in results:
            match_pct = (r['match_count_ref_fast'] / r['total_phrases'] * 100) if r['total_phrases'] > 0 else 100
            writer.writerow([
                r['filename'], f"{r['audio_duration']:.2f}", f"{r['voiced_duration']:.2f}", r['voiced_regions'],
                f"{r['ref_time']:.3f}", f"{r['fast_time']:.3f}", f"{r['speedup']:.2f}",
                f"{r['rmse']:.4f}", f"{r['mae']:.4f}", f"{r['max_err']:.4f}", f"{r['pearson']:.6f}",
                r['ref_segments'], r['fast_segments'], f"{match_pct:.1f}"
            ])
            
    # 2. Stats
    speedups = [r['speedup'] for r in results]
    rmses = [r['rmse'] for r in results]
    maes = [r['mae'] for r in results]
    pearsons = [r['pearson'] for r in results]
    
    stats_md = os.path.join(OUTPUT_DIR, "summary_statistics.md")
    with open(stats_md, 'w') as f:
        f.write("# Benchmark Summary Statistics\n\n")
        f.write("| Metric | Mean | Median | Std Dev | Min | Max |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| Speedup | {np.mean(speedups):.2f}x | {np.median(speedups):.2f}x | {np.std(speedups):.2f} | {np.min(speedups):.2f}x | {np.max(speedups):.2f}x |\n")
        f.write(f"| RMSE (st) | {np.mean(rmses):.4f} | {np.median(rmses):.4f} | {np.std(rmses):.4f} | {np.min(rmses):.4f} | {np.max(rmses):.4f} |\n")
        f.write(f"| MAE (st) | {np.mean(maes):.4f} | {np.median(maes):.4f} | {np.std(maes):.4f} | {np.min(maes):.4f} | {np.max(maes):.4f} |\n")
        f.write(f"| Pearson r | {np.mean(pearsons):.6f} | {np.median(pearsons):.6f} | {np.std(pearsons):.6f} | {np.min(pearsons):.6f} | {np.max(pearsons):.6f} |\n")
        
    # 3. Boundary Histogram & Stats
    all_diffs = []
    for r in results:
        all_diffs.extend(r['bound_diffs'])
        
    if all_diffs:
        plt.figure(figsize=(10, 6))
        
        # Group >4 into bucket 4
        plot_diffs = [min(d, 4) for d in all_diffs]
        bins = np.arange(-0.5, 5.5, 1)
        counts, _ = np.histogram(plot_diffs, bins=bins)
        
        plt.bar(range(5), counts, tick_label=['0', '1', '2', '3', '4+'], color='skyblue', edgecolor='black')
        plt.title('Boundary Displacement Distribution')
        plt.xlabel('Displacement (frames)')
        plt.ylabel('Frequency')
        
        for i, count in enumerate(counts):
            pct = count / len(all_diffs) * 100
            plt.text(i, count, f"{pct:.1f}%", ha='center', va='bottom')
            
        plt.savefig(os.path.join(OUTPUT_DIR, "boundary_displacement.png"))
        plt.close()
        
        exact = sum(1 for d in all_diffs if d == 0)
        pm1 = sum(1 for d in all_diffs if d <= 1)
        pm2 = sum(1 for d in all_diffs if d <= 2)
        pm3 = sum(1 for d in all_diffs if d <= 3)
        gt3 = sum(1 for d in all_diffs if d > 3)
        total_b = len(all_diffs)
        
        with open(stats_md, 'a') as f:
            f.write("\n## Boundary Analysis\n")
            f.write(f"* Exact Match (0 frames): {exact/total_b*100:.1f}%\n")
            f.write(f"* Within 1 frame: {pm1/total_b*100:.1f}%\n")
            f.write(f"* Within 2 frames: {pm2/total_b*100:.1f}%\n")
            f.write(f"* Within 3 frames: {pm3/total_b*100:.1f}%\n")
            f.write(f"* Greater than 3 frames: {gt3/total_b*100:.1f}%\n\n")
            f.write(f"* Max displacement: {np.max(all_diffs)} frames\n")
            f.write(f"* Mean displacement: {np.mean(all_diffs):.2f} frames\n")
            f.write(f"* Median displacement: {np.median(all_diffs):.2f} frames\n")
            f.write(f"* 95th percentile: {np.percentile(all_diffs, 95):.2f} frames\n")
            
    # 4. Intonation Confusion
    total_phrases = 0
    matches = 0
    disagreements = []
    
    for r in results:
        total_phrases += r['total_phrases']
        matches += r['match_count_ref_fast']
        for p in r['intonation_results']:
            if p['ref_label'] != p['fast_label']:
                disagreements.append(p)
                
    with open(os.path.join(OUTPUT_DIR, "intonation_validation.md"), 'w') as f:
        f.write("# Intonation Pipeline Validation (Ref vs Fast)\n\n")
        if total_phrases > 0:
            f.write(f"**Overall Agreement**: {matches/total_phrases*100:.2f}% ({matches}/{total_phrases} phrases)\n\n")
        
        if disagreements:
            f.write("## Disagreements Log\n")
            f.write("| File | Phrase | Ref Slope | Fast Slope | Ref Label | Fast Label |\n")
            f.write("|---|---|---|---|---|---|\n")
            for d in disagreements:
                f.write(f"| {d['filename']} | {d['phrase_text']} | {d['ref_slope']:.4f} | {d['fast_slope']:.4f} | {d['ref_label']} | {d['fast_label']} |\n")
        else:
            f.write("## Disagreements Log\n*No disagreements found. 100% downstream intonation agreement!*\n")

if __name__ == "__main__":
    service = whistress_service.WhiStressService()
    service.load_model()
    
    files = glob.glob("*.wav")
    results = []
    
    for f in files:
        res = process_file(f, service)
        if res:
            results.append(res)
            
    generate_reports(results)
    print("\nBenchmark complete! Results saved to stylization_validation/")
