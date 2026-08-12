import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Add backend to path to import production modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pitch_analysis import extract_pitch, extract_pitch_features, classify_contour
from pitch_stylization_fast import stylize_pitch_contour
from sbcsae_adapter import parse_sbcsae_trn

def plot_phrase(wav_name, iu, pitch_times, raw_f0, stylized_f0, out_path):
    plt.figure(figsize=(10, 4))
    start, end = iu['start'], iu['end']
    
    # Extract phrase slice
    mask = (pitch_times >= start) & (pitch_times <= end)
    p_times = pitch_times[mask]
    r_f0 = raw_f0[mask]
    s_f0 = stylized_f0[mask]
    
    plt.plot(p_times, r_f0, '.', color='gray', alpha=0.5, label='Raw F0')
    plt.plot(p_times, s_f0, '-', color='blue', linewidth=2, label='Stylized F0')
    
    plt.axvline(start, color='red', linestyle='--', alpha=0.5, label='IU Boundary')
    plt.axvline(end, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f"{wav_name} | IU {iu['iu_id']} | Human: {iu['human_reference_class']} | Raw: {iu['raw_prediction']} | Styl: {iu['stylized_prediction']}")
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    wav_dir = "sbc_transcripts_stream"
    trn_dir = "sbc_transcripts_stream"
    out_dir = "sbc_eval_results"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    # Process files 1 to 5
    for i in range(1, 6):
        basename = f"SBC{i:03d}"
        wav_path = os.path.join(wav_dir, f"{basename}.wav")
        trn_path = os.path.join(trn_dir, f"{basename}.trn")
        
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100000:
            print(f"Skipping {basename}: WAV not found or corrupted.")
            continue
        if not os.path.exists(trn_path):
            print(f"Skipping {basename}: TRN not found.")
            continue
            
        print(f"--- Processing {basename} ---")
        ius = parse_sbcsae_trn(trn_path)
        
        # Filter excluded IUs
        valid_ius = [iu for iu in ius if iu['human_reference_class'] != "Excluded"]
        print(f"Found {len(ius)} IUs, {len(valid_ius)} valid for evaluation.")
        
        # 1. Load full audio
        audio, sr = librosa.load(wav_path, sr=16000)
        
        # 2. Extract raw pitch on full audio (Production extraction)
        print("Extracting raw pitch...")
        pitch_times, raw_f0 = extract_pitch(audio, sr=16000)
        
        # Calculate utterance median (ignoring NaNs)
        utterance_median_hz = np.nanmedian(raw_f0)
        
        # 3. Stylize pitch on full audio (Production stylization)
        print("Stylizing pitch...")
        stylized_f0 = stylize_pitch_contour(raw_f0, P=1)
        
        # 4. Evaluate each IU using human boundaries
        for iu in tqdm(valid_ius, desc="Evaluating IUs"):
            # Raw features
            raw_feats = extract_pitch_features(
                iu['start'], iu['end'], pitch_times, raw_f0, utterance_median_hz
            )
            
            # Stylized features
            stylized_feats = extract_pitch_features(
                iu['start'], iu['end'], pitch_times, stylized_f0, utterance_median_hz
            )
            
            raw_slope = raw_feats['overall_slope']
            stylized_slope = stylized_feats['overall_slope']
            
            if raw_slope is None or stylized_slope is None:
                # Insufficient voicing
                continue
                
            raw_pred = classify_contour(raw_feats)
            stylized_pred = classify_contour(stylized_feats)
            
            # Terminal slopes
            raw_terminal = raw_feats['terminal_slope']
            styl_terminal = stylized_feats['terminal_slope']
            
            raw_terminal_pred = classify_contour({'voiced_fraction': raw_feats.get('voiced_fraction', 0), 'overall_slope': raw_terminal}) if raw_terminal is not None else "Insufficient_voicing"
            styl_terminal_pred = classify_contour({'voiced_fraction': stylized_feats.get('voiced_fraction', 0), 'overall_slope': styl_terminal}) if styl_terminal is not None else "Insufficient_voicing"
            
            iu['raw_slope'] = raw_slope
            iu['raw_prediction'] = raw_pred.capitalize()
            iu['stylized_slope'] = stylized_slope
            iu['stylized_prediction'] = stylized_pred.capitalize()
            iu['raw_terminal_slope'] = raw_terminal
            iu['raw_terminal_prediction'] = raw_terminal_pred.capitalize()
            iu['styl_terminal_slope'] = styl_terminal
            iu['styl_terminal_prediction'] = styl_terminal_pred.capitalize()
            iu['voiced_frame_count'] = int(raw_feats.get('voiced_fraction', 0) * len(pitch_times[(pitch_times >= iu['start']) & (pitch_times <= iu['end'])]))
            
            results.append({
                'audio_file': basename,
                'speaker': iu['speaker'],
                'iu_id': iu['iu_id'],
                'start': iu['start'],
                'end': iu['end'],
                'annotation_symbol': iu['annotation_symbol'],
                'human_reference_class': iu['human_reference_class'],
                'raw_slope': raw_slope,
                'raw_prediction': iu['raw_prediction'],
                'stylized_slope': stylized_slope,
                'stylized_prediction': iu['stylized_prediction'],
                'raw_terminal_slope': raw_terminal,
                'raw_terminal_prediction': iu['raw_terminal_prediction'],
                'styl_terminal_slope': styl_terminal,
                'styl_terminal_prediction': iu['styl_terminal_prediction'],
                'voiced_frame_count': iu['voiced_frame_count']
            })
            
            # Diagnostic plots (save one of each type of agreement/disagreement per file max to avoid hundreds of plots)
            # Actually, let's just save a few representative ones overall. We'll do this in a second pass.

    if not results:
        print("No valid IUs processed. Exiting.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "sbcsae_results.csv"), index=False)
    
    # Calculate metrics
    y_true = df['human_reference_class']
    y_raw = df['raw_prediction']
    y_stylized = df['stylized_prediction']
    
    labels = ["Falling", "Level", "Rising"]
    
    raw_acc = accuracy_score(y_true, y_raw)
    styl_acc = accuracy_score(y_true, y_stylized)
    
    y_raw_term = df['raw_terminal_prediction']
    y_styl_term = df['styl_terminal_prediction']
    
    raw_term_acc = accuracy_score(y_true, y_raw_term)
    styl_term_acc = accuracy_score(y_true, y_styl_term)
    
    raw_prec = precision_score(y_true, y_raw, average='macro', labels=labels, zero_division=0)
    styl_prec = precision_score(y_true, y_stylized, average='macro', labels=labels, zero_division=0)
    raw_term_prec = precision_score(y_true, y_raw_term, average='macro', labels=labels, zero_division=0)
    styl_term_prec = precision_score(y_true, y_styl_term, average='macro', labels=labels, zero_division=0)
    
    raw_rec = recall_score(y_true, y_raw, average='macro', labels=labels, zero_division=0)
    styl_rec = recall_score(y_true, y_stylized, average='macro', labels=labels, zero_division=0)
    raw_term_rec = recall_score(y_true, y_raw_term, average='macro', labels=labels, zero_division=0)
    styl_term_rec = recall_score(y_true, y_styl_term, average='macro', labels=labels, zero_division=0)
    
    raw_f1_m = f1_score(y_true, y_raw, average='macro', labels=labels)
    styl_f1_m = f1_score(y_true, y_stylized, average='macro', labels=labels)
    raw_term_f1_m = f1_score(y_true, y_raw_term, average='macro', labels=labels)
    styl_term_f1_m = f1_score(y_true, y_styl_term, average='macro', labels=labels)
    
    raw_f1_w = f1_score(y_true, y_raw, average='weighted', labels=labels)
    styl_f1_w = f1_score(y_true, y_stylized, average='weighted', labels=labels)
    raw_term_f1_w = f1_score(y_true, y_raw_term, average='weighted', labels=labels)
    styl_term_f1_w = f1_score(y_true, y_styl_term, average='weighted', labels=labels)
    
    print("\n--- Evaluation Metrics ---")
    print(f"{'Metric':<15} | {'Raw (Over)':<10} | {'Styl (Over)':<10} | {'Raw (Term)':<10} | {'Styl (Term)':<10}")
    print("-" * 65)
    print(f"{'Accuracy':<15} | {raw_acc:.4f}     | {styl_acc:.4f}     | {raw_term_acc:.4f}     | {styl_term_acc:.4f}")
    print(f"{'Precision (M)':<15} | {raw_prec:.4f}     | {styl_prec:.4f}     | {raw_term_prec:.4f}     | {styl_term_prec:.4f}")
    print(f"{'Recall (M)':<15} | {raw_rec:.4f}     | {styl_rec:.4f}     | {raw_term_rec:.4f}     | {styl_term_rec:.4f}")
    print(f"{'Macro F1':<15} | {raw_f1_m:.4f}     | {styl_f1_m:.4f}     | {raw_term_f1_m:.4f}     | {styl_term_f1_m:.4f}")
    print(f"{'Weighted F1':<15} | {raw_f1_w:.4f}     | {styl_f1_w:.4f}     | {raw_term_f1_w:.4f}     | {styl_term_f1_w:.4f}")
    
    # Confusion Matrices
    print("\n--- Raw Confusion Matrix ---")
    print(confusion_matrix(y_true, y_raw, labels=labels))
    print("\n--- Stylized Confusion Matrix ---")
    print(confusion_matrix(y_true, y_stylized, labels=labels))
    
    # Agreement combinations
    df['raw_correct'] = df['raw_prediction'] == df['human_reference_class']
    df['styl_correct'] = df['stylized_prediction'] == df['human_reference_class']
    
    both_correct = len(df[df['raw_correct'] & df['styl_correct']])
    raw_only = len(df[df['raw_correct'] & ~df['styl_correct']])
    styl_only = len(df[~df['raw_correct'] & df['styl_correct']])
    both_wrong = len(df[~df['raw_correct'] & ~df['styl_correct']])
    
    print("\n--- Agreement Tallies ---")
    print(f"Both correct: {both_correct}")
    print(f"Raw correct / Stylized incorrect: {raw_only}")
    print(f"Raw incorrect / Stylized correct: {styl_only}")
    print(f"Both incorrect: {both_wrong}")
    
    # Save a few diagnostic plots
    print("Saving diagnostic plots...")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    cats = {
        "both_correct": df[df['raw_correct'] & df['styl_correct']],
        "raw_correct": df[df['raw_correct'] & ~df['styl_correct']],
        "styl_correct": df[~df['raw_correct'] & df['styl_correct']],
        "both_incorrect": df[~df['raw_correct'] & ~df['styl_correct']]
    }
    
    for cat_name, cat_df in cats.items():
        if len(cat_df) > 0:
            sample = cat_df.sample(1).iloc[0]
            
            # Load audio/pitch just for this plot
            audio, _ = librosa.load(os.path.join(wav_dir, f"{sample['audio_file']}.wav"), sr=16000)
            p_times, r_f0 = extract_pitch(audio, sr=16000)
            s_f0 = stylize_pitch_contour(r_f0, P=1)
            
            out_path = os.path.join(plot_dir, f"{cat_name}_{sample['audio_file']}_IU{sample['iu_id']}.png")
            plot_phrase(sample['audio_file'], sample, p_times, r_f0, s_f0, out_path)

if __name__ == "__main__":
    main()
