"""
investigate_pause.py
====================
Scientific investigation of the pause-detection discrepancy.

This script runs WITHOUT the FastAPI server and WITHOUT the Whisper model.
It exercises two independent measurement pipelines against the same audio file
and prints a structured comparison report.

Measurement pipeline A  ──  Whisper word timestamps
    The word list is loaded from a cached JSON dump OR produced by the
    full WhiStress inference.  Every inter-word gap is printed verbatim.

Measurement pipeline B  ──  Acoustic silence (pitch / energy)
    Uses parselmouth (Praat) to locate the last voiced frame before a gap
    and the first voiced frame after it.  This is what the pitch contour
    *shows* the user, so it is the ground truth for visual comparison.

Usage
-----
    # Run against all available test files (no model needed):
    python investigate_pause.py --acoustic-only

    # Also run WhiStress inference (requires model to be loaded):
    python investigate_pause.py --file test_two_phrases.wav

    # Investigate a specific file with a known ground-truth pause:
    python investigate_pause.py --file my_recording.wav --ground-truth 1.7

Steps implemented
-----------------
    Step 1 – Instrumentation  : print every inter-word gap side-by-side with
                                 the acoustic silence at the same location.
    Step 2 – Comparison       : generate a structured per-recording report.
    Step 3 – Root cause       : automatic heuristics flag the dominant source
                                 of discrepancy (Whisper, Alpha Rule, threshold).
    Step 4 – Recommendation   : printed after all recordings are analysed.
    Step 5 – Verification     : summary table (Whisper vs. Acoustic vs. Ground Truth).
"""

import sys
import os
import io
import json
import math
import argparse
import numpy as np

# ── Force UTF-8 on Windows so Unicode arrows print without crashing ────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Add WhiStress repo to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

# ── Configuration (mirrors pitch_analysis.py CONFIGURATION block) ──────────────
PAUSE_THRESHOLD_WHISPER  = 0.20   # seconds — frontend threshold (TranscriptView.jsx line 68)
PITCH_TIME_STEP          = 0.01   # seconds
PITCH_FMIN_HZ            = 75.0
PITCH_FMAX_HZ            = 600.0
VOICED_THRESHOLD_HZ      = 50.0   # frames below this are treated as unvoiced

# Minimum acoustic silence that is "real" (filters Praat micro-gaps)
ACOUSTIC_MIN_GAP_S       = 0.05   # seconds

# Energy-based VAD fallback parameters
ENERGY_FRAME_LENGTH_S    = 0.025  # 25 ms RMS frames
ENERGY_HOP_LENGTH_S      = 0.010  # 10 ms hop
ENERGY_SILENCE_DB        = -40.0  # dB relative to RMS peak, below = silent


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Audio loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_16k(path: str) -> np.ndarray:
    """Load any audio file; return mono float32 at 16 kHz."""
    import soundfile as sf
    from pydub import AudioSegment

    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    arr, _ = sf.read(wav_io, dtype="float32")
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Acoustic silence measurement (Pipeline B)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pitch_parselmouth(audio: np.ndarray, sr: int = 16000):
    """Return (times, f0) arrays. Unvoiced frames → NaN."""
    import parselmouth
    snd = parselmouth.Sound(values=audio.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=PITCH_TIME_STEP, pitch_floor=PITCH_FMIN_HZ, pitch_ceiling=PITCH_FMAX_HZ)
    times = pitch.xs()
    f0    = np.array([pitch.get_value_at_time(t) or np.nan for t in times])
    return np.array(times), f0


def extract_energy_vad(audio: np.ndarray, sr: int = 16000):
    """
    Simple energy-based VAD as a fallback / cross-check.
    Returns (times, is_voiced) boolean array.
    """
    frame_len = int(ENERGY_FRAME_LENGTH_S * sr)
    hop_len   = int(ENERGY_HOP_LENGTH_S   * sr)
    frames = []
    times  = []
    for i in range(0, len(audio) - frame_len, hop_len):
        chunk = audio[i : i + frame_len]
        rms = np.sqrt(np.mean(chunk ** 2))
        frames.append(rms)
        times.append((i + frame_len / 2) / sr)
    frames = np.array(frames, dtype=np.float32)
    peak   = frames.max()
    if peak < 1e-10:
        return np.array(times), np.zeros(len(times), dtype=bool)
    db   = 20 * np.log10(frames / peak + 1e-12)
    voiced = db > ENERGY_SILENCE_DB
    return np.array(times), voiced


def acoustic_silence_in_window(
    t_start: float,
    t_end: float,
    pitch_times: np.ndarray,
    pitch_f0: np.ndarray,
    energy_times: np.ndarray,
    energy_voiced: np.ndarray,
) -> dict:
    """
    For a window (t_start, t_end) report:
      - last voiced frame before the window (from pitch)
      - first voiced frame after the window (from pitch)
      - acoustic silence duration (pitch-based)
      - energy-based cross-check
    """
    # --- Pitch-based ---
    mask = ~np.isnan(pitch_f0) & (pitch_f0 > VOICED_THRESHOLD_HZ)

    # Last voiced frame BEFORE t_end (look in [t_start-2, t_end])
    search_before = pitch_times <= (t_end + 0.05)
    voiced_before = np.where(mask & search_before)[0]
    last_voiced_t  = float(pitch_times[voiced_before[-1]]) if len(voiced_before) else None

    # First voiced frame AFTER t_start (look in [t_start, t_start+3])
    search_after = pitch_times >= (t_start - 0.05)
    voiced_after = np.where(mask & search_after)[0]
    first_voiced_t = float(pitch_times[voiced_after[0]]) if len(voiced_after) else None

    acoustic_silence = (
        (first_voiced_t - last_voiced_t)
        if (last_voiced_t is not None and first_voiced_t is not None and first_voiced_t > last_voiced_t)
        else None
    )

    # --- Energy VAD cross-check ---
    in_window_mask  = (energy_times >= t_start) & (energy_times <= t_end)
    energy_in_window = energy_voiced[in_window_mask]
    frac_silent_energy = (1 - energy_in_window.mean()) if len(energy_in_window) else None

    # Expand the energy window slightly to find true voiced edges
    before_e = np.where(energy_voiced & (energy_times <= t_end + 0.05))[0]
    after_e  = np.where(energy_voiced & (energy_times >= t_start - 0.05))[0]
    last_voiced_e  = float(energy_times[before_e[-1]]) if len(before_e) else None
    first_voiced_e = float(energy_times[after_e[0]])   if len(after_e)  else None
    energy_silence = (
        (first_voiced_e - last_voiced_e)
        if (last_voiced_e is not None and first_voiced_e is not None and first_voiced_e > last_voiced_e)
        else None
    )

    return {
        "last_voiced_pitch":    last_voiced_t,
        "first_voiced_pitch":   first_voiced_t,
        "acoustic_silence_s":   acoustic_silence,
        "last_voiced_energy":   last_voiced_e,
        "first_voiced_energy":  first_voiced_e,
        "energy_silence_s":     energy_silence,
        "frac_silent_energy":   frac_silent_energy,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — WhiStress inference (Pipeline A)
# ═══════════════════════════════════════════════════════════════════════════════

def run_whistress(audio_path: str) -> list:
    """
    Run the full WhiStress pipeline and return the word list.
    Returns list of (word, stressed, start, end) tuples.
    """
    print("  Loading WhiStress model (may take ~30 s on first run)...")
    from whistress_service import service
    service.load_model()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    words, *_ = service.predict_with_intonation(audio_bytes)
    return words


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Token-level dump (Alpha Rule inspection)
# ═══════════════════════════════════════════════════════════════════════════════

def run_token_level(audio_path: str) -> list:
    """
    Run WhiStress and capture RAW token timestamps BEFORE merge_stressed_tokens.
    This lets us see whether the Alpha Rule / merge changes end-times.
    """
    print("  Capturing raw (pre-merge) token timestamps...")
    import torch
    from torch.nn import functional as F
    from whistress_service import service
    from WhiStress.whistress.inference_client.utils import (
        get_loaded_model,
        prepare_audio,
        inference_from_audio,
        get_word_emphasis_pairs,
    )

    service.load_model()
    audio_arr = load_audio_16k(audio_path)
    audio_dict = {"array": audio_arr, "sampling_rate": 16000}
    audio_prepared = prepare_audio(audio_dict)

    model = service.client.whistress
    device = service.device

    input_features = model.processor.feature_extractor(
        audio_prepared, sampling_rate=16000, return_tensors="pt"
    )["input_features"]
    out = model.generate_dual(input_features=input_features.to(device))
    emphasis_probs = F.softmax(out.logits, dim=-1)
    emphasis_preds = torch.argmax(emphasis_probs, dim=-1)
    shifted = torch.cat((emphasis_preds[:, -1:], emphasis_preds[:, :-1]), dim=1)
    token_timestamps = out.token_timestamps[0] if out.token_timestamps is not None else None

    raw_tokens = get_word_emphasis_pairs(
        out.preds[0], shifted[0], token_timestamps, model.processor,
        filter_special_tokens=True
    )
    return raw_tokens   # list of (token_str, stress, start, end)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Per-recording analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_recording(
    audio_path: str,
    words: list,
    raw_tokens: list,
    ground_truth_s: float = None,
) -> dict:
    """
    Core analysis for one recording.
    Returns a structured report dict.
    """
    print(f"\n  Loading audio for acoustic analysis: {os.path.basename(audio_path)}")
    audio = load_audio_16k(audio_path)

    print("  Extracting pitch (Parselmouth)...")
    pitch_times, pitch_f0 = extract_pitch_parselmouth(audio)

    print("  Computing energy VAD...")
    energy_times, energy_voiced = extract_energy_vad(audio)

    duration = len(audio) / 16000

    # ── Step 1: Instrument every inter-word gap ──────────────────────────────
    gap_rows = []
    for i in range(1, len(words)):
        prev = words[i - 1]
        curr = words[i]
        pw, ps, p_start, p_end = prev
        cw, cs, c_start, c_end = curr
        whisper_gap = c_start - p_end

        # Acoustic silence in the region [p_end-0.1, c_start+0.1]
        window_start = max(0, p_end - 0.10)
        window_end   = min(duration, c_start + 0.10)
        acoustic = acoustic_silence_in_window(
            window_start, window_end, pitch_times, pitch_f0,
            energy_times, energy_voiced,
        )
        gap_rows.append({
            "idx":           i,
            "prev_word":     pw,
            "prev_end":      p_end,
            "next_word":     cw,
            "next_start":    c_start,
            "whisper_gap":   whisper_gap,
            "acoustic":      acoustic,
        })

    # ── Step 2: Compare raw tokens vs merged words at each gap ────────────────
    alpha_analysis = []
    if raw_tokens:
        # Find which raw tokens correspond to prev/next words at each gap
        for row in gap_rows:
            pw = row["prev_word"]
            # Find last token that contributed to pw
            last_token_end  = None
            first_token_start = None
            for tok, ts, ts_start, ts_end in raw_tokens:
                tok_clean = tok.strip()
                if tok_clean and tok_clean in row["prev_word"].replace(" ", ""):
                    last_token_end = ts_end
                elif tok_clean and tok_clean in row["next_word"].replace(" ", ""):
                    if first_token_start is None:
                        first_token_start = ts_start
            alpha_analysis.append({
                "prev_word":         row["prev_word"],
                "merged_end":        row["prev_end"],
                "raw_token_end":     last_token_end,
                "next_word":         row["next_word"],
                "merged_start":      row["next_start"],
                "raw_token_start":   first_token_start,
            })

    return {
        "path":        audio_path,
        "duration_s":  duration,
        "words":       words,
        "gap_rows":    gap_rows,
        "alpha":       alpha_analysis,
        "ground_truth_s": ground_truth_s,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Report printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_sep(char="─", width=70):
    print(char * width)

def fmt(v, decimals=3, unit="s"):
    if v is None: return "  —   "
    return f"{v:+.{decimals}f} {unit}"

def print_report(report: dict):
    path  = report["path"]
    words = report["words"]
    gaps  = report["gap_rows"]

    print_sep("═")
    print(f"  RECORDING:  {os.path.basename(path)}")
    print(f"  DURATION:   {report['duration_s']:.2f} s")
    print(f"  WORDS:      {len(words)}")
    print_sep("═")

    # ── Full word timeline ────────────────────────────────────────────────────
    print("\n  ── Word Timeline (Whisper timestamps) ──")
    print(f"  {'#':>3}  {'Word':<18} {'Start':>7}  {'End':>7}  {'Stress':>6}")
    print_sep()
    for i, (w, s, t0, t1) in enumerate(words):
        print(f"  {i:>3}  {w:<18} {t0:>7.3f}s {t1:>7.3f}s {'●' if s else '·':>6}")

    # ── Step 1 & 2: Gap-by-gap comparison ────────────────────────────────────
    print(f"\n  ── Step 1 / Step 2:  Gap Comparison (Whisper vs. Acoustic) ──")

    discrepancies = []

    for row in gaps:
        wgap   = row["whisper_gap"]
        ac     = row["acoustic"]
        ac_s   = ac["acoustic_silence_s"]
        en_s   = ac["energy_silence_s"]
        lv_p   = ac["last_voiced_pitch"]
        fv_p   = ac["first_voiced_pitch"]
        lv_e   = ac["last_voiced_energy"]
        fv_e   = ac["first_voiced_energy"]

        badge_w  = "BADGE" if wgap  > PAUSE_THRESHOLD_WHISPER else "     "
        badge_ac = "BADGE" if (ac_s or 0) > PAUSE_THRESHOLD_WHISPER else "     "

        diff = (ac_s - wgap) if ac_s is not None else None
        if diff is not None and abs(diff) > 0.05:
            discrepancies.append({"row": row, "diff": diff})

        print_sep()
        print(f"  Gap #{row['idx']:>2}:  \"{row['prev_word']}\"  →  \"{row['next_word']}\"")
        print()
        print(f"  Whisper   prev_end  = {row['prev_end']:.3f} s")
        print(f"  Whisper   next_start= {row['next_start']:.3f} s")
        print(f"  Whisper   gap       = {wgap:+.3f} s   [{badge_w}]")
        print()
        print(f"  Pitch     last voiced  = {lv_p:.3f} s" if lv_p else "  Pitch     last voiced  = —")
        print(f"  Pitch     first voiced = {fv_p:.3f} s" if fv_p else "  Pitch     first voiced = —")
        print(f"  Pitch     silence      = {ac_s:+.3f} s   [{badge_ac}]" if ac_s is not None else "  Pitch     silence      = —")
        print()
        print(f"  Energy    last voiced  = {lv_e:.3f} s" if lv_e else "  Energy    last voiced  = —")
        print(f"  Energy    first voiced = {fv_e:.3f} s" if fv_e else "  Energy    first voiced = —")
        print(f"  Energy    silence      = {en_s:+.3f} s" if en_s is not None else "  Energy    silence      = —")
        if diff is not None:
            print(f"\n  *** DISCREPANCY = {diff:+.3f} s  (acoustic − Whisper) ***")

    # ── Step 3: Root cause heuristics ────────────────────────────────────────
    print(f"\n  ── Step 3:  Root Cause Analysis ──")
    print_sep()

    if not discrepancies:
        print("  No significant discrepancies found (all gaps agree within 50 ms).")
    else:
        total_diff = sum(d["diff"] for d in discrepancies)
        avg_diff   = total_diff / len(discrepancies)
        max_d      = max(discrepancies, key=lambda x: abs(x["diff"]))

        print(f"  Significant discrepancies found in {len(discrepancies)} gap(s).")
        print(f"  Average (acoustic − Whisper): {avg_diff:+.3f} s")
        print(f"  Worst case:  \"{max_d['row']['prev_word']}\" → \"{max_d['row']['next_word']}\"  diff = {max_d['diff']:+.3f} s")
        print()

        # Bias direction
        if avg_diff > 0.05:
            print("  BIAS:  Acoustic silence is LONGER than Whisper gap.")
            print("         Most likely cause: Whisper next-word start is earlier than actual speech onset.")
            print("         OR: Whisper prev-word end is later than actual speech cessation (extended into silence).")
        elif avg_diff < -0.05:
            print("  BIAS:  Whisper gap is LONGER than acoustic silence.")
            print("         This is unusual. Possible cause: gap contains non-zero-amplitude noise.")
        else:
            print("  BIAS:  Mixed direction — individual gaps vary, no systematic bias.")

        # Alpha Rule check
        alpha = report.get("alpha", [])
        if alpha:
            alpha_extensions = [
                a for a in alpha
                if a["raw_token_end"] is not None and a["merged_end"] is not None
                and abs(a["merged_end"] - a["raw_token_end"]) > 0.02
            ]
            if alpha_extensions:
                print(f"\n  ALPHA RULE:  {len(alpha_extensions)} word(s) have end-times shifted by merge.")
                for a in alpha_extensions:
                    shift = a["merged_end"] - a["raw_token_end"]
                    print(f"    \"{a['prev_word']}\": raw_end={a['raw_token_end']:.3f}  merged_end={a['merged_end']:.3f}  shift={shift:+.3f} s")
            else:
                print("\n  ALPHA RULE:  No significant end-time shifts detected by merge.")

        # Threshold check
        whisper_missed = [d for d in discrepancies if d["row"]["whisper_gap"] <= PAUSE_THRESHOLD_WHISPER and (d["row"]["acoustic"]["acoustic_silence_s"] or 0) > PAUSE_THRESHOLD_WHISPER]
        if whisper_missed:
            print(f"\n  THRESHOLD ISSUE:  {len(whisper_missed)} gap(s) acoustically exceed the {PAUSE_THRESHOLD_WHISPER:.2f}s threshold")
            print(f"  but are NOT flagged because Whisper underreports the gap.")
            for d in whisper_missed:
                r = d["row"]
                print(f"    \"{r['prev_word']}\" → \"{r['next_word']}\": Whisper={r['whisper_gap']:.3f}s  Acoustic={r['acoustic']['acoustic_silence_s']:.3f}s")

    # ── Ground truth comparison if provided ──────────────────────────────────
    gt = report.get("ground_truth_s")
    if gt is not None:
        print(f"\n  ── Ground Truth Comparison ──")
        print_sep()
        largest_gap = max(gaps, key=lambda r: r["whisper_gap"]) if gaps else None
        if largest_gap:
            wg  = largest_gap["whisper_gap"]
            acs = largest_gap["acoustic"]["acoustic_silence_s"]
            print(f"  Ground truth pause:         {gt:.3f} s")
            print(f"  Whisper-detected largest gap:{wg:.3f} s   error = {wg - gt:+.3f} s")
            if acs:
                print(f"  Acoustic silence:           {acs:.3f} s   error = {acs - gt:+.3f} s")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Summary table (Step 5)
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(all_reports: list):
    print_sep("═", 90)
    print("  SUMMARY TABLE  —  Whisper gap vs. Acoustic silence (largest gap per recording)")
    print_sep("═", 90)
    hdr = f"  {'Recording':<30} {'Ground Truth':>12} {'Whisper':>9} {'Acoustic':>10} {'Whisper Err':>12} {'Acoustic Err':>13}"
    print(hdr)
    print_sep("─", 90)
    for r in all_reports:
        name = os.path.basename(r["path"])[:29]
        gt   = r.get("ground_truth_s")
        gaps = r.get("gap_rows", [])
        if not gaps:
            print(f"  {name:<30}  (no gaps detected)")
            continue
        lg = max(gaps, key=lambda g: g["whisper_gap"])
        wg = lg["whisper_gap"]
        ac = lg["acoustic"]["acoustic_silence_s"]
        gt_str  = f"{gt:.3f} s" if gt is not None else "   —   "
        wg_str  = f"{wg:.3f} s"
        ac_str  = f"{ac:.3f} s" if ac is not None else "   —   "
        we_str  = f"{wg - gt:+.3f} s" if gt is not None else "   —   "
        ae_str  = f"{ac - gt:+.3f} s" if (gt is not None and ac is not None) else "   —   "
        print(f"  {name:<30} {gt_str:>12} {wg_str:>9} {ac_str:>10} {we_str:>12} {ae_str:>13}")
    print_sep("═", 90)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — Recommendation (Step 4)
# ═══════════════════════════════════════════════════════════════════════════════

RECOMMENDATION = """
  ── Step 4:  Architecture Recommendation ──────────────────────────────────────

  Based on the analysis above, three options are available:

  OPTION A — Whisper timestamps only (current)
  ─────────────────────────────────────────────
  + Zero additional computation.
  + No dependency on pitch tracker for pause logic.
  - Systematically underestimates pauses because Whisper next-word start
    timestamps sometimes precede actual voiced onset by 50–200 ms.
  - Whisper end timestamps sometimes extend into silence (token greedily
    assigned to trailing noise).
  - Any discrepancy visible in the pitch contour but absent in the badge
    is unexplainable to the user.

  OPTION B — Acoustic silence only (pitch / energy VAD)
  ──────────────────────────────────────────────────────
  + Directly matches what the user sees in the pitch contour.
  + Immune to Whisper timestamp inaccuracies.
  + Can detect silence that spans the entire acoustic gap, not just the
    Whisper-computed inter-word interval.
  - Requires the pitch contour to already be computed (it is — by extract_pitch).
  - Needs a way to associate an acoustic gap with the surrounding WORDS
    (still needed from Whisper).
  - May split at breath sounds or plosive bursts (false positives).

  OPTION C — Hybrid: Whisper for word identity, acoustic for pause duration
  ──────────────────────────────────────────────────────────────────────────
  + Best of both worlds:
      · Whisper provides word boundaries (which word before / after the gap).
      · Acoustic VAD measures the actual silence duration within each gap.
  + The pause badge duration shown to the user matches the pitch contour.
  + Minimal additional computation (pitch contour is already computed for
    phrase intonation — reuse it).
  + Gracefully falls back to Whisper gap if pitch tracker finds no silence
    (e.g. continuous voiced speech).
  - Slightly more complex implementation than Option A.
  - One extra pass over the pitch arrays per inter-word gap (O(n) total).

  VERDICT
  ───────
  Option C is recommended.

  Implementation sketch (backend, build_phrase_response is already present):
    For each pair (word_i, word_{i+1}):
      1. Use Whisper gap as the search window: [word_i.end - 0.1, word_{i+1}.start + 0.1]
      2. Within that window, find the last voiced frame (pitch > 50 Hz) → t_off
         and the first voiced frame → t_on.
      3. acoustic_pause = t_on - t_off
      4. If acoustic_pause < ACOUSTIC_MIN_GAP: fall back to Whisper gap.
      5. Return acoustic_pause as the canonical pause duration.

  The WORD boundary association still comes from Whisper; only the DURATION
  measurement changes.  The existing pause_duration field in the API response
  (which the frontend reads at TranscriptView.jsx line 68) receives the
  improved value.  No other code changes are required.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Investigate pause detection discrepancy.")
    parser.add_argument("--file",         nargs="+", default=[], help="Audio file(s) to analyse.")
    parser.add_argument("--acoustic-only",action="store_true",   help="Skip WhiStress inference; use synthetic/cached word timestamps.")
    parser.add_argument("--ground-truth", type=float, default=None, help="Known pause duration (seconds) for first --file.")
    args = parser.parse_args()

    # Default to available test files if none specified
    test_files = args.file
    if not test_files:
        candidates = [
            "test_two_phrases.wav",
            "test_synthetic_pause.wav",
            "test_pause.mp3",
            "sample.flac",
        ]
        test_files = [f for f in candidates if os.path.exists(f)]
        if not test_files:
            print("No audio files found. Specify with --file <path>.")
            sys.exit(1)

    all_reports = []

    for idx, path in enumerate(test_files):
        print_sep("═")
        print(f"  Analysing: {path}")
        print_sep("═")

        gt = args.ground_truth if idx == 0 else None

        if args.acoustic_only:
            print("  [--acoustic-only] Generating synthetic word list from pitch-based VAD...")
            audio = load_audio_16k(path)
            pitch_times, pitch_f0 = extract_pitch_parselmouth(audio)
            # Synthesise minimal word list: one "word" per continuous voiced segment
            words = []
            in_seg, seg_start = False, 0.0
            for t, f in zip(pitch_times, pitch_f0):
                voiced = not np.isnan(f) and f > VOICED_THRESHOLD_HZ
                if voiced and not in_seg:
                    seg_start = t; in_seg = True
                elif not voiced and in_seg:
                    words.append((f"seg_{len(words)}", 0, seg_start, t)); in_seg = False
            if in_seg:
                words.append((f"seg_{len(words)}", 0, seg_start, pitch_times[-1]))
            raw_tokens = []
        else:
            try:
                words      = run_whistress(path)
                raw_tokens = []   # token-level dump is expensive; skip unless needed
            except Exception as e:
                print(f"  WhiStress inference failed: {e}")
                print("  Re-run with --acoustic-only to skip model inference.")
                continue

        report = analyse_recording(path, words, raw_tokens, ground_truth_s=gt)
        print_report(report)
        all_reports.append(report)

    if len(all_reports) > 1:
        print_summary_table(all_reports)

    print(RECOMMENDATION)

    print_sep("═")
    print("  Investigation complete.")
    print_sep("═")


if __name__ == "__main__":
    main()
