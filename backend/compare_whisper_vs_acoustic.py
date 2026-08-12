"""
compare_whisper_vs_acoustic.py
==============================
Step 1 + Step 2 of the investigation:

For each test file:
  - Run full WhiStress inference to get word timestamps (Pipeline A)
  - Run Parselmouth pitch extraction to get acoustic silence (Pipeline B)
  - Print a side-by-side comparison for every inter-word gap

Also captures RAW token timestamps before merge_stressed_tokens
so we can isolate whether the Alpha Rule / merge shifts end-times.

Run:
    python compare_whisper_vs_acoustic.py
"""

import sys, os, io, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

import soundfile as sf
from pydub import AudioSegment
import parselmouth
import torch
from torch.nn import functional as F

VOICED_THR            = 50.0   # Hz
PAUSE_THRESHOLD_FRONT = 0.20   # seconds — TranscriptView.jsx threshold
SEARCH_EXPAND_S       = 0.15   # expand window each side for acoustic search


def load_audio(path):
    a = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
    b = io.BytesIO(); a.export(b, format="wav"); b.seek(0)
    arr, _ = sf.read(b, dtype="float32")
    return arr, 16000


def get_pitch(arr, sr=16000):
    snd = parselmouth.Sound(values=arr.astype(np.float64), sampling_frequency=sr)
    p   = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    ts  = np.array(p.xs())
    f0  = np.array([p.get_value_at_time(t) or np.nan for t in ts])
    return ts, f0


def acoustic_silence_near(t0, t1, ts, f0, expand=SEARCH_EXPAND_S):
    """
    Given a Whisper-reported gap [t0, t1], expand the search window
    slightly and measure the true acoustic silence:
        last voiced frame before the gap  → t_off
        first voiced frame after the gap  → t_on
    Returns (t_off, t_on, acoustic_duration_s).
    """
    voiced = ~np.isnan(f0) & (f0 > VOICED_THR)
    lo = max(0, t0 - expand)
    hi = min(ts[-1], t1 + expand)

    # Last voiced frame before t1 (within lo…t1+expand)
    before_mask = voiced & (ts <= t1 + expand) & (ts >= lo)
    before_idx  = np.where(before_mask)[0]
    t_off = float(ts[before_idx[-1]]) if len(before_idx) else None

    # First voiced frame after t0 (within t0-expand…hi)
    after_mask = voiced & (ts >= t0 - expand) & (ts <= hi)
    after_idx  = np.where(after_mask)[0]
    t_on = float(ts[after_idx[0]]) if len(after_idx) else None

    if t_off is None or t_on is None or t_on <= t_off:
        return t_off, t_on, None
    return t_off, t_on, t_on - t_off


def run_whistress_with_tokens(audio_path):
    """
    Returns:
        words  – merged word list [(word, stress, start, end), ...]
        tokens – raw token list   [(token, stress, start, end), ...]
    """
    from whistress_service import service
    from WhiStress.whistress.inference_client.utils import (
        prepare_audio, get_word_emphasis_pairs, merge_stressed_tokens
    )

    service.load_model()
    arr, sr = load_audio(audio_path)
    audio_dict = {"array": arr, "sampling_rate": sr}
    audio_prep = prepare_audio(audio_dict)

    model  = service.client.whistress
    device = service.device

    input_features = model.processor.feature_extractor(
        audio_prep, sampling_rate=16000, return_tensors="pt"
    )["input_features"]
    out = model.generate_dual(input_features=input_features.to(device))
    emphasis_probs = F.softmax(out.logits, dim=-1)
    emphasis_preds = torch.argmax(emphasis_probs, dim=-1)
    shifted = torch.cat((emphasis_preds[:, -1:], emphasis_preds[:, :-1]), dim=1)
    tok_ts  = out.token_timestamps[0] if out.token_timestamps is not None else None

    raw_tokens = get_word_emphasis_pairs(
        out.preds[0], shifted[0], tok_ts, model.processor,
        filter_special_tokens=True
    )
    words = merge_stressed_tokens(raw_tokens)
    words = [(w.strip(), s, st, en) for w, s, st, en in words]
    return words, raw_tokens


def sep(c="─", w=72): print(c * w)


def analyse(audio_path, ground_truth_s=None):
    print()
    sep("═")
    print(f"  FILE:  {os.path.basename(audio_path)}")
    sep("═")

    # Pipeline A — WhiStress
    print("  [A] Running WhiStress inference + capturing raw tokens...")
    words, raw_tokens = run_whistress_with_tokens(audio_path)

    # Pipeline B — Acoustic pitch
    print("  [B] Extracting pitch (Parselmouth)...")
    arr, sr = load_audio(audio_path)
    ts, f0  = get_pitch(arr, sr)

    # ── Word timeline ─────────────────────────────────────────────────────────
    print(f"\n  Word timeline ({len(words)} words):")
    sep()
    print(f"  {'#':>3}  {'Word':<18} {'Start':>7}  {'End':>7}  {'St':>3}")
    sep()
    for i, (w, s, t0, t1) in enumerate(words):
        print(f"  {i:>3}  {w:<18} {t0:>7.3f}s {t1:>7.3f}s  {'●' if s else '·'}")

    # ── Raw token timeline ────────────────────────────────────────────────────
    print(f"\n  Raw token timeline ({len(raw_tokens)} tokens, BEFORE merge_stressed_tokens):")
    sep()
    print(f"  {'#':>3}  {'Token':<22} {'Start':>7}  {'End':>7}  {'St':>3}")
    sep()
    for i, (t, s, t0, t1) in enumerate(raw_tokens):
        print(f"  {i:>3}  {repr(t):<22} {t0:>7.3f}s {t1:>7.3f}s  {'●' if s else '·'}")

    # ── Step 1 & 2: Gap comparison ────────────────────────────────────────────
    print(f"\n  Gap-by-gap comparison (Whisper vs. Acoustic):")
    discrepancies   = []
    missed_by_whisper = []

    for i in range(1, len(words)):
        pw, ps, p0, p1 = words[i - 1]
        cw, cs, c0, c1 = words[i]
        whisper_gap = c0 - p1

        t_off, t_on, acoustic = acoustic_silence_near(p1, c0, ts, f0)
        diff = (acoustic - whisper_gap) if acoustic is not None else None

        badge_w = "BADGE" if whisper_gap > PAUSE_THRESHOLD_FRONT else "     "
        badge_a = "BADGE" if (acoustic or 0) > PAUSE_THRESHOLD_FRONT else "     "

        sep()
        print(f"  Gap #{i:>2}:  \"{pw}\"  →  \"{cw}\"")
        print(f"    Whisper   {pw!r:<16} .end   = {p1:.3f} s")
        print(f"    Whisper   {cw!r:<16} .start = {c0:.3f} s")
        print(f"    Whisper   gap                = {whisper_gap:+.3f} s  [{badge_w}]")
        if t_off: print(f"    Acoustic  last voiced         = {t_off:.3f} s")
        if t_on:  print(f"    Acoustic  first voiced        = {t_on:.3f} s")
        if acoustic is not None:
            print(f"    Acoustic  silence              = {acoustic:+.3f} s  [{badge_a}]")
            print(f"    Difference (acoustic-Whisper)  = {diff:+.3f} s")
            if abs(diff) > 0.05:
                discrepancies.append((i, pw, cw, whisper_gap, acoustic, diff))
            if whisper_gap <= PAUSE_THRESHOLD_FRONT and acoustic > PAUSE_THRESHOLD_FRONT:
                missed_by_whisper.append((i, pw, cw, whisper_gap, acoustic))
        else:
            print("    Acoustic  silence              = —  (no gap found in pitch)")

    # ── Alpha Rule inspection ─────────────────────────────────────────────────
    print(f"\n  Alpha Rule inspection (merge end-time shifts):")
    sep()
    print(f"  {'Token':<22} {'Tok.end':>8}  {'Word':<18} {'Merged.end':>11}  Shift")
    sep()
    for i in range(1, len(words)):
        pw, ps, p0, p1 = words[i - 1]
        # Find last token that is part of this word
        tok_end = None
        for tok, ts2, t_start, t_end in raw_tokens:
            if tok.strip() and tok.strip() in pw.replace(" ", ""):
                tok_end = t_end
        if tok_end is not None:
            shift = p1 - tok_end
            flag = " *** SHIFTED" if abs(shift) > 0.02 else ""
            print(f"  {repr(pw):<22} {tok_end:>8.3f}s  {pw:<18} {p1:>10.3f}s  {shift:+.3f}s{flag}")

    # ── Step 3: Root cause ────────────────────────────────────────────────────
    print(f"\n  Step 3 — Root Cause:")
    sep()
    if not discrepancies:
        print("  No significant discrepancies (all gaps agree within 50 ms).")
        print("  Conclusion: Whisper timestamps are accurate for this recording.")
    else:
        diffs = [d[5] for d in discrepancies]
        avg   = sum(diffs) / len(diffs)
        print(f"  Discrepancies in {len(discrepancies)} gap(s).  Avg diff = {avg:+.3f} s")
        if avg > 0.05:
            print("  DIRECTION: Acoustic silence > Whisper gap.")
            print("  LIKELY CAUSE:")
            print("    1. Whisper next-word .start is set EARLIER than actual voiced onset")
            print("       (Whisper 'anticipates' the next word before speech begins).")
            print("    2. OR: Whisper prev-word .end is set LATER than speech cessation")
            print("       (token claims trailing silence as part of the word).")
        if missed_by_whisper:
            print(f"\n  MISSED PAUSES (acoustic > {PAUSE_THRESHOLD_FRONT}s but Whisper gap <= {PAUSE_THRESHOLD_FRONT}s):")
            for idx, pw, cw, wg, ac in missed_by_whisper:
                print(f"    Gap #{idx}: \"{pw}\"→\"{cw}\"  Whisper={wg:.3f}s  Acoustic={ac:.3f}s")

    # ── Ground truth ──────────────────────────────────────────────────────────
    if ground_truth_s and words:
        print(f"\n  Ground truth comparison (GT = {ground_truth_s:.3f} s):")
        sep()
        largest = max(range(1, len(words)), key=lambda i: words[i][2] - words[i-1][3])
        pw, ps, p0, p1 = words[largest - 1]
        cw, cs, c0, c1 = words[largest]
        wg = c0 - p1
        _, _, ac = acoustic_silence_near(p1, c0, ts, f0)
        print(f"  Largest gap: \"{pw}\" → \"{cw}\"")
        print(f"  Ground truth:  {ground_truth_s:.3f} s")
        print(f"  Whisper gap:   {wg:.3f} s   error = {wg - ground_truth_s:+.3f} s")
        if ac: print(f"  Acoustic:      {ac:.3f} s   error = {ac - ground_truth_s:+.3f} s")

    return {"words": words, "discrepancies": discrepancies, "missed_by_whisper": missed_by_whisper}


# ── Summary table ─────────────────────────────────────────────────────────────
def summary_table(results):
    sep("═", 88)
    print("  SUMMARY TABLE")
    sep("═", 88)
    h = f"  {'File':<26} {'Words':>5} {'Discrepancies':>14} {'Missed pauses':>14} {'Avg diff':>9}"
    print(h)
    sep("─", 88)
    for fname, r in results:
        nd = len(r["discrepancies"])
        nm = len(r["missed_by_whisper"])
        diffs = [d[5] for d in r["discrepancies"]]
        avg = f"{sum(diffs)/len(diffs):+.3f} s" if diffs else "   —   "
        print(f"  {os.path.basename(fname):<26} {len(r['words']):>5} {nd:>14} {nm:>14} {avg:>9}")
    sep("═", 88)


if __name__ == "__main__":
    files = [
        ("test_two_phrases.wav",    None),
        ("test_synthetic_pause.wav", None),
        ("test_pause.mp3",           None),
    ]
    results = []
    for path, gt in files:
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue
        r = analyse(path, ground_truth_s=gt)
        results.append((path, r))

    if len(results) > 1:
        print()
        summary_table(results)

    sep("═")
    print("  Investigation complete.")
    sep("═")
