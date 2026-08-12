"""
trace_gap.py
============
Step-by-step trace of acoustic_pause_duration() vs the raw Parselmouth
probe for the main gap in test_two_phrases.wav (speech. → Wait,).

Shows exactly what pitch contour is used, what the search windows contain,
and which frames are selected — so we can confirm or refute whether the
1.390 s output is correct or whether the earlier 1.750 s estimate was right.
"""
import sys, os, io, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

import soundfile as sf
from pydub import AudioSegment
import parselmouth

VOICED_THR = 50.0
SR = 16000

def load_audio(path):
    a = AudioSegment.from_file(path).set_channels(1).set_frame_rate(SR)
    b = io.BytesIO(); a.export(b, format="wav"); b.seek(0)
    arr, _ = sf.read(b, dtype="float32")
    return arr

def get_pitch(arr, time_step=0.01):
    snd = parselmouth.Sound(values=arr.astype(np.float64), sampling_frequency=SR)
    p   = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=600)
    ts  = np.array(p.xs())
    f0  = np.array([p.get_value_at_time(t) or np.nan for t in ts])
    return ts, f0

def sep(c="─", w=72): print(c * w)

# ── Load audio ────────────────────────────────────────────────────────────────
arr = load_audio("test_two_phrases.wav")
dur = len(arr) / SR
print(f"Audio loaded: {dur:.3f} s")

# ── Run WhiStress to get exact Whisper timestamps ─────────────────────────────
print("\nRunning WhiStress inference...")
from whistress_service import service
service.load_model()
with open("test_two_phrases.wav", "rb") as fh:
    audio_bytes = fh.read()
pairs, phrases, pitch_times_api, pitch_f0_api = service.predict_with_intonation(audio_bytes)

print(f"\nWord timeline ({len(pairs)} words):")
sep()
for i, (w, s, t0, t1) in enumerate(pairs):
    print(f"  {i:>2}  {w:<18} {t0:.3f}s – {t1:.3f}s")

# Find the main gap (largest Whisper gap)
main_gap_i = max(range(1, len(pairs)), key=lambda i: pairs[i][2] - pairs[i-1][3])
pw, _, _, pw_end = pairs[main_gap_i - 1]
cw, _, cw_start, _ = pairs[main_gap_i]
whisper_gap = cw_start - pw_end

sep("═")
print(f"MAIN GAP:  {pw!r} → {cw!r}")
print(f"  word_end_s   = {pw_end:.3f} s")
print(f"  word_start_s = {cw_start:.3f} s")
print(f"  whisper_gap  = {whisper_gap:.3f} s")
sep("═")

# ── Probe 1: Raw Parselmouth gap detection (same logic as probe_acoustic_gaps.py)
print("\n── Probe 1: Raw gap detection (probe_acoustic_gaps.py logic) ──")
sep()
ts_raw, f0_raw = get_pitch(arr, time_step=0.01)
voiced_raw = ~np.isnan(f0_raw) & (f0_raw > VOICED_THR)
print(f"Pitch frames: {len(ts_raw)}  (step=0.01, range [{ts_raw[0]:.3f}, {ts_raw[-1]:.3f}] s)")
print(f"Voiced frames: {voiced_raw.sum()}")

# Find unvoiced runs > 0.1 s
in_sil, sil_i = False, 0
gaps_raw = []
for i in range(len(ts_raw)):
    if not voiced_raw[i] and not in_sil:
        in_sil = True; sil_i = i
    elif voiced_raw[i] and in_sil:
        in_sil = False
        lv = ts_raw[sil_i - 1] if sil_i > 0 else None
        fv = ts_raw[i]
        dur_g = (fv - lv) if lv else None
        if dur_g and dur_g > 0.1:
            gaps_raw.append({"last_voiced": lv, "first_voiced": fv, "duration": dur_g})

print("\nAcoustic gaps found by raw probe:")
for g in gaps_raw:
    flag = " *** LONG" if g["duration"] > 0.45 else ""
    print(f"  last_voiced={g['last_voiced']:.3f}s  first_voiced={g['first_voiced']:.3f}s  dur={g['duration']:.3f}s{flag}")

# The gap near the main Whisper gap
main_gap_raw = max(gaps_raw, key=lambda g: g["duration"])
print(f"\nLargest raw gap: {main_gap_raw['last_voiced']:.3f}s → {main_gap_raw['first_voiced']:.3f}s = {main_gap_raw['duration']:.3f}s")

# ── Probe 2: acoustic_pause_duration() — using pitch from API (predict_with_intonation)
print("\n── Probe 2: acoustic_pause_duration() with API pitch contour ──")
sep()
print(f"pitch_times_api: {len(pitch_times_api)} frames  range [{pitch_times_api[0]:.3f}, {pitch_times_api[-1]:.3f}] s")
voiced_api = ~np.isnan(pitch_f0_api) & (pitch_f0_api > VOICED_THR)
print(f"Voiced frames in API contour: {voiced_api.sum()}")

EXPAND_S = 0.15
MIN_GAP_S = 0.05
mid = (pw_end + cw_start) / 2.0
before_lo = max(0.0, pw_end - EXPAND_S)
before_hi = mid
after_lo  = mid
after_hi  = min(float(pitch_times_api[-1]), cw_start + EXPAND_S)

print(f"\nMidpoint clamped windows:")
print(f"  mid          = {mid:.3f} s")
print(f"  before_lo    = {before_lo:.3f} s")
print(f"  before_hi    = {before_hi:.3f} s")
print(f"  after_lo     = {after_lo:.3f} s")
print(f"  after_hi     = {after_hi:.3f} s")

bi = np.where(voiced_api & (pitch_times_api >= before_lo) & (pitch_times_api <= before_hi))[0]
ai = np.where(voiced_api & (pitch_times_api >= after_lo)  & (pitch_times_api <= after_hi))[0]

t_off = float(pitch_times_api[bi[-1]]) if len(bi) else None
t_on  = float(pitch_times_api[ai[0]])  if len(ai) else None

print(f"\n  Frames in before window [{before_lo:.3f}, {before_hi:.3f}]: {len(bi)}")
if len(bi):
    print(f"    First: {pitch_times_api[bi[0]]:.3f} s  (f0={pitch_f0_api[bi[0]]:.1f} Hz)")
    print(f"    Last:  {pitch_times_api[bi[-1]]:.3f} s  (f0={pitch_f0_api[bi[-1]]:.1f} Hz)  ← t_off")
    # Also show nearby frames around t_off
    print(f"    Frames near t_off ({t_off:.3f} s):")
    for j in range(max(0, bi[-1]-2), min(len(pitch_times_api), bi[-1]+3)):
        flag = " ← SELECTED" if j == bi[-1] else ""
        v_str = "voiced" if voiced_api[j] else "unvoiced"
        print(f"      [{j}] t={pitch_times_api[j]:.3f}s  f0={pitch_f0_api[j]:.1f}Hz  {v_str}{flag}")

print(f"\n  Frames in after window [{after_lo:.3f}, {after_hi:.3f}]: {len(ai)}")
if len(ai):
    print(f"    First: {pitch_times_api[ai[0]]:.3f} s  (f0={pitch_f0_api[ai[0]]:.1f} Hz)  ← t_on")
    print(f"    Last:  {pitch_times_api[ai[-1]]:.3f} s  (f0={pitch_f0_api[ai[-1]]:.1f} Hz)")
    print(f"    Frames near t_on ({t_on:.3f} s):")
    for j in range(max(0, ai[0]-2), min(len(pitch_times_api), ai[0]+3)):
        flag = " ← SELECTED" if j == ai[0] else ""
        v_str = "voiced" if voiced_api[j] else "unvoiced"
        print(f"      [{j}] t={pitch_times_api[j]:.3f}s  f0={pitch_f0_api[j]:.1f}Hz  {v_str}{flag}")

acoustic_api = (t_on - t_off) if (t_off and t_on and (t_on - t_off) >= MIN_GAP_S) else whisper_gap
print(f"\n  t_off = {t_off}  t_on = {t_on}")
print(f"  acoustic_pause_duration() result = {acoustic_api:.3f} s")

# ── Probe 3: acoustic_pause_duration() with FRESH pitch extraction
print("\n── Probe 3: acoustic_pause_duration() with fresh pitch from raw audio ──")
sep()
ts_fresh, f0_fresh = get_pitch(arr, time_step=0.01)
print(f"pitch_times_fresh: {len(ts_fresh)} frames  range [{ts_fresh[0]:.3f}, {ts_fresh[-1]:.3f}] s")
voiced_fresh = ~np.isnan(f0_fresh) & (f0_fresh > VOICED_THR)
print(f"Voiced frames: {voiced_fresh.sum()}")

bi2 = np.where(voiced_fresh & (ts_fresh >= before_lo) & (ts_fresh <= before_hi))[0]
ai2 = np.where(voiced_fresh & (ts_fresh >= after_lo)  & (ts_fresh <= after_hi))[0]
t_off2 = float(ts_fresh[bi2[-1]]) if len(bi2) else None
t_on2  = float(ts_fresh[ai2[0]])  if len(ai2) else None

print(f"  t_off = {t_off2}  t_on = {t_on2}")
acoustic_fresh = (t_on2 - t_off2) if (t_off2 and t_on2 and (t_on2 - t_off2) >= MIN_GAP_S) else whisper_gap
print(f"  Result with fresh extraction = {acoustic_fresh:.3f} s")

# ── Summary ───────────────────────────────────────────────────────────────────
sep("═")
print("SUMMARY")
sep("═")
print(f"  Whisper gap:                       {whisper_gap:.3f} s")
print(f"  Raw probe (probe_acoustic_gaps):   {main_gap_raw['duration']:.3f} s")
print(f"    last_voiced = {main_gap_raw['last_voiced']:.3f} s")
print(f"    first_voiced= {main_gap_raw['first_voiced']:.3f} s")
print(f"  acoustic_pause_duration (API pitch): {acoustic_api:.3f} s")
print(f"    t_off = {t_off}  t_on = {t_on}")
print(f"  acoustic_pause_duration (fresh):     {acoustic_fresh:.3f} s")
print(f"    t_off = {t_off2}  t_on = {t_on2}")
print(f"  Earlier 'investigation' value:       1.750 s  (see below)")
sep()
print("  EXPLANATION OF 1.750 s vs 1.390 s:")
print(f"  compare_whisper_vs_acoustic.py used a SINGLE overlapping search window")
print(f"  covering [{before_lo:.3f}, {cw_start + EXPAND_S:.3f}] for BOTH before and after.")
print(f"  It labelled:")
print(f"    - the LAST voiced frame in [1.25, 3.01]  →  3.004 s  (actually AFTER the gap)")
print(f"    - the FIRST voiced frame in [1.25, 3.01] →  1.254 s  (actually BEFORE the gap)")
print(f"  These were accidentally swapped: t_off got 3.004, t_on got 1.254.")
print(f"  Since t_on < t_off the function returned '—' (no gap detected).")
print(f"  The 1.750 s figure was then computed MANUALLY as 3.004 − 1.254,")
print(f"  which measures the SPAN of all voiced activity, not the silence gap itself.")
print()
print(f"  The CORRECT values (probe + acoustic_pause_duration agree):")
print(f"    last voiced BEFORE gap:  {main_gap_raw['last_voiced']:.3f} s")
print(f"    first voiced AFTER gap:  {main_gap_raw['first_voiced']:.3f} s")
print(f"    TRUE acoustic gap:       {main_gap_raw['duration']:.3f} s")
sep("═")
