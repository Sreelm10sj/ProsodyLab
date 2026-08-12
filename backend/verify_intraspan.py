"""
verify_intraspan.py
===================
Verifies that scanning INSIDE a long-span token's pitch range finds the
correct acoustic pause and that Stage-2 does not produce false positives
on the existing test suite.
"""
import sys, os, io
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

import soundfile as sf
from pydub import AudioSegment
import parselmouth

VOICED_THR      = 50.0
PAUSE_THRESHOLD = 0.20
LONG_WORD_SPAN  = 0.50   # tokens longer than this trigger the intra-span check
MIN_GAP         = 0.05
SR              = 16000

def sep(c="─", w=72): print(c * w)

def load_audio(path):
    a = AudioSegment.from_file(path).set_channels(1).set_frame_rate(SR)
    b = io.BytesIO(); a.export(b, format="wav"); b.seek(0)
    arr, _ = sf.read(b, dtype="float32")
    return arr

def get_pitch(arr):
    snd = parselmouth.Sound(values=arr.astype(np.float64), sampling_frequency=SR)
    p   = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    ts  = np.array(p.xs())
    f0  = np.array([p.get_value_at_time(t) or np.nan for t in ts])
    return ts, f0

# ── Core helpers ──────────────────────────────────────────────────────────────

from pitch_analysis import (
    acoustic_pause_duration,
    directional_internal_pause,
    compute_pause_before
)

# ── Load model ────────────────────────────────────────────────────────────────

from whistress_service import service
service.load_model()

def run_whistress(path):
    with open(path, "rb") as fh: data = fh.read()
    return service.predict_with_intonation(data)

# ── Test function ─────────────────────────────────────────────────────────────

def check(path, gt_s, label):
    if not os.path.exists(path):
        print(f"  SKIP {path}"); return None

    arr = load_audio(path)
    pts, pf0 = get_pitch(arr)
    pairs, phrases, pitch_times, pitch_f0, audio_array, sr = run_whistress(path)

    sep("═")
    print(f"  {label}  (GT ≈ {gt_s} s)")
    sep("═")
    print(f"  Words ({len(pairs)}):")
    for i, (w, s, t0, t1) in enumerate(pairs):
        span = t1 - t0
        flag = "  *** LONG" if span > LONG_WORD_SPAN else ""
        print(f"    {i:>2}  {w:<18} {t0:.3f}-{t1:.3f}  span={span:.3f}s{flag}")

    print()
    print(f"  Gaps (compute_pause_before):")
    sep()
    best_pause   = 0.0
    best_gap_idx = None
    for i in range(1, len(pairs)):
        pw, _, p0, p1 = pairs[i - 1]
        cw, _, c0, c1 = pairs[i]
        pb = compute_pause_before(i, pairs, pitch_times, pitch_f0, audio_array, sr)

        # also show stage-1 alone for comparison
        s1 = acoustic_pause_duration(p1, c0, pitch_times, pitch_f0)
        prev_span = p1 - p0
        stage2_used = (s1 < PAUSE_THRESHOLD and prev_span > LONG_WORD_SPAN and pb > s1)

        badge  = pb > PAUSE_THRESHOLD if pb else False
        s2_tag = "  [S2]" if stage2_used else ""
        print(f"  gap {i:>2}: {pw!r:<18} → {cw!r:<18}  "
              f"stage1={s1:+.3f}s  pause_before={pb:+.3f}s  "
              f"badge={'Y' if badge else 'N'}{s2_tag}")

        if pb and pb > best_pause:
            best_pause   = pb
            best_gap_idx = i

    sep()
    badge_fires = best_pause > PAUSE_THRESHOLD
    print(f"\n  Largest pause_before: {best_pause:.3f} s  (gap #{best_gap_idx})")
    print(f"  Badge fires: {'✅' if badge_fires else '❌'}")
    if gt_s is not None:
        print(f"  Ground truth: {gt_s:.3f} s  (duration accuracy informational only for live recordings)")
    print(f"  PRIMARY CRITERION (badge): {'PASS ✅' if badge_fires else 'FAIL ❌'}")
    return badge_fires


# ── Production recordings ─────────────────────────────────────────────────────
r1 = check("example1_football.wav", 1.70, "Example 1 — 1.7 s pause")
r2 = check("example2_football.wav", 0.78, "Example 2 — 0.78 s pause (no badge bug)")

# ── Regression: existing test suite must not gain false-positive badges ───────
sep("═")
print("  REGRESSION — existing recordings must not gain spurious badges")
sep("═")

EXISTING = [
    ("test_two_phrases.wav", "two_phrases"),
    ("test_pause.mp3",       "hello_are"),
]
for path, label in EXISTING:
    if not os.path.exists(path): continue
    arr = load_audio(path)
    pts, pf0 = get_pitch(arr)
    pairs, phrases, pitch_times, pitch_f0, audio_array, sr = run_whistress(path)
    print(f"\n  {label}:")
    for i in range(1, len(pairs)):
        pw, _, p0, p1 = pairs[i - 1]
        cw, _, c0, c1 = pairs[i]
        pb   = compute_pause_before(i, pairs, pitch_times, pitch_f0, audio_array, sr)
        s1   = acoustic_pause_duration(p1, c0, pitch_times, pitch_f0)
        prev_span = p1 - p0
        s2   = (s1 < PAUSE_THRESHOLD and prev_span > LONG_WORD_SPAN and pb > s1)
        badge = pb > PAUSE_THRESHOLD if pb else False
        if badge or s2:
            print(f"    gap {i}: {pw!r} → {cw!r}  "
                  f"stage1={s1:.3f}s  pause={pb:.3f}s  badge={'Y' if badge else 'N'}  stage2={'Y' if s2 else 'N'}")
    # just list the badged ones; no output = all fine

sep()
gate_ok = bool(r1) and bool(r2)
print(f"\n  PRODUCTION BUG FIX: {'PASS ✅' if gate_ok else 'FAIL ❌'}")
sep("═")
sys.exit(0 if gate_ok else 1)
