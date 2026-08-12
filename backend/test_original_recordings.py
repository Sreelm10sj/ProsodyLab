"""
test_original_recordings.py
============================
Validates acoustic_pause_duration() against the two original production
bug recordings.

Expected files (record via the app or any recorder and place here):
    example1_football.wav  — "I really like football ... but I'm tired today"
                              Actual pause ≈ 1.70 s, Whisper reported ≈ 0.80 s
    example2_football.wav  — Same sentence, intentional pause ≈ 0.78 s,
                              Whisper gave 0 s (no badge)

Usage:
    python test_original_recordings.py

The script reports:
  - Whisper word timeline
  - Acoustic pitch contour source (Parselmouth / librosa fallback)
  - Per-gap: Whisper gap, acoustic gap, fallback flag, badge fired?
  - Summary: whether pause duration and badge are now correct
"""
import sys, os, io, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

import soundfile as sf
from pydub import AudioSegment
import parselmouth

VOICED_THR      = 50.0
PAUSE_THRESHOLD = 0.20
SR              = 16000

def sep(c="─", w=70): print(c * w)

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

def acoustic_pause_duration(word_end_s, word_start_s, pitch_times, pitch_f0,
                             expand_s=0.15, min_gap_s=0.05):
    from pitch_analysis import acoustic_pause_duration as _apd
    return _apd(word_end_s, word_start_s, pitch_times, pitch_f0, expand_s, min_gap_s)

_service = None
def run_whistress(path):
    global _service
    if _service is None:
        from whistress_service import service as svc
        svc.load_model()
        _service = svc
    with open(path, "rb") as fh:
        data = fh.read()
    return _service.predict_with_intonation(data)

EXPECTED = {
    "example1_football.wav": {
        "desc":          "Example 1 — 1.70 s pause, Whisper reported ~0.80 s",
        "gt_pause_s":    1.70,
        "whisper_old_s": 0.80,
        "expect_badge":  True,
        "accept_err_s":  0.10,
    },
    "example2_football.wav": {
        "desc":          "Example 2 — 0.78 s pause, no badge at all",
        "gt_pause_s":    0.78,
        "whisper_old_s": 0.00,
        "expect_badge":  True,
        "accept_err_s":  0.10,
    },
}

results = []
missing = []

for fname, spec in EXPECTED.items():
    if not os.path.exists(fname):
        missing.append(fname)
        continue

    sep("═")
    print(f"  FILE: {fname}")
    print(f"  {spec['desc']}")
    sep("═")

    arr = load_audio(fname)
    print(f"  Duration: {len(arr)/SR:.3f} s")

    pairs, phrases, pitch_times, pitch_f0 = run_whistress(fname)
    print(f"\n  Word timeline ({len(pairs)} words):")
    sep()
    for i, (w, s, t0, t1) in enumerate(pairs):
        print(f"    {i:>2}  {w:<20} {t0:.3f}s – {t1:.3f}s  {'●' if s else '·'}")

    # Per-gap analysis
    print(f"\n  Gaps:")
    sep()
    best = None
    for i in range(1, len(pairs)):
        pw, _, _, p1 = pairs[i - 1]
        cw, _, c0, _ = pairs[i]
        from pitch_analysis import acoustic_pause_duration as _apd
        ac = _apd(p1, c0, pitch_times, pitch_f0)
        wg = c0 - p1
        fb = abs(ac - wg) < 0.001   # true if acoustic == whisper (fallback)
        wb = wg > PAUSE_THRESHOLD
        ab = ac > PAUSE_THRESHOLD
        diff = ac - wg
        print(f"  Gap {i:>2}: {pw!r:<18} → {cw!r:<18}  "
              f"Whisper={wg:+.3f}s  Acoustic={ac:+.3f}s  diff={diff:+.3f}s  "
              f"Wbadge={'Y' if wb else 'N'}  Abadge={'Y' if ab else 'N'}  fb={fb}")
        if best is None or wg > best["whisper_gap"]:
            best = {"pw": pw, "cw": cw, "whisper_gap": wg, "acoustic": ac,
                    "whisper_badge": wb, "acoustic_badge": ab}

    print()
    if best:
        gt    = spec["gt_pause_s"]
        old_w = spec["whisper_old_s"]
        err_w = abs(best["whisper_gap"] - gt)
        err_a = abs(best["acoustic"]    - gt)
        badge_ok = (best["acoustic_badge"] == spec["expect_badge"])
        passes   = err_a <= spec["accept_err_s"] and badge_ok

        sep("─")
        print(f"  Largest gap: {best['pw']!r} → {best['cw']!r}")
        print(f"  Ground truth (reported):   {gt:.3f} s")
        print(f"  Whisper gap (historical):  {old_w:.3f} s   error = {old_w - gt:+.3f} s")
        print(f"  Whisper gap (this run):    {best['whisper_gap']:.3f} s   error = {best['whisper_gap'] - gt:+.3f} s")
        print(f"  Acoustic gap:              {best['acoustic']:.3f} s   error = {err_a:+.3f} s")
        print(f"  Badge (acoustic):          {'✅ fires' if best['acoustic_badge'] else '❌ no badge'}  (expected: {'fires' if spec['expect_badge'] else 'no badge'})")
        print(f"  Criterion (err<{spec['accept_err_s']:.2f}s + badge): {'PASS ✅' if passes else 'FAIL ❌'}")
        results.append({"fname": fname, "passes": passes, **best,
                         "gt": gt, "err_a": err_a, "badge_ok": badge_ok})

if missing:
    print()
    sep("═")
    print("  RECORDINGS NOT FOUND — please record and place in:")
    print(f"  {os.path.abspath('.')}")
    print()
    for f in missing:
        spec = EXPECTED[f]
        print(f"  {f}")
        print(f"    {spec['desc']}")
        print(f"    Ground truth pause: {spec['gt_pause_s']:.2f} s")
        print(f"    Record via the browser mic or any audio recorder.")
    sep("═")

if results:
    print()
    sep("═")
    print("  FINAL RESULT")
    sep("═")
    all_pass = all(r["passes"] for r in results)
    for r in results:
        ok = "PASS ✅" if r["passes"] else "FAIL ❌"
        print(f"  {ok}  {r['fname']}")
        print(f"         Acoustic={r['acoustic']:.3f}s  err={r['err_a']:+.3f}s  badge={'✅' if r['acoustic_badge'] else '❌'}")
    print()
    print(f"  OVERALL: {'ALL PASS ✅' if all_pass else 'FAILURES ❌'}")
    sep("═")
    sys.exit(0 if all_pass else 1)
else:
    print("\n  No results — provide the recording files and re-run.")
    sys.exit(2)
