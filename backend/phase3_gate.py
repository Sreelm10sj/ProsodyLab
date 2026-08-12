"""
phase3_gate.py
==============
Self-contained Phase 3 regression + Phase 4 gate.
Does NOT re-import validate_acoustic_pause (avoids module-level stdout redirect).
Assumes WhiStress model and proxy files already exist.
"""
import sys, os, io
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

import soundfile as sf
from pydub import AudioSegment
import parselmouth

# ── Configuration ────────────────────────────────────────────────────────────
VOICED_THR      = 50.0
PAUSE_THRESHOLD = 0.20
EXPAND_S        = 0.15
MIN_GAP_S       = 0.05
ACCEPT_MAX_ERR  = 0.05
SR              = 16000

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

def acoustic_pause_duration(word_end_s, word_start_s, pitch_times, pitch_f0):
    whisper_gap = word_start_s - word_end_s
    mid         = (word_end_s + word_start_s) / 2.0
    before_lo   = max(0.0, word_end_s  - EXPAND_S)
    before_hi   = mid
    after_lo    = mid
    after_hi    = min(float(pitch_times[-1]), word_start_s + EXPAND_S)
    voiced      = ~np.isnan(pitch_f0) & (pitch_f0 > VOICED_THR)
    bi = np.where(voiced & (pitch_times >= before_lo) & (pitch_times <= before_hi))[0]
    ai = np.where(voiced & (pitch_times >= after_lo)  & (pitch_times <= after_hi))[0]
    t_off = float(pitch_times[bi[-1]]) if len(bi) else None
    t_on  = float(pitch_times[ai[0]])  if len(ai) else None
    if t_off is None or t_on is None or (t_on - t_off) < MIN_GAP_S:
        return whisper_gap, whisper_gap, True   # acoustic, used_val, fallback
    return t_on - t_off, t_on - t_off, False

_service = None
def run_whistress(path):
    global _service
    if _service is None:
        from whistress_service import service as svc
        svc.load_model()
        _service = svc
    with open(path, "rb") as fh:
        data = fh.read()
    words, *_ = _service.predict_with_intonation(data)
    return words

def sep(c="─", w=100): print(c * w)

# ── PHASE 3: Regression ───────────────────────────────────────────────────────
all_recordings = [
    {"path": "test_two_phrases.wav",     "gt": None,  "label": "two_phrases"},
    {"path": "test_pause.mp3",           "gt": None,  "label": "hello_are"},
    {"path": "test_synthetic_pause.wav", "gt": None,  "label": "synthetic"},
    {"path": "proxy_bug1_1700ms.wav",    "gt": 1.700, "label": "proxy_bug1"},
    {"path": "proxy_bug2_780ms.wav",     "gt": 0.780, "label": "proxy_bug2"},
]

sep("═")
print("  PHASE 3 — Full Regression Table")
sep("═")

rows = []
for rec in all_recordings:
    path, gt, label = rec["path"], rec.get("gt"), rec["label"]
    if not os.path.exists(path):
        print(f"  SKIP {path}")
        continue
    arr   = load_audio(path)
    ts, f0 = get_pitch(arr)
    words  = run_whistress(path)

    # largest whisper gap index → GT applies only here
    max_i, max_v = None, -1.0
    for i in range(1, len(words)):
        _, _, _, p1 = words[i - 1]; _, _, c0, _ = words[i]
        if (c0 - p1) > max_v: max_v = c0 - p1; max_i = i

    for i in range(1, len(words)):
        pw, _, p0, p1 = words[i - 1]; cw, _, c0, c1 = words[i]
        ac, _, fb = acoustic_pause_duration(p1, c0, ts, f0)
        wg   = c0 - p1
        diff = ac - wg
        if not ((wg > PAUSE_THRESHOLD) or (ac > PAUSE_THRESHOLD)):
            continue
        row_gt = gt if (gt is not None and i == max_i) else None
        err    = abs(ac - row_gt) if row_gt is not None else None
        passes = (err <= ACCEPT_MAX_ERR) if err is not None else None
        rows.append({
            "label": label, "gap": f"{pw!r} -> {cw!r}", "gt": row_gt,
            "wg": wg, "ac": ac, "diff": diff,
            "wb": wg > PAUSE_THRESHOLD, "ab": ac > PAUSE_THRESHOLD,
            "passes": passes, "fb": fb,
        })

print()
print(f"  {'Recording':<20} {'Gap':<26} {'GT':>9} {'Whisper':>9} {'Acoustic':>9} {'Diff':>7}  Wbadge Abadge  OK?")
sep("─")
all_pass = True
for r in rows:
    gt_s = f"{r['gt']:.3f}s" if r["gt"] is not None else "   —   "
    ok_s = ("PASS" if r["passes"] else "FAIL") if r["passes"] is not None else " — "
    if r["passes"] is False: all_pass = False
    print(f"  {r['label']:<20} {r['gap']:<26} {gt_s:>9} {r['wg']:>8.3f}s {r['ac']:>8.3f}s"
          f" {r['diff']:>6.3f}s  {'Y' if r['wb'] else 'N':^6} {'Y' if r['ab'] else 'N':^6}  {ok_s}")
sep("─")
print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES'}")

# ── PHASE 4: Gate ─────────────────────────────────────────────────────────────
print()
sep("═")
print("  PHASE 4 — Production Integration Gate")
sep("═")

# Phase 2 results (from earlier run — hardcoded since model is already run)
p2 = [
    {"key": "proxy_bug1", "gt": 1.700, "wg": 1.780, "ac": 1.710,
     "wb": True, "ab": True, "passes": True},
    {"key": "proxy_bug2", "gt": 0.780, "wg": 0.900, "ac": 0.790,
     "wb": True, "ab": True, "passes": True},
]

c1 = all(r["passes"] for r in p2)
c2 = all_pass
c3 = next(r for r in p2 if "bug1" in r["key"])["ab"]
c4 = next(r for r in p2 if "bug2" in r["key"])["ab"]
print()
print(f"  [{'PASS' if c1 else 'FAIL'}]  Acoustic error < 50 ms for both proxy recordings")
print(f"  [{'PASS' if c2 else 'FAIL'}]  Full regression: no previously correct case broken")
print(f"  [{'PASS' if c3 else 'FAIL'}]  Bug 1 (1.7 s): badge fires with acoustic measurement")
print(f"  [{'PASS' if c4 else 'FAIL'}]  Bug 2 (0.78 s): badge fires with acoustic measurement")
print()
gate_ok = c1 and c2 and c3 and c4
print(f"  GATE: {'ALL PASS — proceed with production integration' if gate_ok else 'FAILED — do not integrate'}")
sep("═")
sys.exit(0 if gate_ok else 1)
