"""
diagnose_football.py  —  frame-level probe of the two production recordings
"""
import sys, os, io
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

def get_pitch(arr):
    snd = parselmouth.Sound(values=arr.astype(np.float64), sampling_frequency=SR)
    p   = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    ts  = np.array(p.xs())
    f0  = np.array([p.get_value_at_time(t) or np.nan for t in ts])
    return ts, f0

def sep(c="─", w=70): print(c * w)

from whistress_service import service
service.load_model()

for fname, gt in [("example1_football.wav", 1.70), ("example2_football.wav", 0.78)]:
    if not os.path.exists(fname):
        print(f"SKIP {fname}"); continue

    sep("═")
    print(f"FILE: {fname}  (ground-truth pause ≈ {gt:.2f} s)")
    sep("═")

    arr = load_audio(fname)
    ts_raw, f0_raw = get_pitch(arr)
    voiced_raw = ~np.isnan(f0_raw) & (f0_raw > VOICED_THR)
    print(f"Duration: {len(arr)/SR:.3f}s   Pitch frames: {len(ts_raw)}   Voiced: {voiced_raw.sum()}")

    # Raw acoustic silence regions > 80 ms
    in_sil, sil_i = False, 0
    print("\nAcoustic silence regions > 80 ms:")
    for i in range(len(ts_raw)):
        if not voiced_raw[i] and not in_sil:
            in_sil = True; sil_i = i
        elif voiced_raw[i] and in_sil:
            in_sil = False
            lv = ts_raw[sil_i - 1] if sil_i > 0 else None
            fv = ts_raw[i]
            dur = (fv - lv) if lv else None
            if dur and dur > 0.08:
                flag = "  *** LONG" if dur > 0.45 else ""
                print(f"  last_voiced={lv:.3f}s  first_voiced={fv:.3f}s  silence={dur:.3f}s{flag}")

    # WhiStress inference
    with open(fname, "rb") as fh:
        data = fh.read()
    pairs, _, pitch_t, pitch_f = service.predict_with_intonation(data)

    print(f"\nWhiStress word timeline ({len(pairs)} words):")
    sep()
    for i, (w, s, t0, t1) in enumerate(pairs):
        span = t1 - t0
        flag = "  *** LONG SPAN" if span > 0.50 else ""
        print(f"  {i:>2}  {w:<18} {t0:.3f} - {t1:.3f}  span={span:.3f}s{flag}")

    # Focus on the football → but pair
    football_idx = next((i for i, (w, *_) in enumerate(pairs)
                         if w.lower().rstrip(".,!?") == "football"), None)
    if football_idx is None or football_idx + 1 >= len(pairs):
        print("  football pair not found"); continue

    pw, _, p0, p1 = pairs[football_idx]
    cw, _, c0, c1 = pairs[football_idx + 1]
    gap = c0 - p1
    mid = (p1 + c0) / 2.0

    sep()
    print(f"\nCRITICAL GAP:  {pw!r} → {cw!r}")
    print(f"  Whisper: {pw!r}.end   = {p1:.3f} s")
    print(f"  Whisper: {cw!r}.start = {c0:.3f} s")
    print(f"  Whisper: {cw!r}.end   = {c1:.3f} s")
    print(f"  Whisper gap = {gap:.3f} s    mid = {mid:.3f} s")

    v = ~np.isnan(pitch_f) & (pitch_f > VOICED_THR)

    # CURRENT algorithm windows (expand-only, mid-clamped)
    bl_cur = max(0.0, p1 - 0.15)
    bh_cur = mid
    al_cur = mid
    ah_cur = min(float(pitch_t[-1]), c0 + 0.15)

    bi_cur = np.where(v & (pitch_t >= bl_cur) & (pitch_t <= bh_cur))[0]
    ai_cur = np.where(v & (pitch_t >= al_cur) & (pitch_t <= ah_cur))[0]
    t_off_cur = float(pitch_t[bi_cur[-1]]) if len(bi_cur) else None
    t_on_cur  = float(pitch_t[ai_cur[0]])  if len(ai_cur) else None
    ac_cur = (round(t_on_cur - t_off_cur, 3)
              if (t_off_cur and t_on_cur and (t_on_cur - t_off_cur) > 0.05)
              else gap)

    print(f"\n  CURRENT algorithm:")
    print(f"    before [{bl_cur:.3f}, {bh_cur:.3f}] → t_off={t_off_cur}  ({len(bi_cur)} voiced frames)")
    print(f"    after  [{al_cur:.3f}, {ah_cur:.3f}] → t_on ={t_on_cur}  ({len(ai_cur)} voiced frames)")
    print(f"    acoustic = {ac_cur:.3f} s  (whisper_gap={gap:.3f} s)")
    print(f"    badge fires: {ac_cur > 0.20}")

    # PROPOSED algorithm windows (word-span-based)
    bl_new = p0
    bh_new = min(p1 + 0.15, mid)
    al_new = max(c0 - 0.15, mid)
    ah_new = min(float(pitch_t[-1]), c1)

    bi_new = np.where(v & (pitch_t >= bl_new) & (pitch_t <= bh_new))[0]
    ai_new = np.where(v & (pitch_t >= al_new) & (pitch_t <= ah_new))[0]
    t_off_new = float(pitch_t[bi_new[-1]]) if len(bi_new) else None
    t_on_new  = float(pitch_t[ai_new[0]])  if len(ai_new) else None
    ac_new = (round(t_on_new - t_off_new, 3)
              if (t_off_new and t_on_new and (t_on_new - t_off_new) > 0.05)
              else gap)

    print(f"\n  PROPOSED algorithm (word-span windows):")
    print(f"    before [{bl_new:.3f}, {bh_new:.3f}] → t_off={t_off_new}  ({len(bi_new)} voiced frames)")
    print(f"    after  [{al_new:.3f}, {ah_new:.3f}] → t_on ={t_on_new}  ({len(ai_new)} voiced frames)")
    print(f"    acoustic = {ac_new:.3f} s  (whisper_gap={gap:.3f} s)")
    print(f"    badge fires: {ac_new > 0.20}")

    print(f"\n  Ground truth: {gt:.3f} s    Error (proposed): {ac_new - gt:+.3f} s")
    print(f"  Result: {'PASS' if abs(ac_new - gt) < 0.15 else 'FAIL'}")
