"""Quick raw acoustic gap probe — run directly, no argparse."""
import sys, os, io
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

import soundfile as sf
from pydub import AudioSegment
import parselmouth

VOICED_THR = 50.0   # Hz — frames below this are unvoiced


def load_audio(path):
    a = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
    b = io.BytesIO(); a.export(b, format="wav"); b.seek(0)
    arr, _ = sf.read(b, dtype="float32")
    return arr


def get_pitch(arr, sr=16000):
    snd = parselmouth.Sound(values=arr.astype(np.float64), sampling_frequency=sr)
    p   = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    ts  = np.array(p.xs())
    f0  = np.array([p.get_value_at_time(t) or np.nan for t in ts])
    return ts, f0


def acoustic_gaps(ts, f0, min_dur=0.08):
    voiced = ~np.isnan(f0) & (f0 > VOICED_THR)
    gaps   = []
    in_sil, sil_i = False, 0

    for i in range(len(ts)):
        if not voiced[i] and not in_sil:
            in_sil = True
            sil_i  = i
        elif voiced[i] and in_sil:
            in_sil = False
            lv = ts[sil_i - 1] if sil_i > 0 else None
            fv = ts[i]
            dur = (fv - lv) if lv is not None else None
            if dur and dur >= min_dur:
                gaps.append({"last_voiced": lv, "first_voiced": fv, "duration": dur})

    if in_sil and sil_i > 0:
        lv = ts[sil_i - 1]
        fv = ts[-1]
        dur = fv - lv
        if dur >= min_dur:
            gaps.append({"last_voiced": lv, "first_voiced": fv, "duration": dur})

    return gaps


files = [
    "test_two_phrases.wav",
    "test_synthetic_pause.wav",
    "test_pause.mp3",
    "sample.flac",
]

for fname in files:
    if not os.path.exists(fname):
        continue
    arr = load_audio(fname)
    dur = len(arr) / 16000
    ts, f0 = get_pitch(arr)
    voiced_frac = np.sum(~np.isnan(f0) & (f0 > VOICED_THR)) / len(f0)
    gaps = acoustic_gaps(ts, f0)

    print("=" * 62)
    print(f"FILE: {fname}  ({dur:.2f} s,  voiced={voiced_frac:.0%})")
    print("=" * 62)
    print(f"  {'Last voiced':>12}  {'First voiced':>13}  {'Acoustic dur':>13}")
    print("-" * 62)
    for g in gaps:
        flag = " *** LONG" if g["duration"] > 0.45 else ""
        print(f"  {g['last_voiced']:>12.3f} s  {g['first_voiced']:>12.3f} s  "
              f"{g['duration']:>12.3f} s{flag}")

print("=" * 62)
print("Done.")
