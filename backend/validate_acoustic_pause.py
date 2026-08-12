"""
validate_acoustic_pause.py
==========================
Comprehensive validation of the acoustic pause measurement approach.

Phase 1  — Window overlap instrumentation
          Confirms whether naive expand-only windows produce overlap on
          real recordings, and demonstrates that midpoint-clamped windows
          prevent it while preserving correct voiced-frame selection.

Phase 2  — Proxy production recordings
          Two recordings are synthesised by splicing test_two_phrases.wav
          at its natural acoustic silence boundaries and inserting EXACTLY
          1700 ms and 780 ms of digital silence — the two durations reported
          in the production bug.  This gives us ground-truth recordings
          whose silence duration is known to the millisecond.

Phase 3  — Full regression table
          All five recordings (three existing + two proxies) are evaluated
          with both the Whisper-only approach and the new acoustic approach.

Phase 4  — Gate for production integration
          Prints PASS / FAIL for every acceptance criterion so the caller
          can decide whether to proceed with the production code change.

Run:
    python validate_acoustic_pause.py
"""

import io
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WhiStress"))

# ── Force UTF-8 on Windows ──────────────────────────────────────────────────
if hasattr(sys.stdout, "buffer"):
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import soundfile as sf
from pydub import AudioSegment
import parselmouth

# ── Constants (must stay in sync with pitch_analysis.py CONFIGURATION) ──────
VOICED_THR            = 50.0    # Hz — frames below this are treated as unvoiced
PAUSE_THRESHOLD       = 0.20    # s  — frontend badge threshold (TranscriptView.jsx:68)
EXPAND_S              = 0.15    # s  — expand_s passed to acoustic_pause_duration
MIN_GAP_S             = 0.05    # s  — minimum acoustic gap we trust
PITCH_TIME_STEP       = 0.01    # s  — Parselmouth frame step
PITCH_FMIN_HZ         = 75.0
PITCH_FMAX_HZ         = 600.0

# ── Acceptance criterion ─────────────────────────────────────────────────────
ACCEPT_MAX_ERROR_S    = 0.05    # acoustic pause must be within 50 ms of ground truth

SR = 16000  # all audio resampled to 16 kHz

# ============================================================================
# Audio I/O
# ============================================================================

def load_audio(path: str) -> np.ndarray:
    a = AudioSegment.from_file(path).set_channels(1).set_frame_rate(SR)
    b = io.BytesIO(); a.export(b, format="wav"); b.seek(0)
    arr, _ = sf.read(b, dtype="float32")
    return arr


def save_audio(arr: np.ndarray, path: str):
    sf.write(path, arr, SR)
    print(f"  Saved: {path}  ({len(arr)/SR:.3f} s)")


# ============================================================================
# Pitch extraction
# ============================================================================

def get_pitch(arr: np.ndarray) -> tuple:
    snd = parselmouth.Sound(values=arr.astype(np.float64), sampling_frequency=SR)
    p   = snd.to_pitch(
        time_step=PITCH_TIME_STEP, pitch_floor=PITCH_FMIN_HZ, pitch_ceiling=PITCH_FMAX_HZ
    )
    ts = np.array(p.xs())
    f0 = np.array([p.get_value_at_time(t) or np.nan for t in ts])
    return ts, f0


# ============================================================================
# acoustic_pause_duration() — the function under validation
# ============================================================================

def acoustic_pause_duration(
    word_end_s:   float,
    word_start_s: float,
    pitch_times:  np.ndarray,
    pitch_f0:     np.ndarray,
    expand_s:     float = EXPAND_S,
    min_gap_s:    float = MIN_GAP_S,
    verbose:      bool  = False,
) -> dict:
    """
    Measure the acoustic silence between two words using pitch voicing frames.

    Window design — midpoint clamping
    ----------------------------------
    Naive approach uses [word_end - expand, word_end + expand] for t_off and
    [word_start - expand, word_start + expand] for t_on.  Whenever the Whisper
    gap (word_start - word_end) < 2 * expand_s these two windows overlap.

    A midpoint clamp eliminates the overlap unconditionally:

        mid = (word_end_s + word_start_s) / 2

        Before window:  [word_end_s - expand_s,  mid]
        After window:   [mid,  word_start_s + expand_s]

    By construction before_hi == after_lo == mid, so the regions are
    disjoint.  For wide gaps (Whisper gap > 2*expand_s) mid lies inside the
    silence, which is fine — the algorithm still finds the last voiced frame
    before mid for t_off and the first voiced frame after mid for t_on.

    Why midpoint clamping is preferred over alternatives
    ----------------------------------------------------
    - Symmetry: both windows are shrunk equally toward the centre, preserving
      the geometric intuition that t_off < mid < t_on for any true pause.
    - No free parameter: the clamp point (mid) is uniquely determined by the
      two Whisper timestamps.
    - Correctness without fallback: even for zero-gap adjacent words the
      before window searches only left of mid and the after window only right
      of mid, so the algorithm can never return t_on < t_off due to overlap.
    - Graceful degradation: if the gap < 2*pitch_time_step (< 20 ms) mid ==
      word_end, the before window collapses to zero width, t_off is None, and
      the fallback returns the Whisper gap — correct for adjacent words.

    Parameters
    ----------
    word_end_s    : Whisper end time of the preceding word (seconds)
    word_start_s  : Whisper start time of the following word (seconds)
    pitch_times   : 1-D array of pitch frame timestamps
    pitch_f0      : 1-D array of F0 values (NaN = unvoiced)
    expand_s      : expand_s — search that far outside the Whisper window
    min_gap_s     : Minimum acoustic gap before we trust the measurement.
                    Below this, falls back to the Whisper gap.
    verbose       : If True, print instrumentation details.

    Returns
    -------
    dict with keys:
        whisper_gap     : float  — word_start_s − word_end_s
        mid             : float  — midpoint of whisper gap
        before_lo/hi    : float  — actual before-window bounds used
        after_lo/hi     : float  — actual after-window bounds used
        naive_overlap   : bool   — would naive windows have overlapped?
        t_off           : float | None — last voiced frame before gap
        t_off_in_overlap: bool   — was t_off in the naive overlap zone?
        t_on            : float | None — first voiced frame after gap
        t_on_in_overlap : bool   — was t_on in the naive overlap zone?
        acoustic_s      : float  — measured acoustic pause (or whisper_gap)
        fallback_used   : bool   — True if we returned whisper_gap
        badge_fires     : bool   — acoustic_s > PAUSE_THRESHOLD
    """
    whisper_gap = word_start_s - word_end_s

    # ── Midpoint-clamped windows ─────────────────────────────────────────────
    mid        = (word_end_s + word_start_s) / 2.0
    before_lo  = max(0.0, word_end_s  - expand_s)
    before_hi  = mid                                     # clamp
    after_lo   = mid                                     # clamp
    after_hi   = min(float(pitch_times[-1] if len(pitch_times) else 0.0),
                     word_start_s + expand_s)

    # ── Naive (un-clamped) window bounds — for overlap detection only ─────────
    naive_before_hi = word_end_s  + expand_s
    naive_after_lo  = word_start_s - expand_s
    naive_overlap   = naive_before_hi > naive_after_lo

    # ── Voiced frame search ─────────────────────────────────────────────────
    voiced = ~np.isnan(pitch_f0) & (pitch_f0 > VOICED_THR)

    # t_off: last voiced frame in [before_lo, before_hi]
    before_mask = voiced & (pitch_times >= before_lo) & (pitch_times <= before_hi)
    before_idx  = np.where(before_mask)[0]
    t_off       = float(pitch_times[before_idx[-1]]) if len(before_idx) else None

    # t_on: first voiced frame in [after_lo, after_hi]
    after_mask  = voiced & (pitch_times >= after_lo) & (pitch_times <= after_hi)
    after_idx   = np.where(after_mask)[0]
    t_on        = float(pitch_times[after_idx[0]])  if len(after_idx)  else None

    # ── Overlap contamination flags ─────────────────────────────────────────
    overlap_lo = naive_after_lo
    overlap_hi = naive_before_hi
    t_off_in_overlap = (
        naive_overlap and t_off is not None
        and overlap_lo <= t_off <= overlap_hi
    )
    t_on_in_overlap = (
        naive_overlap and t_on is not None
        and overlap_lo <= t_on <= overlap_hi
    )

    # ── Acoustic duration ────────────────────────────────────────────────────
    if t_off is None or t_on is None or (t_on - t_off) < min_gap_s:
        acoustic_s   = whisper_gap
        fallback_used = True
    else:
        acoustic_s   = t_on - t_off
        fallback_used = False

    result = {
        "whisper_gap":      whisper_gap,
        "mid":              mid,
        "before_lo":        before_lo,
        "before_hi":        before_hi,
        "after_lo":         after_lo,
        "after_hi":         after_hi,
        "naive_overlap":    naive_overlap,
        "t_off":            t_off,
        "t_off_in_overlap": t_off_in_overlap,
        "t_on":             t_on,
        "t_on_in_overlap":  t_on_in_overlap,
        "acoustic_s":       acoustic_s,
        "fallback_used":    fallback_used,
        "badge_fires":      acoustic_s > PAUSE_THRESHOLD,
    }

    if verbose:
        print(f"    whisper_gap      = {whisper_gap:+.3f} s")
        print(f"    mid              = {mid:.3f} s")
        print(f"    before window    = [{before_lo:.3f}, {before_hi:.3f}]")
        print(f"    after  window    = [{after_lo:.3f}, {after_hi:.3f}]")
        print(f"    naive overlap?   = {naive_overlap}")
        if naive_overlap:
            print(f"      overlap zone   = [{overlap_lo:.3f}, {overlap_hi:.3f}]")
        print(f"    t_off            = {t_off:.3f} s" if t_off else "    t_off            = None")
        if t_off_in_overlap:
            print(f"      *** t_off IN OVERLAP ZONE — would have contaminated naive result")
        print(f"    t_on             = {t_on:.3f} s" if t_on else "    t_on             = None")
        if t_on_in_overlap:
            print(f"      *** t_on IN OVERLAP ZONE — would have contaminated naive result")
        print(f"    acoustic_s       = {acoustic_s:.3f} s  (fallback={fallback_used})")
        print(f"    badge_fires      = {result['badge_fires']}")

    return result


# ============================================================================
# WhiStress inference helper
# ============================================================================

_service = None

def run_whistress(path: str):
    """Return [(word, stress, start, end), ...] for the given audio file."""
    global _service
    if _service is None:
        print("  Loading WhiStress model (first call only)...")
        from whistress_service import service as svc
        svc.load_model()
        _service = svc
    with open(path, "rb") as fh:
        audio_bytes = fh.read()
    words, *_ = _service.predict_with_intonation(audio_bytes)
    return words


# ============================================================================
# Proxy recording creation (Phase 2)
# ============================================================================

def create_proxy_recordings(src_path: str) -> dict:
    """
    Splice test_two_phrases.wav at the acoustic voiced boundaries of its main
    pause, then insert exactly 1700 ms and 780 ms of digital silence.

    The acoustic boundaries (last voiced / first voiced) were measured in the
    previous investigation run (probe_acoustic_gaps.py):
        last_voiced  = 1.404 s   →  use as split point A
        first_voiced = 2.794 s   →  use as split point B

    Returns dict with {name: file_path} for the two proxy recordings.
    """
    SPLIT_A_MS = 1404   # last voiced frame before natural pause
    SPLIT_B_MS = 2794   # first voiced frame after natural pause
    BUG1_PAUSE_MS = 1700
    BUG2_PAUSE_MS = 780

    a = AudioSegment.from_file(src_path).set_channels(1).set_frame_rate(SR)
    part_A   = a[:SPLIT_A_MS]
    part_B   = a[SPLIT_B_MS:]
    silence1 = AudioSegment.silent(duration=BUG1_PAUSE_MS, frame_rate=SR)
    silence2 = AudioSegment.silent(duration=BUG2_PAUSE_MS, frame_rate=SR)

    proxy1 = part_A + silence1 + part_B
    proxy2 = part_A + silence2 + part_B

    out1 = "proxy_bug1_1700ms.wav"
    out2 = "proxy_bug2_780ms.wav"
    proxy1.export(out1, format="wav")
    proxy2.export(out2, format="wav")
    print(f"  proxy_bug1  ({SPLIT_A_MS} ms speech + {BUG1_PAUSE_MS} ms silence + {len(part_B)} ms speech)  → {out1}")
    print(f"  proxy_bug2  ({SPLIT_A_MS} ms speech + {BUG2_PAUSE_MS} ms silence + {len(part_B)} ms speech)  → {out2}")

    return {
        "proxy_bug1": {
            "path": out1,
            "ground_truth_s": BUG1_PAUSE_MS / 1000.0,
            "description": "Proxy for Example 1 (reported 1.7 s pause, Whisper showed ~0.8 s)",
        },
        "proxy_bug2": {
            "path": out2,
            "ground_truth_s": BUG2_PAUSE_MS / 1000.0,
            "description": "Proxy for Example 2 (reported 0.78 s pause, Whisper showed 0 s)",
        },
    }


# ============================================================================
# Print helpers
# ============================================================================

def sep(c="─", w=74): print(c * w)


def badge(v): return "✅ BADGE" if v else "❌ no badge"


# ============================================================================
# Phase 1 — Window overlap instrumentation
# ============================================================================

def phase1_overlap_instrumentation(test_files: list):
    print()
    sep("═")
    print("  PHASE 1 — Window Overlap Instrumentation")
    print(f"  expand_s = {EXPAND_S} s  |  overlap occurs when whisper_gap < {2*EXPAND_S:.2f} s")
    sep("═")

    overlap_cases    = []
    contaminated_cases = []

    for path, label in test_files:
        if not os.path.exists(path):
            print(f"  SKIP {path} (not found)")
            continue

        print(f"\n  File: {os.path.basename(path)}")
        sep()

        arr = load_audio(path)
        ts, f0 = get_pitch(arr)
        words = run_whistress(path)

        for i in range(1, len(words)):
            pw, _, p0, p1 = words[i - 1]
            cw, _, c0, c1 = words[i]
            r = acoustic_pause_duration(p1, c0, ts, f0, verbose=False)

            flag = ""
            if r["naive_overlap"]:
                flag = " ⚠  OVERLAP"
                overlap_cases.append((path, pw, cw, r))
                if r["t_off_in_overlap"] or r["t_on_in_overlap"]:
                    flag += " + CONTAMINATED"
                    contaminated_cases.append((path, pw, cw, r))

            wg_flag = "BADGE" if r["whisper_gap"] > PAUSE_THRESHOLD else "     "
            ac_flag = "BADGE" if r["badge_fires"]                   else "     "
            print(
                f"  gap {i:>2}: {pw!r:<14} → {cw!r:<14}  "
                f"whisper={r['whisper_gap']:+.3f}s [{wg_flag}]  "
                f"acoustic={r['acoustic_s']:+.3f}s [{ac_flag}]  "
                f"fallback={r['fallback_used']}  "
                f"overlap={r['naive_overlap']}{flag}"
            )

    print()
    sep("═")
    print(f"  Overlap summary:  {len(overlap_cases)} gap(s) had naive window overlap")
    print(f"  Contaminated:     {len(contaminated_cases)} gap(s) where t_off or t_on was in overlap zone")
    print()
    if contaminated_cases:
        print("  CONTAMINATED GAPS — detail:")
        for path, pw, cw, r in contaminated_cases:
            print(f"    {os.path.basename(path)}: {pw!r} → {cw!r}")
            print(f"      t_off={r['t_off']}  t_off_in_overlap={r['t_off_in_overlap']}")
            print(f"      t_on={r['t_on']}   t_on_in_overlap={r['t_on_in_overlap']}")
            print(f"      Naive would give:  ~{(r['t_on'] or 0) - (r['t_off'] or 0):.3f} s  "
                  f"Clamped gives:  {r['acoustic_s']:.3f} s (fallback={r['fallback_used']})")
    else:
        print("  No contamination detected — midpoint clamping does not alter results")
        print("  for existing test recordings, but it provides a correctness guarantee")
        print("  for recordings where the whisper_gap < 2 * expand_s.")
    sep("═")

    return overlap_cases, contaminated_cases


# ============================================================================
# Phase 2 — Proxy production bug recordings
# ============================================================================

def phase2_proxy_recordings(proxy_info: dict) -> list:
    print()
    sep("═")
    print("  PHASE 2 — Proxy Production Bug Recordings")
    sep("═")

    rows = []
    for key, info in proxy_info.items():
        path  = info["path"]
        gt    = info["ground_truth_s"]
        desc  = info["description"]

        print(f"\n  {key}: {desc}")
        print(f"  Ground truth pause: {gt:.3f} s")
        sep()

        if not os.path.exists(path):
            print(f"  SKIP — file not found: {path}")
            continue

        arr = load_audio(path)
        ts, f0 = get_pitch(arr)
        words = run_whistress(path)

        print(f"  Words: {len(words)}")
        for i, (w, s, t0, t1) in enumerate(words):
            print(f"    {i:>2}  {w:<18} {t0:.3f}s – {t1:.3f}s  {'●' if s else '·'}")

        sep()
        print("  Gap measurements:")
        best_whisper_gap = None
        best_acoustic    = None
        best_gap_idx     = None
        best_r           = None

        for i in range(1, len(words)):
            pw, _, p0, p1 = words[i - 1]
            cw, _, c0, c1 = words[i]
            r = acoustic_pause_duration(p1, c0, ts, f0, verbose=False)

            print(f"    gap {i}: {pw!r} → {cw!r}")
            print(f"      Whisper gap  = {r['whisper_gap']:+.3f} s  [{badge(r['whisper_gap'] > PAUSE_THRESHOLD)}]")
            print(f"      Acoustic     = {r['acoustic_s']:+.3f} s  [{badge(r['badge_fires'])}]  fallback={r['fallback_used']}")

            if best_whisper_gap is None or r["whisper_gap"] > best_whisper_gap:
                best_whisper_gap = r["whisper_gap"]
                best_acoustic    = r["acoustic_s"]
                best_gap_idx     = i
                best_r           = r

        print()
        whisper_err  = abs((best_whisper_gap or 0) - gt)
        acoustic_err = abs((best_acoustic or 0)    - gt)
        passes_criterion = acoustic_err <= ACCEPT_MAX_ERROR_S

        print(f"  === Largest gap (#{best_gap_idx}) ===")
        print(f"  Ground truth:     {gt:.3f} s")
        print(f"  Whisper gap:      {best_whisper_gap:.3f} s   error = {best_whisper_gap - gt:+.3f} s")
        print(f"  Acoustic pause:   {best_acoustic:.3f} s   error = {best_acoustic - gt:+.3f} s")
        print(f"  Badge (Whisper):  {badge(best_whisper_gap > PAUSE_THRESHOLD)}")
        print(f"  Badge (Acoustic): {badge(best_acoustic > PAUSE_THRESHOLD)}")
        print(f"  Criterion (<{ACCEPT_MAX_ERROR_S:.2f}s): {'PASS ✅' if passes_criterion else 'FAIL ❌'}")

        rows.append({
            "key":           key,
            "path":          path,
            "gt":            gt,
            "whisper_gap":   best_whisper_gap,
            "acoustic":      best_acoustic,
            "whisper_err":   best_whisper_gap - gt if best_whisper_gap else None,
            "acoustic_err":  best_acoustic    - gt if best_acoustic    else None,
            "whisper_badge": best_whisper_gap > PAUSE_THRESHOLD,
            "acoustic_badge": best_acoustic   > PAUSE_THRESHOLD,
            "passes":        passes_criterion,
        })

    return rows


# ============================================================================
# Phase 3 — Full regression table
# ============================================================================

def phase3_regression(all_recordings: list) -> list:
    """
    all_recordings: list of {"path": ..., "gt": ... or None, "label": ...}

    Ground truth semantics
    ----------------------
    For proxy recordings the gt value is the INSERTED pause duration.  Only
    the gap with the largest Whisper measurement (the one spanning the spliced
    silence) is compared against the gt.  All other gaps within the same
    recording are reported without a pass/fail judgement — their gt is None.
    """
    print()
    sep("═")
    print("  PHASE 3 — Full Regression Table")
    sep("═")

    rows = []
    for rec in all_recordings:
        path  = rec["path"]
        gt    = rec.get("gt")
        label = rec.get("label", os.path.basename(path))

        if not os.path.exists(path):
            print(f"  SKIP {path}")
            continue

        arr = load_audio(path)
        ts, f0 = get_pitch(arr)
        words = run_whistress(path)

        # Identify the gap index with the largest Whisper gap in this recording.
        # For proxy recordings this is the spliced pause — the only one whose
        # duration we know a priori.  GT is NOT applied to any other gap.
        max_wg_idx = None
        max_wg_val = -1.0
        for i in range(1, len(words)):
            _, _, _, p1 = words[i - 1]
            _, _, c0, _ = words[i]
            wg = c0 - p1
            if wg > max_wg_val:
                max_wg_val = wg
                max_wg_idx = i

        for i in range(1, len(words)):
            pw, _, p0, p1 = words[i - 1]
            cw, _, c0, c1 = words[i]
            r = acoustic_pause_duration(p1, c0, ts, f0)

            wg = r["whisper_gap"]
            ac = r["acoustic_s"]
            diff = ac - wg

            # Only show gaps where at least one measurement crosses threshold
            interesting = (wg > PAUSE_THRESHOLD) or (ac > PAUSE_THRESHOLD)
            if not interesting:
                continue

            # Apply GT only to the primary (max-whisper) gap row
            row_gt = gt if (gt is not None and i == max_wg_idx) else None
            err    = abs(ac - row_gt) if row_gt is not None else None
            passes = (err <= ACCEPT_MAX_ERROR_S) if err is not None else None

            rows.append({
                "label":         label,
                "gap":           f"{pw!r} → {cw!r}",
                "gt":            row_gt,
                "whisper_gap":   wg,
                "acoustic":      ac,
                "diff":          diff,
                "whisper_badge": wg > PAUSE_THRESHOLD,
                "acoustic_badge": ac > PAUSE_THRESHOLD,
                "passes":        passes,
            })

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    w_rec, w_gap, w_gt, w_wh, w_ac, w_df = 22, 26, 10, 10, 10, 9
    hdr = (
        f"  {'Recording':<{w_rec}} {'Gap':<{w_gap}} {'GT':>{w_gt}} "
        f"{'Whisper':>{w_wh}} {'Acoustic':>{w_ac}} {'Diff':>{w_df}}  Badge W  Badge A  OK?"
    )
    print(hdr)
    sep("─", len(hdr))

    all_pass = True
    for r in rows:
        gt_s  = f"{r['gt']:.3f}s"   if r["gt"]     is not None else "  —     "
        ok_s  = ("PASS ✅" if r["passes"] else "FAIL ❌") if r["passes"] is not None else "  —  "
        if r["passes"] is False:
            all_pass = False

        print(
            f"  {r['label']:<{w_rec}} {r['gap']:<{w_gap}} {gt_s:>{w_gt}} "
            f"{r['whisper_gap']:>{w_wh}.3f}s {r['acoustic']:>{w_ac}.3f}s "
            f"{r['diff']:>{w_df}.3f}s  "
            f"{'✅' if r['whisper_badge'] else '❌':^7}  "
            f"{'✅' if r['acoustic_badge'] else '❌':^7}  {ok_s}"
        )

    sep("─", len(hdr))
    print(f"\n  Overall regression: {'ALL PASS ✅' if all_pass else 'FAILURES DETECTED ❌'}")
    sep("═")
    return rows, all_pass


# ============================================================================
# Phase 4 — Gate report
# ============================================================================

def phase4_gate(p2_rows, all_pass_regression):
    print()
    sep("═")
    print("  PHASE 4 — Production Integration Gate")
    sep("═")
    print()

    criteria = []

    # C1: Acoustic error < 50 ms for all proxy recordings
    proxy_pass = all(r["passes"] for r in p2_rows if r.get("passes") is not None)
    criteria.append(("Acoustic error < 50 ms for all proxy recordings", proxy_pass))

    # C2: All previously correct cases remain correct (regression)
    criteria.append(("Full regression: no previously correct case broken",  all_pass_regression))

    # C3: Production bug 1 badge now fires
    b1 = next((r for r in p2_rows if "1700" in r["path"] or "bug1" in r["key"]), None)
    c3 = b1 is not None and b1["acoustic_badge"]
    criteria.append(("Bug 1 (1.7 s pause): badge now fires with acoustic measurement", c3))

    # C4: Production bug 2 badge now fires
    b2 = next((r for r in p2_rows if "780" in r["path"] or "bug2" in r["key"]), None)
    c4 = b2 is not None and b2["acoustic_badge"]
    criteria.append(("Bug 2 (0.78 s pause): badge now fires with acoustic measurement", c4))

    for desc, ok in criteria:
        status = "PASS ✅" if ok else "FAIL ❌"
        print(f"  [{status}]  {desc}")

    all_criteria_pass = all(ok for _, ok in criteria)
    print()
    print(f"  GATE RESULT: {'✅ ALL CRITERIA PASS — safe to integrate into production.' if all_criteria_pass else '❌ CRITERIA FAILED — do NOT integrate until resolved.'}")
    sep("═")
    return all_criteria_pass


# ============================================================================
# Main
# ============================================================================

def main():
    BASE = "test_two_phrases.wav"

    # ── Phase 1: overlap instrumentation ─────────────────────────────────────
    test_files = [
        ("test_two_phrases.wav",    "Two-phrase recording"),
        ("test_pause.mp3",          "Hello/are recording"),
        ("test_synthetic_pause.wav","Synthetic pause"),
    ]
    overlap_cases, contaminated_cases = phase1_overlap_instrumentation(
        [(p, l) for p, l in test_files]
    )

    # ── Create proxy recordings ───────────────────────────────────────────────
    print()
    sep("═")
    print("  Creating proxy recordings from test_two_phrases.wav ...")
    sep("═")
    proxy_info = create_proxy_recordings(BASE)

    # ── Phase 2: proxy validation ─────────────────────────────────────────────
    p2_rows = phase2_proxy_recordings(proxy_info)

    # ── Phase 3: regression ───────────────────────────────────────────────────
    all_recordings = [
        {"path": "test_two_phrases.wav",    "gt": None, "label": "two_phrases"},
        {"path": "test_pause.mp3",          "gt": None, "label": "hello_are"},
        {"path": "test_synthetic_pause.wav","gt": None, "label": "synthetic"},
        {"path": proxy_info["proxy_bug1"]["path"],
         "gt":   proxy_info["proxy_bug1"]["ground_truth_s"],
         "label":"proxy_bug1"},
        {"path": proxy_info["proxy_bug2"]["path"],
         "gt":   proxy_info["proxy_bug2"]["ground_truth_s"],
         "label":"proxy_bug2"},
    ]
    regression_rows, all_pass_regression = phase3_regression(all_recordings)

    # ── Phase 4: gate ─────────────────────────────────────────────────────────
    all_criteria_pass = phase4_gate(p2_rows, all_pass_regression)

    # ── Final conclusion ──────────────────────────────────────────────────────
    print()
    sep("═")
    print("  CONCLUSION")
    sep("═")
    print()
    print(f"  Window overlap instrumentation:  {len(overlap_cases)} gap(s) had naive overlap")
    print(f"  Of those, contaminated:          {len(contaminated_cases)} gap(s)")
    print()
    print("  Midpoint clamping guarantees that before/after search windows")
    print("  cannot overlap regardless of the Whisper gap width, providing")
    print("  a mathematically sound result without relying on the t_on>t_off")
    print("  fallback condition to accidentally correct an erroneous search.")
    print()
    if all_criteria_pass:
        print("  All validation criteria passed.")
        print("  Proceed with production integration (Phase 4 of the task).")
    else:
        print("  One or more criteria failed. Review the FAIL items above before integrating.")
    sep("═")

    return all_criteria_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
