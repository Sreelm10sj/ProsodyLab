#!/usr/bin/env python3
"""
test_multi_phrase.py
====================
Multi-phrase integration test for the WhiStress pitch-analysis pipeline.

Verifies without loading the WhiStress / Whisper model that:
  1. segment_phrases() detects boundaries at the correct positions.
  2. Each phrase receives an independent contour classification.
  3. A synthetic rising-then-falling sweep is classified correctly.
  4. A real-speech two-phrase audio (created from existing test files)
     produces two entries in the phrases array.
  5. (Optional) The live /api/transcribe endpoint returns multiple entries
     in the phrases array when the server is already running.

The test produces:
  - test_synthetic_pause.wav   (synthetic swept tones + 0.8 s silence)
  - test_two_phrases.wav       (real speech + 0.8 s silence + real speech)
  - test_multi_phrase_synth.png
  - test_multi_phrase_real.png

Usage
-----
    python test_multi_phrase.py
    python test_multi_phrase.py --no-plot
    python test_multi_phrase.py --api-url http://localhost:8000
"""

import argparse
import io
import logging
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Force UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    format="%(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("multi_phrase_test")

from pitch_analysis import (
    PHRASE_BOUNDARY_THRESHOLD_S,
    SLOPE_RISING_THRESHOLD,
    SLOPE_FALLING_THRESHOLD,
    extract_pitch,
    segment_phrases,
    build_phrase_response,
)

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
_RST    = "\033[0m"
W = 70

PASS = f"{_GREEN}✓ PASS{_RST}"
FAIL = f"{_RED}✗ FAIL{_RST}"

_results: List[Tuple[str, bool, str]] = []   # (test_name, passed, detail)


def _check(name: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    line = f"  {tag}  {name}"
    if detail:
        line += f"\n        {_DIM}{detail}{_RST}"
    print(line)
    _results.append((name, condition, detail))
    return condition


def _section(title: str) -> None:
    print(f"\n{'─' * W}\n  {_BOLD}{title}{_RST}\n{'─' * W}")


def _banner(title: str) -> None:
    dbl = "═" * W
    print(f"\n╔{dbl}╗\n║{title:^{W}}║\n╚{dbl}╝")


# ===========================================================================
# 1. Unit tests for segment_phrases()
# ===========================================================================
def run_segment_unit_tests() -> None:
    _section("Unit Tests — segment_phrases()")

    # --- empty input ---
    result = segment_phrases([])
    _check("Empty input → 0 phrases", len(result) == 0)

    # --- single word ---
    words = [("hello", 0, 0.0, 0.5)]
    result = segment_phrases(words)
    _check(
        "Single word → 1 phrase",
        len(result) == 1 and result[0]["word_indices"] == [0],
        f"got {len(result)} phrase(s)",
    )

    # --- multiple words, all gaps below threshold ---
    words = [
        ("I",       0, 0.00, 0.20),
        ("like",    1, 0.25, 0.55),
        ("coffee",  0, 0.60, 0.90),
    ]
    result = segment_phrases(words)
    _check(
        "Three words, max gap 0.05 s → 1 phrase",
        len(result) == 1,
        f"got {len(result)} phrase(s), gaps: {0.25-0.20:.2f}s, {0.60-0.55:.2f}s",
    )

    # --- two groups separated by a long pause (0.80 s) ---
    #
    #   group1:  "Hello  there"   0.00–0.95 s
    #   silence:                  0.95–1.75 s  (0.80 s gap, > threshold 0.45 s)
    #   group2:  "how   are  you" 1.75–2.90 s
    #
    words = [
        ("Hello", 0, 0.00, 0.40),
        ("there", 0, 0.45, 0.95),
        # 0.80 s gap here
        ("how",   0, 1.75, 2.05),
        ("are",   0, 2.10, 2.50),
        ("you",   0, 2.55, 2.90),
    ]
    result = segment_phrases(words)
    _check(
        "Long pause (0.80 s) → 2 phrases",
        len(result) == 2,
        f"got {len(result)} phrase(s); threshold = {PHRASE_BOUNDARY_THRESHOLD_S} s",
    )
    if len(result) == 2:
        _check(
            "  Phrase 1 contains indices [0, 1]",
            result[0]["word_indices"] == [0, 1],
            f"got {result[0]['word_indices']}",
        )
        _check(
            "  Phrase 2 contains indices [2, 3, 4]",
            result[1]["word_indices"] == [2, 3, 4],
            f"got {result[1]['word_indices']}",
        )
        boundary_gap = words[2][2] - words[1][3]   # how.start – there.end
        _check(
            f"  Boundary gap is {boundary_gap:.3f} s (> {PHRASE_BOUNDARY_THRESHOLD_S} s)",
            boundary_gap >= PHRASE_BOUNDARY_THRESHOLD_S,
        )
        _check(
            "  Phrase 1 time: 0.00 → 0.95 s",
            abs(result[0]["start"] - 0.00) < 1e-6 and abs(result[0]["end"] - 0.95) < 1e-6,
        )
        _check(
            "  Phrase 2 time: 1.75 → 2.90 s",
            abs(result[1]["start"] - 1.75) < 1e-6 and abs(result[1]["end"] - 2.90) < 1e-6,
        )

    # --- three groups, two long pauses ---
    words3 = [
        ("A",   0, 0.00, 0.30),
        # 0.60 s gap
        ("B",   0, 0.90, 1.20),
        # 0.70 s gap
        ("C",   0, 1.90, 2.20),
    ]
    result3 = segment_phrases(words3)
    _check(
        "Two long pauses → 3 phrases",
        len(result3) == 3,
        f"got {len(result3)} phrase(s)",
    )

    # --- gap exactly at threshold (should NOT create a new phrase) ---
    words_exact = [
        ("A", 0, 0.00, 0.30),
        ("B", 0, 0.30 + PHRASE_BOUNDARY_THRESHOLD_S - 0.001, 1.00),
    ]
    result_exact = segment_phrases(words_exact)
    _check(
        f"Gap just below threshold ({PHRASE_BOUNDARY_THRESHOLD_S - 0.001:.3f} s) → 1 phrase",
        len(result_exact) == 1,
        f"got {len(result_exact)} phrase(s)",
    )

    # --- gap exactly at threshold (should create a new phrase) ---
    words_at = [
        ("A", 0, 0.00, 0.30),
        ("B", 0, 0.30 + PHRASE_BOUNDARY_THRESHOLD_S, 1.00),
    ]
    result_at = segment_phrases(words_at)
    _check(
        f"Gap exactly at threshold ({PHRASE_BOUNDARY_THRESHOLD_S:.3f} s) → 2 phrases",
        len(result_at) == 2,
        f"got {len(result_at)} phrase(s)",
    )


# ===========================================================================
# 2. Synthetic pitched audio test
# ===========================================================================
def _sine_sweep(
    f_start: float,
    f_end: float,
    duration: float,
    sr: int = 16000,
    amplitude: float = 0.35,
) -> np.ndarray:
    """
    Generate a linear frequency sweep (chirp) with a raised-cosine envelope.

    A raised-cosine envelope makes the signal look more like a voiced speech
    segment and helps Parselmouth track the F0 at the edges.
    """
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Instantaneous frequency (linear interpolation)
    f_inst = f_start + (f_end - f_start) * t / duration
    # Integrate to get phase
    phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
    signal = amplitude * np.sin(phase)
    # Raised-cosine envelope (smooth fade in / fade out)
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
    return (signal * env).astype(np.float32)


def generate_synthetic_two_phrase_audio(
    sr: int = 16000,
    phrase1_duration: float = 1.5,
    silence_duration: float = 0.80,
    phrase2_duration: float = 1.5,
    save_path: str = "test_synthetic_pause.wav",
) -> Tuple[np.ndarray, float, float, float]:
    """
    Construct synthetic audio:
      Phrase 1: rising sweep 100 Hz → 220 Hz  (should classify as "rising")
      Silence : 0.80 s
      Phrase 2: falling sweep 220 Hz → 100 Hz (should classify as "falling")

    Returns (audio_array, phrase1_end, phrase2_start, phrase2_end).
    """
    import soundfile as sf

    seg1 = _sine_sweep(100.0, 220.0, phrase1_duration, sr)
    silence = np.zeros(int(silence_duration * sr), dtype=np.float32)
    seg2 = _sine_sweep(220.0, 100.0, phrase2_duration, sr)

    audio = np.concatenate([seg1, silence, seg2])
    sf.write(save_path, audio, sr)

    phrase1_end   = phrase1_duration
    phrase2_start = phrase1_duration + silence_duration
    phrase2_end   = phrase2_start + phrase2_duration

    logger.info(
        "Synthetic audio written to %s  (%.2f s total)", save_path, len(audio) / sr
    )
    return audio, phrase1_end, phrase2_start, phrase2_end


def run_synthetic_pitch_test(
    show_plot: bool = True,
    output_plot: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[Tuple]]:
    """
    Generate synthetic pitched audio, run the full pitch analysis pipeline
    with matching mock word timestamps, and verify contour classification.

    Returns (pitch_times, pitch_f0, phrases, words) for plotting.
    """
    _section("Synthetic Pitch Test  (no model required)")

    SR = 16000
    P1_DUR = 1.5    # phrase 1 duration (seconds)
    SIL_DUR = 0.80  # silence gap (seconds)
    P2_DUR  = 1.5   # phrase 2 duration (seconds)

    print(f"  Generating synthetic audio:")
    print(f"    Phrase 1 : 100 Hz → 220 Hz rising sweep  (0.00 s – {P1_DUR:.2f} s)")
    print(f"    Silence  : {SIL_DUR:.2f} s  (expected phrase boundary gap)")
    print(f"    Phrase 2 : 220 Hz → 100 Hz falling sweep ({P1_DUR+SIL_DUR:.2f} s – {P1_DUR+SIL_DUR+P2_DUR:.2f} s)")

    audio, p1_end, p2_start, p2_end = generate_synthetic_two_phrase_audio(
        sr=SR,
        phrase1_duration=P1_DUR,
        silence_duration=SIL_DUR,
        phrase2_duration=P2_DUR,
    )

    # Mock word timestamps that place the boundary in the silence gap.
    # The timestamps must have a gap >= PHRASE_BOUNDARY_THRESHOLD_S (0.45 s)
    # between the last word of phrase 1 and the first word of phrase 2.
    #
    #  phrase 1:  word0  word1  word2  (0.00–1.50 s)
    #  phrase 2:  word3  word4  word5  (2.30–3.80 s)
    #  gap between word2.end and word3.start = 2.30 – 1.50 = 0.80 s  ✓
    #
    words: List[Tuple] = [
        ("[ph1-a]", 0, 0.00, 0.45),
        ("[ph1-b]", 1, 0.50, 0.90),
        ("[ph1-c]", 0, 0.95, P1_DUR),     # ends at 1.50 s
        ("[ph2-a]", 0, p2_start, p2_start + 0.45),      # starts at 2.30 s
        ("[ph2-b]", 0, p2_start + 0.50, p2_start + 0.90),
        ("[ph2-c]", 1, p2_start + 0.95, p2_end),        # ends at 3.80 s
    ]
    boundary_gap = words[3][2] - words[2][3]   # ph2-a.start – ph1-c.end

    print(f"\n  Mock word timestamps: 6 words, boundary gap = {boundary_gap:.3f} s")

    # Run pitch extraction
    print("\n  Running extract_pitch() ...")
    t0 = time.time()
    pitch_times, pitch_f0 = extract_pitch(audio, sr=SR)
    elapsed = time.time() - t0
    voiced = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    utterance_median_hz = float(np.median(voiced)) if len(voiced) > 0 else 0.0
    print(f"  Pitch extracted in {elapsed:.3f} s — {len(pitch_times)} frames")
    print(f"  Utterance median F0 : {utterance_median_hz:.1f} Hz")
    print(f"  Voiced frames       : {len(voiced)}/{len(pitch_f0)}  ({100*len(voiced)/len(pitch_f0):.0f}%)")

    # Run full pipeline
    phrases = build_phrase_response(words, pitch_times, pitch_f0)

    # ---- Assertions ----
    print()
    _check(
        "Boundary gap exceeds phrase threshold",
        boundary_gap >= PHRASE_BOUNDARY_THRESHOLD_S,
        f"{boundary_gap:.3f} s >= {PHRASE_BOUNDARY_THRESHOLD_S} s",
    )
    _check(
        "Two phrases detected",
        len(phrases) == 2,
        f"got {len(phrases)}",
    )

    if len(phrases) == 2:
        p1 = phrases[0]["pitch"]
        p2 = phrases[1]["pitch"]

        _check(
            f"Phrase 1 time: {phrases[0]['start']:.2f} → {phrases[0]['end']:.2f} s",
            abs(phrases[0]["start"] - 0.00) < 0.01 and abs(phrases[0]["end"] - P1_DUR) < 0.01,
        )
        _check(
            f"Phrase 2 time: {phrases[1]['start']:.2f} → {phrases[1]['end']:.2f} s",
            abs(phrases[1]["start"] - p2_start) < 0.01 and abs(phrases[1]["end"] - p2_end) < 0.01,
        )
        _check(
            f"Phrase 1 voiced fraction ≥ 0.7",
            p1["voiced_fraction"] is not None and p1["voiced_fraction"] >= 0.70,
            f"got {p1['voiced_fraction']:.2f}",
        )
        _check(
            f"Phrase 2 voiced fraction ≥ 0.7",
            p2["voiced_fraction"] is not None and p2["voiced_fraction"] >= 0.70,
            f"got {p2['voiced_fraction']:.2f}",
        )
        _check(
            f"Phrase 1 overall slope is POSITIVE (rising sweep)",
            p1["overall_slope"] is not None and p1["overall_slope"] > 0,
            f"slope = {p1['overall_slope']:.3f} st/s",
        )
        _check(
            f"Phrase 2 overall slope is NEGATIVE (falling sweep)",
            p2["overall_slope"] is not None and p2["overall_slope"] < 0,
            f"slope = {p2['overall_slope']:.3f} st/s",
        )
        _check(
            f"Phrase 1 classified as 'rising'  (slope ≥ {SLOPE_RISING_THRESHOLD} st/s)",
            p1["pattern"] == "rising",
            f"got '{p1['pattern']}'  (slope {p1['overall_slope']:.3f} st/s)",
        )
        _check(
            f"Phrase 2 classified as 'falling' (slope ≤ {SLOPE_FALLING_THRESHOLD} st/s)",
            p2["pattern"] == "falling",
            f"got '{p2['pattern']}'  (slope {p2['overall_slope']:.3f} st/s)",
        )

        # Print phrase detail table
        print()
        hdr = f"  {'Phrase':<8}  {'Start':>6}  {'End':>6}  {'Voiced':>7}  {'Slope':>10}  {'Pattern':<10}"
        print(hdr)
        print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*10}")
        for pi, phrase in enumerate(phrases):
            pp = phrase["pitch"]
            pat = pp["pattern"]
            arrow = {"rising": "↗", "falling": "↘", "level": "→"}.get(pat, "~")
            slope_s = f"{pp['overall_slope']:+.3f}" if pp["overall_slope"] is not None else "N/A"
            vf_s    = f"{pp['voiced_fraction']:.2f}" if pp["voiced_fraction"] is not None else "N/A"
            print(f"  {pi+1:<8}  {phrase['start']:>6.3f}  {phrase['end']:>6.3f}  "
                  f"{vf_s:>7}  {slope_s:>10} st/s  {arrow} {pat}")

    # ---- Plot ----
    if show_plot or output_plot:
        _plot_results(
            title="Synthetic Pitch Test — Rising / Falling Sweep",
            audio_filename="test_synthetic_pause.wav",
            duration=len(audio) / SR,
            words=words,
            pitch_times=pitch_times,
            pitch_f0=pitch_f0,
            phrases=phrases,
            output_path=output_plot or "test_multi_phrase_synth.png",
        )

    return pitch_times, pitch_f0, phrases, words


# ===========================================================================
# 3. Real-speech two-phrase audio test
# ===========================================================================
def create_two_phrase_real_audio(
    file_a: str = "test_normal.mp3",
    file_b: str = "test_comma.mp3",
    silence_ms: int = 800,
    save_path: str = "test_two_phrases.wav",
    sr: int = 16000,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Concatenate two real speech files with a silence gap using pydub,
    then return the combined numpy array and phrase boundary timestamps.
    """
    from pydub import AudioSegment
    import soundfile as sf

    seg_a = AudioSegment.from_file(file_a).set_channels(1).set_frame_rate(sr)
    seg_b = AudioSegment.from_file(file_b).set_channels(1).set_frame_rate(sr)
    silence = AudioSegment.silent(duration=silence_ms, frame_rate=sr)

    combined = seg_a + silence + seg_b
    wav_io = io.BytesIO()
    combined.export(wav_io, format="wav")
    wav_io.seek(0)
    audio_array, _ = sf.read(wav_io, dtype="float32")

    sf.write(save_path, audio_array, sr)

    dur_a = len(seg_a) / 1000.0
    phrase2_start = dur_a + silence_ms / 1000.0
    phrase2_end   = phrase2_start + len(seg_b) / 1000.0

    logger.info(
        "Two-phrase real audio written to %s  (%.2f s total, gap at %.2f–%.2f s)",
        save_path, len(audio_array) / sr, dur_a, phrase2_start,
    )
    return audio_array, dur_a, phrase2_start, phrase2_end


def run_real_speech_two_phrase_test(
    show_plot: bool = True,
    output_plot: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[Tuple]]:
    """
    Concatenate two real speech test files with an 800 ms silence, provide
    mock word timestamps matching the structure, and verify phrase detection.
    """
    _section("Real-Speech Two-Phrase Test  (no model required)")

    SILENCE_MS = 800
    FILE_A = os.path.join(SCRIPT_DIR, "test_normal.mp3")
    FILE_B = os.path.join(SCRIPT_DIR, "test_comma.mp3")
    SR = 16000

    for path in (FILE_A, FILE_B):
        if not os.path.exists(path):
            print(f"  {_YELLOW}SKIP{_RST} — source file not found: {path}")
            return np.array([]), np.array([]), [], []

    audio_array, dur_a, p2_start, p2_end = create_two_phrase_real_audio(
        file_a=FILE_A,
        file_b=FILE_B,
        silence_ms=SILENCE_MS,
        save_path=os.path.join(SCRIPT_DIR, "test_two_phrases.wav"),
        sr=SR,
    )
    expected_gap = SILENCE_MS / 1000.0  # 0.800 s

    print(f"  Source A            : test_normal.mp3  (0.00 s – {dur_a:.3f} s)")
    print(f"  Silence gap         : {expected_gap:.3f} s  (expected phrase boundary)")
    print(f"  Source B            : test_comma.mp3   ({p2_start:.3f} s – {p2_end:.3f} s)")
    print(f"  Total duration      : {p2_end:.3f} s")

    # Approximate word timestamps — three evenly spaced "words" per segment.
    # What matters is that there is a clear >0.45 s gap between the last word
    # of segment A and the first word of segment B.
    def _even_words(label: str, t_start: float, t_end: float, n: int = 3) -> List[Tuple]:
        step = (t_end - t_start) / n
        return [
            (f"[{label}{i}]", 0, t_start + i * step, t_start + (i + 1) * step - 0.02)
            for i in range(n)
        ]

    words: List[Tuple] = (
        _even_words("A", 0.0, dur_a - 0.05, n=3)
        + _even_words("B", p2_start + 0.05, p2_end, n=3)
    )
    actual_gap = words[3][2] - words[2][3]   # B0.start – A2.end

    print(f"\n  Mock word gap at boundary : {actual_gap:.3f} s  (threshold = {PHRASE_BOUNDARY_THRESHOLD_S} s)")

    # Pitch extraction
    print("\n  Running extract_pitch() ...")
    t0 = time.time()
    pitch_times, pitch_f0 = extract_pitch(audio_array, sr=SR)
    elapsed = time.time() - t0
    voiced = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    median_hz = float(np.median(voiced)) if len(voiced) > 0 else 0.0
    print(f"  Pitch extracted in {elapsed:.3f} s — {len(pitch_times)} frames")
    print(f"  Utterance median F0 : {median_hz:.1f} Hz")

    phrases = build_phrase_response(words, pitch_times, pitch_f0)

    # ---- Assertions ----
    print()
    _check(
        f"Mock word gap ({actual_gap:.3f} s) exceeds threshold ({PHRASE_BOUNDARY_THRESHOLD_S} s)",
        actual_gap >= PHRASE_BOUNDARY_THRESHOLD_S,
    )
    _check(
        "Two phrases detected",
        len(phrases) == 2,
        f"got {len(phrases)}",
    )
    if len(phrases) == 2:
        p1 = phrases[0]
        p2 = phrases[1]
        detected_gap = p2["start"] - p1["end"]

        _check(
            f"Phrase 1 spans segment A  (0.00 – {dur_a:.2f} s)",
            abs(p1["start"] - 0.00) < 0.05 and abs(p1["end"] - (dur_a - 0.05)) < 0.10,
            f"got {p1['start']:.3f} – {p1['end']:.3f} s",
        )
        _check(
            f"Phrase 2 spans segment B  ({p2_start:.2f} – {p2_end:.2f} s)",
            abs(p2["start"] - (p2_start + 0.05)) < 0.10,
            f"got {p2['start']:.3f} – {p2['end']:.3f} s",
        )
        _check(
            f"Boundary gap between phrases = {detected_gap:.3f} s (≥ {PHRASE_BOUNDARY_THRESHOLD_S} s)",
            detected_gap >= PHRASE_BOUNDARY_THRESHOLD_S,
        )
        _check(
            "Each phrase has independent pitch analysis",
            p1["pitch"]["pattern"] is not None and p2["pitch"]["pattern"] is not None,
            f"phrase1={p1['pitch']['pattern']}, phrase2={p2['pitch']['pattern']}",
        )

        print()
        print(f"  {'Phrase':<8}  {'Start':>7}  {'End':>7}  {'Voiced':>7}  {'Slope':>10}  {'Pattern':<10}")
        print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*10}")
        for pi, phrase in enumerate(phrases):
            pp = phrase["pitch"]
            pat = pp["pattern"]
            arrow = {"rising": "↗", "falling": "↘", "level": "→"}.get(pat, "~")
            slope_s = f"{pp['overall_slope']:+.3f}" if pp["overall_slope"] is not None else "N/A"
            vf_s    = f"{pp['voiced_fraction']:.2f}" if pp["voiced_fraction"] is not None else "N/A"
            print(f"  {pi+1:<8}  {phrase['start']:>7.3f}  {phrase['end']:>7.3f}  "
                  f"{vf_s:>7}  {slope_s:>10} st/s  {arrow} {pat}")

    # ---- Plot ----
    if show_plot or output_plot:
        _plot_results(
            title="Real-Speech Two-Phrase Test",
            audio_filename="test_two_phrases.wav",
            duration=p2_end,
            words=words,
            pitch_times=pitch_times,
            pitch_f0=pitch_f0,
            phrases=phrases,
            output_path=output_plot or "test_multi_phrase_real.png",
        )

    return pitch_times, pitch_f0, phrases, words


# ===========================================================================
# 4. Live API endpoint test
# ===========================================================================
def run_api_test(api_url: str, audio_path: str = "test_two_phrases.wav") -> None:
    _section(f"API Endpoint Test — POST {api_url}/api/transcribe")

    import json
    import mimetypes

    # -- health check --
    health_url = f"{api_url}/api/health"
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            health = json.loads(resp.read())
        print(f"  Server status : {health.get('status', '?')}")
        print(f"  Device        : {health.get('device', '?')}")
        print(f"  Model loaded  : {health.get('model_loaded', '?')}")
        if not health.get("model_loaded"):
            print(f"  {_YELLOW}WARNING{_RST} — model not loaded; transcribe will fail.")
    except Exception as exc:
        print(f"  {_YELLOW}Server not reachable{_RST} ({exc})")
        print(f"\n  To run the API test manually after starting the server:")
        print(f"\n    {_CYAN}uvicorn main:app --reload{_RST}")
        print(f"    {_CYAN}curl -X POST -F \"file=@{audio_path}\" {api_url}/api/transcribe | python -m json.tool{_RST}\n")
        return

    # -- transcribe request --
    if not os.path.exists(audio_path):
        audio_path = os.path.join(SCRIPT_DIR, audio_path)
    if not os.path.exists(audio_path):
        print(f"  {_RED}Audio file not found{_RST}: {audio_path}")
        return

    print(f"\n  Uploading {os.path.basename(audio_path)} ...")

    # Build a minimal multipart/form-data request using stdlib only
    boundary = "----WhistressTestBoundary7a6c5e4d"
    with open(audio_path, "rb") as fh:
        audio_data = fh.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(audio_path)}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{api_url}/api/transcribe",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        elapsed = time.time() - t0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        print(f"  {_RED}HTTP {exc.code}{_RST}: {detail}")
        return
    except Exception as exc:
        print(f"  {_RED}Request failed{_RST}: {exc}")
        return

    print(f"  Inference time  : {elapsed:.1f} s")
    print(f"  Transcription   : {result.get('transcription', '—')!r}")
    words_count  = len(result.get("words", []))
    phrases_list = result.get("phrases", [])
    phrases_count = len(phrases_list)

    print(f"  Words returned  : {words_count}")
    print(f"  Phrases returned: {phrases_count}")

    print()
    _check("'phrases' key present in response", "phrases" in result)
    _check(
        "Multiple phrases in API response",
        phrases_count >= 2,
        f"got {phrases_count} phrase(s)",
    )

    if phrases_list:
        print(f"\n  Full phrases array:")
        print(f"  {'#':<4}  {'Start':>7}  {'End':>7}  {'Words':>6}  {'Voiced':>7}  {'Slope':>10}  Pattern")
        print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*10}")
        for pi, ph in enumerate(phrases_list):
            pp = ph.get("pitch", {})
            pat = pp.get("pattern", "—")
            arrow = {"rising": "↗", "falling": "↘", "level": "→"}.get(pat, "~")
            slope_s = f"{pp.get('overall_slope'):+.3f}" if pp.get("overall_slope") is not None else "N/A"
            vf_s    = f"{pp.get('voiced_fraction'):.2f}" if pp.get("voiced_fraction") is not None else "N/A"
            nw = len(ph.get("word_indices", []))
            print(f"  {pi+1:<4}  {ph.get('start', 0):>7.3f}  {ph.get('end', 0):>7.3f}  "
                  f"{nw:>6}  {vf_s:>7}  {slope_s:>10} st/s  {arrow} {pat}")

        # Verify words array is unchanged (has start/end/stressed/word keys)
        words_list = result.get("words", [])
        if words_list:
            sample = words_list[0]
            _check(
                "'words' array still has expected keys (word, stressed, start, end)",
                all(k in sample for k in ("word", "stressed", "start", "end")),
                f"keys found: {list(sample.keys())}",
            )
            _check(
                "'phrases' uses word_indices (no word duplication)",
                "word_indices" in phrases_list[0],
                f"keys in phrases[0]: {list(phrases_list[0].keys())}",
            )


# ===========================================================================
# 5. Plot helper
# ===========================================================================
def _plot_results(
    title: str,
    audio_filename: str,
    duration: float,
    words: List[Tuple],
    pitch_times: np.ndarray,
    pitch_f0: np.ndarray,
    phrases: List[dict],
    output_path: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    voiced_all = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    utterance_median = float(np.median(voiced_all)) if len(voiced_all) > 0 else None

    PHRASE_COLOURS = ["#2c3e6b", "#1e4d3b", "#4a2040", "#3b3a1e"]
    PATTERN_COLOUR = {
        "rising": "#2ecc71", "falling": "#3498db",
        "level": "#f39c12", "insufficient_voicing": "#95a5a6",
    }
    ARROW = {"rising": "↗", "falling": "↘", "level": "→", "insufficient_voicing": "~"}

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [4, 1]},
        facecolor="#1a1a2e",
    )
    ax_p, ax_w = axes
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.spines[:].set_color("#2c2c4e")
        ax.tick_params(colors="#a0a0c0")

    # ── Phrase regions ──────────────────────────────────────────────────────
    f0_vals = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    f0_max = float(np.max(f0_vals)) if len(f0_vals) > 0 else 300.0
    f0_min = float(np.min(f0_vals)) if len(f0_vals) > 0 else 75.0
    f0_range = max(f0_max - f0_min, 10.0)

    for pi, phrase in enumerate(phrases):
        colour = PHRASE_COLOURS[pi % len(PHRASE_COLOURS)]
        ax_p.axvspan(phrase["start"], phrase["end"], alpha=0.20, color=colour, zorder=1)
        ax_w.axvspan(phrase["start"], phrase["end"], alpha=0.20, color=colour, zorder=1)

    # ── Utterance median ────────────────────────────────────────────────────
    if utterance_median:
        ax_p.axhline(
            utterance_median, color="#f39c12", linestyle="--",
            linewidth=1.0, alpha=0.6, zorder=2,
        )

    # ── Phrase boundary lines ───────────────────────────────────────────────
    for pi in range(len(phrases) - 1):
        boundary = (phrases[pi]["end"] + phrases[pi + 1]["start"]) / 2.0
        for ax in (ax_p, ax_w):
            ax.axvline(boundary, color="#e74c3c", linestyle="--",
                       linewidth=1.4, alpha=0.9, zorder=3)
        # Annotate boundary gap
        gap = phrases[pi + 1]["start"] - phrases[pi]["end"]
        ax_p.text(
            boundary, f0_max + f0_range * 0.03,
            f"gap={gap:.2f}s",
            ha="center", fontsize=7.5, color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e",
                      edgecolor="#e74c3c", alpha=0.7, linewidth=0.8),
        )

    # ── F0 contour ──────────────────────────────────────────────────────────
    ax_p.plot(
        pitch_times, pitch_f0,
        color="#00d4ff", linewidth=1.6, alpha=0.9, zorder=4, label="F0 (voiced)",
    )
    voiced_mask = ~np.isnan(pitch_f0) & (pitch_f0 > 0)
    ax_p.scatter(
        pitch_times[voiced_mask], pitch_f0[voiced_mask],
        s=4, color="#00d4ff", alpha=0.45, zorder=5,
    )

    # ── Phrase annotations ──────────────────────────────────────────────────
    for pi, phrase in enumerate(phrases):
        pp = phrase["pitch"]
        pat = pp["pattern"]
        colour = PATTERN_COLOUR.get(pat, "#95a5a6")
        centre_t = (phrase["start"] + phrase["end"]) / 2.0
        slope_s = f"{pp['overall_slope']:+.2f} st/s" if pp["overall_slope"] is not None else ""
        vf_s = f"v={pp['voiced_fraction']:.0%}" if pp["voiced_fraction"] is not None else ""
        ann = f"{ARROW.get(pat, '?')} {pat}\n{slope_s}\n{vf_s}"
        ax_p.annotate(
            ann,
            xy=(centre_t, f0_max + f0_range * 0.12),
            ha="center", va="bottom", fontsize=8, color=colour, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor=colour, alpha=0.85, linewidth=0.9),
        )

    ax_p.set_title(
        f"{title} — {audio_filename}",
        color="#ecf0f1", fontsize=11, pad=12,
    )
    ax_p.set_ylabel("F0 (Hz)", color="#00d4ff", fontsize=10)
    ax_p.yaxis.label.set_color("#00d4ff")
    ax_p.set_xlim(0, duration)

    legend_items = [
        Line2D([0], [0], color="#00d4ff", linewidth=1.6, label="F0 contour"),
        Line2D([0], [0], color="#f39c12", linestyle="--", linewidth=1.0,
               label=f"Utterance median ({utterance_median:.1f} Hz)" if utterance_median else "Median"),
        Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=1.4, label="Phrase boundary"),
    ]
    ax_p.legend(
        handles=legend_items, loc="upper right",
        facecolor="#1a1a2e", edgecolor="#2c2c4e", labelcolor="#ecf0f1", fontsize=8,
    )

    # ── Word timeline ───────────────────────────────────────────────────────
    for i, (word, stress, start, end) in enumerate(words):
        colour = "#f39c12" if stress else "#2c5364"
        dur = max(end - start, 0.005)
        ax_w.barh(0, dur, left=start, height=0.6, color=colour,
                  alpha=0.85, edgecolor="#0a0a1a", linewidth=0.4)
        if dur > 0.06:
            ax_w.text(
                start + dur / 2, 0, word.strip(),
                ha="center", va="center", fontsize=6.5,
                color="#ecf0f1", clip_on=True,
            )

    ax_w.set_xlim(0, duration)
    ax_w.set_ylim(-0.5, 0.5)
    ax_w.set_yticks([])
    ax_w.set_xlabel("Time (seconds)", color="#a0a0c0", fontsize=9)
    ax_w.set_title("Word Timeline  (amber = stressed)", color="#a0a0c0", fontsize=8, pad=3)

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    logger.info("Plot saved to: %s", output_path)
    plt.close()


# ===========================================================================
# 6. Summary
# ===========================================================================
def print_summary() -> None:
    _section("Test Summary")
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"  {PASS}  {passed} / {total}" if failed == 0
          else f"  {passed} passed, {_RED}{failed} failed{_RST}  out of {total}")

    if failed > 0:
        print(f"\n  Failed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    {FAIL}  {name}" + (f"\n              {detail}" if detail else ""))
    print()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-phrase integration test for the pitch-analysis pipeline.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI server (default: http://127.0.0.1:8000).",
    )
    args = parser.parse_args()

    _banner("WhiStress — Multi-Phrase Integration Test")

    # 1. segment_phrases() unit tests (no audio, no model)
    run_segment_unit_tests()

    # 2. Synthetic pitched audio (rising + falling sweeps)
    try:
        run_synthetic_pitch_test(
            show_plot=False,
            output_plot=None if args.no_plot else os.path.join(SCRIPT_DIR, "test_multi_phrase_synth.png"),
        )
    except Exception:
        traceback.print_exc()

    # 3. Real-speech two-phrase audio
    try:
        run_real_speech_two_phrase_test(
            show_plot=False,
            output_plot=None if args.no_plot else os.path.join(SCRIPT_DIR, "test_multi_phrase_real.png"),
        )
    except Exception:
        traceback.print_exc()

    # 4. Live API test (non-blocking — fails gracefully if server is down)
    try:
        run_api_test(
            api_url=args.api_url,
            audio_path=os.path.join(SCRIPT_DIR, "test_two_phrases.wav"),
        )
    except Exception:
        traceback.print_exc()

    # 5. Summary
    print_summary()


if __name__ == "__main__":
    main()
