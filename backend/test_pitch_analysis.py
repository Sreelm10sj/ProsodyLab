#!/usr/bin/env python3
"""
test_pitch_analysis.py
======================
Standalone verification tool for the WhiStress pitch-analysis pipeline.

Runs the complete pipeline (audio → WhiStress word timestamps → pitch
extraction → phrase segmentation → intonation analysis) without starting
FastAPI or the frontend server.

Usage
-----
    python test_pitch_analysis.py
    python test_pitch_analysis.py path/to/audio.wav
    python test_pitch_analysis.py path/to/audio.mp3 --no-plot
    python test_pitch_analysis.py path/to/audio.wav --skip-model
    python test_pitch_analysis.py path/to/audio.wav --output-plot out.png

Arguments
---------
    audio_file      Audio file to analyse (default: sample.flac).
                    Accepts any format supported by pydub/ffmpeg.

    --no-plot       Skip matplotlib plot generation.

    --skip-model    Skip WhiStress model loading and use the entire
                    recording as a single dummy phrase.  Useful for
                    quickly testing pitch extraction and contour
                    classification in isolation without waiting for
                    Whisper to load.

    --output-plot   Save the plot to a file instead of showing it
                    interactively (e.g. --output-plot result.png).
"""

import argparse
import io
import logging
import os
import sys
import time
from typing import List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Force UTF-8 output on Windows — box-drawing and arrow glyphs need it
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup — runnable from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
WHISTRESS_DIR = os.path.join(SCRIPT_DIR, "WhiStress")
if WHISTRESS_DIR not in sys.path:
    sys.path.insert(0, WHISTRESS_DIR)

# ---------------------------------------------------------------------------
# Logging — make the parselmouth fallback warning clearly visible
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("verify_pitch")

# ---------------------------------------------------------------------------
# pitch_analysis imports
# ---------------------------------------------------------------------------
from pitch_analysis import (
    PHRASE_BOUNDARY_THRESHOLD_S,
    VOICED_MIN_FRACTION,
    VOICED_MIN_FRAMES_FOR_SLOPE,
    SLOPE_RISING_THRESHOLD,
    SLOPE_FALLING_THRESHOLD,
    TERMINAL_WINDOW_FRACTION,
    PITCH_TIME_STEP,
    PITCH_FMIN_HZ,
    PITCH_FMAX_HZ,
    extract_pitch,
    extract_pitch_features,
    classify_contour,
    segment_phrases,
    build_phrase_response,
)

# ---------------------------------------------------------------------------
# Terminal display helpers
# ---------------------------------------------------------------------------
_ANSI = {
    "rising":               "\033[32m",   # green
    "falling":              "\033[34m",   # blue
    "level":                "\033[33m",   # amber
    "insufficient_voicing": "\033[90m",   # grey
    "reset":                "\033[0m",
    "bold":                 "\033[1m",
    "dim":                  "\033[2m",
}

CONTOUR_ARROW = {
    "rising":               "↗",
    "falling":              "↘",
    "level":                "→",
    "insufficient_voicing": "~",
}

# Matplotlib colours per pattern (used in plot)
PLOT_COLOUR = {
    "rising":               "#2ecc71",
    "falling":              "#3498db",
    "level":                "#f39c12",
    "insufficient_voicing": "#95a5a6",
}


def _c(key: str, text: str) -> str:
    """Wrap text with an ANSI colour or style."""
    return f"{_ANSI.get(key, '')}{text}{_ANSI['reset']}"


def _fmt(value: Optional[float], unit: str = "", decimals: int = 3) -> str:
    """Format an optional float for display; returns 'N/A' when None."""
    if value is None:
        return _c("dim", "N/A")
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}{unit}"


def _banner(title: str, width: int = 68) -> str:
    dbl = "═" * width
    return f"\n╔{dbl}╗\n║{title:^{width}}║\n╚{dbl}╝"


def _section(title: str, width: int = 68) -> str:
    return f"\n{'─' * width}\n  {_c('bold', title)}\n{'─' * width}"


# ---------------------------------------------------------------------------
# Extractor detection
# ---------------------------------------------------------------------------
def detect_extractor() -> str:
    """Return the name of the pitch extractor that will be active."""
    try:
        import parselmouth  # noqa: F401
        return "Parselmouth (Praat)"
    except ImportError:
        return "librosa.pyin  [fallback — parselmouth not installed]"


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Load any audio file to a mono 16 kHz float32 numpy array.

    Uses the same pydub + soundfile path as the production service so
    that verification results are directly comparable.
    """
    import io
    from pydub import AudioSegment
    import soundfile as sf

    logger.info("Loading audio: %s", audio_path)
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)

    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    array, sr = sf.read(wav_io, dtype="float32")
    logger.info("Audio loaded — duration: %.2f s, sample rate: %d Hz", len(array) / sr, sr)
    return array, sr


# ---------------------------------------------------------------------------
# WhiStress word inference
# ---------------------------------------------------------------------------
def run_whistress(audio_array: np.ndarray, sr: int, device: str = "cpu"):
    """
    Load the WhiStress model and run inference on the audio array.

    Returns a list of (word, stress, start, end) tuples, identical to
    what the production service returns from predict().
    """
    from whistress import WhiStressInferenceClient

    logger.info("Loading WhiStress model on %s ...", device)
    t0 = time.time()
    client = WhiStressInferenceClient(device=device)
    logger.info("Model loaded in %.1f s", time.time() - t0)

    audio_dict = {"array": audio_array, "sampling_rate": sr}
    logger.info("Running WhiStress inference ...")
    t0 = time.time()
    words = client.predict(audio=audio_dict, transcription=None, return_pairs=True)
    logger.info("Inference complete in %.1f s — %d words", time.time() - t0, len(words))
    return words


def make_dummy_words(duration: float) -> List[Tuple]:
    """
    Create a single dummy word spanning the entire recording.

    Used with --skip-model to test pitch analysis without loading Whisper.
    """
    return [("[audio]", 0, 0.0, duration)]


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
def print_report(
    audio_path: str,
    duration: float,
    sr: int,
    extractor: str,
    words: List[Tuple],
    pitch_times: np.ndarray,
    pitch_f0: np.ndarray,
    phrases: List[dict],
    skip_model: bool,
) -> None:
    W = 68

    print(_banner("WhiStress Pitch Analysis — Verification", W))

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n  {'Audio file':<22}: {os.path.basename(audio_path)}")
    print(f"  {'Duration':<22}: {duration:.3f} s")
    print(f"  {'Sample rate':<22}: {sr:,} Hz")
    print(f"  {'Pitch extractor':<22}: {extractor}")
    if skip_model:
        print(f"  {'WhiStress model':<22}: {_c('dim', 'skipped (--skip-model)')}")

    # ── Utterance-level pitch stats ─────────────────────────────────────────
    voiced_all = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    total_frames = len(pitch_f0)

    if len(voiced_all) > 0:
        median_hz = float(np.median(voiced_all))
        voiced_pct = 100.0 * len(voiced_all) / total_frames
        print(f"\n  {'Utterance median F0':<22}: {median_hz:.1f} Hz")
        print(f"  {'Utterance voiced':<22}: {voiced_pct:.1f}%  "
              f"({len(voiced_all)}/{total_frames} frames @ {PITCH_TIME_STEP*1000:.0f} ms/frame)")
    else:
        print(f"\n  {_c('dim', 'No voiced frames detected — all phrases will be insufficient_voicing')}")
        median_hz = None

    # ── Active configuration ─────────────────────────────────────────────────
    print(_section("CONFIGURATION  (active thresholds)", W))
    cfg = [
        ("Phrase boundary threshold",   f"{PHRASE_BOUNDARY_THRESHOLD_S:.3f} s"),
        ("Voiced min fraction",         f"{VOICED_MIN_FRACTION:.2f}"),
        ("Voiced min frames (slope)",   str(VOICED_MIN_FRAMES_FOR_SLOPE)),
        ("Rising slope threshold",      f"≥ {SLOPE_RISING_THRESHOLD:+.1f} st/s"),
        ("Falling slope threshold",     f"≤ {SLOPE_FALLING_THRESHOLD:+.1f} st/s"),
        ("Terminal window",             f"last {TERMINAL_WINDOW_FRACTION*100:.0f}% of voiced frames"),
        ("Pitch analysis range",        f"{PITCH_FMIN_HZ:.0f}–{PITCH_FMAX_HZ:.0f} Hz"),
        ("Frame step",                  f"{PITCH_TIME_STEP*1000:.0f} ms"),
    ]
    for key, val in cfg:
        print(f"  {key:<30}: {val}")

    # ── Word list ────────────────────────────────────────────────────────────
    if words and not skip_model:
        print(_section(f"WORDS  ({len(words)} total from WhiStress)", W))
        print(f"  {'#':>3}  {'Word':<18}  {'Stressed':<8}  {'Start':>7}  {'End':>7}  {'Dur':>6}")
        print(f"  {'─'*3}  {'─'*18}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")
        for i, (word, stress, start, end) in enumerate(words):
            stressed_str = _c("bold", "YES") if stress else _c("dim", "no")
            dur = end - start
            print(f"  {i:>3}  {word:<18}  {stressed_str:<8}  {start:>7.3f}  {end:>7.3f}  {dur:>6.3f}")

    # ── Phrase analysis ──────────────────────────────────────────────────────
    print(_section(
        f"PHRASES  ({len(phrases)} found, boundary threshold = {PHRASE_BOUNDARY_THRESHOLD_S} s)",
        W,
    ))

    for pi, phrase in enumerate(phrases):
        p = phrase["pitch"]
        pattern = p["pattern"]
        arrow = CONTOUR_ARROW.get(pattern, "?")
        indices = phrase["word_indices"]
        phrase_dur = phrase["end"] - phrase["start"]

        if words:
            phrase_text = " ".join(words[i][0].strip() for i in indices)
            phrase_text = phrase_text[:50] + ("…" if len(phrase_text) > 50 else "")
            phrase_text = f'"{phrase_text}"'
        else:
            phrase_text = "(no word data)"

        label_str = _c(pattern, f"{arrow} {pattern.upper()}")

        print(f"\n  {'─'*W}")
        print(f"  Phrase {pi + 1} / {len(phrases)}")
        print(f"  {'─'*W}")
        print(f"  {'Words':<22}: {phrase_text}")
        if len(indices) > 1:
            print(f"  {'Word indices':<22}: {indices[0]} – {indices[-1]}  ({len(indices)} words)")
        else:
            print(f"  {'Word indices':<22}: {indices[0]}")
        print(f"  {'Time span':<22}: {phrase['start']:.3f} s → {phrase['end']:.3f} s  "
              f"({phrase_dur:.3f} s)")
        print(f"  {'Voiced fraction':<22}: {_fmt(p['voiced_fraction'], '', 2)}")
        print(f"  {'Median (utterance-rel)':<22}: {_fmt(p['median_semitone'], ' st')}")
        print(f"  {'Overall slope':<22}: {_fmt(p['overall_slope'], ' st/s')}")
        print(f"  {'Terminal slope':<22}: {_fmt(p['terminal_slope'], ' st/s')}")
        print(f"  {'Pattern':<22}: {label_str}")

        # Show gap to next phrase
        if pi < len(phrases) - 1:
            gap = phrases[pi + 1]["start"] - phrase["end"]
            gap_str = _c("dim", f"{gap:.3f} s pause → new phrase boundary")
            print(f"  {'Boundary gap':<22}: {gap_str}")

    print(f"\n{'─' * W}\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def generate_plot(
    audio_path: str,
    duration: float,
    words: List[Tuple],
    pitch_times: np.ndarray,
    pitch_f0: np.ndarray,
    phrases: List[dict],
    output_path: Optional[str] = None,
    skip_model: bool = False,
) -> None:
    """
    Generate a two-panel pitch contour plot with phrase boundaries overlaid.

    Top panel   : F0 contour (Hz) with phrase regions, median line,
                  phrase boundaries, and contour-type annotations.
    Bottom panel: Word timeline with stress colouring (if model was run).
    """
    try:
        import matplotlib
        if output_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        logger.warning(
            "matplotlib is not installed — skipping plot generation.  "
            "Install it with: pip install matplotlib"
        )
        return

    voiced_all = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    utterance_median_hz = float(np.median(voiced_all)) if len(voiced_all) > 0 else None

    # Compute semitone track for secondary y-axis
    semitone_track = np.full_like(pitch_f0, np.nan)
    if utterance_median_hz is not None:
        voiced_mask = ~np.isnan(pitch_f0) & (pitch_f0 > 0)
        semitone_track[voiced_mask] = 12.0 * np.log2(
            pitch_f0[voiced_mask] / utterance_median_hz
        )

    n_panels = 2 if (words and not skip_model) else 1
    fig_height = 7 if n_panels == 2 else 5

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, fig_height),
        gridspec_kw={"height_ratios": [4, 1] if n_panels == 2 else [4]},
        facecolor="#1a1a2e",
    )
    if n_panels == 1:
        axes = [axes]

    ax_pitch = axes[0]
    ax_pitch.set_facecolor("#16213e")
    fig.patch.set_facecolor("#1a1a2e")

    # ── Phrase region shading ────────────────────────────────────────────────
    phrase_colours = ["#2c3e6b", "#1e4d3b", "#4a2040", "#3b3a1e", "#1e3a4a"]
    for pi, phrase in enumerate(phrases):
        colour = phrase_colours[pi % len(phrase_colours)]
        ax_pitch.axvspan(
            phrase["start"], phrase["end"],
            alpha=0.25, color=colour, zorder=1,
        )

    # ── Utterance median line ────────────────────────────────────────────────
    if utterance_median_hz is not None:
        ax_pitch.axhline(
            utterance_median_hz,
            color="#f39c12", linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=2, label=f"Utterance median ({utterance_median_hz:.1f} Hz)",
        )

    # ── Phrase boundary lines ────────────────────────────────────────────────
    for pi in range(len(phrases) - 1):
        gap_centre = (phrases[pi]["end"] + phrases[pi + 1]["start"]) / 2.0
        ax_pitch.axvline(
            gap_centre, color="#e74c3c", linestyle=":", linewidth=1.2,
            alpha=0.8, zorder=3,
        )

    # ── Voiced/unvoiced F0 track ─────────────────────────────────────────────
    # Plot unvoiced regions as a very dim baseline
    ax_pitch.plot(
        pitch_times, np.where(np.isnan(pitch_f0), 0, np.nan),
        color="#4a4a6a", linewidth=0.5, alpha=0.3, zorder=3,
    )
    # Plot voiced F0 as a bright line (NaN gaps break the line automatically)
    ax_pitch.plot(
        pitch_times, pitch_f0,
        color="#00d4ff", linewidth=1.5, alpha=0.9, zorder=4, label="F0 (voiced)",
    )
    # Voiced frame dots
    voiced_mask = ~np.isnan(pitch_f0) & (pitch_f0 > 0)
    ax_pitch.scatter(
        pitch_times[voiced_mask], pitch_f0[voiced_mask],
        s=3, color="#00d4ff", alpha=0.5, zorder=5,
    )

    # ── Phrase annotations ───────────────────────────────────────────────────
    f0_max = np.nanmax(pitch_f0) if len(voiced_all) > 0 else 300.0
    f0_min = np.nanmin(pitch_f0[voiced_mask]) if voiced_mask.any() else 80.0
    f0_range = f0_max - f0_min

    for pi, phrase in enumerate(phrases):
        p = phrase["pitch"]
        pattern = p["pattern"]
        arrow = CONTOUR_ARROW.get(pattern, "?")
        colour = PLOT_COLOUR.get(pattern, "#95a5a6")
        centre_t = (phrase["start"] + phrase["end"]) / 2.0

        # Annotation text
        vf_str = f"{p['voiced_fraction']:.0%}" if p["voiced_fraction"] is not None else "?"
        slope_str = f"{p['overall_slope']:+.1f} st/s" if p["overall_slope"] is not None else ""
        ann_text = f"{arrow}\n{pattern}\n{slope_str}\nv={vf_str}"

        # Y position: slightly above the max F0
        y_ann = f0_max + f0_range * 0.06

        ax_pitch.annotate(
            ann_text,
            xy=(centre_t, y_ann),
            fontsize=8,
            color=colour,
            ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#1a1a2e",
                edgecolor=colour,
                alpha=0.8,
                linewidth=0.8,
            ),
        )

    # ── Secondary y-axis: semitones ──────────────────────────────────────────
    ax2 = ax_pitch.twinx()
    ax2.set_facecolor("#16213e")
    ax2.plot(
        pitch_times, semitone_track,
        color="#ff6b9d", linewidth=0.8, alpha=0.5, linestyle="-", zorder=3,
    )
    ax2.axhline(0, color="#f39c12", linewidth=0.5, linestyle="--", alpha=0.3)
    ax2.set_ylabel("Semitones (re. utterance median)", color="#ff6b9d", fontsize=9)
    ax2.tick_params(colors="#ff6b9d")
    ax2.yaxis.label.set_color("#ff6b9d")

    # ── Pitch axis labels ────────────────────────────────────────────────────
    title_str = f"Pitch Contour + Phrase Intonation — {os.path.basename(audio_path)}"
    ax_pitch.set_title(title_str, color="#ecf0f1", fontsize=11, pad=10)
    ax_pitch.set_ylabel("F0 (Hz)", color="#00d4ff", fontsize=10)
    ax_pitch.set_xlim(0, duration)
    ax_pitch.tick_params(colors="#a0a0c0")
    ax_pitch.xaxis.label.set_color("#a0a0c0")
    ax_pitch.spines[:].set_color("#2c2c4e")
    ax_pitch.yaxis.label.set_color("#00d4ff")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#00d4ff", linewidth=1.5, label="F0 (voiced)"),
        Line2D([0], [0], color="#f39c12", linewidth=1.0, linestyle="--",
               label=f"Utterance median ({utterance_median_hz:.1f} Hz)" if utterance_median_hz else "Median"),
        Line2D([0], [0], color="#e74c3c", linewidth=1.2, linestyle=":",
               label="Phrase boundary"),
        Line2D([0], [0], color="#ff6b9d", linewidth=0.8, linestyle="-",
               label="Semitones (right axis)", alpha=0.5),
    ]
    ax_pitch.legend(
        handles=legend_elements, loc="upper right",
        facecolor="#1a1a2e", edgecolor="#2c2c4e",
        labelcolor="#ecf0f1", fontsize=8,
    )

    # ── Word timeline panel ──────────────────────────────────────────────────
    if n_panels == 2 and words:
        ax_words = axes[1]
        ax_words.set_facecolor("#16213e")
        ax_words.spines[:].set_color("#2c2c4e")

        for i, (word, stress, start, end) in enumerate(words):
            bar_colour = "#f39c12" if stress else "#2c5364"
            dur = max(end - start, 0.01)
            ax_words.barh(
                0, dur, left=start, height=0.6,
                color=bar_colour, alpha=0.8, edgecolor="#0a0a1a", linewidth=0.5,
            )
            if dur > 0.05:
                ax_words.text(
                    start + dur / 2, 0, word.strip(),
                    ha="center", va="center",
                    fontsize=7, color="#ecf0f1", clip_on=True,
                    fontweight="bold" if stress else "normal",
                )

        # Phrase boundary lines on word panel
        for pi in range(len(phrases) - 1):
            gap_centre = (phrases[pi]["end"] + phrases[pi + 1]["start"]) / 2.0
            ax_words.axvline(gap_centre, color="#e74c3c", linestyle=":", linewidth=1.2, alpha=0.8)

        ax_words.set_xlim(0, duration)
        ax_words.set_ylim(-0.5, 0.5)
        ax_words.set_yticks([])
        ax_words.set_xlabel("Time (seconds)", color="#a0a0c0", fontsize=9)
        ax_words.tick_params(colors="#a0a0c0")

        # Legend patches
        stressed_patch = mpatches.Patch(color="#f39c12", label="Stressed")
        unstressed_patch = mpatches.Patch(color="#2c5364", label="Unstressed")
        ax_words.legend(
            handles=[stressed_patch, unstressed_patch],
            loc="upper right", facecolor="#1a1a2e", edgecolor="#2c2c4e",
            labelcolor="#ecf0f1", fontsize=7,
        )
        ax_words.set_title("Word Timeline", color="#a0a0c0", fontsize=8, pad=4)

    if n_panels == 1:
        axes[0].set_xlabel("Time (seconds)", color="#a0a0c0", fontsize=9)

    plt.tight_layout(pad=1.5)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        logger.info("Plot saved to: %s", output_path)
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Edge-case summary
# ---------------------------------------------------------------------------
def check_edge_cases(
    phrases: List[dict],
    words: List[Tuple],
    pitch_f0: np.ndarray,
) -> None:
    """Print a brief summary of any edge-case conditions detected."""
    issues = []

    voiced_all = pitch_f0[~np.isnan(pitch_f0) & (pitch_f0 > 0)]
    voiced_pct = len(voiced_all) / max(len(pitch_f0), 1)

    if voiced_pct == 0.0:
        issues.append("No voiced frames detected — audio may be silent or noise-only.")
    elif voiced_pct < 0.20:
        issues.append(
            f"Very low overall voicing ({voiced_pct:.0%}) — "
            "most phrases will be insufficient_voicing."
        )

    if len(phrases) == 1:
        issues.append(
            "No pause exceeds the phrase boundary threshold "
            f"({PHRASE_BOUNDARY_THRESHOLD_S} s) — entire recording is one phrase."
        )

    all_insuf = all(p["pitch"]["pattern"] == "insufficient_voicing" for p in phrases)
    if all_insuf and len(phrases) > 0:
        issues.append(
            "All phrases are classified as insufficient_voicing.  "
            "Consider lowering VOICED_MIN_FRACTION or checking the audio quality."
        )

    if not words:
        issues.append("No word data available (--skip-model mode).")

    if issues:
        print(_section("EDGE-CASE NOTES", 68))
        for note in issues:
            print(f"  ⚠  {_c('dim', note)}")
        print()
    else:
        print(f"\n  ✓  No edge-case conditions detected.\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone verification tool for the WhiStress pitch-analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="sample.flac",
        help="Audio file to analyse (default: sample.flac)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help=(
            "Skip WhiStress model loading.  Treats the entire recording as a "
            "single phrase.  Useful for quick pitch-only testing."
        ),
    )
    parser.add_argument(
        "--output-plot",
        metavar="FILE",
        default=None,
        help="Save the plot to a file (e.g. result.png) instead of showing it.",
    )
    args = parser.parse_args()

    audio_path = args.audio_file
    if not os.path.isfile(audio_path):
        # Try relative to SCRIPT_DIR
        candidate = os.path.join(SCRIPT_DIR, audio_path)
        if os.path.isfile(candidate):
            audio_path = candidate
        else:
            logger.error("Audio file not found: %s", audio_path)
            sys.exit(1)

    # ── Step 1: Detect pitch extractor ──────────────────────────────────────
    extractor = detect_extractor()
    logger.info("Pitch extractor: %s", extractor)

    # ── Step 2: Load audio ───────────────────────────────────────────────────
    audio_array, sr = load_audio(audio_path)
    duration = len(audio_array) / sr

    # ── Step 3: WhiStress word inference (or dummy data) ────────────────────
    if args.skip_model:
        logger.info("--skip-model: using dummy single-phrase word data")
        words = make_dummy_words(duration)
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            words = run_whistress(audio_array, sr, device=device)
        except Exception as exc:
            logger.error(
                "WhiStress inference failed: %s\n"
                "  Tip: run with --skip-model to test pitch analysis only.",
                exc,
            )
            sys.exit(1)

    # ── Step 4: Pitch extraction ─────────────────────────────────────────────
    logger.info("Running pitch extraction ...")
    t0 = time.time()
    pitch_times, pitch_f0 = extract_pitch(audio_array, sr=sr)
    logger.info(
        "Pitch extracted in %.2f s — %d frames (%.0f ms/frame)",
        time.time() - t0,
        len(pitch_times),
        PITCH_TIME_STEP * 1000,
    )

    # ── Step 5: Phrase segmentation + intonation analysis ───────────────────
    logger.info("Segmenting phrases and analysing intonation ...")
    phrases = build_phrase_response(words, pitch_times, pitch_f0)
    logger.info("%d phrase(s) detected", len(phrases))

    # ── Step 6: Console report ───────────────────────────────────────────────
    print_report(
        audio_path=audio_path,
        duration=duration,
        sr=sr,
        extractor=extractor,
        words=words,
        pitch_times=pitch_times,
        pitch_f0=pitch_f0,
        phrases=phrases,
        skip_model=args.skip_model,
    )

    # ── Step 7: Edge-case check ──────────────────────────────────────────────
    check_edge_cases(phrases, words, pitch_f0)

    # ── Step 8: Plot ─────────────────────────────────────────────────────────
    if not args.no_plot:
        logger.info("Generating plot ...")
        generate_plot(
            audio_path=audio_path,
            duration=duration,
            words=words,
            pitch_times=pitch_times,
            pitch_f0=pitch_f0,
            phrases=phrases,
            output_path=args.output_plot,
            skip_model=args.skip_model,
        )


if __name__ == "__main__":
    main()
