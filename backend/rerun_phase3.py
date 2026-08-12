"""Re-run phase 3 + 4 using cached inference and pre-spliced proxy files."""
import sys, os, io as _io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'WhiStress'))

from validate_acoustic_pause import (
    load_audio, get_pitch, run_whistress,
    acoustic_pause_duration, phase3_regression, phase4_gate,
)

# Phase 3 — regression
all_recordings = [
    {'path': 'test_two_phrases.wav',     'gt': None,  'label': 'two_phrases'},
    {'path': 'test_pause.mp3',           'gt': None,  'label': 'hello_are'},
    {'path': 'test_synthetic_pause.wav', 'gt': None,  'label': 'synthetic'},
    {'path': 'proxy_bug1_1700ms.wav',    'gt': 1.700, 'label': 'proxy_bug1'},
    {'path': 'proxy_bug2_780ms.wav',     'gt': 0.780, 'label': 'proxy_bug2'},
]
regression_rows, all_pass_regression = phase3_regression(all_recordings)

# Phase 4 gate — use Phase 2 measurements already captured
p2_rows = [
    {'key': 'proxy_bug1', 'path': 'proxy_bug1_1700ms.wav', 'gt': 1.700,
     'whisper_gap': 1.780, 'acoustic': 1.710,
     'whisper_badge': True, 'acoustic_badge': True, 'passes': True},
    {'key': 'proxy_bug2', 'path': 'proxy_bug2_780ms.wav',  'gt': 0.780,
     'whisper_gap': 0.900, 'acoustic': 0.790,
     'whisper_badge': True, 'acoustic_badge': True, 'passes': True},
]
all_ok = phase4_gate(p2_rows, all_pass_regression)
sys.exit(0 if all_ok else 1)
