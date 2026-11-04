"""
backend.training.evaluator

Reconstructed LexiRep Offline Benchmark Evaluator.
Implements the exact historical benchmark methodology:
1. Syllables are mapped to 10-D representations via LexiRepEncoder.
2. Signed relative cosine contrast:
     Delta_i = cos(h_i, z_s) - cos(h_i, z_u)
3. Constrained one-primary-stress prediction:
     y_i* = 1 if i == argmax_{j in w} Delta_j else 0
4. Word-length subsets (L = syllable count of word):
     - B: L == 2 (Bisyllabic words)
     - BT: L in (2, 3) (Bi- and Trisyllabic words)
     - BTQ: L in (2, 3, 4) (Bi-, Tri-, and Quadrisyllabic words)
5. Metrics:
     - B Accuracy: Syllable classification accuracy on B subset (%)
     - BT Accuracy: Syllable classification accuracy on BT subset (%)
     - BTQ Accuracy: Syllable classification accuracy on BTQ subset (%)
     - Balanced Accuracy: balanced_accuracy_score on BTQ subset (%)
     - Total Syllable Accuracy: Full syllable accuracy over all test syllables (%)
     - Word Accuracy: Exact-word primary stress placement across ALL test words (%)

Verified targets on GER production bundle (7,000 syllables, 3,048 words):
  B = 79.91%
  BT = 80.20%
  BTQ = 79.96%
  Balanced Accuracy = 79.70%
  Total Syllable Accuracy = 79.83%
  Word Accuracy = 76.84%
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

# Ensure backend modules are importable
_TESTS_OR_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_OR_TRAIN_DIR)
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)
for p in [_ROOT_DIR, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prosodylab_model_export.lexirep_inference.stress_predictor import LexiRepPredictor


@dataclass
class BenchmarkResult:
    """Structured result of the LexiRep benchmark evaluation."""
    # Summary percentages (0.0 to 100.0)
    b_accuracy: float
    bt_accuracy: float
    btq_accuracy: float
    balanced_accuracy: float
    total_syllable_accuracy: float
    word_accuracy: float

    # Total sample counts
    total_syllables: int
    total_words: int
    correct_syllables: int
    correct_words: int

    # Subset statistics
    b_words: int
    b_syllables: int
    b_correct_syllables: int

    bt_words: int
    bt_syllables: int
    bt_correct_syllables: int

    btq_words: int
    btq_syllables: int
    btq_correct_syllables: int

    # Word-length distribution
    word_length_distribution: Dict[int, int] = field(default_factory=dict)
    word_length_accuracies: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize benchmark results to dictionary with clean decimal formatting."""
        d = asdict(self)
        d["word_length_distribution"] = {str(k): v for k, v in self.word_length_distribution.items()}
        d["word_length_accuracies"] = {str(k): round(v, 2) for k, v in self.word_length_accuracies.items()}
        return d


def evaluate_lexirep_benchmark(
    predictor: LexiRepPredictor,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    W_test: np.ndarray,
) -> BenchmarkResult:
    """
    Evaluate a LexiRep predictor against held-out test features using the official benchmark protocol.

    Args:
        predictor: Instantiated LexiRepPredictor with loaded encoder and prototypes.
        X_test: (N, 768) float32 syllable features.
        Y_test: (N,) binary ground-truth stress labels {0, 1}.
        W_test: (N,) word grouping IDs.

    Returns:
        BenchmarkResult with B, BT, BTQ, balanced, syllable, and word accuracies.
    """
    X_arr = np.asarray(X_test, dtype=np.float32)
    Y_arr = np.asarray(Y_test, dtype=np.int64)
    W_arr = np.asarray(W_test)

    num_syllables = len(X_arr)
    if num_syllables == 0:
        raise ValueError("X_test is empty (0 syllables).")
    if len(Y_arr) != num_syllables or len(W_arr) != num_syllables:
        raise ValueError(
            f"Dimension mismatch: X={num_syllables}, Y={len(Y_arr)}, W={len(W_arr)}."
        )

    # 1. Group syllables into words preserving order
    wmap: Dict[Any, List[int]] = defaultdict(list)
    for idx, wid in enumerate(W_arr):
        wmap[wid].append(idx)

    word_boundaries = list(wmap.values())
    total_words = len(word_boundaries)

    # 2. Run stress prediction via official predictor
    # Features are passed through encoder, contrast is computed against prototypes,
    # and local argmax is applied within each word boundary.
    res = predictor.predict_stress(X_arr, word_boundaries)

    # 3. Reconstruct predicted binary syllable labels
    preds_syl = np.zeros(num_syllables, dtype=np.int64)
    for wp in res.words:
        preds_syl[wp.stressed_syllable_index] = 1

    # 4. Compute subset masks based on word length L(w)
    syl_lengths = np.zeros(num_syllables, dtype=np.int32)
    word_len_counts: Dict[int, int] = defaultdict(int)
    word_len_correct: Dict[int, int] = defaultdict(int)

    correct_words = 0
    for wid, idxs in wmap.items():
        w_len = len(idxs)
        word_len_counts[w_len] += 1
        for i in idxs:
            syl_lengths[i] = w_len

        # Word is correct if all syllable predictions match gold labels
        if np.all(preds_syl[idxs] == Y_arr[idxs]):
            correct_words += 1
            word_len_correct[w_len] += 1

    mask_b = (syl_lengths == 2)
    mask_bt = np.isin(syl_lengths, [2, 3])
    mask_btq = np.isin(syl_lengths, [2, 3, 4])

    # 5. Calculate syllable-level subset accuracies
    b_total = int(np.sum(mask_b))
    b_correct = int(np.sum(preds_syl[mask_b] == Y_arr[mask_b])) if b_total > 0 else 0
    b_acc = (b_correct / b_total * 100.0) if b_total > 0 else 0.0

    bt_total = int(np.sum(mask_bt))
    bt_correct = int(np.sum(preds_syl[mask_bt] == Y_arr[mask_bt])) if bt_total > 0 else 0
    bt_acc = (bt_correct / bt_total * 100.0) if bt_total > 0 else 0.0

    btq_total = int(np.sum(mask_btq))
    btq_correct = int(np.sum(preds_syl[mask_btq] == Y_arr[mask_btq])) if btq_total > 0 else 0
    btq_acc = (btq_correct / btq_total * 100.0) if btq_total > 0 else 0.0

    # 6. Balanced accuracy on BTQ syllable subset
    if btq_total > 0:
        bal_acc = float(balanced_accuracy_score(Y_arr[mask_btq], preds_syl[mask_btq]) * 100.0)
    else:
        bal_acc = 0.0

    # 7. Total syllable and word accuracies
    tot_correct_syl = int(np.sum(preds_syl == Y_arr))
    tot_syl_acc = (tot_correct_syl / num_syllables * 100.0)
    word_acc = (correct_words / total_words * 100.0) if total_words > 0 else 0.0

    # Subset word counts
    b_words = word_len_counts.get(2, 0)
    bt_words = b_words + word_len_counts.get(3, 0)
    btq_words = bt_words + word_len_counts.get(4, 0)

    # Word-length specific word accuracies
    word_len_accs: Dict[int, float] = {}
    for length, cnt in sorted(word_len_counts.items()):
        corr = word_len_correct.get(length, 0)
        word_len_accs[length] = (corr / cnt * 100.0) if cnt > 0 else 0.0

    return BenchmarkResult(
        b_accuracy=round(b_acc, 2),
        bt_accuracy=round(bt_acc, 2),
        btq_accuracy=round(btq_acc, 2),
        balanced_accuracy=round(bal_acc, 2),
        total_syllable_accuracy=round(tot_syl_acc, 2),
        word_accuracy=round(word_acc, 2),
        total_syllables=num_syllables,
        total_words=total_words,
        correct_syllables=tot_correct_syl,
        correct_words=correct_words,
        b_words=b_words,
        b_syllables=b_total,
        b_correct_syllables=b_correct,
        bt_words=bt_words,
        bt_syllables=bt_total,
        bt_correct_syllables=bt_correct,
        btq_words=btq_words,
        btq_syllables=btq_total,
        btq_correct_syllables=btq_correct,
        word_length_distribution=dict(word_len_counts),
        word_length_accuracies=word_len_accs,
    )


def evaluate_bundle_on_dataset(
    bundle_path: str,
    dataset_path: str,
    language: str = "GER",
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Convenience runner to evaluate an exported bundle directly from disk paths.
    """
    if not os.path.isdir(bundle_path):
        raise FileNotFoundError(f"Bundle directory not found: {bundle_path}")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load predictor
    predictor = LexiRepPredictor.from_bundle(
        language=language.upper(),
        bundle_root=bundle_path,
        device=device,
    )

    # Load dataset
    data = np.load(dataset_path)
    if "X_te" in data and "Y_te" in data and "W_te" in data:
        X_te, Y_te, W_te = data["X_te"], data["Y_te"], data["W_te"]
    elif "X_test" in data and "Y_test" in data and "W_test" in data:
        X_te, Y_te, W_te = data["X_test"], data["Y_test"], data["W_test"]
    else:
        raise KeyError(
            f"Dataset '{dataset_path}' does not contain test arrays ('X_te', 'Y_te', 'W_te')."
        )

    return evaluate_lexirep_benchmark(predictor, X_te, Y_te, W_te)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LexiRep model against quarantined test benchmark.")
    parser.add_argument("--bundle", required=True, help="Path to exported LexiRep model bundle directory")
    parser.add_argument("--dataset", required=True, help="Path to .npz dataset file with X_te, Y_te, W_te")
    parser.add_argument("--language", default="GER", help="Language code (GER, ITA, etc.)")
    parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    args = parser.parse_args()

    result = evaluate_bundle_on_dataset(
        bundle_path=args.bundle,
        dataset_path=args.dataset,
        language=args.language,
        device=args.device,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("  LEXIREP QUARANTINED TEST BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Bi-Syllabic (B):               {result.b_accuracy:.2f}%  ({result.b_correct_syllables}/{result.b_syllables} syls, {result.b_words} words)")
        print(f"  Bi + Tri (BT):                 {result.bt_accuracy:.2f}%  ({result.bt_correct_syllables}/{result.bt_syllables} syls, {result.bt_words} words)")
        print(f"  Bi + Tri + Quad (BTQ):         {result.btq_accuracy:.2f}%  ({result.btq_correct_syllables}/{result.btq_syllables} syls, {result.btq_words} words)")
        print(f"  Balanced Accuracy:             {result.balanced_accuracy:.2f}%  (BTQ subset)")
        print(f"  Total Syllable Accuracy:       {result.total_syllable_accuracy:.2f}%  ({result.correct_syllables}/{result.total_syllables} syls)")
        print(f"  Total Word Accuracy:           {result.word_accuracy:.2f}%  ({result.correct_words}/{result.total_words} words)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
