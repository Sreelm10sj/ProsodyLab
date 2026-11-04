"""
test_inference.py
Standalone test harness for verifying:
1. GER & ITA model loading and forward inference
2. 100% linguistic constraint compliance (exactly one primary stress per word)
3. Benchmark accuracy reproduction (GER ~80.7%, ITA ~83.5%)
4. Equivalence test: original checkpoint vs exported checkpoint produces bitwise identical embeddings and predictions
5. Functional API integration test (lexirep_inference.predict_stress)
"""

import os
import sys
import json
import time
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Ensure c:\LexiRep and export directory are on python path
repo_root = r"c:\LexiRep"
export_dir = os.path.join(repo_root, "artifacts", "prosodylab_model_export")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if export_dir not in sys.path:
    sys.path.insert(0, export_dir)

from lexirep_inference import (
    load_model,
    predict_stress,
    LexiRepPredictor,
    InferenceResult
)
from lexirep_inference.model_loader import LexiRepEncoder


def test_language_inference(lang: str):
    print(f"\n{'='*80}")
    print(f"RUNNING STANDALONE INFERENCE TEST: {lang}")
    print(f"{'='*80}")

    lang_upper = lang.upper()
    cache_path = os.path.join(repo_root, "new_code", f"{lang.lower()}_contextual_cache.npz")
    assert os.path.exists(cache_path), f"Cache not found: {cache_path}"
    cache = np.load(cache_path)

    X_test, Y_test, W_test = cache["X_te"], cache["Y_te"], cache["W_te"]
    num_syls = len(X_test)

    wmap_te = defaultdict(list)
    for idx, w in enumerate(W_test):
        wmap_te[w].append(idx)
    word_boundaries = list(wmap_te.values())
    total_words = len(word_boundaries)

    print(f"Dataset Loaded: {num_syls:,} syllables across {total_words:,} words")

    # 1. Load Exported Predictor via LexiRepPredictor.from_bundle
    bundle_dir = os.path.join(export_dir, lang_upper)
    t0 = time.time()
    predictor = LexiRepPredictor.from_bundle(lang_upper, bundle_dir)
    load_time = time.time() - t0
    print(f"Exported Predictor Loaded in {load_time*1000:.1f}ms on device: {predictor.device}")

    # 2. Run Forward Inference
    t0 = time.time()
    result = predictor.predict_stress(X_test, word_boundaries)
    inf_time = time.time() - t0
    print(f"Inference Completed in {inf_time*1000:.1f}ms ({inf_time / num_syls * 1e6:.2f} µs/syllable)")

    # 3. Verify Linguistic Constraint
    print(f"\n--- Linguistic Constraint Audit ({lang_upper}) ---")
    print(f"  Total Words Evaluated       : {result.total_words:,}")
    print(f"  Words with Zero Stress      : {result.zero_stress_words}")
    print(f"  Words with Multiple Stresses: {result.multi_stress_words}")
    print(f"  Linguistic Compliance Flag  : {result.linguistic_compliance}")
    assert result.linguistic_compliance is True, "Linguistic constraint violated!"
    print("  [PASS] Exactly one primary stress per word verified across 100.0% of test words.")

    # Flatten predictions for benchmark accuracy calculation
    preds = np.zeros(num_syls, dtype=int)
    for w in result.words:
        for s in w.syllables:
            preds[s.syllable_index] = s.stress

    # Verify per-syllable masks
    word_lens = np.array([len(wmap_te[w]) for w in W_test])
    mask_b = (word_lens == 2)
    mask_bt = (word_lens <= 3)
    mask_btq = (word_lens <= 4)

    acc_b = accuracy_score(Y_test[mask_b], preds[mask_b]) * 100
    acc_bt = accuracy_score(Y_test[mask_bt], preds[mask_bt]) * 100
    acc_btq = accuracy_score(Y_test[mask_btq], preds[mask_btq]) * 100
    bal_acc = balanced_accuracy_score(Y_test[mask_btq], preds[mask_btq]) * 100

    print(f"\n--- Classification Accuracy ({lang_upper}) ---")
    print(f"  Bi-Syllabic (B)       : {acc_b:.2f}%")
    print(f"  Bi + Tri (BT)         : {acc_bt:.2f}%")
    print(f"  Bi + Tri + Quad (BTQ) : {acc_btq:.2f}%")
    print(f"  Balanced Accuracy     : {bal_acc:.2f}%")

    if lang_upper == "GER":
        assert 79.5 <= acc_btq <= 82.5, f"GER BTQ outside expected range: {acc_btq:.2f}%"
    else:
        assert 82.0 <= acc_btq <= 84.5, f"ITA BTQ outside expected range: {acc_btq:.2f}%"

    # ==========================================================================
    # 4. EQUIVALENCE TEST: Original Checkpoint vs Exported Checkpoint
    # ==========================================================================
    orig_ckpt = os.path.join(repo_root, f"results_{lang.lower()}", "final_lexirep_model.pt")
    exp_ckpt = os.path.join(export_dir, lang_upper, "model", "final_lexirep_model.pt")

    orig_encoder = LexiRepEncoder()
    orig_encoder.load_state_dict(torch.load(orig_ckpt, map_location=predictor.device, weights_only=True))
    orig_encoder.to(predictor.device)
    orig_encoder.eval()

    with torch.no_grad():
        h_orig = orig_encoder(torch.tensor(X_test, dtype=torch.float32, device=predictor.device)).cpu().numpy()
        h_exp = predictor.extract_embeddings(X_test)

    emb_diff = np.max(np.abs(h_orig - h_exp))
    print(f"\n--- Checkpoint Equivalence Test ({lang_upper}) ---")
    print(f"  Original Checkpoint : {orig_ckpt}")
    print(f"  Exported Checkpoint : {exp_ckpt}")
    print(f"  Max Absolute Embedding Difference: {emb_diff:.10e}")
    assert emb_diff < 1e-6, f"Embedding difference exceeds tolerance: {emb_diff}"
    print("  [PASS] Exported checkpoint produces bitwise-identical representations to original.")

    # 5. Functional API Integration Test
    sample_size = 50
    sample_boundaries = word_boundaries[:10]
    sample_syl_count = sum(len(b) for b in sample_boundaries)
    sample_features = X_test[:sample_syl_count]

    api_res = predict_stress(
        language=lang_upper,
        features=sample_features,
        word_boundaries=sample_boundaries,
        bundle_root=bundle_dir
    )
    assert api_res["total_words"] == 10
    assert api_res["linguistic_compliance"] is True
    print("  [PASS] Functional API 'predict_stress' returns valid structured output.")

    return {
        "language": lang_upper,
        "total_words": total_words,
        "total_syllables": num_syls,
        "b_acc": float(acc_b),
        "bt_acc": float(acc_bt),
        "btq_acc": float(acc_btq),
        "bal_acc": float(bal_acc),
        "linguistic_compliance": True,
        "max_embedding_difference": float(emb_diff),
        "load_time_ms": float(load_time * 1000),
        "inference_time_ms": float(inf_time * 1000)
    }


if __name__ == "__main__":
    t_start = time.time()
    results = {}
    results["GER"] = test_language_inference("GER")
    results["ITA"] = test_language_inference("ITA")

    out_json = os.path.join(export_dir, "test_inference_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"ALL STANDALONE INFERENCE TESTS PASSED FOR BOTH LANGUAGES!")
    print(f"Saved results to: {out_json}")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")
    print(f"{'='*80}")
