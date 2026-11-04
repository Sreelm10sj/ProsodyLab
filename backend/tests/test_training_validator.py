"""
backend.tests.test_training_validator

Comprehensive unit tests for dataset validation service in backend/training/validator.py.
Verifies all 15 specification points:
1. Valid synthetic dataset
2. Wrong X rank (1-D, 3-D)
3. Wrong feature dimension (!= 768)
4. Mismatched lengths (len(X) != len(Y), len(X) != len(W))
5. NaN features
6. Inf features
7. Non-binary labels (e.g. 2 or -1)
8. Missing stressed class (all 0)
9. Missing unstressed class (all 1)
10. Word with only 1 syllable (rejected)
11. Word with no stressed syllable (rejected)
12. Insufficient anchor candidates (< min_anchor_words)
13. Non-contiguous word IDs (accepted)
14. Optional word_texts omitted (None accepted)
15. Malformed word_texts (mismatched length, conflicting texts, missing dict keys)
16. Zero-norm vector rejection
17. to_dict() JSON serializability and DatasetValidationError
"""

import unittest
import numpy as np

from backend.training.validator import (
    validate_dataset,
    DatasetValidationResult,
    DatasetValidationError,
)


def _create_valid_synthetic_data(
    num_words: int = 60,
    dim: int = 768,
    seed: int = 42,
):
    """Generate a clean synthetic dataset with 60 words: 50 bisyllabic, 10 trisyllabic."""
    rng = np.random.default_rng(seed)
    w_list = []
    y_list = []

    # 50 bisyllabic words: 1 stressed, 1 unstressed
    for w in range(50):
        w_list.extend([w, w])
        if w % 2 == 0:
            y_list.extend([1, 0])
        else:
            y_list.extend([0, 1])

    # 10 trisyllabic words: 1 stressed, 2 unstressed
    for w in range(50, 50 + 10):
        w_list.extend([w, w, w])
        y_list.extend([1, 0, 0])

    total_syllables = len(w_list)  # 50 * 2 + 10 * 3 = 130 syllables >= 100
    X = rng.standard_normal((total_syllables, dim)).astype(np.float32)
    # Ensure non-zero norms
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * 5.0

    Y = np.array(y_list, dtype=np.int64)
    W = np.array(w_list, dtype=np.int64)

    return X, Y, W


class TestDatasetValidator(unittest.TestCase):
    """Test suite for dataset validation rules."""

    def test_01_valid_synthetic_dataset(self):
        """1. Valid synthetic dataset passes validation with complete stats."""
        X, Y, W = _create_valid_synthetic_data()
        res = validate_dataset(X, Y, W, min_syllables=100, min_anchor_words=5)

        self.assertTrue(res.valid)
        self.assertEqual(len(res.errors), 0)
        self.assertEqual(res.syllable_count, 130)
        self.assertEqual(res.word_count, 60)
        self.assertEqual(res.bisyllabic_anchor_candidate_count, 50)
        self.assertEqual(res.rejected_word_count, 10)
        self.assertEqual(res.stressed_count, 60)
        self.assertEqual(res.unstressed_count, 70)
        self.assertAlmostEqual(res.stress_ratio, 60 / 130, places=3)
        self.assertEqual(res.feature_shape, (130, 768))
        self.assertEqual(res.word_size_distribution, {2: 50, 3: 10})

    def test_02_wrong_x_rank(self):
        """2. Wrong X rank (1-D or 3-D) is rejected."""
        X, Y, W = _create_valid_synthetic_data()
        # 1-D
        res_1d = validate_dataset(X.flatten(), Y, W)
        self.assertFalse(res_1d.valid)
        self.assertTrue(any("2-D" in err for err in res_1d.errors))

        # 3-D
        res_3d = validate_dataset(X[:, :, np.newaxis], Y, W)
        self.assertFalse(res_3d.valid)
        self.assertTrue(any("2-D" in err for err in res_3d.errors))

    def test_03_wrong_feature_dimension(self):
        """3. Feature dimension != 768 is rejected."""
        X, Y, W = _create_valid_synthetic_data(dim=512)
        res = validate_dataset(X, Y, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("768" in err and "512" in err for err in res.errors))

    def test_04_mismatched_lengths(self):
        """4. Length mismatch between X, Y, and W is rejected."""
        X, Y, W = _create_valid_synthetic_data()
        # Y shorter
        res_y = validate_dataset(X, Y[:-5], W)
        self.assertFalse(res_y.valid)
        self.assertTrue(any("Length mismatch" in err for err in res_y.errors))

        # W shorter
        res_w = validate_dataset(X, Y, W[:-5])
        self.assertFalse(res_w.valid)
        self.assertTrue(any("Length mismatch" in err for err in res_w.errors))

    def test_05_nan_features(self):
        """5. NaN values in X are rejected."""
        X, Y, W = _create_valid_synthetic_data()
        X_corrupt = X.copy()
        X_corrupt[5, 10] = np.nan
        res = validate_dataset(X_corrupt, Y, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("NaN" in err for err in res.errors))

    def test_06_inf_features(self):
        """6. Inf values in X are rejected."""
        X, Y, W = _create_valid_synthetic_data()
        X_corrupt = X.copy()
        X_corrupt[12, 45] = np.inf
        res = validate_dataset(X_corrupt, Y, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("infinite" in err for err in res.errors))

    def test_07_non_binary_labels(self):
        """7. Non-binary labels (values other than 0 and 1) are rejected."""
        X, Y, W = _create_valid_synthetic_data()
        Y_corrupt = Y.copy()
        Y_corrupt[10] = 2
        res = validate_dataset(X, Y_corrupt, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("binary labels" in err for err in res.errors))

    def test_08_missing_stressed_class(self):
        """8. Missing stressed class (all 0) is rejected."""
        X, Y, W = _create_valid_synthetic_data()
        Y_all_zero = np.zeros_like(Y)
        res = validate_dataset(X, Y_all_zero, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("no stressed syllables" in err for err in res.errors))

    def test_09_missing_unstressed_class(self):
        """9. Missing unstressed class (all 1) is rejected."""
        X, Y, W = _create_valid_synthetic_data()
        Y_all_one = np.ones_like(Y)
        res = validate_dataset(X, Y_all_one, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("no unstressed syllables" in err for err in res.errors))

    def test_10_word_with_only_one_syllable(self):
        """10. Words with only 1 syllable violate LexiRep requirements and are rejected."""
        X, Y, W = _create_valid_synthetic_data()
        # Make syllable 0 its own isolated word ID 9999
        W_corrupt = W.copy()
        W_corrupt[0] = 9999
        res = validate_dataset(X, Y, W_corrupt)
        self.assertFalse(res.valid)
        self.assertTrue(any("only 1 syllable" in err for err in res.errors))

    def test_11_word_with_no_stressed_syllable(self):
        """11. Words with no stressed syllables are rejected."""
        X, Y, W = _create_valid_synthetic_data()
        # Word 0 has syllables at indices 0, 1. Set both to unstressed (0)
        Y_corrupt = Y.copy()
        Y_corrupt[0] = 0
        Y_corrupt[1] = 0
        res = validate_dataset(X, Y_corrupt, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("no stressed syllables" in err for err in res.errors))

    def test_12_insufficient_anchor_candidates(self):
        """12. Datasets with fewer than min_anchor_words bisyllabic candidates are rejected."""
        X, Y, W = _create_valid_synthetic_data()
        # Set min_anchor_words higher than available candidates (50)
        res = validate_dataset(X, Y, W, min_anchor_words=60)
        self.assertFalse(res.valid)
        self.assertTrue(any("Insufficient bisyllabic anchor" in err for err in res.errors))

    def test_13_non_contiguous_word_ids(self):
        """13. Arbitrary, non-contiguous, non-zero-based word IDs are accepted."""
        X, Y, W = _create_valid_synthetic_data()
        # Map word IDs to large sparse integers: 1000, 2500, 999999...
        unique_w = np.unique(W)
        sparse_map = {old_w: old_w * 1000 + 7 for old_w in unique_w}
        W_sparse = np.array([sparse_map[w] for w in W], dtype=np.int64)

        res = validate_dataset(X, Y, W_sparse, min_syllables=100, min_anchor_words=5)
        self.assertTrue(res.valid)
        self.assertEqual(res.word_count, len(unique_w))
        self.assertEqual(res.bisyllabic_anchor_candidate_count, 50)

    def test_14_optional_word_texts_omitted(self):
        """14. Omitting word_texts (word_texts=None) is completely valid."""
        X, Y, W = _create_valid_synthetic_data()
        res = validate_dataset(X, Y, W, word_texts=None)
        self.assertTrue(res.valid)

    def test_15_word_texts_validation(self):
        """15. Valid word_texts are accepted; malformed/inconsistent word_texts are rejected."""
        X, Y, W = _create_valid_synthetic_data()
        unique_w = np.unique(W)

        # A. Valid dictionary mapping
        valid_dict = {int(w): f"word_{w}" for w in unique_w}
        res_dict = validate_dataset(X, Y, W, word_texts=valid_dict)
        self.assertTrue(res_dict.valid)

        # B. Incomplete dictionary mapping
        incomplete_dict = {int(w): f"word_{w}" for w in unique_w[:10]}
        res_incomp = validate_dataset(X, Y, W, word_texts=incomplete_dict)
        self.assertFalse(res_incomp.valid)
        self.assertTrue(any("missing" in err for err in res_incomp.errors))

        # C. Valid syllable-aligned sequence
        valid_seq = [f"word_{w}" for w in W]
        res_seq = validate_dataset(X, Y, W, word_texts=valid_seq)
        self.assertTrue(res_seq.valid)

        # D. Sequence with length mismatch
        res_seq_mismatch = validate_dataset(X, Y, W, word_texts=valid_seq[:-10])
        self.assertFalse(res_seq_mismatch.valid)
        self.assertTrue(any("does not match syllable count" in err for err in res_seq_mismatch.errors))

        # E. Sequence with conflicting texts within the same word ID
        corrupt_seq = list(valid_seq)
        # Syllables 0 and 1 belong to word 0: assign conflicting texts
        corrupt_seq[0] = "apple"
        corrupt_seq[1] = "banana"
        res_conf = validate_dataset(X, Y, W, word_texts=corrupt_seq)
        self.assertFalse(res_conf.valid)
        self.assertTrue(any("Conflicting word_texts" in err for err in res_conf.errors))

    def test_16_zero_norm_vector_rejection(self):
        """16. Syllable embedding with zero or near-zero norm (<1e-12) is rejected."""
        X, Y, W = _create_valid_synthetic_data()
        X_zero = X.copy()
        X_zero[15] = 0.0  # Zero vector
        res = validate_dataset(X_zero, Y, W)
        self.assertFalse(res.valid)
        self.assertTrue(any("zero or near-zero norm" in err for err in res.errors))

    def test_17_to_dict_and_exception(self):
        """17. Validation result serializes cleanly to dict; DatasetValidationError holds result."""
        X, Y, W = _create_valid_synthetic_data()
        res = validate_dataset(X, Y, W)
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["valid"], True)
        self.assertEqual(d["syllable_count"], 130)
        self.assertEqual(d["word_count"], 60)
        self.assertIn("word_size_distribution", d)

        # Exception packaging
        exc = DatasetValidationError("Validation failed", res)
        self.assertIs(exc.result, res)
        self.assertIn("Validation failed", str(exc))


if __name__ == "__main__":
    unittest.main()
