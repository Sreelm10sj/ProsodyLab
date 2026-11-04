"""
stress_predictor.py
Core inference engine executing LexiRep neural forward pass, relative acoustic contrast,
and linguistic one-primary-stress constraint enforcement.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import torch

from .schemas import SyllablePrediction, WordPrediction, InferenceResult
from .feature_interface import validate_features, validate_word_groupings
from .model_loader import LexiRepEncoder, load_lexirep_bundle


class LexiRepPredictor:
    """
    Production-ready lexical stress predictor for German and Italian.
    Encapsulates PyTorch encoder and prototype-relative contrast evaluation.
    """
    def __init__(
        self,
        language: str,
        encoder: LexiRepEncoder,
        z_s: np.ndarray,
        z_u: np.ndarray,
        config: Dict[str, Any]
    ):
        self.language = language.upper()
        self.encoder = encoder
        self.z_s = z_s.astype(np.float32)
        self.z_u = z_u.astype(np.float32)
        self.config = config

        # Precompute L2 norms of reference prototypes
        self._norm_zs = float(np.linalg.norm(self.z_s)) + 1e-9
        self._norm_zu = float(np.linalg.norm(self.z_u)) + 1e-9

        # Device detection
        self.device = next(encoder.parameters()).device

    @classmethod
    def from_bundle(
        cls,
        language: str,
        bundle_root: Optional[str] = None,
        device: Optional[str] = None
    ) -> "LexiRepPredictor":
        """
        Factory method to instantiate a predictor directly from an exported model bundle.
        """
        encoder, z_s, z_u, config = load_lexirep_bundle(language, bundle_root, device)
        return cls(language, encoder, z_s, z_u, config)

    def extract_embeddings(self, features: np.ndarray) -> np.ndarray:
        """
        Extracts 10-D unit-bounded representations h in [0, 1]^10 from 768-D features.
        """
        features_clean = validate_features(features)
        self.encoder.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(features_clean, dtype=torch.float32, device=self.device)
            h = self.encoder(x_tensor).cpu().numpy()
        return h

    def predict_stress(
        self,
        features: np.ndarray,
        word_boundaries: List[List[int]],
        word_texts: Optional[List[str]] = None
    ) -> InferenceResult:
        """
        Performs end-to-end stress prediction on syllable features.

        Args:
            features: (N, 768) float32 array of wav2vec representations.
            word_boundaries: List of syllable index lists, e.g. [[0, 1], [2, 3, 4], ...].
            word_texts: Optional list of word text strings matching word_boundaries length.

        Returns:
            InferenceResult object with structured word and syllable predictions.
        """
        features_clean = validate_features(features)
        num_syls = len(features_clean)
        boundaries_clean = validate_word_groupings(num_syls, word_boundaries)

        # 1. Forward Pass to 10-D Latent Space
        self.encoder.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(features_clean, dtype=torch.float32, device=self.device)
            h = self.encoder(x_tensor).cpu().numpy()  # (N, 10)

        # 2. Compute Relative Signed Cosine Contrast: Δ = cos(h, z_s) - cos(h, z_u)
        norm_h = np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        sim_s = (h @ self.z_s.T) / (norm_h * self._norm_zs)  # (N, 1)
        sim_u = (h @ self.z_u.T) / (norm_h * self._norm_zu)  # (N, 1)
        diff = (sim_s - sim_u).flatten()                     # (N,)

        # 3. Linguistic Constraint Enforcement (argmax per word)
        word_predictions: List[WordPrediction] = []
        zero_stress_count = 0
        multi_stress_count = 0

        for w_idx, syl_indices in enumerate(boundaries_clean):
            w_text = word_texts[w_idx] if (word_texts and w_idx < len(word_texts)) else f"word_{w_idx}"
            
            # Find best syllable
            local_argmax = int(np.argmax(diff[syl_indices]))
            stressed_global_idx = syl_indices[local_argmax]
            best_diff = float(diff[stressed_global_idx])

            # Calculate confidence margin
            other_diffs = [float(diff[i]) for i in syl_indices if i != stressed_global_idx]
            if len(other_diffs) > 0:
                margin = best_diff - max(other_diffs)
                # Calibrate margin via sigmoid into [0.5, 1.0]
                conf = float(1.0 / (1.0 + np.exp(-margin * 2.0)))
            else:
                conf = 1.0  # Monosyllabic trivial case

            # Build SyllablePrediction list
            syl_preds: List[SyllablePrediction] = []
            stresses_in_word = 0
            for idx in syl_indices:
                is_stressed = 1 if idx == stressed_global_idx else 0
                stresses_in_word += is_stressed
                syl_preds.append(SyllablePrediction(
                    syllable_index=idx,
                    stress=is_stressed,
                    contrast_score=float(diff[idx]),
                    confidence=conf if is_stressed else float(1.0 - conf)
                ))

            if stresses_in_word == 0:
                zero_stress_count += 1
            elif stresses_in_word > 1:
                multi_stress_count += 1

            word_predictions.append(WordPrediction(
                word_id=w_idx,
                word_text=w_text,
                stressed_syllable_index=stressed_global_idx,
                syllables=syl_preds
            ))

        total_words = len(boundaries_clean)
        linguistic_compliance = (zero_stress_count == 0 and multi_stress_count == 0)

        return InferenceResult(
            language=self.language,
            total_words=total_words,
            total_syllables=num_syls,
            linguistic_compliance=linguistic_compliance,
            zero_stress_words=zero_stress_count,
            multi_stress_words=multi_stress_count,
            words=word_predictions
        )
