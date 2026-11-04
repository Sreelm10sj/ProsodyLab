"""
backend.training.anchor_scanner

This scanner reproduces the collaborator's verified anchor-search formulation:
unit-normalized wav2vec syllable embeddings,
stressed-vs-unstressed separation direction,
constrained word-level comparison,
weighted separation score.

Mathematically verified against:
- GER contextual cache: Word 1181 (s=2713, u=2714) with S_train = 68.2231%
- ITA contextual cache: Word 1899 (s=4374, u=4375) with S_train = 72.7906%
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Sequence, Union
import numpy as np
import torch


@dataclass
class AnchorCandidate:
    """Represents a single ranked candidate anchor pair."""
    rank: int
    word_id: int
    stressed_syllable_index: int
    unstressed_syllable_index: int
    separation_score: float
    stressed_vector_norm: float
    unstressed_vector_norm: float
    word_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "word_id": int(self.word_id),
            "stressed_syllable_index": int(self.stressed_syllable_index),
            "unstressed_syllable_index": int(self.unstressed_syllable_index),
            "separation_score": round(self.separation_score, 4),
            "stressed_vector_norm": round(self.stressed_vector_norm, 4),
            "unstressed_vector_norm": round(self.unstressed_vector_norm, 4),
            "word_text": self.word_text,
        }


@dataclass
class AnchorScanResult:
    """Structured result of the complete anchor scan procedure."""
    language: Optional[str]
    total_training_words: int
    total_training_syllables: int
    bisyllabic_candidates: int
    rejected_words: int
    scan_time_seconds: float
    top_anchors: List[AnchorCandidate] = field(default_factory=list)
    selected_anchor: Optional[AnchorCandidate] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "total_training_words": self.total_training_words,
            "total_training_syllables": self.total_training_syllables,
            "bisyllabic_candidates": self.bisyllabic_candidates,
            "rejected_words": self.rejected_words,
            "scan_time_seconds": round(self.scan_time_seconds, 3),
            "top_anchors": [cand.to_dict() for cand in self.top_anchors],
            "selected_anchor": self.selected_anchor.to_dict() if self.selected_anchor else None,
        }


class AnchorScanner:
    """
    Service executing defensible, train-set-only anchor pair discovery.
    Calculates S_train separation score across all bisyllabic word pairs.
    """

    def __init__(
        self,
        max_candidates: Optional[int] = 20,
        batch_size: int = 500,
        device: str = "cpu",
    ):
        """
        Args:
            max_candidates: Maximum ranked candidates to return in top_anchors (None = return all).
            batch_size: Projection batch size for candidate anchor evaluations (default: 500).
            device: Computing device for tensor operations ('cpu' or 'cuda').
        """
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.device = device

    def scan(
        self,
        X: Any,
        Y: Any,
        W: Any,
        word_texts: Optional[Union[Dict[int, str], Sequence[str]]] = None,
        language: Optional[str] = None,
    ) -> AnchorScanResult:
        """
        Execute anchor scan over feature cache.

        Args:
            X: Syllable feature matrix (N, 768).
            Y: Syllable binary stress labels (N,).
            W: Word grouping IDs (N,).
            word_texts: Optional orthographic strings.
            language: Optional dataset language identifier.

        Returns:
            AnchorScanResult containing ranked candidate pairs and rank 1 selected anchor.
        """
        t0 = time.time()

        # Defensive checks
        if not isinstance(X, (np.ndarray, torch.Tensor)):
            X = np.asarray(X)
        if not isinstance(Y, (np.ndarray, torch.Tensor)):
            Y = np.asarray(Y)
        if not isinstance(W, (np.ndarray, torch.Tensor)):
            W = np.asarray(W)

        if X.ndim != 2 or X.shape[1] != 768:
            raise ValueError(f"X must have shape (N, 768), got {X.shape}.")
        if Y.ndim != 1 or len(Y) != len(X):
            raise ValueError(f"Y must have shape (N,), got {Y.shape} vs len(X)={len(X)}.")
        if W.ndim != 1 or len(W) != len(X):
            raise ValueError(f"W must have shape (N,), got {W.shape} vs len(X)={len(X)}.")

        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Dataset is empty (N=0).")

        # Convert to PyTorch tensors on specified device
        target_device = torch.device(self.device)
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=target_device)
        Y_tensor = torch.as_tensor(Y, dtype=torch.long, device=target_device)
        W_list = W.tolist() if isinstance(W, (np.ndarray, torch.Tensor)) else list(W)

        # Check for NaN / Inf
        if torch.isnan(X_tensor).any():
            raise ValueError("X contains NaN values.")
        if torch.isinf(X_tensor).any():
            raise ValueError("X contains Inf values.")

        # Zero-norm check
        raw_norms = torch.norm(X_tensor, p=2, dim=1, keepdim=True)
        zero_mask = raw_norms < 1e-12
        if zero_mask.any():
            zero_indices = torch.where(zero_mask.squeeze(1))[0].tolist()
            raise ValueError(
                f"X contains zero-norm or near-zero-norm feature vector at syllable index {zero_indices[0]}."
            )

        # L2-normalize representations row-wise
        X_norm = X_tensor / raw_norms

        # Construct word map: word_id -> list of syllable indices
        # Defaultdict preserves insertion order in Python 3.7+
        wmap: Dict[Any, List[int]] = defaultdict(list)
        for idx, w_id in enumerate(W_list):
            wmap[w_id].append(idx)

        total_words = len(wmap)

        # Find all bisyllabic words with 1 stressed and 1 unstressed syllable
        b_words: List[Any] = []
        for w_id, idxs in wmap.items():
            if len(idxs) == 2 and Y_tensor[idxs[0]].item() != Y_tensor[idxs[1]].item():
                b_words.append(w_id)

        bisyllabic_candidates = len(b_words)
        rejected_words = total_words - bisyllabic_candidates

        if bisyllabic_candidates == 0:
            scan_time = time.time() - t0
            return AnchorScanResult(
                language=language,
                total_training_words=total_words,
                total_training_syllables=n_samples,
                bisyllabic_candidates=0,
                rejected_words=rejected_words,
                scan_time_seconds=scan_time,
                top_anchors=[],
                selected_anchor=None,
            )

        # Pre-group training words by structure for fast evaluation
        m2_s: List[int] = []
        m2_u: List[int] = []
        other_words: List[Tuple[int, List[int]]] = []

        for w_id, idxs in wmap.items():
            s_candidates = [i for i in idxs if Y_tensor[i].item() == 1]
            u_candidates = [i for i in idxs if Y_tensor[i].item() == 0]

            if not s_candidates:
                continue
            s_idx = s_candidates[0]

            if len(u_candidates) == 1 and len(idxs) == 2:
                m2_s.append(s_idx)
                m2_u.append(u_candidates[0])
            else:
                other_words.append((s_idx, u_candidates))

        m2_s_tensor = torch.tensor(m2_s, dtype=torch.long, device=target_device)
        m2_u_tensor = torch.tensor(m2_u, dtype=torch.long, device=target_device)

        # Build candidate anchor list (word_id, s_idx, u_idx)
        candidates: List[Tuple[Any, int, int]] = []
        s_cand_list: List[int] = []
        u_cand_list: List[int] = []

        for w_id in b_words:
            idxs = wmap[w_id]
            s = idxs[0] if Y_tensor[idxs[0]].item() == 1 else idxs[1]
            u = idxs[1] if Y_tensor[idxs[0]].item() == 1 else idxs[0]
            candidates.append((w_id, s, u))
            s_cand_list.append(s)
            u_cand_list.append(u)

        s_cand_tensor = torch.tensor(s_cand_list, dtype=torch.long, device=target_device)
        u_cand_tensor = torch.tensor(u_cand_list, dtype=torch.long, device=target_device)

        # Exact collaborator scoring formula
        cand_accs: List[float] = []
        base_correct = n_samples - 2 * total_words
        num_candidates = len(candidates)

        for b_start in range(0, num_candidates, self.batch_size):
            b_end = min(b_start + self.batch_size, num_candidates)
            sb = s_cand_tensor[b_start:b_end]
            ub = u_cand_tensor[b_start:b_end]

            diff_vec = X_norm[sb] - X_norm[ub]
            diffs = torch.mm(X_norm, diff_vec.t())  # Shape: (N_train, b_size)

            if len(m2_s_tensor) > 0:
                m2_correct = (diffs[m2_s_tensor] > diffs[m2_u_tensor]).long().sum(dim=0)
            else:
                m2_correct = torch.zeros(b_end - b_start, dtype=torch.long, device=target_device)

            other_correct = torch.zeros(b_end - b_start, dtype=torch.long, device=target_device)
            if other_words:
                for s_idx, u_idxs in other_words:
                    if u_idxs:
                        u_max = torch.max(diffs[u_idxs], dim=0).values
                        other_correct += (diffs[s_idx] > u_max).long()

            tot_correct = base_correct + (m2_correct + other_correct) * 2
            b_accs = (tot_correct.float() / n_samples * 100.0).tolist()
            cand_accs.extend(b_accs)

        # Structure and rank candidates
        raw_norms_flat = raw_norms.squeeze(1)

        ranked_items = [
            (
                candidates[i][0],  # word_id
                candidates[i][1],  # stressed_index
                candidates[i][2],  # unstressed_index
                cand_accs[i],      # separation_score
            )
            for i in range(num_candidates)
        ]

        # Python's list.sort is stable: identical scores preserve candidate insertion order
        ranked_items.sort(key=lambda x: x[3], reverse=True)

        ranked_candidates: List[AnchorCandidate] = []
        for rank_idx, (w_id, s_idx, u_idx, score) in enumerate(ranked_items, start=1):
            s_norm = float(raw_norms_flat[s_idx].item())
            u_norm = float(raw_norms_flat[u_idx].item())

            # Resolve word text if available
            w_text: Optional[str] = None
            if word_texts is not None:
                if isinstance(word_texts, dict):
                    w_text = word_texts.get(w_id)
                elif isinstance(word_texts, (list, tuple, np.ndarray)) and s_idx < len(word_texts):
                    w_text = str(word_texts[s_idx])

            candidate = AnchorCandidate(
                rank=rank_idx,
                word_id=int(w_id),
                stressed_syllable_index=int(s_idx),
                unstressed_syllable_index=int(u_idx),
                separation_score=score,
                stressed_vector_norm=s_norm,
                unstressed_vector_norm=u_norm,
                word_text=w_text,
            )
            ranked_candidates.append(candidate)

        top_anchors = (
            ranked_candidates[: self.max_candidates]
            if self.max_candidates is not None
            else ranked_candidates
        )
        selected_anchor = ranked_candidates[0] if ranked_candidates else None
        scan_time = time.time() - t0

        return AnchorScanResult(
            language=language,
            total_training_words=total_words,
            total_training_syllables=n_samples,
            bisyllabic_candidates=bisyllabic_candidates,
            rejected_words=rejected_words,
            scan_time_seconds=scan_time,
            top_anchors=top_anchors,
            selected_anchor=selected_anchor,
        )

    @classmethod
    def compute_single_anchor_score(
        cls,
        X: Any,
        Y: Any,
        W: Any,
        s_idx: int,
        u_idx: int,
        device: str = "cpu",
    ) -> float:
        """
        Recomputes exact separation score for a specific anchor syllable pair.
        Used for fast server-side verification of client-supplied anchors.
        """
        target_device = torch.device(device)
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=target_device)
        else:
            X_tensor = X.to(dtype=torch.float32, device=target_device)

        if not isinstance(Y, torch.Tensor):
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=target_device)
        else:
            Y_tensor = Y.to(dtype=torch.long, device=target_device)

        W_list = W.tolist() if isinstance(W, (np.ndarray, torch.Tensor)) else list(W)
        n_samples = X_tensor.shape[0]

        raw_norms = torch.norm(X_tensor, p=2, dim=1, keepdim=True)
        X_norm = X_tensor / (raw_norms + 1e-12)

        wmap: Dict[Any, List[int]] = defaultdict(list)
        for idx, w_id in enumerate(W_list):
            wmap[w_id].append(idx)

        total_words = len(wmap)
        base_correct = n_samples - 2 * total_words

        m2_s: List[int] = []
        m2_u: List[int] = []
        other_words: List[Tuple[int, List[int]]] = []

        for w_id, idxs in wmap.items():
            s_candidates = [i for i in idxs if Y_tensor[i].item() == 1]
            u_candidates = [i for i in idxs if Y_tensor[i].item() == 0]
            if not s_candidates:
                continue
            s = s_candidates[0]
            if len(u_candidates) == 1 and len(idxs) == 2:
                m2_s.append(s)
                m2_u.append(u_candidates[0])
            else:
                other_words.append((s, u_candidates))

        m2_s_tensor = torch.tensor(m2_s, dtype=torch.long, device=target_device)
        m2_u_tensor = torch.tensor(m2_u, dtype=torch.long, device=target_device)

        diff_vec = (X_norm[s_idx] - X_norm[u_idx]).unsqueeze(1)
        diffs = torch.mm(X_norm, diff_vec).squeeze(1)

        if len(m2_s_tensor) > 0:
            m2_correct = (diffs[m2_s_tensor] > diffs[m2_u_tensor]).long().sum()
        else:
            m2_correct = torch.tensor(0, dtype=torch.long, device=target_device)

        other_correct = 0
        for s, u_idxs in other_words:
            if u_idxs:
                u_max = torch.max(diffs[u_idxs])
                if diffs[s] > u_max:
                    other_correct += 1

        tot_correct = base_correct + (int(m2_correct.item()) + other_correct) * 2
        return float(tot_correct) / n_samples * 100.0
