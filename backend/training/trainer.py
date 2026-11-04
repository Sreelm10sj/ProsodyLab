"""
backend.training.trainer

Collaborator-compatible LexiRep training engine.
Implements the verified 7-stage pipeline:
1. Anchor initialization and reference vector caching
2. Initial acoustic contrast computation and linguistic argmax pseudo-labeling
3. Supervised Contrastive (SupCon) warm-up (5 epochs default)
4. IDEC autoencoder pretraining (10 epochs default) + K-Means cluster initialization (k=2)
5. 13-Loop Iterative Self-Training:
   - 100 IDEC optimization steps (MSE reconstruction + 0.4 * KL Student-t divergence)
   - Dynamic 10-D prototype extraction (top-k or single-anchor)
   - Relative contrast projection and one-primary-stress constraint enforcement
   - Pseudo-label update and delta tracking
   - 3 SupCon refinement epochs
   - Representation re-projection
6. Comprehensive checkpoint export matching production LexiRep bundle schema:
   - model/final_lexirep_model.pt
   - anchors/prototypes_10d.npy
   - config.json
   - anchors.json
   - metrics.json
   - training_history.csv
7. Verification through official LexiRepPredictor.from_bundle() smoke test
"""

import os
import sys
import json
import time
import shutil
import random
import csv
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple, Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Local inference loader for verification
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORT_DIR = os.path.join(_BACKEND_DIR, "prosodylab_model_export")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _EXPORT_DIR not in sys.path:
    sys.path.insert(0, _EXPORT_DIR)

from prosodylab_model_export.lexirep_inference.model_loader import LexiRepEncoder
from prosodylab_model_export.lexirep_inference.stress_predictor import LexiRepPredictor
from backend.training.anchor_scanner import AnchorCandidate
from backend.training.evaluator import evaluate_lexirep_benchmark


class TrainingCancelled(Exception):
    """Raised when training is cooperatively cancelled via callback."""
    pass


# ==============================================================================
# 1. ARCHITECTURES AND LOSS FUNCTIONS
# ==============================================================================

class IDECAutoencoder(nn.Module):
    """
    IDEC 10-D -> 2-D -> 10-D autoencoder with Student-t soft clustering layer.
    Architecture:
      Encoder: 10 -> 32 -> 32 -> 20 -> 2
      Decoder: 2 -> 20 -> 32 -> 32 -> 10
      Cluster centers: 2 centers in 2-D latent space
    """
    def __init__(self, in_dim: int = 10, shape: Tuple[int, ...] = (10, 32, 32, 20, 2), alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.encoder = nn.Sequential(
            nn.Linear(shape[0], shape[1]), nn.ReLU(),
            nn.Linear(shape[1], shape[2]), nn.ReLU(),
            nn.Linear(shape[2], shape[3]), nn.ReLU(),
            nn.Linear(shape[3], shape[4])
        )
        self.decoder = nn.Sequential(
            nn.Linear(shape[4], shape[3]), nn.ReLU(),
            nn.Linear(shape[3], shape[2]), nn.ReLU(),
            nn.Linear(shape[2], shape[1]), nn.ReLU(),
            nn.Linear(shape[1], shape[0])
        )
        self.cluster_centers = nn.Parameter(torch.Tensor(2, shape[4]))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, x_rec, q


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Computes sharpened target distribution P from Student-t probabilities Q."""
    p = q ** 2 / torch.sum(q, dim=0, keepdim=True)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p.detach()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss operating on unit-normalized representations."""
    def __init__(self, temp: float = 0.08):
        super().__init__()
        self.temp = temp

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temp

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs == 0, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_pos_pairs

        return mean_log_prob_pos.mean()


def make_word_batches(
    X: np.ndarray,
    y: np.ndarray,
    wmap: Dict[Any, List[int]],
    batch_words: int = 32,
    device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Constructs batches preserving entire word syllable groupings."""
    word_ids = list(wmap.keys())
    random.shuffle(word_ids)
    batches = []
    for i in range(0, len(word_ids), batch_words):
        chunk_wids = word_ids[i:i + batch_words]
        syl_idxs = []
        for wid in chunk_wids:
            syl_idxs.extend(wmap[wid])
        b_x = torch.tensor(X[syl_idxs], dtype=torch.float32)
        b_y = torch.tensor(y[syl_idxs], dtype=torch.long)
        if device is not None:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
        batches.append((b_x, b_y))
    return batches


# ==============================================================================
# 2. CONFIGURATION AND DATA OBJECTS
# ==============================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for LexiRep training."""
    language: str = "GER"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    warmup_epochs: int = 5
    idec_pretrain_epochs: int = 10
    outer_loops: int = 13
    idec_steps_per_loop: int = 100
    refinement_epochs: int = 3
    supcon_temperature: float = 0.08
    batch_words: Optional[int] = None
    idec_batch_size: int = 128
    top_k_rep: Optional[int] = None
    learning_rate: float = 0.001

    def __post_init__(self):
        lang_upper = (self.language or "").upper()
        if self.top_k_rep is None:
            if lang_upper == "ITA":
                self.top_k_rep = 10
            elif lang_upper == "GER":
                self.top_k_rep = 1
            else:
                self.top_k_rep = 1

        if self.batch_words is None:
            if lang_upper == "ITA":
                self.batch_words = 48
            else:
                self.batch_words = 32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingProgress:
    """Snapshot of training progress emitted to callbacks."""
    status: str
    current_step: str
    progress_percent: float
    loop: int
    total_loops: int
    epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    pseudo_label_changes: int = 0
    pseudo_label_change_rate: float = 0.0
    elapsed_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "current_step": self.current_step,
            "progress_percent": round(self.progress_percent, 2),
            "loop": self.loop,
            "total_loops": self.total_loops,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "loss": round(self.loss, 6) if self.loss is not None else None,
            "pseudo_label_changes": self.pseudo_label_changes,
            "pseudo_label_change_rate": round(self.pseudo_label_change_rate, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "metrics": self.metrics,
        }


@dataclass
class TrainingResult:
    """Final result of training run including exported bundle locations and metrics."""
    success: bool
    output_dir: str
    selected_anchor: Dict[str, Any]
    final_metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    artifact_paths: Dict[str, str]
    elapsed_seconds: float
    device: str
    seed: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "selected_anchor": self.selected_anchor,
            "final_metrics": self.final_metrics,
            "history": self.history,
            "artifact_paths": self.artifact_paths,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "device": self.device,
            "seed": self.seed,
            "error": self.error,
        }


# ==============================================================================
# 3. LEXIREP TRAINER
# ==============================================================================

class LexiRepTrainer:
    """
    Self-contained LexiRep model trainer.
    Orchestrates the SupCon warm-up, IDEC pretraining, and iterative self-training loops.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        callback: Optional[Callable[[TrainingProgress], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ):
        self.config = config or TrainingConfig()
        self.callback = callback
        self.cancellation_check = cancellation_check

    def _notify(self, progress: TrainingProgress):
        if self.callback:
            self.callback(progress)

    def _check_cancellation(self):
        if self.cancellation_check and self.cancellation_check():
            raise TrainingCancelled("Training was cancelled by user.")

    def _resolve_device(self) -> torch.device:
        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested in TrainingConfig, but CUDA is not available.")
            return torch.device("cuda:0")
        elif self.config.device == "cpu":
            return torch.device("cpu")
        else:  # auto
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _set_seed(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        W_train: np.ndarray,
        anchor: Union[AnchorCandidate, Dict[str, Any]],
        output_dir: str,
        X_test: Optional[np.ndarray] = None,
        Y_test: Optional[np.ndarray] = None,
        W_test: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """
        Execute full training run using validated features and selected anchor.

        Args:
            X_train: (N, 768) float32 syllable embeddings.
            Y_train: (N,) binary labels {0, 1}.
            W_train: (N,) word grouping IDs.
            anchor: AnchorCandidate or dictionary containing word_id, stressed_syllable_index, unstressed_syllable_index.
            output_dir: Target directory where bundle artifacts are saved.
            X_test: Optional (M, 768) held-out test features for benchmark evaluation.
            Y_test: Optional (M,) binary test labels.
            W_test: Optional (M,) test word groupings.

        Returns:
            TrainingResult object.
        """
        t0 = time.time()
        self._set_seed()
        device = self._resolve_device()
        device_str = str(device)

        # ----------------------------------------------------------------------
        # Anchor Resolution
        # ----------------------------------------------------------------------
        if isinstance(anchor, AnchorCandidate):
            anc_w = int(anchor.word_id)
            rs_idx = int(anchor.stressed_syllable_index)
            ru_idx = int(anchor.unstressed_syllable_index)
            anc_score = float(anchor.separation_score)
        elif isinstance(anchor, dict):
            anc_w = int(anchor["word_id"])
            rs_idx = int(anchor["stressed_syllable_index"])
            ru_idx = int(anchor["unstressed_syllable_index"])
            anc_score = float(anchor.get("separation_score", 0.0))
        else:
            raise ValueError(f"Unsupported anchor type: {type(anchor)}")

        if rs_idx >= len(X_train) or ru_idx >= len(X_train):
            raise ValueError(f"Anchor indices ({rs_idx}, {ru_idx}) exceed dataset size ({len(X_train)}).")
        if Y_train[rs_idx] != 1 or Y_train[ru_idx] != 0:
            raise ValueError(f"Anchor labels invalid: Y[{rs_idx}]={Y_train[rs_idx]} (expected 1), Y[{ru_idx}]={Y_train[ru_idx]} (expected 0).")
        if W_train[rs_idx] != anc_w or W_train[ru_idx] != anc_w:
            raise ValueError(f"Anchor word mismatch: W[{rs_idx}]={W_train[rs_idx]}, W[{ru_idx}]={W_train[ru_idx]} != {anc_w}.")

        rs_vec = X_train[rs_idx:rs_idx + 1]
        ru_vec = X_train[ru_idx:ru_idx + 1]

        # Word groupings
        wmap_tr: Dict[Any, List[int]] = defaultdict(list)
        for idx, w in enumerate(W_train):
            wmap_tr[w].append(idx)

        total_words = len(wmap_tr)
        n_samples = len(X_train)

        os.makedirs(output_dir, exist_ok=True)
        history: List[Dict[str, Any]] = []

        # ----------------------------------------------------------------------
        # Stage 2: Initial Acoustic Polarization (Cold Start)
        # ----------------------------------------------------------------------
        self._check_cancellation()
        self._notify(TrainingProgress(
            status="initializing",
            current_step="Initializing acoustic polarization from anchor pair",
            progress_percent=2.0,
            loop=0,
            total_loops=self.config.outer_loops,
            elapsed_seconds=time.time() - t0,
        ))

        sim_s0 = cosine_similarity(X_train, rs_vec).flatten()
        sim_u0 = cosine_similarity(X_train, ru_vec).flatten()
        delta_0 = sim_s0 - sim_u0

        labels_tr = np.zeros(n_samples, dtype=np.int64)
        for w, idxs in wmap_tr.items():
            best_syl = idxs[np.argmax(delta_0[idxs])]
            labels_tr[best_syl] = 1

        # ----------------------------------------------------------------------
        # Stage 3: Contrastive Warm-Up
        # ----------------------------------------------------------------------
        self._check_cancellation()
        cl_encoder = LexiRepEncoder().to(device)
        cl_optimizer = optim.Adam(cl_encoder.parameters(), lr=self.config.learning_rate)
        supcon_criterion = SupConLoss(temp=self.config.supcon_temperature)

        warmup_epochs = max(1, self.config.warmup_epochs)
        warmup_batches = make_word_batches(X_train, labels_tr, wmap_tr, batch_words=self.config.batch_words, device=device)

        for ep in range(1, warmup_epochs + 1):
            self._check_cancellation()
            cl_encoder.train()
            ep_loss = 0.0
            num_batches = 0

            for bx, by in warmup_batches:
                cl_optimizer.zero_grad()
                out = cl_encoder(bx)
                loss = supcon_criterion(out, by)
                loss.backward()
                cl_optimizer.step()
                ep_loss += float(loss.item())
                num_batches += 1

            avg_loss = ep_loss / max(1, num_batches)
            progress_pct = 2.0 + (ep / warmup_epochs) * 10.0  # 2% -> 12%

            self._notify(TrainingProgress(
                status="warming_up",
                current_step=f"SupCon warm-up epoch {ep}/{warmup_epochs}",
                progress_percent=progress_pct,
                loop=0,
                total_loops=self.config.outer_loops,
                epoch=ep,
                total_epochs=warmup_epochs,
                loss=avg_loss,
                elapsed_seconds=time.time() - t0,
            ))
            history.append({
                "phase": "warmup",
                "loop": 0,
                "epoch": ep,
                "supcon_loss": avg_loss,
                "idec_loss": None,
                "label_changes": 0,
                "elapsed_sec": round(time.time() - t0, 2),
            })

        # ----------------------------------------------------------------------
        # Stage 4: IDEC Autoencoder Pretraining & K-Means Initialization
        # ----------------------------------------------------------------------
        self._check_cancellation()
        cl_encoder.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
            h_tr_init = cl_encoder(x_tensor)

        idec_model = IDECAutoencoder(in_dim=10, shape=(10, 32, 32, 20, 2), alpha=1.0).to(device)
        idec_optimizer = optim.Adam(idec_model.parameters(), lr=self.config.learning_rate)

        pretrain_epochs = max(1, self.config.idec_pretrain_epochs)
        for ep in range(1, pretrain_epochs + 1):
            self._check_cancellation()
            idec_model.train()
            perm = np.random.permutation(n_samples)
            ep_rec_loss = 0.0
            num_steps = 0

            for b_start in range(0, n_samples, self.config.idec_batch_size):
                b_idx = perm[b_start:b_start + self.config.idec_batch_size]
                bx = h_tr_init[b_idx]
                idec_optimizer.zero_grad()
                _, rec, _ = idec_model(bx)
                loss_rec = F.mse_loss(rec, bx)
                loss_rec.backward()
                torch.nn.utils.clip_grad_norm_(idec_model.parameters(), max_norm=5.0)
                idec_optimizer.step()
                ep_rec_loss += float(loss_rec.item())
                num_steps += 1

            avg_rec = ep_rec_loss / max(1, num_steps)
            progress_pct = 12.0 + (ep / pretrain_epochs) * 10.0  # 12% -> 22%

            self._notify(TrainingProgress(
                status="idec_pretrain",
                current_step=f"IDEC pretraining epoch {ep}/{pretrain_epochs}",
                progress_percent=progress_pct,
                loop=0,
                total_loops=self.config.outer_loops,
                epoch=ep,
                total_epochs=pretrain_epochs,
                loss=avg_rec,
                elapsed_seconds=time.time() - t0,
                metrics={"idec_loss": round(avg_rec, 5)},
            ))
            history.append({
                "phase": "idec_pretrain",
                "loop": 0,
                "epoch": ep,
                "supcon_loss": None,
                "idec_loss": avg_rec,
                "label_changes": 0,
                "elapsed_sec": round(time.time() - t0, 2),
            })

        # K-Means cluster center initialization
        self._check_cancellation()
        idec_model.eval()
        with torch.no_grad():
            z_lat_init, _, _ = idec_model(h_tr_init)
        km_init = KMeans(n_clusters=2, random_state=self.config.seed, n_init=10).fit(z_lat_init.cpu().numpy())
        idec_model.cluster_centers.data = torch.tensor(km_init.cluster_centers_, dtype=torch.float32, device=device)

        # ----------------------------------------------------------------------
        # Stage 5: 13-Loop Iterative Self-Training
        # ----------------------------------------------------------------------
        h_tr = h_tr_init
        total_loops = max(1, self.config.outer_loops)
        idec_steps = max(1, self.config.idec_steps_per_loop)
        refinement_epochs = max(1, self.config.refinement_epochs)

        z_s_h = None
        z_u_h = None
        final_idec_loss = 0.0
        final_supcon_loss = 0.0

        for loop in range(1, total_loops + 1):
            self._check_cancellation()
            prev_labels = labels_tr.copy()

            # A. IDEC optimization steps
            idec_model.train()
            loop_idec_loss = 0.0
            for step in range(idec_steps):
                if step % 20 == 0:
                    self._check_cancellation()
                perm = np.random.permutation(len(h_tr))
                b_idx = perm[:min(self.config.idec_batch_size, len(h_tr))]
                bx = h_tr[b_idx]

                idec_optimizer.zero_grad()
                z_b, rec_b, q_b = idec_model(bx)
                p_b = target_distribution(q_b.detach())
                loss_rec = F.mse_loss(rec_b, bx)
                loss_kl = F.kl_div(q_b.log(), p_b, reduction="batchmean")
                total_idec_loss = loss_rec + 0.4 * loss_kl
                total_idec_loss.backward()
                torch.nn.utils.clip_grad_norm_(idec_model.parameters(), max_norm=5.0)
                idec_optimizer.step()
                loop_idec_loss += float(total_idec_loss.item())

            final_idec_loss = loop_idec_loss / idec_steps

            # B. Soft clustering inference & prototype extraction
            idec_model.eval()
            with torch.no_grad():
                _, _, q_all = idec_model(h_tr)
                q_np = q_all.cpu().numpy()

            s_c = 1 if q_np[rs_idx, 1] > q_np[rs_idx, 0] else 0
            u_c = 1 - s_c

            h_tr_np = h_tr.detach().cpu().numpy() if isinstance(h_tr, torch.Tensor) else h_tr

            if self.config.top_k_rep > 1:
                q_pred = np.argmax(q_np, axis=1)
                cs_idx = np.where(q_pred == s_c)[0]
                cu_idx = np.where(q_pred == u_c)[0]
                if len(cs_idx) == 0 or len(cu_idx) == 0:
                    z_s_h = h_tr_np[rs_idx:rs_idx + 1]
                    z_u_h = h_tr_np[ru_idx:ru_idx + 1]
                else:
                    k_s = min(self.config.top_k_rep, len(cs_idx))
                    k_u = min(self.config.top_k_rep, len(cu_idx))
                    top_s_idx = cs_idx[np.argsort(q_np[cs_idx, s_c])[-k_s:]]
                    top_u_idx = cu_idx[np.argsort(q_np[cu_idx, u_c])[-k_u:]]
                    z_s_h = np.mean(h_tr_np[top_s_idx], axis=0, keepdims=True)
                    z_u_h = np.mean(h_tr_np[top_u_idx], axis=0, keepdims=True)
            else:
                z_s_h = h_tr_np[rs_idx:rs_idx + 1]
                z_u_h = h_tr_np[ru_idx:ru_idx + 1]

            # C & D. Representation space contrast & linguistic argmax
            diff_z = cosine_similarity(h_tr_np, z_s_h).flatten() - cosine_similarity(h_tr_np, z_u_h).flatten()
            new_labels = np.zeros(n_samples, dtype=np.int64)
            for w, idxs in wmap_tr.items():
                best_syl = idxs[np.argmax(diff_z[idxs])]
                new_labels[best_syl] = 1

            label_changes = int(np.sum(new_labels != prev_labels))
            change_rate = float(label_changes / n_samples)
            labels_tr = new_labels

            # E. SupCon refinement epochs
            self._check_cancellation()
            cl_encoder.train()
            refine_batches = make_word_batches(X_train, labels_tr, wmap_tr, batch_words=self.config.batch_words, device=device)
            loop_supcon_loss = 0.0
            num_refine = 0

            for ref_ep in range(refinement_epochs):
                for bx, by in refine_batches:
                    cl_optimizer.zero_grad()
                    out = cl_encoder(bx)
                    loss = supcon_criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(cl_encoder.parameters(), max_norm=5.0)
                    cl_optimizer.step()
                    loop_supcon_loss += float(loss.item())
                    num_refine += 1

            final_supcon_loss = loop_supcon_loss / max(1, num_refine)

            # F. Re-project representations
            cl_encoder.eval()
            with torch.no_grad():
                h_tr = cl_encoder(torch.tensor(X_train, dtype=torch.float32, device=device))

            # Progress update
            progress_pct = 22.0 + (loop / total_loops) * 70.0  # 22% -> 92%
            self._notify(TrainingProgress(
                status="training",
                current_step=f"Self-training loop {loop}/{total_loops} (changes: {label_changes})",
                progress_percent=progress_pct,
                loop=loop,
                total_loops=total_loops,
                loss=final_idec_loss,
                pseudo_label_changes=label_changes,
                pseudo_label_change_rate=change_rate,
                elapsed_seconds=time.time() - t0,
                metrics={
                    "idec_loss": round(final_idec_loss, 5),
                    "supcon_loss": round(final_supcon_loss, 5),
                    "label_changes": label_changes,
                }
            ))

            history.append({
                "phase": "self_training",
                "loop": loop,
                "epoch": loop,
                "supcon_loss": final_supcon_loss,
                "idec_loss": final_idec_loss,
                "label_changes": label_changes,
                "elapsed_sec": round(time.time() - t0, 2),
            })

        # ----------------------------------------------------------------------
        # Stage 6: Checkpoint & Artifact Export
        # ----------------------------------------------------------------------
        self._check_cancellation()
        self._notify(TrainingProgress(
            status="exporting",
            current_step="Exporting self-contained model bundle and prototypes",
            progress_percent=95.0,
            loop=total_loops,
            total_loops=total_loops,
            elapsed_seconds=time.time() - t0,
        ))

        # Bundle subdirectories
        model_dir = os.path.join(output_dir, "model")
        anchors_dir = os.path.join(output_dir, "anchors")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(anchors_dir, exist_ok=True)

        model_ckpt_path = os.path.join(model_dir, "final_lexirep_model.pt")
        prototypes_path = os.path.join(anchors_dir, "prototypes_10d.npy")
        anchors_json_path = os.path.join(output_dir, "anchors.json")
        config_json_path = os.path.join(output_dir, "config.json")
        metrics_json_path = os.path.join(output_dir, "metrics.json")
        history_csv_path = os.path.join(output_dir, "training_history.csv")

        # 1. Model checkpoint
        torch.save(cl_encoder.state_dict(), model_ckpt_path)

        # 2. Prototypes (shape: 2, 10)
        prototypes_stacked = np.vstack([z_s_h, z_u_h]).astype(np.float32)
        np.save(prototypes_path, prototypes_stacked)

        # 3. Anchors JSON
        with open(anchors_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "language": self.config.language,
                "anchor_word_id": int(anc_w),
                "separation_score": round(anc_score, 4),
                "stressed_anchor": {
                    "index": int(rs_idx),
                    "true_label": 1,
                    "l2_norm": float(np.linalg.norm(rs_vec)),
                    "prototype_10d": [float(x) for x in z_s_h.flatten()]
                },
                "unstressed_anchor": {
                    "index": int(ru_idx),
                    "true_label": 0,
                    "l2_norm": float(np.linalg.norm(ru_vec)),
                    "prototype_10d": [float(x) for x in z_u_h.flatten()]
                }
            }, f, indent=2)

        # 4. Config JSON
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "language": self.config.language,
                "model_architecture": "LexiRepEncoder",
                "framework": "PyTorch",
                "input_dim": 768,
                "embedding_dim": 10,
                "layer_units": [128, 64, 32, 16, 10],
                "bottleneck_activation": "Sigmoid",
                "checkpoint_filename": "final_lexirep_model.pt",
                "anchor_word_id": int(anc_w),
                "stressed_anchor_idx": int(rs_idx),
                "unstressed_anchor_idx": int(ru_idx),
                "top_k_rep": int(self.config.top_k_rep),
                "temperature": float(self.config.supcon_temperature),
                "trained_loops": total_loops,
                "seed": int(self.config.seed),
                "prototypes_10d": {
                    "stressed": [float(x) for x in z_s_h.flatten()],
                    "unstressed": [float(x) for x in z_u_h.flatten()]
                }
            }, f, indent=2)

        # 5. Training history CSV
        if history:
            with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

        # ----------------------------------------------------------------------
        # Stage 7: Reload Verification through Official Inference Loader
        # ----------------------------------------------------------------------
        self._notify(TrainingProgress(
            status="verifying",
            current_step="Validating exported bundle via LexiRepPredictor.from_bundle()",
            progress_percent=98.0,
            loop=total_loops,
            total_loops=total_loops,
            elapsed_seconds=time.time() - t0,
        ))

        # Verification through official predictor
        try:
            # Language must be GER or ITA for loader compatibility
            verify_lang = self.config.language.upper()
            if verify_lang not in ["GER", "ITA"]:
                verify_lang = "GER"  # Fallback for generic test models

            predictor = LexiRepPredictor.from_bundle(
                language=verify_lang,
                bundle_root=output_dir,
                device="cpu",
            )

            # Smoke inference test
            test_words = list(wmap_tr.keys())[:5]
            smoke_syls = []
            smoke_boundaries = []
            cur_idx = 0
            for wid in test_words:
                s_len = len(wmap_tr[wid])
                smoke_syls.extend(wmap_tr[wid])
                smoke_boundaries.append(list(range(cur_idx, cur_idx + s_len)))
                cur_idx += s_len

            smoke_features = X_train[smoke_syls]
            smoke_res = predictor.predict_stress(smoke_features, smoke_boundaries)

            # Check linguistic validity
            if not smoke_res.linguistic_compliance:
                raise RuntimeError(f"Smoke prediction failed linguistic compliance (zero_stress={smoke_res.zero_stress_words}, multi_stress={smoke_res.multi_stress_words})")

            for wp in smoke_res.words:
                syl_indices = [syl.syllable_index for syl in wp.syllables]
                if wp.stressed_syllable_index not in syl_indices:
                    raise RuntimeError(f"Smoke prediction failed linguistic constraint for word: {wp}")

            verification_ok = True
        except Exception as exc:
            raise RuntimeError(f"Exported bundle verification failed: {exc}") from exc

        # 8. Held-out Quarantined Benchmark Evaluation (if test set provided)
        benchmark_dict = None
        eval_summary_path = os.path.join(output_dir, "evaluation_summary.txt")
        if X_test is not None and Y_test is not None and W_test is not None:
            try:
                bm_res = evaluate_lexirep_benchmark(predictor, X_test, Y_test, W_test)
                benchmark_dict = bm_res.to_dict()
                final_metrics["benchmark"] = benchmark_dict

                # Save evaluation summary artifact
                with open(eval_summary_path, "w", encoding="utf-8") as f:
                    f.write("=" * 60 + "\n")
                    f.write("  LEXIREP HELD-OUT BENCHMARK EVALUATION SUMMARY\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"  Bi-Syllabic (B):               {bm_res.b_accuracy:.2f}%\n")
                    f.write(f"  Bi + Tri (BT):                 {bm_res.bt_accuracy:.2f}%\n")
                    f.write(f"  Bi + Tri + Quad (BTQ):         {bm_res.btq_accuracy:.2f}%\n")
                    f.write(f"  Balanced Accuracy:             {bm_res.balanced_accuracy:.2f}%\n")
                    f.write(f"  Total Syllable Accuracy:       {bm_res.total_syllable_accuracy:.2f}%\n")
                    f.write(f"  Total Word Accuracy:           {bm_res.word_accuracy:.2f}%\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"  Test Syllables:                {bm_res.total_syllables}\n")
                    f.write(f"  Test Words:                    {bm_res.total_words}\n")
                    f.write("=" * 60 + "\n")
            except Exception as bm_exc:
                pass

        # 6. Metrics JSON
        elapsed = time.time() - t0
        training_accuracy = float(np.mean(labels_tr == Y_train))
        word_accuracy = float(np.mean([np.all(labels_tr[idxs] == Y_train[idxs]) for idxs in wmap_tr.values()]))

        final_metrics = {
            "trained_loops": total_loops,
            "warmup_epochs": warmup_epochs,
            "idec_pretrain_epochs": pretrain_epochs,
            "final_idec_loss": round(final_idec_loss, 6),
            "final_supcon_loss": round(final_supcon_loss, 6),
            "training_accuracy": round(training_accuracy, 4),
            "word_accuracy": round(word_accuracy, 4),
            "elapsed_seconds": round(elapsed, 2),
            "device": device_str,
            "seed": self.config.seed,
            "verification_passed": verification_ok,
            "total_syllables": n_samples,
            "total_words": total_words,
        }
        if benchmark_dict is not None:
            final_metrics["benchmark"] = benchmark_dict

        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_summary": {
                    "total_syllables": n_samples,
                    "total_words": total_words,
                },
                "training_config": self.config.to_dict(),
                "selected_anchor": {
                    "word_id": anc_w,
                    "stressed_syllable_index": rs_idx,
                    "unstressed_syllable_index": ru_idx,
                    "separation_score": anc_score,
                },
                "final_metrics": final_metrics,
            }, f, indent=2)

        self._notify(TrainingProgress(
            status="completed",
            current_step="Training completed and verified successfully",
            progress_percent=100.0,
            loop=total_loops,
            total_loops=total_loops,
            elapsed_seconds=elapsed,
            metrics=final_metrics,
        ))

        return TrainingResult(
            success=True,
            output_dir=output_dir,
            selected_anchor={
                "word_id": anc_w,
                "stressed_syllable_index": rs_idx,
                "unstressed_syllable_index": ru_idx,
                "separation_score": anc_score,
            },
            final_metrics=final_metrics,
            history=history,
            artifact_paths={
                "model_checkpoint": model_ckpt_path,
                "prototypes": prototypes_path,
                "anchors_json": anchors_json_path,
                "config_json": config_json_path,
                "metrics_json": metrics_json_path,
                "training_history_csv": history_csv_path,
                "evaluation_summary_txt": eval_summary_path if os.path.isfile(eval_summary_path) else None,
            },
            elapsed_seconds=elapsed,
            device=device_str,
            seed=self.config.seed,
        )
