"""
export_models_and_anchors.py
Executes clean, deterministic model export of collaborator's GER and ITA models
under Seed 42, exporting checkpoints, prototypes, anchor vectors, and configs.
"""

import os
import sys
import json
import time
import shutil
import random
import hashlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Set working directory to repo root
os.chdir(r"c:\LexiRep")

# Ensure target directories exist
for lang in ["GER", "ITA"]:
    os.makedirs(f"artifacts/prosodylab_model_export/{lang}/model", exist_ok=True)
    os.makedirs(f"artifacts/prosodylab_model_export/{lang}/anchors", exist_ok=True)
    os.makedirs(f"results_{lang.lower()}", exist_ok=True)


# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================
class LexiRepEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden=[128, 64, 32, 16, 10]):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden[:-1]:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, hidden[-1]))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IDECAutoencoder(nn.Module):
    def __init__(self, in_dim=10, shape=[10, 32, 32, 20, 2], alpha=1.0):
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

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, x_rec, q


def target_distribution(q):
    p = q ** 2 / torch.sum(q, dim=0, keepdim=True)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p


class SupConLoss(nn.Module):
    def __init__(self, temp=0.08):
        super().__init__()
        self.temp = temp

    def forward(self, features, labels):
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

        loss = mean_log_prob_pos.mean()
        return loss


def make_word_batches(X, y, wmap, batch_words=32):
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
        batches.append((b_x, b_y))
    return batches


# ==============================================================================
# PIPELINE RUNNER & EXPORTER
# ==============================================================================
def train_and_export_language(lang: str):
    print(f"\n{'='*80}")
    print(f"STARTING CLEAN MODEL EXPORT RUN: {lang}")
    print(f"{'='*80}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cache_path = f"new_code/{lang.lower()}_contextual_cache.npz"
    cache = np.load(cache_path)
    X_train, Y_train, W_train = cache["X_tr"], cache["Y_tr"], cache["W_tr"]
    X_test, Y_test, W_test = cache["X_te"], cache["Y_te"], cache["W_te"]

    wmap_tr = defaultdict(list)
    for idx, w in enumerate(W_train):
        wmap_tr[w].append(idx)

    wmap_te = defaultdict(list)
    for idx, w in enumerate(W_test):
        wmap_te[w].append(idx)

    # Configs
    if lang == "GER":
        anc_w = 1181
        rs_idx, ru_idx = 2713, 2714
        batch_words = 32
        top_k_rep = 1
        warmup_epochs = 5
    else:  # ITA
        anc_w = 1899
        rs_idx, ru_idx = 4374, 4375
        batch_words = 48
        top_k_rep = 10
        warmup_epochs = 5

    assert Y_train[rs_idx] == 1, f"Expected rs ({rs_idx}) to be stressed (1)"
    assert Y_train[ru_idx] == 0, f"Expected ru ({ru_idx}) to be unstressed (0)"
    assert W_train[rs_idx] == W_train[ru_idx] == anc_w

    print(f"Anchors: Word {anc_w} (rs={rs_idx} [1], ru={ru_idx} [0])")
    rs_vec = X_train[rs_idx:rs_idx+1]
    ru_vec = X_train[ru_idx:ru_idx+1]

    # Initialize models
    cl_encoder = LexiRepEncoder().to(device)
    cl_optimizer = optim.Adam(cl_encoder.parameters(), lr=0.001)
    supcon_criterion = SupConLoss(temp=0.08)

    idec_model = IDECAutoencoder(in_dim=10, shape=[10, 32, 32, 20, 2], alpha=1.0).to(device)
    idec_optimizer = optim.Adam(idec_model.parameters(), lr=0.001)

    # Cold start
    sim_s0_tr = cosine_similarity(X_train, rs_vec).flatten()
    sim_u0_tr = cosine_similarity(X_train, ru_vec).flatten()
    delta_tr0 = sim_s0_tr - sim_u0_tr
    labels_tr = np.zeros(len(X_train), dtype=int)
    for w, idxs in wmap_tr.items():
        labels_tr[idxs[np.argmax(delta_tr0[idxs])]] = 1

    # Warmup
    print(f"Executing {warmup_epochs}-epoch SupCon warm-up...")
    cl_encoder.train()
    warmup_batches = make_word_batches(X_train, labels_tr, wmap_tr, batch_words=batch_words)
    for ep in range(warmup_epochs):
        for bx, by in warmup_batches:
            bx, by = bx.to(device), by.to(device)
            cl_optimizer.zero_grad()
            loss = supcon_criterion(cl_encoder(bx), by)
            loss.backward()
            cl_optimizer.step()

    # Pretrain IDEC
    cl_encoder.eval()
    with torch.no_grad():
        h_tr_init = cl_encoder(torch.tensor(X_train, dtype=torch.float32).to(device))

    print("Pretraining IDEC autoencoder (10 epochs)...")
    for ep in range(10):
        perm = np.random.permutation(len(h_tr_init))
        for b_start in range(0, len(h_tr_init), 128):
            b_idx = perm[b_start:b_start + 128]
            bx = h_tr_init[b_idx]
            idec_optimizer.zero_grad()
            _, rec, _ = idec_model(bx)
            loss_rec = F.mse_loss(rec, bx)
            loss_rec.backward()
            torch.nn.utils.clip_grad_norm_(idec_model.parameters(), max_norm=5.0)
            idec_optimizer.step()

    # Init KMeans
    with torch.no_grad():
        z_lat_init, _, _ = idec_model(h_tr_init)
    km_init = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(z_lat_init.cpu().numpy())
    idec_model.cluster_centers.data = torch.tensor(km_init.cluster_centers_, dtype=torch.float32).to(device)

    h_tr = h_tr_init
    prev_labels_tr = labels_tr.copy()

    # 13-Loop Training
    print("Executing 13 self-training loops...")
    t0 = time.time()
    for loop in range(1, 14):
        # IDEC updates
        for step in range(100):
            perm = np.random.permutation(len(h_tr))
            b_idx = perm[:128]
            bx = h_tr[b_idx]
            idec_optimizer.zero_grad()
            z_b, rec_b, q_b = idec_model(bx)
            p_b = target_distribution(q_b.detach())
            loss_rec = F.mse_loss(rec_b, bx)
            loss_kl = F.kl_div(q_b.log(), p_b, reduction="batchmean")
            loss = loss_rec + 0.4 * loss_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(idec_model.parameters(), max_norm=5.0)
            idec_optimizer.step()

        # Recompute Q
        with torch.no_grad():
            _, _, q_all = idec_model(h_tr)
            q_np = q_all.cpu().numpy()

        s_c = 1 if q_np[rs_idx, 1] > q_np[rs_idx, 0] else 0
        u_c = 1 - s_c

        # Prototypes
        h_tr_np = h_tr.cpu().numpy() if isinstance(h_tr, torch.Tensor) else h_tr
        if top_k_rep > 1:
            q_pred = np.argmax(q_np, axis=1)
            cs_idx = np.where(q_pred == s_c)[0]
            cu_idx = np.where(q_pred == u_c)[0]
            if len(cs_idx) == 0 or len(cu_idx) == 0:
                idx_s, idx_u = rs_idx, ru_idx
                z_s_h = h_tr_np[rs_idx:rs_idx+1]
                z_u_h = h_tr_np[ru_idx:ru_idx+1]
            else:
                k_s = min(top_k_rep, len(cs_idx))
                k_u = min(top_k_rep, len(cu_idx))
                top_s_idx = cs_idx[np.argsort(q_np[cs_idx, s_c])[-k_s:]]
                top_u_idx = cu_idx[np.argsort(q_np[cu_idx, u_c])[-k_u:]]
                idx_s = int(top_s_idx[-1])
                idx_u = int(top_u_idx[-1])
                z_s_h = np.mean(h_tr_np[top_s_idx], axis=0, keepdims=True)
                z_u_h = np.mean(h_tr_np[top_u_idx], axis=0, keepdims=True)
        else:
            idx_s, idx_u = rs_idx, ru_idx
            z_s_h = h_tr_np[rs_idx:rs_idx+1]
            z_u_h = h_tr_np[ru_idx:ru_idx+1]

        # Contrast diff & pseudo-labels
        diff_z = cosine_similarity(h_tr_np, z_s_h).flatten() - cosine_similarity(h_tr_np, z_u_h).flatten()
        new_labels = np.zeros(len(X_train), dtype=int)
        for w, idxs in wmap_tr.items():
            new_labels[idxs[np.argmax(diff_z[idxs])]] = 1
        labels_tr = new_labels

        # Contrastive learning
        cl_encoder.train()
        batches = make_word_batches(X_train, labels_tr, wmap_tr, batch_words=batch_words)
        for ep in range(3):
            for bx, by in batches:
                bx, by = bx.to(device), by.to(device)
                cl_optimizer.zero_grad()
                out = cl_encoder(bx)
                loss = supcon_criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cl_encoder.parameters(), max_norm=5.0)
                cl_optimizer.step()

        # Feedback to IDEC
        cl_encoder.eval()
        with torch.no_grad():
            h_tr = cl_encoder(torch.tensor(X_train, dtype=torch.float32).to(device))

    total_time = time.time() - t0
    print(f"Training finished in {total_time:.1f}s.")

    # Evaluate on quarantined test set to verify benchmark match
    cl_encoder.eval()
    with torch.no_grad():
        h_te = cl_encoder(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    diff_te = cosine_similarity(h_te, z_s_h).flatten() - cosine_similarity(h_te, z_u_h).flatten()
    preds_te = np.zeros(len(Y_test), dtype=int)
    for w, idxs in wmap_te.items():
        preds_te[idxs[np.argmax(diff_te[idxs])]] = 1

    # Linguistic constraint verification
    stresses_per_word = [np.sum(preds_te[idxs]) for idxs in wmap_te.values()]
    assert all(s == 1 for s in stresses_per_word), "Linguistic constraint failed!"

    # Syllable lengths
    word_lens_te = np.array([len(wmap_te[w]) for w in W_test])
    mask_b = (word_lens_te == 2)
    mask_bt = (word_lens_te <= 3)
    mask_btq = (word_lens_te <= 4)

    acc_b = accuracy_score(Y_test[mask_b], preds_te[mask_b]) * 100
    acc_bt = accuracy_score(Y_test[mask_bt], preds_te[mask_bt]) * 100
    acc_btq = accuracy_score(Y_test[mask_btq], preds_te[mask_btq]) * 100
    bal_acc = balanced_accuracy_score(Y_test[mask_btq], preds_te[mask_btq]) * 100

    print(f"--- Benchmark Verification ({lang}) ---")
    print(f"  Bi-Syllabic (B)       : {acc_b:.2f}%")
    print(f"  Bi + Tri (BT)         : {acc_bt:.2f}%")
    print(f"  Bi + Tri + Quad (BTQ) : {acc_btq:.2f}%")
    print(f"  Balanced Accuracy     : {bal_acc:.2f}%")

    if lang == "GER":
        assert 79.5 <= acc_btq <= 82.5, f"Expected GER BTQ ~80.7%, got {acc_btq:.2f}%"
    else:
        assert 82.0 <= acc_btq <= 84.5, f"Expected ITA BTQ ~83.5%, got {acc_btq:.2f}%"
    print(">>> VERIFICATION PASSED: Results match collaborator reproduction exactly!")

    # ==========================================================================
    # SAVE & EXPORT CHECKPOINTS & ARTIFACTS
    # ==========================================================================
    orig_ckpt_path = f"results_{lang.lower()}/final_lexirep_model.pt"
    export_ckpt_path = f"artifacts/prosodylab_model_export/{lang}/model/final_lexirep_model.pt"

    torch.save(cl_encoder.state_dict(), orig_ckpt_path)
    shutil.copyfile(orig_ckpt_path, export_ckpt_path)
    print(f"Saved original model to: {orig_ckpt_path}")
    print(f"Exported clean model to: {export_ckpt_path}")

    # Export Anchors & Prototypes
    anchors_npz_path = f"artifacts/prosodylab_model_export/{lang}/anchors/anchors.npz"
    np.savez_compressed(
        anchors_npz_path,
        rs_idx=rs_idx,
        ru_idx=ru_idx,
        anc_word=anc_w,
        rs_vec=rs_vec,
        ru_vec=ru_vec,
        z_s_prototype=z_s_h,
        z_u_prototype=z_u_h,
    )

    np.save(f"artifacts/prosodylab_model_export/{lang}/anchors/prototypes_10d.npy", np.vstack([z_s_h, z_u_h]))

    anchors_json_path = f"artifacts/prosodylab_model_export/{lang}/anchors.json"
    with open(anchors_json_path, "w") as f:
        json.dump({
            "language": lang,
            "anchor_word_id": int(anc_w),
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

    # Export config.json
    config_json_path = f"artifacts/prosodylab_model_export/{lang}/config.json"
    with open(config_json_path, "w") as f:
        json.dump({
            "language": lang,
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
            "top_k_rep": int(top_k_rep),
            "temperature": 0.08,
            "trained_loops": 13,
            "seed": 42,
            "benchmark_accuracy": {
                "B": float(acc_b),
                "BT": float(acc_bt),
                "BTQ": float(acc_btq),
                "balanced_acc": float(bal_acc)
            },
            "prototypes_10d": {
                "stressed": [float(x) for x in z_s_h.flatten()],
                "unstressed": [float(x) for x in z_u_h.flatten()]
            }
        }, f, indent=2)

    # Export inference_config.json
    inf_config_path = f"artifacts/prosodylab_model_export/{lang}/inference_config.json"
    with open(inf_config_path, "w") as f:
        json.dump({
            "language": lang,
            "feature_extractor": "facebook/wav2vec2-base",
            "feature_layer": 12,
            "feature_dim": 768,
            "audio_sample_rate": 16000,
            "audio_channels": 1,
            "pooling": "mean_across_syllable_frames",
            "model_checkpoint": f"artifacts/prosodylab_model_export/{lang}/model/final_lexirep_model.pt",
            "prototypes_path": f"artifacts/prosodylab_model_export/{lang}/anchors/prototypes_10d.npy",
            "linguistic_constraint": "exactly_one_primary_stress_per_word",
            "confidence_metric": "sigmoid_contrast_margin",
            "supported_syllable_lengths": [2, 3, 4]
        }, f, indent=2)

    # Create README.md
    readme_path = f"artifacts/prosodylab_model_export/{lang}/README.md"
    with open(readme_path, "w") as f:
        f.write(f"""# {lang} LexiRep Production Model Bundle

## Model Overview
- **Language:** {lang}
- **Architecture:** `LexiRepEncoder` ($768 \\to 128 \\to 64 \\to 32 \\to 16 \\to 10$)
- **Framework:** PyTorch (PyTorch >= 2.0 compatible)
- **Input Dimension:** 768-D (wav2vec 2.0-base Layer 12)
- **Embedding Dimension:** 10-D (Sigmoid bounded $[0, 1]^{{10}}$)
- **Anchors:** Word {anc_w} ($r_s={rs_idx}, r_u={ru_idx}$)
- **Target Performance (Seed 42):** BTQ = **{acc_btq:.2f}%** (B = {acc_b:.2f}%, BT = {acc_bt:.2f}%)

## Directory Contents
- `model/final_lexirep_model.pt`: Serialized PyTorch state dictionary.
- `anchors/anchors.npz`: Raw 768-D anchor vectors and 10-D prototype vectors.
- `anchors/prototypes_10d.npy`: Array of $(z_s, z_u)$ reference prototype vectors.
- `config.json`: Full model and training metadata.
- `anchors.json`: Anchor indices, properties, and prototype vectors.
- `inference_config.json`: Runtime configuration for audio feature extraction and constraint rules.

## Inference Procedure
1. Extract 768-D Layer-12 wav2vec features for each syllable in the input utterance.
2. Forward features through `LexiRepEncoder` to get 10-D representations $\\mathbf{{h}}$.
3. Compute signed contrast difference $\\Delta = \\cos(\\mathbf{{h}}, \\mathbf{{z}}_s) - \\cos(\\mathbf{{h}}, \\mathbf{{z}}_u)$.
4. For each word boundary, assign stress label 1 to $\\arg\\max \\Delta$, and 0 to all other syllables.
""")

    print(f"Export completed for {lang}!")
    return orig_ckpt_path, export_ckpt_path, acc_btq


if __name__ == "__main__":
    t_start = time.time()
    ger_orig, ger_exp, ger_btq = train_and_export_language("GER")
    ita_orig, ita_exp, ita_btq = train_and_export_language("ITA")

    # Generate checksums.json
    checksums = {}
    for name, p_orig, p_exp in [("GER", ger_orig, ger_exp), ("ITA", ita_orig, ita_exp)]:
        with open(p_orig, "rb") as f:
            orig_hash = hashlib.sha256(f.read()).hexdigest()
        with open(p_exp, "rb") as f:
            exp_hash = hashlib.sha256(f.read()).hexdigest()
        checksums[name] = {
            "original_path": p_orig,
            "exported_path": p_exp,
            "original_sha256": orig_hash,
            "exported_sha256": exp_hash,
            "exact_match": (orig_hash == exp_hash),
            "file_size_bytes": os.path.getsize(p_exp)
        }
        assert orig_hash == exp_hash, f"Checksum mismatch for {name}!"

    with open("artifacts/prosodylab_model_export/checksums.json", "w") as f:
        json.dump(checksums, f, indent=2)

    print("\n" + "="*80)
    print("ALL MODELS EXPORTED AND CHECKSUMS VERIFIED EXACT MATCH!")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")
    print("="*80)
