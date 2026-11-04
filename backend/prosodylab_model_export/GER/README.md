# GER LexiRep Production Model Documentation

### Model
German (GER)

### Source
- Training Code: [`new_code/LexiRep_GER.ipynb`](file:///c:/LexiRep/new_code/LexiRep_GER.ipynb)
- Dataset Cache: [`new_code/ger_contextual_cache.npz`](file:///c:/LexiRep/new_code/ger_contextual_cache.npz)
- Reference Script: [`artifacts/friend_reproduction/reproduction_runs/run_friend_ger.py`](file:///c:/LexiRep/artifacts/friend_reproduction/reproduction_runs/run_friend_ger.py)

### Checkpoint
- Exported Checkpoint: [`model/final_lexirep_model.pt`](file:///c:/LexiRep/artifacts/prosodylab_model_export/GER/model/final_lexirep_model.pt)
- Original Source Path: `results_ger/final_lexirep_model.pt`
- File Size: 441,888 bytes

### Architecture
- Model Class: `LexiRepEncoder` (PyTorch `nn.Module`)
- Layer Specification: `Linear(768, 128) -> ReLU() -> Linear(128, 64) -> ReLU() -> Linear(64, 32) -> ReLU() -> Linear(32, 16) -> ReLU() -> Linear(16, 10) -> Sigmoid()`
- Input Dimension: 768-D
- Latent Representation: 10-D unit-bounded in $[0, 1]^{10}$
- Total Trainable Parameters: 109,466

### Input
- Upstream Feature Extractor: `facebook/wav2vec2-base`
- Hidden Layer: Layer 12 (final transformer hidden state)
- Audio Requirements: 16 kHz mono WAV, floating point
- Temporal Pooling: Mean-pooling over frames whose centers fall within syllable time boundaries
- Input Tensor Shape: `(num_syllables, 768)`, dtype `float32`

### Anchors
- Anchor Word ID: Word 1181 (Bi-syllabic training word)
- Stressed Anchor ($r_s$): Index 2713, Label 1, L2 norm 7.2073
- Unstressed Anchor ($r_u$): Index 2714, Label 0, L2 norm 6.9336
- Reference Prototypes ($z_s, z_u$): Stored in `anchors/prototypes_10d.npy` and `config.json`

### Inference
- Forward Pass: Pass 768-D syllable vectors through `LexiRepEncoder` to obtain 10-D representations $\mathbf{h}_i$.
- Contrast Scoring: Compute signed relative cosine distance:
  $$\Delta_i = \cos(\mathbf{h}_i, \mathbf{z}_s) - \cos(\mathbf{h}_i, \mathbf{z}_u)$$
- Stress Assignment: For each word $w$, assign stress label 1 to $\arg\max_{j \in w} \Delta_j$, and 0 to all other syllables in $w$.

### Constraint
- Linguistic Rule: Exactly one primary stress per word ($\sum_{j \in w} y_j = 1$).
- Compliance Rate: **100.0%** across all test words (0 zero-stress words, 0 multi-stress words).

### Checksum
- SHA-256 Checksum: `09f6d129439ca07a1b47540f5161b4cce60ad71a68308a9e7ab867dbde4db894`
- Match Verification: **Exact bitwise match** between original `results_ger/final_lexirep_model.pt` and exported model file.

### Verification
- Original vs. Exported Max Embedding Difference: **`0.0000000000e+00`** (bitwise identical)
- Quarantined Test Set Evaluation (7,000 syllables, 3,048 words):
  - Bi-Syllabic (B): **79.91%**
  - Bi + Tri (BT): **80.20%**
  - Bi + Tri + Quad (BTQ): **79.96%**
  - Balanced Accuracy: **79.70%**
- Inference Latency: $\approx 12.2\text{ µs}$ per syllable on GPU.
