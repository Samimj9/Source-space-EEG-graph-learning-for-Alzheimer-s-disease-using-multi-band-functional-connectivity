# Experiment 7 — EEG-Guided Adaptive Residual Multiband GNN (EARM-GNN)

Version 1.0, 2026-10-14. Methodology log entry 43, decisions row 51, DESIGN.md v1.6.
Code: `src/adgraph/models/earm.py`. Config: `configs/experiments/exp7_earm_gnn.yaml`.
Notebook: `notebooks/experiments/Exp7_EARM_GNN.ipynb`. Tests: `tests/test_earm.py`.

**Status: exploratory extension.** It was designed after the Tier 1 GCN results (Exp 2–4) had been
observed. The primary design (DESIGN.md) and every earlier experiment are unchanged. EARM-GNN is not
claimed to be novel or superior: its value is judged only from the pre-specified comparisons below,
evaluated with the same protocol as the other GNNs, and reported descriptively.

## 1. Motivation (from the code inspection and the observed results)

- The Tier 1 GCN propagates with D̃^{-1/2}(A + I)D̃^{-1/2} on dense, fully connected PLV / |ciPLV| graphs.
  With weighted degree ≈ 1 + 67·(mean connectivity), the self-loop share is a few percent for PLV, so
  two layers act largely as connectivity-weighted spatial averaging (over-smoothing risk). Every edge's
  influence is fixed by the EEG value; nothing is learned about which connections matter.
- Constant node features (Exp 2) gave AUCs near 0.5; PSD node features carried most of the signal in
  D2. Regional information should therefore reach the classifier without being repeatedly averaged.

## 2. Architecture

Inputs per window: x ∈ ℝ^{K×R×F}, adj ∈ ℝ^{K×R×R}; K = 4 bands (θ, α1, α2, β), R = 68 DK ROIs,
F = 3 node features per band: log PSD, relative PSD, node strength (mean connectivity of the ROI in that
band's own EEG adjacency; `graphs.dataset.node_strength`). Node features are z-scored per (band, ROI,
feature) with statistics of the training windows of each fold (`NodeFeatureScaler`). PLV and |ciPLV| are
separate variants.

**A. EEG-guided adaptive adjacency.** Shared ROI embeddings U ∈ ℝ^{68×r} (r = 8), band-specific
interaction weights s_k ∈ ℝ^r and biases b_k:

  G_k = σ(b_k + U diag(s_k) Uᵀ) ∈ (0, 1)^{68×68}  (symmetric by construction)
  Ã_k = A_k ⊙ G_k
  P_k = D_k^{-1/2} Ã_k D_k^{-1/2},  D_k = diag(Σ_j A_k,ij)  (degree of the **ungated** EEG adjacency)

- The EEG matrix stays the physiological prior: gates can only attenuate an EEG connection, never
  create one (Ã ≤ A elementwise); no thresholding or top-k.
- Normalizing with the prior degree keeps the operator bounded (0 ≤ Ã ≤ A ⇒ spectral radius of
  P_k ≤ that of D^{-1/2}AD^{-1/2} = 1) and lets a closed gate remove a neighbour's message instead of
  being undone by renormalization. Emphasis is relative: connections whose gates stay open dominate.
- Self-loops are excluded from P_k; the node's own state enters through a separate transformation (B).
- Regularization: λ · mean_{k, i<j} G_k,ij with λ = 0.01, added to the training loss
  (`EARMGNN.regularization`). It is the only explicit gate prior (plus Adam weight decay 1e-4).
- Initialization: b_k = 2 (gates ≈ 0.88, close to the EEG prior), U ~ N(0, 0.1²), s_k = 1.
- Gates are window-independent parameters learned from the training subjects of each fold only.
- 580 parameters (544 + 32 + 4).

**B. Residual graph encoder** (shared across bands; band embedding e_k ∈ ℝ^H, H = 32):

  H⁰_k = X_k W_in + b_in + e_k
  Hˡ_k = Hˡ⁻¹_k + Dropout(GELU(LayerNorm(Hˡ⁻¹_k W_selfˡ + bˡ + P_k Hˡ⁻¹_k W_nbrˡ))),  l = 1, 2, 3
  H_k = [H⁰_k ‖ H¹_k ‖ H²_k ‖ H³_k] W_jk + b_jk  (jumping knowledge)

Distinct self and neighbour transformations, residual connections, layer normalization, GELU, dropout
0.3. H⁰ is a purely node-wise transformation of the regional features, so regional information reaches
the readout without neighbourhood averaging (residual path and JK).

**F. Readout per band:** a_k = softmax_i(w₂ᵀ tanh(W₁ h_k,i + c₁)) (16 hidden units);
z_k = [Σ_i a_k,i h_k,i ‖ mean_i h_k,i ‖ max_i h_k,i] ∈ ℝ^{3H}. Each band keeps its own representation.

**E. Band fusion (attention):** s_k = qᵀ tanh(W z_k + c), β = softmax_k(s), z = Σ_k β_k z_k (32 hidden
units; `multiband.BandFusion`). The band weights β are saved per window (`band_weight_<band>` columns of
`predictions_windows.csv`).

**Classifier:** Dropout(0.3) → Linear(96, 32) → ReLU → Dropout(0.3) → Linear(32, 1) → AD logit; P(AD) =
σ(logit). Subject score = mean of its window probabilities (unchanged pipeline); threshold 0.5.

### Tensor shapes (batch B)

| Stage | Shape |
|---|---|
| x, adj | (B, 4, 68, 3), (B, 4, 68, 68) |
| G, P | (4, 68, 68), (B, 4, 68, 68) |
| H⁰ … H³ | (B, 4, 68, 32) each |
| JK concat → H | (B, 4, 68, 128) → (B, 4, 68, 32) |
| ROI attention a | (B, 4, 68) |
| band embeddings z_k | (B, 4, 96) |
| fused z, band weights β | (B, 96), (B, 4) |
| logit | (B,) |

### Parameters (exact; `scripts/exp7_earm.py inspect` prints the full table)

| Variant | Trainable parameters |
|---|---|
| full | 18,214 |
| ablation A (fixed adjacency) | 17,634 |
| ablation B (mean fusion) | 15,078 |

Largest blocks: JK projection 4,128; 3 residual blocks 6,432; fusion attention 3,136; classifier 3,137;
input + band embedding 256; gates 580; ROI pooling 545.

### Computational cost

≈ 4.6 M multiply-adds per window forward (four bands × three blocks of 68×68×32 propagation plus
68×32×32 transforms, and the JK projection); ≈ 3× that for training. A within-D1 outer fold (≈ 725
training windows × 60 epochs) is ≈ 1–2 TFLOP, negligible for a T4; runtime is dominated by per-step
overhead. Expected: ≈ 10–25 s per outer fold on a T4, ≈ 30–60 min per variant over all settings,
**≈ 3–6 h for the 6 variants** (an estimate, not a measurement).

## 3. Training and evaluation protocol (fixed in advance)

- Same frozen v4 graph run (`graphs_v4_33faa4b89a5d`, `graph_run_id 6d75de9975f0d9ae`), same runner,
  splits, seeds, checkpoints and result writers as every experiment.
- Settings: within-D1 and within-D2 (repeated stratified subject 5-fold, 10 repeats, seeds 0–9: the same
  subject partitions as repeats 0–9 of every other experiment), pooled D1+D2 (stratified by dataset ×
  diagnosis), D1→D2 and D2→D1 (train on all source subjects, test once on all target subjects).
- Fixed hyperparameters, no search, no inner CV, one fit per outer training fold (protocol
  `gnn_fixed_hparams_v1`): Adam (lr 1e-3, weight decay 1e-4, default betas), batch 32, 60 epochs, no
  early stopping, last-epoch model; subject- and class-balanced BCE + gate penalty. The schedule equals
  the Tier 1 GNNs' so that differences are architectural. The EARM-specific values (H = 32, 3 blocks,
  r = 8, λ = 0.01, gate bias 2, dropout 0.3) were set from the size of the problem (43 participants,
  ~18 k parameters, comparable to the Tier 1 GCN) before any Exp 7 result, and are not revised from
  held-out results.
- Every fitted quantity (node scaler, gates, attention, all weights) uses training subjects only; the
  cross-dataset settings fit nothing on the target dataset.
- Saved per outer fold: predictions, band weights, fitted parameters, learned edge gates, training-loss
  history, node-scaler statistics (`units/fold_*.model.npz`), plus the standard summaries and manifests.
- Metrics: subject-level ROC AUC, balanced accuracy, sensitivity, specificity, F1, MCC; mean ± SD over
  repeats; subject-bootstrap 95 % CI; fit time and parameter count per fold.

## 4. Pre-specified comparisons

| Variant | Differs from full by |
|---|---|
| full | — |
| fixed_adjacency (ablation A) | gates disabled: P = D^{-1/2} A D^{-1/2}, no gate penalty |
| mean_fusion (ablation B) | mean instead of attention band fusion |

Each × {PLV, |ciPLV|} = 6 variants. Comparison with completed experiments (Exp 0, 1, 4, 5, 6) from saved
outputs only (`scripts/exp7_earm.py compare`): per-setting tables, ROC / confusion figures, and per-repeat
paired differences on shared repeats. These are **descriptive**; no significance test is computed and no
p-value is reported.

## 5. D1 / D2 diagnostic (read-only)

`scripts/exp7_earm.py diagnose` (`src/adgraph/evaluation/dataset_shift.py`) writes
`<outputs>/diagnostics/dataset_shift_<time>/`: recording quality per dataset × group; channels bad in
≥ 30 % of a dataset's recordings (reasons, whether interpolated, post-interpolation amplitude);
subject-level feature distributions and D1-vs-D2 shift within each group; univariate class separability
(AUC per feature, no model) and whether the AD effect has the same direction in both datasets; and the
prediction behaviour of saved within-dataset results (no discrimination / reversed ranking / near-constant
scores). It does not exclude data or change the pipeline; any defect it reveals is to be reported and
corrected separately with the author's approval.

## 6. Code pin

The Exp 7 notebook runs the commit recorded in `experiments/exp7_earm_gnn/CODE_COMMIT.txt` (set on the
first run; `UPDATE_EXP7_CODE = True` moves only this pin), so Exp 7 can run while Exp 2–6 continue on the
global pin. The commit is also recorded in every Exp 7 manifest and checkpoint identity.

## 7. Limitations

- Designed after observing the GCN results; exploratory; no correction for the additional model family.
- 43 participants: ~18 k parameters are still many relative to the independent units; learned gates and
  attention weights are interpretable only descriptively and may be unstable across folds.
- Gates are global per band (not per participant), so they describe which connections the classifier
  uses, not individual connectivity.
- No permutation tests; differences between variants are descriptive.
