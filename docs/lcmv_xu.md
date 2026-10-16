# D3 LCMV beamforming + Xu-protocol suite (Exp 0–7)

Version 1.1, 2026-10-16. Methodology log entries 50–51, decisions row 55. §4 revised after the full-paper review. A **new, separate** pipeline: outputs only under
`<processed>/D3_LCMV_XU/`, its own code pin (`D3_LCMV_XU/CODE_COMMIT.txt`) and configuration hash. Nothing of D1/D2
or of the earlier D3 (wMNE) work is changed: `graphsD3_v4_da723b33554c`, `experiments_D3/`, `D3_CODE_COMMIT.txt`,
the D1/D2 pin and outputs stay as they are.

Code: `src/adgraph/lcmvxu/` (`recording.py`, `beamformer.py`, `features.py`, `build.py`, `validate.py`),
`configs/lcmv_xu/pipeline.yaml`, `configs/experiments/lx/` (mode B) and `configs/experiments/lx/tenfold/` (mode A),
`scripts/lx_build.py`, `scripts/lx_report.py`, `scripts/lx_run_all.py`. Notebooks: `notebooks/lcmv_xu/`.

Reference: Xu Q, An L, Yang H, Hong K (2025). A multi-graph convolutional network method for Alzheimer's disease
diagnosis based on multi-frequency EEG data with dual-mode connectivity. *Front. Neurosci.* 19:1555657.
Authors' code: github.com/XQJMJ/MF-MGCN (`Model.py`, inspected at its current HEAD).

## 1. Data and recording handling

- ds004504 v1.0.9 provider derivatives already on Drive (`paths.yaml: dataset3.root`); 36 AD + 29 HC, FTD excluded by
  design; 19 channels; electrode coordinates from each `.set` file (entry 46: the Colin27 template the provider
  assigned by an EEGLAB lookup, identical for all participants).
- **Boundaries.** Every EEGLAB `boundary` event cuts the recording (entry 45). Each continuous segment is trimmed to
  an even length and resampled to 250 Hz on its own. Segments are never concatenated as if continuous. The common
  average reference is instantaneous.
- **Margin, the only trimming.** The largest band analytic-kernel half-length, from delta: 825 samples = 3.3 s.
  A window lies at least this far from both ends of its segment, so no filter output inside a window uses samples
  across a discontinuity or a recording end. No fixed edge crop is applied.
- **Five-minute selection (deterministic, `lcmvxu.recording.select`).**
  - If a segment interior holds 300 s, the first 300 s of the first such segment are used: 291 windows.
  - Otherwise the interiors are used in time order as separate pieces until 300 s are reached. Each piece keeps its
    own window grid and pieces are never joined.
  - No piece is added once less than one window is still needed, so the selection never exceeds 300 s (entry 53;
    the build at b38685e added a final 10 s piece there, see §11). Less than 300 s is selected when the recording
    does not hold 300 s as whole windows; this is reported, and the participant is not excluded for it.
  - Checked on the real files: D3-1 (no boundaries) is contiguous with 291 candidates. D3-56 has 56 boundaries
    and 57 segments, no 300 s stretch, so 7 pieces totalling 300 s give 233 candidates. Zero windows violate a
    margin.
- **Windows:** 10 s, 1 s step (90 % overlap).
- **Rejection:** the existing adaptive peak-to-peak rule (median + 5 MAD; > 15 % of channels).
- **Checks that stop the participant with a recorded reason, never silently:**
  - a flat channel (< 0.1 µV);
  - implausible amplitude;
  - invalid coordinates.
- **Inclusion:** fewer than 5 accepted windows gives `below_min_windows`, the pre-specified rule. That participant
  is still listed with its reason.
- **QC report** (`qc_report.csv`; `inspect` gives the recording-level part for all 65 before any source modelling):
  - original duration, boundary count and positions, segment durations, usable segments and interior duration;
  - selection mode and selected duration;
  - candidate, accepted and rejected windows, rejection reasons;
  - LCMV rank and verification error;
  - status and reason.

## 2. Source reconstruction: LCMV beamformer (replaces wMNE in this suite only)

- **Forward model:** fsaverage ico-5 surface source space, 3-layer BEM, the stored electrode positions. Fixed
  cortical-normal orientation (`surf_ori`, `force_fixed`), so the `mean_flip` ROI extraction is well defined.
  Cached per electrode geometry.
- **Data covariance per participant:** its own accepted, average-referenced, broadband (0.5–45 Hz) samples.
  Diagnostic labels and other participants are never used.
- **Rank:** estimated from the data (`mne.compute_rank`; D3-1: 18 = 19 − average reference). The provider's ICA can
  lower it further; the estimated rank is passed to `make_lcmv`.
- **Filter settings:**
  - `reg = 0.05` (loading of 5 % of the mean eigenvalue);
  - `weight_norm = unit-noise-gain-invariant`, robust to rank deficiency;
  - an ad-hoc diagonal noise covariance (whitening of one channel type only);
  - no depth weighting.
  - One broadband filter per participant is used for all bands, so the band powers of one ROI come from the same
    spatial filter.
- **68 ROIs:** the LCMV filters, pushed through an identity probe, then `mean_flip` label extraction over the 68
  Desikan-Killiany labels, sorted by name exactly as before. The result is a 68 × 19 operator W per participant.
- **Verification:** for every participant, W @ x is checked against MNE's `apply_lcmv_raw` plus
  `extract_label_time_course` on a real window. The subject stops if the relative error exceeds 1e-6. The
  identity-probe composition matched `apply_lcmv_raw` to 1.6e-15 on D3-1 data (spherical head model; the fsaverage
  path is verified in the Colab smoke test).
- **Units:** the ROI time courses are noise-normalized, so absolute PSD values are in beamformer units. Log,
  relative PSD, DE differences, PLV and |ciPLV| are unaffected.
- **Limitations:**
  - 19 electrodes give at most 18 independent spatial dimensions, so 68 ROI time courses are strongly smoothed,
    mixed estimates (field spread and leakage). Neighbouring ROIs are not independent.
  - Zero-lag connectivity (PLV) is inflated by leakage; |ciPLV| is the leakage-robust measure.
  - Beamformers suppress perfectly correlated sources.
  - Template head and template electrode positions are used for everyone.
  - The ROI-level results are therefore source-*space* estimates, not measured cortical activity.

## 3. Bands and features (Xu et al.)

- **Bands:** delta 0.5–4, theta 4–8, alpha 8–13 (not split), beta 13–25, gamma 25–45 Hz.
- **Band-pass for phase and DE:** MNE FIR, zero-phase, firwin, 1.0 Hz transitions, applied as a finite analytic
  kernel per continuous segment.
  - Delta's lower transition is capped at 0.5 Hz (stop edge 0 Hz). The provider already high-passed the data at
    0.5 Hz, so delta power near 0.5 Hz is attenuated by the provider's filter.
  - Gamma's upper edge (45 Hz) coincides with the provider's 45 Hz low-pass (about −3 dB).
- **Per window**, for the 68 ROIs and the 19 sensors:
  - Welch PSD (2 s Hann, 50 %): absolute, log10 and relative (0.5–45 Hz) band power;
  - differential entropy DE = ½ ln(2πe·var(band-passed window)), one value per node, band and window. Xu use
    ½ log₂(2πeσ²) (paper Eq. 3): the same quantity scaled by 1/ln 2. That positive constant factor is removed
    exactly by the per-feature standardization every model of this suite applies on its training data, so the
    choice of logarithm base does not change any result. Xu's 32-value DE vector per node is not reproduced (§4);
  - PLV and |ciPLV| between ROIs.
  - Stored as `graphs.store` arrays (`psd_de` / `sensor_psd_de` for DE). Signed ciPLV is not stored.

## 4. Comparison with Xu et al. (paper and code)

Reviewed 2026-10-16 against the full paper (Sections 2.1–3.4, 4.1, 5.4–5.5, Tables 1–2, Figures 3, 8, 9) and the
authors' `Model.py` (methodology log entry 51). Paper statements are cited by section; code statements by `Model.py`.

**Data and samples**
- Paper §3.1: a 5-min segment per participant. Matches our five-minute selection.
- Paper §3.2 and §3.4: 10 s windows with 90 % overlap expanded the data "tenfold", 65 to 650 (360 AD, 290 HC).
  Paper §4.1: 520 training and 130 test samples. Code: exactly 10 samples per participant (`sub-1` … `sub-880`,
  blocks of 10).
- **Unresolved:** how 10 samples arise from 5 minutes. A 300 s segment cut into 10 s windows at a 1 s step gives
  291 windows, not 10. The code's folder name `…10s309s…0.9…880` would fit ten 300 s segments shifted by 1 s
  (300 + 9 = 309 s), but that is 99.7 %, not 90 %, overlap. Neither text nor code settles it.
- Our **mode A** (the first 10 accepted windows: 10 consecutive 10 s windows at a 1 s step) is **our
  operationalisation** of "tenfold by 90 % overlap". It is not a verified reproduction of Xu's samples.
- Our **mode B** (every accepted window of the five minutes, up to 291 per participant) is the mathematically
  consistent reading. Neither mode fabricates samples or changes the overlap.

**Features**
- Paper §2.1 and Eq. 3: Gaussian DE = ½ log₂(2πeσ²) of the band-passed segment, for the bands δ 0.5–4, θ 4–8,
  α 8–13, β 13–25, γ 25–45 Hz. Our bands match exactly; our DE uses the natural logarithm (§3, equivalent after
  standardization).
- Paper §2.2.2 and Table 1: node features X ∈ R^(N×M), M = 32, "the DE calculated from an EEG data segment of
  duration T". Code: `GCNConv(32, 16)`, i.e. 32 input values per node.
- **Unresolved:** how the 32 values per node are formed. The paper does not say; the code reads precomputed CSV
  files. We do not invent a construction. Our node features stay per-window band PSD/DE values, as in Exp 0–7.

**Graphs ("dual-mode connectivity")**
- Paper §2.2.3, Eqs. 4–7:
  - first GCN layer: Pearson correlation between the nodes' DE feature vectors;
  - second GCN layer: binary anatomical-region adjacency (1 if two electrodes share one of five regions: frontal,
    central, parietal, temporal, occipital).
- Code:
  - the first layer reads a precomputed `…CorrelationResult` matrix but casts it to an integer type
    (`torch.tensor(…, dtype=torch.long)`). That truncates every correlation in (−1, 1) to 0, unless the files were
    already binary; edges are then taken from the non-zero entries.
  - the second layer reuses **the same edges**, thresholded at 0.5 (`edge_weight > 0.5`). There is **no**
    anatomical-region matrix in the code.
  - Paper and code therefore differ on the second layer, and possibly on the first.
- **Our suite is not a replication of this design.** Our graphs are 68 LCMV source ROIs with PLV or |ciPLV|
  adjacency (EARM-GNN adds learned gates on that EEG prior). Xu's spatial-connectivity concept could be adapted in
  source space as same-lobe adjacency from the Desikan-Killiany atlas labels. That would be **our adaptation**,
  not Xu's electrode matrix (optional, §7).

**Model and training**
- Matches between the paper (Table 1, §5.4) and the code:
  - learning rate 0.001, batch size 10, 200 epochs, cross-entropy, ReLU;
  - K = 16 (`GCNConv(32, 16)`), L = 2 (`GCNConv(16, 2)`);
  - S = 128 and F = 32 (`Linear(19·2·5 = 190, 128)`, `Linear(128, 32)`, `Linear(32, 2)`);
  - dropout 0.2 after epoch 5 (code only).
- Code only, not in the paper: Adam with weight decay 1e-4. The learning-rate schedule (gamma 0.1) acts only at
  epoch 200, i.e. has no effect.
- Parameters: the coded model has **29,204** (one GCN pair shared by the five bands). The paper states **29,332**,
  128 more, unexplained. Figure 3 draws a separate GCN pair per band, which would give 31,452.
- These dimensions belong to a 19-node, 32-feature design and are **not** transferred to our 68-ROI models, which
  keep their fixed, documented hyperparameters.

**Validation and reporting**
- Paper §4.1: participant-grouped split. All augmented samples of a participant are in one partition; 80 % of
  participants train (520 samples), 20 % test (130 samples). "A 5-fold cross-validation method was utilized during
  the training process" (not specified further).
- Code:
  - five fixed participant-level 80/20 splits (29 AD + 23 HC train, 7 AD + 6 HC validation; fold 4 has 24 + 5 HC).
    Within each split no participant is on both sides;
  - the five validation sets are **not** a partition: they overlap, and 15 AD and 13 HC participants are never
    validated;
  - every metric is the **best value over the 200 epochs** on the validation participants, each metric maximized
    separately;
  - metrics are computed **per sample** (window), not per participant;
  - the positive class is **HC** (`y = 0` for AD, `1` for NC), so the reported precision and recall refer to HC.
- Paper Figures 8–9: K, S and F were chosen by accuracy steps of 1/130, i.e. on the 130 test samples. The final
  96.15 % equals 125/130. **Hyperparameters were therefore selected on the reported test set.**
- **Ours:**
  - one frozen stratified 80/20 participant split (52/13: 7 AD, 6 HC, the same counts as Xu);
  - 5-fold participant CV (a true partition) inside the 80 % only;
  - the holdout is evaluated once, with the last-epoch model and no selection on it;
  - participant-level metrics primary, window-level secondary;
  - the positive class is **AD**: our sensitivity is Xu's specificity, and our precision/recall are not Xu's
    precision/recall.
  - Window-level numbers are the ones comparable in kind to Xu's sample-level numbers; participant-level numbers
    have no counterpart in the paper.

## 5. Validation (identical for Exp 0–7 and both modes)

- **Frozen split** (`evaluation.splits.xu_partition`, `XU_SPLIT`):
  - 20 % of participants per diagnosis are held out with seed 20250702: 13 participants (7 AD, 6 HC);
  - 52 participants form the development set;
  - saved as `xu_split.json` in both graph stores, and re-derived and compared before every experiment run (a
    mismatch refuses to run).
- **`xu_dev_cv`:** 5-fold participant-grouped stratified CV inside the development set. Holdout participants never
  enter it.
- **`xu_holdout`:** fit on all development participants, evaluate once on the holdout participants.
- **Training data only:** normalization, feature selection, class balance (participant- and class-balanced sample
  weights) and, for classical models, the C selection (nested participant-grouped inner CV, `inner_k = 5`).
  GNNs keep their fixed hyperparameters (`gnn_fixed_hparams_v1`, 60 epochs, no search).
- **Repeats:** `n_repeats = 1` (single CV and holdout fit). Raising it adds repeats and reuses the completed ones.
- **Metrics:**
  - participant level (primary): score = mean of the participant's window scores (probability for LR and GNNs,
    decision margin for the SVM);
  - window level (secondary).
  - Both report accuracy, balanced accuracy, sensitivity, specificity, precision, F1, ROC-AUC and the confusion
    matrix, plus ROC curves (`scripts/lx_report.py`).
- **Difference from Xu** (§4): their code reports the best epoch on five overlapping participant-level 80/20
  splits, with sample-level metrics and HC as the positive class. Here: one frozen 80/20 holdout, a 5-fold
  participant CV partition inside the 80 %, the last-epoch model, participant-level metrics primary, AD positive.

## 6. Experiments (existing definitions, five bands; mode A configs only switch the graph store)

| Exp | Config (mode B) | Variants |
|---|---|---|
| 0 | `lx_exp0_sensor_psd_de` | LR, SVM × {PSD [log, rel], DE} (DE is a distinct configuration) |
| 1 | `lx_exp1_source_psd_de` | LR, SVM × {PSD, DE} on 68 LCMV ROIs |
| 2 | `lx_exp2_conn_minimal` | single band, constant node feature, GCN; 5 bands × {PLV, \|ciPLV\|} |
| 3 | `lx_exp3_single_band` | single band + matched-band PSD nodes, GCN; 5 bands × {PLV, \|ciPLV\|} |
| 4 | `lx_exp4_band_mean` | band-averaged adjacency, all-band PSD (F = 10) or DE (F = 5) nodes; {PLV, \|ciPLV\|} |
| 5 | `lx_exp5_multibranch` | one graph per band; shared or band-specific GCN branches × mean / attention fusion; {PLV, \|ciPLV\|} |
| 6 | `lx_exp6_psd_strength` | full model (PSD + strength, attention) for PLV and \|ciPLV\|; PLV ablations: no PSD, no strength, mean fusion, each band dropped |
| 7 | `lx_exp7_earm_gnn` | EARM-GNN (A_adaptive = A_EEG ⊙ G_θ, residual blocks, band attention), fixed-adjacency and mean-fusion ablations; {PLV, \|ciPLV\|} |

**Fits per mode** (variants × (5 CV folds + 1 holdout fit)):
- Exp 0: 24 classical fits (each with an inner 5-fold C search);
- Exp 1: 24 classical fits (each with an inner 5-fold C search);
- Exp 2: 60;
- Exp 3: 60;
- Exp 4: 24;
- Exp 5: 48;
- Exp 6: 60;
- Exp 7: 36 GNN fits.

**Compute:** GNNs use the GPU when available. The device and machine of every result are recorded
(`compute_environment`, per-fold `compute_fingerprint`). They are not part of the checkpoint identity, so folds
completed on CPU and on GPU combine in one result, and the report shows each result's environment.

## 7. Optional future comparison experiments (not implemented)

- **Xu-adapted MF-MGCN in source space (our adaptation):**
  - first layer: Pearson correlation between ROI DE feature vectors, e.g. a DE sequence over consecutive windows
    computed from the saved ROI time courses, without rebuilding the LCMV;
  - second layer: binary same-lobe adjacency of the 68 Desikan-Killiany labels.
  - It would be labelled as an adaptation, not a replication.
- **Xu's training recipe on our inputs:** lr 0.001, batch 10, 200 epochs, as a separate configuration, never by
  changing the fixed GNN protocol.
- **Sample-level metrics with HC as positive class:** for a like-for-like comparison with Xu's Table 2.

## 9. Experiment 8: Xu-inspired dual-adjacency source-space GCN (our adaptation)

Methodology log entry 52, decisions row 56. Model: `src/adgraph/models/xu_dual.py`. Config:
`configs/experiments/lx/lx_exp8_xu_dual_gcn.yaml` (mode A: `tenfold/lx10_exp8_xu_dual_gcn.yaml`). Notebook:
`notebooks/lcmv_xu/LX_Exp8_xu_dual_adjacency.ipynb`. Check (no data): `scripts/lx_exp8_inspect.py`. Exp 8 is a **new** experiment. It takes Xu's paper architecture
(two GCN layers with different adjacencies, per band, flattened into fully connected layers) and gives it our
inputs. It is not a reproduction of MF-MGCN and replaces none of Exp 0–7.

| Element | Xu et al. (paper) | Exp 8 (ours) |
|---|---|---|
| Nodes | 19 electrodes | 68 Desikan-Killiany LCMV source ROIs |
| Node features | 32 DE values per node (construction unpublished) | our stored per-window features: `de` [DE], `psd` [log, rel PSD], `psd_de_strength` [log PSD, rel PSD, DE, node strength] (primary) |
| Layer 1 | Pearson correlation of DE feature vectors | the window's weighted PLV, or \|ciPLV\| as a separate variant, D^-1/2(A+I)D^-1/2; never binarized |
| Layer 2 | binary same-region electrode matrix (5 regions) | fixed atlas matrix: same lobe and same hemisphere; frontal, parietal, temporal, occipital, cingulate, insula. Self-loops added, D^-1/2(A+I)D^-1/2 |
| Bands | 5, GCN per band (Fig. 3; shared in code) | 5, band-specific weights (shared weights as an ablation) |
| Head | flatten 19×2×5 → 128 → 32 → 2 | fusion by flattening 68×2×5 → 32 → 32 → 1 logit |
| Widths | K = 16, S = 128, F = 32 chosen on the test set (Figs. 8–9); L = 2 | K = S = F = 32 (our pre-specified fixed-protocol widths, `gnn_fixed_base.yaml`); L = 2 (Xu Table 1, not tuned); dropout 0.3; nothing tuned |
| Training | lr 0.001, batch 10, 200 epochs, best epoch | lr 0.001, weight decay 1e-4, batch 32, 60 epochs, last epoch (the fixed schedule of Exp 2–7) |
| Validation | see §4 | frozen 80/20 participant split + 5-fold participant CV in the 80 %, as Exp 0–7 |

- **The anatomical layer is atlas-derived group membership** (coarse anatomical proximity), **not** measured
  white-matter structural connectivity. It depends only on the ROI names and order of the graph store (added to
  the variant's identity by `runner.with_store_context`), never on diagnoses, data or predictions.
- **Lobe assignment:** the conventional FreeSurfer lobar grouping of the Desikan-Killiany labels (`xu_dual.DK_LOBES`).
  - Precentral and paracentral → frontal; postcentral → parietal.
  - Entorhinal, parahippocampal, fusiform, temporal pole → temporal.
  - Cingulate (4 labels) and insula are separate groups. Each insula is therefore connected only to itself in the
    hemisphere-specific matrix.
- **Ablations** (PLV, primary features):
  - both layers functional (the analogue of Xu's MF-MGCNII);
  - both layers anatomical (no functional connectivity);
  - bilateral lobes (Xu-like grouping across hemispheres);
  - weights shared across bands (as in Xu's code).
- **Band weights:** no band attention; the bands are concatenated. The `band_weight_*` prediction columns hold the
  uniform 1/5.
- **Parameters:** 24,011 for the primary variant (F = 4), 23,531 with DE only, 23,107 with shared weights. Xu's
  19-electrode model has 29,204 as coded (29,332 in the paper).
- **Mathematics**, per band b, with N(A) = D^-1/2 (A + I) D^-1/2:
  - H1_b = ReLU(N(A_func,b) X_b W1_b + c1_b), with A_func,b the window's PLV or |ciPLV|;
  - H2_b = ReLU(N(A_anat) H1_b W2_b + c2_b);
  - z = flatten([H2_δ … H2_γ]), then logit = w3ᵀ ReLU(W_F ReLU(W_S dropout(z))).
  - Single-adjacency ablations replace A_anat by A_func,b (both layers functional) or A_func,b by A_anat (both
    anatomical); the implementation is otherwise unchanged.
- **Code pin:** Exp 8 runs its own pin (`D3_LCMV_XU/lx_exp8_xu_dual_gcn_CODE_COMMIT.txt`). The suite pin used by
  Exp 0–7 is not moved, so their completed checkpoints stay valid.

## 10. Run order (Colab, *Run all*)

1. `notebooks/lcmv_xu/LX_00_build_dataset.ipynb` (CPU): input check → recording QC → one-participant smoke test →
   full build (only if the smoke test passed; resumable) → validation → QC report.
2. `LX_Exp0` … `LX_Exp7`, one at a time. Each runs mode A, then mode B. Exp 0/1 run on CPU; Exp 2–7 should use a
   T4 GPU.
3. `LX_99_report.ipynb` (CPU): the comparison table, ROC curves and confusion matrices across all experiments.

1b. `LX_01_correct_and_validate.ipynb` (CPU, once, after the build at b38685e): the targeted correction of §11
   and the full validation of both corrected stores. Experiments use only validated stores.

`LX_98_run_all_optional.ipynb` runs all experiments in sequence, but only after `RUN_EVERYTHING = True` is set.

## 11. Correction `corr1` of the dataset built at b38685e (methodology log entry 53)

The full build (65/65 participants; mode B 16,422 windows, mode A 650) failed one validation check:
`D3-23: window [65725, 68225)`. Two participants also reported more than 300 s selected (D3-23 305.62 s, D3-58
309.62 s). Both were investigated on the public ds004504 files with the build's own code:

- **The flagged window is safe.** It is the first window of segment 10 (piece 5) of D3-23. The segment starts at input sample
  129,800 (259.6 s, the boundary itself), and the window starts at input sample 131,450: exactly 1,650 input samples
  = 2 × 825 = the margin after the boundary. No other boundary lies within its margin. The margin equals the
  delta analytic-kernel half-length, so every filtered value in the window uses only samples of its own segment.
  The validator rebuilt the cut as `round(129800 / 500, 3) × 250` in floating point, which is slightly more than
  64,900, so the strict condition `start − margin < cut` fired at exact equality: a false positive from the
  rounded-seconds metadata, not an unsafe window.
- **The over-target selections were a selection bug.** With less than one window still needed (4.38 s for D3-23,
  0.38 s for D3-58), `max(need − got, window)` added one more 10 s piece holding one window each. Those two windows
  lie outside a 300 s selection. They are D3-23 [90130, 92630) (segment 12, piece 7) and D3-58 [79030, 81530)
  (segment 2, piece 2). Both are boundary-safe and accepted.

Fix (code):

- `recording.select` stops instead of adding a piece. D3-23 now has 295.62 s and D3-58 299.62 s. A further piece
  could only hold a window by exceeding 300 s.
- `recording.segment_table` records the exact integer positions: `boundary_cuts_in` (input samples) and
  `segments_in` (`[index, start_in, n_in, orig_start]`).
- `validate.boundary_violations_exact` checks every window in exact rational arithmetic, in its own segment, with
  the same strict conditions. The validator is not relaxed: one sample closer to a boundary still fails, and
  stores without exact positions keep the legacy check.
- New checks:
  - selected EEG ≤ 300 s, both as reported and as covered by the windows;
  - sample weights sum to 1 per participant;
  - window counts equal those in `subjects.csv`.

Correction of the stored dataset (`python scripts/lx_build.py correct`; `lcmvxu.correct`):

- **New stores:** `graphsLX_v1_811690218bfa_corr1` and `graphsLX10_v1_811690218bfa_corr1`. The originals, their 65
  participant checkpoints, the LCMV operators and the pin `b38685e` are not modified.
- **Not recomputed:** no beamformer, connectivity or feature. The kept windows' rows of every aligned array (PLV,
  |ciPLV|, source and sensor PSD and DE) are copied with one mask, in their original order, together with the windows
  table. `sample_weight` is recomputed (1 / windows of the participant). `subjects.csv` gets the corrected selection
  and counts plus the exact positions.
- **Verified first; nothing is written if a check fails:**
  - For all 65 participants, from the EEGLAB header alone (no signal), the stored boundary positions, segment
    durations and segment table are reproduced, and the original selection reproduces every stored candidate
    window exactly.
  - The corrected selection is a subset of the original one.
  - For changed participants only, the signal is reloaded and segmented (no source model). The stored peak-to-peak
    values and decisions must be reproduced, and the adaptive rejection rule is re-applied to the corrected candidate
    set. No kept window may change decision. On the public files: D3-23 threshold 86.26 → 85.96 µV, D3-58
    83.57 → 83.68 µV, no change.
  - The frozen participant split is unchanged, and the mode A windows and arrays are bit-identical (the removed
    windows are not among anyone's first ten).
- **Known, documented consequence, not recomputed by decision:** the LCMV data covariance of D3-23 and D3-58 was
  estimated from their accepted windows including the removed one (one extra clean 10 s window of the same
  participant; no label or other participant involved). `lcmv_covariance_n_windows` records this.
- **Records:** `correction.json`, `correction_participants.csv` and `correction_removed_windows.csv` in both stores.
  The manifest's `correction` field names the source `graph_run_id`.
- **Expected counts:** mode B 16,420 windows; mode A 650. The tool reports the counts it actually finds.

Experiments refuse a store unless its `validation.json` passed for its current `graph_run_id`
(`validate.require_validated`). `latest_complete_graph_run` picks the corrected stores, which were assembled later.
`lx_build.py build` does nothing once the corrected stores exist. It also refuses to recompute checkpoints made by
another code version unless `--allow-recompute` is passed.

