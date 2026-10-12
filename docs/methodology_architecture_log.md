# Methodology and architecture log

This log records why each important methodological or architectural decision was
made, with enough detail to defend or revisit it. It complements
`docs/methods_decisions.md` (a compact table of parameter values) and `DESIGN.md`
(the evaluation protocol).

**Rules for this log**

- Only reasons that can be established from the repository, its history or the
  project author's recorded instructions are given.
- Where the original reason for an older decision cannot be established, the entry
  says **"requires author confirmation"** instead of supplying a retrospective
  justification.
- "v1" refers to the graph set built before this repository existed. Its code is not
  in the repo; its parameters are known only from
  `AD-LEB/processed/run_manifest_20260912_162006.json`.

**Approval vocabulary**

| Status | Meaning |
|---|---|
| *Approved* | The author explicitly approved it in a recorded instruction. |
| *Author-specified* | The author gave the design directly. |
| *Approved in principle* | Covered by the author's general approval of the implementation ("approved in principle", 2026-10-12) without a specific statement. |
| *Pending* | Not yet approved. |

Nothing in this log has been run on the real EEG. Every validating test uses
synthetic data.

---

## Contents

1. [Source-space node and edge definitions](#1-source-space-node-and-edge-definitions)
2. [Harmonized sensor set and shared sensor→ROI operator](#2-harmonized-sensor-set-and-shared-sensorroi-operator)
3. [Signed normal source orientation](#3-signed-normal-source-orientation)
4. [20 s / 50 % windowing and edge trim](#4-20-s--50--windowing-and-edge-trim)
5. [Source PSD node features (and the relative-power denominator)](#5-source-psd-node-features-and-the-relative-power-denominator)
6. [Matching-band PSD in Experiment 3](#6-matching-band-psd-in-experiment-3)
7. [EdgeWeightedGAT](#7-edgeweightedgat)
8. [WeightedGraphSAGE](#8-weightedgraphsage)
9. [GCN as the standard weighted baseline](#9-gcn-as-the-standard-weighted-baseline)
10. [PLV versus |ciPLV|](#10-plv-versus-ciplv)
11. [Preservation and interpretation of signed ciPLV](#11-preservation-and-interpretation-of-signed-ciplv)
12. [Band-averaged versus multi-band graphs](#12-band-averaged-versus-multi-band-graphs)
13. [Experiment 6: PSD + strength node representation](#13-experiment-6-psd--strength-node-representation)
14. [Subject-level splitting](#14-subject-level-splitting)
15. [SVM scores and optional grouped Platt calibration](#15-svm-scores-and-optional-grouped-platt-calibration)
16. [Age sensitivity analysis](#16-age-sensitivity-analysis)
17. [Dense PyTorch rather than PyTorch Geometric](#17-dense-pytorch-rather-than-pytorch-geometric)

---

## 1. Source-space node and edge definitions

**Previous design (v1)**
- Template anatomy fsaverage (ico-5, 20 484 vertices), MNE inverse (loose 0.2,
  depth 0.8, λ² = 1/9).
- Desikan-Killiany `aparc` parcellation (68 ROIs), `mean_flip` extraction.
- PLV in theta 4–8, alpha1 8–10, alpha2 10–13 and beta 13–30 Hz.
- The v1 manifest describes fsaverage as an "adaptation of Paper 1's ICBM152 /
  Brainstorm".

**Problem discovered**
- The parameters and code were not in the repository, so nothing could be
  regenerated from a commit.
- Settings such as the orientation choice (entry 3) and the noise covariance
  were not recorded.

**Scientific / mathematical reason**
- A node is a cortical ROI; an edge is the band-limited phase synchrony between
  two ROI time series in one window.
- These definitions are retained from v1.
- The reasons for the specific choices **require author confirmation**: the
  Desikan-Killiany atlas, `mean_flip`, the MNE inverse parameters, and the band
  edges, especially the split of alpha at 10 Hz.
- What *can* be stated is that the ROI time series are linear projections of the
  sensor data. Graphs therefore inherit field-spread / source-leakage effects;
  this motivates entry 10.

**Final implementation**
- Node *i* is ROI *i* of 68 (`aparc`, sorted by label name, `unknown` /
  `corpuscallosum` excluded, count asserted to be 68).
- Edge (*i*, *j*) in band *b* is PLV or |ciPLV| over one 20 s window.
- Self-loops are zero. Matrices are dense, with no thresholding at build time.

**Alternatives considered**
- Other atlases (Destrieux, Schaefer) and other connectivity estimators (wPLI,
  AEC) were not evaluated.
- Sparsification is available at the model stage (`sparsify`, fitted on
  training data), with `none` as the default.

**Effect on interpretation**
- Results are conditional on template anatomy (no individual MRI) and on
  name-based electrode positions (no digitization). Both are limitations of the
  data (`docs/data_inventory.md`).
- ROI-level claims are about atlas regions, not sources.

**Code / config**
- `configs/pipeline.yaml` (`source`, `connectivity`)
- `src/adgraph/source/inverse.py`, `src/adgraph/connectivity/plv.py`, `src/adgraph/pipeline.py`

**Validating tests**
- `test_plv.py::test_identical_and_lagged_signals`, `test_matrix_properties_plv_ciplv_signed`,
  `test_band_analytic_isolates_band`
- `test_end_to_end.py::test_pipeline_on_synthetic_edf`
- The fsaverage operator itself is verified at runtime against MNE's full inverse
  (`verify_operator`). It is not unit-tested here because fsaverage cannot be
  downloaded in the test environment.

**Approval status**
- Node and edge definition approved (2026-10-12, "PLV primary + ciPLV
  sensitivity", Exp 0–6 structure).
- Atlas, inverse parameters and bands are inherited from v1, so their rationale
  requires author confirmation.

---

## 2. Harmonized sensor set and shared sensor→ROI operator

**Previous design (v1)**
- M1/M2 were dropped for D1 only, giving 30 channels for D1 and 32 for D2.
- One forward model was built per dataset.

**Problem discovered**
- The two datasets went through different source models. The v1 manifest itself
  lists this as a "documented asymmetry/limitation".
- This makes the cross-dataset gate in `DESIGN.md` §7 impossible to pass.

**Scientific / mathematical reason**
- If sensor sets differ, the forward and inverse operators differ. Dataset
  differences in the graphs would then be confounded with processing differences.
- Why v1 dropped the mastoids for D1 only **requires author confirmation**.

**Final implementation**
- M1/M2 are dropped in both datasets, giving 30 identical channels in the order
  of `electrodes_positions_32.loc`.
- One forward model, inverse operator and `mean_flip` extraction are collapsed
  into a single 68 × 30 matrix W, shared by both datasets.
- W is verified against `apply_inverse_raw` + `extract_label_time_course` on
  random data. The run aborts if the relative error exceeds 1e-6.

**Alternatives considered**
- Keeping the mastoids in both datasets (requires D1 files to contain them; v1
  evidence suggests they do).
- Separate per-dataset operators (rejected because of the gate).

**Effect on interpretation**
- Processing is identical across datasets, so remaining dataset differences
  reflect acquisition, cohort or recording effects.
- Choosing to drop the mastoids rather than keep them **requires author
  confirmation**.

**Code / config**
- `configs/pipeline.yaml` (`channels`)
- `src/adgraph/source/inverse.py`, `src/adgraph/pipeline.py::verify_operator`
- `src/adgraph/experiments/gate.py`

**Validating tests**
- `test_io.py::test_common_channels_harmonized`
- `test_end_to_end.py::test_run_writes_outputs` (shared-operator flag in the manifest)
- `test_runner.py::test_cross_dataset_gate_blocks_unverified_runs`

**Approval status**
- Approved in principle. The mastoid choice awaits explicit confirmation
  (methods log #15).

---

## 3. Signed normal source orientation

**Previous design (v1)**
- Not recorded.

**Problem discovered**
- MNE's default `pick_ori=None` for a loose-orientation inverse returns the
  vector norm per vertex: non-negative and without sign.
- It could not be established from the v1 manifest whether v1 used it.

**Scientific / mathematical reason**
- PLV needs the instantaneous phase of an oscillating, signed signal. A norm
  time course rectifies the oscillation, which changes its spectrum and phase.
- `mean_flip` relies on signed sources.
- `pick_ori='normal'` keeps the signed component normal to the cortex.

**Final implementation**
- `source.pick_ori: normal`. It is part of W and recorded in the manifest.

**Alternatives considered**
- Vector solutions with per-ROI PCA. These are more complex and were not
  needed for a linear operator.

**Effect on interpretation**
- Phase-based edges are meaningful. Any comparison with v1 graphs is invalid
  until v1's choice is known.

**Code / config**
- `configs/pipeline.yaml` (`source.pick_ori`), `src/adgraph/source/inverse.py`

**Validating tests**
- Runtime operator verification only, as in entry 1.

**Approval status**
- Approved in principle (methods log #18).

---

## 4. 20 s / 50 % windowing and edge trim

**Previous design**
- v1 and pipeline v2 used 40 s windows with a 20 s step and a 10 s edge trim.

**Problem discovered**
- None technical. The author changed the primary segmentation (2026-10-12).

**Scientific / mathematical reason**
- The author froze 20 s windows with 50 % overlap for both datasets. The
  scientific reason for 20 s **requires author confirmation**.
- Mathematical consequences, recorded so they are not mistaken for reasons:
  - **Samples per window:** 5000 per window at 250 Hz.
  - **Cycles per window:** roughly 80 cycles at 4 Hz and 260 at 13 Hz.
  - **PLV estimation bias:** the bias for independent signals scales roughly as
    1/√(number of independent cycles). It is higher than with 40 s windows but
    identical for all subjects.
  - **Window count:** about twice as many windows per subject.
  - **Overlap:** consecutive windows share 10 s, so they are strongly dependent
    (see entry 14).
- The 10 s trim at each end removes filter edge transients and settling. Its
  original v1 rationale **requires author confirmation**.
- Because of the trim, a nominal 300 s recording gives 27 windows rather than 29.

**Final implementation**
- `window_s: 20`, `step_s: 10` (5000 / 2500 samples), `edge_trim_s: 10`.
- All filtering happens on the continuous signal before segmentation.
- Window IDs are stable (index among candidate windows before rejection).

**Alternatives considered**
- 40 s windows (v1 / v2) remain configurable. Other lengths were not studied.

**Effect on interpretation**
- More, shorter windows. Window-level metrics become even less meaningful, so
  subject-level aggregation is essential.
- PLV values are not numerically comparable with 40 s results.

**Code / config**
- `configs/pipeline.yaml` (`windowing`)
- `src/adgraph/preprocessing/windows.py`, `src/adgraph/pipeline.py::process_recording`

**Validating tests**
- `test_windows.py::test_primary_segmentation_is_frozen_at_20s_50pct`,
  `test_window_starts_respect_trim_and_step`, `test_expected_n_windows_matches_starts`,
  `test_expected_n_windows_examples` (27 vs 29)
- `test_end_to_end.py::test_pipeline_on_synthetic_edf`

**Approval status**
- 20 s / 50 % approved and frozen (2026-10-12).
- Trim kept as configurable, "do not change it yet".

---

## 5. Source PSD node features (and the relative-power denominator)

**Previous design (v2)**
- Log10 Hilbert-envelope power per ROI and band, as candidate node features.

**Problem discovered**
- The author requested explicit source-space PSD: absolute, log and relative
  power per window, ROI and band, with a configurable estimator.
- The relative-power denominator (1–45 Hz) had to be checked against the
  preprocessing passband.

**Scientific / mathematical reason**
- *PSD.* Band power describes local spectral activity of each ROI, the node
  attribute. AD-related spectral slowing is the motivation stated for the PSD
  baselines (Paper 2 / Exp 0); the exact literature basis **requires author
  confirmation**.
- *Estimator.* Welch with 2 s Hann segments and 50 % overlap gives 0.5 Hz
  resolution and 19 averaged segments per 20 s window. Bands are half-open
  [lo, hi), so adjacent bands never share a bin.
- *Relative-power denominator.* Relative power is only meaningful if the
  denominator contains frequencies preprocessing actually preserves.
  - Preprocessing is a zero-phase FIR band-pass 0.1–45 Hz (MNE `firwin`). Both
    frequencies are **passband edges**.
  - Measured response at 250 Hz: −0.03 dB at 0.1 Hz, ≈0 dB at 1 Hz, −0.01 dB at
    44.5 Hz, −0.04 dB at 45 Hz, −6 dB at 50.6 Hz, −48 dB at 56.25 Hz.
  - The 1–45 Hz denominator (bins 1.0 … 44.5 Hz) therefore lies entirely inside
    the preserved passband and is **not changed**.
  - The low-pass is not below 45 Hz, so no adjustment is required.
  - Why the lower limit is 1 Hz rather than 0.1 Hz **requires author
    confirmation**. A plausible but unconfirmed reason is excluding the drift
    region.

**Final implementation**
- `features/psd.py` provides `welch` (default), `multitaper` and `periodogram`.
- Absolute power is the sum of bins in [lo, hi) times Δf. Log is log10.
  Relative power is absolute / power in [1, 45).
- Computed for 68 ROIs and the 30 sensors.
- `check_relative_range` makes the pipeline fail, rather than silently adapt, if
  the denominator is configured outside the passband.

**Alternatives considered**
- Multitaper (configurable).
- A denominator of 4–30 Hz (the four bands only), which would make relative
  powers sum to 1. Not adopted.
- 0.1–45 Hz, which includes drift.

**Effect on interpretation**
- Relative power is a fraction of 1–45 Hz power. The four bands cover 26/44 of
  that range, so relative powers do not sum to 1.
- Absolute source power is in arbitrary units (template inverse). Log and
  relative forms are the interpretable features.

**Code / config**
- `configs/pipeline.yaml` (`psd`), `src/adgraph/features/psd.py`, `src/adgraph/pipeline.py`

**Validating tests**
- `test_psd.py::test_psd_integrates_to_variance`, `test_band_power_locates_oscillation`,
  `test_bands_are_half_open_and_disjoint`, `test_relative_range_inside_preserved_passband`,
  `test_relative_range_outside_passband_is_rejected`

**Approval status**
- Source PSD extraction approved (2026-10-12).
- Denominator check performed. Unchanged, pending author acknowledgement.

---

## 6. Matching-band PSD in Experiment 3

**Previous design**
- Exp 3 single-band graphs carried the PSD of all four bands (F = 8).

**Problem discovered**
- With all-band node features, a single-band "connectivity + PSD" model also
  sees spectral information from bands its graph does not represent.
- Exp 3 would then mix "connectivity of band b" with "full spectrum", confounding
  the band-specific question.

**Scientific / mathematical reason**
- Each band's graph is paired with that band's local activity, so node and edge
  describe the same frequency range. Example: the theta graph gets theta [log,
  rel] PSD.
- The decision and its rationale were specified by the author (2026-10-12).

**Final implementation**
- `exp3_conn_psd.yaml`: `psd_bands: matched`, `psd_kinds: [log, rel]`, F = 2.
- All-band PSD is kept only as the secondary `exp3b_conn_psd_allbands.yaml`
  (F = 8).
- Exp 4 (band-averaged adjacency) keeps all-band PSD.

**Alternatives considered**
- All-band PSD (now Exp 3b).
- Log-only or relative-only PSD (configurable through `psd_kinds`).

**Effect on interpretation**
- Exp 3 vs Exp 2 isolates the contribution of same-band local power to a
  same-band graph.
- Exp 3b vs Exp 3 shows whether other-band spectral information adds anything.

**Code / config**
- `configs/experiments/exp3_conn_psd.yaml`, `exp3b_conn_psd_allbands.yaml`
- `src/adgraph/graphs/dataset.py::build_graph`

**Validating tests**
- `test_dataset.py::test_experiment_node_feature_layouts` (theta graph ← theta PSD,
  beta graph ← beta PSD, F = 2 / 8), `test_psd_node_features_matched_and_all`
- `test_configs.py::test_exp3_primary_uses_matching_band_psd`

**Approval status**
- Author-specified (2026-10-12).

---

## 7. EdgeWeightedGAT

**Previous design**
- Called "GAT" in configs (`gat`).

**Problem discovered**
- Every graph is fully connected and weighted.
- Canonical GAT on a complete graph ignores edge weights.
- With the constant node features of Exp 2, every message `W h_j` is identical.
  Attention then averages identical vectors, so the output is the same for every
  graph, whatever the connectivity.

**Scientific / mathematical reason**
- With constant x, canonical GAT gives h'_i = Σ_j α_ij W x = W x for all i and
  all graphs, because Σ_j α_ij = 1.
- The modification adds the edge weight to the attention logit,
  e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j + c_h A_ij).
- It also adds the edge weight to the message, m_ij = W h_j + A_ij u_h. The
  output then depends on connectivity.
- Edges with A_ij = 0 (after optional sparsification) are masked; self-loops are
  kept.

**Final implementation**
- `models/edge_weighted_gat.py::EdgeWeightedGATConv`, encoder name
  `edge_weighted_gat`, reported as **EdgeWeightedGAT**.
- `edge_in_message` is configurable.
- The plain name `gat` is rejected by `build_encoder`.

**Alternatives considered**
- Canonical GAT, rejected because it is degenerate for Exp 2.
- Edge features in attention only (PyG `GATConv` with `edge_dim`). Still
  degenerate with constant features.
- Transformer-style edge-conditioned attention. Not implemented.

**Effect on interpretation**
- Results must not be compared to, or described as, canonical GAT.
- Both attention and messages are connectivity-weighted.

**Code / config**
- `src/adgraph/models/edge_weighted_gat.py`, `src/adgraph/models/encoders.py`
- `configs/experiments/gnn_base.yaml` (`encoder_kwargs`)

**Validating tests**
- `test_models.py::test_constant_features_still_see_connectivity`,
  `test_edge_weighted_gat_masks_absent_edges`, `test_node_permutation_invariance`,
  `test_forward_shapes`, `test_canonical_names_are_not_offered`

**Approval status**
- Approved, with the explicit naming requirement (2026-10-12).

---

## 8. WeightedGraphSAGE

**Previous design**
- Called "GraphSAGE" (`graphsage`).

**Problem discovered**
- GraphSAGE's mean aggregator of constant node features is the same constant for
  every node and graph. The same holds for an edge-weighted mean
  Σ A_ij h_j / Σ A_ij.
- Exp 2 would therefore carry no connectivity information.

**Scientific / mathematical reason**
- The weighted *sum* Σ_j A_ij h_j / (R − 1) gives each node its strength times
  the feature, which is connectivity-dependent.
- The (R − 1) scaling keeps the magnitude independent of graph size.

**Final implementation**
- `models/weighted_graphsage.py::WeightedSAGEConv`, encoder name
  `weighted_graphsage`, reported as **WeightedGraphSAGE**.
- Aggregator `sum` by default; `mean` and `max` (weighted) are configurable.
- The plain name `graphsage` is rejected.

**Alternatives considered**
- Canonical mean-aggregator GraphSAGE, rejected because it is degenerate for
  Exp 2.
- An LSTM aggregator. Not applicable to an unordered weighted neighbourhood.

**Effect on interpretation**
- This is not canonical GraphSAGE. Its neighbourhood aggregation is an explicit
  function of edge weights.

**Code / config**
- `src/adgraph/models/weighted_graphsage.py`
- `configs/experiments/gnn_base.yaml` (`encoder_kwargs.aggr`)

**Validating tests**
- `test_models.py::test_constant_features_still_see_connectivity`, `test_conv_shapes`,
  `test_node_permutation_invariance`, `test_canonical_names_are_not_offered`

**Approval status**
- Approved, with the naming requirement (2026-10-12).

---

## 9. GCN as the standard weighted baseline

**Previous design**
- Unchanged since first implementation.

**Problem discovered**
- None. The author asked for GCN to remain the standard baseline (2026-10-12).

**Scientific / mathematical reason**
- Kipf & Welling's propagation D̂^{−1/2}(A + I)D̂^{−1/2} H W applies to weighted
  adjacency without modification: degrees become weighted degrees.
- With constant features the output is a function of normalized weighted degrees,
  so GCN is not degenerate for Exp 2.
- No change to the canonical operator was needed, so it serves as the
  unmodified reference.

**Final implementation**
- `models/gcn.py::DenseGCNConv` (encoder `gcn`).
- Default encoder for Exp 4–6.

**Alternatives considered**
- ChebNet and other spectral filters. Not implemented.

**Effect on interpretation**
- GCN results are directly comparable to the standard GCN literature, given the
  weighted adjacency.

**Code / config**
- `src/adgraph/models/gcn.py`, `configs/experiments/gnn_base.yaml`

**Validating tests**
- `test_models.py::test_forward_shapes`, `test_node_permutation_invariance`,
  `test_constant_features_still_see_connectivity`

**Approval status**
- Approved (2026-10-12).

---

## 10. PLV versus |ciPLV|

**Previous design (v1)**
- PLV only.

**Problem discovered**
- Source-space ROI time series are linear mixtures, so field spread and leakage
  produce zero-lag coupling. PLV is fully sensitive to this.

**Scientific / mathematical reason**
- Let C_jk = ⟨e^{i(φ_j−φ_k)}⟩.
- PLV = |C_jk| counts zero-lag and lagged synchrony alike.
- ciPLV = Im C / √(1 − (Re C)²) (Bruna et al., 2018) is zero for purely zero-lag
  coupling and corrects the imaginary part for the real part.
- Its magnitude |ciPLV| is symmetric and non-negative in [0, 1], so it is a
  valid weighted adjacency.
- PLV is kept as primary to stay consistent with the reference pipeline
  ("Paper 1"; the exact reason for its choice there **requires author
  confirmation**). |ciPLV| is the leakage-robust sensitivity analysis.

**Final implementation**
- `plv` and `ciplv` (= |ciPLV|) are separate arrays and separate variants. They
  are never merged.
- Both are validated as symmetric, zero-diagonal, in [0, 1].

**Alternatives considered**
- wPLI, orthogonalized AEC. Not implemented.

**Effect on interpretation**
- A PLV result that survives under |ciPLV| is less likely to be leakage-driven.
- A result present only under PLV may reflect zero-lag mixing.

**Code / config**
- `src/adgraph/connectivity/plv.py`, `src/adgraph/graphs/store.py::check_adjacency`
- `graph.connectivity` in experiment configs

**Validating tests**
- `test_plv.py::test_identical_and_lagged_signals`, `test_matrix_properties_plv_ciplv_signed`
- `test_dataset.py::test_connectivity_switch_selects_right_array`, `test_plv_and_ciplv_differ`,
  `test_adjacency_contract`, `test_strength_from_magnitude_adjacency`
- `test_configs.py::test_plv_and_ciplv_are_separate_variants`

**Approval status**
- Approved: PLV primary, ciPLV sensitivity (2026-10-12).

---

## 11. Preservation and interpretation of signed ciPLV

**Previous design**
- The magnitude only. Then the signed value was added with documentation that
  described the sign as "which ROI leads".

**Problem discovered**
- The "leads" wording implies directed connectivity or causal direction, which
  the quantity does not establish (author correction, 2026-10-12).

**Scientific / mathematical reason**
- Signed ciPLV_jk carries the sign of Im C_jk, i.e. of the sine of the mean phase
  difference φ_j − φ_k.
- It is antisymmetric (ciPLV_kj = −ciPLV_jk) and therefore defined relative to
  the ROI ordering.
- The sign is circular: an offset of −π/4 and one of 7π/4 give the same sign.
- A consistent non-zero phase offset can arise from common drive, different
  regional dynamics or residual mixing. It is not evidence of information flow.

**Final implementation**
- The signed matrix is stored as `ciplv_signed` (configurable
  `output.save_signed_ciplv`) and validated as antisymmetric in [−1, 1].
- `build_graph` refuses it as an adjacency.
- Documentation states that the sign reflects the non-zero-lag phase
  relationship relative to ROI ordering, not directed or causal connectivity.

**Alternatives considered**
- Discarding the sign (loses descriptive information).
- Signed or directed graph models. Not implemented.

**Effect on interpretation**
- Any use of `ciplv_signed` is descriptive.
- All GNN results use |ciPLV| only.

**Code / config**
- `src/adgraph/connectivity/plv.py::phase_sync`, `src/adgraph/graphs/store.py`
- `configs/pipeline.yaml` (`output.save_signed_ciplv`)

**Validating tests**
- `test_plv.py::test_matrix_properties_plv_ciplv_signed`,
  `test_signed_ciplv_sign_follows_phase_offset_and_roi_order`
- `test_dataset.py::test_signed_ciplv_is_not_an_adjacency`, `test_store_validates_signed_ciplv`

**Approval status**
- Author-specified (2026-10-12).

---

## 12. Band-averaged versus multi-band graphs

**Previous design**
- Defined at the first implementation of the experiment framework.

**Problem discovered**
- The paper's claim concerns multi-band structure. A fair simple baseline was
  needed.

**Scientific / mathematical reason**
- The band-averaged graph A_mean = ¼ Σ_b A_b (Exp 4) keeps total connectivity
  but discards band specificity.
- The multi-band model (Exp 5) keeps the four graphs and learns band embeddings
  and their fusion.
- If Exp 5 beats Exp 4 on identical splits, band-specific structure carries
  information beyond the average.
- Fusion options:
  - mean / concat (fixed)
  - gate (independent sigmoid weights)
  - attention (softmax, bands compete)
  - Learned fusion (gate / attention) is the candidate model; band weights are
    saved for inspection.
- Encoders can be shared (band-symmetric under mean fusion) or band-specific.

**Final implementation**
- `graph.band_mode: mean | multi`
- `MultiBandGNN(shared_encoder, fusion)`
- Exp 4 and Exp 5 configs; PLV and |ciPLV| are separate variants.

**Alternatives considered**
- Concatenating the four adjacency matrices into one multigraph layer. Not
  implemented.

**Effect on interpretation**
- Exp 5 vs Exp 4 tests multi-band structure.
- Learned band weights are descriptive, not causal importance.

**Code / config**
- `src/adgraph/graphs/dataset.py::build_graph`, `src/adgraph/models/multiband.py`
- `exp4_band_mean.yaml`, `exp5_multiband.yaml`

**Validating tests**
- `test_dataset.py::test_band_modes`
- `test_models.py::test_shared_vs_band_specific_parameters`,
  `test_mean_fusion_is_band_symmetric_with_shared_encoder`, `test_fusion_output_dims`,
  `test_forward_shapes`

**Approval status**
- Approved (Exp 0–6 structure, 2026-10-12).

---

## 13. Experiment 6: PSD + strength node representation

**Previous design**
- [PSD(log, rel), strength] per band, kinds implicit.

**Problem discovered**
- The node feature order and content needed to be explicit (author
  specification).

**Scientific / mathematical reason**
- X_{i,b} = [log absolute PSD_{i,b}, relative PSD_{i,b}, strength_{i,b}] combines:
  - the ROI's local spectral activity in band b;
  - its overall coupling in band b: strength = mean of row *i* of band b's
    adjacency, excluding the diagonal.
- Strength is computed from the adjacency the model uses (PLV or |ciPLV|), so it
  lies in [0, 1] and never uses the signed ciPLV.
- Bands are not collapsed. Each component can be ablated: no PSD, no strength, no
  learned fusion, each band dropped, PLV vs |ciPLV|.

**Final implementation**
- `exp6_multiband_psd_strength.yaml`: F = 3, attention fusion, 8 named
  ablations × 2 connectivity measures.

**Alternatives considered**
- Raw adjacency rows as node features. Available as `identity` / not used.
- Clustering coefficient and other graph metrics. Not implemented.

**Effect on interpretation**
- Exp 6 vs Exp 5 measures the added value of the strength channel.
- The ablations attribute performance to components.

**Code / config**
- `src/adgraph/graphs/dataset.py::node_strength`, `build_graph`
- `exp6_multiband_psd_strength.yaml`

**Validating tests**
- `test_dataset.py::test_strength_and_combined_features`,
  `test_strength_from_magnitude_adjacency`, `test_experiment_node_feature_layouts`
- `test_configs.py::test_exp6_ablations`

**Approval status**
- Author-specified (2026-10-12).

---

## 14. Subject-level splitting

**Previous design**
- None. This is the first evaluation code.

**Problem discovered**
- Windows of one subject are strongly dependent: same recording and same
  person, and under 50 % overlap consecutive windows share half their samples.
- Window-level splitting would put near-duplicate data on both sides of a split.

**Scientific / mathematical reason**
- The unit of inference is the subject.
- Folds are built from one row per subject, stratified by label (pooled: by
  dataset × label), then expanded to windows.
- Subject score = mean of window scores.
- Every fitted step sees only training subjects:
  - scaling, selection, PCA
  - sparsification thresholds
  - covariate scaling
  - hyperparameter inner CV
  - early stopping, calibration

**Final implementation**
- `evaluation/splits.py` (repeated stratified subject K-fold, LOSO, cross-dataset).
- `fold_indices` re-checks subject and window-ID disjointness.
- Role-checked `DataView`s: `fit` accepts only `train`.
- Repeat, fold and grid counts are **provisional** and configurable.

**Alternatives considered**
- Window-level CV, rejected as leaky.
- Leave-one-subject-out (implemented as a sensitivity scheme).

**Effect on interpretation**
- Reported performance estimates generalization to unseen subjects.

**Code / config**
- `src/adgraph/evaluation/splits.py`, `src/adgraph/graphs/dataset.py`
- `src/adgraph/experiments/runner.py`, `configs/experiments/base.yaml`

**Validating tests**
- `test_splits.py` (all, incl. `test_overlapping_windows_never_cross_folds`,
  `test_inner_folds_stay_inside_training`, `test_overlap_is_detected`)
- `test_leakage.py::test_fit_on_test_view_is_refused`
- `test_runner.py::test_classical_experiments_all_settings`

**Approval status**
- Author-specified (evaluation hierarchy, 2026-10-12).
- Protocol sizes provisional.

---

## 15. SVM scores and optional grouped Platt calibration

**Previous design**
1. `SVC(probability=True)`, whose internal calibration CV splits windows.
2. Then sklearn `CalibratedClassifierCV` (window-level CV).
3. Then `GroupedPlattSVC` (subject-grouped calibration) for every SVM.

**Problems discovered**
1. Window-level calibration CV puts windows of one subject on both the fitting
   and the calibration side (author correction).
2. For the primary baseline, probability calibration is unnecessary. ROC-AUC
   depends only on the ranking of scores, and calibration adds a fitted step and
   its sample-size cost (author correction, 2026-10-12).

**Scientific / mathematical reason**
- The SVC decision function f(x) is a monotone score.
- ROC-AUC on f equals ROC-AUC on any monotone transform of f, including Platt
  probabilities.
- Subject aggregation is the mean margin over windows; the fixed threshold is 0,
  the SVM's own decision boundary.
- Calibrated probabilities are only needed when probabilities themselves are
  interpreted. In that case calibration must be grouped by subject.

**Final implementation**
- `model.calibration: none` (default; `score_type = decision_function`,
  threshold `evaluation.decision_threshold = 0`).
- `model.calibration: grouped_platt` is optional. It uses `GroupedPlattSVC`:
  StratifiedGroupKFold over subjects, Platt fit on out-of-fold margins.
- Predictions are stored as `score`; the summary and manifest record the score
  type and threshold.
- LR and GNNs output probabilities, threshold 0.5.

**Alternatives considered**
- Window-level calibration, rejected.
- Isotonic calibration, not implemented.

**Effect on interpretation**
- SVM AUCs are unaffected by calibration.
- SVM threshold-based metrics use the margin at 0.
- Mean-margin aggregation weights confident windows more than averaging
  probabilities would.

**Code / config**
- `src/adgraph/models/classical.py` (`score_type`, `GroupedPlattSVC`)
- `src/adgraph/experiments/runner.py` (`model_score_type`, `score_threshold`)
- `configs/experiments/classical_base.yaml`, `base.yaml`

**Validating tests**
- `test_leakage.py::test_primary_svm_uses_decision_function`,
  `test_svm_calibration_is_subject_grouped`, `test_grouped_platt_requires_groups`
- `test_metrics.py::test_decision_scores_use_their_own_threshold`
- `test_runner.py::test_svm_variant_reports_decision_scores`

**Approval status**
- Author-specified (2026-10-12).

---

## 16. Age sensitivity analysis

**Previous design**
- `DESIGN.md` v1.1 described residualizing EEG features on age (not implemented).
- Age-only baseline implemented.

**Problem discovered**
- In D2, HC are about 62 years old and AD about 80 (`docs/data_inventory.md`).
  Age-related EEG changes could drive classification.

**Scientific / mathematical reason**
- Comparing age-only, EEG-only and EEG + age on identical splits shows whether
  EEG adds information beyond age.
- The author specified the integration:
  - Classical models append age, standardized on training subjects, to the
    feature vector after the EEG transforms, so it is never removed by selection
    or mixed into PCA.
  - GNNs concatenate it to the pooled graph embedding immediately before the
    classifier, so it does not alter graph encoding or band fusion.
- Statistics are computed per training subject: one value per subject, so
  window counts do not weight them.

**Final implementation**
- `covariates: [age]`, `CovariateScaler`, `MultiBandGNN(n_covariates)`.
- `configs/experiments/age/<exp>_plus_age.yaml` extends each EEG-only config.

**Alternatives considered**
- Residualizing features on age (earlier DESIGN wording).
- Age-matched subsets. Neither implemented.

**Effect on interpretation**
- EEG + age ≈ age-only means EEG adds little beyond age.
- EEG-only ≫ age-only, with EEG + age > age-only, supports EEG information
  beyond age.
- Not run yet. The D1 subject `sub18` has age 7 in the sheet, which will enter
  these models as is unless corrected (**requires author confirmation**).

**Code / config**
- `src/adgraph/models/covariates.py`, `classical.py`, `multiband.py`, `trainer.py`
- `src/adgraph/graphs/dataset.py::add_covariates`, `configs/experiments/age/`

**Validating tests**
- `test_leakage.py::test_age_covariate_scaled_on_training_subjects_only`
- `test_models.py::test_covariates_enter_only_the_head`, `test_gnn_model_with_age`
- `test_runner.py::test_eeg_plus_age_runs`
- `test_configs.py::test_age_configs_cover_eeg_experiments`

**Approval status**
- Author-specified (2026-10-12); implemented, not run.

---

## 17. Dense PyTorch rather than PyTorch Geometric

**Previous design**
- `CLAUDE.md` initially listed `torch_geometric` in the stack (no code used it).

**Problem discovered**
- Not a defect. The models needed edge-weight-aware variants (entries 7–8) on
  fixed-size, fully connected graphs.

**Scientific / mathematical reason**
- Every graph has the same 68 nodes and a dense 68 × 68 weighted adjacency.
  Batched dense tensors (B, K, R, R) express all operations as matrix products
  with no sparse indexing overhead.
- They also make the modified layers short and auditable.
- Avoiding PyG removes a version-sensitive dependency from Colab setups.

**Final implementation**
- `models/gcn.py`, `edge_weighted_gat.py`, `weighted_graphsage.py`,
  `multiband.py`, `trainer.py` in plain PyTorch.

**Alternatives considered**
- PyG `GCNConv` / `GATConv` / `SAGEConv`. Its standard GAT/SAGE would not provide
  the required edge-weighted behaviour.

**Effect on interpretation**
- None on the science.
- Memory scales with B·K·R², which is small for R = 68.

**Code / config**
- `src/adgraph/models/`, `requirements.txt` (no PyG)

**Validating tests**
- `test_models.py` (all)

**Approval status**
- Approved (2026-10-12).

---

## Open items requiring author confirmation

1. Reasons for v1 choices:
   - Desikan-Killiany atlas, `mean_flip`
   - MNE inverse parameters (loose 0.2, depth 0.8, λ² = 1/9)
   - band edges (including the 10 Hz alpha split)
   - the 10 s edge trim
   - the adaptive rejection constants
   - The v1 manifest gives a reason only for replacing the fixed ±80 µV
     threshold.
2. The scientific rationale for 20 s windows (the decision itself is approved).
3. Dropping (rather than keeping) M1/M2 in both datasets.
4. The 1 Hz lower limit of the relative-power denominator. The 1–45 Hz range is
   verified to be inside the passband.
5. The D1 `sub18` age value (7) before the age analyses are run.
6. Final repeat, fold, epoch and grid sizes (currently provisional).
