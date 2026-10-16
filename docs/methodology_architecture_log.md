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
18. [Figures from saved outputs only](#18-figures-from-saved-outputs-only)
19. [Subject-level rather than window-level plotting](#19-subject-level-rather-than-window-level-plotting)
20. [PLV and |ciPLV| as separate visual conditions](#20-plv-and-ciplv-as-separate-visual-conditions)
21. [Uncertainty display: repeats and subject bootstrap](#21-uncertainty-display-repeats-and-subject-bootstrap)
22. [Significance only from saved, corrected statistics](#22-significance-only-from-saved-corrected-statistics)
23. [Display-only thresholds, exploratory projections, fusion weights](#23-display-only-thresholds-exploratory-projections-fusion-weights)
24. [Publication style and provenance](#24-publication-style-and-provenance)
25. [No representative band for Exp 2 / Exp 3](#25-no-representative-band-for-exp-2--exp-3)
26. [Main metrics: balanced accuracy and ROC-AUC](#26-main-metrics-balanced-accuracy-and-roc-auc)
27. [Primary models](#27-primary-models)
28. [PLV primary, |ciPLV| predefined robustness analysis](#28-plv-primary-ciplv-predefined-robustness-analysis)
29. [Frozen main-paper figure set](#29-frozen-main-paper-figure-set)
30. [Comparable timing and single-model embeddings](#30-comparable-timing-and-single-model-embeddings)
31. [Checkpointing and resume](#31-checkpointing-and-resume)
32. [Pre-real-run reliability safeguards](#32-pre-real-run-reliability-safeguards)
33. [Pre-real-data audit fixes and the one-experiment-at-a-time workflow](#33-pre-real-data-audit-fixes-and-the-one-experiment-at-a-time-workflow)
34. [Memory audit of source reconstruction and graph construction](#34-memory-audit-of-source-reconstruction-and-graph-construction)
35. [Progress display, final Colab verification, alpha-band filter verification](#35-progress-display-final-colab-verification-alpha-band-filter-verification)
36. [Connectivity band-pass transition widths](#36-connectivity-band-pass-transition-widths)

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

## 18. Figures from saved outputs only

**Previous design**
- None. This is the first plotting code.

**Problem discovered**
- Figures must be reproducible and must not contain hand-entered or invented numbers.

**Scientific / mathematical reason**
- A figure is a view of saved data, so it must be regenerable from committed config +
  commit hash + the saved files (CLAUDE.md rule 7).
- Plot functions therefore take only:
  - an index of saved experiment outputs;
  - a saved graph run;
  - an optional saved statistics file.
- They never train, never recompute features from EEG, and contain no numeric results.

**Final implementation**
- `src/adgraph/visualization/`:
  - `results.py`: indexes `manifest.json` + `metrics_repeats.csv` directories.
  - `figures.py`: registry and driver; `configs/figures/paper.yaml` holds the inventory.
  - `writer.py`: saves each figure with its exact plotted data.
- The runner now also saves the inputs some figures need:
  - `n_parameters` and `fit_seconds` (`hparams.csv`);
  - optional `embeddings.npz`;
  - `roi_coords` in graph runs (fsaverage label centroids).
- Duplicated results across runs are an error unless `duplicates: latest` is chosen
  explicitly.

**Alternatives considered**
- Plotting inside the runner. Rejected: it would couple figures to training.

**Effect on interpretation**
- Every plotted value can be traced to a source file, recorded with sha256 in the manifest.

**Code / config**
- `src/adgraph/visualization/*`, `configs/figures/*.yaml`, `scripts/make_figures.py`
- `src/adgraph/experiments/runner.py`, `src/adgraph/source/inverse.py::roi_centroids`

**Validating tests**
- `test_visualization.py::test_performance_values_come_from_saved_metrics`,
  `test_generalization_ci_is_centred_on_saved_estimate`, `test_roc_matches_saved_subject_auc`,
  `test_confusion_matches_saved_summary`, `test_index_and_duplicates`, `test_make_figures_full_inventory`

**Approval status**
- Author-specified (2026-10-12).
- Figure selections were resolved by the author before any result (entries 25–29).

---

## 19. Subject-level rather than window-level plotting

**Previous design**
- None.

**Problem discovered**
- Each subject contributes many windows. With 20 s / 50 % overlap, consecutive windows
  share half their samples.
- Plotting windows as points, or computing error bars over windows, would treat strongly
  dependent samples as independent observations. That inflates apparent precision roughly
  by √(windows per subject) and weights subjects by their recording length.

**Scientific / mathematical reason**
- The subject is the unit of inference (entry 14).
- Every figure aggregates to subjects first:
  - classification figures use subject scores (mean of window scores);
  - PSD and connectivity figures average windows within subject before group means and SDs;
  - fusion weights are averaged per subject;
  - embeddings are averaged per subject.

**Final implementation**
- `results.subject_scores`, `anatomy.subject_level`, `fusion.subject_band_weights`.
- No figure function uses window rows as plotted observations.

**Alternatives considered**
- Window-level plots with subject colouring. Rejected: they still suggest window-level n.

**Effect on interpretation**
- n in every figure is the number of subjects (stated in titles, legends and metadata).

**Code / config**
- `src/adgraph/visualization/subjects.py`, `anatomy.py`, `spectral.py`, `connectivity.py`,
  `fusion.py`, `embedding.py`

**Validating tests**
- `test_psd_heatmap_subject_level_and_stats_guard` (mean of subject means, not of windows)
- `test_subject_scores_one_point_per_subject_anonymized`, `test_roc_matches_saved_subject_auc`
  (n = subjects), `test_embeddings`

**Approval status**
- Author-specified (2026-10-12).

---

## 20. PLV and |ciPLV| as separate visual conditions

**Previous design**
- None.

**Problem discovered**
- Combining PLV and |ciPLV| results (averaging, or picking the better per model) would mix
  the primary analysis with its sensitivity analysis, and would hide the leakage check that
  |ciPLV| provides.

**Scientific / mathematical reason**
- PLV includes zero-lag coupling; |ciPLV| suppresses it (entry 10).
- Showing both side by side, with the same model, splits and seeds, lets the reader see
  whether a result depends on zero-lag coupling.
- Paired comparisons are only valid when folds are identical. The paired figure verifies
  `folds.csv` equality and refuses unmatched pairs.

**Final implementation**
- `CONDITION_STYLE`: PLV = blue filled circle, |ciPLV| = vermillion open square.
- Distinct in hue, shape and fill, so they stay distinguishable in grayscale.
- Signed ciPLV is never plotted as connectivity.

**Alternatives considered**
- Separate figures per condition (still possible via `conditions`).

**Effect on interpretation**
- Any claim must state which condition it rests on.

**Code / config**
- `visualization/style.py`, `performance.py::matched_condition_pairs`

**Validating tests**
- `test_paired_conditions_require_identical_folds`, `test_performance_values_come_from_saved_metrics`
  (conditions kept apart), `test_connectivity_matrices_and_network` (signed ciPLV refused)

**Approval status**
- Author-specified (2026-10-12).

---

## 21. Uncertainty display: repeats and subject bootstrap

**Previous design**
- None.

**Problem discovered**
- Per-fold metrics come from 4–5 test subjects: AUC is unstable or undefined there.
- Cross-dataset settings have a single split.

**Scientific / mathematical reason**
- One CV repeat yields one prediction per subject, so its metric is a subject-level estimate.
- The spread across repeats shows split variability. It is not a confidence interval,
  because repeats reuse the same subjects.
- For a CI, subjects are resampled (bootstrap of the repeat-averaged subject scores, as in
  `summary.json`).
- The marker is then the metric of those repeat-averaged scores, the quantity the CI
  describes, not the mean of per-repeat metrics.

**Final implementation**
- `uncertainty: repeats` (points + mean ± SD) or `bootstrap` (point estimate + 95 % CI from
  `summary.json`).
- The generalization figure always uses the bootstrap CI.

**Alternatives considered**
- Per-fold boxplots (rejected).
- Pooled window bootstrap (rejected: wrong unit).

**Effect on interpretation**
- SD across repeats ≠ uncertainty about population performance.
- Figure metadata states which one is shown.

**Code / config**
- `visualization/performance.py::strip_panel`, `collect_metrics`

**Validating tests**
- `test_generalization_ci_is_centred_on_saved_estimate`,
  `test_performance_values_come_from_saved_metrics`

**Approval status**
- Proposed; pending author approval.

---

## 22. Significance only from saved, corrected statistics

**Previous design**
- None.

**Problem discovered**
- Significance marks computed ad hoc in plotting code would:
  - be untracked tests;
  - typically be uncorrected for multiplicity;
  - often use the wrong unit (windows).

**Scientific / mathematical reason**
- Inference belongs to a separate, documented analysis.
- The plotting code only displays results from a statistics CSV whose rows are validated:
  - unit = subject;
  - named test;
  - n_subjects > 0;
  - corrected p in [0, 1] and not below the raw p;
  - non-empty correction method and family.
- Without such a file, figures are labelled `descriptive` and show no stars.
- For fusion weights, the AD vs HC split itself is refused without the statistics.

**Final implementation**
- `visualization/stats.py` (`load_stats`, `lookup`, `stars`). Comparison ids per figure
  family are documented in docs/figures.md.

**Alternatives considered**
- Computing tests in figures. Rejected.

**Effect on interpretation**
- Readers can tell descriptive from inferential figures (`kind` in `_meta.json`).

**Code / config**
- `visualization/stats.py`, `performance.annotate_from_stats`, `spectral.fig_psd_heatmap`,
  `fusion.fig_fusion_weights`

**Validating tests**
- `test_stats_validation`, `test_performance_significance_only_from_stats`,
  `test_fusion_weights_and_group_split_guard`, `test_psd_heatmap_subject_level_and_stats_guard`

**Approval status**
- Author-specified (2026-10-12).

---

## 23. Display-only thresholds, exploratory projections, fusion weights

**Previous design**
- None.

**Problem discovered**
- Three plots invite over-interpretation:
  - Network plots need an edge selection to be readable. Selecting by magnitude on the
    full labelled data would look like a finding.
  - PCA/UMAP clusters look like evidence.
  - Attention weights look like importance.

**Scientific / mathematical reason**
- **Network edges.** Selected only for display (top-N by magnitude). The selection is
  labelled in the figure title, the data file (`display_rank`) and the metadata
  (`display_only_threshold.statistical = false`).
- **Projections.** Unsupervised (labels unused) and titled "exploratory, not statistical
  evidence".
- **Graph embeddings.** Taken from one fitted model. Embedding spaces of different folds
  are not aligned, so pooling them would be meaningless.
- **Fusion weights.** Describe how a fitted model combines bands, not biological importance.

**Final implementation**
- `connectivity.fig_connectivity_network`, `embedding.py`, `fusion.py`.

**Alternatives considered**
- Statistical edge thresholds (need a saved statistics file; not implemented).
- Aligned embeddings across folds (not meaningful).

**Effect on interpretation**
- All three are descriptive only.

**Code / config**
- `visualization/connectivity.py`, `embedding.py`, `fusion.py`

**Validating tests**
- `test_connectivity_matrices_and_network`, `test_embeddings`,
  `test_fusion_weights_and_group_split_guard`

**Approval status**
- Author-specified (2026-10-12).

---

## 24. Publication style and provenance

**Previous design**
- None.

**Problem discovered**
- Figures must meet journal requirements and be reproducible.

**Scientific / mathematical reason**
- **Style:**
  - Central style (`configs/figures/style.yaml`): column widths, font sizes, line widths,
    marker sizes.
  - Okabe–Ito palette, safe for common colour-vision deficiencies. Categories also differ
    by marker and fill, for grayscale.
  - `cividis` sequential map (monotonic lightness); `RdBu_r` diverging map with symmetric
    limits. Diverging sign is lost in grayscale, so values are in the data files.
- **Output:** PDF/SVG with editable text, plus PNG at 600 dpi.
- **Provenance:** every figure saves its plotted data and metadata. The manifest has the
  config, sources (sha256), timestamp and git commit. Files are never overwritten.
- **Synthetic outputs** are watermarked.

**Final implementation**
- `visualization/style.py`, `writer.py`, `configs/figures/style.yaml`.

**Alternatives considered**
- Interactive figures. Not used for publication.

**Effect on interpretation**
- None on the science; ensures reproducibility.

**Code / config**
- as above

**Validating tests**
- `test_writer_never_overwrites_and_records_provenance`, `test_style_sizes`,
  `test_make_figures_full_inventory`

**Approval status**
- Author-specified (2026-10-12). Exact journal widths and fonts require author
  confirmation for the target journal.

---

## 25. No representative band for Exp 2 / Exp 3

**Previous design**
- `paper.yaml` represented Exp 2 and Exp 3 in the overall comparison by their alpha1
  variant (GCN), marked as a placeholder.

**Problem discovered**
- Exp 2/3 are single-band models. Representing each by one band forces a band choice.
- Any choice made after seeing results would be selection on test performance, inflating
  the apparent performance of Exp 2/3 in the overall figure.

**Scientific / mathematical reason**
- Selecting the maximum of four correlated estimates biases the selected estimate upward:
  the winner's curse.
- Keeping Exp 2/3 band-specific removes that degree of freedom.
- The band question is answered where it belongs: the band-specific comparison
  (Fig. 4 / S07 / S08).

**Final implementation**
- The overall comparison (Fig. 3, S01–S05, S21) uses `entries_overall` = Exp 0 (LR, SVM),
  Exp 1 (LR, SVM), Exp 4, Exp 5 (attention, shared encoder) and Exp 6 full.
- Exp 2/3 appear only in `band_comparison` figures, across all four bands.

**Alternatives considered**
- A pre-specified single band. Rejected: no a priori basis was established.
- The best band. Rejected: selection bias.

**Effect on interpretation**
- Fig. 3 compares only models without a band choice.
- Exp 2/3 conclusions are band-specific statements.

**Code / config**
- `configs/figures/paper.yaml` (`entries_overall`)

**Validating tests**
- `test_visualization.py::test_frozen_main_figure_set_and_selections` (Exp 2/3 absent from
  every non-band figure)

**Approval status**
- Author decision (2026-10-12), made before any real result.

---

## 26. Main metrics: balanced accuracy and ROC-AUC

**Previous design**
- The main performance figure showed five metrics.

**Problem discovered**
- Five metrics per model dilute the main message and multiply implicit comparisons.

**Scientific / mathematical reason**
- **Balanced accuracy** is insensitive to the class imbalance (D2 has 10 AD / 13 HC) and
  evaluates the fixed decision rule.
- **ROC-AUC** is threshold-free and uses the scores directly. For the primary SVM those are
  margins, so no calibration is needed.
- Together they cover the decision and ranking aspects.
- Accuracy, F1, sensitivity and specificity are still computed and stay available.

**Final implementation**
- `metrics_main` = [balanced_accuracy, roc_auc] for Fig. 3 and Fig. 6.
- `metrics_supp` (accuracy, F1, sensitivity, specificity) in S01.
- All metrics remain in `metrics_repeats.csv` and `summary.json`.

**Alternatives considered**
- MCC as a main metric (available in tables).

**Effect on interpretation**
- Main-text claims rest on balanced accuracy and ROC-AUC.

**Code / config**
- `configs/figures/paper.yaml`

**Validating tests**
- `test_frozen_main_figure_set_and_selections`

**Approval status**
- Author decision (2026-10-12).

---

## 27. Primary models

**Previous design**
- Placeholder primary list (Exp 6 full, Exp 5, Exp 1 SVM, Exp 0 SVM).

**Problem discovered**
- Primary models must be fixed before results.

**Scientific / mathematical reason**
- Three models span the paper's argument:
  - **Exp 0 SVM:** the conventional sensor-space spectral baseline.
  - **Exp 1 SVM:** what source localization alone adds.
  - **Exp 6 full:** the proposed graph model.
- The SVM is the baseline representative because its decision function gives
  calibration-free scores (entry 15).

**Final implementation**
- `entries_primary` is the default for ROC (S10), confusion (S11), subject scores (S12) and
  Fig. 6 (generalization).
- Using the same three in Fig. 6 is a consequence of this decision, applied for
  consistency.

**Alternatives considered**
- Including Exp 4/5 in ROC/generalization. Available by configuration, supplementary.

**Effect on interpretation**
- Primary claims compare these three models.

**Code / config**
- `configs/figures/paper.yaml` (`entries_primary`)

**Validating tests**
- `test_frozen_main_figure_set_and_selections`

**Approval status**
- Author decision (2026-10-12).
- Applying the same set to Fig. 6 follows from it; author confirmation welcome.

---

## 28. PLV primary, |ciPLV| predefined robustness analysis

**Previous design**
- Entries 10 and 20 already kept PLV and |ciPLV| separate.

**Problem discovered**
- The hierarchy had to be explicit in the figure documentation, so that |ciPLV| results
  cannot later be promoted to primary when they look better.

**Scientific / mathematical reason**
- Fixing the hierarchy in advance prevents choosing the more favourable measure after
  the fact.
- |ciPLV| answers a predefined question: is the result robust to removing zero-lag
  coupling?
- A direct comparison is meaningful only on identical subjects, folds and repeats
  (paired), because split variability would otherwise be confounded with the measure.

**Final implementation**
- `docs/figures.md` "Pre-result design decisions" §1.
- Fig. 2b and all main performance figures show PLV as primary. |ciPLV| appears as the
  second condition or in supplementary figures (S06, S17, S18b).
- `matched_condition_pairs` refuses pairs with different `folds.csv`.

**Alternatives considered**
- None. A post-hoc choice of measure is excluded.

**Effect on interpretation**
- The primary claim rests on PLV.
- |ciPLV| results qualify robustness and do not replace it.

**Code / config**
- `docs/figures.md`, `visualization/performance.py`

**Validating tests**
- `test_paired_conditions_require_identical_folds`

**Approval status**
- Author decision (2026-10-12).

---

## 29. Frozen main-paper figure set

**Previous design**
- Nine main figures, including ROC and age sensitivity.

**Problem discovered**
- Main-paper content should be fixed before results, so that it is not shaped by which
  results look best.

**Scientific / mathematical reason**
- Pre-specifying the main figures (and therefore the main claims) limits selective
  reporting. The figure order follows the paper's argument:
  1. pipeline
  2. source characterization
  3. overall performance
  4. band specificity
  5. ablations
  6. generalization
  7. multi-band interpretation

**Final implementation**
- Main set:
  - Fig. 1 conceptual, drawn later and not generated
  - Fig2a / Fig2b, Fig3, Fig4, Fig5, Fig6, Fig7 in `paper.yaml`
- Supplementary: ROC, confusion matrices, detailed |ciPLV| plots, subject-score
  distributions, PCA/embedding views, network views, complexity plots, age sensitivity.
- Promotion later requires a logged reason.

**Alternatives considered**
- ROC in the main text. Kept supplementary by author decision.

**Effect on interpretation**
- Supplementary figures support but do not define the main claims.

**Code / config**
- `configs/figures/paper.yaml`, `docs/figures.md`

**Validating tests**
- `test_frozen_main_figure_set_and_selections`, `test_make_figures_full_inventory`

**Approval status**
- Author decision (2026-10-12).

---

## 30. Comparable timing and single-model embeddings

**Previous design**
- Fit times were plotted without environment information.
- The graph-embedding figure defaulted to repeat 0, fold 0.

**Problem discovered**
- **Fit time** depends on hardware (CPU model, GPU, threads) and software versions.
  Combining runs from different Colab sessions or machines would compare environments,
  not models.
- **Embeddings** from independently trained models are in unaligned coordinate systems:
  arbitrary rotations, permutations and scalings of the latent space. Merging them, or
  relying on an implicit default, risks an uninterpretable plot.

**Scientific / mathematical reason**
- Timing is only comparable under a fixed environment. Parameter counts are
  environment-independent.
- An embedding plot is only meaningful within one fitted model.

**Final implementation**
- Each experiment manifest stores `compute_environment` (`provenance.compute_environment`):
  - device, CPU model and count, GPU name and CUDA version;
  - torch threads, platform, library versions;
  - a fingerprint of these, excluding host name and time.
- `fig_complexity` with `fit_seconds` raises `IncomparableTiming` if environments differ or
  are missing, and writes the environment into the figure metadata.
- `fig_graph_embedding` requires explicit integer `repeat` and `fold` and uses only that
  model's embeddings. It is titled exploratory, and its metadata records
  `single_fitted_model: true`.

**Alternatives considered**
- Normalizing times by a benchmark. Not reliable.
- Procrustes-aligning embeddings across folds. Adds a modelling step; not adopted.

**Effect on interpretation**
- Runtime claims are valid only within the recorded environment.
- Embedding plots describe one model.

**Code / config**
- `src/adgraph/provenance.py`, `experiments/runner.py::run_device`
- `visualization/performance.py::fig_complexity`, `visualization/embedding.py`

**Validating tests**
- `test_fit_time_requires_one_environment`, `test_graph_embedding_requires_one_fitted_model`

**Approval status**
- Author decision (2026-10-12).

---

## 31. Checkpointing and resume

**Previous design**
- Graph construction kept every subject in memory and saved the whole cohort once, at
  the end, into a new timestamped directory.
- The sensor→ROI operator (fsaverage fetch, forward model, inverse, ROI extraction) was
  rebuilt on every run.
- Experiments wrote their outputs only after all repeats, folds and permutations had
  finished, again into a timestamped directory.

**Problem discovered**
- Colab sessions disconnect and time out. Under the previous design a disconnect lost all
  work since the start of the run, and a rerun started from zero.
- Timestamped directories meant a rerun could not find its own earlier partial work, and
  nothing stopped results from different configurations or graph runs being mixed by hand.

**Scientific / engineering reason**
- Recomputation must not change results. A result may only be reused if it was produced
  from the same configuration, the same graph run, the same split, repeat, fold and seed,
  and the same code.
- An interrupted or corrupted output must never be mistaken for a completed one.

**Final implementation**
- `src/adgraph/checkpoint.py`:
  - Atomic writes: temp file, fsync, `os.replace`.
  - A unit is complete only when its marker `<name>.done.json` exists, its `identity`
    equals the expected identity, and every listed payload file has the recorded SHA-256.
    Payload is written first and the marker last.
  - States: complete / missing / incompatible / corrupt. Corrupt or incompatible units are
    moved to `_superseded/<stamp>/`, never deleted or overwritten.
  - `CODE_VERSIONS` per unit type; bumping one makes old units incompatible.
  - `status.csv` (pending / running / completed / failed) is informational only;
    completion is always decided by the marker.
- Graph construction (`pipeline.run`):
  - Output directory `graphs_v3_<config hash>/` (no timestamp), so a rerun of the same
    command finds its own work. `run_identity.json` guards against a different config.
  - One unit per subject under `subjects/<key>/`. Identity: config hash, operator id,
    input path, size and mtime, code version.
  - A subject that raises an error is recorded as failed and retried on the next run.
  - The cohort files (`graphs.npz`, `windows.csv`, `subjects.csv`, `manifest.json`) are
    assembled from the subject units. The assembly is redone only when the units change.
    The manifest carries a `graph_run_id`.
  - The operator is cached as `operator_<id>.npz` keyed by the source-model hash (source
    config, channels, montage, reference, MNE version). fsaverage is fetched only when
    no valid cached operator exists.
- Experiments (`runner.run_variant`):
  - Output directory `<out>/<experiment>/<variant>/<setting>/` (no timestamp).
  - Units: each outer fold (predictions, metadata, embeddings), each inner-CV trial (one
    hyperparameter combination on one inner fold) and each permutation.
  - Unit identity: training-config hash, `graph_run_id`, setting, split id (hash of the
    train/test subjects), repeat, fold, seed, trial parameters or permutation index, code
    version.
  - The training hash excludes evaluation-only keys (`n_repeats`, `cross_n_seeds`,
    `n_bootstrap`, `permutation`). Raising `n_repeats` therefore reuses existing folds.
    Any training-relevant change gives a new identity.
  - An existing directory with a different identity raises `IncompatibleCheckpoint`.
    Only `--force` supersedes it.
  - Final files are written atomically, with a final marker written last.
- CLI (`build_graphs.py`, `run_experiment.py`):
  - `--resume` is the default.
  - `--status` reports progress without running anything.
  - `--force` recomputes intentionally (new directory or superseded outputs).
  - `--no-resume` refuses an existing directory.
- Figures: `index_results` ignores `_superseded/` and runs without a valid final marker.
  Notebooks pick graph runs with `latest_complete_graph_run`.

**Alternatives considered**
- Keep timestamped directories and add a separate resume pointer. More state to keep
  consistent; not adopted.
- Use the status table as the source of truth. A crash can leave it stale; not adopted.
- Content-hash the raw EDF inputs. Robust, but reads every file on every run over Drive;
  size + mtime used instead (see caveats).
- Delete incompatible outputs. Rejected: outputs are never destroyed (CLAUDE.md rule 6).

**Effect on interpretation**
- None on results: a resumed run produces the same outputs as an uninterrupted run with
  the same seeds (tested).
- Caveats:
  - Input identity uses file size and mtime, not a content hash. An edited file with
    unchanged size and mtime would not be detected. *(Resolved by entry 32.)*
  - `CODE_VERSIONS` must be bumped by hand when a change alters what a unit computes.
    *(Resolved by entry 32.)*
  - Failed subjects are retried automatically on every run until they succeed.

**Code / config**
- `src/adgraph/checkpoint.py`, `src/adgraph/pipeline.py`, `src/adgraph/experiments/runner.py`
- `src/adgraph/graphs/store.py::graph_run_id`, `src/adgraph/visualization/results.py::index_results`
- `scripts/build_graphs.py`, `scripts/run_experiment.py`

**Validating tests**
- `tests/test_checkpoint.py` (synthetic data, simulated crashes, corrupted and torn
  files, config changes, `--force`, CLI subprocess, figure-only reads)
- `tests/test_runner.py::test_outputs_never_overwritten`

**Approval status**
- Author-specified (resumability requirements, 2026-10-12). Implementation details
  approved in principle.

---

## 32. Pre-real-run reliability safeguards

**Previous design** (entry 31)
- **Storage.** Output locations were not checked, and the notebook's smoke test wrote to
  `/content/smoke`.
- **Code.** Unit identities depended on code only through the manual `CODE_VERSIONS`
  labels.
- **Input files.** Recordings were identified by path, size and mtime.
- **Dependencies.**
  - `edfio` was an unpinned test dependency.
  - Tests that need `edfio` or `torch` used `pytest.importorskip`, so a missing package
    produced a green run with skips. This happened once after a container reset: 6 graph
    tests were skipped.

**Problem discovered**
- **Storage.** In Colab only the mounted Drive survives a disconnect. A checkpoint
  written elsewhere defeats resumption. A path under `/content/drive` written while Drive
  is not mounted silently lands on the VM disk.
- **Code.** Forgetting to bump `CODE_VERSIONS` after a numerical change would silently
  reuse stale scientific results.
- **Input files.** Path, size and mtime do not identify content. An EDF replaced in place
  can keep its size, and a re-mounted Drive changes paths without changing content.
- **Dependencies.** A suite that passes with skipped graph-construction tests does not
  validate the real-run environment.

**Scientific / engineering reason**
- A cached result may be reused only if it is the result the current code would compute
  from the current data. Both "the current code" and "the current data" must be checked
  automatically, not by convention.

**Final implementation**
- **Persistent storage** (`src/adgraph/storage.py`):
  - A location is ephemeral if it is under `/content` but not under `/content/drive`.
  - It is also ephemeral if it is under `/content/drive` while that is not a mount point.
  - In a Colab runtime, any location outside `/content/drive` is ephemeral.
  - These locations are checked:
    - `pipeline.run`: run directory, operator cache, fsaverage `subjects_dir` and input
      hash cache;
    - `run_variant`: the output root, for non-synthetic stores;
    - the CLIs (`build_graphs`, `run_experiment`, `make_figures`, `inspect_raw`): their
      output paths.
  - An ephemeral location raises `EphemeralStorageError` before anything is written.
    `--allow-ephemeral-storage` downgrades this to a warning.
  - Synthetic runs are exempt.
  - The notebook smoke test now writes to Drive (`graphs_v3_<hash>_limit1/`).
- **Automatic code fingerprint** (`checkpoint.code_identity(scope)`):
  - It contains:
    - the git commit;
    - SHA-256 over the source files of the package parts that determine the computation
      (`CODE_SCOPES`: operator, graph, experiment; visualization is excluded, so figure
      edits never invalidate results);
    - the versions of the numerical libraries (`SCOPE_LIBRARIES`).
  - It is included in the identities of:
    - the operator;
    - subject units;
    - the graph assembly, and therefore `graph_run_id`;
    - experiment directories and all their units.
  - The source hash also catches uncommitted edits, and it works without git.
  - `CODE_VERSIONS` stays as a documentation label only.
  - `--status` reports which identity fields changed (`changed: code / input / metadata`).
- **Raw-data fingerprint** (`io/fingerprint.py`):
  - Subject identity uses the EDF's SHA-256 instead of path and mtime. It also includes a
    hash of the subject's metadata row (label, age, sex, MMSE, ACE-R, comment), so a
    corrected sheet recomputes only the affected subject.
  - Participant sheets and the electrode layout file are hashed into the assembly identity
    and the manifest (`auxiliary_inputs`).
  - The manifest lists each input's SHA-256.
  - Hashes are cached in `<cache_dir>/input_sha256.json`:
    - keyed by absolute path and validated by size and `mtime_ns`;
    - written after each newly hashed file;
    - re-read in full with `--rehash-inputs`.
- **Dependencies:**
  - `edfio==0.4.18` is pinned in `requirements.txt` and in the `pyproject` test extra.
  - `provenance.verify_environment()` checks every exact pin before a real run (CLIs).
  - The pytest plugin `tests/dependency_guard.py` makes any "could not import" skip fail
    the session.
  - The notebooks run `pytest -q -rs` before the smoke test.
- **Resuming across commits.** `run_identity.json` and `--status` record the commit. The
  notebooks have a `COMMIT` variable that pins the checkout for resuming.

**Alternatives considered**
- **Source hash only, without the git commit.** This would let commits that touch only
  documentation keep the cache. Not adopted: the author asked for the commit in the
  identity. The cost is that any new commit invalidates checkpoints, which is why the
  notebooks offer commit pinning.
- **Hashing the whole repository.** Rejected: notebook, test and documentation edits would
  invalidate results; the commit already covers them.
- **Re-hashing every input file on every resume.** Rejected: it reads the whole dataset
  over the Drive mount each time.
- **Hard-failing on a dirty working tree.** Not adopted: the source hash already captures
  the edits. `git_dirty` is recorded in manifests and `--status`.

**Effect on interpretation**
- None on results.
- Any reused unit was produced by byte-identical relevant source, the same commit, the same
  numerical library versions and byte-identical input content.
- Remaining limitations:
  - **Hash cache.** A file edited in place with unchanged size and nanosecond mtime would be
    served from the hash cache. `--rehash-inputs` closes this.
  - **New commits.** Any new commit triggers recomputation. Resume with the run's commit.
  - **Libraries.** Library versions outside `SCOPE_LIBRARIES` (and the GPU/driver) are
    recorded in `compute_environment` but are not part of the identity.

**Code / config**
- `src/adgraph/storage.py`, `src/adgraph/io/fingerprint.py`
- `src/adgraph/checkpoint.py` (`code_identity`, `CODE_SCOPES`, `identity_diff`)
- `src/adgraph/provenance.py` (`source_fingerprint`, `git_dirty`, `verify_environment`)
- `src/adgraph/pipeline.py`, `src/adgraph/experiments/runner.py`, `scripts/*.py`
- `requirements.txt`, `pyproject.toml`, `tests/dependency_guard.py`, notebooks 01–03

**Validating tests**
- `tests/test_reliability.py` (synthetic only):
  - storage classification;
  - refusal before writing (library, experiment, CLI);
  - source edits within and outside the scope;
  - a code, commit or library change recomputes graph units without bumping
    `CODE_VERSIONS`;
  - a code change is refused in experiments and recomputed under `--force`;
  - the hash cache (including the documented size/mtime limit and `--rehash-inputs`);
  - a content change recomputes only that subject;
  - moved data is not recomputed; a metadata change recomputes only that subject;
  - hashes are cached across resumes;
  - pin checks;
  - a dependency skip fails the suite.

**Approval status**
- Author-specified (reliability requirements before the first real-data run, 2026-10-12).
  Implementation details approved in principle.

---

## 33. Pre-real-data audit fixes and the one-experiment-at-a-time workflow

**Previous design**
- **Notebooks.**
  - Notebook 02 had commented-out experiment commands.
  - Shell steps (`!cmd`) neither stopped "Run all" on failure nor showed subprocess output.
  - The smoke test wrote to ephemeral `/content`.
- **Experiments.** There was no per-experiment runtime estimate or report.
- **Audit.** A read-only audit of the graph pipeline, the experiment framework and the figure
  pipeline (2026-10-13, synthetic data only) found the defects listed below.

**Problem discovered** (verified by reading the code and by synthetic reproduction)
1. **Inner-trial unit collision (blocker).** Inner-CV trial checkpoints were named by the
   outer split's subject sets, but their identity includes the seed. D1 has only 45 × 45
   distinct 5-fold test folds, so splits recur across 20 repeats. Every `within_D1` run
   (all configs with a grid) stopped with `IncompatibleCheckpoint`.
2. **GNN loss normalised per batch.** The loss was `Σ w·l / Σ w(batch)`, so every batch
   counted equally whatever its weight. This only approximates the documented "one vote per
   subject, classes balanced" weighting (methods decisions #29).
3. **Degenerate dataset-probe permutation test.** Labels were permuted within each dataset.
   For the dataset-identity target they are constant within a dataset, so the "permuted"
   labels equalled the real ones and p = 1.
4. **Collisions between analyses.**
   - Runs with `target: dataset` or `scheme: loso` wrote to the primary result directory.
   - The figure index could plot a probe, LOSO or calibrated run as a primary result.
   - Results from different graph runs could be mixed in one figure.
5. **Figure errors and gaps were hidden.**
   - Every exception became a "skipped" figure, so code bugs looked like missing inputs.
   - Figures drawn with some entries missing said nothing.
   - A latent selection bug: an empty `where` match selected zero columns.
6. **S20 and S21b inputs.**
   - S20 needed Exp 6 embeddings, which no config saved.
   - S21b compared fit times by one fingerprint per run, but a resumed run can mix machines.
7. **`run_experiment.py`.**
   - A failed cross-dataset gate aborted all remaining variants.
   - `--variants plv` also matched `ciplv`.
8. **Metadata.**
   - A participant row with an unrecognised Group (e.g. "Control") was silently dropped,
     and its recording was then processed with no label.
   - A missing sex became the string "NAN".
9. **Diagnostics only.**
   - The header report could not show EDF units.
   - Channel-name suffixes `-A1/-A2/-M1/-M2` were not recognised.
10. **Drive I/O.** Classical inner-CV trials took milliseconds each but wrote a checkpoint
    each. Exp 0 alone writes about 12 700 units, roughly an hour of Drive writes for about
    5 minutes of computation.

**Scientific / engineering reason**
- Defects 1–8 could crash the real run or produce results that differ from the documented
  protocol, or attribute them to the wrong analysis.
- The author asked to run and inspect one experiment at a time without editing anything,
  keeping the full protocol (20 repeats, tuning, all model configurations), with a realistic
  cost estimate before launching (instruction of 2026-10-13).

**Final implementation**
- **Fixes:**
  1. Inner trials are named after their outer fold unit (`inner_fold_rRRR_fFF_…`).
  2. GNN loss is `Σ w·l / batch size`, with weights of mean 1 over the training set.
  3. The probe permutes across all subjects.
  4. Separate analyses and provenance:
     - `runner.result_dir` adds `__target-<t>` and `__scheme-<s>` to the variant directory;
     - `index_results` keeps primary runs only and records `graph_run_id` and `git_commit`;
     - `make_figures` refuses results from several graph runs, or from a graph run other
       than `--graphs`.
  5. Figure errors and gaps:
     - a typed `MissingInputs` exception;
     - code errors fail the figure run after the other figures are written;
     - missing entries are recorded per figure (`missing_inputs`, `incomplete`);
     - the empty-selection bug is fixed.
  6. Exp 6 `full` saves embeddings; each fold records its compute fingerprint.
  7. `run_experiment.py` skips gated cross settings with a message, matches variant tokens
     exactly, and reuses the dataset across settings.
  8. Metadata:
     - an unrecognised Group raises;
     - recordings without a label stop graph construction before any processing;
     - a missing sex becomes "".
  9. EDF header units are reported, and reference suffixes are recognised.
  10. Inner-trial checkpoints are kept only for GNNs. Outer folds are checkpointed for all
      models. Results are identical.
  11. GNN training tensors are moved to the device once per fit, not per batch. Batches and
      results are identical.
- **One experiment at a time:**
  - `scripts/experiment.py estimate | run | status | report` handles one config:
    - `run` covers all its variants and settings in the order within → pooled → cross
      (gated);
    - it isolates failures and writes a per-experiment status file;
    - it never starts another experiment.
  - `adgraph.experiments.estimate`:
    - exact fit counts from the real splits;
    - one fit timed per architecture on the current runtime;
    - the measured Drive checkpoint cost;
    - remaining time.
  - `adgraph.experiments.report` writes a results table and single-experiment diagnostic
    figures (D1–D4), labelled as not comparative.
- **Notebooks (generated by `scripts/make_notebooks.py`):**
  - 01 builds the graphs;
  - one notebook per experiment config: Exp0–Exp6, plus Exp3b, baselines, the dataset probe
    and EEG + age in `additional/`;
  - 03 makes the comparative figures;
  - an advanced whole-plan notebook that starts nothing unless `RUN_EVERYTHING = True`.
- **Shared setup cell:**
  - locates the data and a writable output folder on Drive;
  - pins the code commit on Drive for all sessions;
  - installs the pinned environment;
  - stops "Run all" on any failure.
- **Automatic checks in notebook 01:** inputs, tests and EDF headers are checked before any
  graph is built.

**Alternatives considered**
- **Running the whole plan from one notebook.** Rejected by the author: each result must be
  reviewed before the next experiment starts.
- **Reducing repeats or grids to save time.** Rejected by the author: the full protocol is kept.
- **Per-trial Drive checkpoints for classical models.** Rejected: a write costs more than the
  fit. Outer folds remain checkpointed.

**Effect on interpretation**
- No method or protocol parameter changed.
- Fix 2 changes GNN optimisation numerically, to match the documented weighting.
- Fix 3 makes the probe's permutation p-value meaningful.
- Fixes 4–5 prevent misattributed or silently incomplete figures.
- **Open author decision (not changed):** the connectivity band-pass filters use MNE's
  automatic transition bands. Their −6 dB bands are theta 3.0–9.0, alpha1 7.0–11.25,
  alpha2 8.8–14.6 and beta 11.4–33.7 Hz, so alpha1 and alpha2 overlap substantially.
  Narrower transitions would change the method and need approval (open item 8).

**Code / config**
- `src/adgraph/experiments/{runner,plan,estimate,report}.py`, `src/adgraph/models/trainer.py`
- `src/adgraph/visualization/{results,figures,performance,subjects,embedding,fusion,connectivity}.py`
- `src/adgraph/io/{participants,eeg}.py`, `src/adgraph/pipeline.py`, `src/adgraph/colab.py`
- `scripts/{experiment,make_notebooks,colab_setup,run_plan,run_experiment,inspect_raw,make_figures}.py`
  (`estimate_runtime_synthetic.py` was removed in entry 35)
- `configs/experiments/exp6_multiband_psd_strength.yaml`, `configs/run_plan.yaml`, `notebooks/`

**Validating tests**
- `tests/test_audit_fixes.py`:
  - recurring outer split;
  - probe permutation;
  - result directories;
  - variant filter;
  - loss normalisation;
  - S20 embeddings;
  - unrecognised Group;
  - channel suffixes;
  - unlabelled recordings;
  - missing and incomplete figures;
  - code errors vs missing inputs;
  - graph-run mismatch;
  - primary-only index.
- `tests/test_notebooks.py`:
  - notebooks match the generator;
  - one notebook per config;
  - each runs only its experiment;
  - the advanced notebook is disabled by default;
  - the CLI estimate → run → status → report on classical and GNN configs.
- `tests/test_colab_plan.py`: setup helpers, code pin, input check, run plan.
- `tests/test_checkpoint.py` (updated): classical inner searches are redone on resume; GNN
  inner trials are reused.

**Approval status**
- One-experiment-at-a-time workflow, full protocol and estimates: author-specified (2026-10-13).
- Bug fixes: approved in principle (author asked to fix identified bugs).
- Band-filter widths: pending (open item 8).

---

## 34. Memory audit of source reconstruction and graph construction

**Previous design** (audited 2026-10-13 by code reading and `tracemalloc` profiling of a
synthetic worst-case recording: 20 min, 500 Hz, 34 channels, 41 MB EDF, larger than any
AD-LEB EDF, which are 20–39 MB)
- **Already sound:**
  - Recordings were processed one subject at a time.
  - The source model was reduced once to the 68 × 30 operator `W`, built from a 30-column
    identity probe and verified on 200 random samples. It is cached on Drive, and a real
    recording only ever goes through `W @ data`. The full 20 484-vertex source time series is
    never formed.
  - Subject checkpoints were atomic, with the marker written last.
- **Wasteful:**
  - `read_raw_edf(preload=True)` loaded every EDF channel at the original rate before picking.
  - Band-pass + Hilbert ran on all 68 ROIs at once, and the previous band's analytic signal
    was still alive while the next was computed.
  - A float32 copy of the ROI signals was always made, even when not saved.
  - The previous subject's arrays stayed referenced while the next subject was processed.
  - Assembly kept per-subject lists and then concatenated them, so the cohort was held twice.
  - `atomic_savez` built the whole compressed archive in RAM.
  - `GraphStore.validate` ran `allclose(a, aᵀ)` over whole cohort arrays (about 3–4× their
    size in temporaries), at assembly and at every experiment load.
- **Unmonitored:** memory was not recorded anywhere.

**Problem discovered**
- No out-of-memory risk at AD-LEB sizes:
  - worst-case peaks were about 0.3 GB for loading and preprocessing, and 1.5 GB for
    connectivity and PSD;
  - Colab has about 12.7 GB.
- The connectivity peak was still about 9× the ROI signal size. Assembly and validation
  scaled at 2–4× the cohort size, which matters for longer recordings or larger cohorts.
- Without memory logging, an OOM kill could not be diagnosed afterwards.

**Final implementation** (no scientific parameter changed; outputs verified identical)
- **Loading:** EDFs are opened lazily; only the used channels are picked and then loaded.
- **Connectivity:**
  - `band_analytic(..., chunk_rows=8)` filters and Hilbert-transforms 8 ROIs at a time into
    one preallocated array. ROIs are independent, so the values are identical.
  - Each band's analytic signal is released before the next band.
- **ROI copy:** the float32 ROI time series is made only when `save_roi_timeseries` is true.
- **Between subjects:** a subject's arrays are deleted after its checkpoint is saved, and
  garbage is collected (`MemoryTracker`).
- **Assembly:**
  - pass 1 reads the small per-subject CSVs to size the cohort;
  - the cohort arrays are preallocated once;
  - pass 2 copies one subject at a time, with shape checks.
- **Saving:** `atomic_savez` and `atomic_save_npy` stream into a temp file (fsync, rename),
  so the compressed archive is never held in RAM.
- **Validation:** adjacency and signed-ciPLV checks run in blocks of 256 windows. The checks
  are the same; the temporaries are bounded.
- **Monitoring (`adgraph.memory`):**
  - current RSS, peak RSS (VmHWM, reset per unit through `/proc/self/clear_refs`), and
    available and total system memory;
  - per subject: `mem_*` columns in `subject.json` and `subjects.csv`, and a log line;
  - per run: `memory_log.csv` with one row each for the operator, every subject and the
    assembly, appended across sessions.
- **After an OOM kill:**
  - completed units are untouched (marker last; payload validated by SHA-256);
  - the interrupted unit has no valid marker and is recomputed;
  - leftover temp files of killed writes are removed at the start of the next run
    (`cleanup_temp_files`; never part of a unit).

**Measured effect** (same synthetic worst case)
- Load peak: 323 → 181 MB.
- Connectivity and PSD peak: 1 492 → 772 MB, now about the continuous ROI signals plus one
  band's analytic signal, which is the unavoidable floor for continuous-signal filtering.
- Assembly peak: below 1.5× the cohort arrays (was above 2×).

**Alternatives considered**
- **complex64 analytic signals.** Rejected: halves memory but changes numerical precision.
- **Windowed (not continuous) filtering.** Rejected: changes the method (methods decision #21).
- **Memory-mapped cohort arrays.** Not needed at this size; compressed `npz` cannot be
  memory-mapped.

**Effect on interpretation**
- None. Results are identical within floating-point tolerance (tested).

**Code / config**
- `src/adgraph/memory.py`, `src/adgraph/pipeline.py` (`process_recording`, `run`,
  `assemble`, `MemoryLog`)
- `src/adgraph/connectivity/plv.py`, `src/adgraph/io/eeg.py`
- `src/adgraph/checkpoint.py` (`_atomic_stream`, `cleanup_temp_files`),
  `src/adgraph/graphs/store.py`

**Validating tests** (`tests/test_memory.py`, synthetic only)
- Chunked band-analytic and bounded `process_recording` are identical to the unbounded path.
- Lazy EDF loading reads identical samples.
- The per-recording peak stays below 4.5× the ROI signal size.
- Retained memory does not grow across subjects.
- Memory QC columns and `memory_log.csv` are written.
- Assembly peak stays below 1.5× the cohort arrays, with identical content.
- A real SIGKILL during a subject write: completed units are byte-identical after the restart,
  only the killed and pending subjects are recomputed, and leftover temp files are removed.

**Approval status**
- Author-specified audit (2026-10-13). Implementation approved in principle: no scientific
  parameter changed.

---

## 35. Progress display, final Colab verification, alpha-band filter verification

**Previous design**
- **No progress display.** Long stages only printed log lines, and `adgraph.colab.sh`
  forwarded subprocess output line by line, which breaks carriage-return progress bars.
- **Cost estimate in notebooks.** Each experiment notebook ran a runtime-estimate stage, and
  the README carried a static cost table.
- **Failures did not stop the notebook.** A failed experiment unit or failed recording was
  recorded and printed, but *Run all* still finished without an error.
- **Unverified filters.** The alpha1/alpha2 filter overlap from entry 33 was reported from
  −6 dB points only and was not documented in detail.

**Problem discovered**
- Runs take hours to days with no visible progress, and a resumed run needs to show what was
  recovered from Drive.
- The author asked that no cost-estimation stage or table be part of the workflow
  (2026-10-13).
- Incomplete work must not look like a successful run.

**Final implementation**
- **`adgraph.progress`.** One tqdm line per stage, never nested (tested: at most one active
  bar). Library code reports context with `progress.set(...)` and items with
  `progress.advance()`; both are no-ops without an active bar.
- **Graph construction:**
  - bars for input SHA-256, the source operator (cached or built: fetch fsaverage →
    forward/inverse → verification) and all recordings (`n/43`);
  - the recordings line shows the current subject, dataset and stage (loading, preprocessing,
    windowing and rejection, source reconstruction, connectivity per band with window count,
    source and sensor PSD, saving the checkpoint);
  - it also counts done / resumed / failed / pending recordings;
  - then bars for assembly (per subject) and saving the cohort.
- **Experiments:**
  - one line for the whole experiment, counting outer folds plus permutations;
  - it shows the current variant, setting, repeat and fold, inner-CV trial (or refit), GNN
    epoch and permutation, plus finished / resumed / failed / skipped / pending units.
- **Figures:** one line through the figures, with made / skipped / incomplete / error counts.
- **Checkpoint-aware counting:**
  - graphs pre-check each subject's marker identity without hashing the payload;
  - experiments pre-read `experiment_status`, now including permutations;
  - recovered items count as done from the start, are never recomputed, and are not timed;
  - a subject whose marker matched but whose payload is invalid is taken back off the count
    and recomputed.
- **Honest ETA.** tqdm's rate-based ETA is never shown.
  - Recordings: from at least two recordings processed in this session.
  - Experiments: only for the current variant and setting, from at least two of its folds
    computed in this session. Whole-experiment ETAs would mix units that differ in cost by
    orders of magnitude.
- **Colab rendering:**
  - `sh` forwards raw output chunks, so `\r` redraws update one line;
  - redraws are rate-limited (`ADGRAPH_PROGRESS_INTERVAL`, default 2 s);
  - logging goes through `tqdm.contrib.logging`, so log lines print above the bar;
  - `ADGRAPH_PROGRESS=0` disables the bars.
- **Logs:**
  - runtime (`seconds`) and peak memory per recording in `subjects.csv` and
    `memory_log.csv`;
  - per experiment unit in `_run_status.csv` (`seconds`, `mem_peak_mb`).
- **Notebooks:**
  - the cost-estimate stage and table are removed (`experiment.py estimate` remains as an
    optional command; `scripts/estimate_runtime_synthetic.py` is removed);
  - experiment notebooks display results, runtime and memory, then raise a visible error if
    any unit failed or is incompatible;
  - notebook 01 raises after its QC summary if any recording failed (`below_min_windows` is
    the pre-specified exclusion, not an error);
  - `notebooks/README.md` documents the exact Google Drive output structure.
- **Alpha-band verification.** `scripts/inspect_band_filters.py` and `docs/band_filters.md`
  measure the filters actually applied (impulse check against MNE `filter_data`). Findings:
  - the passbands match the nominal bands within about 0.15 Hz;
  - MNE's automatic transition bands (2–3.25 Hz at the alpha edges) make the effective −6 dB
    bands alpha1 ≈ 7.0–11.25 Hz and alpha2 ≈ 8.76–14.62 Hz, overlapping on 8.76–11.25 Hz;
  - for flat input, about 46 % of alpha1 and 42 % of alpha2 output power lies outside the
    nominal band;
  - alpha2 passes 67 % of the amplitude at 9 Hz and 92 % at 9.5 Hz.

  **Nothing changed** (open item 8).

**Alternatives considered**
- **Nested bars** (one per level). Rejected: multi-line bars need cursor-movement escape
  codes that Colab's text output does not render reliably, and they clutter the output.
- **In-kernel widget bars.** Rejected: they would require running the pipeline inside the
  notebook kernel instead of isolated subprocesses.

**Effect on interpretation**
- None. Display and logging only. Results and the protocol are unchanged.

**Code / config**
- `src/adgraph/progress.py`, `src/adgraph/colab.py` (`sh`)
- `src/adgraph/pipeline.py`, `src/adgraph/experiments/{plan,runner}.py`,
  `src/adgraph/models/trainer.py`, `src/adgraph/visualization/figures.py`,
  `src/adgraph/checkpoint.py` (`marker_matches`)
- `scripts/{build_graphs,experiment,run_plan,make_figures,make_notebooks,inspect_band_filters}.py`
- `notebooks/`, `docs/band_filters.md`, `requirements.txt` / `pyproject.toml` (tqdm)

**Validating tests** (`tests/test_progress.py`, synthetic only)
- ETA comes only from items processed in this session.
- `sh` forwards `\r` redraws.
- Graph progress on a fresh run (4 done) and on resume (4 resumed, none recomputed, no ETA).
- Experiment progress across a simulated disconnect: recovered folds counted as done from the
  start; the third run recomputes nothing.
- Inner-trial, fold, permutation and epoch context are shown.
- Figure progress counts.
- Never more than one active bar.
- `tests/test_notebooks.py` checks there is no cost stage and that failures end experiment
  notebooks visibly.

**Approval status**
- Author-specified (2026-10-13). Filter changes: pending author decision (open item 8).

---

## 36. Connectivity band-pass transition widths

**Previous design**
- Phase bands were filtered with MNE FIR, zero phase, `firwin`, and MNE's automatic transition
  widths: 2–3.25 Hz at the alpha edges, 7.5 Hz at 30 Hz.
- This was never chosen explicitly. It was the library default.

**Problem discovered** (entry 35; `docs/band_filters.md`)
- The passbands matched the nominal bands, but the wide transitions made the effective −6 dB
  bands overlap: alpha1 7.0–11.25 Hz, alpha2 8.76–14.62 Hz.
- On synthetic signals, coupling only at 9 Hz (inside alpha1) gave alpha2 PLV 0.67 (in-band
  0.86). Coupling only at 11.5 Hz appeared in alpha1 (0.34) and beta (0.38). Band-specific
  connectivity was therefore not band-specific.

**Scientific reason**
- Band-specific claims (single-band experiments, band comparisons, fusion weights) require
  that a band's PLV / |ciPLV| reflects phase coupling within that band.
- The author asked to keep the nominal bands and choose a more selective, defensible design
  (instruction of 2026-10-13).

**Final implementation**
- Config: `connectivity.transition_bandwidth_hz: 1.0` sets one transition width on every edge
  (lower edge capped at `l_freq`; `null` reproduces MNE 'auto').
- `connectivity.plv.band_analytic(..., transition_bandwidth_hz)` applies it;
  `pipeline.process_recording` passes the configured value.
- Filters are 3.3 s long (825 samples at 250 Hz).
- `check_filters_fit_edge_trim` runs at every graph run and refuses filters whose half-length
  reaches the 10 s edge trim.
- `CODE_VERSIONS["graph_subject"]` is now `graph-subject-2`.
- Nothing else changed: nominal bands, filtering of the continuous signal, zero-phase FIR,
  Hilbert phase, PLV/ciPLV, windows and PSD.

**Alternatives considered** (synthetic comparison, `scripts/compare_band_filter_designs.py`)
- **1.5 Hz.** Leaves residual leakage (alpha2 PLV 0.20 for 9 Hz coupling).
- **0.5 Hz.** Doubles filter length (6.6 s) and temporal smearing, with no measurable gain in
  specificity.
- **Keeping 'auto' and reporting the effective bands.** Rejected: band-specific
  interpretation would be confounded.
- **Merging or redefining bands.** Excluded by the author.

**Effect on interpretation**
- Off-band leakage falls to the noise floor: alpha2 PLV 0.11 for 9 Hz coupling, against a
  floor of 0.10.
- In-band coupling recovery is maintained or slightly improved (0.90 vs 0.86).
- The null PLV floor rises slightly (alpha1 0.084 → 0.110) because of the narrower effective
  bandwidth. This applies equally to all subjects and classes.
- The −6 dB overlap between neighbouring bands is now ±0.5 Hz around the shared edge.
- No real-data graphs existed with the previous design, and the config hash changes, so no
  result mixes designs.

**Code / config**
- `configs/pipeline.yaml`, `src/adgraph/connectivity/plv.py`, `src/adgraph/pipeline.py`
- `scripts/inspect_band_filters.py`, `scripts/compare_band_filter_designs.py`
- `docs/band_filters.md`

**Validating tests** (`tests/test_band_filters.py`, synthetic only)
- Frozen nominal bands and width.
- Passbands on the nominal edges, with −6 dB at edge ∓ 0.5 Hz and a stopband below −50 dB.
- alpha1/alpha2 overlap of 9.5–10.5 Hz or less.
- Filters fit inside the edge trim, and too-narrow transitions are refused.
- 9 Hz and 11.5 Hz coupling stays in its band, with PLV and |ciPLV| at the noise floor in the
  other band; the old design is shown to leak.
- The pipeline passes the configured width.

**Approval status**
- Author-specified (instruction to recommend and implement, 2026-10-13). Resolves open item 8.

---

## 37. EDF channel labels are electrode numbers: mapping through the `.loc` layout

**Problem (real-data header check, Notebook 01 cell 3, commit `d3498570`)**
- The strict header check failed for all 43 recordings.
- The EDF headers do not name electrodes. D1 has `EEG E01` … `EEG E32` (1000 Hz); D2 has
  `Misc ExG1` … `Misc ExG32` (1024 Hz).
- The pipeline only recognised 10–20 names, so every channel was "unrecognised" and every used
  channel "missing". No signal had been read; no graph unit existed.

**Evidence for the mapping**
- The author confirmed that both datasets were recorded with the same 32-electrode cap, whose
  layout is `AD-LEB/electrodes_positions_32.loc` (Drive file id `1fsfZUkd1ng-GnuiWtIlglQTTS78t6eva`,
  SHA-256 `da108d02…0bd4`; `docs/data_inventory.md`).
- That file was read (read-only, via the Drive connector) on 2026-10-13. Its order, index 1–32,
  is Fp1 Fpz Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 M1 T7 C3 Cz C4 T8 M2 CP5 CP1 CP2 CP6 P7 P3 Pz
  P4 P8 POz O1 Oz O2. This is exactly `channels.recorded` in `configs/pipeline.yaml`, which the
  original pipeline already used.
- The polar angles in the `.loc` agree with the `standard_1020` positions of the same labels
  (within 20°; tested), so the labels in the file and the montage positions are consistent.
- 32 numbered EDF channels and 32 layout entries: channel *k* is electrode *k*.

**Correction**
- `adgraph.io.layout`: `parse_loc` (validated: 4 fields per line, indices 1..N, unique labels,
  dot padding and CRLF handled); `verify_layout` (order must equal `channels.recorded`);
  `find_layout_file` (root first, then `EEGs/`; a lone `… (1).loc` copy is accepted, different
  copies are refused); `check_layout` (path + SHA-256 for logs).
- `channel_mapping` accepts two conventions, never mixed in one recording: 10–20 names (as
  before) or index-only labels (`EEG E01`, `E1`, `Misc ExG1`, `ExG 01`, zero padding, trailing
  dots). It refuses mixed inventories and repeated channel numbers; numbers beyond 32 and other
  channels (e.g. `Status`) are reported as unrecognised and not used.
- The same function serves the header check (`inspect_headers`, `scripts/inspect_raw.py`) and
  loading (`load_raw`), so the gate's verdict and the loaded channels cannot disagree (tested).
- `load_raw` sets every kept channel to type `eeg` (D2's `Misc` channels are read as misc),
  drops M1/M2, orders the 30 channels canonically and applies the `standard_1020` montage.
  Everything downstream (preprocessing, 250 Hz resampling, operator, sensor PSD, source
  reconstruction, connectivity, graphs) sees the same 30 named channels as before.
- The strict header check first verifies the `.loc` file (printing its path, SHA-256 and the
  index → label order) and fails if it is missing or differs. It also fails on an ambiguous
  inventory or a missing used channel. `pipeline.run` verifies the layout whenever paths.yaml
  names one. The layout file was already hashed into the assembly identity and manifest.
- `colab.write_paths_yaml` locates the layout with `find_layout_file`.

**Data-driven check of the mapping (QC only)**
- `qc_spatial_rho` in `subjects.csv`: the Spearman correlation, per recording after
  preprocessing, between inter-electrode distance and |signal correlation|. Volume conduction
  makes it clearly negative when channels sit where the montage says. A shifted or shuffled
  mapping gives a value near 0 (tested on synthetic data). Values above −0.1 are logged as a
  warning and listed in Notebook 01's quality summary. Nothing is changed or excluded on it.

**Not changed**
- Bands, filters, reference, rejection, windows, inverse, parcellation, the 30-channel set,
  the montage and the unit scaling (D1 ×1e-6, D2 ×1).
- The EDF and `.loc` files (only read).

**Effect on cached results**
- The commit changes `code_identity`. Graph subjects and the operator cache from earlier
  commits are therefore incompatible and are recomputed. None existed: the check failed before
  any processing.

---

## 38. Amplitude check failure in the first real-data smoke test; empty runs fail explicitly

**What happened (Notebook 01, smoke test, commit `ba4d574`)**
- The layout check, the input SHA-256 checks and the fsaverage operator succeeded (relative
  inverse verification error 1.3e-15).
- Both smoke-test recordings stopped at the preprocessing amplitude check
  (`preprocessing/clean.py`):
  - D1-1: median channel std 0.000221 V (221 µV);
  - D2-2: 0.000322 V (322 µV);
  - plausible range `units.plausible_std_v` = [1e-7, 2e-4] V (0.1–200 µV).
- The run reported `done: 0, failed: 2` but still assembled an **empty cohort** and wrote a
  completion marker (`assembly.done.json`).

**What the check measures** (traced in code)
- It is the median over the 30 channels of each channel's standard deviation.
- The signal is taken after resampling to 250 Hz and the 0.1–45 Hz band-pass.
- It is measured in the **recording reference**, i.e. before bad-channel handling and before
  the average reference.
- It covers the **whole recording**, including the first and last 10 s that windowing later
  discards.
- It is a plain standard deviation, so a few large transients can dominate it.
- Purpose (decision 20): catch unit errors in either dataset, which shift amplitudes by 10³
  or 10⁶.

**Unit chain** (traced in code)
- MNE reads the EDF physical values and converts them with the header's physical dimension:
  `V` × 1, `mV` × 1e-3, `uV`/`µV` × 1e-6.
- `load_raw` then multiplies by `units.scale`: D1 × 1e-6 (header `V`, values µV), D2 × 1.
- The reported values exclude the gross errors:
  - if D1 were rescaled twice, its value would be 2.2e-10 V, not 2.2e-4 V;
  - if it were not rescaled at all, its value would be 221 V;
  - an mV/µV mix-up would give 0.22 V or 0.22 µV.
- Both datasets give a similar 200–300 µV through different chains (× 1e-6 vs × 1). A shared
  scale error would be a coincidence.
- **Hypothesis to test, not a conclusion.** 200–300 µV is plausible for a 0.1 Hz high-passed
  signal that still carries:
  - electrode drift / skin potentials;
  - a common-mode or reference signal (removed later by the average reference);
  - start-up transients.

  Typical resting EEG in the bands that are analysed (1–45 Hz, average reference) is a few
  to tens of µV.
- Genuinely noisy EEG remains possible and is distinguished by the diagnostic below.

**Diagnostic (read-only; no pipeline change)**
- `adgraph.preprocessing.amplitude.amplitude_report`, `scripts/diagnose_amplitude.py` and
  Notebook 01 section 3b.
- Run on the two smoke-test recordings, or any `--keys`, it reports:
  - the EDF header per channel: physical dimension, MNE's conversion factor, physical and
    digital limits, resolution, prefiltering;
  - per-channel and median statistics at six stages: MNE volts, after the dataset scale,
    the QC stage, + 1 Hz high-pass, + average reference, both. The statistics are std,
    robust std (1.4826 MAD), percentiles of |x − median|, peak-to-peak, DC offset and
    linear trend;
  - the QC value without the 10 s edges;
  - band fractions of the 0.1–45 Hz power (0.1–1 Hz drift share);
  - the common-mode variance fraction;
  - 20 s block amplitudes over time;
  - clipping at the EDF physical limits, flat samples, 50/60 Hz line noise, and the alpha
    peak and its spectral density.
- The QC-stage value is computed with the same calls as `preprocess()`. `preprocess` now uses
  a shared `resample_and_bandpass` helper, with identical numerics, and the two values agree
  exactly in a test.
- It prints evidence statements with fixed criteria:
  - **units**: the 1–45 Hz average-referenced std inside 2–100 µV;
  - **drift**: ≥ 50 % of the power below 1 Hz;
  - **common mode**: ≥ 50 % of the variance shared by all channels;
  - **transients**: std / robust std > 1.5;
  - **clipping**: > 0.1 % of samples at the limits;
  - **edges**: whole-recording vs edge-trimmed value.
- These criteria are evidence for the author, not an automatic decision.

**Decision status**
- The threshold, the check's stage and the scaling are **unchanged**.
- No error is suppressed: the smoke test keeps failing until the cause is decided.
- If the diagnostic shows correct units, with the excess due to drift, common mode or edges, a
  candidate revision for the author would be:
  - apply the check to the signal that actually enters the features: edge-trimmed, 1–45 Hz,
    average reference, robust std;
  - keep a range that still separates the 10³/10⁶ unit errors.

  This is a methods decision. It needs the diagnostic's numbers and the author's approval,
  and it would be logged as a new decision row.
- **Open item 11.**

**Empty / incomplete runs** (implemented)
- `pipeline.run` checks the subject outcomes before assembly (`check_run_complete`). If any
  subject failed, or no subject produced graphs (e.g. all below the 5-window rule), it:
  - writes `last_attempt.json` (failed subjects, errors, statuses);
  - keeps `subjects/<key>/failed.json`, `status.csv` and `memory_log.csv`;
  - moves a previously assembled cohort in that directory to `_superseded/`;
  - raises `GraphRunIncomplete`. `build_graphs.py` exits with `FAILED: …`, so the notebook stops.
- `assemble` itself refuses failed subjects or an empty cohort. A completion marker therefore
  always means a complete, non-empty cohort.
- This replaces the earlier behaviour, which assembled the successful subjects with failed
  ones listed. A full run with a failed subject is no longer assembled. Fix the cause and
  rerun; only the failed subjects are recomputed.
- Experiments refuse unusable graph runs (`graphs.store.require_complete_cohort`, called by
  `runner.run_variant` and by `pipeline.latest_complete_graph_run`). A run is refused if it
  has:
  - no graphs;
  - a failed subject, in `subjects.csv` or in the manifest units;
  - a dataset without graphs;
  - been a `--limit` smoke test.

  `latest_complete_graph_run` does not fall back to an older run, so an empty cohort
  assembled by earlier code is refused too.

**Effect on checkpoints**
- The new commit changes `code_identity`. The cached operator is therefore rebuilt once (as
  after every commit), and graph subject units are recomputed.
- The empty smoke-test directory `graphs_v3_<hash>_limit1/` from commit `ba4d574` contains
  an empty `graphs.npz`, `manifest.json` and `assembly.done.json`. It is never selected:
  - smoke-test directories are excluded from `latest_complete_graph_run`;
  - `require_complete_cohort` refuses it.
- On the next smoke-test attempt with the new code, that directory is reused because its
  config hash is unchanged:
  - if the attempt fails, the empty assembly is moved to `_superseded/`;
  - if it succeeds, the assembly is rebuilt and the old files are superseded.

  Nothing is deleted.

---

## 39. Amplitude QC on the analysed interval, unit safeguard, bad channels on the analysed interval

**Evidence (author's run of `scripts/diagnose_amplitude.py`, commit `128ed95`, real data)**

| Measurement | D1-1 | D2-2 |
|---|---:|---:|
| Former check value (whole recording, recording reference) | 221.1 µV | 322.5 µV |
| Same without the first/last 10 s | 17.5 µV | 20.2 µV |
| Robust std at that stage | 11.7 µV | 15.8 µV |
| Power below 1 Hz | 95 % | 92 % |
| Largest 20 s block std | 1030 µV (start) | 825.8 µV (end) |

- Units are correct: the robust and edge-trimmed values are ordinary EEG amplitudes in both
  datasets.
- The failures came from a very large transient at the start of D1-1 and at the end of D2-2,
  inside the 10 s that windowing discards.
- The former statistic (plain std over the whole recording) was dominated by those edges, so
  the check measured data that is never analysed.

**Correction (author-approved 2026-10-13)** — `preprocessing/clean.py` `preprocess`, in order:
1. Resample to 250 Hz and band-pass 0.1–45 Hz: **unchanged**. The analysed signal stays
   0.1–45 Hz.
2. **Unit safeguard** (new, `units.gross_robust_std_v` = [1e-7, 1e-3] V):
   - statistic: median over channels of the robust std (1.4826 × MAD) of the **whole**
     0.1–45 Hz recording;
   - MAD is barely moved by transients that occupy a small fraction of the samples;
   - the range is one decade beyond physiological scalp EEG on each side. A 10³ or 10⁶ scale
     error of typical EEG (10 µV → 10 mV, 10 V, 10 nV or 10 pV) falls outside, about two decades
     from either bound;
   - it is evaluated first, before bad-channel detection, whose flat-channel rule uses an
     absolute threshold;
   - it does not depend on the edge trim or the reference.
3. **Bad-channel detection on the analysed interval**:
   - the 1 Hz high-pass is applied to the continuous signal first, so its 1.65 s half-length
     stays inside the trim; the samples are then cut to the edge-trimmed interval;
   - the rules and thresholds are unchanged: flat, robust z of log std > 5, 90th-percentile
     neighbour |correlation| < 0.4; spherical-spline interpolation;
   - reason: in a synthetic test, a transient confined to the discarded first 3 s flagged a
     channel. Its interpolation changed PLV by up to 0.19 and log10 PSD by up to 0.36 in
     **every** retained window, so data that is never analysed was deciding the
     interpolation;
   - the author chose this option. What whole-recording detection would have flagged is
     still recorded (`bad_channels_whole_recording`, QC only).
4. Average reference: **unchanged**.
5. **Amplitude plausibility** (`units.plausible_std_v`, range **unchanged**, 0.1–200 µV):
   - statistic: median over channels of the std of the **analysed interval of the final
     signal**, i.e. 0.1–45 Hz, interpolated, average reference: the signal the windows are cut
     from;
   - genuinely noisy recordings still fail (tested with ~500 µV over the analysed interval);
   - a recording where interpolation leaves no signal (e.g. every channel flagged) now fails
     here instead of passing silently.

**Window rejection: unchanged and fully active**
- Adaptive peak-to-peak threshold (median + 5 MAD per subject); reject a window when more than
  15 % of its channels exceed it.
- Every candidate window is now saved in `subjects/<key>/qc_windows.csv`: kept/rejected,
  channels over threshold, max/median ptp, threshold, decision, and whether it lies within the
  band-pass reach of the discarded edges.

**Edge leakage through filtering (item 6 of the author's request)**
- The 0.1–45 Hz FIR (MNE firwin, `l_freq` 0.1 Hz with a 0.1 Hz transition) is **33 s long,
  half-length 16.5 s**. This exceeds the 10 s trim, so transients in the discarded edges leak up
  to 16.5 s into the analysed interval: in the first and last two windows.
- The leak is the high-pass complement, i.e. content **below ~0.1 Hz**. Synthetic tests:
  - **sensor signal**: unchanged in the middle of the recording (< 1e-9 V);
  - **edge windows**: the leak adds slow offsets. The unchanged adaptive rejection removed the
    two edge windows that the leak made abnormal (3× the ptp of the others);
  - **kept windows**: ptp ratio vs the middle windows < 1.5 (`qc_edge_window_ptp_ratio` in
    `subjects.csv`; warning above 2);
  - **PSD features** (source, sensor, absolute, relative) of retained windows: unchanged
    (< 1e-4).
- **PLV/|ciPLV|** of retained windows change by ≤ 0.0025 for 3 mV transients (about 300× the
  EEG). The change decays with distance from the edges (about 1.5e-4 in the middle).
  - Cause: the existing connectivity design. The analytic signal is computed over the whole
    continuous band-passed ROI signal (entry 34), and the Hilbert kernel decays as 1/t, so an
    edge transient slightly shifts phases everywhere.
  - The sensor data and the PSD do not change.
  - Not changed here (PLV/ciPLV definitions are frozen). Open item 12.
- The amplitude plausibility value includes the leaked slow offsets of the edge windows, so it
  rises when edge transients are present. That is measured, not hidden.

**Line noise (item 7)**
- The diagnostic's former `line50_ratio` was computed on the **unfiltered** recording at its
  original rate, so it was meaningful but did not say so.
- The 0.1–45 Hz FIR attenuates 50 Hz by only **−4.5 dB** (−6 dB at 50.6 Hz; 60 Hz −58 dB), so
  50 Hz mains largely remains in the signal seen by the amplitude checks, bad-channel rules and
  window rejection.
- It is outside every feature (bands ≤ 30 Hz with 1 Hz transitions; relative-power denominator
  [1, 45) Hz).
- The diagnostic now reports 50/60 Hz RMS before and after the band-pass, plus the filter's gain
  (`line_noise`, `LINE NOISE` evidence line). No pipeline change.

**QC outputs added (no change to features)**
- `subjects.csv`: `gross_robust_std_uv`, `median_std_uv` (now the analysed-interval,
  average-referenced plausibility statistic), `median_robust_std_uv`, `whole_recording_std_uv`
  (former statistic, QC only), `bad_channels_whole_recording`, `n_windows_rejected`,
  `qc_edge_window_ptp_ratio`.
- `subjects/<key>/qc_channels.csv`, per channel:
  - std and robust std on the analysed interval, in the recording reference;
  - whole-recording std;
  - std after the 1 Hz high-pass and its robust z;
  - neighbour correlation;
  - bad reason, and the bad reason whole-recording detection would give;
  - interpolated flag;
  - final std and robust std.
- `scripts/inspect_graph_subjects.py` and Notebook 01 step 4b give a read-only per-subject
  inspection: QC tables, a window timeline, ROI and sensor band power, mean PLV/|ciPLV| per band.
- The cross-dataset gate's amplitude-overlap check (DESIGN.md §7) now compares the new
  `median_std_uv`.

**Effect on checkpoints**
- `configs/pipeline.yaml` gains `units.gross_robust_std_v`, so the config hash and the run
  directory change.
- Earlier smoke-test directories stay untouched and are never selected (smoke tests are
  excluded, and empty cohorts are refused).
- The new commit changes `code_identity`, so the operator is rebuilt once.
- Unchanged and reused: fsaverage (`cache/subjects`) and the input SHA-256 cache.

---

## 40. Pre-specified dataset-wide QC audit (bad channels, connectivity boundary sensitivity)

**Context (2026-10-14).** The real-data edge-sensitivity check at `7fd5a69` reproduced the stored
smoke-test graphs exactly. It found:
- D1-1: A vs B PLV max 0.027, max subject-mean difference 0.0005;
- D2-2: A vs B PLV max 0.215, max subject-mean difference 0.020, theta most affected, five
  interpolated channels (F7, FC1, T7, T8, CP5).

**Decision.** Before any pipeline change, run a read-only audit of all 43 recordings:
- Part 1, bad channels;
- Part 2, production A vs reflected-edge B with identical accepted windows.

The plan, metrics and decision criteria are in `docs/qc_boundary_audit.md` and were written and
committed **before** any dataset-wide result.

**Key choices:**
- Subjects are the units.
- B is not ground truth: synthetic ground truth showed that B's own reflection dominates the
  boundary-touching windows, while interior differences track the true dependence on non-analysed data.
- The primary boundary criterion is relative to the between-subject spread of each connection.
- Interpolation and boundary sensitivity are assessed separately.
- No decision may use classification performance.

**Implementation.** `scripts/audit_quality_boundary.py`:
- read-only against the production code at `7fd5a69`, which is checked at start-up;
- resumable per recording;
- writes only to a separate Drive folder.

The production modules, the graphs, the cached operators and the experiments are untouched.

**Revision 1 (same day, still before any dataset-wide result, author's request).**
- Significance only flags a comparison for investigation; every group comparison reports p, effect
  size (rank-biserial, difference), 95 % bootstrap CI and practical magnitude.
- The normalised boundary metric uses the between-subject SD within each dataset, with a floor of
  0.005, and is always reported next to the absolute differences.
- 0.10/0.25 are provisional QC categories.
- The ≥ 30 % bad-channel criterion triggers only a *test* of a reduced common montage, with identical
  channels in both datasets and a recomputed operator.
- All per-recording and per-band metrics are saved.
- Audit script and plan only; production code unchanged.

**Approval status.** Audit requested by the author. Decisions on options A–E follow the results.

---

## 41. Boundary-Aware Filtering and Hilbert Phase Estimation for Source-Space EEG Functional Connectivity

Version 1.0, 2026-10-14. The full decision record (original design, audit evidence, new algorithm,
unchanged items, effect on the dataset, reproducibility, limitations, Methods and Supplementary text,
before/after table) is `docs/boundary_processing.md`.

**Decision (author, after the completed 43-recording audit of entry 40).**
- Correct the boundary processing and rebuild all graphs as pipeline v4, in a new directory
  `graphs_v4_<config hash>/`.
- The 30-channel montage and subject-specific interpolation are kept. No reduced-montage experiment.

**Original (v3).** The 10 s at each end were excluded from the windows but processed with everything
else:
- FFT resampling and the 33 s (8251-tap) 0.1–45 Hz FIR ran on the whole recording;
- the band-pass ran on the whole ROI signal;
- the analytic signal came from an FFT Hilbert transform of the whole recording, which is non-local
  and assumes the recording is periodic.

**Evidence.**
- The largest maximum subject-mean PLV differences between production and reflected-edge handling
  were 0.177 (D1-8), 0.164 (D2-12) and 0.117 (D1-14).
- Theta was the most sensitive band in 29 of 43 recordings.
- Sensitivity was associated with band-specific edge severity (ρ = 0.755) but not with the number of
  interpolated channels (ρ = 0.248, p = 0.109).
- Boundary windows were the most sensitive (median p95 0.0218 at distance 0 vs 0.00085 at ≥ 6 steps).
- D2 was more sensitive than D1 (p = 0.0066), and edge characteristics differed: D1 transients at the
  start, D2 at the end.
- These are sensitivities to boundary handling, not errors relative to a ground truth.

**New (v4).**
1. Discard 10 s per side at the original rate before resampling and filtering.
2. Explicit `reflect_limited` padding for resampling and every FIR.
3. A finite analytic kernel g = h + i𝓗{h} (825 taps; Hilbert-kernel energy outside the support
   ≤ 3.8e-7; negative-frequency leakage ≤ −76 dB), applied by linear convolution.
4. A margin of 4125 + 412 = 4537 samples (18.148 s) from the cropped ends (`margin_policy:
   full_support`). Accepted windows then depend on no padded sample through any FIR.
5. The 20 s / 10 s window grid starts at the first valid sample.

Identical for D1 and D2.

**Verified by implementation tests** (`tests/test_boundary_processing.py`, synthetic):
- outputs are bit-identical under any change confined to the discarded edges, at 1000 and 1024 Hz;
- interior agreement with the v3 analytic signal: PLV/|ciPLV| within 1e-4;
- the analytic FIR is local, while the FFT Hilbert transform is not;
- margins and window grid;
- end-to-end run with the validation script.

**Unchanged.**
- Recordings, labels, montage and interpolation.
- 250 Hz resampling, the 0.1–45 Hz FIR and the average reference.
- Source model, operator, ROIs and bands.
- Band-pass kernels, PLV/ciPLV definitions and PSD features.
- 20 s / 50 % windows, rejection and the minimum-window rule.

**Effect.**
- 3–4 fewer candidate windows per recording, from the margin.
- Completed rebuild (`graphs_v4_33faa4b89a5d`, `graph_run_id` `6d75de9975f0d9ae`, commit `e637368`,
  23/23 integrity checks passed): 43/43 recordings, no failures or exclusions; 1,706 candidate
  windows, 179 rejected (10.5 %), **1,527 graphs** (D1 907, D2 620; AD 729, HC 798), 6,108
  band-specific adjacency matrices per connectivity measure. v3 had 1,640 graphs (−113, −6.9 %).
- The source operator (`c9a4fbf869a03321`) was verified against the full inverse (relative error
  1.27e-15). The cached-operator comparison found no operator with the same source-model identity,
  so equality with the `7fd5a69` operator was not tested directly.
- v3 checkpoints are never reused: the configuration hash, `CODE_VERSIONS["graph_subject"]` and the code
  identity all change.

**Approval status.**
- The correction and rebuild were approved by the author.
- The four implementation choices (margin policy `full_support`, window grid from the first valid
  sample, QC on the cropped signal, operator rebuild) were approved before launch
  (`docs/boundary_processing.md` §9).
- The v4 graph dataset is **frozen**; Exp 0–6 and all later analyses use it.

---

## 42. Fixed-hyperparameter GNN protocol and Tier 1 schedule (protocol amendment for computational feasibility)

Version 1.0, 2026-10-14. Author decision. DESIGN.md v1.5, decisions row 50.

**Situation when the decision was made.**
- Exp 0 and Exp 1 were complete (results committed in `results/tables/`).
- Exp 2 had run under the nested protocol for ≈ 1 h 40 min: 1 of 120 units (PLV / theta / GCN / within-D1,
  subject-level AUC 0.522) and 102 of 7,248 outer folds. **This result had been observed before the
  amendment.**
- Measured cost: ≈ 59 s per outer fold, i.e. 17 GNN fits (4-point grid × 4 inner folds + refit, 60 epochs
  each). Extrapolated: ≈ 180–290 GPU hours for Exp 2 alone and ≈ 900–1,100 h for Exp 2–6.

**Decision.**
1. **No GNN hyperparameter search of any kind** (no grid, random or Bayesian search, no inner CV, no
   epoch selection). One fixed configuration per architecture, `configs/experiments/gnn_fixed_base.yaml`
   (protocol `gnn_fixed_hparams_v1`, recorded in every variant config, manifest and checkpoint identity):

   | Item | Value |
   |---|---|
   | Graph-convolution layers | 2 (GCN: symmetric-normalized weighted adjacency with self-loops) |
   | Hidden dimension | 32 (EdgeWeightedGAT: 4 heads × 8, edge terms in attention and messages; WeightedGraphSAGE: sum aggregation) |
   | Dropout | 0.3 |
   | Readout | mean + max over ROIs; MLP head with 32 hidden units |
   | Multi-band | shared encoder; fusion as specified by the experiment (mean / attention) |
   | Optimizer | Adam (PyTorch defaults β = (0.9, 0.999), ε = 1e-8), learning rate 1e-3, weight decay (L2) 1e-4 |
   | Batch size / epochs | 32 / 60, no early stopping; the model after the last epoch is evaluated |
   | Loss | binary cross-entropy with subject- and class-balanced window weights (unchanged) |
   | Node-feature scaling | z-score per (band graph, ROI, feature), fitted on the training windows of the fold |
   | Seeds | outer splits: stratified subject folds with seed 0 + r for repeat r (identical to every other experiment); model seed 0 + 1000·r + fold (Python, NumPy, PyTorch, CUDA) |

   **Source of the values:** they are the defaults of `gnn_base.yaml`, committed on 2026-10-12 before any
   GNN result existed (hidden 32 and weight decay 1e-4 were also the base values around which the grid
   was defined). Adam with learning rate 1e-3 is the optimizer's conventional default; two layers, hidden
   width 32 and moderate dropout are conventional for small dense graphs. No value was chosen from
   observed performance; the inner-CV selections of the observed nested run were not inspected.
2. **Execution:** each outer training fold (subject-level; unchanged splits) is fitted exactly once and the
   model predicts that fold's held-out test subjects once. Every fitted transform (node-feature scaler,
   covariate scaler) is fitted on the training subjects of the fold only. Enforced: a config under this
   protocol with a grid fails to load (`experiments.config.validate`); tested in
   `tests/test_fixed_hparams.py` (one fit per outer fold, training subjects only, no inner units, same
   splits as other experiments).
3. **Tier 1 schedule (10 repeats, seeds 0–9):**
   - Exp 2 and Exp 3: GCN, 4 bands × {PLV, |ciPLV|}, settings S1, S2, S4 (8 variants each);
   - Exp 4: both measures, all settings;
   - Exp 5: shared GCN encoder × {mean, attention} fusion × both measures, all settings;
   - Exp 6: full model with both measures, all settings; the seven ablations (no PSD, no strength, no
     learned fusion, each band dropped) with PLV in S1, S2, S4.
   Configs `exp{2,3,4,5,6}_*_fixedhp.yaml`. A named variant may restrict its settings (`settings` key,
   honoured by `plan_units` and `run_experiment.py`).
4. **Tier 2 (optional, secondary, never delaying Tier 1):** EdgeWeightedGAT and WeightedGraphSAGE in
   Exp 2/3, band-specific encoders and concat / gate fusion in Exp 5, |ciPLV| Exp 6 ablations
   (`*_fixedhp_tier2.yaml`), Exp 3b and EEG + age GNN analyses (`*_fixedhp*.yaml`), same fixed values.
5. **Permutation tests for GNNs are deferred.** No permutation p-value is reported unless a valid test is
   completed. Subject-level ROC AUC, balanced accuracy, F1, sensitivity, specificity, MCC, bootstrap CIs
   and per-subject predictions are kept.

**Selection rules.** The GCN-only Tier 1 follows from the design (Exp 4–6 are GCN-only, so the single-band
vs multi-band comparison is architecture-matched); the Exp 5 fusions mirror Exp 6 full / no learned
fusion; the ablation settings follow the primary settings (S1, S2) plus the cheap S4. None of these uses
the observed Exp 2 result.

**Separation of results.**
- The nested-CV Exp 2 checkpoint stays on Drive (`experiments/exp2_conn_minimal/`) as a historical
  record. It is not reported as a result, not pooled with fixed-protocol results and not shown in figures
  (`configs/figures/paper.yaml` reads only `*_fixedhp` experiments).
- Fixed-protocol results live in `experiments/<name>_fixedhp/`; their identity includes the protocol, the
  new code commit and the graph run `6d75de9975f0d9ae`.
- Exp 0 and Exp 1 (commit `e637368`, 20 repeats, nested inner CV over C for LR/SVM) are kept as they are.
  Paired comparisons with GNNs use repeats 0–9, whose subject splits are identical.

**Limitations to state in the paper.**
- The amendment was made after one nested-CV GNN result and the classical results had been observed.
- Fixed hyperparameters may be suboptimal for some experiments; GNN performance is reported as that of a
  pre-specified configuration, not of a tuned model.
- 10 repeats instead of 20 (bootstrap CIs over subjects are affected little).
- No permutation p-values for GNNs until tests are run.

**Approval status.** Approved by the author on 2026-10-14 (fixed hyperparameters, Tier 1 / Tier 2
schedule, deferred permutation tests).

---

## 43. Experiment 7: EARM-GNN (exploratory extension after the Tier 1 GCN results)

Version 1.0, 2026-10-14. Author request. Full specification: `docs/earm_gnn.md`.

**Situation.** Tier 1 results observed before this decision: Exp 2 (constant node features) near AUC 0.5
in D1 and D2; Exp 3 alpha2 within-D2 AUC 0.857 (PLV) / 0.865 (|ciPLV|) but within-D1 0.319 / 0.303;
Exp 4 D1 0.440 / 0.424, D2 0.855 / 0.864, pooled 0.663 / 0.656, cross-dataset 0.30–0.48. The code
inspection of the GCN (weighted, fixed normalization D̃^{-1/2}(A + I)D̃^{-1/2} on dense graphs) indicated
strong smoothing and no learned edge relevance.

**Decision.** Add Experiment 7, EARM-GNN, as an exploratory extension with the same protocol as the other
GNNs (fixed hyperparameters, no search, one fit per outer fold, 10 repeats with the shared splits, all five
settings, PLV and |ciPLV| separate):
- A: EEG-guided adaptive adjacency A ⊙ G, G = σ(b_k + U diag(s_k) Uᵀ) (rank 8, symmetric), propagation
  normalized by the ungated EEG degree (bounded operator), gate penalty λ = 0.01 · mean gate;
- B: three residual blocks with separate self / neighbour transforms, LayerNorm, GELU, dropout, and
  jumping knowledge including the node-wise input embedding;
- C/D: per band [log PSD, relative PSD, node strength], four separate band graphs, shared encoder with a
  band embedding;
- E: softmax attention fusion over bands (weights saved per window);
- F: attention + mean + max pooling, small MLP head. 18,214 parameters (full).
- Pre-specified ablations: fixed adjacency (no gates) and mean fusion. No other ablations.
- Short read-only D1 / D2 diagnostic (QC, interpolated channels, feature shift and separability,
  prediction behaviour); no exclusion of D1, no pipeline change.
- Engineering: a model can add a training penalty (`regularization()`); `evaluation.save_model_state`
  saves fitted parameters, gates and the training history per outer fold (off for every earlier
  experiment, whose behaviour is unchanged); Exp 7 has its own code pin so it can run while Exp 2–6
  continue on the global pin.

**Reporting rules.** EARM-GNN is reported as an exploratory model developed after observing the GCN's
limitations; no novelty or superiority is claimed without evidence; comparisons with earlier experiments
are descriptive (paired by shared CV repeats); both datasets are reported; within-D2 results are not
independent validation.

**Approval status.** Requested by the author on 2026-10-14; fixed EARM hyperparameters set before any Exp 7
result.

---

## 44. Dataset 3: OpenNeuro ds004504 (AD vs HC), provider-preprocessed, source space

Version 1.0, 2026-10-14. Author decisions. Full record: `docs/dataset_d3.md`.

**Decisions (author, 2026-10-14).**
1. Add OpenNeuro ds004504 (release 1.0.9) as **Dataset 3**: the 36 AD and 29 HC participants, from the
   provider's preprocessed derivatives. The 23 FTD participants are excluded by design and are never
   downloaded.
2. **Role:** third cohort with its own within-dataset evaluation (`within_D3`), reported next to D1/D2.
   Cross-dataset tests with D1/D2 come later; they need a combined store and a revised §7 gate.
3. **Preprocessing:** minimal, nothing repeated.
   - The provider's 0.5–45 Hz Butterworth, A1–A2 reference, ASR and ICA are used as given.
   - Added: the 10 s edge crop, resampling to 250 Hz, common average reference, the same windows,
     rejection, source model, ROIs, bands, connectivity and PSD as D1/D2.
   - No second band-pass (`preprocessing.bandpass: false`).
   - Bad-channel rules are evaluated and reported but never acted on (`interpolate_bads: false`).
   - Boundary margin = analytic-kernel half-length only (412 samples).
4. **Representation:** source space as for D1/D2 (fsaverage, 68 DK ROIs). The operator is rebuilt for
   the 19-channel montage.
5. **First experiments:**
   - Exp 0 and Exp 1 (LR, SVM);
   - age-only baseline;
   - Exp 6 full GCN (PLV, |ciPLV|);
   - Exp 7 EARM-GNN with two ablations.

   All use the existing protocols: classical 20 repeats with nested C; GNNs fixed hyperparameters,
   10 repeats.

**Engineering.**
- Pipeline configs may declare:
  - `datasets` (default D1, D2);
  - `run_prefix` (`graphsD3`: D3 runs are never selected for D1/D2);
  - `input.format` (`eeglab`);
  - `channels.verify_layout`;
  - `preprocessing.bandpass`.
- Experiment configs may declare `graph_family: D3`. Such configs read `graphsD3_v*` runs and write
  `experiments_D3/`, so every results root holds one graph run.
- D3 notebooks share their own code pin (`D3_CODE_COMMIT.txt`).
- The D1/D2 pipeline config, graph run, experiments and notebooks are unchanged.
- Download: resumable, verified by size and MD5 against the OpenNeuro listing. The release DOI, the
  CHANGES line and a SHA-256 per file are recorded in `download_manifest.json`.

**Approval status.** Approved by the author (role, preprocessing, representation, first experiments)
on 2026-10-14.

**Addendum (2026-10-15, author approval; montage inspection before the first D3 build).**
- **Flat channels:** with bad-channel interpolation off (D3), a confirmed flat channel raises
  `FlatChannelError` naming the subject and channel, before referencing. The criterion is the existing
  `flat_std_v`: std < 0.1 µV over the analysed, 1 Hz high-passed interval, two orders of magnitude below
  physiological EEG. Nothing is interpolated, excluded or rebuilt automatically. Other flagged channels stay
  report-only. D1/D2, where flat channels are interpolated, is unchanged.
- **Inspection script:** `inspect_graph_subjects.py` takes the channel and ROI counts from the subject's arrays
  (no fixed 30).
- **Real-data checks:** `scripts/d3_verify_smoke.py` gates the full D3 build (`CONFIRM_D3_FULL_BUILD`).

---

## 45. Dataset 3: boundary-aware processing of EEGLAB discontinuities; separate D3 code pin

**Date:** 2026-10-15. **Requested by:** the author, after the real D3 input check (56 of 65 recordings contain
EEGLAB `boundary` events; D3-56 has 56). **Code:** `src/adgraph/preprocessing/discontinuities.py`,
`pipeline.process_subject` / `process_recording`, `preprocessing/clean.preprocess(segments=...)`,
`configs/pipeline_d3.yaml` (`discontinuities: {source: eeglab_boundary, handling: split}`). **Tests:**
`tests/test_d3_discontinuities.py`.

**Problem.** The provider's ASR cleaning removed data segments and EEGLAB marks each junction with a `boundary`
event. The file then contains samples that are adjacent in the array but not in time. A 20 s window across such
a junction mixes non-contiguous data, and resampling or filtering across it spreads the jump into its
neighbourhood. Entry 44 had deferred the decision until the counts were known.

**How MNE represents the events** (MNE 1.13.2 `_read_annotations_eeglab`, confirmed on an EEGLAB `.set` written
in the test):
- onset = (latency − 1) / srate, in seconds from the first sample; a boundary has latency k + 0.5, so it lies
  between two samples;
- duration = EEGLAB `event.duration` / srate, which is the number of samples REMOVED there. Those samples are
  not in the file, so the duration does not affect the file's time line; it is 0 or NaN for many events;
- events with a NaN latency are dropped by MNE; the description stays `boundary`, and MNE does not treat it as a
  bad segment.

**Rule.**
1. A boundary at fractional sample position p = onset·srate cuts the recording before sample ⌈p⌉. Events at the
   very start or end make no cut; the count still reports them.
2. The edge crop (10 s per side) is applied first, as before; cuts inside the analysed interval split it into
   continuous segments.
3. Each segment is sliced by sample index and trimmed at its end to the largest length that resamples to a whole
   number of samples (500 → 250 Hz: even; at most 1 sample). It is resampled on its own with a fixed 4 s
   (2,000-sample) mirrored padding, and band-passed only if the configuration band-passes (D3 does not).
   - Reason for the trim: MNE's FFT resampling of an odd length spreads the output over the exact duration,
     a sub-sample time shift.
   - Reason for the fixed padding: MNE's `npad='auto'` changes the FFT length with the segment length and
     shifted some segments by one input sample.
   - Measured with band-limited test signals (3–44 Hz): beyond the margin, the relative error of the resampled
     segment is ≤ 2.4·10⁻⁴.
4. The 1 Hz QC high-pass (bad-channel and plausibility checks) is applied per segment. The average reference and
   the 68 × 19 ROI operator are instantaneous and act on the concatenated segments. The analytic FIR (half-length
   412) also acts on the concatenation; this is exact for every accepted window because its support never leaves
   the window's own segment (rule 5).
5. Windows: the same 20 s / 10 s grid, started at each segment's first valid sample, at least the margin
   (412 samples) from both segment ends. A segment shorter than 5,000 + 2 × 412 = 5,824 samples (23.3 s) holds
   no window and is not processed.
6. Window rejection, the ≥ 5-window rule, source model, ROIs, bands, PLV, |ciPLV| and PSD are unchanged. No
   recording is excluded for having boundaries. A recording whose segments are all too short gets status
   `below_min_windows` through the existing rule.
7. Stored `start_sample` / `stop_sample` stay on the original 250 Hz time line. New columns: `segment_index`,
   `segment_start_sample`, `segment_stop_sample`.

**Report** (per participant, in `subjects.csv`):
- `n_boundary_events`, `n_discontinuities_analysed`, `n_segments`, `n_segments_usable`, `usable_duration_s`;
- `n_windows_continuous_grid`: windows the continuous grid would have placed, and how many of them would cross a
  cut;
- `n_windows_excluded_discontinuity`;
- `n_windows_total` (candidates), `n_windows_rejected` (QC), `n_windows_kept` (accepted);
- `discontinuity_samples`: cut positions on the 250 Hz time line.

**Verification gate.** The gate is no longer "no boundary events in the files". It is ZERO accepted windows
crossing a discontinuity or its margin, with every window inside one segment at least the margin from its ends.
- `d3_verify_smoke.py` checks this on the smoke subject and, in memory, on the recording with the most events.
- `validate_graph_run.py` checks it on the full run.

**D1/D2.** Unchanged. `configs/pipeline.yaml` has no `discontinuities` key, so the D1/D2 path runs the same code
as before (the frozen run `graphs_v4_33faa4b89a5d` is not touched). The D3 config hash changes, so D3 graphs go
to a new `graphsD3_v4_<hash>` directory.

**Code pins.**
- Before this entry, every D3 notebook first ran the shared setup on `CODE_COMMIT.txt` and then switched to
  `D3_CODE_COMMIT.txt`. A D3 session could therefore move the D1/D2 pin (with `UPDATE_CODE`, or automatically
  when no D1/D2 unit had been completed).
- Now `scripts/colab_setup.py --family D3` uses only `D3_CODE_COMMIT.txt` and D3 checkpoints (`graphsD3_v*`,
  `experiments_D3/`). It reads `CODE_COMMIT.txt` only to print it.
- `scripts/code_pin.py restore-d1d2` (notebook `advanced/Restore_D1D2_code_pin.ipynb`) sets `CODE_COMMIT.txt`
  back to the last pin in its history that predates Dataset 3 (expected `1b6988a`). It keeps the history and
  changes no result. It refuses if any D1/D2 unit was completed under the current pin.
- The Exp 7 pin is unchanged.

---

## 46. Dataset 3: electrode coordinates from the EEGLAB files (provenance: a shared template)

**Date:** 2026-10-15. **Requested by:** the author: the D3 forward and inverse models must use the electrode
coordinates supplied with the recordings, not a template substituted by this pipeline. D3 only; D1/D2 unchanged.
**Code:** `src/adgraph/io/chanlocs.py`, `io/eeg.load_raw`, `pipeline.subject_geometries` / `operator_identity` /
`get_operator` / `run` / `process_subject`, `configs/pipeline_d3.yaml` (`channels.coordinates`),
`scripts/d3_data.py check`, `scripts/d3_verify_smoke.py`. **Tests:** `tests/test_d3_coordinates.py`.

**Inspection of the real files** (all 65 AD/HC recordings of ds004504 v1.0.9, derivatives and raw BIDS files; only
the `chanlocs`, `chaninfo` and `history` variables were read, from the public OpenNeuro bucket, byte ranges only;
the Drive copy is the same content, MD5-verified at download):

| Property | Finding |
|---|---|
| Channels with coordinates | 19 of 19 in every file (Fp1 … Pz, provider order), all X/Y/Z finite |
| Units, axes | mm; EEGLAB axes (X nose, Y left, Z up), `chaninfo.nosedir = '+X'`; radius 83–117 mm |
| Provenance | every file: `chaninfo.filename` and `EEG.history` = `pop_chanedit(EEG, 'lookup', 'D:\eeglab2022.1\plugins\dipfit\standard_BEM\elec\standard_1005.elc')` |
| Subject-specific? | **No.** All 65 files hold identical coordinates (max difference 0.0 mm) |
| Relation to the template | identical (≤ 7·10⁻¹⁴ mm) to MNE's `colin27_1005` / `colin27_1020` (FieldTrip `standard_1005.elc`): Colin27 positions in MNI space |
| Fiducials | none (`nodatchans` = the A1/A2 mastoid references only) |
| BIDS sidecars | no `electrodes.tsv` / `coordsystem.json`; `EEGPlacementScheme` "10-20" |
| MNE's EEGLAB reader as is | labels the MNI coordinates as `head` frame with fiducials invented at ±97 mm; with `trans='fsaverage'` every electrode would land 46–57 mm from its stored position. Not usable without explicit frame handling |
| The template used so far (`standard_1020` → `colin27_1020`, `mri` frame, fsaverage fiducials) | positions given to the forward model identical to the stored ones (≤ 1.4·10⁻¹⁴ mm) |

The stored coordinates are therefore valid and interpretable, but they are the provider's template lookup, not
individual measurements. They cannot represent individual head geometry.

**Implementation.**
- `channels.coordinates.source: eeglab_chanlocs`: every D3 recording's montage is built from its own `chanlocs`.
  `make_standard_montage` is never called for D3.
- **Frame.** It is established from the provider's lookup record: `chaninfo.filename`, or a `'lookup'` in
  `EEG.history`, must name an accepted DIPFIT `standard_BEM` file. Those positions are MNI (Colin27). They are
  placed in the `mri` frame with the fsaverage MNI fiducials, exactly as MNE handles its colin27 templates, so
  `trans='fsaverage'` maps them onto fsaverage MRI space.
- **Checks; any failure raises `CoordinateError`, with no template fallback:**
  - all 19 channels present with finite 3D coordinates;
  - head radius consistent with the configured units (mm);
  - nose = +X, left/right sides correct;
  - an accepted lookup record. An unrecorded or other source, such as an individual digitisation in an unknown
    frame, stops the subject.
- **Operators.** One verified 68 × 19 operator per distinct geometry. The identity includes the geometry hash and
  positions. Every subject records `coord_source`, `coord_frame`, `coord_lookup`, `montage_hash`,
  `coord_template_deviation_mm` and `operator_id`. A subject whose geometry differs from its operator's stops.
  With one geometry, as in ds004504, there is one operator. The manifest stores the geometry with the operator,
  and all operators if there are several.
- **New identities.** The config hash changes, so D3 uses a new `graphsD3_v4_<hash>` run (old smoke output kept).
  The operator identity changes, so the old template-based operator is not reused (it stays in the cache).
- **Smoke checks** (`d3_verify_smoke.py`):
  - (1) the coordinates come from the file, and the operator's positions equal them;
  - (1) the head → fsaverage MRI transform returns the stored positions (< 0.01 mm);
  - (1) every subject is on its own geometry's operator;
  - (6) PSD, PLV and |ciPLV| are finite, in range and symmetric.
  - Provenance is saved in `d3_montage_checks.json`.

**Expected effect.** None numerically: the stored positions are the template used before. The change makes the
geometry come from the data with an explicit frame. It also stops any future recording whose positions differ or
whose frame cannot be established. The limitation that D3 source estimates use template electrode positions on a
template head (fsaverage) is unchanged and must be stated.

---

## 47. Investigation: the channel-mapping QC statistic `qc_spatial_rho` (no pipeline change)

**Date:** 2026-10-15. **Requested by:** the author (warnings in D1, D2 and D3; D3-1 rho = +0.109499 although its
coordinates were verified, entry 46). **Tool:** `scripts/diagnose_spatial.py` (read-only). Nothing in the
pipeline, the configurations, the graph runs or the pins was changed.

**What the statistic is.** `io.layout.spatial_consistency`, called in `pipeline.process_subject`: Spearman rho
between inter-electrode distance and **|r|** (absolute Pearson correlation). It is computed over all samples of the
preprocessed signal: cropped, resampled, band-passed for D1/D2, interpolated for D1/D2, **average-referenced**.
Positions are `info["chs"][i]["loc"]`, head frame, the same index `i` as the data rows. A warning is logged when
rho is not below −0.1 (entry 37).

**Evidence.**

| Signal | rho(dist, \|r\|) | rho(dist, r) | Far third of pairs: mean r, % negative |
|---|---|---|---|
| D3-1, before average reference (A1–A2; common signal = 96.5 % of variance, mean r = +0.96) | −0.676 | −0.676 | +0.95, 0 % |
| D3-1, after average reference (pipeline; reproduces +0.109499) | **+0.109** | **−0.798** | −0.41, 93 % |
| Simulation, 400 random dipoles, 3-shell sphere, D3 montage, reference-free | +0.066 | −0.968 | −0.40, 100 % |
| same, average reference | +0.041 | −0.912 | −0.37, 98 % |
| same, D1/D2 30-channel montage, average reference | −0.071 | −0.898 | −0.37, 99 % |

**Can each statistic detect a wrong mapping?** Same signal, positions shuffled 1000×:
- D3-1, |r|: the correct mapping's +0.109 is above 95.5 % of the shuffled mappings.
- D3-1, signed r: no shuffle reached −0.798 (5–95 % of the null: [−0.146, +0.099]).
- Swapping one distant pair raises the signed rho to −0.41 (Fp1↔O2) or −0.55 (F7↔T6).
- Swapping neighbouring electrodes is barely visible to either statistic.

**Conclusion.**
- **Volume conduction:** nearby electrodes are positively correlated, and distant electrodes **negatively**
  correlated. Each source produces a dipolar field with opposite poles; this holds even reference-free, and
  average referencing removes the common signal.
- **Effect of |r|:** taking |r| folds the strong negative far-field correlations into large values, so with a
  correct mapping the statistic sits near 0 (or slightly positive). The −0.1 threshold therefore warns for correct
  mappings by construction. Its validation used a purely positive Gaussian mixing kernel without referencing
  (`tests/test_channel_mapping.py`), which does not produce these negative correlations.
- **Before average reference** the |r| statistic was negative (−0.68), but there it is dominated by the common
  reference signal.
- **No mapping or geometry error is indicated** for D3-1: the signed statistic is −0.80, far outside the shuffle
  null.

**Proposed correction (not applied; needs the author's decision; D3 first, D1/D2 would only ever be
re-evaluated as QC):** use the signed correlation (drop `np.abs`) on the same signal, and warn when the observed
rho is not clearly better than shuffled mappings, e.g. a permutation p ≥ 0.01, instead of a fixed −0.1. The QC
statistic would then be redefined; the graphs are unaffected, since nothing depends on it.

---

## 48. Dataset 3: Experiment 3 (single bands) added

**Date:** 2026-10-15. **Requested by:** the author, after the D3 graph run was built and validated (65
participants, 3,974 windows) and D3 Exp 0 and Exp 1 had been completed. Exp 2, 4 and 5 are not added for D3.
**Config:** `configs/experiments/d3/d3_exp3_conn_psd_fixedhp.yaml`. **Notebook:**
`notebooks/d3/experiments/D3_Exp3_conn_psd_single_band.ipynb`. **Test:**
`tests/test_d3.py::test_d3_exp3_reuses_exp3_and_runs_on_d3_only`.

- **Definition unchanged.** The config extends the Tier 1 `exp3_conn_psd_fixedhp.yaml`; the test checks that every
  variant's graph, node features, model, training, evaluation and protocol equal the D1/D2 ones. Concretely:
  - 4 bands (theta, alpha1, alpha2, beta), each evaluated on its own, × {PLV, |ciPLV|} = 8 variants;
  - single-band source connectivity graph; nodes = matching-band [log, relative] source PSD; GCN;
  - `gnn_fixed_hparams_v1`: no search, no inner CV, 60 epochs, no early stopping;
  - 10 repeats × 5 stratified participant folds, seeds as for every other experiment;
  - training-fold-only, per-participant and class-balanced weights; participant-level metrics.
- **D3 only.**
  - `graph_family: D3`, setting `within_D3`: the frozen D3 graph run (`graphsD3_v4_*`) is read, and results go
    to `experiments_D3/d3_exp3_conn_psd_fixedhp/`.
  - Mixing with D1/D2 settings is refused (`validate`).
  - No code, graph construction, preprocessing or source model changed.
- **Code pin.** The experiment runs its own pin, `experiments_D3/d3_exp3_conn_psd_fixedhp/CODE_COMMIT.txt`
  (as Exp 7 does). `D3_CODE_COMMIT.txt`, under which the D3 graph run and D3 Exp 0/1 were produced, and the
  D1/D2 pin are not moved.
- **Timing.** It was added after the D3 Exp 0/1 results were observed. Its definition is the pre-existing Exp 3,
  and no D3 result entered any choice.

---

## 49. Dataset 3: Experiments 2, 4 and 5 added

**Date:** 2026-10-15. **Requested by:** the author. The frozen D3 graph run `graphsD3_v4_da723b33554c` has 65
participants and 3,974 graphs; D3 Exp 0/1 are complete and Exp 6 is running.
**Configs:** `configs/experiments/d3/d3_exp{2_conn_minimal,4_band_mean,5_multiband}_fixedhp.yaml`.
**Notebooks:** `notebooks/d3/experiments/D3_Exp2_conn_minimal.ipynb`, `D3_Exp4_band_mean.ipynb`,
`D3_Exp5_multiband.ipynb`. **Test:** `tests/test_d3.py::test_d3_gnn_experiments_reuse_d1d2_definitions_and_run_on_d3_only`
(also covers Exp 3).

- **Definitions unchanged**, as for D3 Exp 3 (entry 48). Each config extends the Tier 1 D1/D2 config. The test
  checks that every variant's graph, node features, model, training, evaluation and protocol equal the D1/D2 ones:
  - Exp 2: single band, constant node feature, GCN; 4 bands × {PLV, |ciPLV|} = 8 variants.
  - Exp 4: band-mean adjacency, [log, rel] PSD of all bands (F = 8), GCN; 2 variants.
  - Exp 5: four band graphs, shared GCN, mean / attention fusion, matched-band PSD; 4 variants.
  - All use `gnn_fixed_hparams_v1`: 10 repeats × 5 stratified participant folds, with training-fold-only weights
    and participant-level metrics.
- **D3 only.** `within_D3`, results in `experiments_D3/<experiment>/`, each experiment with its own code pin
  (`experiments_D3/<experiment>/CODE_COMMIT.txt`). `D3_CODE_COMMIT.txt` and the D1/D2 pin are not moved. No code,
  graph, preprocessing or source-model change.
- **Timing.** Added after D3 Exp 0/1 results were observed. The definitions are the pre-existing ones, and no D3
  result entered any choice.

---

## 50. D3 LCMV beamforming + Xu et al. (2025) protocol suite (Exp 0-7), a separate pipeline

**Date:** 2026-10-16. **Requested by:** the author. **Full description:** `docs/lcmv_xu.md`. **Code:**
`src/adgraph/lcmvxu/`, `configs/lcmv_xu/pipeline.yaml`, `configs/experiments/lx/`, `scripts/lx_*.py`,
`notebooks/lcmv_xu/`.

A new pipeline next to, not instead of, the existing ones. Outputs only in `<processed>/D3_LCMV_XU/`. It has its own
code pin (`colab_setup.py --family LX`), configuration hash and checkpoint scope (`lx_graph`). D1/D2, the wMNE D3
graph run `graphsD3_v4_da723b33554c`, `experiments_D3/`, `D3_CODE_COMMIT.txt` and the D1/D2 pin are not touched.

**Decisions:**
- **Source model:** per-participant time-domain LCMV beamformer on fsaverage with fixed cortical-normal
  orientation. Covariance from the participant's own accepted broadband samples, rank estimated from the data,
  `reg = 0.05`, unit-noise-gain-invariant weights. `mean_flip` over the 68 Desikan-Killiany ROIs, same order as
  before. Every participant's 68 × 19 operator is verified against MNE's `apply_lcmv_raw`.
- **Bands:** Xu's five bands; alpha is not split. 1.0 Hz FIR transitions, delta lower transition capped at 0.5 Hz.
- **Features:** PSD (abs/log/rel), differential entropy, PLV and |ciPLV|.
- **Recordings:** split at every EEGLAB boundary; each segment resampled on its own.
  - The only trimming is the 3.3 s delta-kernel margin from each segment end.
  - Five minutes are selected deterministically: the first uninterrupted 300 s, else separate pieces in time
    order, never joined.
  - 10 s windows with a 1 s step; the existing peak-to-peak rejection; the pre-specified ≥ 5-window rule.
  - The participant QC report lists every participant.
- **650 samples:** the authors' code reads 10 samples per participant (65 × 10). Mode A reproduces that count with
  the first 10 accepted windows, i.e. the literal 90 %-overlap tenfold reading. Mode B uses every accepted window
  of the five minutes (up to 291 per participant). The paper's "309 s" and the 32 DE values per node could not be
  verified; this is documented and not imitated.
- **Validation:** one frozen, stratified 80/20 participant split (seed 20250702; 52 development / 13 holdout),
  identical for all experiments and both modes. Settings `xu_dev_cv` (5-fold participant CV inside the development
  set) and `xu_holdout` (one fit on the development set, evaluated on the holdout). Participant-level metrics are
  primary.
  - Xu's code instead reports the best epoch on five overlapping participant-level 80/20 splits (corrected in
    entry 51). Here the last-epoch model is
    used, with no selection on evaluation data.
- **Experiments:** Exp 0-7 keep their existing definitions with five bands. DE variants are added as distinct
  configurations: Exp 0, 1 (classical) and Exp 4.
  - Exp 5 adds the band-specific branches of its Tier 2 definition.
  - Exp 6 keeps its ablation types for five bands.
  - Exp 7 is EARM-GNN unchanged.
- **Shared code changes:** `graphs.store.PSD_KINDS` gains the optional kind `de`; the settings `xu_dev_cv` and
  `xu_holdout`; the experiment families `LX` / `LX10`. Existing stores, settings and families behave as before.

**Checks run before delivery** (no long tests, by request):
- imports and configuration expansion (16 configs);
- recording handling on two real files (D3-1 and D3-56): zero window/boundary violations;
- LCMV operator vs `apply_lcmv_raw`: 1.6e-15, on a spherical head model, because fsaverage is not downloadable in
  this environment; the fsaverage path is checked by the Colab smoke test;
- feature shapes and finiteness;
- split disjointness;
- a short synthetic run of Exp 1, 5, 6 and 7 in both settings.

---

## 51. Review of the LCMV / Xu suite against the full Xu et al. (2025) paper

**Date:** 2026-10-16. **Requested by:** the author, with the full paper (Front. Neurosci. 19:1555657). It was
compared with the authors' `Model.py` and with commit b38685e. Details: `docs/lcmv_xu.md` §4.

**Result: no dataset or experiment change is needed.** The build may be running; configurations, schemas, hashes,
outputs and checkpoints are untouched. Only the documentation is corrected.

- **Confirmed matches:**
  - 5-min segment per participant;
  - 10 s windows, 90 % overlap (1 s step);
  - the five bands, exactly;
  - Gaussian DE of the band-passed signal (natural log vs. Xu's log₂: a constant factor, removed by
    standardization);
  - participant-grouped 80/20 split with the same counts (52/13; 7 AD, 6 HC);
  - all samples of a participant in one partition.
- **Corrected statements** (`docs/lcmv_xu.md`, entry 50):
  - Xu's five "folds" are five overlapping participant-level 80/20 splits, not a partition; 15 AD and 13 HC
    participants are never validated;
  - Xu's code has no anatomical-region adjacency: its second layer reuses the first layer's edges thresholded
    at 0.5, and it casts the correlation matrix to integers;
  - Xu's code reports per-sample metrics, the best of 200 epochs, with HC as the positive class;
  - the paper's K, S and F were selected on its test set (accuracy steps of 1/130);
  - the coded model has 29,204 parameters against 29,332 in the paper.
- **Unresolved:** how 10 samples per participant arise from 5 minutes ("309 s" in the code's folder name), and how
  the 32 DE values per node are formed. Mode A remains our operationalisation, labelled as such.
- **Not transferred:** Xu's 19-node dimensions (K = 16, S = 128, F = 32, batch 10, 200 epochs) to our 68-ROI
  models.

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
7. The target journal's width and font requirements (figure selections were resolved:
   entries 25–29).
8. ~~Connectivity band-pass transition widths~~: resolved in entry 36 (1.0 Hz transitions, nominal bands unchanged).
9. Permutation tests (off by default): the null uses `permutation.n_repeats` CV repeats per
   permutation while the observed statistic averages all repeats, which makes the test
   conservative. Match them (e.g. compare repeat 0, or use equal repeats) before enabling it.
10. No module produces the corrected, subject-level statistics file that figures can show
    significance from (`docs/figures.md`). All figures are currently descriptive.
11. ~~Preprocessing amplitude check~~: resolved in entry 39 (edge transients; check moved to
    the analysed interval, unit safeguard added, bad channels on the analysed interval).
12. Hilbert transform over the whole continuous ROI signal (entry 39): edge transients shift
    PLV/|ciPLV| of retained windows slightly (≤ 0.0025 for 3 mV transients, decaying with
    distance). Decide after the smoke-test inspection whether this needs a change (e.g. analytic
    signal computed on the analysed interval only); frozen definitions are unchanged until then.
