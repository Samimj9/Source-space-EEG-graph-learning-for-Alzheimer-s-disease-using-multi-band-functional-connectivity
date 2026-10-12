# Methods decisions log

The single source of truth for methodological choices, especially
deviations from the reference papers ("Paper 1" = source-space pipeline
reference; "Paper 2" = the publication of Dataset 2; full citations TBD).
Add an entry whenever a parameter changes.

Rows 1–11 are seeded from `processed/run_manifest_20260912_162006.json` (pipeline v1, code not in this repo).
Pipeline v2 (`configs/pipeline.yaml`, `src/adgraph/pipeline.py`) supersedes the graph sets from v1; rows 15–23 record what changed.

| # | Stage | Decision | Reference choice | Reason | Status |
|---|---|---|---|---|---|
| 1 | Preprocessing | Resample to 250 Hz; band-pass 0.1–45 Hz | | | adopted |
| 2 | Preprocessing | Average reference | not specified in Paper 1 | | adopted |
| 3 | Preprocessing | `standard_1020` montage, matched by channel name | landmark-based coregistration | no digitization available | adopted (limitation) |
| 4 | Preprocessing | D1 EDF ×1e-6 rescale | n/a | headers mislabel µV as V | adopted |
| 5 | Preprocessing | Drop M1/M2 in D1 only | | | superseded by #15 |
| 6 | Windowing | 40 s windows, 20 s step, 10 s edge trim; keep subject if ≥5 windows | | | superseded by #24 |
| 7 | Rejection | Per-subject adaptive threshold: median + 5·MAD of peak-to-peak; reject window if >15 % of channels exceed it | fixed ±80 µV | the fixed threshold did not transfer to this data | adopted |
| 8 | Source | fsaverage, ico-5 (20 484 vertices), 3-layer BEM 5120 | ICBM152 / Brainstorm | MNE-Python toolchain | adopted |
| 9 | Source | MNE inverse, loose 0.2, depth 0.8, λ² = 1/9 | | | adopted |
| 10 | Source | Desikan-Killiany (aparc), 68 ROIs, `mean_flip` | | | adopted |
| 11 | Connectivity | PLV, theta 4–8, alpha1 8–10, alpha2 10–13, beta 13–30 Hz | | | adopted |
| 12 | Data | Use all 43 subjects and recordings as provided; no duplicate-based exclusion | n/a | project decision (2026-10-12) | adopted |
| 13 | Evaluation | Subject-grouped CV, subject-level metrics primary | | windows are not independent | adopted |
| 14 | Evaluation | Frozen evaluation design v1.0: S1/S2 within-dataset (primary), S3 pooled (secondary), S4 cross-dataset (gated). Repeated 5-fold ×20 with nested 4-fold, subject level. See `DESIGN.md` | | pre-registration before classification | adopted |
| 15 | Harmonization (v2) | Drop M1/M2 in **both** datasets → identical 30-channel set (order from `electrodes_positions_32.loc`); average reference over those 30 | | mastoids sit at the edge of the template head model and were kept in D2 only; identical sensor sets are required by the `DESIGN.md` §7 gate | adopted |
| 16 | Source (v2) | **One** forward model, inverse operator and ROI extraction shared by D1 and D2, collapsed into a single 68×30 sensor→ROI matrix and verified at runtime against MNE's full inverse | separate forward model per dataset (v1) | removes a dataset-specific processing difference; needed for the gate | adopted |
| 17 | Source (v2) | Noise covariance: ad hoc diagonal (`mne.make_ad_hoc_cov`) | not recorded (v1) | resting state has no baseline/empty-room data | adopted |
| 18 | Source (v2) | `pick_ori='normal'`: signed normal component of the loose-orientation solution | not recorded (v1) | the default (`None`) returns the unsigned norm, which destroys the phase needed for PLV and makes `mean_flip` meaningless | adopted |
| 19 | Preprocessing (v2) | Bad channels: same automatic rule for both datasets (flat, robust-z of log std > 5, 90th-pct neighbour correlation < 0.4 on 1–45 Hz), spherical-spline interpolation **before** re-referencing; no subject excluded | not handled (v1); sheets give only bad-channel counts, not names | the comment column cannot be used, so the rule must be data-driven and identical | adopted |
| 20 | Preprocessing (v2) | Units: D1 ×1e-6 kept, plus an automatic plausibility check (median channel std must be 0.1–200 µV after filtering) for both datasets | fixed rescale only (v1) | guards against unit errors in either dataset | adopted |
| 21 | Connectivity (v2) | Band filtering + Hilbert applied to the **continuous** ROI signals, then windows are cut | unspecified (v1) | avoids filter edge effects at every window boundary | adopted |
| 22 | Connectivity (v2) | ciPLV (corrected imaginary PLV) stored next to PLV; PLV stays primary | PLV only | source leakage inflates zero-lag PLV; ciPLV is the leakage-robust sensitivity analysis | adopted (sensitivity) |
| 23 | Graphs (v2) | Stored per window: dense PLV and ciPLV (4 bands × 68 × 68, no self-loops) and log10 ROI band power (candidate node features); no thresholding at build time | | sparsification and node features are model-stage choices, made inside training folds | adopted; power replaced by #25 |
| 24 | Windowing (v3) | **Frozen primary segmentation: 20 s windows, 50 % overlap (5000 / 2500 samples at 250 Hz), same for D1 and D2**; 10 s edge trim and ≥5-window rule kept; filtering on the continuous signal before segmentation | 40 s / 20 s step (v1, v2) | project decision (2026-10-12) | adopted |
| 25 | Features (v3) | Source-ROI and sensor PSD per window and band: absolute, log10 and relative (to 1–45 Hz) power; Welch 2 s Hann, 50 % overlap (configurable: multitaper, periodogram); half-open bands | Hilbert-envelope log power (v2) | node features for Exp 3–6; sensor PSD for Exp 0 | adopted |
| 26 | Graphs (v3) | Each window carries `window_id`, subject/dataset IDs, label, sample positions and QC; optional continuous ROI time series per subject | | traceability of every sample | adopted |
| 27 | Models | Dense PyTorch GCN / GAT / GraphSAGE (no PyG). GAT uses edge weights in attention logits and messages; GraphSAGE uses weighted-sum aggregation by default | PyG layers | with constant node features, plain GAT and mean-aggregation SAGE give identical outputs for every graph | **needs approval** |
| 28 | Experiments | Exp 2/3 run single-band graphs (each band a separate variant); Exp 4 node features = PSD of all bands; Exp 5 = matched-band PSD; Exp 6 = [PSD, strength] per band, attention fusion as the learned fusion; PSD node features = {log, rel}; Exp 4–6 encoder = GCN | | see `docs/experiments.md` | **needs approval** |
| 29 | Training | Loss weights: one vote per subject, classes balanced by subject count; Adam, 60 epochs, batch 32, no early stopping by default; inner-CV grid: hidden {16, 32} × weight decay {1e-4, 1e-3}; LR/SVM C ∈ {0.001 … 10} | | | **needs approval** |
