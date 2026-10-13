# Experiment framework

This document covers the code that runs Experiments 0–6 and the baselines. Nothing in this
repository has been run on the real EEG; every test uses synthetic data
(`adgraph.synthetic`). Evaluation follows `DESIGN.md`.

## 1. Data flow

Graph construction and the experiments are separate. Their only interface is the
graph-run directory (schema: `src/adgraph/graphs/store.py`).

```
scripts/build_graphs.py ──► processed/graphs_v3_<config hash>/
                              graphs.npz  windows.csv  subjects.csv  manifest.json  [roi_timeseries/]
                                              │
scripts/run_experiment.py --config configs/experiments/<exp>.yaml --graphs <run dir>
   GraphStore.load ─► build_dataset (stateless, per window) ─► subject-level splits
   ─► per outer split: inner-CV hyperparameter selection ─► model.fit(train view)
   ─► predict(test view) ─► window scores ─► subject score (mean) ─► metrics
```

### Segmentation (frozen primary setting)

The segmentation is the same for D1 and D2:

- **20 s windows with 50 % overlap.** At 250 Hz that is 5000-sample windows with a
  2500-sample step.
- **Edge trim:** 10 s is discarded at each end of the recording.
- **Filtering first:** sensor band-pass, bad-channel interpolation, re-referencing,
  the ROI projection and the per-band ROI filtering + Hilbert transform are all
  applied to the **continuous** signal before cutting windows.

Candidate windows per recording (before artifact rejection):
`n = floor((duration − 2·10 − 20) / 10) + 1`. For example, 300 s gives 27 windows
and 600 s gives 57 (`adgraph.preprocessing.windows.expected_n_windows`).

A nominal 300 s recording yields 27 windows, not 29, because of the 10 s trim at
each end (`windowing.edge_trim_s`). Without the trim, the formula
`floor((300 − 20) / 10) + 1` gives 29. The trim is configurable and is kept at
10 s for now.

Consecutive windows share 10 s of signal. Splits are therefore built on subjects,
and every window of a subject is always on the same side of a split
(`tests/test_splits.py::test_overlapping_windows_never_cross_folds`).

### Per-window record (one row of `windows.csv` ↔ index *i* of every array)

| Field | Content |
|---|---|
| `window_id` | `<dataset>-<subject>-w<index>`, where index counts candidate windows before rejection (stable) |
| `key`, `dataset`, `subject_id`, `label`, `y` | subject identity and label |
| `window_index`, `start_sample`, `stop_sample`, `tmin`, `tmax` | position in the preprocessed recording |
| `sample_weight` | 1 / number of kept windows of the subject |
| `qc_frac_channels_over_threshold`, `qc_max_ptp_uv` | window QC |
| `subject_*` | subject QC: bad channels, rejection threshold, window counts, median amplitude |
| `plv[i]`, `ciplv[i]` | (4, 68, 68) connectivity, theta/alpha1/alpha2/beta, zero diagonal |
| `ciplv_signed[i]` | (4, 68, 68) signed ciPLV, antisymmetric, in [−1, 1]; optional; never an adjacency |
| `psd_abs[i]`, `psd_log[i]`, `psd_rel[i]` | (4, 68) source ROI band power |
| `sensor_psd_*[i]` | (4, 30) electrode band power (Experiment 0) |
| `roi_timeseries/<key>.npy` | optional continuous (68, n_times) ROI signals; a window is `[:, start:stop]` |

PLV is the primary connectivity measure and ciPLV is the sensitivity analysis. They
are separate arrays and separate experiment variants, and the code never combines
them.

**ciPLV sign handling**

- `phase_sync` computes the signed ciPLV, `Im C / sqrt(1 − (Re C)²)`. It lies in
  [−1, 1] and is antisymmetric.
- Its sign is the sign of the non-zero-lag (imaginary) component of the mean
  phase relationship φ_j − φ_k. It is therefore defined relative to the ROI
  ordering (j, k) and flips when the order is swapped.
- The sign does **not** establish directed connectivity or causal direction.
  Phase is circular, and common drive or differing regional dynamics can produce
  a consistent phase offset without any directed interaction.
- Every GNN adjacency and every connectivity strength uses the symmetric,
  non-negative **magnitude** |ciPLV|, stored as `ciplv`. It has a zero diagonal
  and lies in [0, 1], like PLV.
- The signed matrix is stored separately as `ciplv_signed`
  (`output.save_signed_ciplv`), for descriptive use only. `build_graph` refuses
  it as an adjacency.
- Both connectivity arrays are checked on load: symmetric, zero diagonal, values
  in [0, 1] (`check_adjacency`).

### PSD features (`src/adgraph/features/psd.py`, `configs/pipeline.yaml: psd`)

- **Estimator** (configurable): Welch (default: 2 s Hann segments, 50 % overlap,
  mean averaging, linear detrend, 0.5 Hz resolution), multitaper (DPSS, configurable
  bandwidth) or Hann periodogram. All return a one-sided density.
- **Absolute band power:** the sum of PSD bins with `lo ≤ f < hi`, times the bin
  width. Bands are half-open, so they never share a bin.
- **Log power:** log10 of the absolute power.
- **Relative power:** absolute power / power in 1–45 Hz.
- **Coverage:** computed per window for the 68 ROI signals (source PSD) and the 30
  preprocessed electrodes (sensor PSD).

## 2. Experiments

| Exp | Config | Input | Models | Variants |
|---|---|---|---|---|
| 0 | `exp0_sensor_psd.yaml` | electrode PSD, 30 ch × 4 bands × {log, rel} = 240 features | LR, SVM | 2 |
| 1 | `exp1_source_psd.yaml` | ROI PSD, 68 × 4 × {log, rel} = 544 features | LR, SVM | 2 |
| 2 | `exp2_conn_minimal.yaml` | one band graph; edges PLV or ciPLV; constant node feature | GCN, EdgeWeightedGAT, WeightedGraphSAGE | 2 conn × 4 bands × 3 = 24 |
| 3 | `exp3_conn_psd.yaml` | one band graph; nodes = **matching-band** ROI PSD [log, rel] (F = 2) | GCN, EdgeWeightedGAT, WeightedGraphSAGE | 24 |
| 3b | `exp3b_conn_psd_allbands.yaml` | secondary: one band graph; nodes = ROI PSD of all 4 bands (F = 8) | as Exp 3 | 24 |
| 4 | `exp4_band_mean.yaml` | `A_mean` = mean of the 4 band matrices; nodes = ROI PSD (8) | GCN (configurable) | 2 conn |
| 5 | `exp5_multiband.yaml` | 4 band graphs; nodes = matched-band PSD (2) | GCN; shared / band-specific; mean / concat / gate / attention fusion | 2 × 2 × 4 = 16 |
| 6 | `exp6_multiband_psd_strength.yaml` | 4 band graphs; `X_i,b = [log_abs_PSD_i,b, rel_PSD_i,b, strength_i,b]` (3), strength from band b's PLV or \|ciPLV\| | GCN + attention fusion | 2 conn × 8 ablations = 16 |
| – | `baseline_age.yaml` | age | LR | 1 |
| – | `baseline_conn_classical.yaml` | upper-triangle PLV / ciPLV (4 × 2278 = 9112) | LR, SVM | 4 |
| – | `probe_dataset.yaml` | ROI PSD, target = dataset (pooled only) | LR | 1 |

Experiment 6 ablations: `full`, `no_psd`, `no_strength`, `no_learned_fusion` (mean
fusion), and `drop_<band>` for each band. Each runs for PLV and for ciPLV.

### Age sensitivity analysis (DESIGN.md §5)

| Model type | Configs |
|---|---|
| Age only | `baseline_age.yaml` |
| EEG only | the experiment configs above (`covariates: []`) |
| EEG + age | `configs/experiments/age/<exp>_plus_age.yaml` for Exp 0–6 and the connectivity baseline |

Each EEG + age config `extends` its EEG-only config and sets `covariates: [age]`,
so the two use identical splits, grids and seeds.

### Provisional training protocol

The following are configurable and are **not** frozen as the final scientific
evaluation:

- 20 repeats × 5 folds, 4-fold inner CV
- epochs, early stopping
- hyperparameter grids
- bootstrap and permutation counts

Override any of them per run, e.g. `--override evaluation.n_repeats=5`.

Each config `extends` `classical_base.yaml` or `gnn_base.yaml`, which in turn extend
`base.yaml` (the frozen evaluation protocol):

- **`sweep`** expands into variants (cartesian product).
- **`variants`** adds named overrides on top of the sweep.
- **`hparams`** is a grid searched by subject-level inner CV. It never creates
  variants.

Any key can be changed from the command line without editing Python, e.g.
`--override model.encoder=gat`.

## 3. Tensor shapes

N = number of windows, K = number of band graphs, R = 68 ROIs, F = node features per graph.

| Model | Input | Shape |
|---|---|---|
| LR / SVM | `X` | (N, n_features); Exp 0: 240, Exp 1: 544, age: 1, conn: 9112 (all bands, log+rel) |
| GNN, single band (Exp 2, 3, 3b) | `x`, `adj` | (N, 1, 68, F), (N, 1, 68, 68); F = 1 (Exp 2), 2 (Exp 3), 8 (Exp 3b) |
| GNN, band mean (Exp 4) | `x`, `adj` | (N, 1, 68, 8), (N, 1, 68, 68) |
| GNN, multi-band (Exp 5) | `x`, `adj` | (N, 4, 68, 2), (N, 4, 68, 68) |
| GNN, multi-band (Exp 6) | `x`, `adj` | (N, 4, 68, 3), (N, 4, 68, 68); K = 3 when a band is dropped |
| `MultiBandGNN.forward` | x (B, K, R, F), adj (B, K, R, R) | logits (B,), band weights (B, K) |
| encoder (per band graph) | x (B, R, F), adj (B, R, R) | node embeddings (B, R, H) |
| readout `mean_max` | (B, R, H) | (B, 2H) |
| fusion `concat` / others | (B, K, D) | (B, K·D) / (B, D) |

The models are dense implementations in plain PyTorch (`src/adgraph/models/`), with
no PyTorch Geometric dependency. Every graph has the same 68 nodes and is fully
connected and weighted, so batched dense matrix operations are the natural form.

**Encoder names in code and in the paper**

| Config name | Paper name | What it is |
|---|---|---|
| `gcn` | GCN | Standard Kipf & Welling GCN on the weighted adjacency (symmetric normalization with self-loops). This is the standard weighted-adjacency baseline. |
| `edge_weighted_gat` | EdgeWeightedGAT | **Modified** GAT. The edge weight enters the attention logits (`c_h·A_ij`) and the messages (`A_ij·u_h`). |
| `weighted_graphsage` | WeightedGraphSAGE | **Modified** GraphSAGE. Edge-weighted neighbour aggregation, weighted sum by default. |

Canonical unweighted GAT and GraphSAGE are not implemented. Their plain names are
rejected by `build_encoder`, so results cannot be mislabelled. The modifications
are needed because, on fully connected graphs with constant node features
(Experiment 2), canonical GAT and mean-aggregation GraphSAGE give the same output
for every graph.

**Covariates (EEG + age)**

- `cov` has shape (N, 1). Each model standardizes it with training-subject
  statistics.
- Classical models append it to the EEG feature vector after
  scaling/selection/PCA.
- `MultiBandGNN` concatenates it to the fused (pooled) graph embedding right
  before the final MLP: head input = D_fused + 1.

## 4. Leakage protection (built into the code)

1. **Splits** are built from one row per subject and only then expanded to windows.
   `fold_indices` re-checks subject and window-ID disjointness for every split.
2. **Stateless versus fitted transforms.**
   - Per-window, stateless transforms are computed once in
     `graphs/dataset.py`: band selection, band averaging, node strength, PSD
     features, PLV/ciPLV choice.
   - Every fitted transform lives *inside* the estimator: standardization,
     `SelectKBest`, PCA, node-feature scaling, quantile sparsification,
     early-stopping split.
3. **Roles.**
   - Data reach models only as `DataView`s with a role (`train`/`val`/`test`).
   - Every `fit` calls `require_train`, so fitting on a validation or test view
     raises `LeakageError`.
   - The test view is created only after the model has been fitted.
4. **Hyperparameters** are selected by subject-level inner CV inside the training
   view (`select_hparams`). Early stopping (off by default) draws validation
   subjects from the training subjects only.
5. **Loss weights** are computed from the training view: one vote per subject and
   classes balanced by subject count (`training_weights`).
6. **SVM scores.**
   - The primary SVM baseline (`model.calibration: none`) uses the
     `decision_function` margin as its score. ROC-AUC and the subject-level
     aggregation (mean margin) use it directly, with the fixed threshold 0
     (`evaluation.decision_threshold`). No calibration is fitted.
   - `model.calibration: grouped_platt` is optional, only for when calibrated
     probabilities are explicitly required. Platt scaling is then fitted on
     out-of-fold margins from a **subject-grouped** internal CV
     (`GroupedPlattSVC`, `StratifiedGroupKFold`), and the threshold is 0.5.
   - sklearn's window-level `CalibratedClassifierCV` is never used.
   - LR and GNNs output probabilities with threshold 0.5.
   - Every run records `score_type` and the threshold in `summary.json` and
     `manifest.json`.
7. **Age** (EEG + age models) is standardized with the training subjects' mean
   and SD (`CovariateScaler`, one value per subject), inside each fold.
8. **Cross-dataset runs** are refused unless the graph run passes the DESIGN.md §7
   gate (`experiments/gate.py`).

## 5. Outputs (`<out>/<experiment>/<variant>/<setting>/`, checkpointed, never overwritten)

Analyses other than the primary one get their own variant directory, so they can never
overwrite or be mistaken for primary results: `<variant>__target-dataset` (dataset probe) and
`<variant>__scheme-loso`. The figure index uses primary results only (diagnosis target, k-fold,
uncalibrated).

| File | Content |
|---|---|
| `predictions_windows.csv` | per window and repeat: identifiers, y, score (probability, or SVM margin) (+ learned band weights for multi-band models) |
| `predictions_subjects.csv` | subject score (mean of window scores) per repeat |
| `metrics_repeats.csv` | subject-level accuracy, balanced accuracy, F1, sensitivity, specificity, ROC-AUC, MCC, confusion matrix, per repeat; pooled runs add per-dataset rows |
| `summary.json` | mean/SD/percentiles across repeats, repeat-averaged metrics with a subject bootstrap 95 % CI, window-level metrics, optional permutation test |
| `folds.csv`, `hparams.csv`, `manifest.json` | split assignment, selected hyperparameters with inner scores, `n_parameters` and `fit_seconds` per split, provenance |
| `embeddings.npz` | optional (`evaluation.save_embeddings: true`, GNNs): fused graph embeddings of train and test windows of every split, from that split's single fitted model, with key, label, dataset |

These files contain per-subject predictions, so they belong in Drive and are never
committed.

**Checkpoints in the same directory**

| File | Content |
|---|---|
| `_identity.json` | experiment identity: training-config hash, `graph_run_id`, setting, code version, automatic code fingerprint (git commit + source SHA-256 + library versions) |
| `units/fold_r<rrr>_f<ff>.*` + `.done.json` | one outer fold: predictions, fold table, hparam row, optional embeddings |
| `units/inner_fold_r<rrr>_f<ff>_*` | one inner-CV trial (parameter set × inner fold) of that outer fold; **GNNs only** (a classical fit takes milliseconds, less than writing a checkpoint to Drive, so classical inner searches are redone on resume; their outer folds are checkpointed) |
| `units/perm_*` | one permutation |
| `status.csv` | pending / running / completed / failed per outer fold |
| `final.done.json` | validates the final outputs above (written last) |

On restart, the same command loads validated units and computes only what is missing.
`--status` reports progress, and `--force` moves the directory to `<out>/_superseded/`.
Results produced by different code (another commit, edited source or other library versions)
are never reused. They raise `IncompatibleCheckpoint`, and `--status` shows `changed: code`.
To resume after new commits were pushed, check out the run's commit. For real graph runs
`--out` must be on the mounted Drive; ephemeral Colab storage is refused (methodology log
entry 32).

## 6. Running experiments: one at a time (Colab)

The normal workflow runs **one experiment per notebook** and reviews it before starting the
next one (`notebooks/README.md` lists the notebooks and their estimated cost):

```
notebooks/01_build_graphs.ipynb                 graphs + QC (once)
notebooks/experiments/Exp0_sensor_psd.ipynb ... Exp6_multiband_psd_strength.ipynb
notebooks/additional/                           Exp3b, baselines, dataset probe, EEG + age (one config each)
notebooks/03_comparative_figures.ipynb          paper figures across experiments (afterwards)
notebooks/advanced/Run_all_experiments_ADVANCED.ipynb   whole run plan; starts nothing unless enabled
```

Each experiment notebook calls `scripts/experiment.py` for its own config only:

| Action | What it does |
|---|---|
| `status` | progress per (variant, setting) from the checkpoints; computes nothing |
| `estimate` | exact number of model fits of the full protocol (repeats × outer folds × (inner folds × grid + refit)), one real fit timed per architecture on the current runtime and graph run, the checkpoint write time on Drive, and the time remaining (completed folds subtracted); saved in `_estimates/` |
| `run` | every variant and setting of this config, with the config's full protocol; order: within-dataset settings (all variants) → pooled → cross-dataset (only if the DESIGN.md §7 gate passes, otherwise recorded as skipped); resumable; a failing unit is recorded and the others continue; never starts another experiment |
| `report` | results table + single-experiment diagnostic figures from saved outputs |

Per experiment, Drive holds `<processed>/experiments/<experiment>/`:

| Path | Content |
|---|---|
| `<variant>/<setting>/` | results and checkpoints of one (variant, setting) (table above) |
| `_run_status.csv` | state of every (variant, setting) of the last run (completed / failed / skipped + message) |
| `_estimates/estimate_<time>.csv/.json` | runtime estimates |
| `_reports/report_<time>/` | `results_summary.csv` (subject-level mean ± SD over repeats, bootstrap 95 % CI, window-level AUC), `progress.csv`, `README.txt`, `figures/` |

**Single-experiment diagnostics vs comparative figures.** A report's figures (`D1_performance`,
`D2_roc_<setting>`, `D3_confusion_<setting>`, `D4_subject_scores_<setting>`) show only that
experiment's variants, for checking it. They select or rank nothing and are not paper figures.
The comparative paper figures (`configs/figures/paper.yaml`, `docs/figures.md`) are generated
by `03_comparative_figures` from the saved results of several experiments.

### Estimated cost (full protocol, before launching)

See `notebooks/README.md` for the per-experiment table. It is measured by
`scripts/estimate_runtime_synthetic.py` with the same estimator on synthetic data of the real
size (43 subjects × 27 windows) and 2 CPU threads, excluding checkpoint writes. Each notebook
measures again on its runtime (GPU if available) and the real graph run before training
(section 1), and shows the remaining time when resuming.

### Advanced: whole plan, other commands

`scripts/run_plan.py` runs every config of `configs/run_plan.yaml` in priority order
(`--status` shows progress). It is never started automatically: only the advanced notebook
calls it, and only after `RUN_EVERYTHING = True` is set deliberately.

```bash
python scripts/run_experiment.py --config configs/experiments/exp5_multiband.yaml --graphs <run dir> --dry-run
python scripts/run_experiment.py --config configs/experiments/exp5_multiband.yaml --graphs <run dir> \
    --settings within_D1 within_D2 --out /content/drive/MyDrive/AD-LEB/processed/experiments
```

`run_experiment.py --variants` matches whole name tokens (`plv` selects `plv__*`, not `ciplv__*`).
Cross-dataset settings whose gate fails are skipped with a message instead of aborting.
`--synthetic` runs any config on synthetic data, to test the framework only.
