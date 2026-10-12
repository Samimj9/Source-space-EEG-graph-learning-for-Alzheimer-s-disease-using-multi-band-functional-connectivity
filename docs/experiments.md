# Experiment framework

This document covers the code that runs Experiments 0–6 and the baselines. Nothing in this
repository has been run on the real EEG; every test uses synthetic data
(`adgraph.synthetic`). Evaluation follows `DESIGN.md`.

## 1. Data flow

Graph construction and the experiments are separate. Their only interface is the
graph-run directory (schema: `src/adgraph/graphs/store.py`).

```
scripts/build_graphs.py ──► processed/graphs_v3_<stamp>_<hash>/
                              graphs.npz  windows.csv  subjects.csv  manifest.json  [roi_timeseries/]
                                              │
scripts/run_experiment.py --config configs/experiments/<exp>.yaml --graphs <run dir>
   GraphStore.load ─► build_dataset (stateless, per window) ─► subject-level splits
   ─► per outer split: inner-CV hyperparameter selection ─► model.fit(train view)
   ─► predict(test view) ─► window probabilities ─► subject probability (mean) ─► metrics
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
| `psd_abs[i]`, `psd_log[i]`, `psd_rel[i]` | (4, 68) source ROI band power |
| `sensor_psd_*[i]` | (4, 30) electrode band power (Experiment 0) |
| `roi_timeseries/<key>.npy` | optional continuous (68, n_times) ROI signals; a window is `[:, start:stop]` |

PLV is the primary connectivity measure and ciPLV is the sensitivity analysis. They
are separate arrays and separate experiment variants, and the code never combines
them.

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
| 2 | `exp2_conn_minimal.yaml` | one band graph; edges PLV or ciPLV; constant node feature | GCN, GAT, GraphSAGE | 2 conn × 4 bands × 3 = 24 |
| 3 | `exp3_conn_psd.yaml` | one band graph; nodes = ROI PSD of all 4 bands {log, rel} (8) | GCN, GAT, GraphSAGE | 24 |
| 4 | `exp4_band_mean.yaml` | `A_mean` = mean of the 4 band matrices; nodes = ROI PSD (8) | GCN (configurable) | 2 conn |
| 5 | `exp5_multiband.yaml` | 4 band graphs; nodes = matched-band PSD (2) | GCN; shared / band-specific; mean / concat / gate / attention fusion | 2 × 2 × 4 = 16 |
| 6 | `exp6_multiband_psd_strength.yaml` | 4 band graphs; `X_i,b = [PSD_i,b (log, rel), strength_i,b]` (3) | GCN + attention fusion | 2 conn × 8 ablations = 16 |
| – | `baseline_age.yaml` | age | LR | 1 |
| – | `baseline_conn_classical.yaml` | upper-triangle PLV / ciPLV (4 × 2278 = 9112) | LR, SVM | 4 |
| – | `probe_dataset.yaml` | ROI PSD, target = dataset (pooled only) | LR | 1 |

Experiment 6 ablations: `full`, `no_psd`, `no_strength`, `no_learned_fusion` (mean
fusion), and `drop_<band>` for each band. Each runs for PLV and for ciPLV.

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
| GNN, single band (Exp 2, 3) | `x`, `adj` | (N, 1, 68, F), (N, 1, 68, 68); F = 1 (Exp 2), 8 (Exp 3) |
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
6. **Cross-dataset runs** are refused unless the graph run passes the DESIGN.md §7
   gate (`experiments/gate.py`).

## 5. Outputs (`<out>/<experiment>_<stamp>/<variant>/<setting>/`, never overwritten)

| File | Content |
|---|---|
| `predictions_windows.csv` | per window and repeat: identifiers, y, prob (+ learned band weights for multi-band models) |
| `predictions_subjects.csv` | subject probability per repeat |
| `metrics_repeats.csv` | subject-level accuracy, balanced accuracy, F1, sensitivity, specificity, ROC-AUC, MCC, confusion matrix, per repeat; pooled runs add per-dataset rows |
| `summary.json` | mean/SD/percentiles across repeats, repeat-averaged metrics with a subject bootstrap 95 % CI, window-level metrics, optional permutation test |
| `folds.csv`, `hparams.csv`, `manifest.json` | split assignment, selected hyperparameters with inner scores, provenance |

These files contain per-subject predictions, so they belong in Drive and are never
committed.

## 6. Commands (to be run later in Colab, on real graphs)

```bash
python scripts/run_experiment.py --config configs/experiments/exp5_multiband.yaml --graphs <run dir> --dry-run
python scripts/run_experiment.py --config configs/experiments/exp5_multiband.yaml --graphs <run dir> \
    --settings within_D1 within_D2 --out /content/drive/MyDrive/AD-LEB/processed/experiments
```

`--synthetic` runs any config on synthetic data, to test the framework only.
