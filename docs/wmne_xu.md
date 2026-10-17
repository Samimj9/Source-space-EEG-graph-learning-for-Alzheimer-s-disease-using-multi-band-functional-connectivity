# wMNE versus LCMV under the frozen Xu protocol (`D3_WMNE_XU`)

Version 1.0, 2026-10-17. Methodology log entry 55, decisions row 58. Purpose: determine whether the inverse solution
(wMNE vs LCMV) affects AD/HC classification, under the same participant-independent validation and the same models.
The LCMV suite (`docs/lcmv_xu.md`) is not modified: its stores, results, checkpoints and pin
(`D3_LCMV_XU/CODE_COMMIT.txt`, currently `299aed72`) are only read.

## 1. Two comparisons

| | A. Historical wMNE, validation-matched | B. Matched wMNE (primary) |
|---|---|---|
| Graph store | existing `graphsD3_v4_da723b33554c` (reused, read only) | new `D3_WMNE_XU/graphsWX_v1_<hash>` (mode B) and `graphsWX10_v1_<hash>` (mode A) |
| Windows | 20 s, 50 % overlap, the v4 boundary-aware grid | identical to LCMV: 10 s, 1 s step, the same five-minute selection and accepted windows |
| Bands | theta, alpha1, alpha2, beta | identical to LCMV: delta, theta, alpha, beta, gamma (Xu) |
| Inverse | wMNE | wMNE (same settings as A) |
| Experiments | existing D3 definitions Exp 0-7 (`configs/experiments/wx_hist/`) | LCMV definitions Exp 0-8, inherited unchanged (`configs/experiments/wx/`, `wx/tenfold/`) |
| Validation | frozen LCMV split and folds (verified) | frozen LCMV split and folds (verified) |
| Isolates the inverse? | no (windows, bands, model definitions differ) | yes (the inverse is the only intended difference) |

A tells how much of the earlier D3 numbers depended on the old validation protocol (`within_D3`, 10 × 5-fold on 65
participants) versus the frozen 52 / 13 protocol. B is the wMNE-versus-LCMV comparison.

The historical store supports the protocol directly: all 65 participants have status `ok`, with the same keys and
labels as the LCMV stores (checked on Drive, 2026-10-17). The frozen split and folds depend only on the participant
table, so they are identical. This is verified at run time (§3). Exp 8 has no historical counterpart: its primary
variants need DE features, which the historical store does not hold, and it is defined for the five Xu bands.

## 2. Matched wMNE dataset (comparison B)

Built by the same `lcmvxu` code as the LCMV stores (`scripts/wx_build.py`, `configs/wmne_xu/pipeline.yaml`).

**Configuration.** The configuration is identical to `configs/lcmv_xu/pipeline.yaml` except for:
- `output_subdir: D3_WMNE_XU`;
- `store_names`;
- `source`.

`tests/test_wmne_xu.py` enforces this.

**Shared with LCMV:**
- the inputs (ds004504 derivatives);
- boundary splitting and per-segment resampling;
- the exact five-minute selection, which never exceeds 300 s (entry 53);
- the 3.3 s margin;
- 10 s / 1 s windows;
- the adaptive peak-to-peak rejection;
- the Xu bands and band-pass filters;
- the PSD, DE, PLV and |ciPLV| definitions;
- the fsaverage ico-5 source space, the 3-layer BEM, `mindist` and the electrode positions;
- the 68 Desikan-Killiany labels in sorted order, with `mean_flip`;
- mode A (first 10 accepted windows) and mode B;
- `sample_weight` = 1 / windows of the participant;
- the frozen split.

**Inverse.** The settings are the historical wMNE ones (`configs/pipeline_d3.yaml`):
- depth-weighted minimum norm (`MNE`, depth 0.8);
- loose orientation 0.2 on the free-orientation forward;
- lambda2 = 1/9;
- ad-hoc diagonal noise covariance;
- the signed normal component.

The resulting 68 × 19 operator uses no recording. It is identical for all participants, who share one electrode
geometry. On every participant it is checked against MNE's `apply_inverse_raw` (relative error ≤ 1e-6). LCMV
differs by definition: it uses a per-participant data covariance and the fixed-normal forward.

**Differences that cannot be eliminated:**
1. The inverse itself, including its orientation model (loose 0.2 vs fixed normal). This is the comparison.
2. The LCMV covariance of D3-23 and D3-58 kept the one clean window that was removed by the `corr1` correction
   (entry 53). That window affects nothing in wMNE.
3. Absolute source PSD is on different scales (beamformer units vs nA·m). Log, relative, DE, PLV and |ciPLV| are
   comparable.

**Match.** `wx_build.py match` compares each matched store with the corrected LCMV store of its mode
(`lcmvxu.match`), checking that the following are identical:
- the configuration, apart from the inverse;
- participants, labels and status;
- the selections: durations, pieces, exact segment tables and cuts;
- the window counts;
- every window row: id, participant, start / stop sample, segment, piece and tenfold flag;
- the weights;
- `xu_split.json`;
- bands, ROI order and channels.

It also checks that the sensor features agree to float32 precision. Those features do not involve the inverse, so
this is the Exp 0 consistency control. Source arrays differ, and only their summaries are reported. The result is
written to `match.json`.

**Outputs:** `<processed>/D3_WMNE_XU/` (§6). Per-participant checkpoints make the build resumable after a Colab
disconnect. Completed participants are skipped. A checkpoint from another code version is never silently recomputed:
`build` refuses unless `--allow-recompute` is given.

## 3. Validation protocol (both comparisons)

- **Families:** `WX` (matched, mode B), `WX10` (matched, mode A) and `D3H` (historical), with settings `xu_dev_cv`
  and `xu_holdout`.
- **Pre-run check.** Before any unit runs, `scripts/experiment.py` calls `lcmvxu.protocol.verify` against the
  validated LCMV store of the same mode. It requires:
  - the same 65 participants and labels;
  - `xu_partition` = the saved 52 development / 13 holdout participants;
  - every outer split of `xu_dev_cv` (5 folds) and `xu_holdout` with the same training and test participants as on
    the LCMV store, built exactly as the runner builds them, with the config's k, repeats and seed;
  - no participant on both sides of a split;
  - no holdout participant in a development fold.
- **Matched stores** additionally need a passed `validation.json` and a passed `match.json` for their current
  `graph_run_id` and the current LCMV reference.
- **Unchanged from the LCMV experiments:**
  - training-only scaling and class weights;
  - the classical models' inner CV, on training participants only;
  - the fixed 60-epoch GNN schedule with no early stopping;
  - seeds;
  - subject-level aggregation (the mean window score) and the reporting.
- The holdout is never used for selection, checkpointing or stopping.
- A later 200-epoch / early-stopping protocol must be added to both suites together, never to wMNE alone.

## 4. Experiments

| Mode | Configs | Units (variants × 2 settings) |
|---|---|---|
| B, matched | `configs/experiments/wx/wx_exp{0..8}_*.yaml` | Exp 0: 8, Exp 1: 8, Exp 2: 20, Exp 3: 20, Exp 4: 8, Exp 5: 16, Exp 6: 20, Exp 7: 12, Exp 8: 20 |
| A, matched | `configs/experiments/wx/tenfold/wx10_exp{0..8}_*.yaml` | as mode B |
| historical | `configs/experiments/wx_hist/wxh_exp{0..7}_*.yaml` | Exp 0: 4, Exp 1: 4, Exp 2: 16, Exp 3: 16, Exp 4: 4, Exp 5: 8, Exp 6: 4, Exp 7: 12 |

Each `xu_dev_cv` unit has 5 folds, and each `xu_holdout` unit has 1 fit. The matched configs `extend` the LX configs
and change only `experiment` and `graph_family`, so the definitions cannot drift (tested). Exp 8 is part of this
suite under the suite's own pin; its architecture code is shared and unchanged.

## 5. Comparison report

`scripts/wx_compare.py` (notebook `WX_99_compare`) reads completed results only and pairs LCMV and matched wMNE by
mode × experiment × variant × setting. For each pair it reports participant-level ROC-AUC, balanced accuracy and F1,
window-level ROC-AUC, the wMNE − LCMV differences, and a same-participants flag. Historical results are reported
separately. The report is descriptive.

## 6. Colab workflow (does not interfere with the running LCMV experiments)

| Order | Notebook | Runtime |
|---|---|---|
| 1 | `notebooks/wmne_xu/WX_00_build_dataset.ipynb` | CPU, standard RAM |
| 2 | `notebooks/wmne_xu/WX_Exp0_sensor_psd_de.ipynb` … `WX_Exp8_xu_dual_adjacency.ipynb` (one at a time) | Exp 0/1 CPU; Exp 2-8 T4 GPU |
| 2b (optional) | `notebooks/wmne_xu/historical/WXH_Exp0_*.ipynb` … `WXH_Exp7_*.ipynb` | Exp 0/1 CPU; Exp 2-7 T4 GPU |
| 3 | `notebooks/wmne_xu/WX_99_compare.ipynb` | CPU |

- Every WX notebook uses the pin `D3_WMNE_XU/CODE_COMMIT.txt` (`--family WX`, flag `UPDATE_WX_CODE`). The first WX
  notebook run records the branch head there.
- The LCMV pin and the LCMV notebooks are not touched. A WX session can run beside an LCMV session, because their
  folders and pins are separate.
- Do not move the LCMV pin while LCMV experiments run.

**Drive layout:**

```
<processed>/D3_WMNE_XU/
  CODE_COMMIT.txt
  graphsWX_v1_<hash>/   graphsWX10_v1_<hash>/     (+ _limit1 smoke-test runs)
      subjects/<key>/ graphs.npz windows.csv subjects.csv manifest.json xu_split.json qc_report.csv
      validation.json match.json
  cache/                                           wMNE forward model, input hashes
  experiments_A_xu_tenfold/wx10_exp*/              experiments_B_all_windows/wx_exp*/
  experiments_H_historical_graphs/wxh_exp*/
  reports/compare_<time>/
```

**Expected cost:**
- Build: the full LCMV build took about 40 minutes on CPU for 65 participants (Drive timestamps). The wMNE build does
  the same per-participant work without the LCMV fit, plus one forward model and inverse computed once. Expect about
  30-45 minutes on a standard CPU runtime.
- Peak RAM: one participant at a time, about 1-2 GB, plus assembly of about 3.5 GB for mode B. `match` needs about
  6 GB.
- Drive space: about 5 GB (2.3 GB `graphs.npz` for mode B, plus checkpoints and ROI time series).
- Experiments: the same compute as the corresponding LX notebooks.
