# Dataset 3 (D3): OpenNeuro ds004504, AD vs HC

Version 1.0, 2026-10-14. Methodology log entry 44, decisions row 52, DESIGN.md v1.7.
Code: `src/adgraph/io/bids.py`, `src/adgraph/io/openneuro.py`, `configs/pipeline_d3.yaml`,
`configs/experiments/d3/`, `scripts/d3_data.py`. Notebooks: `notebooks/d3/`. Tests: `tests/test_d3.py`.

## 1. Source and cohort

- **Dataset:** Miltiadous et al. (2023), "A dataset of EEG recordings from: Alzheimer's disease,
  Frontotemporal dementia and Healthy subjects", OpenNeuro **ds004504**, release 1.0.9 (2026-05-20),
  doi:10.18112/openneuro.ds004504.v1.0.9, licence CC0. Data descriptor: *Data* 8(6):95,
  doi:10.3390/data8060095.
- **Recording:** AHEPA University Hospital, Thessaloniki; Nihon Kohden EEG 2100; eyes closed, sitting;
  19 scalp electrodes (10-20: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2);
  500 Hz; about 13.5 min (AD, 5.1–21.3) and 13.8 min (HC, 12.5–16.5).
- **Used:** groups A (AD, **n = 36**) and C (healthy controls, **n = 29**) only. Group F (frontotemporal
  dementia, n = 23) is **excluded by design** (the study question is AD vs HC). It is never downloaded,
  and its keys are recorded in each D3 graph run's manifest (`registry.excluded_by_design`). No other
  participant is excluded; the only exclusion rule remains "fewer than 5 clean windows".

| Group | n | Age, mean ± SD (range) | MMSE, mean ± SD | Female |
|---|---|---|---|---|
| AD | 36 | 66.4 ± 7.9 (49–79) | 17.8 ± 4.5 | 24 |
| HC | 29 | 67.9 ± 5.4 (57–78) | 30.0 ± 0.0 | 11 |

Unlike D2, D3's AD and HC groups have similar ages; the sex ratio differs between groups. These figures
come from the public `participants.tsv` and are aggregates only.

## 2. Data used: the provider's preprocessed derivatives

`derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set` (EEGLAB). Provider preprocessing, per the dataset
README:
1. Butterworth band-pass 0.5–45 Hz;
2. re-referencing to A1–A2;
3. Artifact Subspace Reconstruction (EEGLAB);
4. ICA (RunICA), removing components that ICLabel classified as eye or jaw artifacts.

## 3. Processing in this project (`configs/pipeline_d3.yaml`)

Author decision: use the provider's cleaning as given and add only what our pipeline needs.

| Step | D1 / D2 (frozen v4) | D3 |
|---|---|---|
| Input | EDF, 30 of 32 channels | EEGLAB `.set` derivatives, all 19 channels |
| Edge crop | 10 s per side before any processing | same |
| Resampling | 250 Hz (FFT, `reflect_limited`) | same |
| Sensor band-pass | 0.1–45 Hz FIR (8251 taps) | **none** (provider 0.5–45 Hz Butterworth; nothing filtered twice) |
| Artifact cleaning | none at the sensor level | provider ASR + ICA (as released) |
| Bad channels | rules evaluated on the analysed interval, flagged channels interpolated | same rules **evaluated and reported only** (`qc_channels.csv`); no interpolation added. A **confirmed flat** channel (std of the analysed 1 Hz high-passed signal < 0.1 µV, `flat_std_v`) stops that subject with `FlatChannelError` naming subject and channel; nothing is interpolated or excluded automatically |
| Reference | common average (projection) | common average (from A1–A2) |
| Amplitude checks | unit safeguard + plausibility | same |
| Boundary margin | 4125 + 412 = 4537 samples (18.148 s) | analytic kernel only: **412 samples (1.648 s)**, since no sensor FIR is applied |
| Windows / rejection / minimum | 20 s, 50 % overlap; adaptive peak-to-peak rule; ≥ 5 windows | identical |
| Source model | fsaverage ico-5, MNE (λ² = 1/9, loose 0.2, depth 0.8, normal), DK 68 ROIs, `mean_flip` | identical (operator 68 × 19, built and verified for the 19-channel montage) |
| Bands, PLV, \|ciPLV\|, PSD | theta 4–8, alpha1 8–10, alpha2 10–13, beta 13–30 Hz; analytic FIR; Welch | identical |

How runs and results are kept apart:
- **Graph runs:** `<processed>/graphsD3_v4_<config hash>/`. The D1/D2 selection only ever considers
  `graphs_v*` runs, so the two families never mix.
- **Experiment results:** `<processed>/experiments_D3/`. Each results root therefore holds results from a
  single graph run.
- **Code pin:** D3 has its own pin, `<processed>/D3_CODE_COMMIT.txt`; the D1/D2 pin is untouched.

## 3b. Real-data checks before the full build

`scripts/d3_verify_smoke.py` (notebook D3_01 step 3b) runs on the one-subject smoke test. Its inputs are
selected explicitly: the smoke run and the operator recorded in its manifest, and the frozen D1/D2 run by
`graph_run_id`. It checks:
1. electrode labels and positions;
2. no flat channel in the reconstruction;
3. operator 68 × 19 with the configured channels, verified against the full MNE inverse;
4. ROI names and order identical to D1/D2;
5. no discontinuities.

The full build cell runs only after `CONFIRM_D3_FULL_BUILD = True` is set.

## 4. Evaluation

- **Setting `within_D3`:** repeated stratified subject-level 5-fold CV, with the same seeds and rules as
  S1/S2. Classical models use 20 repeats with the nested C grid; GNNs use the fixed-hyperparameter
  protocol with 10 repeats. All windows of a participant always stay in one partition.
- **Experiments run first** (author choice):
  - Exp 0 and Exp 1 (LR, SVM);
  - the age-only baseline;
  - Exp 6 full multi-band GCN (PLV, |ciPLV|);
  - Exp 7 EARM-GNN with its two ablations.

  Configs are in `configs/experiments/d3/`.
- **Cross-dataset tests D3 ↔ D1/D2 are planned but not yet implemented.** They require:
  - a store combining the frozen D1/D2 run with the D3 run, in source space only, since the sensor
    sets differ (30 vs 19 channels);
  - a revision of the DESIGN.md §7 gate. Its rule of "identical preprocessing" cannot hold: D3 uses the
    provider's ASR/ICA cleaning and 19 electrodes.

  Until then no D3 cross-dataset result exists.

## 5. Limitations

- **Different preprocessing:** D3 uses the provider's cleaning (ASR, ICA with ICLabel), D1/D2 our own.
  Connectivity and PSD differences between cohorts can therefore reflect preprocessing as well as
  physiology.
- **Fewer electrodes:** with 19 electrodes, source estimates on the 68 ROIs are spatially coarser than
  with 30 channels, and neighbouring ROIs share more of the same sensor information. |ciPLV| is the
  leakage-robust sensitivity analysis.
- **Filtering before our crop:** the provider filtered and cleaned whole recordings before release. Our
  10 s crop and margin cover our own filters, not the provider's.
- **Possible discontinuities:** if the provider's cleaning removed data segments, the `.set` files
  contain EEGLAB "boundary" events. `scripts/d3_data.py check` counts them per recording. The pipeline
  does not yet exclude windows that span one: if any are found, that is reported and decided before
  graphs are built.
- **Relative-PSD range:** the provider's Butterworth low-pass is about −3 dB at 45 Hz, the top of our
  relative-PSD range [1, 45) Hz.
- **Literature on this dataset:** many published results on ds004504 split windows rather than subjects.
  The dataset authors' AHEPA benchmark (Miltiadous et al., 2026, *Cognitive Neurodynamics*,
  doi:10.1007/s11571-026-10464-w) reviews those results and proposes a standardized reporting checklist.
  Our subject-level protocol avoids window leakage, and the paper should follow that checklist and cite
  the dataset as its authors request.
