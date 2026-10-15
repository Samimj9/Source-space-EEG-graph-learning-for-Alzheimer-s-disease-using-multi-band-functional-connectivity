# Dataset 3 (D3): OpenNeuro ds004504, AD vs HC

Version 1.2, 2026-10-15. Methodology log entries 44–46, decisions rows 52–54, DESIGN.md v1.7.
Code: `src/adgraph/io/bids.py`, `src/adgraph/io/openneuro.py`, `configs/pipeline_d3.yaml`,
`configs/experiments/d3/`, `scripts/d3_data.py`, `src/adgraph/preprocessing/discontinuities.py`,
`src/adgraph/io/chanlocs.py`. Notebooks: `notebooks/d3/`. Tests: `tests/test_d3.py`, `tests/test_d3_discontinuities.py`,
`tests/test_d3_coordinates.py`.

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
| Discontinuities | none in the EDFs | split at every EEGLAB `boundary` event; each continuous segment resampled on its own (§3a) |
| Resampling | 250 Hz (FFT, `reflect_limited`) | same |
| Sensor band-pass | 0.1–45 Hz FIR (8251 taps) | **none** (provider 0.5–45 Hz Butterworth; nothing filtered twice) |
| Artifact cleaning | none at the sensor level | provider ASR + ICA (as released) |
| Bad channels | rules evaluated on the analysed interval, flagged channels interpolated | same rules **evaluated and reported only** (`qc_channels.csv`); no interpolation added. A **confirmed flat** channel (std of the analysed 1 Hz high-passed signal < 0.1 µV, `flat_std_v`) stops that subject with `FlatChannelError` naming subject and channel; nothing is interpolated or excluded automatically |
| Reference | common average (projection) | common average (from A1–A2) |
| Amplitude checks | unit safeguard + plausibility | same |
| Boundary margin | 4125 + 412 = 4537 samples (18.148 s) | analytic kernel only: **412 samples (1.648 s)**, since no sensor FIR is applied |
| Windows / rejection / minimum | 20 s, 50 % overlap; adaptive peak-to-peak rule; ≥ 5 windows | identical |
| Electrode positions | 30-channel `.loc` layout → `standard_1020` template | the coordinates stored in each `.set` file, frame from the provider's lookup record (§3b) |
| Source model | fsaverage ico-5, MNE (λ² = 1/9, loose 0.2, depth 0.8, normal), DK 68 ROIs, `mean_flip` | identical (operator 68 × 19 per distinct electrode geometry, built and verified) |
| Bands, PLV, \|ciPLV\|, PSD | theta 4–8, alpha1 8–10, alpha2 10–13, beta 13–30 Hz; analytic FIR; Welch | identical |

How runs and results are kept apart:
- **Graph runs:** `<processed>/graphsD3_v4_<config hash>/`. The D1/D2 selection only ever considers
  `graphs_v*` runs, so the two families never mix.
- **Experiment results:** `<processed>/experiments_D3/`. Each results root therefore holds results from a
  single graph run.
- **Code pin:** D3 has its own pin, `<processed>/D3_CODE_COMMIT.txt`. The D3 setup cell
  (`colab_setup.py --family D3`) never reads it for checkout nor writes `CODE_COMMIT.txt` (the D1/D2 pin).
  `notebooks/advanced/Restore_D1D2_code_pin.ipynb` restores the D1/D2 pin if a D3 session moved it earlier.

## 3a. Discontinuities (EEGLAB `boundary` events; methodology log entry 45)

The provider's cleaning removed data segments in 56 of the 65 recordings (D3-56: 56 events). MNE reads each
event as an annotation at (latency − 1)/srate; its duration is the number of removed samples, which are not in
the file. Processing:
1. the recording is cut before sample ⌈onset·srate⌉ of every boundary inside the analysed interval;
2. each continuous segment is trimmed to an even length and resampled on its own (fixed 4 s padding);
3. windows use the same 20 s / 10 s grid inside each segment, at least 412 samples (1.648 s) from both segment
   ends; segments shorter than 23.3 s give no window;
4. reference, source operator, analytic FIR, rejection, PLV, |ciPLV| and PSD are unchanged.

No recording is excluded for having boundaries. Per participant, `subjects.csv` reports boundary events,
segments (all / usable), the windows of the continuous grid and how many of them would have crossed a cut,
windows excluded for a discontinuity, candidates, windows rejected by QC and accepted windows.
`discontinuity_samples` holds the cut positions on the 250 Hz time line.

## 3b. Electrode coordinates (methodology log entry 46)

The montage is built from each file's `EEG.chanlocs`, never from a template chosen by the pipeline. What the 65
files contain (inspected from the released files):
- 19 channels with finite 3D coordinates in mm, EEGLAB axes (nose +X), no fiducials;
- identical in all 65 files;
- assigned by the provider with `pop_chanedit(..., 'lookup', '...dipfit/standard_BEM/elec/standard_1005.elc')`;
- equal to MNE's Colin27 template (≤ 7·10⁻¹⁴ mm).

They are therefore a **shared template, not individual measurements**. Their frame (MNI/Colin27) follows from
that lookup record. They are placed in the `mri` frame with the fsaverage fiducials, as MNE does for its Colin27
templates.

Coordinates that are missing, non-finite, inconsistent with the units or orientation, or without an accepted
lookup record stop the subject (`CoordinateError`); there is no template fallback. Operators are built per
distinct geometry. Each subject records `coord_source`, `coord_frame`, `coord_lookup`, `montage_hash`,
`coord_template_deviation_mm` and `operator_id`.

MNE's EEGLAB reader alone would label these MNI coordinates as the head frame and misplace every electrode by
46–57 mm. The pipeline therefore never uses that montage.

## 3c. Real-data checks before the full build

`scripts/d3_verify_smoke.py` (notebook D3_01 step 3b) runs on the one-subject smoke test. Its inputs are
selected explicitly: the smoke run and the operator recorded in its manifest, and the frozen D1/D2 run by
`graph_run_id`. It checks:
1. electrode labels and the coordinates stored in the `.set` file (frame established, operator built from them,
   head → MRI transformation verified, every subject on its own geometry's operator);
2. no flat channel in the reconstruction;
3. operator 68 × 19 with the configured channels, verified against the full MNE inverse;
4. ROI names and order identical to D1/D2;
5. discontinuities are handled: on the smoke subject and, in memory, on the recording with the most boundary
   events, ZERO accepted windows cross a discontinuity or its margin, and every window lies inside one segment
   at least the margin from its ends. Boundary events themselves are allowed;
6. PSD, PLV and |ciPLV| finite, in range and symmetric.

Provenance (coordinate source, frame, montage hash, operator id) is saved in `d3_montage_checks.json`.

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
- **Template electrode positions:** the electrode coordinates supplied with ds004504 are the Colin27 template
  (EEGLAB lookup), identical for all participants, on a template head (fsaverage). Individual head and electrode
  geometry is not available, so source-space estimates carry template-geometry error for every participant.
- **Fewer electrodes:** with 19 electrodes, source estimates on the 68 ROIs are spatially coarser than
  with 30 channels, and neighbouring ROIs share more of the same sensor information. |ciPLV| is the
  leakage-robust sensitivity analysis.
- **Filtering before our crop:** the provider filtered and cleaned whole recordings before release. Our
  10 s crop and margin cover our own filters, not the provider's.
- **Discontinuities:** the provider's cleaning removed data segments (EEGLAB "boundary" events in 56 of
  65 recordings). Windows never cross them (§3a), but the removed data reduce the usable recording time, and
  short segments between close boundaries contribute no window. The amount differs between participants and is
  reported per participant.
- **Relative-PSD range:** the provider's Butterworth low-pass is about −3 dB at 45 Hz, the top of our
  relative-PSD range [1, 45) Hz.
- **Literature on this dataset:** many published results on ds004504 split windows rather than subjects.
  The dataset authors' AHEPA benchmark (Miltiadous et al., 2026, *Cognitive Neurodynamics*,
  doi:10.1007/s11571-026-10464-w) reviews those results and proposes a standardized reporting checklist.
  Our subject-level protocol avoids window leakage, and the paper should follow that checklist and cite
  the dataset as its authors request.
