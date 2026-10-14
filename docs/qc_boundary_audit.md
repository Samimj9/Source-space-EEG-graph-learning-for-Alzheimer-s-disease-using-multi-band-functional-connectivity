# Dataset-wide QC audit: bad channels and connectivity boundary sensitivity

Status: **plan and pre-specified decision criteria, written 2026-10-14 before any dataset-wide
result was seen** (methodology log entry 40). Revised the same day, still before any result, at the
author's request; see "Revision 1" below. The audit is read-only. The production code stays at
commit `7fd5a69`, and no graph, operator, checkpoint or experiment is modified. Nothing is changed in
the pipeline until the author has seen the results and approved a decision.

Tool: `scripts/audit_quality_boundary.py`. In Colab it is read from its own commit (see "Running
it") while the production working tree stays at `7fd5a69`.

## 1. What is already known

Real data, author's runs at `7fd5a69`:

| | D1-1 | D2-2 |
|---|---|---|
| A vs B PLV p95 / max | 0.00035 / 0.02747 | 0.00438 / 0.21479 |
| Max subject-mean difference | 0.00050 | 0.02036 |
| Most sensitive band | | theta |
| Interpolated channels | none | F7, FC1, T7, T8, CP5 |
| Edge transient | start (block std ~1030 µV) | end (~826 µV) |

- Both recordings reproduced the stored smoke-test arrays exactly.
- F7, T7 and CP5 are neighbours along the left temporal chain.

Synthetic ground truth (methodology log entry 39 and the sensitivity analysis of 2026-10-14):
- The production FFT Hilbert transform makes accepted windows depend on data outside the analysed
  interval, at **both** ends (circular FFT), most in the first and last window.
- B (reflected edges) is insensitive to edge transients, but its own reflection changes the first and
  last window by about 0.02–0.025 PLV. That is 4× the true contamination of A by a 3 mV transient.
- **From the second window inward, A − B tracked the true contamination of A** (e.g. 0.0028 vs
  0.0026).
- So differences in boundary-touching windows measure sensitivity to the *choice of boundary
  handling*, not proven contamination.
- Differences in interior windows measure the dependence of retained windows on non-analysed data.

## 2. Design

### Units and verification
- The unit of every comparison is the **recording/subject**: 20 D1 + 23 D2. Windows are never treated
  as independent observations.
- Production functions are used unchanged: `load_raw`, `preprocess`, `process_recording`, the cached
  operator (read-only, refused if absent), `band_analytic` and `phase_sync`.
- Where stored outputs exist (the smoke-test units D1-1, D2-2), the audit verifies:
  - the per-channel QC table;
  - the bad channels;
  - the accepted-window count;
  - the PLV and |ciPLV| arrays (exact equality expected).
- The Part 2 run must find the same bad channels as Part 1.

### Part 1: bad channels (load + preprocess only; no source model)
Per recording:
- interpolated channels and the reason (flat / amplitude z / low correlation);
- the channels whole-recording detection would have flagged (QC only);
- amplitude statistics, accepted windows;
- edge-transient severity of the final signal: RMS ratio of the discarded 10 s to the analysed
  interval, start and end, and peak ratio;
- the largest cluster of neighbouring interpolated electrodes (neighbours = montage distance ≤ 1.5 ×
  the median nearest-neighbour distance; 2–6 per electrode);
- the fraction of the interpolated channels' neighbours that were interpolated as well.

Recordings whose preprocessing fails are recorded as QC findings, not errors.

Report 1:
- the per-recording table, with every field saved (`per_recording.csv`);
- per-channel detection frequency by dataset, with AD/HC counts and reasons (`channel_frequency.csv`);
- the distribution of interpolated counts;
- AD vs HC within each dataset and D1 vs D2 (`group_comparisons.csv`), for: interpolated count, largest
  cluster, edge severity, amplitude. Each comparison gives:
  - medians, the median difference with a 95 % bootstrap CI (subjects resampled within each group),
    the rank-biserial effect size and the Mann–Whitney p;
  - for AD vs HC combined over datasets: the dataset-weighted mean difference with a stratified
    bootstrap CI and a permutation p (labels permuted within dataset);
- systematic electrodes;
- where D2-2 sits in the distribution and which recordings share its channels.

### Part 2: boundary sensitivity, A vs B, identical accepted windows
- **A** = production.
- **B** = the 10 s outside the analysed interval replaced by a mirror image of the adjacent analysed
  data. B has the same length, windows, band filters, Hilbert transform and PLV/ciPLV formulas.
- Per recording, for PLV and |ciPLV|, it reports:
  - |A − B| median / p95 / max over windows × bands × ROI pairs;
  - the same per band;
  - by window position (first / middle / last accepted) and by distance from the boundary;
  - the same statistics excluding boundary-touching windows (`excl_boundary`, distance ≥ 1) and the
    next ones (`excl_boundary2`, distance ≥ 2);
  - the max subject-mean difference (the mean over the accepted windows of A minus that of B);
  - the 10 largest (band, ROI pair) subject-mean differences;
  - the band-limited ROI edge severity (RMS ratio per band and side).

Report 2 (every table is saved as CSV in the report folder, so no conclusion rests on categories
alone):
- per recording, every statistic for PLV and |ciPLV| (`per_recording.csv`):
  - all windows, excluding boundary-touching windows, and excluding the next ones;
  - median / p95 / max of |A − B| and the subject-mean max;
- per recording and band (`per_band.csv`);
- the per-window values by distance from the boundary (`per_window_by_distance.csv`) and by position
  (`by_position.csv`);
- Spearman associations, across subjects, with edge severity and interpolated count
  (`associations.csv`);
- D1 vs D2 and AD vs HC at the subject level, with effect sizes, CIs and p (`group_comparisons.csv`);
- recordings flagged for investigation, with their bands and ROI pairs (`flagged_recordings.csv`);
- ROI concentration;
- **the normalised boundary metric (§3.2), always next to the absolute differences:**
  - `normalised_per_recording.csv`, `normalised_per_recording_band.csv`;
  - the denominators per dataset and band (`between_subject_sd.csv`, with the share of floored
    cells);
  - its group comparisons (`normalised_group_comparisons.csv`).

The per-recording units (`part1/<key>/`, `part2/<key>/`: JSON, CSV, NPZ) keep the raw per-window,
per-band and per-ROI-pair quantities for any further analysis.

### Resumability, storage, cost
- **Checkpointing:**
  - one checkpointed unit per recording and part, with an atomic payload and the completion marker
    written last (`adgraph.checkpoint`);
  - the identity is the audit version, file name, size, mtime, production commit and config hash;
  - rerunning skips completed recordings; reports never recompute anything.
- **Outputs** are under `<--out>/qc_boundary_audit_7fd5a69/`:
  - `part1/<key>/`, `part2/<key>/`;
  - `reports/report{1,2}_<UTC time>/`, never overwritten;
  - `manifest.json`.
- **Estimates**, measured on synthetic recordings (5–10 min, 1000/1024 Hz), with Colab expected to be
  up to ~2× slower:

| | Part 1 | Part 2 |
|---|---|---|
| CPU time per second of EEG | 0.015–0.04 s | 0.06–0.15 s |
| Peak RAM per recording | < 1 GB | ~1–1.5 GB for a 10 min recording |

  - For 43 recordings of ~4.6 h of EEG: Part 1 ~5–15 min, Part 2 ~20–45 min, plus Drive reading of
    ~1.3 GB of EDF per part.
  - Storage is ~0.3 MB per recording (~15 MB) plus ~2 MB per report.
  - `estimate` prints the exact durations before anything runs.

## 3. Pre-specified criteria (fixed before results; never tuned on classification performance)

**Principles.**
- The recommendation weighs the **magnitudes** of the effects: absolute differences in PLV/|ciPLV|
  units, normalised differences, their uncertainty and their distribution over recordings, bands and
  ROI pairs.
- Thresholds and categories are **provisional QC flags** that direct attention. They are not validated
  physiological cut-offs and do not trigger decisions on their own.
- A p < 0.05 marks a comparison **for investigation**. It never triggers a rebuild, an exclusion or a
  montage change by itself.
- Every group comparison is reported with p, effect size (rank-biserial; difference in medians or
  means), 95 % bootstrap CI and the practical magnitude in the metric's own units.
- No decision is made by the audit tool. Every change needs the author's approval.

### 3.1 Bad channels (Part 1)
- **E1 Systematic electrode:** interpolated in ≥ 30 % of the recordings of a dataset. This is a
  **trigger to test** a reduced common montage, never automatic removal (§3.3).
- **R1 Recording with too many interpolated channels:** more than 20 % of the 30 channels (≥ 7). This
  is the existing `bad_channels.max_fraction_warn`.
- **R2 Interpolation over a gap:** a cluster of ≥ 3 neighbouring interpolated electrodes. Spherical
  splines are least reliable where neighbours are missing.
- **R3 Existing rule:** fewer than 5 clean windows, the only pre-specified exclusion in DESIGN.md.
- **G1 Group imbalance in interpolation:** the AD − HC difference within dataset, reported with effect
  size and CI. Flagged for investigation as a potential confound when p < 0.05 or when the CI excludes
  0 with a difference of ≥ 1 channel. Never used to exclude.

### 3.2 Boundary sensitivity (Part 2)
- **Absolute (always reported):** |A − B| in PLV/|ciPLV| units, per recording and band:
  - all windows and the interior windows (distance ≥ 1, ≥ 2);
  - median / p95 / max;
  - the subject-mean difference (the quantity that enters subject-level analyses).
- **Normalised (always reported alongside):** for each (band, ROI pair),
  ratio = |subject-mean A − subject-mean B| / max(SD, floor):
  - SD is the between-subject SD of the subject-mean A, **computed within the recording's own dataset**
    (D1 recordings use the D1 SD, D2 recordings the D2 SD), so a D1/D2 difference in spread cannot
    distort the ratio;
  - **denominator floor 0.005** (PLV/|ciPLV| units). Connections that barely vary between subjects
    (e.g. |ciPLV| near zero) would otherwise turn negligible absolute differences into large ratios;
  - the share of floored cells is reported per dataset and band, and where it is large the absolute
    values take precedence;
  - the recording's own value contributes to its dataset's SD (20 and 23 subjects).
  - It is summarised per recording by the median, p95 and max over cells, and per band.
- **Provisional QC categories** (per-recording ratio p95): ≤ 0.10, 0.10–0.25, > 0.25. They are QC
  flags fixed before the results, not validated cut-offs. They describe the distribution, and the
  recommendation does not depend on them alone.
- **Dataset or diagnosis dependence:** D1 vs D2 and AD vs HC (within dataset) on the absolute and
  normalised per-recording statistics, with effect size, CI and p. p < 0.05 is a flag for
  investigation.
- **Interior persistence flag:** p95 of |A − B| at distance ≥ 2 above 0.005 in a recording.
- **Unusual-recording flag** (for investigation): robust z of log(PLV subject-mean max) > 3.5, or PLV
  subject-mean max > 0.01, or PLV max excluding boundary-touching windows > 0.05. D2-2 (0.0204) already
  meets the second condition.

### 3.3 How the evidence maps to the options (for the author's decision; nothing is automatic)

**Interpolation**
- **A (keep the 30-channel montage with subject-specific interpolation)** is supported when no
  electrode is systematic, R1/R2 flags are rare and the interpolation burden is similar between AD and
  HC within dataset (small effect, CI around 0).
- **C (test a reduced common montage)** is warranted when an E1 electrode exists. The test must:
  - drop **identical retained channels in both datasets**, even if the electrode is systematic in only
    one;
  - use a **source operator recomputed** for exactly that channel set (forward model, inverse, ROI
    operator, verification);
  - be run first as a sensitivity analysis on the same subjects.

  Whether the reduced montage becomes primary is decided by the author from signal-quality evidence,
  never from classification results.
- **B (exclude recordings)** is considered only for recordings meeting R1 or R3, with a logged reason
  and a DESIGN.md amendment, and always accompanied by the all-recordings analysis. R2 alone motivates
  a sensitivity analysis, not exclusion.

**Boundary**
- **E (retain the production pipeline and report this audit as a sensitivity analysis)** is supported
  when absolute subject-mean differences are small relative to the between-subject spread for nearly
  all recordings. Typically most recordings would sit in the lowest provisional category, with no
  practically relevant D1/D2 or AD/HC difference and no interior persistence.
- **D (revise the boundary handling and regenerate the graphs)** is warranted by **material
  magnitudes**, not by p-values:
  - large normalised or absolute subject-mean differences in a substantial share of recordings, or in
    the interior windows;
  - a dataset- or diagnosis-related difference whose effect size and CI indicate a practically relevant
    magnitude.

  The new method would be chosen on synthetic ground truth and boundary diagnostics, never on
  classification results, and graphs are regenerated only after the author's approval.
- **Intermediate evidence:** E plus a pre-specified secondary analysis without boundary-touching
  windows (computing it needs the author's approval).

**General**
- Interpolation and boundary sensitivity are separate problems assessed on separate evidence. Channel
  removal (C) is not assumed to change boundary sensitivity, and vice versa.
- No electrode, recording or preprocessing choice is made from classification performance on any test
  set.

**Revision 1 (2026-10-14, before any dataset-wide result, author's request).**
- p-values are flags for investigation, never automatic rebuild triggers.
- Effect sizes, CIs and practical magnitudes are reported with every comparison.
- The normalising SD is computed within each dataset, with a documented floor (0.005); absolute and
  normalised values are reported together.
- 0.10/0.25 are provisional QC categories, not validated cut-offs.
- The 30 % bad-channel criterion triggers a reduced-montage **test** with identical channels in both
  datasets and a recomputed operator.
- All per-recording and per-band metrics are preserved and saved.

### 3.4 Implications to check for the evaluation settings (DESIGN.md)
- **Pooled:** an artifact that differs by dataset is harmless only if it is not also associated with
  diagnosis. G1 and the AD vs HC tests of Part 2 address this. The dataset-identity probe (DESIGN.md
  §6) remains the global check.
- **D1→D2 / D2→D1:** a dataset-specific QC difference (e.g. more interpolations or larger edge
  effects in D2) is a technical domain shift that can lower transfer independently of biology. Report
  it next to the cross-dataset gate.
- **LOSO (leave one subject out):** subject-level artifacts add noise to the held-out subject. They
  bias the results only if they correlate with diagnosis.

## 4. Running it (Colab)

The production code stays pinned at `7fd5a69` (Cell 1 of Notebook 01 with `UPDATE_CODE = False`). The
audit script is read from its own commit (`AUDIT_COMMIT`, given with the instructions) into
`/content/audit/`, outside the repository, so the audit code is versioned separately and the pinned
working tree stays clean. The order is `estimate`, `part1`, `report1`, then `part2`, `report2`.
- Every command can be rerun after a disconnect: completed recordings are skipped.
- `status` shows the progress.
- Outputs go only to the separate Drive folder given with `--out`.
