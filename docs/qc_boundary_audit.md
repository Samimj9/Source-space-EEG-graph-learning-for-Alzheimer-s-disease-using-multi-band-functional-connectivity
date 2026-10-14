# Dataset-wide QC audit: bad channels and connectivity boundary sensitivity

Status: **plan and pre-specified decision criteria, written 2026-10-14 before any dataset-wide
result was seen** (methodology log entry 40). The audit is read-only. The production code stays at
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
- the per-recording table;
- per-channel detection frequency by dataset, with AD/HC counts;
- the distribution of interpolated counts;
- AD vs HC per dataset: Mann–Whitney, plus a permutation test of the within-dataset AD − HC
  difference with labels permuted within dataset;
- D1 vs D2;
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

Report 2:
- the per-recording and per-band tables;
- the distance profile;
- Spearman associations, across subjects, with edge severity and the number of interpolated channels;
- D1 vs D2 and AD vs HC: per-recording statistics, Mann–Whitney within dataset, stratified
  permutation;
- unusually sensitive recordings, with their bands and ROI pairs;
- ROI concentration;
- the **primary relative criterion** (§3.2).

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

### 3.1 Bad channels (Part 1)
- **E1 Systematic electrode:** interpolated in ≥ 30 % of the recordings of a dataset.
- **R1 Recording with too many interpolated channels:** more than 20 % of the 30 channels (≥ 7). This
  is the existing `bad_channels.max_fraction_warn`.
- **R2 Interpolation over a gap:** a cluster of ≥ 3 neighbouring interpolated electrodes. Spherical
  splines are least reliable where neighbours are missing.
- **R3 Existing rule:** fewer than 5 clean windows, the only pre-specified exclusion in DESIGN.md.
- **G1 Group imbalance:** the AD − HC difference in interpolated counts, labels permuted within
  dataset, with p < 0.05. This is reported as a potential confound, not used to exclude.

### 3.2 Boundary sensitivity (Part 2)
- **Primary (relative):** for each (band, ROI pair), the ratio of |subject-mean A − subject-mean B| to
  the pooled within-dataset between-subject SD of the subject-mean A, summarised per recording by its
  95th percentile.
  - **Negligible:** ≥ 90 % of the recordings at ≤ 0.10 and none > 0.25.
  - **Moderate:** otherwise, as long as ≤ 10 % of the recordings are > 0.25.
  - **Material:** > 10 % of the recordings > 0.25.
- **Dependence:** D1 vs D2 (Mann–Whitney) or AD vs HC (stratified permutation) differences in the
  relative p95 or in the PLV subject-mean max with p < 0.05.
- **Interior persistence:** p95 of |A − B| at distance ≥ 2 above 0.005 in any recording.
- **Unusual recording:** robust z of log(PLV subject-mean max) > 3.5, or PLV subject-mean max > 0.01,
  or PLV max excluding boundary-touching windows > 0.05. D2-2 (0.0204) already meets the second
  condition.

### 3.3 Decision rules (applied separately to the two problems)

**Interpolation**
- **A (keep the 30-channel montage with subject-specific interpolation)** applies when there is no E1
  electrode and R1/R2 flags are rare (≤ 10 % of recordings) with no G1 imbalance.
- **C (test a reduced common montage)** applies when an E1 electrode exists:
  - the same electrodes are removed in both datasets;
  - a new operator is computed for the reduced set;
  - run first as a sensitivity analysis. Making it primary is the author's decision.
- **B (exclude recordings)** is considered only for recordings meeting R1 or R3, and only with a
  logged reason and a DESIGN.md amendment. Any such exclusion is accompanied by the analysis with all
  recordings. R2 alone triggers a sensitivity analysis, not exclusion.

**Boundary**
- **E (retain the production pipeline, report this audit as a sensitivity analysis)** applies when the
  effect is negligible, with no dependence and no interior persistence.
- **D (revise boundary handling and regenerate the graphs)** applies when the effect is material, or
  any dependence exists, or interior persistence occurs in more than one recording. The new method is
  chosen on synthetic ground truth and boundary diagnostics, never on classification results.
- **Moderate effect:** E plus, as a pre-specified secondary analysis, results recomputed without
  boundary-touching windows. Recomputing graphs needs the author's approval.

**General**
- Channel removal (C) is not assumed to change boundary sensitivity, and vice versa. Each problem is
  assessed on its own evidence.
- No electrode, recording or preprocessing choice is made from classification performance on any
  test set.

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

The production code stays pinned at `7fd5a69`. The audit script is read from its own commit (the
`AUDIT_COMMIT` printed in the instructions), so the audit code is versioned separately:

```python
# after Cell 1 of notebooks/01_build_graphs.ipynb (UPDATE_CODE = False, pin 7fd5a69)
AUDIT_COMMIT = '<commit given in the instructions>'
AUDIT_OUT = '/content/drive/MyDrive/AD-LEB-audit'          # separate Drive folder
import subprocess
from pathlib import Path
Path('/content/audit').mkdir(exist_ok=True)
Path('/content/audit/audit_quality_boundary.py').write_text(subprocess.check_output(
    ['git', 'show', f'{AUDIT_COMMIT}:scripts/audit_quality_boundary.py'], cwd='/content/repo', text=True))
sh(f"python /content/audit/audit_quality_boundary.py estimate --out '{AUDIT_OUT}'", cwd='/content/repo')
```

Then run `part1`, `report1`, `part2` and `report2` in that order with the same call. Each command can
be rerun after a disconnect and continues where it stopped; `status` shows the progress.
