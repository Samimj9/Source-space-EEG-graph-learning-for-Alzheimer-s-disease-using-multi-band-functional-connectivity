# Boundary-Aware Filtering and Hilbert Phase Estimation for Source-Space EEG Functional Connectivity

Methodological decision record, **version 1.0** (2026-10-14). This is methodology log entry 41 and
decisions-log row 49. The pipeline version is v4 (`configs/pipeline.yaml`, `version: 4`, `boundary:`
section).

Code:
- `src/adgraph/preprocessing/boundary.py`;
- `src/adgraph/connectivity/plv.py` (`analytic_kernel`, `band_analytic_fir`);
- `src/adgraph/pipeline.py` (`process_subject`, `process_recording`).

The code commit is recorded in §8 and in every graph-run manifest. Numbers marked **[final]** are
filled in from the rebuild's `validation_<time>/validation.json` once the 43-recording reconstruction
has completed. Everything else describes the implemented code.

---

## 1. Original implementation (pipeline v3, production commit `7fd5a69`)

Per recording, on the continuous signal of the **whole recording**:
1. Load the 30 common channels (volts; D1 × 1e-6).
2. Resample to 250 Hz with MNE's FFT method (`Raw.resample`, default padding).
3. Band-pass 0.1–45 Hz with an MNE zero-phase FIR (firwin, Hamming):
   - 8251 taps (33.0 s), with transition bands of 0.1 Hz (lower) and 11.25 Hz (upper);
   - so every output sample depends on input up to 16.5 s away.
4. Unit safeguard on the whole recording.
5. Bad channels detected on the edge-trimmed interval of a 1 Hz high-passed copy filtered over the
   whole recording; spherical-spline interpolation; average reference; amplitude plausibility on the
   edge-trimmed interval.
6. Windows of 20 s with a 10 s step, placed from 10 s after the start to 10 s before the end
   (`edge_trim_s`). Adaptive peak-to-peak rejection.
7. ROI signals = 68 × 30 operator × the whole sensor signal.
8. For each band, a zero-phase FIR band-pass (825 taps, 1 Hz transitions, half-length 1.648 s), then
   the analytic signal by `scipy.signal.hilbert`: one FFT of the **entire** band-passed recording.
9. PLV, signed ciPLV and |ciPLV| per accepted window.

The 10 s at each end were excluded from the windows but **not from the processing**. Their content,
including large start/end transients, entered:
- the 33 s sensor filter, which spreads content up to 16.5 s;
- the band-pass filters (1.648 s);
- the FFT-based Hilbert transform. This transform is non-local: its kernel decays only as 1/t, and the
  FFT treats the recording as periodic, so content at one end also wraps around to the other end.

## 2. Reason for revision (completed 43-recording audit; `docs/qc_boundary_audit.md`)

The audit compared:
- **A**, the exact production computation;
- **B**, a diagnostic alternative with reflected recording edges, on identical accepted windows.

All 43 recordings (20 D1, 23 D2) were processed. A reproduced the stored production arrays exactly for
D1-1 and D2-2.

- **Largest changes in PLV:**

| Recording | p95 \|A−B\| | max \|A−B\| | max subject-mean \|A−B\| |
|---|---:|---:|---:|
| D1-8 | 0.0368 | 0.855 | 0.177 |
| D2-12 | 0.0331 | 0.742 | 0.164 |
| D1-14 | 0.0343 | 0.740 | 0.117 |
| D2-4 | 0.0186 | 0.596 | 0.079 |
| D2-22 | 0.0276 | 0.628 | 0.069 |
| D1-4 | 0.0076 | 0.494 | 0.052 |

- **Theta** was the most sensitive PLV band in 29 of the 43 recordings (13 D1, 16 D2).
- **Sensitivity rose with edge-transient severity:** Spearman ρ = 0.755 between band-specific edge
  severity and the maximum subject-mean PLV difference. It was not significantly associated with the
  number of interpolated channels (ρ = 0.248, p = 0.109).
- **Boundary windows were the most sensitive:** the median p95 PLV difference was 0.0218 at distance 0
  from the boundary and 0.00085 at ≥ 6 window steps. Some large differences persisted beyond the first
  and last windows.
- **Dataset differences:**
  - the per-recording PLV p95 difference was higher in D2 (median 0.00492) than in D1 (0.00054),
    Mann–Whitney p = 0.0066;
  - the median edge RMS ratios showed different boundary characteristics: D1 start 47.8 / end 6.8,
    D2 start 0.70 / end 82.8.
- **Normalised subject-mean PLV sensitivity** (provisional QC categories, not validated thresholds):
  37/43 recordings ≤ 0.10, 3 between 0.10 and 0.25, 3 > 0.25. For |ciPLV|: 37, 4 and 2.

The audit measured **sensitivity of the estimates to boundary handling**. It did not measure error
relative to a ground truth, and B is not a validated reference.

## 3. New design (pipeline v4)

### 3.1 Crop before any filtering
- The `edge_trim_s` = 10 s at each end is discarded at the **original sampling rate**, before
  resampling (`boundary.crop_edges`): 10 000 samples for D1 at 1000 Hz, 10 240 for D2 at 1024 Hz.
- The discarded samples, which carry the large start/end transients, never enter any filter, the
  bad-channel detection, the interpolation, the reference, the source operator or the phase
  estimation.
- A synthetic implementation test shows that any change confined to the discarded 10 s leaves every
  output bit-identical, for both sampling rates (`tests/test_boundary_processing.py`).

### 3.2 Continuous filtering with explicit padding
The cropped signal is processed continuously as before:
- resampled to 250 Hz (MNE FFT method);
- band-passed 0.1–45 Hz (the same 8251-tap FIR);
- 1 Hz high-passed for bad-channel detection.

All three operations pad explicitly with MNE's `reflect_limited` (`boundary.filter_pad`):
- the signal is extended by its mirror image about the end sample (x[−k] = x[k]), limited to the
  signal length, then zeros;
- this is MNE's default, now fixed in the configuration;
- the mirror extension avoids the step that zero-padding would create at a data boundary.

Under the boundary margin (§3.4) no padded sample reaches an accepted window through any FIR filter.
The padding therefore only affects samples that are excluded from analysis (exception: §7).

### 3.3 Hilbert analytic signal by a finite complex FIR
The band-pass and the Hilbert transform are combined into one finite analytic kernel per band:

&nbsp;&nbsp;&nbsp;&nbsp;g = h + i·𝓗{h}

- h is the production zero-phase band-pass kernel (825 taps).
- 𝓗{h} is its Hilbert transform, computed exactly by FFT on a 64 × 825 zero-padded copy of h and
  truncated to the same 825 taps, centre-aligned.
- Because the Hilbert transform is linear and shift-invariant, the analytic signal of the band-passed
  ROI signal is a = x ∗ g, up to the truncation of 𝓗{h}. The convolution is computed as a **linear**
  convolution (`scipy.signal.fftconvolve`, mode `same`), not a circular one.
- Each analytic sample depends only on the 825 input samples centred on it (±1.648 s). There is no
  periodic wrap-around and no dependence on distant samples.
- **Numerical properties**, per band, recorded in the manifest:

| Property | Value |
|---|---|
| Hilbert-kernel energy outside the 825 taps | ≤ 3.8 × 10⁻⁷ |
| Negative-frequency leakage | ≤ −76 dB |
| Interior agreement with the production filter + FFT Hilbert (synthetic) | PLV and \|ciPLV\| within 4 × 10⁻⁵ |

### 3.4 Boundary-exclusion margin (computed from the filters)
- Accepted windows lie at least M samples inside the cropped signal.
- Policy `full_support` (configured):

&nbsp;&nbsp;&nbsp;&nbsp;M = (L_sensor − 1)/2 + (L_analytic − 1)/2 = 4125 + 412 = **4537 samples = 18.148 s**

  - L_sensor = 8251 taps is the 0.1–45 Hz FIR;
  - L_analytic = 825 taps is the analytic kernel.
- Every sample of an accepted window therefore depends, through every FIR in the chain, only on
  recorded samples of the analysed interval.
- The alternative policy `feature_band` (M = 412 + 37 = 449 samples, 1.796 s) uses only the half-length
  of the analytic kernel and of the 45 Hz low-pass part. It is implemented but not configured (§9).

### 3.5 Windows and rejection
- Windows of 5000 samples (20 s) with a step of 2500 samples (10 s, 50 % overlap) tile [M, N − M) of the
  cropped signal (N samples at 250 Hz), starting at its first sample:

&nbsp;&nbsp;&nbsp;&nbsp;s_m = M + 2500·m, m = 0, …, ⌊(N − 2M − 5000)/2500⌋

- Start samples are stored relative to the original recording (offset 2500 samples = 10 s).
- The rejection rule is unchanged: a window is rejected when more than 15 % of its channels exceed the
  subject's adaptive peak-to-peak threshold (median + 5·MAD over all candidate windows × channels).
- The minimum of 5 clean windows is unchanged.

### 3.6 Same rules for both datasets
- All parameters are in seconds, or in samples at the common 250 Hz rate.
- The crop is exactly 10 s at both original rates.
- D1 and D2 run through the same code path and configuration.

## 4. What remains unchanged
- All 43 recordings and their labels.
- The common 30-channel montage (M1/M2 dropped), with no global removal of F7, T7, T8 or any other
  electrode.
- Subject-specific bad-channel detection (same rules and thresholds) and spherical-spline
  interpolation.
- The unit scaling, the 250 Hz resampling (FFT method), the 0.1–45 Hz FIR and the average reference.
- The amplitude checks (same statistics and ranges).
- The fsaverage source model, MNE inverse (λ² = 1/9, loose 0.2, depth 0.8, normal orientation),
  Desikan–Killiany 68 ROIs, `mean_flip`, and the shared 68 × 30 operator.
- The frequency bands: theta 4–8, alpha1 8–10, alpha2 10–13, beta 13–30 Hz, and the band-pass kernels
  (825 taps, 1 Hz transitions).
- The PLV, signed ciPLV and |ciPLV| definitions (§S5).
- The PSD features: Welch, 2 s Hann segments, 50 % overlap, absolute / log / relative over [1, 45) Hz;
  source and sensor.
- 20 s windows with 50 % overlap, the rejection rule and the minimum-window rule.

Two QC steps now run on the cropped signal: the unit safeguard and the spatial-consistency QC. The 1 Hz
high-pass for bad-channel detection is also computed on the cropped signal; its 1.65 s boundary region
lies within the data the rules already used.

## 5. Effect on the dataset

| | v3 (production, `7fd5a69`, audited) | v4 (boundary-aware) |
|---|---|---|
| First window start | 10 s | 28.148 s |
| Candidate windows for a recording of T s | ⌊(T − 40)/10⌋ + 1 | ⌊(T − 76.296)/10⌋ + 1 |
| Candidate windows lost per recording | | 3 or 4 |
| Accepted windows (graphs) | 1,640 | **[final]** |
| Recordings with ≥ 5 accepted windows | | **[final]** |

The reduction is caused only by the boundary margin (§3.4). The per-recording old and new counts are in
`validation_<time>/dataset_report.csv` (columns `reference_accepted_windows`, `accepted_windows`).

## 6. Reproducibility
- **Configuration:** `configs/pipeline.yaml` (version 4, `boundary` section); its hash names the output
  directory `graphs_v4_<config hash>/`.
- **Code commit:** see §8. It is also recorded as `git_commit` (clean tree) in every subject
  checkpoint, `manifest.json` and `validation.json`.
- **Software:** `mne == 1.13.2` (pinned), `edfio == 0.4.18` (pinned), numpy ≥ 1.26, scipy ≥ 1.11. The
  exact versions used are recorded in `manifest.json` (`versions`).
- **Filter lengths:**

| Filter | Taps | Duration |
|---|---|---|
| Sensor band-pass | 8251 | 33.0 s |
| 45 Hz low-pass part | 75 | |
| Band-pass and analytic kernels | 825 | 3.3 s |
| Bad-channel detection 1 Hz high-pass | 825 | |

- **Padding:** `reflect_limited` for resampling and every FIR on the continuous signal; implicit zeros
  for the linear convolution with g (inside the margin only).
- **Margin:** 4537 samples (18.148 s).
- **Crop:** 10 s per side at the original rate.
- **Manifest:** `manifest.json` → `boundary_processing` records all of the above, including the
  per-band kernel properties, plus the source-operator identity, its verification error (< 10⁻⁶) and
  the comparison with the cached operator of `7fd5a69`.
- **Outputs:** `<processed>/graphs_v4_<config hash>/`. Earlier graph runs, checkpoints, experiments and
  audits are left untouched. Experiments select the highest pipeline version (`latest_complete_graph_run`).

## 7. Scientific limitations
- **The audit measured sensitivity, not error.** It established that PLV/|ciPLV| depended on how the
  recording boundaries were handled. It did not establish the error of either computation relative to
  the true phase synchrony, and the reflected-edge alternative B is not physiologically validated.
- **The correction removes known mechanisms, not every artifact.** It removes the mechanisms identified
  in the original implementation: discarded edge content entering long filters, and the non-local,
  periodic FFT Hilbert transform. It does not remove artifacts inside the analysed interval; those
  remain subject to bad-channel interpolation and window rejection.
- **FFT resampling is not compactly supported.** It is an ideal low-pass at 125 Hz with reflect-limited
  padding. Its residual boundary influence beyond the margin was ≈ 10⁻⁷ of the signal SD or less in a synthetic
  implementation check.
- **Truncation of the Hilbert kernel** causes negative-frequency leakage of ≤ −76 dB.
- **Fewer windows per recording** reduce the per-subject averaging, by about 3–4 windows.
- **Reported numbers changed.** Results computed with v3 graphs are not comparable with v4 results.
  v3-based numbers are not reported as v4.

## 8. Code version
- **Corrected code commit: [final]**, the commit that introduces this record. Exact hash: see §8 of this
  file at release, and `git_commit` in every v4 manifest.

## 9. Implementation choices requiring the author's approval before the rebuild
1. **Margin policy:** `full_support` (18.148 s, configured) vs `feature_band` (1.796 s; v4 candidate
   counts then about one window lower than v3).
2. **Window grid anchored at the first valid sample:** window start times differ from v3 even where
   windows overlap.
3. **The unit safeguard and the spatial-consistency QC** run on the cropped signal.
4. **The source operator is rebuilt**, because its identity includes the code version, and it is
   compared with the cached `7fd5a69` operator (expected identical).

---

## 10. Methods subsection (publication text; numbers marked [final] are completed after the rebuild)

**Boundary-Aware Signal Processing and Functional Connectivity Estimation.**

For each recording, the first and last 10 s were discarded at the original sampling rate (1000 Hz for
dataset 1, 1024 Hz for dataset 2) before any further processing, so that acquisition onset and offset
transients could not enter subsequent filters.

The remaining signal was:
- resampled to 250 Hz (FFT-based resampling);
- band-pass filtered between 0.1 and 45 Hz with a zero-phase finite impulse response (FIR) filter
  (window method, Hamming window, 8251 coefficients);
- subjected to bad-channel detection, spherical-spline interpolation and re-referencing to the common
  average.

All filters on the continuous signal used reflection padding at the signal ends. Source time series of
68 Desikan–Killiany regions were obtained with a fixed linear operator derived from the fsaverage
template (MNE, λ² = 1/9; sign-flipped mean within regions).

For each frequency band (theta 4–8 Hz, alpha1 8–10 Hz, alpha2 10–13 Hz, beta 13–30 Hz), the analytic
signal of the band-limited regional time series was computed by linear convolution with a finite
complex kernel g = h + i𝓗{h}:
- h is a zero-phase band-pass FIR filter (825 coefficients, 1 Hz transition bands);
- 𝓗{h} is its Hilbert transform, truncated to the same 825 coefficients.

Each analytic sample therefore depended only on input samples within ±1.648 s, and no periodic
continuation of the signal was assumed.

Analysis windows of 20 s with 50 % overlap were placed at least M = 4537 samples (18.148 s) from both
ends of the processed signal. M is the sum of the half-lengths of the 0.1–45 Hz filter and of the
analytic kernel, so no sample within an accepted window depended on padded values through any FIR
filter. Windows were rejected when more than 15 % of channels exceeded a subject-specific peak-to-peak
threshold (median + 5 MAD).

For regions j and k within a window of T samples, with phases φ = arg a:

&nbsp;&nbsp;&nbsp;&nbsp;C_jk = T⁻¹ Σ_t exp(i(φ_j(t) − φ_k(t)))
&nbsp;&nbsp;&nbsp;&nbsp;PLV_jk = |C_jk|
&nbsp;&nbsp;&nbsp;&nbsp;|ciPLV_jk| = |Im C_jk| / √(1 − (Re C_jk)²)

Identical processing parameters were applied to both datasets. The procedure yielded **[final]** graphs
from **[final]** recordings, compared with 1,640 graphs under the previous boundary handling. The
difference results from the boundary-exclusion margin.

This procedure removes the dependence of the retained windows on discarded recording edges and on
periodic signal continuation. It does not guarantee the absence of other artifacts within the analysed
interval.

---

## Supplementary Methods

**S1 Filter design.**
- **Sensor band-pass:**
  - MNE `create_filter`, `fir_design='firwin'` (Hamming), zero phase;
  - l_freq = 0.1 Hz with a 0.1 Hz transition, h_freq = 45 Hz with an 11.25 Hz transition;
  - 8251 taps at 250 Hz (33.0 s);
  - −6 dB at 0.05 Hz and 50.6 Hz.
- **Band-pass kernels:**
  - same design, transition width 1.0 Hz on both edges (lower edge capped at l_freq);
  - 825 taps (3.3 s);
  - nominal band edges are passband edges (`docs/band_filters.md`).
- **Bad-channel detection:** 1 Hz high-pass, 825 taps.

**S2 Padding.**
- MNE `reflect_limited` for resampling (`npad='auto'`) and FIR filtering: mirror image about the end
  sample, limited to the signal length, then zeros.
- Linear convolution with g extends the signal with zeros. Samples within 412 of either end of the
  convolution output are inside the margin and never used.

**S3 Hilbert implementation.**
- 𝓗{h} = Im(hilbert(h₀)), where h₀ is h zero-padded to 64 × 825 = 52 800 samples (centred); the
  central 825 samples are kept.
- g = h + i𝓗{h}; analytic signal a = fftconvolve(x, g, mode='same') per region (linear convolution).
- Recorded per band: kernel energy outside the support ≤ 3.8 × 10⁻⁷ and negative-frequency leakage
  ≤ −76 dB (`manifest.boundary_processing.analytic_kernels`).

**S4 Window indexing.**
- N = samples of the cropped signal at 250 Hz; M = 4537; window length 5000, step 2500.
- Candidate starts: s_m = M + 2500 m for m = 0 … ⌊(N − 2M − 5000)/2500⌋.
- Stored as s_m + 2500 (relative to the original recording, `windows.csv: start_sample`).
- `window_index` = m. The rejected windows and the reasons are in `qc_windows.csv` of each subject.

**S5 Connectivity definitions.**
- C_jk = T⁻¹ Σ_t z_j(t) z̄_k(t), with z = a/|a|.
- PLV_jk = |C_jk|.
- Signed ciPLV_jk = Im C_jk / √(1 − (Re C_jk)²), with the denominator floored at 10⁻¹², made exactly
  antisymmetric, and the diagonal set to 0.
- |ciPLV| is its magnitude and is used as the adjacency. Signed ciPLV is stored but never used as an
  adjacency.

**S6 Numerical implementation.**
- float64 throughout, with float32 storage of the arrays.
- Analytic signals are computed per band in chunks of 8 regions (identical results).
- The operator is verified against the full MNE inverse (relative error < 10⁻⁶).
- Checkpoints carry SHA-256 identities of inputs and code.

**S7 Configuration and software.**
- `configs/pipeline.yaml` (v4).
- Exact library versions and the code commit are in `manifest.json`.

**S8 Audit summary motivating the change:** §2.

---

## Before-versus-after methodology table

| Processing component | Original design (v3) | Revised design (v4) | Reason for modification | Downstream graphs affected |
|---|---|---|---|---|
| Edge handling | 10 s excluded from windows only; the whole recording processed | 10 s discarded before resampling and filtering | Edge transients entered filters and the phase estimate | Yes |
| Resampling to 250 Hz | MNE FFT, whole recording, default padding | MNE FFT on the cropped signal, `reflect_limited` explicit | Edges no longer enter | Yes (numerically) |
| 0.1–45 Hz FIR (8251 taps) | Whole recording, default padding | Cropped signal, `reflect_limited` explicit | As above | Yes (numerically) |
| Bad-channel detection | Rules on the trimmed interval of a whole-recording 1 Hz high-pass | Same rules on the cropped signal (high-pass on the cropped signal) | Consistency with the crop | Possibly (channel decisions) |
| Interpolation, reference, operator | — | Unchanged | — | No (except through inputs) |
| Band-pass (825 taps) | Whole ROI signal | Real part of the analytic kernel on the cropped ROI signal | Locality | Yes (numerically) |
| Analytic signal | FFT Hilbert of the whole band-passed recording (non-local, periodic) | Finite complex FIR g = h + i𝓗{h}, linear convolution | Remove wrap-around and long-range dependence | Yes |
| Boundary margin | None beyond the 10 s trim | M = 4537 samples (18.148 s) from the cropped ends, computed from the filters | Accepted windows independent of padding | Yes (window count) |
| Window grid | From 10 s, step 10 s | From 10 s + M, step 10 s | Follows from the margin | Yes (positions, count) |
| Window rejection, minimum windows | Adaptive ptp, > 15 % channels; ≥ 5 windows | Unchanged | — | Through the candidate set |
| PLV / ciPLV definitions, bands, PSD | — | Unchanged | — | No |
