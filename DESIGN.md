# DESIGN.md: Evaluation design (frozen before classification)

This document fixes the evaluation protocol **before any classifier is
trained**. Any change after classification starts must be logged here (with
date and reason) and in `docs/methods_decisions.md`. Results produced under an
earlier version of this design are reported against that version.

Status: **frozen v1.0 (2026-10-12)**. No classification has been run.

---

## 1. Cohort

All subjects and recordings are used as provided. There is no exclusion based
on file-content similarity.

| Dataset | Subjects | AD | HC | Recording used |
|---|---|---|---|---|
| D1 (2017) | 20 | 10 | 10 | `sub<N>.edf` (resting state) |
| D2 (2019) | 23 | 10 | 13 | `<id>_<date>_1.edf` (RestingState) |
| Pooled | 43 | 20 | 23 | |

A subject is excluded only by the pre-specified pipeline rule (fewer than 5
clean windows after artifact rejection). Any such exclusion is listed in the
run manifest and in the results tables.

**Unit of analysis.** The model sees one multi-band graph per 40 s window, but
the **subject** is the unit of inference. The subject key is
`(dataset, subject_id)`, e.g. `D1-3`, `D2-3`.

---

## 2. The four evaluation settings

| # | Setting | Train | Test | Role |
|---|---|---|---|---|
| S1 | Within-D1 | D1 subjects (CV) | held-out D1 subjects | **Primary** |
| S2 | Within-D2 | D2 subjects (CV) | held-out D2 subjects | **Primary** |
| S3 | Pooled D1+D2 | D1+D2 subjects (CV) | held-out D1+D2 subjects | Secondary |
| S4a | Cross-dataset | all D1 | all D2 | Conditional (gated, §7) |
| S4b | Cross-dataset | all D2 | all D1 | Conditional (gated, §7) |

### Primary vs secondary

- **Primary analyses: S1 and S2**, each reported separately. They are never
  averaged into one headline number, because the datasets differ in
  acquisition, cohort and age structure.
- **Primary comparison:** the multi-band source-space GNN against the
  pre-specified baselines (§5) in S1 and S2.
- **Secondary: S3.** It describes performance when data are combined and is
  used to inspect dataset effects. It is not a claim of generalization.
- **Conditional / exploratory: S4.** It runs only if the harmonization gate
  (§7) passes. If the gate fails, the paper reports the reason instead of a
  number.

---

## 3. Subject-level split strategy

### 3.1 Within-dataset (S1, S2)

- **Outer loop:** repeated stratified K-fold over subjects. K = 5, with
  **20 repeats** using seeds `0..19`.
  - Folds are built from the list of subjects, stratified by label. Each
    subject's windows all go to that subject's fold.
  - D1 (10/10): each test fold has 4 subjects (2 AD, 2 HC).
  - D2 (10/13): each test fold has 4–5 subjects with 2 AD each, and
    stratification keeps the HC share as even as possible.
- **Inner loop (model selection):** stratified K = 4 subject-level CV inside
  each outer training set. It chooses the hyperparameters from a
  pre-specified, finite grid recorded in the config. The outer test subjects
  are never seen during selection.
- **Sensitivity analysis:** leave-one-subject-out (LOSO) with the same inner
  loop, reported in the supplement.
- Per outer fold, the following are fitted **on training subjects only**:
  feature or edge scaling, any data-driven graph threshold, any feature
  selection, early-stopping validation (taken from the inner split, never the
  outer test fold), and class weighting.
- Window-level `sample_weight = 1 / n_windows(subject)` is used in the loss,
  so subjects with more windows don't dominate training.

### 3.2 Pooled (S3)

- The same outer and inner scheme as §3.1, with subjects from both datasets.
  **Stratification is by `dataset × label`** (4 strata), so every test fold
  holds both datasets and both classes in proportion.

### 3.3 Cross-dataset (S4)

- Train on **all** subjects of the source dataset. Hyperparameters are
  selected by the §3.1 inner CV run on the source dataset only.
- Test **once** on all subjects of the target dataset.
- No target-dataset data are used for fitting, selection, thresholding or
  early stopping.
- A transductive variant, with per-dataset unsupervised standardization of
  target features, may be reported **only as a separately labelled
  sensitivity analysis**. It is never the main S4 result.

### 3.4 Determinism

Seeds for splits, model initialization and data loading are fixed and written
to the run manifest. The exact fold assignment (subject → fold, per repeat) is
saved as a CSV next to the outputs in Drive.

---

## 4. Subject-level prediction and metrics

### 4.1 From windows to subjects

- **Subject probability** is the mean of that subject's window probabilities
  for AD.
- **Decision threshold** is fixed at 0.5. It is not tuned on test data.
- Window-level predictions are kept for secondary reporting.

### 4.2 Metrics

Primary metrics, at subject level:

- **ROC AUC**
- **Balanced accuracy**

Secondary metrics:

- At subject level: sensitivity (AD recall), specificity (HC recall), F1 for
  AD, Matthews correlation coefficient (MCC), and the confusion matrix.
- At window level: AUC and balanced accuracy, clearly labelled as window
  level.

Each metric is computed as follows:

- **Within each repeat,** out-of-fold subject predictions are concatenated
  across the K folds, giving one prediction per subject, and the metric is
  computed once on that set. This is better than averaging per-fold AUCs from
  4–5 test subjects.
- **Across repeats,** the summary is the mean ± SD, plus the 2.5th–97.5th
  percentile range.
- **Uncertainty:** each subject's out-of-fold probability is averaged across
  the 20 repeats. A 95% CI then comes from 2,000 bootstrap resamples **of
  subjects** on those averaged predictions.
- **Significance against chance:** a subject-level label-permutation test.
  Labels are permuted across subjects, the full CV pipeline is re-run, and
  1,000 permutations are used (at least 200 if compute-limited, with the
  number reported). The p-value is computed as
  `(1 + #perm ≥ observed) / (1 + n_perm)`.
- **Model comparisons** (GNN vs. baseline on the same folds) use paired
  differences per repeat with the corrected resampled t-test of Nadeau and
  Bengio. They are reported as effect size + CI. With small N, "no
  significant difference" is reported as such.
- **Per-dataset breakdown in S3:** pooled out-of-fold predictions are also
  scored separately for D1 subjects and D2 subjects (§6).
- **Per-subject predictions** (subject key, true label, mean probability per
  repeat) are saved to Drive for every setting. They are never committed.

### 4.3 Multiplicity and reporting discipline

- One primary model configuration and one primary baseline set are declared
  in `configs/` before S1/S2 are run.
- Every configuration evaluated on outer test folds is logged. The count of
  configurations tried is reported in the paper.
- No configuration is selected by outer-test performance.

---

## 5. Comparators (same folds, same subjects)

1. **Chance / majority class.**
2. **Age-only logistic regression** in S1–S4. D2 has a strong age imbalance
   (HC ≈ 62 y, AD ≈ 80 y), so the GNN must be read against this baseline.
3. **Classical ML on the same connectivity:** an L2-regularized logistic
   regression and a linear SVM on vectorized upper-triangle PLV (68 ROIs × 4
   bands), with standardization fitted on training data.
4. **Single-band GNN vs. multi-band GNN,** the ablation that supports the
   paper's multi-band claim.
5. **Age-adjusted sensitivity analysis,** S2 in particular. Edge features are
   residualized on age with a regression fitted on training subjects only, and
   the model is retrained.

---

## 6. Pooled analysis and dataset identity (S3)

- `dataset` (D1/D2) is carried through every table, graph object and
  prediction file as **metadata**.
- **Main S3 model:** dataset identity is **not** a model input. It is not a
  node or graph feature and is not used as a covariate. No per-dataset
  harmonization is applied.
- **Dataset-effect diagnostics**, all reported:
  1. **Per-dataset scores** from the pooled model (§4.2). These are compared
     with S1/S2 to show whether pooling helps or hurts each dataset.
  2. **Dataset-identity probe.** The same model and pipeline are trained to
     predict `dataset` instead of diagnosis, with the same CV. A high
     dataset-prediction AUC means strong site/acquisition signal that the
     diagnostic model could exploit.
  3. **Within-class dataset comparison.** HC(D1) vs HC(D2) and AD(D1) vs
     AD(D2) are compared on band-wise mean PLV and global graph metrics.
  4. **Label × dataset balance.** AD fraction is 50% in D1 and 43% in D2.
     Because it is not identical, a model could partly score by detecting the
     dataset. The probe in (2) quantifies this risk.
- **Sensitivity variant:** per-dataset feature standardization, fitted on the
  training subjects of each dataset within each fold. It is reported next to
  the main S3 model, never instead of it.
- S3 results are **not** evidence of cross-dataset generalization. A pooled
  model can succeed by learning dataset-specific rules.

---

## 7. Gate for cross-dataset testing (S4)

S4 runs only when **every "must" item** below is satisfied and documented in
`docs/methods_decisions.md`. The verification is recorded in the S4 run
manifest. Until then, S4 results are not generated.

### 7.1 Pipeline identity ("must")

| Item | Requirement | Current state (manifest 2026-09-12) |
|---|---|---|
| Channel set | Identical channel names and count entering source modelling for D1 and D2 | ❌ D1 has 30 channels (M1/M2 dropped) and D2 has 32 (M1/M2 kept). This must be harmonized to a common set. |
| Electrode positions | Same montage source (`standard_1020` by name) for both | ✅ |
| Reference | Average reference computed over the same common channel set | ⚠ It is average, but over different sets, so this follows from fixing the channel set. |
| Bad channels | Same detection and handling rule (e.g. interpolate listed bad channels, or none) for both | ❓ The rule is not defined yet. |
| Units | Amplitudes verified in volts for both (D1 ×1e-6 fix verified by comparing amplitude/PSD ranges with D2) | ⚠ The fix is applied but not cross-checked against D2. |
| Resampling / filtering | Same target rate (250 Hz), same filter design and parameters | ✅ |
| Windowing / rejection | Same window length, step, trim, rejection rule and minimum-window rule | ✅ |
| Forward model | Same template (fsaverage), source space (ico-5), BEM, and the **same sensor set**. Ideally one forward model is shared by both datasets. | ❌ There are two forward models on different sensor sets. |
| Inverse | Same method, `loose`, `depth`, `λ²` and **noise-covariance** choice | ⚠ The noise covariance is not recorded in the manifest. |
| ROIs | Same parcellation (aparc, 68), same ROI order, same extraction mode (`mean_flip`) | ✅ |
| Bands | Identical band edges and the same band-pass/Hilbert implementation | ✅ |
| Graph construction | Same PLV estimator, same window length (PLV bias depends on sample count), same sparsification/threshold rule, same node features, same edge normalization | ✅ PLV and window length; the threshold and node features are still to be fixed. |
| Provenance | Both datasets processed by the same code commit and the same config hash in one pipeline run | ❓ |

### 7.2 Comparability checks (must be run and reported; they inform interpretation)

- Recording condition: whether both datasets are eyes-closed (or eyes-open)
  resting state. This has to be confirmed with the data owner and
  documented. A mismatch is a major caveat.
- Original sampling rate, amplifier and recording duration per dataset.
- Diagnostic criteria and cognitive ranges, e.g. MMSE (D1 AD 3–21,
  D2 AD 7–20) and the HC definition.
- Age distributions per class per dataset.
- Distribution-shift diagnostics: the dataset-identity probe (§6), and
  HC-vs-HC and AD-vs-AD comparisons of band PSD and band-wise mean PLV.

If any 7.1 item cannot be satisfied, S4 is reported as **"not performed:
pipeline not harmonizable"** along with the reason.

---

## 8. What counts as evidence

**Within-dataset discrimination (S1, S2).** In a given dataset, the model must
meet all of the following at subject level:

- AUC 95% CI lower bound > 0.5
- Permutation p < 0.05
- Balanced accuracy above the majority baseline

**Added value of the method.** The multi-band GNN must beat the age-only
baseline, and must match or beat the classical PLV baseline, with a paired
difference CI that excludes 0. In D2 especially, it must beat the age-only
baseline and keep its effect after age adjustment (§5.5). The multi-band >
single-band ablation supports the multi-band claim.

**Generalization to new subjects.** S1 and S2 results meeting the above in
**both** datasets.

**Cross-dataset generalization.** This requires a passed gate (§7) and:

- *Strong evidence:* **both** S4a and S4b show subject-level AUC CI lower
  bound > 0.5 and permutation p < 0.05, and both beat the transferred
  age-only baseline.
- *Partial evidence:* only one direction meets these criteria. It is reported
  as asymmetric transfer, with discussion of why (sample size, cohort,
  acquisition).
- *Supportive, not sufficient:* consistent discriminative bands and ROIs
  across D1 and D2, for example overlap of top-ranked edges or band
  importance.

**Not evidence of generalization:**

- window-level metrics
- pooled (S3) results alone
- any result using a threshold, normalization or hyperparameter tuned on test
  subjects
- the best of several configurations chosen after viewing test results
- the transductive S4 variant on its own
- performance that is matched by the age-only baseline

---

## 9. Outputs per run

Saved to Drive under `processed/eval/<setting>_<timestamp>/`, never committed:

- run manifest (config, config hash, commit, seeds, software versions)
- subject→fold assignment
- per-window and per-subject predictions
- per-repeat metrics
- permutation null distribution

Aggregate tables only (no per-subject rows) go to `results/tables/`.
