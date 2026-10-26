# DESIGN.md: Evaluation design (frozen before classification)

This document fixes the evaluation protocol **before any classifier is
trained**. Any change after classification starts must be logged here (with
date and reason) and in `docs/methods_decisions.md`. Results produced under an
earlier version of this design are reported against that version.

Status: **v2.0 (2026-10-18), the final framework** (`docs/final/framework.md`, methodology log entry 58, decisions row 61). v2.0 governs every result of release final-1.0 (ds004504, LCMV and wMNE source space); v1.0-v1.7 below remain the design of the earlier D1 / D2 / D3 results, which are reported against those versions. v1.7 (2026-10-14) added Dataset 3; v1.6 the exploratory Experiment 7; v1.5 amended the GNN training protocol (fixed hyperparameters) after Exp 0, Exp 1 and the first nested-CV Exp 2 result had been observed.

Changelog:

- **v2.4 (2026-10-26, amendment A3; implemented, pending the author's approval, no holdout evaluated):** in the frozen
  final analysis exp8 is replaced by exp8a (the only protocol change; `holdout.yaml` is kept as the historical plan);
  Exp 0-6 units are frozen in the release namespace at the release pin, Exp 8a units in the A2 namespace at the A2 pin,
  linked by an A3 master record with checksums; the holdout stays descriptive, once per unit, never used for selection.
  Participant-level paired comparisons with Holm correction and selection-bias estimates (methodology log entry 61).
- **v2.3 (2026-10-23, amendment A2; post-design, pre-result for Exp 8):** on the recorded atlas lobes12 the Exp 8
  anatomical layer (`lobe_hemisphere`) is the zero matrix; Exp 8 is run as `exp8a_xu_dual_gcn_surface` with a
  cortical-surface contiguity layer 2, in its own namespace and code pin (`FINAL_v1/amendments/A2_exp8_surface`).
  Exp 0-7, the release pin, the atlas and graph decisions and the protocol are unchanged. Final-analysis integration:
  proposal in methodology log entry 60.

- **v2.2 (2026-10-19, after the exact-BEM resolution result; author authorization):** two additional atlas arms,
  dk68 the reference: **lobes12** (recommended by the unchanged `atlas_resolution` rule) and **r18** (exploratory,
  post-result approval, amendment A1; fails the LCMV own-region dominance criterion). Their stores are built from the
  original recordings only under `experiments.decisions.atlas_authorization`; notebook 04 compares the three atlases
  on the pre-specified models (development CV only; exploratory, uncorrected) and selects nothing. Methodology log
  entry 59b.

- **v2.1 (2026-10-18, before any final-framework result; author approval of the 18-region investigation):**
  - **Atlas:** r18 stores (LCMV, wMNE; modes B, A) built from the original recordings by the unchanged pipeline,
    conditional on the anatomical checks and the exact-BEM resolution; validated, matched and QC'd. A controlled,
    development-only dk68 vs r18 comparison (`configs/final/roi_comparison.yaml`: Exp 1, Exp 4 `plv__psd`, Exp 5
    `plv__separate__mean`; both inverses; mode B; dense PLV; same participants, folds, seeds) reports resolution /
    reliability and classification evidence separately; the author records the analysis atlas. dk68 stays the
    reference.
  - **Order:** graph-construction comparison after the atlas decision, on that atlas only.
  - **Holdout:** evaluated only in the frozen final analysis (`configs/final/holdout.yaml`: every pre-specified variant
    of Exp 0-6 and 8, mode B, both inverses, on the recorded atlas and graph), after development CV is complete;
    refused by the code otherwise. Exp 7 (exploratory) and mode A (sensitivity) are development-CV only.

- **v2.0 (2026-10-18, final framework; author request "one unified, final implementation"; methodology log entry 58, decisions row 61):**
  - **When / what was known:** after the LCMV-Xu and wMNE-Xu suites (Exp 0-8, fixed-60, ES200, R1) had been run and their
    development-CV and holdout results inspected (methodology log entries 50-57). The holdout is therefore **no longer
    untouched**: in v2.0 it is evaluated once per final configuration and reported descriptively, never used for any
    selection.
  - **Data:** ds004504 only (65 participants: 36 AD / 29 HC; FTD excluded by design), the four validated stores (LCMV /
    wMNE x window modes B / A, 68 Desikan-Killiany ROIs, 5 Xu bands), read only and locked. Mode B is primary, mode A
    the sensitivity analysis.
  - **Settings:** the frozen participant split of the Xu suites only: `xu_dev_cv` (5-fold participant CV inside the 52
    development participants; folds identical to the earlier suites) and `xu_holdout` (fit on the 52, test once on the
    13). The D1 / D2 settings S1-S5 are not part of v2.0.
  - **Experiments:** Exp 0-8 as in the Xu suites, unchanged definitions, except that shared-encoder multiband models
    always carry the explicit one-hot band identity (entry 57); the band-permutation-invariant variants are not final
    experiments. No architecture added.
  - **Training:** GNNs exactly 60 epochs (protocol `gnn_fixed60_v1`, the v1.5 fixed configuration; no early stopping,
    no search); classical models keep their participant-level inner CV for C.
  - **Reporting:** participant-level pooled out-of-fold AUC primary, with mean / range of fold AUCs, balanced accuracy,
    sensitivity, specificity, confusion matrix, DeLong / bootstrap / Wilson CIs; LCMV vs wMNE and mode B vs A paired
    DeLong (descriptive).
  - **New, pre-specified before any v2.0 result:** participant QC gate (category A blocks until decided; nobody excluded
    automatically); graph-construction comparison with a development-only selection rule (dense kept unless a
    construction is better by >= 0.02 AUC in >= 3 of 4 cells); atlas choice on source-resolution criteria, not on
    classification results (`docs/final/parcellation.md`).
- **v1.7 (2026-10-14):** **Dataset 3** (OpenNeuro ds004504; 36 AD, 29 HC; FTD excluded by design; `docs/dataset_d3.md`, methodology log entry 44) is added as a third cohort with its own within-dataset setting **S5 `within_D3`** (same subject-level repeated stratified 5-fold CV, metrics and evidence criteria as S1/S2). D3 uses the provider's preprocessed derivatives with minimal added processing, so its pipeline is not identical to D1/D2. Cross-dataset tests involving D3 are not part of this version: they need a combined source-space store and a revision of the §7 gate. S1–S4 and every D1/D2 result are unchanged.
- **v1.6 (2026-10-14):** Experiment 7 (EARM-GNN, `docs/earm_gnn.md`, methodology log entry 43) is added as an **exploratory extension** designed after the Tier 1 GCN results had been observed. It uses the v1.5 protocol unchanged (fixed hyperparameters, no inner CV, 10 repeats with the shared splits, all settings, PLV and |ciPLV| separate) with two pre-specified ablations (fixed adjacency, mean fusion). The primary analyses, comparators and evidence criteria (§2, §5, §8) are unchanged; Exp 7 comparisons are descriptive.
- **v1.5 (2026-10-14, protocol amendment for computational feasibility; author decision; methodology log entry 42, decisions row 50):**
  - **When:** after Exp 0 and Exp 1 (complete) and after ONE nested-CV GNN result had been observed (Exp 2, PLV / theta / GCN / within-D1, subject-level AUC 0.522; 1 of 120 Exp 2 units). The nested protocol was measured at ≈ 59 s per outer fold (17 GNN fits), i.e. weeks of GPU time for Exp 2–6.
  - **GNN hyperparameters are fixed; there is no hyperparameter search of any kind and no inner CV for GNNs.** One configuration per architecture (`configs/experiments/gnn_fixed_base.yaml`, protocol `gnn_fixed_hparams_v1`): 2 graph-convolution layers, hidden 32, dropout 0.3, mean+max readout, MLP head 32; Adam, learning rate 1e-3, weight decay 1e-4, batch size 32, 60 epochs, no early stopping, the model after the last epoch is evaluated. These are the defaults defined in `gnn_base.yaml` before any GNN result (2026-10-12), not values chosen from results. Each outer training fold is fitted exactly once; outer splits, seeds and all fitting-on-training-subjects rules are unchanged (§3).
  - **Tier 1 schedule (10 repeats, seeds 0–9):** Exp 2 and Exp 3 with GCN, 4 bands × {PLV, |ciPLV|}, settings S1, S2, S4; Exp 4 both measures, all settings; Exp 5 shared GCN encoder × {mean, attention} fusion × both measures, all settings; Exp 6 full model with both measures, all settings, and its seven ablations with PLV in S1, S2, S4. **Tier 2 (optional, secondary):** EdgeWeightedGAT and WeightedGraphSAGE in Exp 2/3, the other Exp 5 variants, |ciPLV| Exp 6 ablations. No band, fusion, encoder or experiment was chosen from held-out results.
  - **Permutation tests for GNNs are deferred**; no permutation p-value is reported unless a valid test is completed (§4.2, §8).
  - The nested-CV Exp 2 checkpoint is kept on Drive as a historical record (`experiments/exp2_conn_minimal/`) and is never pooled with or reported as a fixed-protocol result. Classical models (Exp 0, Exp 1, baselines) keep their nested inner CV over their pre-specified C grid.
- **v1.4 (2026-10-14):** graph construction moves to pipeline v4, "Boundary-Aware Filtering and Hilbert Phase Estimation" (methodology log entry 41, `docs/boundary_processing.md`):
  - the 10 s edges are discarded before any filtering;
  - padding is explicit;
  - the analytic signal is a finite complex FIR;
  - windows keep a boundary margin of 4537 samples (18.148 s), so there are fewer windows per recording.

  The window length, overlap, bands, connectivity definitions, montage, interpolation, source model and every evaluation rule are unchanged. The same rules apply to D1 and D2.

  **Frozen v4 graph dataset** (input of every experiment): `graphs_v4_33faa4b89a5d`, `graph_run_id` `6d75de9975f0d9ae`, commit `e637368`; 43/43 recordings (D1 20, D2 23), none excluded; 1,706 candidate windows, 179 rejected; **1,527 graphs** (D1 907, D2 620; AD 729, HC 798); 6,108 band-specific adjacency matrices per connectivity measure; 23/23 integrity checks passed (v3: 1,640 graphs).
- **v1.1:** primary segmentation frozen at 20 s windows with 50 % overlap (both
  datasets). Experiment set 0–6 fixed (§10). The code for every setting exists,
  but nothing has been run on real data.
- **v1.2:**
  - Exp 3 primary uses matching-band PSD (F = 2); all-band PSD becomes
    secondary Exp 3b.
  - GAT and GraphSAGE are reported as EdgeWeightedGAT and WeightedGraphSAGE.
  - GNNs use |ciPLV|; signed ciPLV is stored separately.
  - SVM calibration is subject-grouped.
  - The age sensitivity analysis is implemented.
  - The repeat, fold, epoch and grid counts are **provisional** (§3).
- **v1.3:**
  - The primary SVM is scored by its decision function, with threshold 0.
    Grouped Platt calibration is optional (§4.1).
  - The signed-ciPLV sign is not interpreted as direction.
  - The relative-PSD denominator is checked against the preprocessing passband.
  - See `docs/methodology_architecture_log.md`.

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

**Unit of analysis.** The model sees one sample per **20 s window**. Windows
overlap by 50 % (5000-sample windows, 2500-sample step at 250 Hz), with the
same rule for D1 and D2. All filtering happens on the continuous signal before
segmentation. The **subject** is the unit of inference. The subject key is
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

> Frozen here: subject-level splitting, nested subject-level inner CV for
> selection, the per-fold fitting rules, and the fixed threshold. **Provisional
> and configurable, not yet final:**
> - K = 5 outer folds, 20 repeats, inner K = 4
> - epochs, early stopping, hyperparameter grids
> - bootstrap and permutation counts
>
> These numbers are defaults in `configs/experiments/base.yaml`.
>
> **v1.5 amendment (GNNs only):** GNNs use fixed hyperparameters and no inner
> CV (`configs/experiments/gnn_fixed_base.yaml`), with 10 repeats (seeds 0–9);
> see the changelog. The inner loop below applies to the classical models.

### 3.1 Within-dataset (S1, S2)

- **Outer loop:** repeated stratified K-fold over subjects. K = 5, with
  **20 repeats** using seeds `0..19`.
  - Folds are built from the list of subjects, stratified by label. Each
    subject's windows all go to that subject's fold.
  - D1 (10/10): each test fold has 4 subjects (2 AD, 2 HC).
  - D2 (10/13): each test fold has 4–5 subjects with 2 AD each, and
    stratification keeps the HC share as even as possible.
- **Inner loop (model selection; classical models only from v1.5):** stratified K = 4 subject-level CV inside
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
  selected by the §3.1 inner CV run on the source dataset only (classical
  models; GNNs use their fixed configuration, v1.5).
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

- **Subject score** is the mean of that subject's window scores for AD.
  - LR, GNNs and the optional calibrated SVM give probabilities.
  - The primary SVM gives its `decision_function` margin. ROC-AUC uses the
    margin directly, with no calibration.
- **Decision threshold** is fixed on the score's own scale: 0.5 for
  probabilities, 0 for SVM margins. It is not tuned on test data.
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
- **Uncertainty:** each subject's out-of-fold score is averaged across
  the 20 repeats. A 95% CI then comes from 2,000 bootstrap resamples **of
  subjects** on those averaged predictions.
- **Significance against chance:** a subject-level label-permutation test.
  Labels are permuted across subjects, the full CV pipeline is re-run, and
  1,000 permutations are used (at least 200 if compute-limited, with the
  number reported). The p-value is computed as
  `(1 + #perm ≥ observed) / (1 + n_perm)`.
  **v1.5:** permutation tests are deferred (none has been run, for any model).
  No permutation p-value is reported unless a valid test, re-running the full
  CV pipeline, is completed; until then the §8 criterion "permutation p < 0.05"
  is reported as not assessed.
- **Model comparisons** (GNN vs. baseline on the same folds) use paired
  differences per repeat with the corrected resampled t-test of Nadeau and
  Bengio. They are reported as effect size + CI. With small N, "no
  significant difference" is reported as such.
- **Per-dataset breakdown in S3:** pooled out-of-fold predictions are also
  scored separately for D1 subjects and D2 subjects (§6).
- **Per-subject predictions** (subject key, true label, mean score per
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
   (`baseline_conn_classical.yaml`; ciPLV is a separate variant.)
4. **Single-band GNN vs. multi-band GNN,** the ablation that supports the
   paper's multi-band claim (Experiments 2/3 vs. 4/5).
5. **Age-adjusted sensitivity analysis,** S2 in particular. Edge features are
   residualized on age with a regression fitted on training subjects only, and
   the model is retrained. *Implemented as "EEG + age" instead:*
   - Classical models append age, standardized on the training subjects, to the
     feature vector.
   - GNNs concatenate it to the pooled graph embedding before the classifier.
   - Configs: `configs/experiments/age/`.
   - The comparison is age-only vs. EEG-only vs. EEG + age on identical splits.

The age-only baseline is `baseline_age.yaml`.

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

| Item | Requirement | Pipeline v4 (`configs/pipeline.yaml`) | Verified by |
|---|---|---|---|
| Channel set | Identical channel names and count entering source modelling for D1 and D2 | ✅ 30 channels in both datasets (M1/M2 dropped), canonical order | loader raises on a missing channel; channel order is asserted against the operator |
| Electrode positions | Same montage source for both | ✅ `standard_1020` by name (MNE ≥1.13 calls it `colin27_1020`, with identical positions) | |
| Reference | Average reference computed over the same common channel set | ✅ | |
| Bad channels | Same detection and handling rule for both | ✅ the same automatic rule, plus interpolation | per-subject list in `subjects.csv` |
| Units | Amplitudes verified in volts for both | ✅ the D1 rescale, plus a plausibility check for both datasets | `median_std_uv` in `subjects.csv`, to be compared between D1 and D2 |
| Resampling / filtering | Same target rate (250 Hz), same filter design and parameters | ✅ | |
| Windowing / rejection | Same window length, step, trim, rejection rule and minimum-window rule | ✅ 20 s / 50 % overlap for both; 10 s cropped before filtering and a 4537-sample boundary margin for both (v4) | `manifest.segmentation`, `manifest.boundary_processing` |
| Boundary handling | Same crop, padding, analytic-signal method and margin for both | ✅ v4: crop at the original rate, `reflect_limited` padding, finite analytic FIR, margin computed from the filters | `manifest.boundary_processing`, `scripts/validate_graph_run.py` |
| Forward model | Same template, source space, BEM and **same sensor set** | ✅ **one** forward model shared by both datasets | `manifest.source_operator.shared_across_datasets` |
| Inverse | Same method, `loose`, `depth`, `λ²` and noise covariance | ✅ ad hoc covariance, `pick_ori='normal'`, all recorded | runtime check against the full inverse (`verification_rel_error`) |
| ROIs | Same parcellation, ROI order and extraction mode | ✅ | `roi_names` in the manifest and `graphs.npz` |
| Bands | Identical band edges and the same band-pass/Hilbert implementation | ✅ (v4: finite analytic FIR for both) | `manifest.boundary_processing.analytic_kernels` |
| Graph construction | Same estimator, window length, sparsification, node features and edge normalization | ✅ at build time (dense PLV/ciPLV and band power). Sparsification and node features are fixed at the model stage and applied identically to both datasets | model config |
| Provenance | Both datasets processed by the same code commit and config hash in one run | ✅ a single `run()` covers D1 and D2 | `git_commit`, `config_hash` in the manifest |

**Gate status:** the 7.1 items are satisfied *by design* since pipeline v2
(v3 adds 20 s windows and PSD features; v4 adds the boundary-aware processing). The gate is formally
passed once a full v4 run has completed, with:

- `verification_rel_error` < 1e-6
- no subject in error
- `median_std_uv` ranges of D1 and D2 that overlap

Graph sets from pipeline v1 (`processed/*_20260912_*`, `*_20260915_*`) do not
pass the gate and are not used.

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
- Permutation p < 0.05 (v1.5: deferred; reported as not assessed until run)
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

Saved to Drive under
`processed/experiments/<experiment>_<timestamp>/<variant>/<setting>/`, never
committed. The file list is in `docs/experiments.md` §5:

- run manifest (config, config hash, commit, seeds, software versions)
- subject→fold assignment
- per-window and per-subject predictions
- per-repeat metrics
- permutation null distribution

Aggregate tables only (no per-subject rows) go to `results/tables/`.

---

## 10. Experiments

Full details (inputs, tensor shapes, configs, leakage architecture) are in
`docs/experiments.md`.

| Exp | Question | Input | Models |
|---|---|---|---|
| 0 | Conventional sensor-space spectral baseline (Paper 2) | electrode PSD | LR, SVM |
| 1 | Does source localization alone help? | 68-ROI source PSD | LR, SVM |
| 2 | Value of source connectivity alone | single-band PLV / \|ciPLV\| graph, constant node feature | GCN, EdgeWeightedGAT, WeightedGraphSAGE |
| 3 | Connectivity + local spectral activity | single-band graph, matching-band ROI PSD [log, rel] (Exp 3b: all bands, secondary) | GCN, EdgeWeightedGAT, WeightedGraphSAGE |
| 4 | Simple graph baseline | band-averaged graph, ROI PSD node features | GNN |
| 5 | Multi-band GNN (candidate: learned fusion) | 4 band graphs, matched-band PSD | GNN, shared/band-specific encoders, mean/concat/gate/attention fusion |
| 6 | Richest model, with ablations | 4 band graphs, [log PSD, rel PSD, strength] per band | GNN + attention fusion; ablations: no PSD, no strength, no learned fusion, each band dropped |

**v1.5:** every GNN experiment above runs under the fixed-hyperparameter protocol
(`*_fixedhp` configs). Tier 1 (primary): Exp 2/3 GCN (S1, S2, S4), Exp 4, Exp 5
shared encoder × {mean, attention}, Exp 6 full (both measures) + PLV ablations
(S1, S2, S4); 10 repeats. EdgeWeightedGAT, WeightedGraphSAGE, the other Exp 5
variants and the |ciPLV| ablations are Tier 2 (secondary, optional).

Conventions:

- PLV is primary in every experiment. ciPLV is run as a separate, parallel
  sensitivity analysis and is never merged with PLV.
- All experiments use the same subject-level splits (same seeds) in each
  setting, so paired comparisons (§4.2) are valid.
- The primary comparison (§2) is Experiment 5 or 6 (learned fusion) against
  Experiments 0, 1 and 4, the age-only baseline, and the classical-connectivity
  baseline, in S1 and S2.
