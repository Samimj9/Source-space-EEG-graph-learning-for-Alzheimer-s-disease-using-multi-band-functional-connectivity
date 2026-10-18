# Participant quality control before classification

Code `adgraph.qc` (`participants`, `report`, `fold`, `difficult`); notebook `01_participant_qc`;
`python scripts/final.py qc --inverse <lcmv|wmne> --mode <B|A>`. Methodology log entry 58. Tests `tests/test_qc.py`.

**No participant is excluded or reweighted by any rule.** All 65 participants stay in the analysis unless the author
documents a scientific reason, which is then a logged protocol amendment (it changes the frozen split), never an
automatic QC action.

## 1. When it runs

Every experiment call (`plan.open_context`) loads, or computes once, the QC report of its store
(`FINAL_v1/qc/<inverse>_<mode>_<atlas>/`, completion marker tied to the store's graph_run_id and the QC code). The
experiment refuses to start while a category A finding has no recorded decision, or when a decision asks for an
exclusion. QC cannot be skipped: `plan.run_experiment` refuses a real (non-synthetic) store passed without its QC report
(tested), and `01_participant_qc` computes the four reports before any experiment notebook.

## 2. Three categories, kept apart

| Category | Meaning | Detection | Consequence |
|---|---|---|---|
| **A** technical corruption | the data are not a valid measurement | non-finite values in any array; PLV / \|ciPLV\| outside [0, 1], non-zero diagonal or asymmetric; non-positive absolute power; a feature constant over all windows of a participant (flat); degenerate connectivity (mean off-diagonal PLV of a window > 0.99: identical ROI signals); duplicate window ids; window counts disagreeing with `subjects.csv`; a source operator that does not reproduce MNE's own inverse on the participant's data (`lcmv_` / `wmne_verification_rel_error` > 1e-6, the builder's tolerance) | blocks the experiments until the author records a decision |
| **B** statistically unusual, valid | a real but atypical EEG | label-blind participant summaries (per band: ROI-median log / relative PSD and DE, sensor log PSD, mean PLV and \|ciPLV\| node strength and its spread over ROIs; across-window variability; rejection statistics and amplitude (`median_std_uv`); recording structure (boundary events, segments, analysed pieces); the participant's LCMV operator rank; source-minus-sensor log power; the spread of source log power over ROIs per band, which a degenerate source estimate collapses); modified z-score (Iglewicz & Hoaglin; median / MAD of the 52 development participants) > 3.5 for any summary, or robust Mahalanobis distance (MCD on the leading 5 principal components of the robust z-scores; chi-square 0.999) | flagged and reported; kept |
| **C** difficult to classify | consistently misclassified | from development-CV out-of-fold predictions of the completed models: misclassified by >= 75 % of the models (`scripts/final.py difficult`) | reported with its A / B flags; kept |

The diagnosis is never used to compute a summary or a threshold (tested); it only annotates the report (flags by
diagnosis, description only).

**Technical vs physiological vs difficult.** Category A uses fixed technical criteria that no valid EEG can meet
(they are independent of the data distribution and of the diagnosis). Category B is relative to the development
participants: AD shifts the spectrum towards slower rhythms, so an AD participant can be a robust outlier for valid
physiological reasons; B is therefore never a reason to exclude, only a pointer to inspect (sensor vs source power,
rejection rate, recording structure). Category C is a property of the models, computed after classification, and
only described.

## 3. Fold-level thresholds (training participants only)

For each outer split the runner recomputes the modified z-scores of every participant against that split's training
participants only (`qc.fold.fold_qc`) and records the flags in `fold_qc.csv` and `training.csv`. The protocol's
`qc.fold_policy: report` means these flags document how test participants relate to the training distribution; they
never remove, reweight or otherwise influence training or prediction.

## 4. Recording a decision (category A only)

`<processed>/FINAL_v1/qc/decisions.yaml` is written by the author, never by code:

```yaml
D3-23:
  decision: keep          # keep | exclude (exclude stops the experiments: it needs a logged protocol amendment)
  reason: "isolated non-finite DE value in one window traced to ...; window retained"
  date: 2026-10-20
```

## 5. Outputs

`participant_summaries.csv`, `window_summaries.csv.gz`, `technical_problems.csv` (A), `robust_flags.csv` and
`flagged_features.csv` (B), `qc_summary.json`, `qc_report.md`, `pca.png` (an exploratory projection, labelled as such;
not a result), `qc.done.json`; `difficult_<time>.csv` (C).
