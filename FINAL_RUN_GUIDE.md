# Final run guide (release final-1.0)

Branch `claude/unified-final-framework`. Open each notebook from `notebooks/README.md` (Colab links), *Runtime → Run
all*, one notebook at a time, in this order (`configs/final/run_plan.yaml`, tested). Every notebook is resumable: if
Colab disconnects, open it again and *Run all* again; finished work is kept on Drive and skipped. Details:
`docs/final/framework.md`. **How to begin: §6.**

## 1. Execution order

| # | Notebook | Runtime | Real EEG? | What it does | Estimated time |
|---|---|---|---|---|---|
| 1 | `00_setup_and_stores` | CPU | reads stores | verifies the 4 validated 68-ROI stores, writes the release lock, runs the test suite once | 15-25 min |
| 2 | `01_participant_qc` | CPU | reads stores (diagnostics) | QC of the 4 dk68 stores; **stops if a category A finding needs your decision** | 10-20 min |
| 3 | `02_parcellation` | CPU | diagnostics, then **reconstruction** | r18 atlas checks; analytic resolution with the stores' BEM; resolution of the stores' own operators; reliability; then (approved, conditional) the **r18 stores from the original recordings** for LCMV and wMNE, modes B and A, validated, matched to each other and window by window to dk68; QC of the 4 r18 stores | 2.5-3.5 h |
| 4 | `04_roi_68_vs_18` | GPU | **training** (development CV) | dk68 vs r18: Exp 1 (4 variants), Exp 4 `plv__psd`, Exp 5 `plv__separate__mean`, both inverses, mode B, dense PLV; report; **you record the atlas** | ≈ 3-4 h CPU equiv. |
| 5 | `03_graph_construction` | GPU | **training** (development CV) | graph diagnostics and the 11-construction comparison on the recorded atlas; the recorded selection | dk68 ≈ 24 h CPU equiv.; r18 less (not measured) |
| 6-14 | `Exp0` … `Exp8` | CPU (0, 1) / GPU | **training** (development CV) | one experiment each; LCMV B, wMNE B, LCMV A, wMNE A (Exp 0: sensor B, A) on the recorded atlas and graph | see below |
| 15 | `99_results_report` | GPU | **training** (holdout fits) | development-CV report, difficult participants; **freeze** the final analysis; holdout of the frozen units only; final tables | ≈ 13 h CPU equiv. for dk68 (GPU less) |

Development CV only until notebook 99. The holdout is evaluated once, descriptively, for the units of the frozen final
analysis (`configs/final/holdout.yaml`: every pre-specified variant of Exp 0-6 and 8, mode B, both inverses, on the
recorded atlas and graph); the code refuses it for anything else (ROI comparison, graph comparison, Exp 7, mode A).

**Time estimates** are derived from measured fit times, not guessed: 60-epoch fits measured on 2 CPU threads at the
mode A and mode B sizes of the 68-ROI stores, consistent with the saved mode A runs on a Colab CPU runtime (Exp 5
21-32 s, Exp 8 11-13 s per fit). Per fold fit in mode B on Colab CPU: Exp 2-4 ≈ 3 min, Exp 5 ≈ 10, Exp 6 ≈ 12, Exp 7
≈ 30, Exp 8 ≈ 5 min; mode A ≈ 1/25 of that. Development CV of Exp 0-8 on dk68 (both inverses, both modes, 5 fits per
variant and context): Exp 2 ≈ 5 h, Exp 3 ≈ 5 h, Exp 4 ≈ 2 h, Exp 5 ≈ 13 h, Exp 6 ≈ 20 h, Exp 7 ≈ 30 h, Exp 8 ≈ 8 h
CPU-equivalent (≈ 84 h in total; Exp 0, 1 < 1 h each). r18 graphs are 18 x 18 instead of 68 x 68, so r18 runs should
be faster; this has not been measured. A T4 GPU has not been measured for these models either; the real per-fold
seconds appear in every status table after the first fold. The dk68 development-CV units of notebook 04 are reused by
the experiment notebooks if dk68 is recorded (and by notebook 03 for its `dense` cells).

**Storage** (Drive): the r18 stores need ≈ 1-2 GB in total for both inverses and both modes (mode B assembled arrays
≈ 0.26 GB per inverse for 16,420 windows: PLV and |ciPLV| 5 x 18 x 18 float32 per window plus PSD / DE; participant
checkpoints of the same size; continuous 18-ROI signals ≈ 0.35 GB per inverse; mode A ≈ 0.01 GB). Notebook 02 prints
the real sizes (`build_stores.py size`). Results of all notebooks: a few hundred MB (predictions, metrics, figures).
Keep ≥ 5 GB free on Drive.

## 2. Configurations

* Stores (read only): LCMV `D3_LCMV_XU/graphsLX_v1_811690218bfa_corr1` (B), `graphsLX10_v1_811690218bfa_corr1` (A);
  wMNE `D3_WMNE_XU/graphsWX_v1_c32bc6c67f0e` (B), `graphsWX10_v1_c32bc6c67f0e` (A); 68 DK ROIs, 5 Xu bands.
* New r18 stores (notebook 02): `D3_LCMV_XU/graphsLX-r18_v1_<hash>`, `graphsLX10-r18_v1_<hash>`,
  `D3_WMNE_XU/graphsWX-r18_v1_<hash>`, `graphsWX10-r18_v1_<hash>` (`configs/source_graphs.yaml` with
  `source.atlas: r18`; otherwise identical); 18 regions, 9 per hemisphere (`source.atlas`).
* Protocol: frozen 52 / 13 split, `xu_dev_cv` 5 folds; `xu_holdout` only in the frozen final analysis
  (`configs/final/holdout.yaml`, notebook 99); participant-level metrics primary; holdout descriptive only
  (inspected before; never used for any selection).
* GNNs: `gnn_fixed60_v1`, exactly 60 epochs, Adam 1e-3, wd 1e-4, batch 32, no early stopping, no search.
* **Graph constructions** (`configs/final/graphs.yaml`; k = round(f (R - 1)): R = 68 → k = 7 / 13 / 20, R = 18 →
  k = 2 / 3 / 5):

  | name | rule |
  |---|---|
  | `dense` | weighted PLV / \|ciPLV\| (reference, default) |
  | `knn_union_f10/f20/f30` | union kNN, k = 7 / 13 / 20 (dk68), 2 / 3 / 5 (r18) |
  | `knn_mutual_f10/f20/f30` | mutual kNN, same k, isolated nodes repaired |
  | `proportional_f20` | global strongest edges, density 20 % |
  | `contiguity`, `contiguity_knn_union_f20` | restricted to cortically contiguous regions |
  | `feature_similarity_knn_union_f20` | PSD / DE similarity graph (reported, never selected) |

  Selection (development CV only, on the recorded atlas): Exp 4 `plv__psd` and Exp 5 `plv__separate__mean`, LCMV
  and wMNE, mode B; a construction replaces `dense` only if mean AUC ≥ dense + 0.02 and better in ≥ 3 of 4 cells
  (a pre-specified exploratory rule, not a superiority test).

## 3. Three Scientific Decisions Before Training

Full classification (Exp 0-8) does not start until these three decisions are settled. Nothing below used a
classification result, and nothing read the holdout. Status (2026-10-18): r18 stores and the ROI comparison approved
(§3.1), the graph comparison approved on the recorded atlas (§3.2), QC as described (§3.3).

### 3.1 Cortical ROI resolution (68 vs 34 vs 12-18 regions)

**What was investigated.** Source resolution of four nested atlases (merged whole Desikan-Killiany labels of one
hemisphere): `dk68`, `dk34_gyral`, `r18` (9 regions per hemisphere), `lobes12`. For both inverses, with the stores'
exact inverse settings, from the fsaverage forward model and the 19 template electrodes: spatial rank of the
leadfield, rank of the ROI operator (ROI signals forced to be linear combinations of the others), cross-talk and
own-region dominance from the region resolution matrix, leakage correlation. Also whether coarse-ROI signals can be
recomputed from the stores' saved intermediates. No EEG and no diagnosis enter the resolution analysis
(`docs/final/parcellation.md`).

**Quantitative evidence** (approximate 3-layer BEM; a sphere model agrees within ≈ 0.02):

| | dk68 | dk34_gyral | r18 | lobes12 |
|---|---|---|---|---|
| redundant ROI signals (operator rank 18 / 18 / 18 / 12) | 50 | 16 | 0 | 0 |
| median cross-talk, wMNE / LCMV | 0.85 / 0.86 | 0.71 / 0.72 | 0.49 / 0.53 | 0.39 / 0.46 |
| ROIs dominated by their own region, wMNE / LCMV | 0.53 / 0.57 | 0.74 / 0.74 | 0.89 / 0.89 | 1.00 / 1.00 |

The average-referenced leadfield has rank 18 (entropy-effective rank 13.3; 10 singular values hold 95 % of the
energy). With 68 ROIs most of the 68 x 68 connectivity entries are shaped by leakage, not by distinct regions.
*Reconstruction from intermediates:* the saved operators W (rank 18, reject the constant) let the average-referenced
sensor signals be recovered exactly from the saved ROI signals (pinv(W) * roi; tested), but the analytic-filter
margins were not saved, so the first and last few windows of every piece could not be recomputed exactly; averaging
68-ROI signals or PLV matrices is never used (sign flips, phase synchrony does not average).

**Limitations.** Template anatomy and template electrode positions (as in the stores); the stores' exact 5120 BEM
was unavailable here (notebook 02 reruns the analysis with it); LCMV uses a model covariance rather than each
participant's data covariance; resolution under independent region activity. The two numeric thresholds of the rule
(median cross-talk ≤ 0.60, own-region dominance ≥ 0.80) were set after these provisional numbers were seen; the
threshold-free criterion "no more regions than the operator rank" alone gives the same choice.

**Exact-BEM result (notebook 02, 2026-10-19):** r18 LCMV own-region dominance 0.778 < 0.80 (wMNE 0.889; median
cross-talk 0.536 / 0.511), so r18 does not meet the pre-specified rule, which recommends **lobes12**. Notebook 02 stops
before the r18 build, as designed; nothing was changed in response (methodology log entry 59a). The text below
describes the r18 plan as approved before this result; it needs your new decision.

**Recommended default (before the exact-BEM result).** **r18** as the analysis atlas for both inverses, with **dk68** kept as the reference /
sensitivity analysis (and the link to all earlier results), **conditional on** notebook 02 (exact BEM, the stores'
own operators, validated r18 stores) and the development-only comparison of notebook 04. Code never makes r18 the
analysis atlas: you record it (`final.py choose-atlas`). A rank-18 sensor space does not prove that 18 regions are
independently or accurately localized: the median r18 region still receives about half of its estimate from other
regions.

**Alternatives requiring development-only validation.** None is selected by classification results (by design the
atlas follows source resolution only). `dk34_gyral` and `lobes12` stay available as sensitivity analyses, reported
descriptively on development CV.

**Additional computation.** Yes: r18 stores for LCMV and wMNE, modes B and A, built from the original recordings with
the unchanged pipeline (`scripts/build_stores.py build | validate | match | match-atlas --atlas r18`; notebook 02 §4;
≈ 40 min LCMV + 30-45 min wMNE CPU). New directories next to the 68-ROI stores; those are not touched.
`final.py reconstruct-check` (notebook 02 §3, read-only) measures the reconstructable fraction on the real stores.

**Approved (2026-10-18):** building the r18 stores, conditional on the atlas checks and the exact-BEM resolution
(notebook 02 stops otherwise). **Still yours:** the analysis atlas, after notebook 04 (recorded once in
`FINAL_v1/decisions/atlas.json` with your reason; also log it in `docs/methods_decisions.md`). If you keep dk68, the
paper states the leakage limitation quantified here.

**The controlled comparison (notebook 04, `configs/final/roi_comparison.yaml`).** Same 52 development participants,
same five folds, mode B, same training and seeds (seeds depend on the fold only; tested), LCMV and wMNE: Exp 1
(LR / SVM x PSD / DE), Exp 4 `plv__psd`, Exp 5 `plv__separate__mean`, GNNs on dense PLV at 60 epochs. Reported
separately: (1) source localization / resolution: forced redundancy, cross-talk, own-region dominance of each
participant's operators actually built, against the stores' forward model; split-half feature reliability; (2)
classification: participant-level pooled out-of-fold AUC (DeLong CI), balanced accuracy, F1, sensitivity,
specificity, paired r18 - dk68 differences (paired DeLong; participant-bootstrap CIs). The summary is descriptive
("no material difference" below |0.02| mean AUC difference, "tendency" when ≥ 75 % of the cells agree in sign,
otherwise "inconsistent"); it never declares a winner, and an AUC difference is not evidence about localization.

### 3.2 Graph construction (dense, top-k, union kNN, mutual kNN)

**What was investigated.** Every construction of `configs/final/graphs.yaml` for symmetry, self-loops, weight
handling, isolated nodes and density (tested on synthetic graphs and the synthetic store); the GCN normalization;
how much of each ROI's own PSD / DE node feature survives GCN propagation for each construction
(`docs/final/graph_construction.md` §4).

**Quantitative evidence.** All candidates are symmetric, have a zero diagonal (the model adds exactly one unit
self-loop: D^-1/2 (A + I) D^-1/2, finite also for an isolated node), keep the exact weights of retained edges, never
keep zero or negative edges, and kNN constructions leave no node isolated. GCN self-weight (share of an ROI's own
feature per layer): dense 0.05, union kNN k = 7 / 13 / 20: 0.20 / 0.14 / 0.10, mutual kNN k = 13: 0.17; across-ROI
feature variance kept after one layer (independent features, lower bound): dense ≈ 0.003, union kNN k = 7 ≈ 0.10.
Directed per-node top-k is not symmetric; it is analysis only (`symmetrize: none`), not a candidate.

**Limitations.** These are structural properties and a mechanism (dense PLV graphs make GCN layers average PSD / DE
over almost all ROIs), not evidence that kNN classifies better; real PSD / DE are spatially smooth, so the real
retained variance is higher than the lower bound. GNN run time does not fall with sparsity (dense batched products).

**Recommended default.** **`dense`** weighted PLV / |ciPLV| stays the default. kNN is not declared better.

**Alternatives requiring development-only validation.** The pre-specified comparison (`configs/final/selection.yaml`,
notebook 03): `knn_union_f10/f20/f30`, `knn_mutual_f10/f20/f30` (k = round(f (R - 1)): 7 / 13 / 20 for 68 ROIs,
2 / 3 / 5 for 18), `proportional_f20`, `contiguity`, `contiguity_knn_union_f20` (feature-similarity graph reported,
never selectable). Reference models Exp 4 `plv__psd` and Exp 5 `plv__separate__mean`, both inverses, mode B,
development CV only. A candidate replaces dense only if mean participant AUC ≥ dense + 0.02 and it is better in ≥ 3
of 4 cells; the decision is written once, with its atlas, to `FINAL_v1/decisions/graph_construction.json`. The
holdout plays no part (tested). It runs once, on the atlas recorded in notebook 04 (k scales with R), never on dk68
merely because it was the earlier default. It is a pre-specified exploratory decision rule, not a test of
statistical superiority: read the per-cell AUCs with their CIs.

**Additional computation.** Yes: dk68 ≈ 24 CPU-equivalent hours (220 fold fits; GPU faster); r18 less (not
measured), notebook 03 after 04.

**Approved (2026-10-18):** the comparison with the stated candidates, k fractions, 0.02 margin and 3-of-4 rule, on
the recorded atlas.

### 3.3 Participant-level QC

**What was investigated.** A QC stage before every experiment (`docs/final/qc.md`): label-blind participant
summaries of spectral features (source and sensor PSD / DE per band), source estimates (source-minus-sensor power,
spread of source power over ROIs, LCMV operator rank, operator verification error), connectivity (PLV / |ciPLV|
strength and spread) and signal-quality metadata (rejection rate, amplitude, boundary events, segments, pieces).

**Quantitative evidence** (synthetic stores; the real reports are produced by notebook 01): injected NaN, invalid
PLV, flat features, wrong window counts and an operator that does not reproduce MNE (> 1e-6) are detected as
category A for exactly the damaged participant; a participant with 10^4 times more power is flagged category B and
kept; summaries are identical with flipped labels; reference statistics do not change when holdout values change;
the gate blocks unresolved category A findings and refuses "exclude" decisions; a real store is never run without
its QC report (all tested).

**Limitations.** Category B thresholds (modified z > 3.5; MCD chi-square 0.999) are conventional, not tuned; with
52 reference participants a few valid participants will be flagged by chance; AD itself shifts the spectrum, so B
flags can be physiological. Category C exists only after classification.

**Recommended default.** Three categories kept apart. A (technical corruption, fixed criteria independent of data
distribution and diagnosis) blocks the experiments until you record a decision; B (unusual but valid) is reported
and kept; C (difficult) is reported after classification. **Nobody is excluded or reweighted by code.** Fold-level
flags are recomputed against each fold's training participants only and are report-only (`qc.fold_policy: report`).

**Alternatives requiring development-only validation.** None proposed. Any learned QC rule that would affect
training (e.g. down-weighting B participants) would need its thresholds fitted on each fold's training participants
only and a pre-specified development-CV evaluation; it is not implemented.

**Additional computation.** Notebook 01 (dk68) and notebook 02 §5 (the four r18 stores), ≈ 10-20 min CPU each, no
training. Every experiment opens its store's QC report and refuses to start without it.

**Needs your approval.** A `keep` / documented decision in `FINAL_v1/qc/decisions.yaml` for every category A finding
notebook 01 reports; any exclusion is a logged protocol amendment (it changes the frozen split).

## 4. Output directories (Drive)

Inside the release folder (everything the notebooks compute except the r18 stores):

```
<processed>/FINAL_v1/CODE_COMMIT.txt, stores.lock.json
<processed>/FINAL_v1/qc/<inverse>_<mode>_<atlas>/, qc/decisions.yaml (yours), qc/difficult_<time>.csv
<processed>/FINAL_v1/parcellation/atlas_r18_<time>/, resolution_<time>/, store_resolution_<inverse>_<atlas>_<time>/,
                                  reliability_<inverse>_<mode>_<atlas>_<time>/, reconstruction_<inverse>_<mode>_<atlas>_<time>.csv
<processed>/FINAL_v1/stores/        dated re-checks of store records (never overwrite a store's own record)
<processed>/FINAL_v1/roi_comparison/compare_<time>/   classification.csv, *_resolution.csv, feature_reliability.csv,
                                                      summary.json, report.md
<processed>/FINAL_v1/graphs/<inverse>_<mode>_<atlas>/diagnostics_<time>/
<processed>/FINAL_v1/decisions/atlas.json (yours, notebook 04), graph_construction.json (notebook 03),
                               final_analysis.json (notebook 99)
<processed>/FINAL_v1/results/<lcmv|wmne|sensor>/mode_<B|A>/<atlas>/<graph|none>/<experiment>/<variant>/<setting>/
<processed>/FINAL_v1/reports/tables_<time>/
```

Outside the release folder, next to the 68-ROI stores (new, versioned by configuration hash; written by notebook 02):

```
<processed>/D3_LCMV_XU/graphsLX-r18_v1_<hash>/, graphsLX10-r18_v1_<hash>/     LCMV r18, modes B / A
<processed>/D3_WMNE_XU/graphsWX-r18_v1_<hash>/, graphsWX10-r18_v1_<hash>/     wMNE r18, modes B / A
<processed>/D3_LCMV_XU/cache/, D3_WMNE_XU/cache/                               the existing forward-model / input-hash
                                                                               caches (reused; the forward model does
                                                                               not depend on the atlas)
<cache_dir>/subjects/fsaverage/                                                fsaverage (MNE download, if missing)
```

Each r18 store has its own `manifest.json` (configuration and its hash, code commit, software versions, input
SHA-256s, timestamps), `status.csv`, per-participant checkpoints, `validation.json`, `match.json` (wMNE) and
`atlas_match.json`. The validated 68-ROI stores, all earlier results and all earlier code pins are only read.

## 5. Needs your decision

1. Category A QC findings, if notebook 01 or 02 reports any (`qc/decisions.yaml`; §3.3).
2. After notebook 04: the analysis atlas (`ATLAS_DECISION` in its last cell; §3.1). Notebook 03 and the experiment
   notebooks stop until it is recorded.
3. After notebook 99 §1 (development CV complete): nothing; the freeze and the holdout follow the pre-specified plan.
   Review the holdout as descriptive only.
4. The deletion proposals of `docs/final/inventory.md` (nothing is deleted otherwise).

## 6. Begin here (notebook 00)

1. Open `notebooks/README.md` on GitHub (branch `claude/unified-final-framework`) and click
   `00_setup_and_stores.ipynb` (*Open in Colab*). A CPU runtime is enough.
2. *Runtime → Run all*. Allow Google Drive access when asked. Edit nothing (`UPDATE_CODE` stays `False`).
3. The setup cell clones the branch, finds `AD-LEB` / the outputs folder on Drive, writes `configs/paths.yaml`,
   records the code commit in `FINAL_v1/CODE_COMMIT.txt` (all later notebooks use that commit), and installs the
   pinned environment.
4. Expected result: a table with the four dk68 stores all `passed = True`, four "locked" lines, then the test suite
   (passes once per commit). If a store fails, stop and send the table.
5. Then open `01_participant_qc`, and continue in the order of §1, one notebook at a time. Stop points: a category A
   finding (01, 02 §5), 02 not building r18 (it says why), and the atlas decision at the end of 04.
