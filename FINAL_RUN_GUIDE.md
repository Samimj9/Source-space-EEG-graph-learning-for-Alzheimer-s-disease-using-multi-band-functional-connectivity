# Final run guide (release final-1.0)

Branch `claude/unified-final-framework`. Open each notebook from `notebooks/README.md` (Colab links), *Runtime → Run
all*, one notebook at a time, in this order (`configs/final/run_plan.yaml`, tested). Every notebook is resumable: if
Colab disconnects, run it again. Details: `docs/final/framework.md`.

## 1. Execution order

| # | Notebook | Runtime | What it does | Estimated time |
|---|---|---|---|---|
| 1 | `00_setup_and_stores` | CPU | verifies the 4 validated stores, writes the release lock, runs the test suite once | 15-25 min |
| 2 | `01_participant_qc` | CPU | QC of the 4 stores; **stop here if a category A finding needs your decision** (`docs/final/qc.md`) | 10-20 min |
| 3 | `02_parcellation` | CPU | source resolution per atlas with the stores' BEM, atlas recommendation, feature reliability | 15-30 min |
| 4 | `03_graph_construction` | GPU | graph diagnostics, then the development-CV comparison of 11 constructions, then the recorded selection | ~24 h CPU equiv. |
| 5 | `Exp0_sensor_psd_de` | CPU | sensor PSD / DE, LR and SVM (modes B, A) | < 1 h |
| 6 | `Exp1_source_psd_de` | CPU | source PSD / DE, LR and SVM (LCMV, wMNE x B, A) | ~1 h |
| 7 | `Exp2_conn_minimal` | GPU | 10 variants | ~6 h CPU equiv. |
| 8 | `Exp3_single_band` | GPU | 10 variants | ~6 h |
| 9 | `Exp4_band_mean` | GPU | 4 variants | ~2.5 h |
| 10 | `Exp5_multibranch` | GPU | 8 variants | ~16 h |
| 11 | `Exp6_psd_strength` | GPU | 10 variants | ~24 h |
| 12 | `Exp7_earm_gnn` | GPU | 6 variants (exploratory) | ~36 h |
| 13 | `Exp8_xu_dual_gcn` | GPU | 10 variants | ~10 h |
| 14 | `99_results_report` | CPU | all tables, LCMV vs wMNE, mode B vs A, difficult participants | minutes |

Each experiment notebook runs, in order, LCMV mode B, wMNE mode B, LCMV mode A, wMNE mode A (Exp 0: sensor B, A),
each variant in development CV (5 folds) and once on the holdout. GNNs read the graph construction recorded by
notebook 03 (`--graph selected`) and stop if it is missing.

**Time estimates** are derived from measured fit times, not guessed: 60-epoch fits measured on 2 CPU threads at the
mode A and mode B sizes, consistent with the saved mode A runs on a Colab CPU runtime (Exp 5 21-32 s, Exp 8 11-13 s
per fit). Per fold fit in mode B on Colab CPU: Exp 2-4 ≈ 3 min, Exp 5 ≈ 10, Exp 6 ≈ 12, Exp 7 ≈ 30, Exp 8 ≈ 5 min;
mode A ≈ 1/25 of that. The table gives CPU-equivalent totals for both inverses and both modes (6 fits per variant
and context). A T4 GPU has not been measured for these models and should be faster; the real per-fold seconds appear
in every status table after the first fold. Total: ≈ 100 CPU-equivalent hours of GNN training + ≈ 24 h for the graph
comparison.

## 2. Configurations

* Stores (read only): LCMV `D3_LCMV_XU/graphsLX_v1_811690218bfa_corr1` (B), `graphsLX10_v1_811690218bfa_corr1` (A);
  wMNE `D3_WMNE_XU/graphsWX_v1_c32bc6c67f0e` (B), `graphsWX10_v1_c32bc6c67f0e` (A); 68 DK ROIs, 5 Xu bands.
* Protocol: frozen 52 / 13 split, `xu_dev_cv` 5 folds + `xu_holdout`; participant-level metrics primary; holdout
  descriptive only (inspected before; never used for any selection).
* GNNs: `gnn_fixed60_v1`, exactly 60 epochs, Adam 1e-3, wd 1e-4, batch 32, no early stopping, no search.
* **Graph constructions** (`configs/final/graphs.yaml`; k = round(f (R - 1)), R = 68 → k = 7 / 13 / 20):

  | name | rule |
  |---|---|
  | `dense` | weighted PLV / \|ciPLV\| (reference, default) |
  | `knn_union_f10/f20/f30` | union kNN, k = 7 / 13 / 20 |
  | `knn_mutual_f10/f20/f30` | mutual kNN, k = 7 / 13 / 20, isolated nodes repaired |
  | `proportional_f20` | global strongest edges, density 20 % |
  | `contiguity`, `contiguity_knn_union_f20` | restricted to cortically contiguous regions |
  | `feature_similarity_knn_union_f20` | PSD / DE similarity graph (reported, never selected) |

  Selection (development CV only): Exp 4 `plv__psd` and Exp 5 `plv__separate__mean`, LCMV and wMNE, mode B; a
  construction replaces `dense` only if mean AUC ≥ dense + 0.02 and better in ≥ 3 of 4 cells.

## 3. Three Scientific Decisions Before Training

Full classification (Exp 0-8) does not start until you have reviewed these three decisions. Nothing below used a
classification result, and nothing read the holdout.

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

**Recommended default.** **r18** as the primary atlas for both inverses, with **dk68** kept as the reference /
sensitivity analysis (and the link to all earlier results). Until r18 stores exist, everything runs on dk68.

**Alternatives requiring development-only validation.** None is selected by classification results (by design the
atlas follows source resolution only). `dk34_gyral` and `lobes12` stay available as sensitivity analyses, reported
descriptively on development CV.

**Additional computation.** Yes: r18 stores for LCMV and wMNE, built from the recordings with the unchanged pipeline
(`scripts/build_stores.py build | validate | match --atlas r18`; notebook 02 §4; ≈ 40 min LCMV + 30-45 min wMNE CPU).
New directories; the 68-ROI stores are not touched. `final.py reconstruct-check` (notebook 02 §3b, read-only)
measures the reconstructable fraction on the real stores.

**Needs your approval.** (a) building the r18 stores; (b) making r18 primary (`release.PRIMARY["atlas"]`, logged in
`docs/methods_decisions.md`); otherwise dk68 stays primary and the paper states the leakage limitation.

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
of 4 cells; the decision is written once to `FINAL_v1/decisions/graph_construction.json`. The holdout plays no part
(tested). If r18 is adopted, run this comparison on the r18 stores (k scales with R).

**Additional computation.** Yes, ≈ 24 CPU-equivalent hours (220 fold fits; GPU faster), notebook 03.

**Needs your approval.** Running notebook 03 (it trains models, though on development folds only); the 0.02 margin
and the k fractions as stated, before it runs.

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

**Additional computation.** Notebook 01 only (≈ 10-20 min CPU, no training).

**Needs your approval.** A `keep` / documented decision in `FINAL_v1/qc/decisions.yaml` for every category A finding
notebook 01 reports; any exclusion is a logged protocol amendment (it changes the frozen split).

## 4. Output directories (Drive)

```
<processed>/FINAL_v1/CODE_COMMIT.txt, stores.lock.json
<processed>/FINAL_v1/qc/<inverse>_<mode>_dk68/, qc/decisions.yaml (yours), qc/difficult_<time>.csv
<processed>/FINAL_v1/parcellation/resolution_<time>/, reliability_<inverse>_<mode>_dk68_<time>/
<processed>/FINAL_v1/graphs/<inverse>_<mode>_dk68/diagnostics_<time>/
<processed>/FINAL_v1/decisions/graph_construction.json
<processed>/FINAL_v1/results/<lcmv|wmne|sensor>/mode_<B|A>/dk68/<graph|none>/<experiment>/<variant>/<setting>/
<processed>/FINAL_v1/reports/tables_<time>/
```

Nothing outside `FINAL_v1/` is written; the stores and all earlier results and pins are only read.

## 5. Needs your decision

1. Category A QC findings, if notebook 01 reports any (`qc/decisions.yaml`; §3.3).
2. Building the r18 stores and making r18 the primary atlas (notebook 02 §4, then a logged change of
   `release.PRIMARY["atlas"]`; §3.1).
3. Running the development-only graph-construction comparison of notebook 03 with its stated margin and k (§3.2).
4. The deletion proposals of `docs/final/inventory.md` (nothing is deleted otherwise).
