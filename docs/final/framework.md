# The final framework (release final-1.0)

DESIGN.md v2.0, methodology log entry 58, decisions row 61. Branch `claude/unified-final-framework`; the earlier
branch `claude/ecstatic-davinci-a2e8mh` and the tag `backup/pre-unification-607b5ea` stay the recovery point for every
earlier result.

## 1. What it is

One code path for the AD vs HC classification of OpenNeuro ds004504 (65 participants: 36 AD, 29 HC; FTD excluded by
design) in source space:

```
validated store (inverse x window mode x atlas) --verify + lock--> participant QC gate
   --> [02] r18 stores from the recordings (validated, matched to LCMV and to dk68) + their QC
   --> [04] dk68 vs r18 comparison (development CV) --> the author records the atlas
   --> [03] graph-construction comparison on that atlas (development CV) --> recorded selection
   --> Exp 0-8 (one config each; development CV) --> [99] freeze the final analysis --> holdout once (descriptive)
   --> combined tables / comparisons
```

* **Inverse** (`lcmv` | `wmne`) and **window mode** (`B` all accepted windows, primary; `A` 10 windows per
  participant, sensitivity) are run-time parameters; the experiment definitions, models, training and evaluation are
  identical for every context (tested: `tests/test_source_parity.py`). Exp 0 runs in sensor space once per mode.
* **Atlas** (`dk68`, or `r18` with its own stores; `dk34_gyral`, `lobes12` available) and **graph construction**
  (`dense` default; `configs/final/graphs.yaml`) are run-time parameters too; the experiment notebooks read the
  recorded decisions (`--atlas selected --graph selected`; `adgraph.experiments.decisions`).

| Part | Where |
|---|---|
| store configuration (both inverses, every atlas) | `configs/source_graphs.yaml`, `adgraph.sourcegraphs.config.resolve` |
| store building / validation / matching | `adgraph.sourcegraphs`, `scripts/build_stores.py` |
| store verification and the release lock | `adgraph.stores` |
| release, version, output root | `adgraph.release` |
| participant QC | `adgraph.qc`, [qc.md](qc.md) |
| graph construction and diagnostics | `adgraph.graphs.construction`, `adgraph.graphs.diagnostics`, [graph_construction.md](graph_construction.md) |
| parcellation / source resolution | `adgraph.source.atlas`, `adgraph.source.resolution`, [parcellation.md](parcellation.md) |
| experiment configs | `configs/final/{protocol,classical,gnn}.yaml`, `configs/final/experiments/exp0-8*.yaml` |
| runner, planner, report, selection | `adgraph.experiments.{runner,plan,report,selection}` |
| ROI comparison, recorded decisions, frozen final analysis | `adgraph.experiments.{roi_comparison,decisions}`, `configs/final/{roi_comparison,holdout}.yaml` |
| participant-level evaluation | `adgraph.evaluation.{splits,metrics,reporting}` |
| CLI / notebooks | `scripts/final.py`, `scripts/make_notebooks.py` -> `notebooks/` |
| execution order | `configs/final/run_plan.yaml`, [`FINAL_RUN_GUIDE.md`](../../FINAL_RUN_GUIDE.md) |

## 2. The validated stores (read only)

| inverse / mode | directory (under `<processed>`) | config hash |
|---|---|---|
| LCMV B / A | `D3_LCMV_XU/graphsLX_v1_811690218bfa_corr1`, `graphsLX10_v1_811690218bfa_corr1` | 811690218bfa |
| wMNE B / A | `D3_WMNE_XU/graphsWX_v1_c32bc6c67f0e`, `graphsWX10_v1_c32bc6c67f0e` | c32bc6c67f0e |

`resolve(inverse, "dk68")` returns, dict for dict, the configurations these stores were built with (tested against
`legacy/configs/{lcmv_xu,wmne_xu}/pipeline.yaml` and the configs at the build commits 299aed72 / 5b3223c7), so the
stores are found by their hash and never rebuilt. `stores.verify_store` checks, without writing: resolved directory and
config hash, assembled and complete cohort, window mode, inverse, the five Xu bands, the atlas's ROI names in store
order, `validation.json` passed for the current graph_run_id, wMNE `match.json` passed against the LCMV store of the
same mode, 65 participants (36 AD / 29 HC), the saved 52 / 13 partition equal to `xu_partition` and to the reference
LCMV store. The first verification records each store's graph_run_id in `FINAL_v1/stores.lock.json`; a store whose
content changed afterwards is refused.

## 3. One version system

`adgraph.release`: release `final-1.0`, output root `<processed>/FINAL_v1/`, ONE code pin `FINAL_v1/CODE_COMMIT.txt`
(set by the first notebook session, moved only with `UPDATE_CODE = True`). It replaces the per-suite pins of the earlier
work (D1/D2, D3, LX, WX, ES200, R1, Exp 8); those pin files stay on Drive untouched and are only reported. Every result
manifest records `release.release_manifest()` (release, git commit, dirty flag, source fingerprint, library versions);
every checkpoint unit carries the code identity, so a code change is never silently reused (`IncompatibleCheckpoint`).

## 4. Experiments (configs/final/experiments)

| Exp | Model | Variants |
|---|---|---|
| 0 | sensor PSD / DE, LR and linear SVM | 4 |
| 1 | source PSD / DE, LR and linear SVM | 4 |
| 2 | minimal GCN: one band's connectivity, constant node feature | 10 (5 bands x PLV / \|ciPLV\|) |
| 3 | single-band GCN with that band's PSD (log, rel) | 10 |
| 4 | GCN on the band-averaged adjacency, all-band PSD or DE as columns (intentional band averaging) | 4 |
| 5 | multibranch GNN: shared GCN + one-hot band identity, or one GCN per band; mean / attention fusion | 8 |
| 6 | node-feature / band ablations of the shared band-identity GCN with attention fusion | 10 |
| 7 | EARM-GNN (exploratory; learned band embedding and edge gates) and its two ablations | 6 |
| 8 | Xu-inspired dual-adjacency GCN (band-specific weights; functional + atlas layer) and ablations | 10 |

No architecture was added. The band-permutation-invariant shared-encoder variants of the earlier suites (Exp 5
`shared__*`, Exp 6 originals) are not final experiments; `experiments.config.validate` refuses a shared-encoder
multiband model with symmetric fusion and no band identity unless it is declared `band_invariant_baseline`.
Every final multiband variant is tested to be sensitive to the band order (`tests/test_final_audit.py`).

## 5. Training

* **GNNs (Exp 2-8):** protocol `gnn_fixed60_v1`: exactly 60 epochs, no early stopping, no hyperparameter search,
  no inner CV, the model after the last epoch is evaluated; Adam lr 1e-3, weight decay 1e-4, batch 32, participant-
  and class-balanced BCE (each participant one vote, both classes equal weight), seed = 1000 x repeat + fold. Node
  features are z-scored per (band graph, ROI, feature) with training statistics; the graph construction is applied to
  every view (only `quantile` is fitted, on training windows). The training history (loss and gradient norm per
  epoch, relative loss change over the last 10 epochs, training-set AUC of the final model, graph statistics) is
  saved per fold; with `save_model_state` the parameters, EARM edge gates and scaler statistics too. The training loop
  reproduces the 607b5eac trainer bit for bit (checked on four architectures).
* **Classical (Exp 0, 1):** standardization fitted on the training windows, C in {0.001, 0.01, 0.1, 1, 10} chosen by
  participant-level inner 5-fold CV (ROC-AUC) inside each outer training set, refit on the whole outer training set;
  LR scored by P(AD), the SVM by its decision margin (threshold 0).

## 6. Evaluation (participant level; adgraph.evaluation.reporting)

* Frozen split `xu_partition` (seed 20250702, test fraction 0.2): 52 development / 13 holdout participants
  (holdout: D3-3, 12, 15, 16, 22, 30, 33, 38, 42, 45, 51, 57, 64; 7 AD / 6 HC). `xu_dev_cv`: stratified 5-fold
  participant CV inside the development set (seed 0; identical to the folds of the earlier suites, pinned in
  `tests/test_splits.py`); `xu_holdout`: fit on the 52, evaluate once on the 13.
* Participant score = mean of its window scores. Reported per (variant, setting): pooled out-of-fold participant
  ROC-AUC (primary), mean / SD / min / max of the five within-fold AUCs, balanced accuracy, sensitivity (AD),
  specificity (HC), F1, MCC, confusion matrix, DeLong 95 % CI, participant-bootstrap 95 % CIs, Wilson CIs of
  sensitivity and specificity, window-level AUC (secondary), score distribution, participant predictions, fold
  training diagnostics, fold QC.
* **Holdout policy:** the holdout was inspected in the earlier LX / WX experiments, so it is not an untouched test set.
  It is evaluated once per unit of the frozen final analysis (`configs/final/holdout.yaml`: every pre-specified
  variant of Exp 0-6 and 8, mode B, both inverses, on the recorded atlas and graph), after the atlas and graph
  decisions and after all of these units completed development CV (`decisions.freeze`), and reported descriptively.
  The code refuses the holdout of a real store for any other unit (`decisions.require_frozen`, in
  `plan.run_experiment`). It is never used to select an architecture, a graph construction, k, the atlas, QC
  thresholds or anything else.
* Mode B is the primary analysis, mode A the reduced-window sensitivity analysis; LCMV vs wMNE and mode B vs A are
  compared on the same participants with the paired DeLong test (descriptive, no multiplicity correction implied).

## 7. Pre-specified selection rules (configs/final/selection.yaml)

* **Graph construction:** chosen by development CV only (`experiments.selection`): the reference models Exp 4
  `plv__psd` and Exp 5 `plv__separate__mean`, both inverses, mode B, every construction; a construction replaces
  `dense` only if its mean development AUC is >= dense + 0.02 AND it beats dense in >= 75 % of the 4 cells; the
  feature-similarity graph is reported, never selected; an incomplete comparison keeps `dense`. Recorded once in
  `FINAL_v1/decisions/graph_construction.json` (with the atlas it was made on), read by Exp 2-8 (`--graph
  selected`). It runs on the recorded atlas, after notebook 04. An exploratory pre-specified rule, not a superiority
  test.
* **Atlas:** never chosen by classification performance alone. Recommendation by source resolution: the finest
  atlas whose ROI estimates (both inverses) have no forced linear redundancy, median cross-talk <= 0.60 and own-region
  dominance >= 0.80 ([parcellation.md](parcellation.md)); r18 stores approved and built in notebook 02; the
  development-only dk68 vs r18 comparison of notebook 04 reports resolution / reliability and classification
  evidence separately; the AUTHOR records the atlas (`FINAL_v1/decisions/atlas.json`). dk68 stays the reference.

## 8. Leakage rules (enforced in code, tested)

Splits are built from participants, never windows (`fold_indices` re-checks); every fitted transform lives in a model
whose `fit` accepts only a training view (`LeakageError` otherwise): scalers, feature selection, PCA, SVM calibration,
node-feature scaling, graph-construction thresholds; the classical inner CV runs inside the outer training
participants; QC outlier thresholds are fitted on reference / training participants only and never change training.

## 9. Outputs (never overwritten)

```
<processed>/FINAL_v1/
  CODE_COMMIT.txt  stores.lock.json
  qc/<inverse>_<mode>_<atlas>/        participant QC (qc.md);  qc/decisions.yaml (author)
  graphs/<inverse>_<mode>_<atlas>/diagnostics_<time>/
  parcellation/atlas_<atlas>_<time>/, resolution_<time>/, store_resolution_<inverse>_<atlas>_<time>/,
               reliability_<inverse>_<mode>_<atlas>_<time>/
  roi_comparison/compare_<time>/
  decisions/atlas.json (author), graph_construction.json, final_analysis.json
  results/<inverse|sensor>/mode_<B|A>/<atlas>/<graph|none>/<experiment>/<variant>/<xu_dev_cv|xu_holdout>/
      predictions_windows.csv predictions_participants.csv metrics.json fold_metrics.csv folds.csv training.csv
      fold_qc.csv manifest.json units/ status.csv _identity.json final.done.json
  results/.../<experiment>/_run_status.csv , _reports/report_<time>/
  reports/tables_<time>/  all_results.csv  lcmv_vs_wmne.csv  mode_B_vs_A.csv
```

Coarser-atlas stores are not under `FINAL_v1/`: they are written next to the 68-ROI stores,
`<processed>/D3_LCMV_XU/graphsLX{,10}-r18_v1_<hash>/` and `<processed>/D3_WMNE_XU/graphsWX{,10}-r18_v1_<hash>/`
(`scripts/build_stores.py`), each with its own manifest, validation, match and atlas-match records.
