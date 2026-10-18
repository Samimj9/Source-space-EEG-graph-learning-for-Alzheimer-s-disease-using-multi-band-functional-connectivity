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

## 3. ROI resolution recommendation

**18 regions (`r18`)** is recommended for 19-channel EEG: with 68 ROIs, 50 of the 68 signals are exact linear
combinations of the others (rank ≤ 18) and the median ROI gets ≈ 85 % of its variance from other regions; r18 has no
forced redundancy, about half the cross-talk and 89 % own-region dominance, for both inverses
(`docs/final/parcellation.md`; provisional sphere-model numbers, confirmed with the BEM by notebook 02). It requires
building r18 stores (your approval; ≈ 1.5 h CPU; notebook 02 §4). Until then everything runs on the validated
68-ROI stores, which remain the reference analysis in either case.

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

1. Category A QC findings, if notebook 01 reports any (`qc/decisions.yaml`).
2. Building the r18 stores and making r18 the primary atlas (notebook 02 §4, then a logged change of
   `release.PRIMARY["atlas"]`).
3. The deletion proposals of `docs/final/inventory.md` (nothing is deleted otherwise).
