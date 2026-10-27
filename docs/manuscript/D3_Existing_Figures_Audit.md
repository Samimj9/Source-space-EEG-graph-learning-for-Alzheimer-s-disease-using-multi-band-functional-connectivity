# D3 existing figures — reuse audit

Scope: every figure was **downloaded from Drive and viewed as a rendered image** (not judged from its filename). Drive folder: `1X-r_c1RxJIuXy53-t2TjAjR15ho_lwq9` (consolidated development analysis, created 2026-10-27). The parent path of that folder under `AD-LEB/processed/FINAL_v1/` was not resolvable through the connector; file IDs are given instead (open as `https://drive.google.com/file/d/<ID>/view`). Manuscript scope excludes Exp0/Exp1, so any figure that plots them cannot be used unchanged.

## 1. Consolidated figures (7 files, each as PDF + PNG)

| Existing file (ID of PNG) | Experimental source | What it shows / what I found | Visual quality | Main / supplement | Reuse / redraw | Replacement |
|---|---|---|---|---|---|---|
| `fig1_primary_forest` (1q55jaEobsQIZupk8EfztJJPZsafzLZHM) | development_results_all.csv, mode B, recorded atlas/graph; DeLong 95% CI | Forest of **all** variants incl. **Exp0 (sensor) and Exp1 (classical)**, 67 rows, tiny labels, legend overlaps the last rows, title says "the best is optimistic" | readable only at full size; 2256 x 2579 px; too tall for a journal column | exclude from the paper | **Do not use** (scope: contains Exp0/Exp1; unreadable at column width). Keep as an internal overview. | `fig04` (GNN families) and Table I |
| `fig2_family_summary` (1ClANvRxvjEbje4mIRkmNGHELi1s8Lpi4) | same | Strip plot of AUC per family; **includes Exp0 and Exp1** | clean, colour-blind-safe palette | would be the basis of Fig. 4a | **Redraw** without Exp0/Exp1, label dots as configurations (not CIs) | `fig04a` |
| `fig3_lcmv_vs_wmne` (1RT4YhjfxgVUL72RGvAIq8goVmK1TGRRL) | comparisons.csv family LCMV-vs-wMNE (124 pairs) | Matched LCMV vs wMNE scatter. **The 124 pairs include 8 Exp1 pairs** (visible top-right at AUC about 0.9); caption text says "pink = supported" but no pair is pink | good; square; clear diagonal | supplement at most | **Redraw GNN-only (116 pairs)** | `fig06b` |
| `fig4_modeB_vs_modeA` (1RZiVMt-CAa0mMCEkYjskdhayqTEJBe7O) | comparisons.csv family modeB-vs-modeA (128 pairs) | one pink pair (Exp7 LCMV |ciPLV| mean fusion: B 0.861 vs A 0.609). **128 pairs include 8 Exp1 + 4 Exp0 pairs** | good | supplement | **Redraw GNN-only (116 pairs)** | `fig06b` |
| `fig5_ablations` (1qNxIlz_Z9-BrtThTsC9JDgUKVgEgD5kJ) | comparisons.csv Exp8a-ablations (12) + Exp6-ablations (6), per inverse | forest of AUC differences with paired bootstrap CIs, axis label states "uncorrected" | good content, but very long y-labels and the Holm p-values are not shown | supplement (Exp6 half) / basis of Fig. 6a (Exp8a half) | **Reuse the data, redraw** to show Holm p and split Exp8a vs Exp6 | `fig06a`, `figS_exp6` |
| `fig6_selection_bias` (1SGVCQzhbbMO9cdWK5odNMRnwZHISaGRs) | selection_check.json | observed best vs "optimism-corrected" vs null 95th percentile; **includes the classical (Exp 1) sets**. The plotted blue point is **observed minus optimism**, not the out-of-bag winner AUC quoted in interpretation.md (e.g. GNN-all LCMV: 0.790 plotted vs 0.808 OOB) | clear | supplement | **Redraw** GNN-only, show both estimators and say what was resampled | `figS_selection` |
| `fig7_plv_vs_ciplv` (1UPJNNIou1R2oHaSSwf7jEo197f63DU9V) | comparisons.csv PLV-vs-ciPLV (46 pairs, GNN only) | matched scatter, no pair supported; title mentions pink points but none exist | good | supplement | reuse unchanged is acceptable; included in the new `fig06b` for compactness | `fig06b` |

## 2. Per-unit diagnostics inside the experiment folders (sampled)

The recursive search found, in the `xu_dev_cv` / variant folders of the release and A2 trees, the files `roc_xu_dev_cv.png`, `confusion_xu_dev_cv.png`, `training_loss.png` (about 20+ copies, one set per experiment/context; IDs in `source_inventory` when the notebook is run), `roc_holdout.png` (single-model holdout), and legacy `D2_roc_xu_*` figures (D2 is out of scope).
* `roc_xu_dev_cv.png` (example ID 1RB0Mro3USL-PHm156ZVjxlC6apUWhQAI): overlay of several variants of one experiment, titled "(diagnostic)", legend font tiny; not suitable for the paper, fine as supplementary diagnostics. It does not show the primary model.
* `confusion_xu_dev_cv.png` (example 1HNRqaZZaUUNRADyOH_0zvY-UgHIYmWs-): multi-panel confusion matrices per variant; low resolution (about 750 px wide).
* `training_loss.png` (example 19hYSmdOq7y8-GaOb-lQsIC9yLUh7G-u-): mean training loss of the five folds over the fixed 60 epochs; losses reach about 0.03–0.08 with no validation curve. Useful for a supplementary convergence remark (no early stopping); it cannot show over-fitting because no validation loss is saved.
* No saved QC, atlas, connectivity-matrix or adjacency figures were found by title search (`qc`, `atlas`, `resolution`, `adjacency`); this does not prove none exist. The recursive listing of the whole `FINAL_v1` tree was not completed (Drive connector pagination), so the audit of per-experiment folders is a sample, not exhaustive.

## 3. Figures that do not yet exist and are generated by the package
`fig01` pipeline, `fig02` Exp8a architecture, `fig03` atlas resolution (centroid maps + metrics + AUC by atlas), `fig04` family/band heatmap, `fig05` ROC + confusion (from saved participant predictions), `fig06` ablations/robustness; supplementary `figS_selection`, `figS_holdout`, `figS_graph`, `figS_exp6`, `figS_bands`. All are reproduced by `manuscript_figures/build_all_figures.py`.

## 4. Suggested captions for the reused material
* fig5 (supplement): "AUC difference between the full model and each variant (paired participant-bootstrap 95% interval, uncorrected); no comparison is supported after Holm correction."
* fig7 (supplement): "Matched PLV versus |ciPLV| AUC for 46 GNN configurations (development CV, n = 52); no pair differs after Holm correction."
