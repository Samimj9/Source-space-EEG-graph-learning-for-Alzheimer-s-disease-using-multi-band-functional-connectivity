# manuscript_figures — reproducible figures for the D3 source-space EEG-GNN manuscript

Reads **saved outputs only**. No training, inference, source reconstruction or new evaluation. Exp0/Exp1 (classical baselines) are excluded from every figure.

| File | Purpose |
|---|---|
| `build_all_figures.py` | all plotting functions + CLI (`python build_all_figures.py --processed .../AD-LEB/processed`) |
| `D3_Manuscript_Figure_Generation.ipynb` | Colab notebook that runs the same functions (`make_notebook.py` regenerates it) |
| `captions.json` | journal-style captions keyed by figure id |
| `requirements.txt` | numpy, pandas, scipy, scikit-learn, matplotlib (nbformat only for the notebook generator) |
| `generated/` | PDF, SVG, PNG (600 dpi) of every figure (new run overwrites; originals on Drive are never touched) |
| `figure_manifest.csv`, `source_inventory.csv`, `run_info.json` | written by each run: outputs, inputs with SHA-256, skipped figures |

## Inputs (read-only)
`development_results_all.csv`, `comparisons.csv`, `selection_check.json`, `resolution_by_atlas.csv` (found under `FINAL_v1` or passed with `--resolution-csv`), per-unit `predictions_participants.csv` (path = `<processed>/<rel_dir>/predictions_participants.csv`, `rel_dir` from the results table), the single-model holdout predictions (`--holdout-pred`), and the committed atlas tables `src/adgraph/source/atlas_tables.json`.

## Safeguards (tested)
* every unit is located by its full identity (experiment, variant, inverse, window mode, atlas, graph); exactly one saved row must match;
* prediction files must hold one score per participant (52; holdout 13), and the recomputed AUC and confusion matrix must equal the saved table, otherwise the run stops (a corrupted file aborts with `recomputed AUC ... != saved ...`);
* output paths inside `FINAL_v1` / amendment namespaces are refused; no result file is ever opened for writing;
* a missing input raises `MissingInput` and that figure is reported as skipped — nothing is simulated or hand-entered.

## Figures
| id | content | needs |
|---|---|---|
| fig01 | pipeline diagram | none (drawn) |
| fig02 | Exp8a architecture (shapes verified against `xu_dual.py`; 5,771 parameters counted by instantiation) | none (drawn) |
| fig03 | atlas geometry, resolution metrics, AUC/BA by atlas | atlas tables, resolution CSV, results CSV |
| fig04 | family distribution + Exp3 band heatmap (+ BA heatmap S5) | results CSV |
| fig05 | participant-level ROC + confusion matrices + paired DeLong | predictions of 3 units, results CSV |
| fig06 | Exp8a ablations + matched robustness (GNN-only subsets) | comparisons CSV |
| S1–S4 | selection optimism, descriptive holdout, graph construction, Exp6 | selection JSON, holdout predictions, results CSV |

`render_cortical_surface()` (optional) renders an atlas on the fsaverage inflated surface with MNE; it needs MNE and the fsaverage files cached by notebook 02 and was **not** executed in the offline test build.

## Status of this build
Offline test run with local copies of the Drive files (CSV tables, three unit predictions, the holdout predictions, a copy of `selection_check.json` and `resolution_by_atlas.csv`): all 10 figure groups built. The Colab path (Drive discovery of `resolution_by_atlas.csv`, `predictions_participants.csv` via `rel_dir`) is implemented but has not been executed against the live Drive tree.
