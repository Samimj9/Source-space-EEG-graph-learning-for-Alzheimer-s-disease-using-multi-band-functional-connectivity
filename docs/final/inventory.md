# Inventory, archive and proposed deletion list

Nothing listed here has been deleted. Old results stay on Drive and old code stays in Git; this file proposes what
could be removed **after the author's explicit approval**, and what must never be. The final framework writes only
under `<processed>/FINAL_v1/` and never into any folder below.

## 1. Google Drive (`<processed>` = `My Drive/AD-LEB-processed` or `AD-LEB/processed`; read-only listing 2026-10-18)

### Keep (inputs of the final framework, or provenance of reported numbers)

| Path | What | Why |
|---|---|---|
| `D3_LCMV_XU/graphsLX_v1_811690218bfa_corr1`, `graphsLX10_v1_811690218bfa_corr1` | validated LCMV stores, modes B / A | **final inputs** (locked by `stores.lock.json`) |
| `D3_WMNE_XU/graphsWX_v1_c32bc6c67f0e`, `graphsWX10_v1_c32bc6c67f0e` | validated wMNE stores, matched to LCMV | **final inputs** |
| `D3_LCMV_XU/graphsLX_v1_811690218bfa`, `graphsLX10_v1_811690218bfa` | LCMV stores before the corr1 correction | provenance of corr1 (the correction summary and the participant LCMV operators / ROI time series live here) |
| `graphsD3_v4_da723b33554c` | historical wMNE D3 graphs (20 s windows, 4 bands) | provenance of the earlier D3 results |
| `graphs_v4_33faa4b89a5d` | D1 / D2 graph run (frozen v4 dataset of DESIGN.md v1.4) | provenance of the D1 / D2 results |
| `OpenNeuro/ds004504` (My Drive) | provider derivatives | raw input; needed to build coarser-atlas stores |
| `AD-LEB/EEGs`, `AD-LEB/Info participants` | D1 / D2 recordings and sheets | raw input (shared folder, owner: collaborator) |
| `cache/` | fsaverage, forward models, operators, input hashes, test markers | speeds up rebuilds; safe to keep |
| `CODE_COMMIT.txt`, `D3_CODE_COMMIT.txt`, `D3_LCMV_XU/CODE_COMMIT.txt`, `D3_WMNE_XU/CODE_COMMIT.txt`, `*/ES200_CODE_COMMIT.txt`, `*/R1_CODE_COMMIT.txt` | historical code pins | identify the commits of the earlier results; never moved by the final framework |

### Archive (earlier results; not used by the final framework, kept as the record of earlier analyses)

| Path | What |
|---|---|
| `experiments/` | D1 / D2 experiments (nested-CV Exp 2 record, fixed-hyperparameter Tier 1 / 2, Exp 7) |
| `experiments_D3/` | earlier D3 (wMNE `graphsD3_v4`) experiments |
| `D3_LCMV_XU/experiments_A_xu_tenfold/`, `experiments_B_all_windows/` | LX suite results (fixed-60, ES200, R1 folders) |
| `D3_WMNE_XU/experiments_A_xu_tenfold/`, `experiments_B_all_windows/`, `experiments_H_historical_graphs/` | WX suite results |
| `D3_LCMV_XU/reports/`, `D3_WMNE_XU/reports/`, `diagnostics/` | earlier reports and diagnostics |
| `header_report_*.csv`, `d3_header_report_*.csv` | input header checks |

Recommendation: keep all of them until the paper is accepted; if space is needed, move (not delete) them into one
`<processed>/_archive_pre_final/` folder; results of the final framework never read them.

### Proposed for deletion (only on explicit approval)

Smoke-test stores (`--limit 1`, one participant; never inputs of any experiment, refused by every experiment runner):

```
<processed>/graphs_v4_33faa4b89a5d_limit1
<processed>/graphsD3_v4_b72f61b88060_limit1
<processed>/graphsD3_v4_da723b33554c_limit1
<processed>/D3_LCMV_XU/graphsLX_v1_811690218bfa_limit1
<processed>/D3_LCMV_XU/graphsLX10_v1_811690218bfa_limit1
<processed>/D3_WMNE_XU/graphsWX_v1_c32bc6c67f0e_limit1
<processed>/D3_WMNE_XU/graphsWX10_v1_c32bc6c67f0e_limit1
```

and any `_superseded/` folders created by deliberate `--force` reruns (inspect before deleting). Nothing else.

## 2. Repository (branch `claude/unified-final-framework`)

The final framework: `src/adgraph` (without `legacy/` it is the framework of docs/final/framework.md), `configs/final/`,
`configs/source_graphs.yaml`, `scripts/{final,build_stores,colab_setup,make_notebooks,make_atlas_tables,d3_data,
inspect_band_filters,compare_band_filter_designs}.py`, `notebooks/`, `tests/`, `docs/final/`, `FINAL_RUN_GUIDE.md`.

Kept and tested, not part of the final pipeline:

* `src/adgraph/legacy/` (importable): the D1 / D2 / `graphsD3_v4` graph builder `pipeline.py` (with differential
  entropy, methodology log entry 58), its amplitude diagnosis and synthetic EDF helpers; `scripts/legacy/` (its
  command-line tools); `configs/pipeline.yaml`, `configs/pipeline_d3.yaml` (its configurations). Tests:
  `test_end_to_end`, `test_boundary_processing`, `test_memory`, `test_amplitude_qc`, `test_band_filters`,
  `test_d3*`, `test_checkpoint`, `test_reliability`, `test_incomplete_graph_run`, `test_legacy_de`, ...

Archived with `git mv` (history preserved; not importable, not collected by pytest):

| Folder | Content |
|---|---|
| `legacy/src/adgraph/` | D1 / D2 visualization package, EdgeWeightedGAT, WeightedGraphSAGE, age covariates, cross-dataset gate, dataset-shift diagnostic, experiment comparison / cost estimation |
| `legacy/configs/` | `experiments/` (D1 / D2, D3, LX, WX, ES200, R1 experiment configs), `figures/`, `run_plan.yaml`, `lcmv_xu/pipeline.yaml` and `wmne_xu/pipeline.yaml` (the store configurations, reproduced exactly by `configs/source_graphs.yaml`; tested) |
| `legacy/scripts/` | per-suite runners and pins: `experiment.py`, `run_experiment.py`, `run_plan.py`, `code_pin.py`, `lx_build.py`, `wx_build.py`, `lx_report.py`, `wx_compare.py`, `es_report.py`, `exp7_earm.py`, `lx_exp8_inspect.py`, `lx_run_all.py`, `make_figures.py`, `source_scale_check.py`, the old `make_notebooks.py` |
| `legacy/notebooks/` | every earlier notebook (D1 / D2 Exp 0-7, additional, advanced, D3, LX, WX, ES200, R1 corrected) |
| `legacy/tests/` | tests of removed functionality, verbatim (§3) |

**Proposed for deletion (only on approval, e.g. after publication; Git history keeps them anyway):** everything under
`legacy/` at the repository root (exact list: `docs/final/legacy_files.txt`, 1 path per line). Never `src/adgraph/legacy/`
while the earlier D1 / D2 numbers may still be cited (rule 7: regenerable from a committed config and commit).

## 3. Retired tests and what replaces them

| Retired (legacy/tests/) | Tested | Final-framework replacement |
|---|---|---|
| `test_runner.py`, `test_configs.py`, `test_fixed_hparams.py` | D1 / D2 settings, configs, nested GNN CV | `test_final_framework.py` (configs, protocol, end-to-end runs), `test_splits.py` |
| `test_early_stopping.py`, `test_lx_schedule.py` | ES200 protocol, LX scheduling | refused by `validate` (fixed 60 epochs), `test_final_framework.py` |
| `test_colab_plan.py`, `test_notebooks.py` | multi-pin run plan, old notebooks | `test_final_framework.py::test_run_plan_covers_the_final_architecture`, `::test_notebooks_are_up_to_date_and_thin`, `::test_one_release_pin_and_historical_pins_untouched` |
| `test_visualization.py`, figure tests of `test_audit_fixes.py`, `test_checkpoint.py`, `test_progress.py` | D1 / D2 paper figures | per-experiment diagnostics of `experiments.report` |
| experiment parts of `test_checkpoint.py`, `test_reliability.py`, `test_incomplete_graph_run.py` | resume, incompatibility, storage guard on the old runner | `test_final_checkpoint.py` |
| `test_dataset.py` age / probe / covariates, `test_leakage.py` age, `test_earm.py` comparison scripts | D1 / D2 secondary analyses | removed with the analyses; layouts in `test_dataset.py::test_final_node_feature_layouts_and_band_alignment` |
| `test_wmne_xu.py` per-suite parts, `test_xu_audit.py` | LX / WX families, R1 variants | `test_source_parity.py`, `test_final_audit.py` |
