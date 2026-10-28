# Cross-dataset D3 <-> D2: run guide

Design: `docs/cross_dataset/design.md`. Notebooks: `notebooks/cross_dataset/` on branch
`claude/cross-dataset-d3-d2-19ch` (links in `notebooks/cross_dataset/README.md`). One notebook at a time, each with
*Runtime -> Run all*. **Exp 8a may keep running**: these notebooks use their own branch, their own code pin
(`<outputs>/CROSS_D3_D2_19CH_v1/CODE_COMMIT.txt`) and their own output folder; they only read `FINAL_v1/`, the A2
results and the D3 stores.

## 0. Before anything (author)

1. Read the design document, in particular section 12 (decisions D1-D10, amendment X1 in 12.8) and section 4.3
   (what is not harmonized).
2. Confirm the D2 recording condition (eyes closed / open; decision D6) and, if known, the D2 amplifier and online
   reference.

## 1. `00_CrossDataset_Audit_and_Build.ipynb` (CPU)

**Authorized now (2026-10-26): Part A only.** Leave `AUTHORIZED_BY = ''` and `FREEZE_CAMPAIGN = False`; the notebook
then stops after Part A. Send the printed audit tables (mapping audit, EDF headers and annotations, D3 store table,
decision audit, demographics / overlap, filter response, D2 inspection, D3 line noise).

| Part | What runs | Needs |
|---|---|---|
| Setup, environment | own pin; the Exp 7 numerical environment restored if needed | nothing |
| A1 `audit` | mapping table, layout hash, geometry (D3 `.set` positions vs `.loc`), D2 EDF technical headers, D3 stores verified (read only), inherited decisions | nothing (reads only) |
| A1 (cont.) | the D2 cohort (`d2_cohort.json`: inventory 23, analysis 22, D2-19 excluded by amendment X1), demographics by diagnosis (analysis cohort; the 23-recording inventory separately) and the age-overlap feasibility (descriptive), the verified D2 filter response, the EDF annotation texts (recording condition, D6) | nothing |
| A2 `inspect-d2-units`, `inspect-d2`, `inspect-d3-line` | every EDF channel's unit and band-passed amplitude of all 23 recordings (mixed `V` / `uV` recordings listed channel by channel; D2-19's recorded evidence re-checked, amendment X1); per D2 recording processed as the store will be, WITHOUT and WITH the D2 50 Hz notch: 50 Hz power, 100 Hz harmonic, residual narrowband peaks, covariance change, accepted windows, band-power change per analysis band (`diag_*` = the first audit's diagnostic notch); the D3 line-noise audit (no source model, no graph, labels removed) | nothing |
| B `build-d2`, `validate-d2`, `compat` | the D2 stores of the 22-participant analysis cohort (smoke `--limit 2` first, then LCMV and wMNE, modes B and A), validation, compatibility report, shift diagnostics, D2 QC on the label-masked store, harmonization manifest | `AUTHORIZED_BY = '<name>'` after your written authorization |
| C `freeze` | the frozen campaign (before any training) | `FREEZE_CAMPAIGN = True` |

Stop points: `AUDIT FAILED` (send the printed table), a failed store validation, `COMPATIBILITY FAILED`. Review the
compatibility report and the shift diagnostics before freezing.

## 2. `01_CrossDataset_D3_to_D2.ipynb` (GPU, T4)

Set `AUTHORIZED_BY` once (records the `train` authorization), then *Run all*. Six contexts: LCMV / wMNE / sensor
(Exp 0) x modes B / A. Every unit fits on the 52 D3 development participants and scores the blinded D2 windows.
Disconnected? Open again, *Run all*: completed units are skipped. No metric is shown here.

## 3. `02_CrossDataset_D2_to_D3.ipynb` (GPU, T4)

Same, D2 -> D3, plus the source-only D2 cross-validation (5 folds x 1 repeat, the primary D3 protocol) of every
unit. The optional last cell of notebooks 01 / 02 runs the stability analysis (seeds 1-4, D2 CV 5 x 5) only with a
separate `STABILITY_AUTHORIZED_BY`.

## 4. `03_CrossDataset_Comparison_Report.ipynb` (CPU)

`evaluate` (the only step that reads target labels; frozen units only), then `report`: tables 1-4, forest plots, age
figures, README with the disclosures, under `CROSS_D3_D2_19CH_v1/reports/primary_<time>/`.

## Command-line equivalents (inside the notebooks' environment)

```
python scripts/cross_dataset.py audit
python scripts/cross_dataset.py inspect-d2-units
python scripts/cross_dataset.py inspect-d2
python scripts/cross_dataset.py inspect-d3-line
python scripts/cross_dataset.py authorize --authorization-stage build_d2 --by "<name>"
python scripts/cross_dataset.py build-d2 --inverse lcmv --limit 2 && python scripts/cross_dataset.py validate-d2 --inverse lcmv --limit 2
python scripts/cross_dataset.py build-d2 --inverse lcmv && python scripts/cross_dataset.py validate-d2 --inverse lcmv
python scripts/cross_dataset.py build-d2 --inverse wmne && python scripts/cross_dataset.py validate-d2 --inverse wmne
python scripts/cross_dataset.py compat --inverse lcmv --mode B          # and the other three contexts
python scripts/cross_dataset.py freeze
python scripts/cross_dataset.py authorize --authorization-stage train --by "<name>"
python scripts/cross_dataset.py run --direction D3_to_D2 --inverses lcmv --modes B
python scripts/cross_dataset.py authorize --authorization-stage stability --by "<name>"   # only if authorized
python scripts/cross_dataset.py run --stage stability --direction D3_to_D2 --inverses lcmv --modes B
python scripts/cross_dataset.py status
python scripts/cross_dataset.py evaluate && python scripts/cross_dataset.py report
python scripts/cross_dataset.py smoke --quick                           # synthetic, temp dir, any time
```

## Output layout on Drive

```
CROSS_D3_D2_19CH_v1/
  CODE_COMMIT.txt  authorizations.json
  audit/        channel_mapping_audit.csv  d2_edf_headers.csv  reference_geometry.json  audit.json  d2_inspection.csv
                demographics.csv  demographics_d2_inventory.csv  d2_cohort.csv  d2_cohort.json
                age_overlap_feasibility.csv  d2_filter_response.json  d3_line_noise.csv
                d2_units.csv  d2_notch_response.json
                <inverse>_mode_<m>/compatibility_report.csv  shift_diagnostics.csv  harmonization_manifest.json
                <inverse>_mode_<m>/d2_qc_blind/
  stores/       (each store: cohort.json = amendment X1 record) graphsXD2L_v1_<hash>/  graphsXD2L10_v1_<hash>/  graphsXD2W_v1_<hash>/  graphsXD2W10_v1_<hash>/  cache/
  plan/         frozen.json
  results/      <direction>/<inverse>/mode_<m>/<atlas>/<graph>/<experiment>/<variant>/transfer/seed_<s>/
                (target_windows_blind.csv  target_participants_blind.csv  training.json  model.npz  transfer.done.json)
                D2_to_D3/.../source_cv/ (units/, predictions_participants.csv, metrics.json)
  evaluation/   primary/ (metrics_long.csv  predictions_participants.csv  paired_comparisons.csv  age_*.csv
                demographics.csv  summary.json)
  reports/      primary_<time>/ (tables, figures, README.md)
```
