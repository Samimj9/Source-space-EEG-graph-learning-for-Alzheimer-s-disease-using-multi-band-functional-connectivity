# A3 single-model holdout (selected after development CV)

Methodology log entry 62; `configs/final/amendments/a3_single_model.yaml`; code `src/adgraph/a3/single.py`,
`scripts/a3_single_holdout.py`; notebook `notebooks_a3/A3_06_single_model_holdout.ipynb`.

**What it is.** A separate, lightweight, explicitly *development-CV-selected* holdout evaluation of ONE unit of the frozen
A3 inventory, requested by the author to reduce computation: `exp8a_xu_dual_gcn_surface` /
`plv__psd_de_strength__abl_anatomical_both`, wMNE, mode B, lobes12, dense (development AUC 0.873 as reported by the author;
checked against the frozen record, rounded to 3 decimals, and refused if different), A2 pin
`7fa470c668b8cce64ca152372422638be3ee8203`.

**What it is not.** It is not completion of the full A3 holdout protocol. The other 115 frozen configurations are not
evaluated by this path. The 13-participant D3 holdout was inspected in earlier work: the result is descriptive only, is not
external validation, and informs no selection. The model was chosen after development CV, and its development AUC, being the
maximum over many configurations, is optimistic. The manifest records the model's rank among the frozen GNN units by
development AUC (computed from the frozen records) and says "highest" only if it is.

**Nothing existing changes.** The A3 master freeze, both frozen records, the 116-unit inventory, every development result
and both pins are read and never written. New files only: `A3_final/single_model/selection_<id>.json` (written once),
`runs/run_<id>_<time>.json`, `report_<time>/`, and the holdout result folder of this one unit in the A2 namespace (by the
pinned code).

## Steps (what `A3_06` runs)

1. **Environment.** The interpreter is restored to the environment recorded by *this unit's* development result (one file, no scan
   of the ~320 results), then compared with it (`a3/environment.py`: identical or documented-compatible passes, otherwise refused).
2. **`a3.py single-verify` (read only).** Master freeze and both records verified (checksums, unit sets from configuration, pins,
   QC decisions, store lock); the exact unit must be in the frozen A2 record and its AUC must match; the unit's development
   result is verified (completion marker, payload checksums, metrics / predictions hashes, produced at the A2 pin). Not re-read:
   the other units' development results (the full `a3.py verify` still checks them).
3. **`a3.py single-holdout`.** After the checks it writes the selection manifest (once; states the selection after development CV,
   that the other frozen configurations are not evaluated, and the descriptive-only status), creates a clean worktree at the A2
   pin, and runs `scripts/a3_single_holdout.py` there. That driver applies the same gates as the pinned `amendment.holdout`
   (amendment initialized; the namespace's frozen record exists, names this commit and lists the unit; the variant selector
   matches exactly one unit) and then calls the pinned `open_context` and `run_experiment(["xu_holdout"])` for that unit only:
   unchanged model, fixed-60 protocol, fit on the 52 development participants, one evaluation on the 13 holdout participants,
   threshold 0.5, participant-level checkpointing. A completed unit is not launched again; an interrupted one resumes.
4. **`a3.py single-report`.** Reads only the selection manifest, the frozen records and this unit's development and holdout folders
   (does not need the other units): participant predictions, AUC (DeLong and bootstrap CI), balanced accuracy (bootstrap CI),
   sensitivity / specificity (Wilson CI), F1, MCC, confusion matrix, ROC, and the frozen development AUC for a descriptive
   comparison, with the manifest's statement. Each report goes to a new folder with output checksums.

## Reading it

With 13 participants (about 7-8 AD and 5-6 HC) every interval is wide; the development-versus-holdout difference is not a test,
and a high or low holdout value must not lead to choosing or discarding a model.
