# Amendment A3 — exact Colab instructions

All notebooks are in `notebooks_a3/` on branch `claude/a3-final-analysis` (links: `notebooks_a3/README.md`). Each: open,
*Runtime → Run all*, allow Drive. Resumable. **Do not run A3_03 / A3_04 until you have reviewed the freeze and approved A3.**
The release notebooks (`notebooks/`, including `99_results_report`) are unchanged and should not be used for the final
analysis: 99 cannot complete its freeze because exp8 has no result on lobes12.

| Step | Notebook | Runtime | What happens | Writes |
|---|---|---|---|---|
| 1 | `A3_01_development_report` | CPU | audit (stops if a pin, commit, count or environment check fails), consolidated Exp 0-8a report, comparisons, selection checks, figures, interpretation, D3 → D2 manifest | `A3_final/{audit,reports,d2_manifest}/` only |
| 2 | `A3_02_freeze` | CPU | write-once probe (prints whether this Drive folder accepts hard links; the freeze works either way), audit, **dry run** (checks + 116-unit inventory with development AUCs). `WRITE_FREEZE = False` writes nothing | nothing (default) |
| — | *you review* | | `docs/final/a3_review_checklist.md`, the inventory printed by step 2, the report of step 1 | |
| — | *you approve* | | edit `configs/final/amendments.yaml: A3` → `approved_by: <you>`, `status: approved`; commit and push to `claude/a3-final-analysis` | |
| 3 | `A3_02_freeze` again | CPU | set `WRITE_FREEZE = True`, Run all: writes the release record, the A2 record and `A3_final/freeze.json` once, then verifies. Refuses if anything differs | the 2 frozen records + master |
| 4 | `A3_03_holdout_exp0_6` | GPU | restores the numerical environment of the completed units, checks out the **release pin** in a worktree, runs `final.py holdout --exp exp0 … exp6`; verifies the freeze before/after each call | holdout results in `FINAL_v1/results/` (by the pinned code) |
| 5 | `A3_04_holdout_exp8a` | GPU | same, at the **A2 pin**: `final.py holdout --amendment A2 --exp exp8a` | holdout results in the A2 namespace |
| 6 | `A3_05_final_report` | CPU | verifies the freeze and that every unit completed; new development report and a separate descriptive holdout report | `A3_final/reports/` |

If Colab disconnects: open the notebook and *Run all* again (completed units are returned unchanged, interrupted ones
recomputed). If the A3 code changes before the freeze exists, the A3 pin moves automatically; after it exists use
`UPDATE_CODE = True` deliberately (this never touches the release or A2 pins). `ACCEPT_ENVIRONMENT_DIFFERENCE` stays `False`.

## Rerunning A3_01 after the numerical-environment correction

1. Open `notebooks_a3/A3_01_development_report.ipynb` from branch `claude/a3-final-analysis` (Colab: *File → Open
   notebook → GitHub*, or the link in `notebooks_a3/README.md`). A CPU runtime is enough.
2. *Runtime → Run all*. The setup cell clones the branch head; the A3 code pin is not frozen yet, so it moves to the new
   commit automatically. If it reports an older pin, set `UPDATE_CODE = True` in the first cell and run again.
3. Expected audit output: all `[PASS]`, and one `[WARN] documented compatible differences only (Python patch version,
   PyTorch cpu / CUDA build)` naming the two environments, followed by the table of results per namespace, experiment and
   environment. Any `[FAIL]` stops the notebook and is a genuine mismatch to investigate.
4. Outputs (new files, nothing overwritten): `A3_final/audit/audit_*.json`, `environment_inventory_*.csv`,
   `environment_summary_*.csv`, and the report folder `A3_final/reports/development_*/` with the environment provenance
   section in `development_results.md`.

## Command line (same logic; from a checkout of the A3 branch with `configs/paths.yaml`)

```
python scripts/a3.py plan --no-drive          # the 116 units, from configs only (no Drive)
python scripts/a3.py audit                    # read-only reproducibility audit
python scripts/a3.py freeze                   # dry run; --write only after approval
python scripts/a3.py verify
python scripts/a3.py holdout --part release --exp exp3 --python <restored python> --repo . --base /content
python scripts/a3.py holdout --part A2 --exp exp8a --python <restored python> --repo . --base /content
python scripts/a3.py holdout-status | report | holdout-report | d2-manifest
python scripts/a3.py smoke                    # synthetic end-to-end check (temporary root)
```
