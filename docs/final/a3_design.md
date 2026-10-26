# Amendment A3 — design record

Methodology log entry 61; configuration `configs/final/amendments.yaml: A3`, `configs/final/amendments/holdout_A3.yaml`,
`configs/final/amendments/a3_statistics.yaml`. Code: `src/adgraph/a3/`, `scripts/a3.py`, `scripts/a3_setup.py`,
`scripts/make_a3_notebooks.py`, notebooks `notebooks_a3/`. **Status: implemented and tested; not approved; no holdout run.**

## 1. What is amended, and why

The pre-specified final analysis (`configs/final/holdout.yaml`, unchanged, SHA-256 `fc1a6988…571ecc`) contains exp8. On the
recorded atlas lobes12 exp8 has no development-CV result (its anatomical layer is empty; every unit failed before
training), so it cannot be frozen. Amendment A2 defined exp8a (same 10 variants, cortical-surface anatomical layer) and
completed its development CV in its own namespace. **A3 substitutes exp8a for exp8 in the frozen final analysis. This is the
only protocol change.** Exp 7, window mode A, unrecorded atlases and graph constructions remain development-only; the
holdout (13 participants, inspected in earlier work) is descriptive, evaluated once per unit, never used for selection.

## 2. Namespaces, pins and records

```
FINAL_v1/CODE_COMMIT.txt                      release pin 59a9ab7      (read only)      Exp 0-6 dev results + holdout code
FINAL_v1/results/                             Exp 0-7 dev results      (read only; holdout results later, by the pinned code)
FINAL_v1/decisions/final_analysis.json        release frozen record    written ONCE by the A3 freeze (Exp 0-6, commit 59a9ab7)
FINAL_v1/amendments/A2_exp8_surface/CODE_COMMIT.txt     A2 pin 7fa470c (read only)
FINAL_v1/amendments/A2_exp8_surface/results/            Exp 8a dev results (read only; holdout results later)
FINAL_v1/amendments/A2_exp8_surface/decisions/final_analysis.json   A2 frozen record, ONCE (Exp 8a, commit 7fa470c)
FINAL_v1/amendments/A3_final/CODE_COMMIT.txt  A3 pin (A3 code; free to move until freeze.json exists)
FINAL_v1/amendments/A3_final/freeze.json      master record: SHA-256 of both records, pins, protected files
FINAL_v1/amendments/A3_final/{reports,audit,d2_manifest,manifests}/    A3 outputs (never overwritten)
```

## 3. Gates (all tested)

| Gate | Where |
|---|---|
| holdout.yaml unchanged; plan = holdout.yaml with `exp8 -> exp8a`; Exp 7 / mode A outside | `freeze.preconditions` |
| both pins as recorded; A2 base pin = release pin; atlas lobes12 / graph dense recorded | `freeze.preconditions` |
| every planned unit has a complete (marker + payload checksums), development-CV result whose identity names the part's pin | `freeze.dev_results` |
| no conflicting frozen record; no holdout result before the freeze | `freeze.preconditions` |
| A3 approved by the author (`approved_by`, `status: approved`) and clean checkout | `freeze.freeze(write=True)` |
| records written once (hard link; never overwritten); master written last; rerun completes an interrupted freeze | `namespace.write_once`, `freeze._matches` |
| record / result / pin / store-lock / QC-decision tampering detected | `freeze.verify` (before and after each holdout call) |
| holdout without master, in a worktree not at the pin, or with an interpreter that does not import it | `holdout.run_part` |
| holdout of a unit outside the record, under another commit, mode A, Exp 7 | pinned code: `decisions.require_frozen`, `amendment.holdout` |
| A3 code outside the experiment code identity scope (no completed checkpoint invalidated) | `tests/test_a3.py` (git diff vs the A2 pin) |

## 4. Statistics and reading rules

Unit = participant; matched participants only; paired DeLong + stratified participant bootstrap CI of the AUC difference;
Holm within a comparison family; *supported* only if Holm p < 0.05 **and** the CI excludes 0. The best of many variants is
reported with a bootstrap optimism estimate, a max-statistic permutation p and a selection-aware best-classical versus
best-GNN difference. Windows are never samples. All development results are on 52 participants; non-significance is not
equivalence. Development versus holdout is descriptive, in the holdout report only.

## 5. What A3 deliberately does not do

It does not edit `holdout.yaml`; move a pin; retrain, delete or overwrite any result; compute QC; modify stores; read holdout
metrics in the development report; use D2 data or outcomes; or run anything without the author's approval.
