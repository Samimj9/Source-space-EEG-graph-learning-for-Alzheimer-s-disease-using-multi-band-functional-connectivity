# Unified framework: progress log (resumable)

Branch `claude/unified-final-framework`, started 2026-10-17 from `607b5ea` (tag `backup/pre-unification-607b5ea`;
the old branch `claude/ecstatic-davinci-a2e8mh` and every Drive pin are untouched and remain the recovery point).

| Stage | State | Notes |
|---|---|---|
| 1 audit existing repository | done | LX / WX share `lcmvxu`; 144 configs, 93 notebooks, 5 pin files; D1/D2 code unused by the D3 suites except 5 helpers of `pipeline.py` |
| 2 unified architecture | done | `release`, `stores`, `sourcegraphs.config` / `registry`, `configs/source_graphs.yaml` (reproduces both store hashes) |
| 3 consolidate LX and WX | done | `lcmvxu` -> `sourcegraphs`; one config per experiment; inverse as run context; parity tested per variant |
| 4 graph construction | done | `graphs.construction`, `graphs.diagnostics`, `configs/final/graphs.yaml`; selection rule `experiments.selection` |
| 5 QC / outliers | done | `qc` (A / B / C, fold-level, gate); `tests/test_qc.py` |
| 6 parcellation | done (provisional numbers) | atlases, `source.resolution`; approximate-BEM and sphere analyses here (agree); exact-BEM rerun in notebook 02; r18 recommended (stores need approval) |
| 7 Exp 0-8 | done | `configs/final/experiments/`; band identity for every shared multiband encoder |
| 8 fixed 60 epochs | done | `gnn_fixed60_v1`; trainer bit-identical to 607b5ea |
| 9 reporting | done | `evaluation.reporting`, `experiments.report` (DeLong, bootstrap, Wilson; LCMV vs wMNE, B vs A) |
| 10 tests + smoke tests | done | full suite + `scripts/final.py smoke` (results in the commit message / final report) |
| 11 fixes | done | feature-similarity strength; legacy DE (+ memory view); recommend_atlas empty table; paired DeLong zero variance |
| 12 docs + notebooks | done | `docs/final/*`, `FINAL_RUN_GUIDE.md`, CLAUDE.md, DESIGN v2.0, log entry 58, decisions row 61; 14 notebooks |
| 13 commit + push | done | see `git log` |

## Review of the 62 original refactor changes (2026-10-18)

Every changed file was read. Splits (only the frozen protocol; identical folds), training (identical loop),
normalization (training-only), labels (diagnosis only), store loading (verified + locked), aggregation (mean window
score) and evaluation (pooled + fold AUC, CIs) are correct. Defects found and fixed: the feature-similarity graph's
node strength; the test suite did not load (`conftest` imported a removed helper); entry points, notebooks, docs and
tests were missing; the legacy builder was not importable (now `adgraph.legacy`).

## Three decisions before training (2026-10-18)

ROI resolution, graph construction and participant QC: evidence, defaults and approvals in `FINAL_RUN_GUIDE.md` §3.

## Final preparation pass (2026-10-18; methodology log entry 59)

r18 investigation approved: atlas checks, real-operator resolution, r18 stores from the recordings (validated, matched
to LCMV and to dk68), r18 QC, the controlled dk68 vs r18 comparison (notebook 04), atlas decision by the author, graph
comparison on that atlas, development CV only until the frozen final analysis (notebook 99).

## Atlas arms (2026-10-19; methodology log entries 59a, 59b)

Checkpoint diagnostics fixed for the corrected LCMV store; exact-BEM rule: lobes12 recommended, r18 not eligible
(unchanged). Authorized: lobes12 (rule) and r18 (exploratory, amendment A1); notebook 04 compares dk68, lobes12, r18
and selects nothing.

## Pending (author)

1. Category A QC decisions, if any (notebook 01).
2. The analysis atlas after notebook 04 (r18 stores approved; built in notebook 02).
3. Review of the graph-construction decision (notebook 03, on that atlas).
4. Deletion proposals of `inventory.md`.
5. Authorization to run the classification notebooks.
