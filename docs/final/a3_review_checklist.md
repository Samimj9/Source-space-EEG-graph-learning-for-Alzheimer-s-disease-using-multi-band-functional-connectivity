# Amendment A3 — review checklist (before authorizing the freeze and the holdout)

Nothing below has been executed on real data. The unit list is `docs/final/a3_freeze_inventory.csv` (generated from
configuration by `python scripts/a3.py plan --no-drive`; 116 rows).

## Units that would be frozen (atlas lobes12, graph dense, window mode B, LCMV and wMNE)

| Part | Pin | Experiment | Variants × contexts | Units |
|---|---|---|---|---|
| release | `59a9ab7` | exp0 (sensor space, dk68, no graph) | 4 variants × 1 context | 4 |
| release | `59a9ab7` | exp1 | 4 × 2 | 8 |
| release | `59a9ab7` | exp2 | 10 × 2 | 20 |
| release | `59a9ab7` | exp3 | 10 × 2 | 20 |
| release | `59a9ab7` | exp4 | 4 × 2 | 8 |
| release | `59a9ab7` | exp5 | 8 × 2 | 16 |
| release | `59a9ab7` | exp6 | 10 × 2 | 20 |
| A2 | `7fa470c` | exp8a | 10 × 2 | 20 |
| | | **total** | | **116** (= `holdout.yaml` with exp8 → exp8a) |

Not frozen, development-only: Exp 7; window mode A; atlases dk68 / r18 comparisons; non-selected graph constructions;
the original exp8 (failed, kept as provenance).

## Review items

- [ ] The single protocol change (exp8 → exp8a) and its justification (log entry 61) are acceptable.
- [ ] `configs/final/holdout.yaml` is unchanged (SHA-256 `fc1a6988…`); `holdout_A3.yaml` differs only by the substitution.
- [ ] Dry run of `A3_02` on Drive: every check PASS; inventory shows 116 units with development AUCs; the units' commits are the pins.
- [ ] `A3_01` audit: 40 Exp 8a configurations, 200 / 200 folds, release results all at `59a9ab7`, one numerical environment, no completed original Exp 8 result.
- [ ] Interpretation, comparison and selection-bias sections of the development report read correctly (descriptive vs supported; Exp 8a node-strength caveat).
- [ ] Holdout use understood: descriptive, 13 previously inspected participants, once per unit, no selection, no tuning.
- [ ] D3 → D2 manifest and its recommended source-only set are acceptable as a pre-specified list; D2 untouched.
- [ ] Approve A3: `approved_by` and `status: approved` in `configs/final/amendments.yaml`, committed.

## Safety properties (tested; see `tests/test_a3.py`)

Pins and completed results untouched; frozen records written once and checksummed; any later change to a record, a
development result, a pin, the store lock or the QC decisions fails `a3.py verify`; holdout refused without the master, at
another commit, for unfrozen units, mode A and Exp 7; participant-disjoint fitting (52 development / 13 holdout); resumable.
