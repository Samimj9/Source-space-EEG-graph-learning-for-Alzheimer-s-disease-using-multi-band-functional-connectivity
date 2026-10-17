# 200-epoch early-stopping protocol for the LCMV-Xu and wMNE-Xu GNNs (`gnn_fixed_hparams_es200_v1`)

Version 1.0, 2026-10-17. Methodology log entry 56, decisions row 59. Applies to the GNN experiments Exp 2–8 of both
suites. Classical Exp 0 / 1 (LR, SVM) have no epochs and are unchanged. The fixed 60-epoch protocol
(`gnn_fixed_hparams_v1`) and all of its results stay as they are. The two protocols are separate, versioned
experiments.

## 1. Protocol

| | 60-epoch (existing) | early stopping (new) |
|---|---|---|
| Configs | `configs/experiments/{lx,wx}/…` | `configs/experiments/{lx,wx}_es/…`; they `extend` the 60-epoch configs and change only `protocol` and `training` |
| Architecture, lr, weight decay, batch size, dropout, seeds, splits, reporting | — | identical |
| Epochs | 60, the last epoch is used | at most 200 |
| Monitor | — | inner-validation loss: participant- and class-balanced BCE, the training loss without the regularizer |
| Improvement | — | val < best − `min_delta`, with `min_delta = 1e-4` (absolute; about 0.015 % of ln 2, so float-noise changes in a ≈0.69-scale loss do not reset patience) |
| Patience | — | 10 consecutive epochs without improvement |
| Model used | after epoch 60 | weights of the best epoch (restored) |
| Refit | — | none (`refit: false`; §3) |

**Inner validation.**
- In every outer fit, the training participants alone are split once, with `evaluation.splits.subject_holdout`.
- 20 % of each class, at least one, become validation participants. Every window of a participant stays on one
  side. The split is seeded by the fit's seed.
- Scalers and the sparsifier are fitted on the inner-fit participants only.
- `GNNModel.fit` receives only the outer training view (`require_train`), so the outer test participants cannot
  enter. The runner also checks that every inner validation participant belongs to the outer training set.

| Outer fit | Training participants | Inner validation (≈) | Never used for stopping |
|---|---|---|---|
| `xu_dev_cv`, each of the 5 folds | about 41-42 development participants | 8-9 | that fold's development test participants, and the 13 holdout participants |
| `xu_holdout` | the 52 development participants | 11 (6 AD, 5 HC) | the 13 holdout participants |

**Recorded per outer fit** (`hparams.csv`, one row per fold):
- `max_epochs`, `epochs_run`, `best_epoch` (counted from 1), `stopped_early`, `best_val_loss`, `patience`,
  `min_delta`, `restored_best`;
- `n_inner_fit_participants`, `n_inner_val_participants`, `inner_val_keys`;
- `history_train_loss` and `history_val_loss`, the full curves.

`scripts/es_report.py --suite LX|WX` summarises these and plots the curves.

## 2. Comparability of LCMV and wMNE

- The `wx_es` configs equal the `lx_es` configs except for the experiment name and graph family. This is tested
  variant by variant (`tests/test_early_stopping.py`).
- Both suites train with the same commit:
  - The ES notebooks of each suite pin `ES200_CODE_COMMIT.txt` in their suite folder.
  - On its first run, a suite's ES pin adopts the other suite's ES pin if that exists.
  - The notebooks warn if the two pins differ.
  - `wx_compare.py` reports `same_code_commit` for every pair.

## 3. Final model and holdout evaluation

The holdout model is the early-stopped model: trained on about 41 of the 52 development participants, stopped on
the other 11, evaluated once on the 13 holdout participants. The development CV evaluates exactly the same
procedure on each fold, so the CV estimate and the holdout estimate describe one procedure.

A refit on all 52 participants for the selected number of epochs is implemented (`early_stopping.refit: true`;
tested) but not used. A refit transfers an epoch count chosen with about 41 participants to a 52-participant
training set, where the optimum may differ, and the refitted weights are never validated. Enabling it would be a
separate, versioned protocol for both suites.

## 4. Versioning, pins and resumption

- **Results:**
  - `D3_LCMV_XU/experiments_{A_xu_tenfold,B_all_windows}/lx10_es_*/` and `lx_es_*/`;
  - `D3_WMNE_XU/experiments_{A_xu_tenfold,B_all_windows}/wx10_es_*/` and `wx_es_*/`.

  These are new experiment folders. The 60-epoch folders (`lx_exp*`, `lx10_exp*`, `wx_exp*`, `wx10_exp*`) are never
  written by the ES notebooks.
- **Identity.** Each unit's checkpoint identity includes the training configuration and the code fingerprint, so a
  run of one protocol can never resume into the other's results (`IncompatibleCheckpoint`; tested). Interrupted ES
  runs resume per outer fold.
- **Pins:**

  | Pin file | Used by |
  |---|---|
  | `D3_LCMV_XU/CODE_COMMIT.txt` | LX 60-epoch notebooks (`299aed72`) |
  | `D3_LCMV_XU/lx_exp8_xu_dual_gcn_CODE_COMMIT.txt` | LX Exp 8, 60-epoch |
  | `D3_WMNE_XU/CODE_COMMIT.txt` | WX build and 60-epoch notebooks |
  | `D3_LCMV_XU/ES200_CODE_COMMIT.txt` | LX ES notebooks (new) |
  | `D3_WMNE_XU/ES200_CODE_COMMIT.txt` | WX ES notebooks (new) |

  The ES notebooks read the suite pin, never move it, and then check out their own ES pin.
- **Graph code untouched.** This change does not modify any file that determines graph construction. Graph stores,
  the WX build at its pin and running builds are unaffected.
- **Exp 8:** the ES notebooks run Exp 8 at the shared ES pin. The architecture code (`models/xu_dual.py`) is
  unchanged, so no separate Exp 8 pin is needed for the ES protocol.
