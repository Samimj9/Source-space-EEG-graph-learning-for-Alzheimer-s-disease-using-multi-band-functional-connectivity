# Correctness audit of the LCMV-Xu and wMNE-Xu experiments; band identity (corrected variants R1)

Version 1.0, 2026-10-17. Methodology log entry 57, decisions row 60. Tests: `tests/test_xu_audit.py`, together with
the existing suite tests.

## 1. Finding

A `MultiBandGNN` with a shared encoder (`shared_encoder: true`) and mean or attention fusion is **invariant to
permuting the band graphs**:
- one encoder processes every band's graph;
- no input says which band it is;
- per-band z-scoring (`NodeFeatureScaler`, scope `node`) also removes each band's typical level;
- mean and attention fusion are symmetric over bands.

The model therefore cannot represent effects whose direction differs between bands, such as "theta up, alpha down".
An exact paired test makes this concrete. HC participants are band-swapped copies of AD participants. The
original model scores every pair identically, so its AUC is exactly 0.5 (`test_band_identity_restores_band_direction_signals`).

This is not a coding error: the code does what it specifies. It is an **unintended architectural limitation**.

| Experiment | Variants affected |
|---|---|
| Exp 5 | `shared__mean`, `shared__attention` |
| Exp 6 | all 10 variants (shared GCN with attention or mean fusion) |

These apply in both suites (LX, WX), both modes (A, B) and both protocols (fixed 60, ES200).

## 2. Correction (separate, versioned architectural variant)

`model.band_identity: onehot` appends a constant one-hot band indicator (K columns) to every band's node features
before the encoder. The shared weights can then act band-specifically. The default `none` leaves the original model
bit-identical (tested). The new configs carry the original definitions unchanged except for this key:

| Corrected experiment | Variants | Configs |
|---|---|---|
| `{lx,wx}{,10}_exp5_bandid` | `{plv,ciplv} × shared__{mean,attention}__bandid` (4) | `lx/`, `lx/tenfold/`, `wx/`, `wx/tenfold/` |
| `{lx,wx}{,10}_exp6_bandid` | all 10 Exp 6 variants with band identity | same |
| early-stopping versions | same variants, protocol `gnn_fixed_hparams_es200_v1` | `lx_es/`, `lx_es/tenfold/`, `wx_es/`, `wx_es/tenfold/` |

- **Originals:** the original Exp 5 / Exp 6 experiments and results remain as baselines.
- **Notebooks:** `notebooks/{lcmv_xu,wmne_xu}/corrected/{LX,WX}_R1_Exp{5,6}_bandid_{fixed60,es200}.ipynb`.
- **Pins:** each suite has its own pin `R1_CODE_COMMIT.txt`. The second suite adopts the first suite's commit, so LCMV
  and wMNE run identical code.
- **Untouched:** the suite pins, the Exp 8 pin and the ES200 pins.

## 3. Architectures checked for band identity

Every multiband variant of every LX / WX config, in both modes and both protocols, has a test that the model's
output changes under a band permutation exactly when the architecture is meant to be band-aware.

| Experiment | How bands enter | Band identity | Verdict |
|---|---|---|---|
| Exp 0 / 1 (LR, SVM) | flattened feature columns `band × channel/ROI`, fixed order | column position | PASS (intentional) |
| Exp 2 / 3 | one band per experiment | not applicable | PASS (intentional) |
| Exp 4 | adjacency averaged over bands (K = 1); node features = every band's PSD as separate columns | column position | PASS (averaging intentional) |
| Exp 5 separate | one encoder per band | separate parameters | PASS |
| Exp 5 shared, Exp 6 | one shared encoder, symmetric fusion | none | LIMITATION, corrected variant R1 |
| Exp 7 EARM-GNN | shared residual encoder + learned band embedding + band-specific edge gates | learned (initialised equal, separated by training) | PASS |
| Exp 8 Xu dual GCN | band-specific weights; flattened band × ROI × output order into the FC layers (also with the shared-weights ablation) | parameters and flatten order | PASS |

Band order (delta → gamma) is preserved from the store through `build_graph` to the model; tested for every
feature and adjacency position. Attention fusion normalises over the band axis (sum 1, tested). Exp 8's atlas
adjacency is built from the store's own ROI names, in array order.

## 4. Labels, scores and evaluation (all experiments, both suites)

All checked; PASS:
- **Labels:** AD = 1, HC = 0, from the provider codes A / C (`io.bids.load_participants_tsv`); FTD is excluded.
- **Orientation:** classical scores are `predict_proba[:, 1]` (sklearn classes `[0, 1]`) or the SVM margin
  (positive = AD). GNN scores are `sigmoid(logit)` trained with BCE against `y`. For every model family, AD-positive
  synthetic signal gives participant AUC > 0.85; no inversion.
- **Aggregation:** participant score = mean of the window scores, with keys and labels aligned; a label that varies
  within a participant raises.
- **Thresholds:** 0 for the uncalibrated SVM margin, 0.5 for probabilities, in every config.
- **Training-only fitting:** scaling, feature selection, inner CV and inner validation use training participants
  only (`require_train`).
- **Splits:** the frozen split and folds are verified before each run.

Pooled versus fold AUC is reported, not a defect. The primary development-CV metric is the pooled out-of-fold
participant AUC, which ranks scores of five separately trained fold models together. Fold models that are calibrated
differently can lower it relative to the mean within-fold AUC. The reports (`lx_report.py`, `wx_compare.py`) now
also give `participant_mean_fold_auc` and the fold range. The primary metric is unchanged.

## 5. Which results stay valid

- **Valid as reported:**
  - Exp 0–4, Exp 5 separate, Exp 7, Exp 8, in both suites, modes and protocols;
  - Exp 5 shared and Exp 6, as results of their original (band-permutation-invariant) architecture.
- **To be read with the limitation:** Exp 5 shared and Exp 6 cannot use band-specific directions. Their R1 variants
  are the corrected comparison.
- **New runs needed:** only the R1 experiments (§2). No original result is overwritten or needs rerunning.
