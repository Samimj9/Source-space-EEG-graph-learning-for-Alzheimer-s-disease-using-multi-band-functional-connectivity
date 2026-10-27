# Cross-dataset generalization D3 <-> D2 on a matched 19-electrode representation: experimental design

Version 1.2 (2026-10-26), branch `claude/cross-dataset-d3-d2-19ch`, methodology log entries 61-63. **Status: designed,
synthetically validated, author decisions D1-D10 recorded (section 12); only the audit-only Part A of notebook 00 is
authorized. No real-data graph has been built, the campaign is not frozen, no model has been trained and no target
label has been evaluated.** The stages that touch real data beyond the audit (the D2 stores, the training, the
stability analysis) need the author's recorded authorization (section 10). No result exists yet, and nothing in this
document reports or anticipates one.

Code: `src/adgraph/crossdataset/` (`channels`, `d2store`, `namespace`, `compat`, `plan`, `transfer`, `evaluate`,
`report`, `synthetic`), `scripts/cross_dataset.py`. Configs: `configs/cross_dataset/` (`channel_map_19.yaml`,
`d2_store.yaml`, `campaign.yaml`). Notebooks: `notebooks/cross_dataset/` (00-03). Tests: `tests/test_cross_channels.py`,
`tests/test_cross_dataset.py`. Run guide: `docs/cross_dataset/run_guide.md`.

## 1. Objective

How well do the EEG graph neural networks of Exp 3-8a, trained on one independent AD dataset, classify AD vs HC in
another dataset **without any target-domain training** (zero-shot)?

* **Direction A, D3 -> D2:** train on D3 only (the 52 frozen development participants), test on all 23 D2 participants.
* **Direction B, D2 -> D3:** train on D2 only (all 23), test on all 65 D3 participants, reported for all 65 and
  separately for the 52 development and the 13 holdout participants (the 13 were inspected in earlier work:
  not an untouched holdout).

Nothing is fitted on the target: no model parameter, scaler, graph-construction parameter, class weight, threshold,
calibration, hyperparameter, preprocessing setting or model choice. Target labels are read only by the evaluation,
after the campaign is frozen. Unsupervised domain adaptation is out of scope; if ever done, it is a separate,
explicitly labelled supplementary experiment and is never called zero-shot.

## 2. Audit of the repository and the data (findings that shape the design)

| # | Finding | Consequence |
|---|---|---|
| A1 | The D3 stores of the selected atlas lobes12 exist, are validated, matched (wMNE to LCMV; window by window to dk68) and locked by the release. | Reused **read only** (`stores.open_store(lock=False)`: verified against the release lock, never re-locked). No new D3 representation is needed (section 5). |
| A2 | The D3 store builder (`sourcegraphs.build`) is dataset-agnostic below the loader: segments, five-minute selection, rejection, LCMV / wMNE, analytic FIR, PLV, |ciPLV|, PSD, DE. | The D2 stores use the **same per-participant code** with a D2 loader (two backward-compatible hooks: a `loader` argument and an optional "no ds004504 split" in assembly). The D3 stores are verified by config hash, graph_run_id and the release lock, not by code fingerprint, so they are unaffected. |
| A3 | D2 EDFs carry channel numbers only (`Misc ExG1`-`ExG32`); channel k = electrode k of `electrodes_positions_32.loc`. The layout text in the test suite is byte-identical to the Drive file (SHA-256 `da108d02...`). | An explicit, hash-pinned mapping table (section 3). |
| A4 | In the colin27 10-05 template that D3's provider used (EEGLAB `standard_1005.elc`, = MNE `colin27_1005`), T3/T7, T4/T8, T5/P7, T6/P8 have **identical** positions; every D3 recording carries exactly those template positions (methodology log entry 46). | D2 uses **exactly the D3 head geometry**: one forward model, the same wMNE operator, the same LCMV forward (section 3.3). |
| A5 | The experiment runner fits a model on a `train` view and scores any view; every fitted transform lives inside the model. | Source-only fit + scoring of another store needs no change to models, graphs or the runner (`transfer.run_unit`). |
| A6 | `evaluation.splits.make_splits` is hard-wired to ds004504. | The cross-dataset splits are built in `crossdataset` (D3 development keys = `xu_partition`, checked against the store's `xu_split.json`; D2 CV with the same stratified participant folds). |
| A7 | D2 was analysed within-dataset in earlier work (legacy D1 / D2: 30 channels, wMNE, 68 ROIs, 20 s windows); the model families were designed with knowledge of those results and of earlier D3 results; the D3 holdout was inspected earlier. | Disclosed in every report (`plan.DISCLOSURES`); D2 is an independent dataset for this transfer but not an untouched one. No configuration is chosen on D2 or on transfer results. |
| A8 | Exp 8 is undefined on lobes12 (zero anatomical layer); amendment A2 defines Exp 8a. Exp 8a is running in Colab at its own pin. | The campaign uses `exp8a` from A2's config (unchanged), refuses `exp8`, and never reads or writes the A2 namespace except its completed development-CV results (read only, as the D3 source-only estimate). |
| A9 | D2 metadata: AD about 80 y (66-92), HC about 62 y (55-71) (`docs/data_inventory.md`); D3: similar ages. | Age analyses are part of the primary report (section 9). |
| A10 | Unknown from the repository: the D2 amplifier, online reference, hardware filters and **eyes condition**. | The audit extracts the EDF technical header fields (prefiltering, units, ranges; never patient / recording identity); the eyes condition must be confirmed by the author (decision D6). |

## 3. Electrode harmonization (19 matched electrodes)

### 3.1 The mapping table (`configs/cross_dataset/channel_map_19.yaml`)

| D3 (store order) | D2 label | EDF index | EDF label | .loc theta, radius | colin27 MNI (mm) |
|---|---|---|---|---|---|
| Fp1 | Fp1 | 1 | Misc ExG1 | -18, 0.479 | -29.44, 83.92, -6.99 |
| Fp2 | Fp2 | 3 | Misc ExG3 | 18, 0.479 | 29.87, 84.90, -7.08 |
| F3 | F3 | 5 | Misc ExG5 | -39, 0.318 | -50.24, 53.11, 42.19 |
| F4 | F4 | 7 | Misc ExG7 | 39, 0.318 | 51.84, 54.30, 40.81 |
| C3 | C3 | 15 | Misc ExG15 | -90, 0.247 | -65.36, -11.63, 64.36 |
| C4 | C4 | 17 | Misc ExG17 | 90, 0.247 | 67.12, -10.90, 63.58 |
| P3 | P3 | 25 | Misc ExG25 | -141, 0.321 | -53.01, -78.79, 55.94 |
| P4 | P4 | 27 | Misc ExG27 | 141, 0.321 | 55.67, -78.56, 56.56 |
| O1 | O1 | 30 | Misc ExG30 | -169, 0.482 | -29.41, -112.45, 8.84 |
| O2 | O2 | 32 | Misc ExG32 | 169, 0.482 | 29.84, -112.16, 8.80 |
| F7 | F7 | 4 | Misc ExG4 | -54, 0.491 | -70.26, 42.47, -11.42 |
| F8 | F8 | 8 | Misc ExG8 | 54, 0.491 | 73.04, 44.42, -12.00 |
| **T3** | **T7** | 14 | Misc ExG14 | -90, 0.5 | -84.16, -16.02, -9.35 |
| **T4** | **T8** | 18 | Misc ExG18 | 90, 0.5 | 85.08, -15.02, -9.49 |
| **T5** | **P7** | 24 | Misc ExG24 | -126, 0.491 | -72.43, -73.45, -2.49 |
| **T6** | **P8** | 28 | Misc ExG28 | 126, 0.491 | 73.06, -73.07, -2.54 |
| Fz | Fz | 6 | Misc ExG6 | 0, 0.234 | 0.31, 58.51, 66.46 |
| Cz | Cz | 16 | Misc ExG16 | 0, 0 | 0.40, -9.17, 100.24 |
| Pz | Pz | 26 | Misc ExG26 | 180, 0.239 | 0.32, -81.11, 82.61 |

Never read: Fpz, FC5, FC1, FC2, FC6, M1, M2, CP5, CP1, CP2, CP6, POz, Oz. The final canonical order is the order of the
validated D3 stores (`configs/source_graphs.yaml`), so the sensor arrays of both datasets align row by row. (The
anatomical order Fp1, Fp2, F7, F3, Fz, ... is a different listing of the same 19 electrodes.)

### 3.2 Verification (each failure stops immediately; `channels`, tests)

1. Table: 19 rows = the D3 store channels in order; 19 distinct EDF indices; exactly the four equivalences; identity
   for every other electrode; EDF label = `Misc ExG<index>`. The table itself has a hash recorded in every D2 store
   and in the frozen campaign.
2. Layout: SHA-256 of the `.loc` file; the label at every table index is the table's D2 label with the table's polar
   coordinates; the 13 unmapped labels are exactly the excluded list.
3. EDF (every recording): the header holds every number 1..32 exactly once as an index-only label; the 19 mapped
   numbers are read **by number** (never by position), renamed to the D3 labels and put in the D3 order.
4. Geometry: the colin27 positions of the D2 labels equal those of the D3 labels and the table (<= 0.01 mm); every
   `.loc` direction lies within 15 deg of its position on a sphere fitted to the 19 positions (observed 1.4-12.4 deg);
   the angle-minimising assignment of the 19 `.loc` directions to the 19 positions is the table (no swap); odd
   electrodes left, even right, midline at x ~ 0, `.loc` theta on the matching side. Units (mm -> m), axes (RAS) and
   frame (MRI / MNI with the fsaverage fiducials, as for D3) are those of the D3 geometry.
5. Audit outputs (`audit/channel_mapping_audit.csv`, `audit/audit.json`): one row per channel with mapping, EDF
   index, coordinates and angle; per recording, that all 19 are present once.

### 3.3 Head geometry and source model

D2 has no digitized positions; neither has D3 (template lookup). The D2 geometry is **the established D3 geometry**
(read from the D3 `.set` files; all 65 identical), recorded under its own source label with the D3 geometry hash, and
verified against the `.loc` layout (3.2.4). Hence one forward model (fsaverage ico-5, BEM, mindist 5 mm) and the
identical wMNE operator for both datasets; LCMV uses the same forward with each participant's own covariance. No
coordinate is substituted silently: the substitution of a template is the documented harmonization step, and the
`.loc` agreement is a hard gate. **Identical template coordinates do not establish identical physical electrode
placement** (decision D1): neither dataset has digitized positions, caps and heads differ, and the shared geometry
only removes a modelling difference (two forward models), not the real placement error, which remains part of the
source-localization uncertainty of both datasets.

### 3.4 Order of operations

EDF header -> select 19 channels by number -> crop -> band-pass -> crop -> per-segment resampling -> **common average
of the same 19 electrodes** -> rejection -> source reconstruction. The original 30 / 32 D2 channels never enter the
reference or the inverse. Tested: a 5 mV signal on all 13 unselected EDF channels leaves the loaded data, the common
average (over 19 channels) and the segments bit-identical (`tests/test_cross_channels.py`).

## 4. Matched signal processing

### 4.1 Identical to the D3 stores (inherited configuration keys; compared key by key)

250 Hz (each continuous segment resampled on its own); common average reference of the 19 electrodes; five bands
(delta 0.5-4, theta 4-8, alpha 8-13, beta 13-25, gamma 25-45 Hz); analytic FIR band filters (1 Hz transitions) and
the derived window margin; 10 s windows, 1 s step; five-minute deterministic selection; adaptive peak-to-peak rejection
(median + 5 MAD, > 15 % of channels); minimum 5 windows; mode A = first 10 accepted windows; fsaverage ico-5, BEM,
mindist 5 mm; LCMV (reg 0.05, unit-noise-gain-invariant, rank = data, covariance of the participant's own accepted
samples) and wMNE (loose 0.2, depth 0.8, lambda2 1/9, normal component); lobes12 regions with mean_flip, same order;
Welch PSD (2 s Hann, 50 %), absolute / log / relative (0.5-45 Hz) band power, differential entropy; PLV and |ciPLV|
stored separately; flat-channel and amplitude-plausibility checks. The D2 configuration is
`resolve(inverse, 'lobes12')` + the overrides of `d2_store.yaml`; the test suite checks that only those keys differ.

### 4.2 D2-specific steps (the only differences in the configuration)

| Step | D2 | Why |
|---|---|---|
| Input | EDF, 19 of 32 channels by number | section 3 |
| Edge crop | 10 s at each end before any filter | legacy D2 rule (entry 41); D3 used as released |
| Band-pass | 0.5-45 Hz Butterworth, order 4, zero phase (forward-backward) | reproduces the provider's documented D3 filter; the provider's order and phase are not documented |
| Unit checks | every selected channel: EDF physical dimension in {V, mV, uV, nV}; robust std of the band-passed signal in [1e-7, 1e-3] V; otherwise the participant stops (`UnitError`, naming the channel) | mixed `V` / `uV` headers exist (D2-19); MNE scales each channel by its own unit, a mislabelled channel would be off by 1e6 (12.7) |
| **Line-noise notch (D2 only)** | zero-phase FIR notch (firwin), stop band 49-51 Hz, 1 Hz transitions, after the band-pass, before the settle crop, resampling, common average, rejection and source reconstruction | acquisition-specific 50 Hz interference measured in Part A (12.2); D3 is provider-preprocessed and not re-filtered |
| Settle crop | 8 s at each end after band-pass + notch (was 5 s) | band-pass impulse response < 1e-4 of its peak beyond 3.73 s + the notch FIR's exact half-length 3.30 s = 7.03 s |
| Geometry | the D3 positions | section 3.3 |
| Unit scale | none beyond MNE's per-channel EDF unit | amplitude checked per channel and per participant |

### 4.3 What is not harmonized, and why (decision D2)

* **ASR / ICA.** D3 was cleaned by its provider (ASR; ICA with ICLabel removing eye and jaw components); D2 is raw. The
  pinned stack has no validated ASR implementation, and ICA + ICLabel would add a classifier-dependent, threshold-
  dependent cleaning step whose settings would have to be chosen for D2 without a validation reference. Applying them
  "to mimic D3" would replace one documented difference with an unvalidated one. **Primary analysis: no ASR / ICA on
  D2**; the identical adaptive window rejection is the shared artifact control. The difference is reported as
  *different* and is a candidate explanation of any transfer loss. A possible pre-specified sensitivity arm (ICA with
  `mne-icalabel`, fixed thresholds, applied identically to both datasets' raw data where available) is listed in
  section 12 and is not implemented.
* **Line noise (harmonized by a D2-specific correction).** D2 receives an acquisition-specific 50 Hz notch; D3 remains
  provider-preprocessed and is not re-filtered (12.2). The two datasets therefore reach comparable line-noise levels by
  different routes; this is a dataset-specific correction of a dataset-specific artifact, not identical
  preprocessing.
* **Reference.** D3: A1-A2 then common average; D2: recording reference then common average. The common average of the
  same electrodes removes any common reference signal exactly, so this is *identical* after referencing (the
  provider's ASR / ICA did act in A1-A2 space).
* Acquisition hardware, site, sampling rate (500 vs 1024 Hz), recording length, data removed by the provider, eyes
  condition and population are *different* and are stated, never hidden.

## 5. Stores and compatibility

New: `<processed>/CROSS_D3_D2_19CH_v1/stores/graphsXD2{L,W}{,10}_v1_<hash>` (LCMV / wMNE, mode B / A), all 23 D2
resting-state participants, built resumably (one checkpoint per participant), validated (`validation.json`), no
ds004504 split. D3: the validated lobes12 stores (Exp 1, 3-8a) and the LCMV dk68 store (Exp 0 sensor features, as in
the release), read only; no new D3 representation is created because every property is already identical (4.1) and
the D2 representation was built to match them.

The compatibility report (`audit/<inverse>_mode_<m>/compatibility_report.csv`, `compat.compatibility`) lists every
property as *identical* (verified in code or data: configuration keys, ROI names and order, band order and edges,
sensor channels and order, array set and trailing dimensions, definitions, finiteness, adjacency validity,
labels, unique identities, disjoint participant keys, graph construction, normalization convention), *approximate*
(band-pass, line noise, original rate, edge handling) or *different* (artifact cleaning, site / hardware, recording
condition, population, continuity). Any unexpected difference fails the gate. Label-blind shift diagnostics
(`shift_diagnostics.csv`: windows per participant, rejections, amplitude, relative band power, mean PLV / |ciPLV| per
band; per dataset, never per diagnosis) and D2 participant QC on the label-masked store are saved with it; the
harmonization manifest records the mapping, both store identities and the D2 preprocessing.

## 6. Models (unchanged definitions)

| Experiment | Variants | Role |
|---|---|---|
| Exp 3 single-band GCN, matched PSD | 10 (PLV / |ciPLV| x 5 bands) | GNN |
| Exp 4 band-mean GCN, PSD or DE | 4 | GNN |
| Exp 5 multibranch, shared (band one-hot) / separate, mean / attention | 8 | GNN |
| Exp 6 PSD / strength / fusion / band ablations | 10 | GNN |
| Exp 7 EARM-GNN and ablations | 6 | GNN, exploratory |
| Exp 8a Xu-inspired dual GCN, surface anatomical layer (amendment A2) | 10 | GNN, exploratory |
| Exp 0 sensor PSD / DE (LR, SVM) | 4 | reference |
| Exp 1 source PSD / DE (LR, SVM) | 4 | reference |

All 56 variants are loaded from the existing YAMLs (`plan.variants`), never redefined; the original Exp 8 is refused.
GNNs: `gnn_fixed60_v1` (exactly 60 epochs, no early stopping, no search); classical: inner participant CV for C inside
the source training participants only.

## 7. Training and zero-shot evaluation protocol

* **Source participants.** D3 -> D2: the 52 development participants of the frozen split (= `xu_partition`, checked
  against the store's `xu_split.json`); the 13 holdout participants never enter. D2 -> D3: all 23.
* **Fit.** A fresh model per unit (direction x inverse x mode x experiment x variant x seed), seed 0 = the protocol
  seed (for D3 -> D2 this is exactly the release's holdout-fit configuration). Every fitted transform (node-feature
  scaler, graph construction, subject- and class-balanced weights, classical inner CV) sees only the source view.
* **Scoring.** The target store is blinded (labels -1 / "masked", participant metadata removed) before its dataset is
  built; tensor shapes, feature names, band and ROI order and adjacency kind are checked equal; the fitted model scores
  every target window; one mean P(AD) per participant (SVM: mean margin) is saved without labels.
* **Threshold.** 0.5 (SVM margin 0); never tuned.
* **Source-only estimates.** D2 -> D3: participant-stratified 5-fold CV of D2, **one repeat**, the primary D3
  protocol (decision D7). D3 -> D2: the completed release / A2 development CV of the same variant, used only if its
  training-config hash, D3 store graph_run_id, setting, experiment, variant and run context all equal this campaign's
  (decision D9; else reported as missing or mismatched; optional recomputation is not part of the primary plan).
* **Seeds.** Primary: seed 0. The separately authorized **stability analysis** (decisions D7, D10;
  `campaign.yaml: stability`, authorization `stability`) adds training seeds 1-4 and the repeated D2 source CV
  (5 folds x 5 repeats, `source_cv_stability/`); it never selects anything and is reported apart from the primary
  analysis.
* **Freeze.** Before any training, `plan/frozen.json` records the campaign (hash), all units, the store graph_run_ids,
  the code commit and the decision audit (recorded atlas lobes12 and graph dense; A2 record; no target evaluation
  yet). Training and evaluation refuse to run without it, with another campaign or at another commit.
* **Model-family comparison.** Exploratory and descriptive; no winner is selected; every variant is reported.

## 8. Evaluation and reporting

Per direction (never pooled), per inverse and window mode (never pooled), per unit, at the participant level:

* ROC-AUC with DeLong 95 % CI and a participant bootstrap CI (2000 draws, stratified by diagnosis); balanced accuracy,
  sensitivity (AD), specificity (HC) with Wilson and bootstrap CIs; F1, MCC, confusion matrix; n AD / n HC;
* source-only CV AUC and balanced accuracy; transfer degradation (target - source) with a CI from independent
  participant bootstraps of both;
* D2 -> D3 strata: all 65 (primary), 52 development, 13 holdout (descriptive; inspected in earlier work, so not an
  untouched holdout);
* paired comparisons only on identical target participants: each GNN variant vs the reference baseline (Exp 1 LR on
  source PSD, same direction / inverse / mode), paired DeLong, Holm within each direction x inverse x mode family;
  labelled exploratory; small ranking differences are not interpreted (D2 has 23 participants);
* per-participant predictions with provenance: key, subject id, true label, mean P(AD), number of windows, source and
  target dataset, inverse, mode, experiment, variant, seed, threshold, D3 partition, git commit, source fingerprint
  (on Drive only).

Windows are never treated as independent observations.

## 9. Age, sex and acquisition shift

* Demographics by dataset and diagnosis (age mean / SD / range, sex, MMSE).
* **Age-only baseline:** logistic regression on age, (a) fitted on the source and applied to the target, (b) participant
  CV inside the source. In D2, age separates the groups strongly (AD 66-92 y, HC 55-71 y); a strong D3 -> D2 AUC is
  therefore not evidence of a disease-specific EEG signature unless the EEG models exceed this baseline and survive the
  analyses below.
* Per unit: Spearman association of the target scores with age inside each diagnosis; the **age-overlap subset**
  (target participants within [max(min AD age, min HC age), min(max AD age, max HC age)]; an AUC only with >= 3
  participants per class, else "not estimable"); and y ~ age + score vs y ~ age (likelihood ratio). All exploratory,
  n stated; with D2's age ranges the overlap is expected to be small.
* Discussion items (report README and paper): sex distribution, recording hardware, referencing, sampling rate,
  filter characteristics, artifact removal, source-localization uncertainty (template anatomy and positions for both
  datasets; 19 electrodes; 12 lobar regions). Dataset shift is distinguished from disease effects;
  electrode matching does not remove domain differences.

## 10. Reproducibility, checkpointing and safety

* **Namespace** `<processed>/CROSS_D3_D2_19CH_v1/` with its own code pin (`colab_setup.py --namespace cross`); the
  release pin, the A2 pin, `FINAL_v1/` (decisions, lock, QC, results) and the D3 stores are read only; `namespace.guard`
  refuses any write path outside the namespace (tested with snapshots of every protected file).
* **Authorizations** (`authorizations.json`, written once by `authorize --by <name>`): `build_d2` before any D2 store
  (also a smoke build) and `train` before real-data training. Without them the code refuses.
* **Manifests**: every D2 store (config, config hash, base D3 config hash, mapping hash, geometry, inputs with SHA-256,
  code identity, versions); the harmonization manifest; the frozen campaign; every transfer unit's marker (identity:
  campaign hash, training-config hash, source and target graph_run_id, source participants, seed, code identity) and
  `training.json` (variant config, inputs, training history).
* **Resume**: D2 store participants, transfer units and D2 CV folds are checkpointed with validated markers (payload
  first, marker last); completed units are reused; another identity raises `IncompatibleCheckpoint`; `force` moves
  to `_superseded/`; nothing is deleted.
* **Code identity**: scopes `cross_store` and `cross_experiment` (the release scopes plus `crossdataset`); the release
  scopes and their fingerprints are unchanged, so the running Exp 8a and the completed units are unaffected.
* **Environment**: the notebooks restore the Exp 7 numerical environment (`numerical_env`) and run every step with it.
* **Data**: recordings, participant sheets and every per-participant output stay on Drive; nothing is committed.

## 11. Results tables and figures (specification; filled only from saved outputs)

* **Table 1 (main, one per direction):** rows = experiment / variant (grouped: reference, GNN, exploratory GNN),
  columns per inverse x mode: source-only CV AUC; target AUC [95 % DeLong CI]; balanced accuracy [bootstrap CI];
  sensitivity / specificity [Wilson]; F1; MCC; degradation in AUC [CI]. n AD / n HC in the header.
* **Table 2:** D2 -> D3 by D3 partition (development 52, holdout 13; holdout descriptive and
  previously inspected).
* **Table 3:** paired comparisons vs the reference baseline (difference, SE, p, Holm p; exploratory).
* **Table 4:** demographics; age-only baseline (transfer and within-source); age-overlap subset; score-age
  associations; score-adds-to-age tests.
* **Table S1:** compatibility report (identical / approximate / different); **S2:** channel mapping audit; **S3:**
  shift diagnostics; **S4:** stability seeds (if authorized).
* **Figure 1:** forest plot per direction x inverse x mode: target AUC with CIs per variant, source-only CV marked,
  chance line. **Figure 2:** participant score vs age by diagnosis for the reference baseline (and any variant
  discussed). **Figure 3 (planned):** source vs target AUC per variant. **Figure 4 (planned):** dataset-shift
  diagnostics (label-blind distributions). Rules: participants, not windows; PLV and |ciPLV| separate; no significance
  marks except from the saved, corrected comparison table; synthetic outputs watermarked; never overwritten.

## 12. Recorded decisions (author, 2026-10-26)

| # | Decision | Recorded |
|---|---|---|
| D1 | D3 template geometry for D2 | approved, provided the `.loc` mapping and the real EDF headers pass every check; identical template coordinates do not establish identical physical placement (3.3) |
| D2 | ASR / ICA on D2 | not in the primary analysis; the preprocessing mismatch with D3 is a stated limitation (4.3) |
| D3 | 50 Hz notch | measured in Part A (12.2), then: D2-specific zero-phase FIR notch 49-51 Hz (author, 2026-10-26); D3 not re-filtered; decided before any graph build and any classification |
| D4 | D2 band-pass | provisionally approved as a harmonization choice, not a reproduction of the provider's filter (order and phase undocumented); response verified (12.1) |
| D5 | D2 edge crops | 10 s before the band-pass; the post-filter crop is 8 s (was 5 s) to cover the notch FIR (12.2) |
| D6 | D2 recording condition | **unknown** (12.3); stated in the campaign with its implication, checked at the freeze |
| D7 | D2 source-only CV | 5 participant-stratified folds x 1 repeat (primary); 5 repeats only in the stability analysis |
| D8 | D2 -> D3 target | all 65 D3 participants, with the 52 development and the 13 holdout participants reported separately; the 13 were inspected in earlier work and are **not an untouched holdout** |
| D9 | D3 source-only estimate | reuse of a completed release / A2 development-CV result only with exactly matching store and training identities |
| D10 | Seeds 1-4 | deferred to the separately authorized stability analysis |

### 12.1 D2 filter response (verified; `audit/d2_filter_response.json`, tested)

Zero-phase 4th-order Butterworth 0.5-45 Hz at 1024 Hz (squared magnitude): -48.8 dB at 0.25 Hz, -6.02 dB at 0.5 Hz,
-0.03 dB at 1 Hz, 0.00 dB from 2 to 13 Hz, -0.01 dB at 20 Hz, -0.06 dB at 25 Hz, -0.29 dB at 30 Hz, -2.78 dB at 40 Hz,
-6.02 dB at 45 Hz, -10.6 dB at 50 Hz, -21.4 dB at 60 Hz; impulse response below 1e-4 of its peak beyond 3.73 s (the
5 s post-filter crop covers it). Consequence: the upper gamma band (40-45 Hz) and the lowest delta frequencies are
attenuated in D2; the provider's D3 filter has the same pass band and family but an undocumented order and phase, so
the gamma and low-delta attenuation of the two datasets can differ (an *approximate* property in the compatibility
report).

### 12.2 Line noise (decision D3: measured in Part A, then corrected in D2 only; methodology log entry 63)

**Part A finding (author's run, 2026-10-26; label-blind).** On the analysed 250 Hz data, the 49-51 Hz power relative
to 35-45 Hz has a median of +22.61 dB in D2 and -56.74 dB in D3; the median D2 fraction of variance at 47-53 Hz is
26.7 % (D2-20: 61.8 %). A diagnostic IIR notch applied to the already resampled segments changed the broadband
covariance (the LCMV input) by a median 51.6 % (D2-20: 98.9 %) and flipped 38.5 % of the window-acceptance decisions
of D2-19. All 23 D2 recordings kept at least 129 accepted windows without a notch. Line noise therefore does not touch
the features directly (they end at 45 Hz) but dominates the LCMV covariance and changes the rejection.

**Correction (author, 2026-10-26).** D2 only: zero-phase FIR notch (MNE `notch_filter`, firwin, stop band 49-51 Hz,
1 Hz transitions, `reflect_limited` padding), at the input rate, after the band-pass and the per-channel unit checks,
before the settle crop, resampling, the common average, rejection and source reconstruction. Verified
(`audit/d2_notch_response.json`, tests): -54 to -58 dB at 49-51 Hz; |gain| < 0.01 dB everywhere in 0.5-45 Hz; with
the band-pass -68.5 dB at 50 Hz and -57.9 dB at 100 Hz; FIR half-length 3.30 s, hence the 8 s settle crop;
deterministic (bit-identical reruns) and finite. On synthetic recordings with strong injected 50 Hz the line-band
variance fraction falls from 77 % to 1 %, the 50 Hz peak disappears, and the power of every analysis band changes by
< 0.001 dB. D3 is not notched (its validated stores are unchanged).

**Two notches, not to be confused.** The `diag_*` columns of the inspection are the first audit's *diagnostic* notch:
an IIR notch on the resampled 250 Hz segments, applied to data whose window grid and crops were fixed without a
notch. The *preprocessing* notch acts at 1024 Hz on the continuous band-passed recording before the 8 s crop and the
resampling, so its numbers (`*_before` / `*_after`, `covariance_rel_change_actual_notch`,
`n_windows_accepted_without_notch` vs `n_windows_accepted`, `rejection_flip_fraction_actual_notch`) are the ones that
describe the stores; the two are reported side by side and are not assumed equal. Residual interference after the
notch is checked by the 100 Hz harmonic ratio and a narrowband-peak detector (>= 10 dB over the +-5 Hz median, 1-120
Hz). Note: after the notch, the 49-51 Hz ratio measures the noise floor in the notch's transition bins, not a residual
line; the peak detector is the specific test.

### 12.3 Recording condition (decision D6)

D3: eyes closed (provider README). D2: **unknown**. Searched (2026-10-26): the repository documentation, the AD-LEB
folder on Drive (no protocol or README for the 2019 data), the 2019 participant sheet (columns: ID, age, gender, group,
ACE-R and subscores, MMSE, data quality, comments; no condition field). Remaining evidence: the EDF annotation texts,
listed by the Part A audit (`audit/d2_edf_headers.csv`, `audit.json: d2_edf_annotations`), and the data owner. If D2
was recorded eyes open or mixed, posterior alpha and the theta / alpha balance differ from eyes-closed D3 for reasons
unrelated to diagnosis; a transfer loss or gain may then reflect the condition. Until established, every cross-dataset
result carries this limitation and the condition is reported as unknown.

### 12.4 Age, sex and data-quality confounding in D2 (from the participant sheet; aggregates only)

| D2 | n | Age mean (range) | Female / male |
|---|---|---|---|
| AD | 10 | 80.3 (66-92) | 7 / 3 |
| HC | 13 | 61.9 (55-71) | 4 / 9 |

The pre-specified age-overlap interval is [66, 71] years and holds 1 AD and 4 HC participants: the age-overlap AUC is
**not estimable** (at least 3 per class required) and will be reported as such. The analyses that remain informative
are the age-only baseline (transfer and within-source), the within-diagnosis score-age association and the test of
whether a model's score adds to age (section 9). In addition, the sheet's quality comments (bad channels, movement)
concern 7 of 10 AD and 3 of 13 HC recordings: recording quality is also associated with diagnosis in D2, so
artifact-related signal differences are a second non-disease route to separation. D3's groups have similar ages
(66.4 and 67.9 years). These are descriptive facts about the cohorts; no EEG quantity was combined with a label to
obtain them.

### 12.5 Prior exposure of D2 and the origin of the D2 preprocessing (item 6)

D2 was analysed within-dataset in earlier work (legacy pipeline: 30 channels, 0.1-45 Hz FIR, interpolation, wMNE,
68 ROIs, 20 s windows; graph runs and experiments under `processed/graphs_v*` and `processed/experiments/`). Every D2
choice of this experiment comes from elsewhere: the 19 electrodes from the D3 montage; the 0.5-45 Hz Butterworth band
from the D3 provider's documentation; the 10 s pre-filter crop from the pre-specified, label-blind boundary audit and
rule (entries 40-41: "No decision may use classification performance"); the 5 s post-filter crop from the filter's
impulse response; everything else from the D3 store configuration. No D2 classification outcome informed any of them,
and the legacy choices that were tuned on D1 / D2 data (interpolation, the 0.1 Hz high-pass, 20 s windows) are not
used.

### 12.7 EDF units (Part A finding: mixed `V;uV` headers in D2-19)

The Part A header audit showed D2-19 with physical dimensions `V;uV`. MNE scales every EDF channel by its own header
unit, so mixed headers are correct when each header matches its stored values (tested: the same signal stored in V
in some channels and in uV in others loads to identical volts). A header that does NOT match its values would be off
by a factor 1e6. Hence: `inspect-d2-units` lists every channel of every recording (label, EDF index, selected or not,
unit, physical / digital range, the band-passed robust std MNE returns), identifying exactly which channels of D2-19
carry which unit; and the D2 loader fails fast (`UnitError`, naming recording and channel) when a selected channel's
unit is not V, mV, uV or nV or its band-passed robust std is outside [0.1 uV, 1 mV]. Nothing is ever rescaled to pass.
**Result (author's Part A run, 2026-10-27).** Only D2-19 has mixed units. Its 32 channels: 26 in uV (ranges +-4,246
to +-53,740 uV, one 16-bit step 0.13-1.64 uV) and 6 in V. Two V channels, M1 (`ExG13`, unselected) and T7 = T3
(`ExG14`, selected), have a +-0.14 V range (one step about 4.3 uV) and plausible amplitudes (robust std 184 and 102
uV). Four V channels, all SELECTED, T8 = T4 (`ExG18`), P7 = T5 (`ExG24`), P8 = T6 (`ExG28`) and O1 (`ExG30`), plus the
unselected CP1 (`ExG20`), have a +-1 V range (one step 30.5 uV) and a robust std of 2.8e-27 V: their stored samples are
constant, i.e. they contain no recorded EEG (consistent with the sheet's "8-10 bad channels" note for this
recording). It is not a unit error: MNE reads every channel in volts correctly; there is simply nothing to read. All
other 22 recordings: every selected channel plausible. The unit audit now also reports, per channel, the
quantization step (`lsb_uv`), the fraction of identical stored samples (`constant_fraction`), the number of distinct
values and the plain std, and a `finding` (`flat`, `no_eeg_below_quantization`, `too_small`, `too_large`,
`unsupported_unit`, `ok`).

**Consequence and the decision it needs (author; not taken by code).** The D2 loader stops D2-19 (`UnitError`: no EEG
in T4, T5, T6, O1), so the D2 stores cannot be assembled with the current rules (a store with a failed participant is
never assembled). 4 of the 19 matched electrodes (21 %), at the right temporal and left occipital edge of the montage,
carry nothing. Options:
1. **Exclude D2-19 from both directions by a logged amendment** (a technical, label-blind fact established from the
   stored samples before any graph or classification): D2 becomes 22 participants (9 AD / 13 HC; AD age 79.7 y, range
   66-92; 6 F / 3 M). This is a deviation from rule 4 (the only pre-specified exclusion is fewer than 5 clean
   windows) and must be recorded as such, with the participant's diagnosis (AD) stated.
2. **Interpolate the 4 channels** (spherical spline from the 15 remaining matched electrodes; the legacy D1 / D2
   pipeline interpolated bad channels). Not recommended: 21 % synthetic channels, two adjacent pairs at the montage
   edge where splines extrapolate, interpolated channels are linear mixtures of their neighbours and inflate sensor
   and source connectivity (PLV / |ciPLV|), and D3 is not interpolated (a new asymmetry).
3. Use other D2 channels for these positions: excluded by design (the matched 19-electrode representation).
Recommendation: option 1, decided and recorded before the first build; the D2 cohort, the age summaries and every
report then state 22 / 23 with the reason.

### 12.6 Part A (audit only; authorized 2026-10-26)

Verified from this session through the Drive connector (read only): the layout file on Drive has SHA-256
`da108d02...bd4` (the pinned hash); the RestingState folder holds exactly 23 EDFs (20.5-37.2 MB, about 312-568 s at
32 channels x 1024 Hz if the files are plain EDF); the participant sheet has 23 rows (10 AD, 13 HC). The EDF headers,
the D3 geometry, the D3 store verification, the recording-level inspection and the line-noise audit run in Colab
(notebook 00 Part A); the build, the freeze, the training and any target evaluation stay blocked.

## 13. Compute (relative estimate; no timing has been measured on real data)

Primary units: 432 transfer units (2 directions x 2 modes x (48 GNN + 4 Exp 1 variants) x 2 inverses + 4 Exp 0
variants x 2 modes x 2 directions) + 25 D2 CV folds for each D2 -> D3 unit. Relative to one release development-CV
fold of the same variant (about 41 D3 training participants): a D3 -> D2 fit about 1.25 folds, a D2 -> D3 fit about
0.5, a D2 CV fold about 0.4; mode A units are a small fraction of mode B. With one D2 CV repeat (decision D7) the
D2 CV costs about 5 x 0.4 = 2 fold-equivalents per D2 -> D3 unit; the stability analysis multiplies the transfer
units by 5 (seeds 0-4) and the D2 CV by 5.
