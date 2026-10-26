# CLAUDE.md

Guidance for Claude (and humans) working in this repository.

## What this project is

Code for a technical research paper: **source-space EEG graph learning for Alzheimer's disease (AD) vs. healthy control (HC)
classification using multi-band functional connectivity.**

The current code is the **final framework, release final-1.0** (`docs/final/framework.md`, `FINAL_RUN_GUIDE.md`,
DESIGN.md v2.0, methodology log entry 58) on OpenNeuro ds004504 (65 participants, 36 AD / 29 HC):

```
validated store (LCMV | wMNE  x  window mode B | A  x  atlas) --verify + lock--> participant QC gate
  --> Exp 0-8 (one config each; graph construction for GNNs) --> participant-level evaluation (52 / 13 frozen split)
```

Earlier work (D1 / D2 with wMNE, the first D3 pipeline, the LX / WX suites with their ES200 and R1 variants) is archived
in `legacy/` (and on branch `claude/ecstatic-davinci-a2e8mh`); its D1 / D2 graph builder stays importable and tested as
`adgraph.legacy`.

The pipeline runs in **Google Colab with Google Drive mounted**. This repo holds
code, configs, docs and small summary tables only.

## Hard rules

1. **Never commit data.** No raw EEG (`.edf`, `.gdf`, `.set`, `.fif`, `.bdf`),
   no participant spreadsheets (`.xlsx`, `.csv` with participant rows), no
   processed graphs (`.pkl`, `.npy`, `.npz`, `.pt`, `.h5`), no model
   checkpoints. Participant metadata (age, sex, MMSE, ACE-R) is
   health data; treat it as sensitive even when de-identified. `.gitignore`
   enforces most of this; check `git status` before every commit anyway.
2. **No hard-coded Drive paths in library code.** All paths come from
   `configs/paths.yaml` (copied from `configs/paths.example.yaml`, not
   committed) or function arguments.
3. **Split by subject, never by window.** Windows from the same recording are
   highly correlated. Every train/val/test split, CV fold, scaler fit and
   feature selection must be grouped by `(dataset, subject_id)`. Report
   subject-level metrics (aggregate window predictions per subject) as the
   primary result; window-level metrics are secondary.
4. **Use all subjects and recordings as provided.** Don't exclude subjects
   or run duplicate-file analyses. The only exclusion is the pre-specified
   pipeline rule (fewer than 5 clean windows). Evaluation follows
   `DESIGN.md`, which is frozen. Change it only with a logged reason.
5. **Every deviation from the reference papers is recorded** in
   `docs/methods_decisions.md` with the reason. The existing run manifest
   already lists several; keep that log the single source of truth.
6. **Every pipeline run writes a manifest** (JSON: parameters, software
   versions, git commit, input file list, timestamp) next to its outputs in
   Drive.
   - Outputs are never overwritten. Completed checkpoint units are
     immutable.
   - Intentional recomputation (`--force`) goes to a fresh directory, or
     moves old outputs to `_superseded/`.
   - Output directories are keyed by config hash (needed for resuming), not
     by timestamp; timestamps live in manifests and markers.
7. Don't present a number in the paper that can't be regenerated from a
   committed config + a commit hash.

## Data location (Google Drive)

The data lives in the shared Drive folder **`AD-LEB`** (owner: a collaborator;
shared with the project author), accessed in Colab through a shortcut at
`/content/drive/MyDrive/AD-LEB`.

```
AD-LEB/
├── EEGs/
│   ├── Dataset 1 (2017)/          sub1.edf ... sub20.edf          (20 files, resting state)
│   └── Dataset 2 (2019)/
│       ├── RestingState/          <id>_<date>_1.edf               (23 files)  <- used
│       ├── PictureNaming/         <id>_<date>_2.gdf               (task, not used)
│       └── ButtonPress/           <id>_<date>_3.gdf               (task, not used)
├── Info participants/
│   ├── Info Participants_2017.xlsx   ID, Age, Gender, Group, MMSE, Comments
│   └── Info Participants_2019.xlsx   Id, Age, Gender, Group, ACE-R (+subscores), MMSE, Comments
├── electrodes_positions_32.loc       32-ch EEGLAB polar layout (incl. M1/M2)
└── processed/                        pipeline outputs (graphs .pkl, summaries .csv, manifests .json)
```

Full inventory, cohort counts and known data issues: `docs/data_inventory.md`.

Subject identity: Dataset 2 filenames are `<subject>_<DDMYYYY>_<run>.edf`; the
subject key is the leading integer. Dataset 1 is `sub<N>.edf`. Subject IDs
overlap between datasets, so the unique key is always `(dataset, subject_id)`,
e.g. `D1-3`, `D2-3`.

### Dataset 3 (OpenNeuro ds004504; `docs/dataset_d3.md`, methodology log entry 44)

- 36 AD + 29 HC, provider-preprocessed derivatives; FTD excluded by design (never downloaded).
- Downloaded (once, already done) by `scripts/d3_data.py` to `My Drive/OpenNeuro/ds004504`
  (`paths.yaml: dataset3.root`); the source-graph stores are built from these derivatives.
- Shared preprocessing rules (both the stores and the legacy `configs/pipeline_d3.yaml` builder):
  - no second band-pass; bad channels reported only;
  - recordings split at EEGLAB `boundary` events, each segment resampled on its own; no window crosses a
    discontinuity or its margin (`preprocessing/discontinuities.py`, methodology log entry 45);
  - electrode coordinates from each `.set` file's `chanlocs` (frame from the provider's lookup record, no
    template fallback, one operator per geometry; `io/chanlocs.py`, methodology log entry 46). They are a
    shared Colin27 template lookup, not individual measurements.
- The historical wMNE graphs `graphsD3_v4_*`, their experiments (`experiments_D3/`) and the pin
  `D3_CODE_COMMIT.txt` belong to the earlier work. Never mix them, or D1 / D2 results, with the final framework's.

### The final framework (release final-1.0; `docs/final/`, methodology log entry 58)

- **Stores (read only, locked):** LCMV `D3_LCMV_XU/graphsLX{,10}_v1_811690218bfa_corr1`, wMNE
  `D3_WMNE_XU/graphsWX{,10}_v1_c32bc6c67f0e` (mode B / A; 68 DK ROIs, 5 Xu bands). One store configuration
  `configs/source_graphs.yaml`; `sourcegraphs.config.resolve(inverse, atlas)` reproduces both stores' configurations
  exactly (tested). Never rebuild or modify them. Coarser atlases get new stores (`scripts/build_stores.py`), built by
  notebook 02 from the original recordings into `D3_LCMV_XU/graphsLX{,10}-<atlas>_v1_<hash>` and
  `D3_WMNE_XU/graphsWX{,10}-<atlas>_v1_<hash>`, validated, matched (wMNE to LCMV; each window by window to its dk68
  store, `atlas_match.json`) and QC'd before use. A build needs `experiments.decisions.atlas_authorization`: lobes12
  is rule-eligible (exact BEM); r18 is exploratory and needs the author's recorded approval (amendment A1,
  `configs/final/amendments.yaml`, `final.py approve-exploratory`). Never alter the rule, its thresholds or a verdict.
- **One version system:** `adgraph.release`, outputs under `<processed>/FINAL_v1/` (except the coarser-atlas stores,
  next to the 68-ROI stores), one code pin
  `FINAL_v1/CODE_COMMIT.txt`. The earlier pins (`CODE_COMMIT.txt`, `D3_CODE_COMMIT.txt`, LX / WX / ES200 / R1 pins) are
  historical: never moved, never written by the final framework.
- **Protocol:** frozen 52 / 13 participant split (`xu_dev_cv` 5 folds, `xu_holdout`); GNNs exactly 60 epochs
  (`gnn_fixed60_v1`), no early stopping, no search; classical models participant-level inner CV for C. The holdout was
  inspected in earlier work: descriptive only, never used for any selection. Everything runs development CV only;
  the holdout is evaluated once, in notebook 99, for the units of the frozen final analysis
  (`configs/final/holdout.yaml`, `experiments.decisions`); the code refuses it for anything else.
- **Experiments:** `configs/final/experiments/exp0-8*.yaml`; inverse, window mode, atlas and graph construction are
  run-time context (`experiments.config.with_context`). LCMV and wMNE variants must stay identical apart from the
  inverse (tested). A shared-encoder multiband model must carry `model.band_identity: onehot` (validate refuses it
  otherwise, unless declared `band_invariant_baseline`; methodology log entry 57).
- **Amendments (own namespace and pin):** amendment A2 runs Exp 8 on lobes12 as `exp8a_xu_dual_gcn_surface`
  (cortical-surface anatomical layer) in `FINAL_v1/amendments/A2_exp8_surface/` with its own `CODE_COMMIT.txt`
  (`experiments.amendment`, `scripts/colab_setup.py --amendment A2`, `final.py ... --amendment A2`; methodology log
  entry 60). Never move the release pin for an amendment, never let an amendment write in the release namespace, and
  never change `src/adgraph/qc/` (the release's QC reports are reused across commits only while that code is unchanged).
- **Cross-dataset D3 <-> D2 (branch `claude/cross-dataset-d3-d2-19ch`; `docs/cross_dataset/`, methodology log entry 61):**
  `adgraph.crossdataset`, `scripts/cross_dataset.py`, namespace `CROSS_D3_D2_19CH_v1/` with its own pin. 19 matched
  electrodes by EDF number (`configs/cross_dataset/channel_map_19.yaml`, hash-pinned); source-only fits, blinded target,
  frozen campaign; D2 stores and real-data training only after the author's recorded authorization. Never write
  outside that namespace.
- **QC before classification:** `adgraph.qc`; category A blocks the experiments until the author records a decision in
  `FINAL_v1/qc/decisions.yaml`; nobody is excluded or reweighted by code (docs/final/qc.md).
- **Graph construction / atlas:** `configs/final/graphs.yaml`, `configs/final/selection.yaml`; the atlas is recorded by
  the author (`final.py choose-atlas`) after the source-resolution evidence and the development-only dk68 vs lobes12
  vs r18 comparison (notebook 04, `configs/final/roi_comparison.yaml`; 04 itself selects nothing), never by
  classification results alone; the graph
  construction is then chosen on that atlas by the pre-specified development-CV rule only (notebook 03).

## Repository layout

```
DESIGN.md           Evaluation design (v2.0 = the final framework; v1.x = the earlier results).
FINAL_RUN_GUIDE.md  Notebook order, configurations, ROI recommendation, output directories.
configs/            final/ (protocol, classical, gnn, graphs, selection, roi_comparison, holdout, run_plan,
                    experiments/exp0-8),
                    source_graphs.yaml (stores), pipeline*.yaml (legacy builder), paths template.
src/adgraph/        The package.
  sourcegraphs/     LCMV / wMNE store building, validation, matching, protocol checks
  stores.py, release.py   store verification + lock; the release (version, pin, output root)
  io/ preprocessing/ source/ features/ connectivity/   shared signal processing (source/atlas.py, resolution.py)
  graphs/           store schema, datasets, graph construction + diagnostics
  qc/               participant QC (categories A / B / C), fold-level QC
  models/           LR / SVM, dense GCN multiband GNN, EARM-GNN, Xu dual GCN, trainer
  evaluation/       participant splits, metrics (DeLong, bootstrap, Wilson), reporting
  experiments/      configs, planner, runner, report, selection, roi_comparison, decisions (atlas / graph / frozen holdout)
  legacy/           frozen D1 / D2 / graphsD3_v4 builder (importable, tested)
notebooks/          Generated Colab notebooks of the final framework (00-03, Exp0-Exp8, 99).
scripts/            final.py (the CLI), build_stores.py, colab_setup.py, make_notebooks.py, legacy/ (builder tools).
tests/              Unit / synthetic tests (no real EEG).
docs/               final/ (framework, graph construction, QC, parcellation, inventory), logs, earlier docs.
legacy/             Archived earlier code, configs, notebooks and tests (not importable; docs/final/inventory.md).
results/            Small, shareable outputs only. No per-subject data.
```

Notebooks stay thin: setup cell + calls of `scripts/final.py` / `scripts/build_stores.py`. Logic written in a notebook
moves into the package before it is relied on.

## Colab workflow

Notebooks are generated by `scripts/make_notebooks.py`; never edit them by hand. Order: `configs/final/run_plan.yaml`
(`00_setup_and_stores`, `01_participant_qc`, `02_parcellation`, `04_roi_68_vs_18`, `03_graph_construction`,
`Exp0`-`Exp8`, `99_results_report`; 04 before 03 on purpose), **one notebook at a time**, each *Runtime → Run all*.

- **Scope.** An experiment notebook runs only its own experiment (`scripts/final.py status | run | report --exp
  expN`) over the contexts of its `CONTEXTS` cell (the only cell meant to be edited). It never starts another
  experiment.
- **Setup cell.** Identical in every notebook: mounts Drive, clones `claude/unified-final-framework`, runs
  `scripts/colab_setup.py` (locates `AD-LEB`, chooses the output folder, writes `configs/paths.yaml`, pins the code in
  `<output>/FINAL_v1/CODE_COMMIT.txt` (`UPDATE_CODE = True` moves it deliberately), installs `requirements.txt`).
- **Shell steps** go through `adgraph.colab.sh` (shows output, raises on failure). Never use `!cmd` in a notebook.
- **Protocol.** Notebooks run the config's full protocol. Never change repeats, grids, epochs or model
  configurations to save time without the author's approval.
- **Progress bars** (`adgraph.progress`, methodology log entry 35): one tqdm line per stage; library code calls
  `progress.set(...)` / `progress.advance()`; items recovered from checkpoints count as done.

## Legacy graph builder (D1 / D2, pipeline v4, boundary-aware)

Not used by the final framework (whose stores come from `adgraph.sourcegraphs`); kept so earlier results stay
reproducible. It computes differential entropy with the definition of the source-graph stores (methodology log
entry 58).

- Parameters: `configs/pipeline.yaml`. Code: `src/adgraph/legacy/pipeline.py`. Archived Colab driver:
  `legacy/notebooks/01_build_graphs.ipynb`.
- Order: `scripts/legacy/inspect_raw.py` (header check), then
  `scripts/legacy/build_graphs.py --limit 1` (smoke test), then the full
  `scripts/legacy/build_graphs.py`.
- **Frozen primary segmentation:** 20 s windows, 50 % overlap (5000 / 2500
  samples at 250 Hz), same for both datasets, cut after all filtering of the
  continuous signal.
- **Boundary handling (v4; methodology log entry 41, `docs/boundary_processing.md`):**
  - the 10 s edges are cropped before any filtering;
  - padding is explicit (`reflect_limited`);
  - the analytic signal is a finite complex FIR (`band_analytic_fir`), never an FFT Hilbert transform
    of the whole recording;
  - windows keep the computed margin (4537 samples) from the cropped ends.
  
  Do not reintroduce whole-recording filtering or FFT Hilbert transforms in the v4 path. The v3 path is
  kept only for reproducing earlier results.
- Each run writes `processed/graphs_v4_<config hash>/` with
  `subjects/<key>/` (per-subject checkpoints), `status.csv`, and after
  assembly `graphs.npz`, `windows.csv`, `subjects.csv` and `manifest.json`
  (with `graph_run_id`). The schema is
  in `src/adgraph/graphs/store.py`, and row *i* of `windows.csv` matches index
  *i* of the arrays.
- PLV is primary and ciPLV is the sensitivity analysis. They are stored and
  evaluated separately and never merged.
- Graph adjacencies use the magnitude |ciPLV| (`ciplv`). The signed,
  antisymmetric ciPLV (`ciplv_signed`) is never used as an adjacency.

## Experiments

- Configs: `configs/final/experiments/` (Exp 0-8), shared `configs/final/{protocol,classical,gnn}.yaml`. One experiment
  at a time: `scripts/final.py status | run | report --exp expN --inverse lcmv|wmne --mode B|A [--atlas selected]
  [--graph selected]` (development CV only); the holdout only via `final.py freeze` + `final.py holdout` (notebook 99).
- Keep store building and experiments separate. The verified store is the only interface between them.
- Leakage rules are enforced in code:
  - Stateless per-window features go in `graphs/dataset.py`; graph constructions are stateless per window except
    `quantile`.
  - Anything fitted lives inside an estimator, whose `fit` accepts only a `train` `DataView`.
  - Never fit a scaler, selector, threshold or QC reference outside a model or the training participants.
- The only GNN encoder is the standard weighted-adjacency `gcn` (EdgeWeightedGAT / WeightedGraphSAGE are archived; in
  writing always use those names, never plain GAT or GraphSAGE).
- The primary SVM is scored by its `decision_function` margin, with no calibration and threshold 0.
- Signed ciPLV's sign is the sign of the non-zero-lag phase relation for the ROI order. Never describe it as directed
  or causal connectivity; it is never an adjacency.
- Experiment 7 (EARM-GNN) is exploratory (designed after earlier results); comparisons are descriptive.
- Decision rationale lives in `docs/methodology_architecture_log.md`.
- **Do not run experiments or the pipeline on real data unless the user asks.**
  `scripts/final.py smoke`, `--synthetic`-style runs and `pytest` are always fine.
- Tests: `pytest` (synthetic data only; the fsaverage steps are stubbed or approximated).

## Figures

The D1 / D2 paper-figure tooling (`visualization`, `configs/figures/`, `scripts/make_figures.py`) is archived in
`legacy/`. The final framework writes per-experiment diagnostic figures from saved outputs only
(`experiments.report`); they are diagnostics, not paper figures. Rules for any figure: read only saved outputs,
never hard-code a result value, plot participants not windows, keep PLV and |ciPLV| separate, no significance marks
unless they come from a saved, corrected, participant-level statistics file, label exploratory projections, never
overwrite, watermark synthetic figures. The pre-result figure decisions of `docs/figures.md` apply to the earlier
D1 / D2 figures.

## Checkpointing and resume (see methodology log entries 31–32)

- **Mechanism.** Every unit of work is checkpointed with atomic writes and a
  validated completion marker: identity + SHA-256 of payload files
  (`src/adgraph/checkpoint.py`). Payload is written first, marker last, so an
  interrupted unit is never treated as complete.
- **Graphs:**
  - Each subject is saved as soon as it finishes. On restart, completed
    subjects are skipped.
  - Missing, interrupted, corrupted or incompatible subjects are recomputed,
    and failed subjects are retried.
  - The sensor→ROI operator is cached under `<cache_dir>/operators` by its
    source-model hash, so fsaverage and the forward/inverse model are only
    built when needed.
  - The cohort is reassembled only when the subject units change.
- **Experiments:**
  - Each outer fold, inner-CV trial and permutation is saved on completion.
  - Unit identity = training-config hash + `graph_run_id` + setting + split
    id + repeat + fold + seed (+ trial params / permutation index) + code
    version.
  - Raising `n_repeats` reuses existing repeats.
  - A different config or graph run raises `IncompatibleCheckpoint`; results
    are never silently reused.
- **CLI:**
  - `--resume` is the default and continues the same command.
  - `--status` shows completed / partial / pending / failed without running
    anything.
  - `--force` recomputes intentionally: a fresh directory, or old outputs
    moved to `_superseded/`.
  - `--no-resume` refuses an existing directory.
- **Code fingerprint (automatic).** Every identity carries `code_identity`:
  - the git commit;
  - a SHA-256 of the source files that determine the computation (this also
    catches uncommitted edits);
  - the versions of the numerical libraries.

  Any new commit therefore invalidates graph units, which are recomputed
  automatically, and experiment directories, which raise
  `IncompatibleCheckpoint` until `--force`. To resume an interrupted run,
  pin the run's commit (`COMMIT` in the notebooks). `CODE_VERSIONS` remains
  as a documentation label only; forgetting to bump it cannot cause reuse.
- **Input fingerprint.** EDF recordings are identified by their content
  SHA-256, not by path or mtime, so moving the Drive mount does not
  invalidate them. Each subject's identity also hashes its metadata row.
  Participant sheets and the layout file are hashed into the assembly
  identity and the manifest. Hashes are cached in
  `<cache_dir>/input_sha256.json`, keyed by path + size + mtime;
  `--rehash-inputs` ignores the cache.
- **Persistent storage.** Real runs refuse any output, checkpoint or cache
  location that is not on the mounted Google Drive, including `/content`,
  other VM paths in Colab, and `/content/drive` when Drive is not mounted.
  The override is `--allow-ephemeral-storage`. Synthetic runs are exempt.
- **Environment.** Real-run CLIs verify the exact pins in
  `requirements.txt` (`mne`, `edfio`). In `pytest`, a skip caused by a
  missing dependency fails the session (`tests/dependency_guard.py`).
- **Figures** only read final, validated experiment outputs and never train.
  Superseded and unfinished results are ignored.
- **Memory (methodology log entry 34).** Graph construction holds one subject at a
  time and releases it after its checkpoint is saved. Recordings only ever go through
  the 68×30 operator (no full source time series). Band-pass + Hilbert run in ROI
  chunks, one band at a time. Assembly preallocates the cohort once. Peak RAM per
  subject is in `subjects.csv` (`mem_*`) and in the run's `memory_log.csv`. Do not
  trade memory for changed numerics (e.g. complex64, windowed filtering).

## Conventions

- Python 3.10+ (Colab default). Core stack: `mne`, `numpy`, `scipy`, `pandas`,
  `scikit-learn`, `torch` (dense GNNs, no `torch_geometric`). `mne` is pinned in
  `requirements.txt` (forward/inverse results must not drift).
- Set and log random seeds (numpy, torch, python) for every run.
- Units: keep EEG in volts inside MNE objects. Dataset 1 EDFs are mislabeled
  (values are µV but header says V): rescale ×1e-6 on load and log it.
- Amplitude checks (methodology log entries 38-39): a unit safeguard (`units.gross_robust_std_v`,
  robust std of the whole recording) and the plausibility check (`units.plausible_std_v`, std of
  the analysed interval after interpolation and average reference). Bad channels are detected on
  the analysed interval. On failure, run `scripts/diagnose_amplitude.py` (read-only) and decide
  from its evidence; never widen a range or rescale to get past it.
- Before approving a full graph build, inspect the smoke test (`scripts/inspect_graph_subjects.py`,
  Notebook 01 step 4b).
- A graph run with any failed subject, or with no graphs, is not assembled
  (`GraphRunIncomplete`); experiments refuse empty, partial or smoke-test runs.
- Channels: the EDFs label channels by number only (D1 `EEG E01`–`E32` at 1000 Hz, D2
  `Misc ExG1`–`ExG32` at 1024 Hz). Channel *k* is electrode *k* of
  `AD-LEB/electrodes_positions_32.loc`, which is verified against `channels.recorded` at run
  time (`adgraph.io.layout`, methodology log entry 37). Never map them by any other rule.
- Bands (Hz): theta 4–8, alpha1 8–10, alpha2 10–13, beta 13–30. Connectivity band-pass:
  MNE FIR zero-phase firwin with 1.0 Hz transitions on every edge
  (`connectivity.transition_bandwidth_hz`; `docs/band_filters.md`, methodology log entry 36).
- Commit messages: imperative, describe the scientific change (e.g. "Group CV
  folds by subject").

## Working with Claude on this repo

- Ask before changing any methodological parameter (band edges, inverse
  method, rejection threshold, parcellation); these are paper decisions, not
  code-style choices. When one changes, update `docs/methods_decisions.md`.
- Don't download raw EEG into the repo working tree. Inspection of Drive files
  happens in Colab or via the Drive connector, read-only.
- Prefer small, testable functions in `src/` over notebook code.
