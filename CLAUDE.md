# CLAUDE.md

Guidance for Claude (and humans) working in this repository.

## What this project is

Code for a technical research paper: **source-space EEG graph learning for
Alzheimer's disease (AD) vs. healthy control (HC) classification using
multi-band functional connectivity.**

Pipeline (one graph per EEG window, per frequency band):

```
raw EEG (EDF) -> preprocessing -> windowing + artifact rejection
  -> source reconstruction (fsaverage, MNE) -> Desikan-Killiany 68 ROIs
  -> band-limited PLV connectivity (theta, alpha1, alpha2, beta)
  -> multi-band graphs (68 nodes) -> GNN / baselines -> subject-level evaluation
```

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
- Downloaded by `notebooks/d3/D3_01_download_and_build_graphs.ipynb` to `My Drive/OpenNeuro/ds004504`
  (`paths.yaml: dataset3.root`).
- Pipeline `configs/pipeline_d3.yaml`:
  - no second band-pass;
  - bad channels reported only;
  - recordings split at EEGLAB `boundary` events, each segment resampled on its own; no window crosses a
    discontinuity or its margin (`preprocessing/discontinuities.py`, methodology log entry 45);
  - electrode coordinates from each `.set` file's `chanlocs` (frame from the provider's lookup record, no
    template fallback, one operator per geometry; `io/chanlocs.py`, methodology log entry 46). They are a
    shared Colin27 template lookup, not individual measurements;
  - runs in `graphsD3_v4_*`.
- Experiments `configs/experiments/d3/` (`graph_family: D3`, setting `within_D3`); results in `experiments_D3/`.
- Own code pin `D3_CODE_COMMIT.txt` (`colab_setup.py --family D3`); D3 notebooks never change the D1/D2 pin
  `CODE_COMMIT.txt` (`scripts/code_pin.py`).
- Never mix D3 graph runs or results with D1/D2 ones. Cross-dataset D3 ↔ D1/D2 is not implemented yet.

### D3 LCMV beamforming + Xu protocol suite (`docs/lcmv_xu.md`, methodology log entry 50)

- Separate pipeline: `src/adgraph/lcmvxu/`, `configs/lcmv_xu/pipeline.yaml`, `configs/experiments/lx/` (mode B, all
  windows) and `lx/tenfold/` (mode A, 10 samples per participant); notebooks `notebooks/lcmv_xu/`.
- Outputs only under `<processed>/D3_LCMV_XU/` (graphs `graphsLX_v1_*` / `graphsLX10_v1_*`, results
  `experiments_B_all_windows/` / `experiments_A_xu_tenfold/`); own pin `D3_LCMV_XU/CODE_COMMIT.txt` (`--family LX`).
- Per-participant LCMV, 68 DK ROIs, Xu's five bands, 10 s / 1 s windows, frozen 80/20 participant split
  (`xu_dev_cv`, `xu_holdout`). Never mix with D1/D2 or the wMNE D3 runs.

### D3 matched wMNE / Xu comparison (`docs/wmne_xu.md`, methodology log entry 55)

- Same `lcmvxu` pipeline with `configs/wmne_xu/pipeline.yaml` (identical to LCMV-Xu except `source`: historical wMNE);
  `scripts/wx_build.py` (build, validate, `match` against the corrected LCMV stores); notebooks `notebooks/wmne_xu/`.
- Outputs only under `<processed>/D3_WMNE_XU/` (stores `graphsWX_v1_*` / `graphsWX10_v1_*`, experiments
  `experiments_{A_xu_tenfold,B_all_windows,H_historical_graphs}/`); own pin `D3_WMNE_XU/CODE_COMMIT.txt` (`--family WX`).
- Families `WX`, `WX10` (configs `configs/experiments/wx/`, inheriting the LX definitions) and `D3H` (historical
  `graphsD3_v4` graphs, `configs/experiments/wx_hist/`); every run verifies the frozen LCMV split and folds.
- Never write into `D3_LCMV_XU/` from this suite, and never add early stopping to only one of the two suites.

## Repository layout

```
DESIGN.md           Frozen evaluation design (settings, splits, metrics, cross-dataset gate).
configs/            YAML configs: pipeline.yaml, paths template, experiments/. Committed.
src/adgraph/        Importable Python package (all reusable logic lives here).
  io/               loading EDF + participant metadata, subject registry
  preprocessing/    filtering, resampling, referencing, montage, windowing, rejection
  source/           forward model, inverse, parcellation
  features/         PSD band power
  connectivity/     band-limited PLV and ciPLV
  graphs/           store schema (store.py), experiment inputs and DataViews (dataset.py)
  models/           LR/SVM (classical.py), dense GCN / EdgeWeightedGAT / WeightedGraphSAGE, fusion, trainer
  evaluation/       subject-level splits, metrics, statistics
  experiments/      config loading/variants, cross-dataset gate, runner
  visualization/    publication figures from saved outputs only (docs/figures.md)
notebooks/          Thin Colab notebooks: mount Drive, install, call src/. Numbered by stage.
scripts/            CLI entry points mirroring the notebooks (for reproducibility).
tests/              Unit tests on synthetic data only (no real EEG in tests).
docs/               Data inventory, methods decisions log, paper notes.
results/            Small, shareable outputs only (aggregate tables, figures). No per-subject data.
```

Notebooks stay thin: setup cells + calls into `src/adgraph`. Logic that is
written in a notebook gets moved into the package before it is relied on.

## Colab workflow

Notebooks are generated by `scripts/make_notebooks.py`; never edit them by hand.
`notebooks/README.md` holds the index, Colab links and the Drive output layout. Each notebook is run
with *Runtime → Run all* and needs no editing.

- **Order:**
  1. `01_build_graphs` (graphs + QC);
  2. **one experiment notebook at a time** (`notebooks/experiments/Exp0..Exp6`, plus
     `notebooks/additional/` for Exp3b, baselines, the dataset probe and EEG + age);
  3. `03_comparative_figures`.
- **Scope.** An experiment notebook runs only its own config, via
  `scripts/experiment.py status | run | report`. It never starts another experiment.
  The whole-plan runner (`scripts/run_plan.py`, `configs/run_plan.yaml`) is only reachable
  from `notebooks/advanced/`, and only after `RUN_EVERYTHING = True` is set deliberately.
- **Setup cell.** The first cell is identical in every notebook. It mounts Drive, clones
  the branch and runs `scripts/colab_setup.py`, which:
  - locates `AD-LEB`;
  - picks `AD-LEB/processed` when writable, else `My Drive/AD-LEB-processed`;
  - writes `configs/paths.yaml`;
  - pins the code commit in `<output>/CODE_COMMIT.txt`, so all sessions and notebooks use
    one commit (`UPDATE_CODE = True` moves the pin deliberately);
  - installs `requirements.txt`.
- **Shell steps** go through `adgraph.colab.sh`. It shows the output and raises on
  failure, so "Run all" stops. Never use `!cmd` in a notebook.
- **Checks before graphs.** Notebook 01 runs these automatically: the input check (20 + 23
  recordings, 10/10 and 10/13 labels), the test suite (once per commit and environment)
  and `inspect_raw.py --strict`.
- **Protocol.** Experiment notebooks run the config's full protocol (repeats, nested CV,
  grids). They show live progress, then a results table, runtime and memory per unit, and
  single-experiment diagnostic figures. Those diagnostics are not paper figures. Failures end
  the notebook with a visible error after the results are shown. There is no cost-estimation
  stage.
- **Never change** repeats, grids or model configurations to save time without the author's
  approval.
- **Progress bars** (`adgraph.progress`, methodology log entry 35):
  - one tqdm line per stage, never nested: subjects, assembly, experiment folds +
    permutations, figures;
  - library code calls `progress.set(...)` / `progress.advance()`, which are no-ops without
    an active bar;
  - items recovered from checkpoints are counted as done from the start;
  - an ETA appears only from items processed in this session;
  - `adgraph.colab.sh` forwards `\r` redraws so the bar updates in place in Colab.

## Graph construction (pipeline v4, boundary-aware)

- Parameters: `configs/pipeline.yaml`. Code: `src/adgraph/pipeline.py`.
  Colab driver: `notebooks/01_build_graphs.ipynb`.
- Order: `scripts/inspect_raw.py` (header check), then
  `scripts/build_graphs.py --limit 1` (smoke test), then the full
  `scripts/build_graphs.py`.
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

- Configs: `configs/experiments/` (Exp 0–6 plus baselines). One experiment at a
  time: `scripts/experiment.py estimate|run|status|report --config <cfg>` (what the
  notebooks call). Lower-level runner: `scripts/run_experiment.py`. Details:
  `docs/experiments.md` §6.
- Keep graph construction and experiments separate. The graph-run directory
  is the only interface between them.
- Leakage rules are enforced in code:
  - Stateless per-window features go in `graphs/dataset.py`.
  - Anything fitted lives inside an estimator, whose `fit` accepts only a
    `train` `DataView`.
  - Never fit a scaler, selector or threshold outside a model.
- Encoders are `gcn` (standard), `edge_weighted_gat` and `weighted_graphsage`.
  In writing, always call the latter two EdgeWeightedGAT and
  WeightedGraphSAGE, never plain GAT or GraphSAGE.
- The primary SVM is scored by its `decision_function` margin, with no
  calibration and threshold 0. Optional Platt calibration (`grouped_platt`) and
  every inner split are subject-grouped.
- Signed ciPLV's sign is the sign of the non-zero-lag phase relation for the ROI
  order. Never describe it as directed or causal connectivity.
- Decision rationale lives in `docs/methodology_architecture_log.md`.
- EEG + age configs live in `configs/experiments/age/`.
- Repeats, folds, epochs and grids are provisional for classical models. **GNNs use fixed
  hyperparameters with no search or inner CV** (`configs/experiments/gnn_fixed_base.yaml`, protocol
  `gnn_fixed_hparams_v1`, DESIGN.md v1.5, methodology log entry 42): Tier 1 = `*_fixedhp.yaml`
  (10 repeats), Tier 2 = `*_fixedhp_tier2.yaml`. Never add a grid to a fixed-protocol config (it fails
  to load), and never mix nested-CV GNN results (historical `exp2_conn_minimal/`) with fixed ones.
- **Experiment 7 (EARM-GNN, `docs/earm_gnn.md`, methodology log entry 43)** is an exploratory extension designed
  after the Tier 1 results: same protocol, ablations fixed adjacency and mean fusion, own code pin
  (`experiments/exp7_earm_gnn/CODE_COMMIT.txt`). Report it as exploratory; comparisons are descriptive.
- **Do not run experiments or the pipeline on real data unless the user asks.**
  `--synthetic` and `pytest` are always fine.
- Tests: `pytest` (synthetic data only; the fsaverage steps are stubbed).

## Figures

- Figures are generated by `scripts/make_figures.py` with `configs/figures/paper.yaml`
  (inventory) and `configs/figures/style.yaml`. Every figure is documented in
  `docs/figures.md`.
- Figures read only saved outputs. Never hard-code a result value in plotting code or
  config.
- Plot subjects, not windows.
- Keep PLV and |ciPLV| separate.
- No significance marks unless they come from a saved, corrected, subject-level
  statistics file.
- Label display-only thresholds and exploratory projections as such.
- Each figure is saved with its data, metadata and a manifest, and is never overwritten.
  Synthetic figures are watermarked and are never results.
- The pre-result figure decisions are frozen; see the "Pre-result design decisions"
  section of `docs/figures.md`:
  - Exp 2/3 have no representative band.
  - Main figures are Fig. 1–7.
  - The primary models are Exp 0 SVM, Exp 1 SVM and Exp 6 full.
  - PLV is primary and |ciPLV| is the robustness analysis.
  - Changing any of these needs a logged reason.

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
