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
   Drive. Outputs are timestamped, never overwritten.
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

Standard first cells for every notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/samimj9/Source-space-EEG-graph-learning-for-Alzheimer-s-disease-using-multi-band-functional-connectivity.git /content/repo
%cd /content/repo
!pip install -q -e .
```

Then load `configs/paths.yaml` (create it in Colab from `paths.example.yaml`
pointing at `/content/drive/MyDrive/AD-LEB`). Cache expensive artifacts
(forward models, source time courses) in Drive under `processed/cache/`, keyed
by dataset and config hash.

## Graph construction (pipeline v3)

- Parameters: `configs/pipeline.yaml`. Code: `src/adgraph/pipeline.py`.
  Colab driver: `notebooks/01_build_graphs.ipynb`.
- Order: `scripts/inspect_raw.py` (header check), then
  `scripts/build_graphs.py --limit 1` (smoke test), then the full
  `scripts/build_graphs.py`.
- **Frozen primary segmentation:** 20 s windows, 50 % overlap (5000 / 2500
  samples at 250 Hz), same for both datasets, cut after all filtering of the
  continuous signal.
- Each run writes `processed/graphs_v3_<UTC stamp>_<config hash>/` with
  `graphs.npz`, `windows.csv`, `subjects.csv` and `manifest.json`. The schema is
  in `src/adgraph/graphs/store.py`, and row *i* of `windows.csv` matches index
  *i* of the arrays.
- PLV is primary and ciPLV is the sensitivity analysis. They are stored and
  evaluated separately and never merged.
- Graph adjacencies use the magnitude |ciPLV| (`ciplv`). The signed,
  antisymmetric ciPLV (`ciplv_signed`) is never used as an adjacency.

## Experiments

- Configs: `configs/experiments/` (Exp 0–6 plus baselines). Runner:
  `scripts/run_experiment.py`. Details: `docs/experiments.md`.
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
- Repeats, folds, epochs and grids are provisional.
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

## Conventions

- Python 3.10+ (Colab default). Core stack: `mne`, `numpy`, `scipy`, `pandas`,
  `scikit-learn`, `torch` (dense GNNs, no `torch_geometric`). `mne` is pinned in
  `requirements.txt` (forward/inverse results must not drift).
- Set and log random seeds (numpy, torch, python) for every run.
- Units: keep EEG in volts inside MNE objects. Dataset 1 EDFs are mislabeled
  (values are µV but header says V): rescale ×1e-6 on load and log it.
- Bands (Hz): theta 4–8, alpha1 8–10, alpha2 10–13, beta 13–30.
- Commit messages: imperative, describe the scientific change (e.g. "Group CV
  folds by subject").

## Working with Claude on this repo

- Ask before changing any methodological parameter (band edges, inverse
  method, rejection threshold, parcellation); these are paper decisions, not
  code-style choices. When one changes, update `docs/methods_decisions.md`.
- Don't download raw EEG into the repo working tree. Inspection of Drive files
  happens in Colab or via the Drive connector, read-only.
- Prefer small, testable functions in `src/` over notebook code.
