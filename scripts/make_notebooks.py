"""Generate every Colab notebook from one template (keeps them identical and thin).

    python scripts/make_notebooks.py          # rewrite notebooks/ ; tests check they are up to date

Layout:
    notebooks/01_build_graphs.ipynb                 graphs + QC (run first)
    notebooks/experiments/Exp<k>_*.ipynb            ONE experiment each (Exp0-Exp6)
    notebooks/additional/*.ipynb                    one per secondary analysis (Exp3b, baselines,
                                                    dataset probe, EEG + age)
    notebooks/03_comparative_figures.ipynb          paper figures across experiments
    notebooks/advanced/Run_all_experiments_ADVANCED.ipynb   whole plan; does nothing unless enabled
    notebooks/README.md                             index with Colab links and the Drive output layout

Every notebook: Runtime -> Run all, nothing to edit. Scientific code lives in src/adgraph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adgraph.experiments.config import expand_variants, hparam_grid, load_experiment

REPO = Path(__file__).resolve().parents[1]
OWNER_REPO = "Samimj9/Source-space-EEG-graph-learning-for-Alzheimer-s-disease-using-multi-band-functional-connectivity"
BRANCH = "claude/ecstatic-davinci-a2e8mh"
REPO_URL = f"https://github.com/{OWNER_REPO}.git"
COLAB = f"https://colab.research.google.com/github/{OWNER_REPO}/blob/{BRANCH}/notebooks"

# (folder, notebook name, config, title, one-line description)
EXPERIMENTS = [
    ("experiments", "Exp0_sensor_psd", "exp0_sensor_psd", "Experiment 0 · sensor PSD (LR, SVM)",
     "Electrode band power (30 channels × 4 bands × {log, relative}) with logistic regression and SVM."),
    ("experiments", "Exp1_source_psd", "exp1_source_psd", "Experiment 1 · source PSD (LR, SVM)",
     "Source ROI band power (68 ROIs × 4 bands × {log, relative}) with logistic regression and SVM."),
    ("experiments", "Exp2_conn_minimal", "exp2_conn_minimal", "Experiment 2 · single-band graphs, minimal node features",
     "One band graph (PLV or |ciPLV|) per model, constant node feature; GCN, EdgeWeightedGAT, WeightedGraphSAGE."),
    ("experiments", "Exp3_conn_psd", "exp3_conn_psd", "Experiment 3 · single-band graphs + matching-band PSD",
     "One band graph per model; node features = matching-band ROI PSD [log, relative]; three encoders."),
    ("experiments", "Exp4_band_mean", "exp4_band_mean", "Experiment 4 · band-averaged graph",
     "Mean of the 4 band adjacencies; node features = ROI PSD of all bands; GCN."),
    ("experiments", "Exp5_multiband", "exp5_multiband", "Experiment 5 · multi-band GNN",
     "4 band graphs, matched-band PSD nodes; shared / band-specific encoders × mean / concat / gate / attention fusion."),
    ("experiments", "Exp6_multiband_psd_strength", "exp6_multiband_psd_strength",
     "Experiment 6 · multi-band GNN with PSD + strength (full model and ablations)",
     "4 band graphs, nodes = [log PSD, relative PSD, strength] per band, attention fusion; 8 ablations × PLV/|ciPLV|."),
    ("additional", "Exp3b_conn_psd_allbands", "exp3b_conn_psd_allbands",
     "Experiment 3b (secondary) · single-band graphs + all-band PSD", "Secondary Experiment 3 node features (all bands)."),
    ("additional", "Baseline_age_only", "baseline_age", "Baseline · age only", "Age as the only predictor."),
    ("additional", "Baseline_connectivity_classical", "baseline_conn_classical",
     "Baseline · connectivity features with LR / SVM", "Flattened connectivity matrices with classical models."),
    ("additional", "Probe_dataset_identity", "probe_dataset", "Dataset-identity probe (D1 vs D2)",
     "Predicts the dataset instead of the diagnosis (DESIGN.md §6), pooled setting only."),
] + [
    ("additional", f"Age_{name}", f"age/{cfg}_plus_age", f"EEG + age · {title}", f"{title} with age as a covariate.")
    for name, cfg, title in [
        ("Exp0_sensor_psd", "exp0_sensor_psd", "Experiment 0"), ("Exp1_source_psd", "exp1_source_psd", "Experiment 1"),
        ("Exp2_conn_minimal", "exp2_conn_minimal", "Experiment 2"), ("Exp3_conn_psd", "exp3_conn_psd", "Experiment 3"),
        ("Exp4_band_mean", "exp4_band_mean", "Experiment 4"), ("Exp5_multiband", "exp5_multiband", "Experiment 5"),
        ("Exp6_multiband_psd_strength", "exp6_multiband_psd_strength", "Experiment 6"),
        ("Baseline_connectivity_classical", "baseline_conn_classical", "connectivity baseline")]
]

SETUP = f'''# ===== Setup: nothing to edit. Mounts Drive, gets the code, installs the pinned environment. =====
UPDATE_CODE = False   # leave False. True only if you were asked to switch to newer code
                      # (all notebooks then use the newest commit; unfinished work is recomputed).
BRANCH = '{BRANCH}'
REPO_URL = '{REPO_URL}'

import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')

def _run(cmd, cwd=None):  # shows output and stops "Run all" on failure
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode:
        raise RuntimeError(f'failed: {{" ".join(cmd)}}')

os.chdir('/content')
_run(['rm', '-rf', '/content/repo'])
_run(['git', 'clone', '-q', '-b', BRANCH, REPO_URL, '/content/repo'])
os.chdir('/content/repo')
_run([sys.executable, 'scripts/colab_setup.py'] + (['--update-code'] if UPDATE_CODE else []))
sys.path.insert(0, '/content/repo/src')   # the package, for this notebook's own cells

from adgraph.colab import sh
from adgraph.config import load_paths
PATHS = load_paths('configs/paths.yaml')
OUT = PATHS['processed_dir']
print('outputs on Drive:', OUT)
'''

HOWTO = """
**How to use:** *Runtime → Run all*. Nothing to edit. Allow Google Drive access when asked.

If Colab disconnects or the runtime ends, open the notebook again and *Run all* again: finished
work is kept on Google Drive and skipped, and the computation continues where it stopped.
Everything (checkpoints, results, figures, manifests) is written to Drive, never to the
temporary Colab disk.
"""


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict], gpu: bool = False) -> dict:
    meta = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


def protocol(cfg_path: str) -> tuple[list[dict], str]:
    variants = expand_variants(load_experiment(REPO / "configs" / "experiments" / f"{cfg_path}.yaml"))
    v = variants[0]
    ev = v["evaluation"]
    grid = len(hparam_grid(v))
    gnn = v["input"] == "graph"
    text = (f"- **{len(variants)} variant(s):** " + ", ".join(f"`{x['variant']}`" for x in variants[:24])
            + (" …" if len(variants) > 24 else "") + "\n"
            f"- **Settings:** " + ", ".join(v["settings"]) + " (cross-dataset only if the DESIGN.md §7 gate passes)\n"
            f"- **Protocol (from the config, unchanged):** {ev['n_repeats']} repeats × {ev['k']}-fold subject-level "
            f"outer CV; inner {ev['inner_k']}-fold subject-level CV over {grid} hyperparameter combination(s); "
            f"refit on the full training fold" + (f"; {v['training']['epochs']} epochs" if gnn else "") + ".")
    return variants, text


def experiment_notebook(folder, name, cfg_path, title, desc) -> tuple[dict, dict]:
    variants, proto = protocol(cfg_path)
    experiment = variants[0]["experiment"]
    gnn = variants[0]["input"] == "graph"
    config = f"configs/experiments/{cfg_path}.yaml"
    nb = notebook([
        md(f"# {title}\n\n{desc}\n\n**Runs only this experiment** (`{config}`): every predefined variant and "
           f"setting, with nested cross-validation, evaluation and checkpointing. It never starts another "
           f"experiment.\n\n{proto}\n" + HOWTO + f"""
**Progress** is shown as one updating line: completed outer folds of the whole experiment
(previously completed folds recovered from Drive count as done immediately), current variant,
setting, repeat and fold, inner-CV trial{", training epoch" if gnn else ""}, and the counts of finished /
resumed / failed / skipped units. An ETA is shown only for the current variant and setting, from folds
computed in this session.{" Large GNN experiments take several Colab sessions: just *Run all* again after each disconnect." if gnn else ""}

Results are saved under `<outputs>/experiments/{experiment}/` on Drive: per-fold/repeat checkpoints,
window and subject predictions, metrics per repeat, selected hyperparameters, summaries, manifests,
runtime and peak memory per unit (`_run_status.csv`), and a report folder (results table +
**single-experiment diagnostic figures**). The comparative paper figures across experiments are made
separately (`03_comparative_figures.ipynb`).

Requires the graphs from `01_build_graphs.ipynb`."""),
        code(SETUP),
        md("## 1. This experiment: progress so far (from the checkpoints on Drive)"),
        code(f"""CONFIG = '{config}'   # this notebook's experiment
EXPERIMENT = '{experiment}'
import torch
print('runtime:', ('GPU ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU only')
""" + ("""if not torch.cuda.is_available():
    print('NOTE: GNN experiment on CPU. It works, but a GPU runtime (Runtime > Change runtime type > T4 GPU) '
          'is much faster. Interrupt now if you want to switch; nothing is lost.')
""" if gnn else "") + """sh(f"python scripts/experiment.py status --config {CONFIG}")"""),
        md("## 2. Run (or resume) this experiment\nSaves every completed fold to Drive as it goes. "
           "Units that fail are recorded and the others continue; failures are reported at the end of "
           "section 3 (after the results are shown) and stop *Run all* with an error."),
        code("""RUN_ERROR = None
try:
    sh(f"python scripts/experiment.py run --config {CONFIG}")
except RuntimeError as exc:
    RUN_ERROR = exc
    print('\\nSome units failed or are incompatible; details after the results below.')"""),
        md("## 3. Results of this experiment (saved outputs only)\nSubject-level metrics: mean ± SD over CV "
           "repeats, subject-bootstrap 95 % CI. Runtime and peak memory per variant and setting. "
           "Diagnostic figures of this experiment only; not the comparative paper figures. Nothing is "
           "selected or ranked here."),
        code("""sh(f"python scripts/experiment.py report --config {CONFIG}")
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
exp_dir = Path(OUT) / 'experiments' / EXPERIMENT
report = sorted((exp_dir / '_reports').glob('report_*'))[-1]
print('report folder on Drive:', report)
pd.set_option('display.max_rows', 500, 'display.width', 250)
table = pd.read_csv(report / 'results_summary.csv')
display(table) if len(table) else print('no completed results yet')
status = pd.read_csv(exp_dir / '_run_status.csv') if (exp_dir / '_run_status.csv').exists() else pd.DataFrame()
if len(status):
    print('runtime and peak memory of the last run (seconds, MB):')
    display(status[['variant', 'setting', 'state', 'seconds', 'mem_peak_mb', 'message']])
for png in sorted((report / 'figures').glob('D[12]_*.png')):
    print(png.name)
    display(Image(filename=str(png)))
print('all figures:', sorted(p.name for p in (report / 'figures').glob('*.png')))
bad = status[status['state'].isin(['failed', 'incompatible'])] if len(status) else status
if RUN_ERROR is not None or len(bad):
    raise RuntimeError(f"{len(bad)} unit(s) of {EXPERIMENT} did not complete (see the table above and "
                       f"{exp_dir / '_run_status.csv'}); completed units are saved. Fix the cause, then Run all "
                       "again: only the incomplete units are recomputed.")"""),
    ], gpu=gnn)
    entry = {"folder": folder, "name": name, "title": title, "config": config, "gpu": gnn}
    return nb, entry


def build_graphs_notebook() -> dict:
    return notebook([
        md("# 01 · Build source-space graphs (pipeline v3)\n" + HOWTO + """
Steps (each stops the notebook with a clear message if something is wrong):
1. setup and input check: 20 + 23 recordings, labels 10 AD / 10 HC and 10 AD / 13 HC;
2. synthetic test suite (once per code version, about 10–15 min);
3. EDF header check (all 30 channels used by the pipeline present);
4. smoke test on the first subject of each dataset (downloads fsaverage into the Drive cache once);
5. full graph construction for all 43 recordings (several hours; resumable);
6. quality summary (including peak memory per subject; also saved in `memory_log.csv`).

**Progress** is shown as updating bars: input hashing, the source operator (built once, then loaded
from the Drive cache), then one line for all recordings (e.g. `12/43`) with the current subject,
dataset and processing stage (loading, preprocessing, source reconstruction, connectivity per band,
PSD, saving the checkpoint), the counts of done / resumed / failed / pending recordings and, once two
recordings have been processed in this session, an ETA. Then assembly and saving of the cohort.
Recordings already completed on Drive count as done immediately and are not recomputed.
Runtime and peak memory per recording are saved in `subjects.csv` and `memory_log.csv`.

Then open one experiment notebook in `notebooks/experiments/` (e.g. `Exp0_sensor_psd.ipynb`).
A GPU is not needed for this notebook."""),
        code(SETUP),
        md("## 1. Input check (file names and participant sheets only)"),
        code("from adgraph.colab import check_inputs\ncheck_inputs(PATHS)"),
        md("## 2. Synthetic test suite (no EEG)\nMust pass with zero dependency-related skips. Skipped "
           "automatically if it already passed for this code."),
        code("from adgraph.colab import run_tests_once\nrun_tests_once(OUT)"),
        md("## 3. EDF header check (reads no signal)\nFirst verifies the electrode layout "
           "(`AD-LEB/electrodes_positions_32.loc`) against `channels.recorded`: the EDFs label channels "
           "by number only (D1 `EEG E01`-`E32`, D2 `Misc ExG1`-`ExG32`), and channel *k* is electrode *k* "
           "of that file. Then every recording must have the 30 channels used (M1/M2 dropped)."),
        code("""sh(f"python scripts/inspect_raw.py --paths configs/paths.yaml --out '{OUT}' --strict")"""),
        md("## 4. Smoke test: first subject of each dataset\nWritten to Drive as `graphs_v3_<hash>_limit1/`; "
           "it is never used for analysis."),
        code('sh("python scripts/build_graphs.py --paths configs/paths.yaml --limit 1")'),
        md("## 5. Full graph construction (resumable)\nProgress is saved per subject. If the runtime stops, "
           "*Run all* again."),
        code('sh("python scripts/build_graphs.py --paths configs/paths.yaml --status")\n'
             'sh("python scripts/build_graphs.py --paths configs/paths.yaml")'),
        md("## 6. Quality summary"),
        code("""import json
import pandas as pd
from adgraph.experiments.gate import gate_failures
from adgraph.graphs.store import GraphStore
from adgraph.pipeline import latest_complete_graph_run

run = latest_complete_graph_run(OUT)
m = json.load(open(run / 'manifest.json'))
print('graph run:', run)
print('commit', m['git_commit'], '| config', m['config_hash'], '| graph_run_id', m['graph_run_id'],
      '| operator error', m['source_operator']['verification_rel_error'])
print(json.dumps(m['summary']['per_dataset'], indent=1))
print('subjects not ok:', m['summary']['subjects_not_ok'] or 'none')
fails = gate_failures(GraphStore.load(run, keys=()))
print('cross-dataset gate (DESIGN.md §7):', 'PASSED' if not fails else 'NOT passed: ' + '; '.join(fails))
mem = pd.read_csv(run / 'memory_log.csv') if (run / 'memory_log.csv').exists() else None
if mem is not None:
    print(f"memory: highest peak {mem['mem_peak_mb'].max():.0f} MB per unit "
          f"(machine total {mem['mem_total_mb'].max():.0f} MB; lowest available {mem['mem_available_before_mb'].min():.0f} MB)")
s = pd.read_csv(run / 'subjects.csv')
from IPython.display import display
display(s[[c for c in ['key', 'label', 'orig_sfreq', 'duration_s', 'median_std_uv', 'n_bad_channels',
                       'n_windows_kept', 'n_windows_total', 'channel_naming', 'qc_spatial_rho', 'seconds',
                       'mem_peak_mb', 'status'] if c in s]])
if 'qc_spatial_rho' in s:   # channel-mapping QC: nearby electrodes correlate more (clearly negative expected)
    print('distance vs |correlation| rho per recording: median', round(s['qc_spatial_rho'].median(), 2),
          '| not clearly negative (> -0.1):', s.loc[s['qc_spatial_rho'] > -0.1, 'key'].tolist() or 'none')
errors = s[s['status'].astype(str).str.startswith('error')]
if len(errors):   # below_min_windows is the pre-specified exclusion, not an error
    raise RuntimeError(f"{len(errors)} recording(s) failed: {errors[['key', 'status']].to_dict('records')}. "
                       "Completed recordings are saved; fix the cause and Run all again (only failed ones rerun).")"""),
    ])


def comparative_notebook(configs: list[str]) -> dict:
    return notebook([
        md("# 03 · Comparative figures (across experiments)\n" + HOWTO + """
Builds the **paper figures** (`configs/figures/paper.yaml`, `docs/figures.md`) that compare
experiments, **only from saved outputs**. Run it after the experiments you want to compare have
finished; it trains nothing and starts no experiment. Figures whose inputs do not exist yet are
skipped, and figures drawn with some inputs still missing are listed as INCOMPLETE (with what is
missing). Each run writes a new folder `figures/figures_<time>/` with figures, plotted data and a
manifest. Single-experiment diagnostics are in each experiment's own report folder instead."""),
        code(SETUP),
        md("## 1. Which experiments have results"),
        code(f"""from pathlib import Path
import pandas as pd
from adgraph.experiments.plan import experiments_overview
from adgraph.graphs.store import GraphStore
from adgraph.pipeline import latest_complete_graph_run

GRAPHS = latest_complete_graph_run(OUT)
RESULTS = f"{{OUT}}/experiments"
CONFIGS = {json.dumps(configs)}
print('graph run:', GRAPHS)
pd.set_option('display.width', 250)
experiments_overview(CONFIGS, GraphStore.load(GRAPHS, keys=()), RESULTS)"""),
        md("## 2. Generate the comparative figures"),
        code("""sh(f"python scripts/make_figures.py --config configs/figures/paper.yaml --results '{RESULTS}' "
   f"--graphs '{GRAPHS}' --out '{OUT}/figures'")
from IPython.display import Image, display
latest = sorted(Path(f"{OUT}/figures").glob('figures_*'))[-1]
print('figures in', latest)
for png in sorted(latest.glob('Fig*.png')):
    print(png.name)
    display(Image(filename=str(png)))
print('supplementary:', sorted(p.name for p in latest.glob('S*.png')))"""),
    ])


def advanced_notebook() -> dict:
    return notebook([
        md("# ADVANCED · run every experiment of `configs/run_plan.yaml`\n" + HOWTO + """
**Optional. Not part of the normal workflow.** The normal workflow is one experiment notebook at a
time (`notebooks/experiments/`), checking each result before starting the next.

This notebook can run *all* experiments in priority order (≈ hundreds of GPU-hours with the full
protocol). With the default `RUN_EVERYTHING = False`, *Run all* only shows the progress of the whole
plan and **starts nothing**. To launch the full plan you must deliberately set
`RUN_EVERYTHING = True` in section 2."""),
        code(SETUP),
        md("## 1. Progress of the whole plan (computes nothing)"),
        code('sh("python scripts/run_plan.py --paths configs/paths.yaml --status")'),
        md("## 2. Launch everything (disabled by default)"),
        code("""RUN_EVERYTHING = False   # deliberately set True to run ALL experiments of configs/run_plan.yaml
if RUN_EVERYTHING:
    sh("python scripts/run_plan.py --paths configs/paths.yaml")
else:
    print('Not started (RUN_EVERYTHING = False). Use the per-experiment notebooks instead.')"""),
    ], gpu=True)


DRIVE_LAYOUT = """\
<outputs>/                          AD-LEB/processed (or My Drive/AD-LEB-processed if that is read-only)
├── CODE_COMMIT.txt                 code commit used by every session and notebook
├── header_report_<time>.csv        EDF header checks (notebook 01, step 3)
├── cache/
│   ├── subjects/fsaverage/         fsaverage template (downloaded once)
│   ├── operators/operator_<id>.npz + .done.json   verified 68x30 sensor->ROI operator
│   ├── input_sha256.json           content hashes of EDFs and sheets
│   └── tests_passed/<commit>_<env>.txt            test-suite pass record
├── graphs_v3_<config hash>_limit1/ smoke test (never used for analysis)
├── graphs_v3_<config hash>/        graph run used by the experiments
│   ├── run_identity.json  status.csv  memory_log.csv
│   ├── subjects/<D1-k|D2-k>/       per-subject checkpoint: arrays.npz windows.csv subject.json subject.done.json
│   ├── graphs.npz  windows.csv  subjects.csv  manifest.json  assembly.done.json
│   └── _superseded/<time>/         replaced checkpoints (never deleted)
├── experiments/
│   └── <experiment>/               one folder per experiment notebook
│       ├── <variant>/<setting>/    predictions_windows.csv predictions_subjects.csv metrics_repeats.csv
│       │                           folds.csv hparams.csv summary.json manifest.json [embeddings.npz]
│       │                           final.done.json _identity.json status.csv units/ (fold, inner-trial,
│       │                           permutation checkpoints)
│       ├── _run_status.csv         state, runtime (s) and peak memory (MB) per variant and setting
│       ├── _estimates/             (only if `experiment.py estimate` is run manually)
│       └── _reports/report_<time>/ results_summary.csv progress.csv README.txt figures/ (diagnostics D1-D4)
└── figures/figures_<time>/         comparative paper figures (notebook 03): figure files, data,
                                    metadata, figure_manifest.json
"""


def readme(entries: list[dict]) -> str:
    rows = "\n".join(
        f"| [{e['folder']}/{e['name']}.ipynb]({COLAB}/{e['folder']}/{e['name']}.ipynb) | `{e['config']}` | "
        f"{'GPU (T4)' if e['gpu'] else 'CPU'} |" for e in entries)
    return f"""# Colab notebooks

Open a notebook in Colab (links below), then *Runtime → Run all*. Nothing needs to be edited;
everything is saved to Google Drive, and *Run all* again resumes after a disconnect (finished work
is shown as done and never recomputed). Progress is shown as updating progress bars. These files
are generated by `scripts/make_notebooks.py` (do not edit them by hand).

1. [01_build_graphs.ipynb]({COLAB}/01_build_graphs.ipynb): graphs + QC (first, once).
2. One experiment at a time ([notebooks/experiments](experiments)): each runs **only** its own
   experiment with the full protocol, saves its results and shows them. Review, then choose the next.
3. [03_comparative_figures.ipynb]({COLAB}/03_comparative_figures.ipynb): paper figures comparing
   experiments (after the experiments you need have finished).

Optional: secondary analyses in [notebooks/additional](additional), and
[advanced/Run_all_experiments_ADVANCED.ipynb]({COLAB}/advanced/Run_all_experiments_ADVANCED.ipynb)
(whole plan; starts nothing unless explicitly enabled).

## Experiment notebooks

| Notebook | Config | Runtime type |
|---|---|---|
{{rows}}

## Google Drive output structure

```
{{DRIVE_LAYOUT}}```
""".replace("{rows}", rows).replace("{DRIVE_LAYOUT}", DRIVE_LAYOUT)


def main(nbdir: Path = REPO / "notebooks") -> dict[Path, dict]:
    nbdir = Path(nbdir)
    outputs = {nbdir / "01_build_graphs.ipynb": build_graphs_notebook(),
               nbdir / "advanced" / "Run_all_experiments_ADVANCED.ipynb": advanced_notebook()}
    entries = []
    for folder, name, cfg, title, desc in EXPERIMENTS:
        nb, entry = experiment_notebook(folder, name, cfg, title, desc)
        outputs[nbdir / folder / f"{name}.ipynb"] = nb
        entries.append(entry)
    outputs[nbdir / "03_comparative_figures.ipynb"] = comparative_notebook([e["config"] for e in entries])
    for path, nb in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (nbdir / "README.md").write_text(readme(entries))
    for old in ("02_experiments.ipynb", "03_figures.ipynb"):           # replaced by the files above
        (nbdir / old).unlink(missing_ok=True)
    print(f"{len(outputs)} notebooks + README written")
    return outputs


if __name__ == "__main__":
    main()
