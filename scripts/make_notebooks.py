"""Generate every Colab notebook from one template (keeps them identical and thin).

    python scripts/make_notebooks.py          # rewrite notebooks/ ; tests check they are up to date

Layout:
    notebooks/01_build_graphs.ipynb                 graphs + QC (run first)
    notebooks/experiments/Exp<k>_*.ipynb            ONE experiment each (Exp0-Exp7)
    notebooks/additional/*.ipynb                    one per secondary analysis (Exp3b, baselines,
                                                    dataset probe, EEG + age)
    notebooks/03_comparative_figures.ipynb          paper figures across experiments
    notebooks/advanced/Run_all_experiments_ADVANCED.ipynb   whole plan; does nothing unless enabled
    notebooks/advanced/Restore_D1D2_code_pin.ipynb          restores the D1/D2 code pin (maintenance)
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
    ("experiments", "Exp2_conn_minimal", "exp2_conn_minimal_fixedhp",
     "Experiment 2 · single-band graphs, minimal node features (fixed hyperparameters)",
     "One band graph (PLV or |ciPLV|) per model, constant node feature; GCN (Tier 1)."),
    ("experiments", "Exp3_conn_psd", "exp3_conn_psd_fixedhp",
     "Experiment 3 · single-band graphs + matching-band PSD (fixed hyperparameters)",
     "One band graph per model; node features = matching-band ROI PSD [log, relative]; GCN (Tier 1)."),
    ("experiments", "Exp4_band_mean", "exp4_band_mean_fixedhp", "Experiment 4 · band-averaged graph (fixed hyperparameters)",
     "Mean of the 4 band adjacencies; node features = ROI PSD of all bands; GCN."),
    ("experiments", "Exp5_multiband", "exp5_multiband_fixedhp", "Experiment 5 · multi-band GNN (fixed hyperparameters)",
     "4 band graphs, matched-band PSD nodes; shared GCN encoder, mean / attention fusion (Tier 1)."),
    ("experiments", "Exp6_multiband_psd_strength", "exp6_multiband_psd_strength_fixedhp",
     "Experiment 6 · multi-band GNN with PSD + strength (fixed hyperparameters)",
     "4 band graphs, nodes = [log PSD, relative PSD, strength] per band, attention fusion; full model "
     "(PLV, |ciPLV|) + 7 PLV ablations (Tier 1)."),
    ("additional", "Tier2_Exp2_other_encoders", "exp2_conn_minimal_fixedhp_tier2",
     "Tier 2 (optional) · Experiment 2 with EdgeWeightedGAT and WeightedGraphSAGE",
     "Secondary: the Exp 2 single-band graphs with the two other encoders, same fixed hyperparameters."),
    ("additional", "Tier2_Exp3_other_encoders", "exp3_conn_psd_fixedhp_tier2",
     "Tier 2 (optional) · Experiment 3 with EdgeWeightedGAT and WeightedGraphSAGE",
     "Secondary: the Exp 3 single-band graphs with the two other encoders, same fixed hyperparameters."),
    ("additional", "Tier2_Exp5_other_fusions", "exp5_multiband_fixedhp_tier2",
     "Tier 2 (optional) · Experiment 5 band-specific encoders and concat / gate fusion",
     "Secondary: the Exp 5 variants not in Tier 1, same fixed hyperparameters."),
    ("additional", "Tier2_Exp6_ciplv_ablations", "exp6_multiband_psd_strength_fixedhp_tier2",
     "Tier 2 (optional) · Experiment 6 ablations with |ciPLV|",
     "Secondary: the seven Exp 6 ablations with |ciPLV|, same fixed hyperparameters."),
    ("additional", "Exp3b_conn_psd_allbands", "exp3b_conn_psd_allbands_fixedhp",
     "Experiment 3b (secondary) · single-band graphs + all-band PSD (fixed hyperparameters)",
     "Secondary Experiment 3 node features (all bands)."),
    ("additional", "Baseline_age_only", "baseline_age", "Baseline · age only", "Age as the only predictor."),
    ("additional", "Baseline_connectivity_classical", "baseline_conn_classical",
     "Baseline · connectivity features with LR / SVM", "Flattened connectivity matrices with classical models."),
    ("additional", "Probe_dataset_identity", "probe_dataset", "Dataset-identity probe (D1 vs D2)",
     "Predicts the dataset instead of the diagnosis (DESIGN.md §6), pooled setting only."),
] + [
    ("additional", f"Age_{name}", f"age/{cfg}_plus_age", f"EEG + age · {title}", f"{title} with age as a covariate.")
    for name, cfg, title in [
        ("Exp0_sensor_psd", "exp0_sensor_psd", "Experiment 0"), ("Exp1_source_psd", "exp1_source_psd", "Experiment 1"),
        ("Exp2_conn_minimal", "exp2_conn_minimal_fixedhp", "Experiment 2"),
        ("Exp3_conn_psd", "exp3_conn_psd_fixedhp", "Experiment 3"),
        ("Exp4_band_mean", "exp4_band_mean_fixedhp", "Experiment 4"),
        ("Exp5_multiband", "exp5_multiband_fixedhp", "Experiment 5"),
        ("Exp6_multiband_psd_strength", "exp6_multiband_psd_strength_fixedhp", "Experiment 6"),
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

SETUP_D3 = (SETUP
            .replace("# ===== Setup: nothing to edit.", "# ===== Dataset 3 setup: nothing to edit.")
            .replace("UPDATE_CODE = False   # leave False. True only if you were asked to switch to newer code\n"
                     "                      # (all notebooks then use the newest commit; unfinished work is recomputed).",
                     "UPDATE_D3_CODE = False   # leave False. True only if you were asked to switch D3 to newer code\n"
                     "# Dataset 3 has its own code pin, <outputs>/D3_CODE_COMMIT.txt (all D3 notebooks run that commit);\n"
                     "# the D1/D2 pin CODE_COMMIT.txt is only reported, never read for checkout or changed here.")
            .replace("_run([sys.executable, 'scripts/colab_setup.py'] + (['--update-code'] if UPDATE_CODE else []))",
                     "_run([sys.executable, 'scripts/colab_setup.py', '--family', 'D3']\n"
                     "     + (['--update-code'] if UPDATE_D3_CODE else []))")
            + "print('Dataset 3 folder:', PATHS['dataset3']['root'])\n")
assert "UPDATE_D3_CODE" in SETUP_D3 and "--family" in SETUP_D3

SETUP_MAINTENANCE = (SETUP
                     .replace("UPDATE_CODE = False   # leave False. True only if you were asked to switch to newer code\n"
                              "                      # (all notebooks then use the newest commit; unfinished work is recomputed).\n",
                              "# maintenance: runs the newest code of the branch and reads or writes no code pin\n")
                     .replace("_run([sys.executable, 'scripts/colab_setup.py'] + (['--update-code'] if UPDATE_CODE else []))",
                              "_run([sys.executable, 'scripts/colab_setup.py', '--no-pin'])"))
assert "--no-pin" in SETUP_MAINTENANCE

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
    settings = [s for s in v["settings"] if any(s in x["settings"] for x in variants)]
    restricted = sorted({x["variant"] for x in variants if x["settings"] != v["settings"]})
    if grid == 1:
        search = ("fixed hyperparameters, no hyperparameter search and no inner CV: one fit per outer training "
                  "fold" + (f" (protocol `{v['protocol']['name']}`)" if v.get("protocol") else ""))
    else:
        search = (f"inner {ev['inner_k']}-fold subject-level CV over {grid} hyperparameter combination(s); "
                  f"refit on the full training fold")
    text = (f"- **{len(variants)} variant(s):** " + ", ".join(f"`{x['variant']}`" for x in variants[:24])
            + (" …" if len(variants) > 24 else "") + "\n"
            f"- **Settings:** " + ", ".join(settings) + " (cross-dataset only if the DESIGN.md §7 gate passes)"
            + (f"; variants with their own settings: {', '.join(f'`{x}`' for x in restricted)}" if restricted else "")
            + "\n"
            f"- **Protocol (from the config, unchanged):** {ev['n_repeats']} repeats × {ev['k']}-fold subject-level "
            f"outer CV; {search}" + (f"; {v['training']['epochs']} epochs" if gnn else "") + ".")
    return variants, text


# experiments added after their family's results existed run their own code pin (as Exp7 does), so the family
# pin (and every completed result and graph run under it) never has to move for them
OWN_PIN = {"d3/d3_exp2_conn_minimal_fixedhp", "d3/d3_exp3_conn_psd_fixedhp", "d3/d3_exp4_band_mean_fixedhp",
           "d3/d3_exp5_multiband_fixedhp"}


def own_pin_cells(results: str, experiment: str, config: str) -> list[dict]:
    return [md(f"""## 0. This experiment's code pin
This experiment runs the commit recorded in its own folder (`{results}/{experiment}/CODE_COMMIT.txt`, set on the
first run). The family pin used by the setup cell, the graph run and every other experiment's results stay
unchanged. Leave `UPDATE_EXPERIMENT_CODE = False`; `True` moves only this experiment's pin to the newest code."""),
            code(f"""UPDATE_EXPERIMENT_CODE = False
import subprocess, sys
from pathlib import Path
from adgraph.colab import resolve_code_pin

EXP_DIR = Path(OUT) / '{results}' / '{experiment}'
head = subprocess.check_output(['git', '-C', '/content/repo', 'rev-parse', f'origin/{{BRANCH}}'], text=True).strip()
has_work = EXP_DIR.exists() and any(EXP_DIR.rglob('*.done.json'))
EXP_COMMIT, _ = resolve_code_pin(EXP_DIR / 'CODE_COMMIT.txt', head, update=UPDATE_EXPERIMENT_CODE, has_work=has_work)
_run(['git', '-C', '/content/repo', 'checkout', '-q', EXP_COMMIT])
if not Path('/content/repo', '{config}').exists():
    raise RuntimeError(f'pin {{EXP_COMMIT[:10]}} lacks {config}: set UPDATE_EXPERIMENT_CODE = True and run again')
for m in [m for m in sys.modules if m == 'adgraph' or m.startswith('adgraph.')]:
    del sys.modules[m]                     # this notebook's own imports now come from this experiment's commit
from adgraph.colab import sh
print('{experiment} code commit:', EXP_COMMIT)
for pin in ('D3_CODE_COMMIT.txt', 'CODE_COMMIT.txt'):
    if (Path(OUT) / pin).exists():
        print(f'{{pin}} (not used here, unchanged):', (Path(OUT) / pin).read_text().split()[0])""")]


def experiment_notebook(folder, name, cfg_path, title, desc) -> tuple[dict, dict]:
    variants, proto = protocol(cfg_path)
    experiment = variants[0]["experiment"]
    gnn = variants[0]["input"] == "graph"
    config = f"configs/experiments/{cfg_path}.yaml"
    d3 = variants[0].get("graph_family") == "D3"
    results = "experiments_D3" if d3 else "experiments"
    nb = notebook([
        md(f"# {title}\n\n{desc}\n\n**Runs only this experiment** (`{config}`): every predefined variant and "
           f"setting, with subject-level cross-validation, evaluation and checkpointing. It never starts another "
           f"experiment.\n\n{proto}\n" + HOWTO + f"""
**Progress** is shown as one updating line: completed outer folds of the whole experiment
(previously completed folds recovered from Drive count as done immediately), current variant,
setting, repeat and fold, inner-CV trial (if any){", training epoch" if gnn else ""}, and the counts of finished /
resumed / failed / skipped units. An ETA is shown only for the current variant and setting, from folds
computed in this session.{" Large GNN experiments take several Colab sessions: just *Run all* again after each disconnect." if gnn else ""}

Results are saved under `<outputs>/{results}/{experiment}/` on Drive: per-fold/repeat checkpoints,
window and subject predictions, metrics per repeat, selected hyperparameters, summaries, manifests,
runtime and peak memory per unit (`_run_status.csv`), and a report folder (results table +
**single-experiment diagnostic figures**). The comparative paper figures across experiments are made
separately (`03_comparative_figures.ipynb`).

Requires the graphs from `""" + ("d3/D3_01_download_and_build_graphs.ipynb" if d3 else "01_build_graphs.ipynb")
           + "`." + (" Dataset 3 (OpenNeuro ds004504, AD vs HC): graph runs `graphsD3_v4_*`, results in "
                     "`experiments_D3/`, never mixed with D1/D2." if d3 else "")),
        code(SETUP_D3 if d3 else SETUP),
        *(own_pin_cells(results, experiment, config) if cfg_path in OWN_PIN else []),
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
exp_dir = Path(OUT) / '""" + results + """' / EXPERIMENT
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


EXP7_CONFIG = "configs/experiments/exp7_earm_gnn.yaml"


def exp7_notebook() -> tuple[dict, dict]:
    """Experiment 7 (EARM-GNN): model check, training/evaluation, comparison, D1/D2 diagnostic."""
    variants, proto = protocol("exp7_earm_gnn")
    nb = notebook([
        md("""# Experiment 7 · EEG-Guided Adaptive Residual Multiband GNN (EARM-GNN)

**Exploratory extension**, developed after the Tier 1 GCN results (Exp 2-4) had been observed
(methodology log entry 43, `docs/earm_gnn.md`). It is evaluated exactly like the other GNNs: frozen v4
graph run (1,527 windows, 43 participants), the same subject-level splits and seeds, fixed hyperparameters
(no search, no inner CV, one fit per outer training fold), PLV and |ciPLV| as separate variants, plus two
pre-specified ablations: **fixed adjacency** (no learned edge gates) and **mean fusion** (no band attention).
Nothing is chosen from held-out results, and both datasets are reported.

""" + proto + "\n" + HOWTO + """
**What this notebook does:** (0) pins Exp7's own code commit, (1) checks the frozen graph run and the
features, builds every variant, shows layers, parameter counts and tensor shapes and runs a small
forward/backward smoke test, (2) trains and evaluates every variant and setting (resumable; every outer
fold is saved to Drive), (3) shows the results, ROC curves and confusion matrices, (4) compares EARM-GNN
with the GCN experiments and the PSD baselines (saved outputs only; descriptive), (5) runs a short
read-only D1 / D2 diagnostic, (6) prints a final summary.

Use a **GPU runtime** (*Runtime → Change runtime type → T4 GPU*). Results:
`<outputs>/experiments/exp7_earm_gnn/` on Drive. Other experiments are never touched."""),
        code(SETUP),
        md("""## 0. Exp7 code pin
Exp7 runs the commit recorded in its own folder (`experiments/exp7_earm_gnn/CODE_COMMIT.txt`, set on the
first run), so it can run while the other experiments stay on the global pin (`CODE_COMMIT.txt`, unchanged).
Leave `UPDATE_EXP7_CODE = False`; `True` moves only Exp7's pin to the newest code (unfinished Exp7 work is then
recomputed)."""),
        code("""UPDATE_EXP7_CODE = False
import subprocess, sys
from pathlib import Path
from adgraph.colab import resolve_code_pin

EXP7_DIR = Path(OUT) / 'experiments' / 'exp7_earm_gnn'
head = subprocess.check_output(['git', '-C', '/content/repo', 'rev-parse', f'origin/{BRANCH}'], text=True).strip()
has_work = EXP7_DIR.exists() and any(EXP7_DIR.rglob('*.done.json'))
EXP7_COMMIT, _ = resolve_code_pin(EXP7_DIR / 'CODE_COMMIT.txt', head, update=UPDATE_EXP7_CODE, has_work=has_work)
_run(['git', '-C', '/content/repo', 'checkout', '-q', EXP7_COMMIT])
for f in ('src/adgraph/models/earm.py', 'scripts/exp7_earm.py', 'configs/experiments/exp7_earm_gnn.yaml'):
    if not Path('/content/repo', f).exists():
        raise RuntimeError(f'Exp7 pin {EXP7_COMMIT[:10]} lacks {f}: set UPDATE_EXP7_CODE = True and run again')
for m in [m for m in sys.modules if m == 'adgraph' or m.startswith('adgraph.')]:
    del sys.modules[m]                     # this notebook's own imports now come from the Exp7 commit
from adgraph.colab import sh
print('Exp7 code commit:', EXP7_COMMIT)
print('global pin of the other experiments (unchanged):', (Path(OUT) / 'CODE_COMMIT.txt').read_text().split()[0])"""),
        md("""## 1. Frozen graph run, features, architecture and smoke test
Checks that the selected graph run is the frozen v4 run (`graph_run_id 6d75de9975f0d9ae`), that PLV,
|ciPLV|, log and relative PSD are present, finite and valid, and that node strength equals the strength of
each band's own EEG adjacency. Builds every variant and prints its parameter count, the layer structure,
the trainable parameters, tensor shapes of one forward pass, and runs two training epochs on a quarter of
the participants to confirm finite losses and gradients reaching the edge gates. This is NOT an evaluation."""),
        code(f"""CONFIG = '{EXP7_CONFIG}'   # this notebook's experiment
EXPERIMENT = 'exp7_earm_gnn'
import torch
print('runtime:', ('GPU ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU only')
if not torch.cuda.is_available():
    print('NOTE: no GPU. It works on CPU but is much slower (Runtime > Change runtime type > T4 GPU).')
sh("python scripts/exp7_earm.py inspect")
sh(f"python scripts/experiment.py status --config {{CONFIG}}")"""),
        md("""## 2. Train and evaluate (resumable)
Every completed outer fold (predictions, fitted parameters, learned edge gates, training history) is saved
to Drive at once; after a disconnect, *Run all* again and only unfinished folds are computed."""),
        code("""RUN_ERROR = None
try:
    sh(f"python scripts/experiment.py run --config {CONFIG}")
except RuntimeError as exc:
    RUN_ERROR = exc
    print('\\nSome units failed or are incompatible; details at the end of the notebook.')"""),
        md("""## 3. Results of Experiment 7 (saved outputs only)
Subject-level metrics: mean ± SD over CV repeats, subject-bootstrap 95 % CI; ROC curves, confusion matrices
and subject scores per setting; runtime per unit."""),
        code("""sh(f"python scripts/experiment.py report --config {CONFIG}")
import pandas as pd
from IPython.display import Image, display
report = sorted((EXP7_DIR / '_reports').glob('report_*'))[-1]
pd.set_option('display.max_rows', 500, 'display.width', 250)
table = pd.read_csv(report / 'results_summary.csv')
display(table) if len(table) else print('no completed results yet')
status = pd.read_csv(EXP7_DIR / '_run_status.csv') if (EXP7_DIR / '_run_status.csv').exists() else pd.DataFrame()
if len(status):
    display(status[['variant', 'setting', 'state', 'seconds', 'mem_peak_mb', 'message']])
for png in sorted((report / 'figures').glob('D[123]_*.png')):
    print(png.name)
    display(Image(filename=str(png)))"""),
        md("""## 4. Comparison with the GCN experiments and the PSD baselines
Saved outputs only (nothing retrained): EARM-GNN full and its two ablations, Exp 6 / 5 / 4 GCN, Exp 1 and
Exp 0 PSD models. Paired differences use only the CV repeats both results share (identical subject splits);
they are **descriptive** (no significance test). Experiments not run yet are simply absent."""),
        code("""sh("python scripts/exp7_earm.py compare")
cmp = sorted((EXP7_DIR / '_comparison').glob('comparison_*'))[-1]
read = lambda name: pd.read_csv(cmp / name) if (cmp / name).stat().st_size > 1 else pd.DataFrame()
display(pd.read_csv(cmp / 'auc_by_setting.csv', index_col=0).round(3))
full = read('comparison_table.csv')
if len(full):
    display(full[['model', 'connectivity', 'setting', 'n_repeats', 'roc_auc_mean', 'roc_auc_sd', 'roc_auc_ci95_low',
                  'roc_auc_ci95_high', 'balanced_accuracy_mean', 'sensitivity_mean', 'specificity_mean', 'f1_mean',
                  'n_parameters', 'fit_seconds_mean']].round(3))
diff = read('paired_differences.csv')
if len(diff):
    print('EARM-GNN full minus comparator (subject-level ROC AUC, per shared repeat; descriptive):')
    display(diff[diff['metric'] == 'roc_auc'].drop(columns=['reference', 'metric']).round(3))
for png in sorted((cmp / 'figures').glob('E7_*.png')):
    print(png.name)
    display(Image(filename=str(png)))"""),
        md("""## 5. Short D1 / D2 diagnostic (read-only)
Recording quality and interpolated (e.g. persistently flat) channels per dataset, feature distributions and
dataset shift, class separability of simple features **without a model**, and whether poor D1 results are
near-constant, reversed or non-discriminating predictions. Nothing is reprocessed or excluded; D1 stays in
every analysis. Written to `<outputs>/diagnostics/dataset_shift_<time>/`."""),
        code("""sh("python scripts/exp7_earm.py diagnose")"""),
        md("""## 6. Final summary
Subject-level ROC AUC [95 % CI] per setting for EARM-GNN, its ablations, the GCN experiments and the PSD
baselines, and the training-loss histories of EARM-GNN. Within-D2 results are one dataset's
cross-validation, not independent validation."""),
        code("""import numpy as np
import matplotlib.pyplot as plt
if len(full):
    f = full.assign(auc=full.apply(lambda r: f"{r.roc_auc_mean:.3f} [{r.roc_auc_ci95_low:.2f}, {r.roc_auc_ci95_high:.2f}]",
                                   axis=1))
    order = [s for s in ['within_D1', 'within_D2', 'pooled', 'cross_D1_to_D2', 'cross_D2_to_D1'] if s in set(f['setting'])]
    display(f.pivot_table(index=['model', 'connectivity'], columns='setting', values='auc', aggfunc='first',
                          sort=False).reindex(columns=order))
hist = {}
for npz in EXP7_DIR.glob('*__full/*/units/fold_*.model.npz'):
    hist.setdefault((npz.parts[-4], npz.parts[-3]), []).append(np.load(npz)['history_train_loss'])
if hist:
    fig, ax = plt.subplots(figsize=(7, 4))
    for (variant, setting), curves in sorted(hist.items()):
        ax.plot(np.mean(curves, 0), label=f'{variant} / {setting} ({len(curves)} folds)')
    ax.set_xlabel('epoch'); ax.set_ylabel('weighted training loss (mean over folds)'); ax.legend(fontsize=7)
    ax.set_title('EARM-GNN full: training loss (training subjects only)')
    plt.show()
print('\\n'.join([
    'How to read this:',
    '* EARM-GNN is an exploratory extension designed after the Tier 1 GCN results were seen; report it as such.',
    '* Ablation A (fixed adjacency) and B (mean fusion) each differ from the full model in one component;',
    '  the paired differences are descriptive. No permutation test was run: claim no significance.',
    '* Both datasets are primary (S1, S2). Poor D1 performance is a result, not a reason to drop D1 (section 5).',
    '* Fixed hyperparameters, set before any Exp7 result; nothing was tuned on test subjects.']))"""),
        code("""bad = status[status['state'].isin(['failed', 'incompatible'])] if len(status) else status
if RUN_ERROR is not None or len(bad):
    raise RuntimeError(f"{len(bad)} unit(s) of {EXPERIMENT} did not complete (see {EXP7_DIR / '_run_status.csv'}); "
                       "completed units are saved. Fix the cause, then Run all again: only incomplete units are recomputed.")
print('Experiment 7 complete.')"""),
    ], gpu=True)
    entry = {"folder": "experiments", "name": "Exp7_EARM_GNN", "title": "Experiment 7 · EARM-GNN",
             "config": EXP7_CONFIG, "gpu": True}
    return nb, entry


# ============================================================================ Dataset 3 (ds004504)
D3_EXPERIMENTS = [
    ("D3_Exp0_sensor_psd", "d3/d3_exp0_sensor_psd", "D3 · Experiment 0 · sensor PSD (LR, SVM)",
     "Dataset 3 (ds004504, AD vs HC): electrode band power (19 channels × 4 bands × {log, relative})."),
    ("D3_Exp1_source_psd", "d3/d3_exp1_source_psd", "D3 · Experiment 1 · source PSD (LR, SVM)",
     "Dataset 3: source ROI band power (68 ROIs × 4 bands × {log, relative})."),
    ("D3_Baseline_age_only", "d3/d3_baseline_age", "D3 · Baseline · age only",
     "Dataset 3: age as the only predictor (DESIGN.md §5)."),
    ("D3_Exp2_conn_minimal", "d3/d3_exp2_conn_minimal_fixedhp",
     "D3 · Experiment 2 · single-band connectivity only (GCN)",
     "Dataset 3: each band on its own (theta, alpha1, alpha2, beta), single-band source connectivity graph with a "
     "constant node feature, GCN; PLV and |ciPLV| (the Tier 1 Exp 2 design, methodology log entry 49)."),
    ("D3_Exp3_conn_psd_single_band", "d3/d3_exp3_conn_psd_fixedhp",
     "D3 · Experiment 3 · single-band connectivity + PSD (GCN)",
     "Dataset 3: each band on its own (theta, alpha1, alpha2, beta), single-band source connectivity graph with "
     "matching-band [log, relative] PSD node features, GCN; PLV and |ciPLV| (the Tier 1 Exp 3 design, "
     "methodology log entry 48)."),
    ("D3_Exp4_band_mean", "d3/d3_exp4_band_mean_fixedhp",
     "D3 · Experiment 4 · band-averaged graph (GCN)",
     "Dataset 3: band-averaged adjacency of the four bands, nodes = [log, relative] source PSD of all four bands, "
     "GCN; PLV and |ciPLV| (the Tier 1 Exp 4 design, methodology log entry 49)."),
    ("D3_Exp5_multiband", "d3/d3_exp5_multiband_fixedhp",
     "D3 · Experiment 5 · multi-band GNN (shared GCN, mean / attention fusion)",
     "Dataset 3: four band graphs, one shared GCN encoder, mean or attention fusion, nodes = matching-band "
     "[log, relative] PSD; PLV and |ciPLV| (the Tier 1 Exp 5 design, methodology log entry 49)."),
    ("D3_Exp6_GCN_full", "d3/d3_exp6_gcn_full", "D3 · Experiment 6 · multi-band GCN, full model",
     "Dataset 3: four band graphs, nodes = [log PSD, relative PSD, strength], attention fusion; PLV and |ciPLV|."),
    ("D3_Exp7_EARM_GNN", "d3/d3_exp7_earm_gnn", "D3 · Experiment 7 · EARM-GNN (+ two ablations)",
     "Dataset 3: EARM-GNN full, fixed adjacency and mean fusion; PLV and |ciPLV| (exploratory, docs/earm_gnn.md)."),
]

def d3_build_notebook() -> dict:
    return notebook([
        md("""# D3 · 01 · Download ds004504 and build its source-space graphs

**Dataset 3** = OpenNeuro **ds004504** (Miltiadous et al., 2023; doi:10.18112/openneuro.ds004504, CC0): eyes-closed
resting-state EEG, 19 channels (10-20), 500 Hz. Used: the **36 AD and 29 HC** participants, from the provider's
**preprocessed derivatives**; the 23 FTD participants are excluded by design and never downloaded
(methodology log entry 44, `docs/dataset_d3.md`).

Processing (`configs/pipeline_d3.yaml`): the provider's 0.5-45 Hz filter, ASR and ICA cleaning are used as given
(nothing repeated). Recordings are split at the EEGLAB 'boundary' events that the provider's cleaning left
(data removed there): each continuous segment is resampled on its own and no window crosses a discontinuity.
The forward and inverse models use the electrode coordinates stored in the `.set` files (methodology log entry 46;
the provider assigned them by an EEGLAB lookup of the DIPFIT Colin27/MNI template, so they are a shared template, not
individual measurements), one verified operator per distinct electrode geometry. Added: the 10 s edge crop, resampling to 250 Hz, common average reference, the same 20 s /
50 % windows, window rejection, fsaverage source model, 68 Desikan-Killiany ROIs, bands, PLV / |ciPLV| and PSD
as D1/D2. Bad-channel rules are evaluated and reported only (no interpolation added).
""" + HOWTO + """
**Code pin:** the setup cell runs the commit in `<outputs>/D3_CODE_COMMIT.txt` (Dataset 3's own pin); the
D1/D2 pin `CODE_COMMIT.txt` is never changed by D3 notebooks.

**Steps:** 1. download (≈ 2.2 GB to `My Drive/OpenNeuro/ds004504`, resumable, MD5-verified) ·
2. input check (36 AD / 29 HC, 19 channels in every file, EEGLAB boundary events and stored electrode
coordinates per recording) · 3. smoke test (first subject) and its inspection ·
3b. real-data montage checks · 4. full graph construction (65 recordings, resumable; **only after you set
`CONFIRM_D3_FULL_BUILD = True`**) · 5. validation and completion report · 6. summary.
Output: `<outputs>/graphsD3_v4_<config hash>/` (never mixed with the D1/D2 graph runs). Experiments on D3:
`notebooks/d3/experiments/`. A GPU is not needed for this notebook."""),
        code(SETUP_D3),
        md("## 1. Download ds004504 (AD and HC derivatives only)"),
        code('sh("python scripts/d3_data.py download")'),
        md("## 2. Input check (participants and recording headers; no signal is read)"),
        code('sh("python scripts/d3_data.py check")'),
        md("## 3. Smoke test: first subject, then its inspection (diagnostics, not paper figures)"),
        code("""sh("python scripts/build_graphs.py --config configs/pipeline_d3.yaml --limit 1")
sh("python scripts/inspect_graph_subjects.py --config configs/pipeline_d3.yaml --limit 1")
from IPython.display import Image, display
from adgraph.config import load_yaml
from adgraph.pipeline import graph_run_dir
smoke = graph_run_dir(OUT, load_yaml('configs/pipeline_d3.yaml'), 1)
for png in sorted(sorted(smoke.glob('inspection_*'))[-1].glob('*.png')):
    display(Image(str(png)))"""),
        md("""## 3b. Real-data montage checks (must all PASS)
Explicitly selected inputs (never "the newest"): the D3 smoke run and the operator recorded in its manifest,
the frozen D1/D2 run by `graph_run_id 6d75de9975f0d9ae`, the header report of step 2. Checks: (1) electrode
labels and the coordinates stored in the `.set` file: valid 3D positions for all 19 channels, coordinate frame
established from the provider's lookup record, the operator built from exactly these positions, the head ->
fsaverage MRI transformation returning them, every subject on the operator of its own geometry; (2) no confirmed flat channel entered reconstruction (a flat
channel stops its subject with `FlatChannelError`); (3) inverse operator 68 × 19 with the configured channels,
verified against the full MNE inverse; (4) 68 ROI names and order identical to D1/D2; (5) EEGLAB 'boundary'
discontinuities are handled (methodology log entry 45): the recording is split at every boundary, each continuous
segment is processed on its own and windows lie inside one segment, at least the margin from its ends. The check
runs the smoke subject and, in memory (nothing saved), the recording with the most boundary events, prints the
per-participant table (boundary events, segments, candidate windows, windows excluded for a discontinuity,
windows rejected by QC, accepted windows) and requires **zero accepted windows crossing a discontinuity or its
margin**; (6) PSD, PLV and |ciPLV| finite, in range and 68 × 68 per band. The coordinate source, frame, montage
hash and operator id are saved in `d3_montage_checks.json`. A failure stops the notebook here."""),
        code('sh("python scripts/d3_verify_smoke.py")'),
        md("""## 4. Full graph construction (resumable)
Starts only after you have reviewed steps 2-3b: set `CONFIRM_D3_FULL_BUILD = True` and run this cell (and the
ones below). Progress is saved per subject; if the runtime stops, *Run all* again."""),
        code("""CONFIRM_D3_FULL_BUILD = False   # set True only after every check of step 3b passed and the smoke test was reviewed
if not CONFIRM_D3_FULL_BUILD:
    raise RuntimeError('Full D3 graph construction not started (CONFIRM_D3_FULL_BUILD = False). Review steps 2-3b, '
                       'then set CONFIRM_D3_FULL_BUILD = True and run this cell and the cells below.')
sh("python scripts/build_graphs.py --config configs/pipeline_d3.yaml --status")
sh("python scripts/build_graphs.py --config configs/pipeline_d3.yaml")"""),
        md("## 5. Validate the D3 graph dataset and write the completion report"),
        code('sh("python scripts/validate_graph_run.py --prefix graphsD3")'),
        md("## 6. Summary"),
        code("""import json
import pandas as pd
from adgraph.pipeline import latest_complete_graph_run
run = latest_complete_graph_run(OUT, 'graphsD3')
m = json.load(open(run / 'manifest.json'))
print('graph run:', run)
print('graph_run_id', m['graph_run_id'], '| commit', m['git_commit'], '| operator error',
      m['source_operator']['verification_rel_error'])
print('electrode geometry:', {k: v for k, v in m['source_operator'].get('geometry', {}).items() if k != 'pos_m'},
      '| operators:', len(m.get('source_operators', [])) or 1)
print('per dataset:', m['summary']['per_dataset'])
print('excluded by design:', len(m['registry'].get('excluded_by_design', {}).get('D3', [])), 'FTD participants')
s = pd.read_csv(run / 'subjects.csv')
display(s[[c for c in ['key', 'label', 'duration_s', 'median_std_uv', 'n_bad_channels', 'qc_spatial_rho', 'status']
          if c in s]])
print('discontinuities per participant (EEGLAB boundary events; windows of 20 s):')
display(s[[c for c in ['key', 'n_boundary_events', 'n_segments', 'n_segments_usable', 'usable_duration_s',
                       'n_windows_continuous_grid', 'n_windows_excluded_discontinuity', 'n_windows_total',
                       'n_windows_rejected', 'n_windows_kept', 'status'] if c in s]])
print(s.groupby('label')['n_windows_kept'].agg(['count', 'sum', 'median']))"""),
    ])


def d3_comparison_notebook() -> dict:
    return notebook([
        md("""# D3 · 99 · Comparison of the Dataset 3 experiments (saved outputs only)

Subject-level results on Dataset 3 (ds004504, 36 AD / 29 HC, within-D3 cross-validation): EARM-GNN and its two
ablations, the multi-band GCN, source and sensor PSD models and the age-only baseline. Tables, ROC curves,
confusion matrices and per-repeat paired differences (descriptive; no significance test). Nothing is trained.
Run after the D3 experiment notebooks; missing experiments are simply absent.""" + HOWTO),
        code(SETUP_D3),
        md("## 1. Comparison"),
        code("""sh("python scripts/exp7_earm.py compare --family D3")
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
cmp = sorted((Path(OUT) / 'experiments_D3' / '_comparison').glob('comparison_*'))[-1]
display(pd.read_csv(cmp / 'auc_by_setting.csv', index_col=0).round(3))
t = pd.read_csv(cmp / 'comparison_table.csv') if (cmp / 'comparison_table.csv').stat().st_size > 1 else pd.DataFrame()
if len(t):
    display(t[['model', 'connectivity', 'n_repeats', 'roc_auc_mean', 'roc_auc_sd', 'roc_auc_ci95_low',
               'roc_auc_ci95_high', 'balanced_accuracy_mean', 'sensitivity_mean', 'specificity_mean', 'f1_mean',
               'n_parameters', 'fit_seconds_mean']].round(3))
for png in sorted((cmp / 'figures').glob('D3_*.png')):
    print(png.name)
    display(Image(filename=str(png)))"""),
    ])


def build_graphs_notebook() -> dict:
    return notebook([
        md("# 01 · Build source-space graphs (pipeline v4, boundary-aware)\n" + HOWTO + """
Pipeline v4 (methodology log entry 41, `docs/boundary_processing.md`): the 10 s at each end are discarded
before any filtering, every filter pads explicitly, the analytic signal is a finite complex FIR, and
windows keep a computed boundary margin. Outputs go to a NEW directory `graphs_v4_<config hash>/`;
earlier graph runs, checkpoints, experiments and audits are not touched.

Steps (each stops the notebook with a clear message if something is wrong):
1. setup and input check: 20 + 23 recordings, labels 10 AD / 10 HC and 10 AD / 13 HC;
2. synthetic test suite (once per code version, about 10–15 min);
3. EDF header check (all 30 channels used by the pipeline present);
4. smoke test on the first subject of each dataset (downloads fsaverage into the Drive cache once);
5. full graph construction for all 43 recordings (several hours; resumable);
5b. validation of the assembled dataset (shapes, finite values, ranges, symmetry, window margins) and
    the completion report (`validation_<time>/dataset_report.csv`, `validation.json`);
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
        md("## 3b. Amplitude diagnostic (read-only)\nReads the first recording of each dataset (the smoke-test "
           "subjects) and breaks the preprocessing amplitude check down by stage: EDF header units, MNE's unit "
           "conversion, dataset scale, band-pass, 1 Hz high-pass, average reference, edges, drift, clipping, "
           "spectra (methodology log entry 38). It writes only to `diagnostics/amplitude_<time>/` and changes "
           "nothing. Other recordings: add `--keys D1-3 D2-5`."),
        code("""sh(f"python scripts/diagnose_amplitude.py --paths configs/paths.yaml --out '{OUT}'")
from pathlib import Path
from IPython.display import Image, display
display(Image(str(sorted(Path(OUT, 'diagnostics').glob('amplitude_*/amplitude.png'))[-1])))"""),
        md("## 4. Smoke test: first subject of each dataset\nWritten to Drive as `graphs_v3_<hash>_limit1/`; "
           "it is never used for analysis."),
        code('sh("python scripts/build_graphs.py --paths configs/paths.yaml --limit 1")'),
        md("## 4b. Inspect the smoke-test subjects (read-only)\nPreprocessing QC (unit safeguard, amplitude "
           "on the analysed interval, bad channels), every candidate window kept or rejected and why, ROI and "
           "sensor band power, mean PLV and |ciPLV| per band. Review these before the full build (step 5). "
           "Diagnostics, not paper figures (methodology log entry 39)."),
        code("""sh("python scripts/inspect_graph_subjects.py --paths configs/paths.yaml --limit 1")
from pathlib import Path
from IPython.display import Image, display
from adgraph.config import load_yaml
from adgraph.pipeline import graph_run_dir
smoke = graph_run_dir(OUT, load_yaml('configs/pipeline.yaml'), 1)
for png in sorted(sorted(smoke.glob('inspection_*'))[-1].glob('*.png')):
    display(Image(str(png)))"""),
        md("## 5. Full graph construction (resumable)\nProgress is saved per subject. If the runtime stops, "
           "*Run all* again."),
        code('sh("python scripts/build_graphs.py --paths configs/paths.yaml --status")\n'
             'sh("python scripts/build_graphs.py --paths configs/paths.yaml")'),
        md("## 5b. Validate the graph dataset and write the completion report\nImplementation checks only "
           "(shapes, finite values, ranges, symmetry, window grid and boundary margins, manifest). With the "
           "boundary audit folder present, the accepted windows of the audited production run are listed next "
           "to the new counts. Stops the notebook if any check fails."),
        code("""from pathlib import Path
AUDIT_PART2 = Path('/content/drive/MyDrive/AD-LEB-audit/qc_boundary_audit_7fd5a69/part2')   # optional reference
ref = f" --reference-audit '{AUDIT_PART2}'" if AUDIT_PART2.is_dir() else ''
sh("python scripts/validate_graph_run.py --paths configs/paths.yaml" + ref)"""),
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
display(s[[c for c in ['key', 'label', 'orig_sfreq', 'duration_s', 'gross_robust_std_uv', 'median_std_uv',
                       'median_robust_std_uv', 'n_bad_channels', 'n_windows_kept', 'n_windows_total',
                       'qc_edge_window_ptp_ratio', 'channel_naming', 'qc_spatial_rho', 'seconds',
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


def restore_pin_notebook() -> dict:
    return notebook([
        md("""# ADVANCED · Restore the D1/D2 code pin (maintenance, run once)
""" + HOWTO + """
Dataset 3 now has its own code pin (`<outputs>/D3_CODE_COMMIT.txt`, methodology log entry 45). Before that, D3
notebooks used the shared pin `CODE_COMMIT.txt`, which a D3 session could move to a Dataset 3 commit. This
notebook sets `CODE_COMMIT.txt` back to the last D1/D2 pin that predates Dataset 3, read from the pin's own
history (expected: `1b6988a`, the fixed-hyperparameter commit). The old lines stay in the file as history.

It changes **only that text file**: no graph, checkpoint or result is modified. It refuses (and changes nothing)
if any D1/D2 unit was completed under the current pin. Running it again is harmless (it then reports
"unchanged"). It runs the newest code without reading or writing any pin."""),
        code(SETUP_MAINTENANCE),
        md("## 1. Pins and the commits recorded by completed units (before)"),
        code('sh("python scripts/code_pin.py status")'),
        md("## 2. Restore the D1/D2 pin"),
        code('sh("python scripts/code_pin.py restore-d1d2")'),
    ])


DRIVE_LAYOUT = """\
<outputs>/                          AD-LEB/processed (or My Drive/AD-LEB-processed if that is read-only)
├── CODE_COMMIT.txt                 code commit used by every D1/D2 session and notebook
├── D3_CODE_COMMIT.txt              Dataset 3's own code pin (notebooks/d3; never changes CODE_COMMIT.txt)
├── header_report_<time>.csv        EDF header checks (notebook 01, step 3)
├── diagnostics/amplitude_<time>/   read-only amplitude diagnostic (notebook 01, step 3b)
├── cache/
│   ├── subjects/fsaverage/         fsaverage template (downloaded once)
│   ├── operators/operator_<id>.npz + .done.json   verified 68x30 sensor->ROI operator
│   ├── input_sha256.json           content hashes of EDFs and sheets
│   └── tests_passed/<commit>_<env>.txt            test-suite pass record
├── graphs_v3_<config hash>_limit1/ earlier smoke tests (pipeline v3; kept, never used for analysis)
├── graphs_v4_<config hash>_limit1/ smoke test of the boundary-aware pipeline (never used for analysis)
├── graphs_v4_<config hash>/        graph run used by the experiments (highest version, entry 41)
│   ├── run_identity.json  status.csv  memory_log.csv  last_attempt.json (failed subjects of the last run)
│   ├── subjects/<D1-k|D2-k>/       per-subject checkpoint: arrays.npz windows.csv subject.json subject.done.json
│   │                               qc_channels.csv qc_windows.csv (failed.json: the error of a failed subject)
│   ├── inspection_<time>/          read-only per-subject QC figures (notebook 01, step 4b)
│   ├── graphs.npz  windows.csv  subjects.csv  manifest.json  assembly.done.json
│   │                               (only when every subject succeeded and the cohort is not empty)
│   ├── validation_<time>/          dataset_report.csv validation.json (notebook 01, step 5b)
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
│   └── exp7_earm_gnn/              Experiment 7 (EARM-GNN): as above, plus CODE_COMMIT.txt (Exp7's own code pin),
│                                   units/fold_*.model.npz (fitted parameters, edge gates, training history),
│                                   _inspect/ (model check), _comparison/comparison_<time>/ (tables + figures)
├── diagnostics/dataset_shift_<time>/  read-only D1 / D2 diagnostic (Exp7 notebook)
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
(whole plan; starts nothing unless explicitly enabled). Maintenance:
[advanced/Restore_D1D2_code_pin.ipynb]({COLAB}/advanced/Restore_D1D2_code_pin.ipynb) (sets the D1/D2 code
pin back to its pre-Dataset 3 commit; changes no result).

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
               nbdir / "advanced" / "Run_all_experiments_ADVANCED.ipynb": advanced_notebook(),
               nbdir / "advanced" / "Restore_D1D2_code_pin.ipynb": restore_pin_notebook()}
    entries = []
    for folder, name, cfg, title, desc in EXPERIMENTS:
        nb, entry = experiment_notebook(folder, name, cfg, title, desc)
        outputs[nbdir / folder / f"{name}.ipynb"] = nb
        entries.append(entry)
    nb, entry = exp7_notebook()
    outputs[nbdir / "experiments" / "Exp7_EARM_GNN.ipynb"] = nb
    entries.append(entry)
    d1d2_configs = [e["config"] for e in entries]
    outputs[nbdir / "d3" / "D3_01_download_and_build_graphs.ipynb"] = d3_build_notebook()
    for name, cfg, title, desc in D3_EXPERIMENTS:
        nb, entry = experiment_notebook("d3/experiments", name, cfg, title, desc)
        outputs[nbdir / "d3" / "experiments" / f"{name}.ipynb"] = nb
        entries.append(entry)
    outputs[nbdir / "d3" / "D3_99_comparison.ipynb"] = d3_comparison_notebook()
    outputs[nbdir / "03_comparative_figures.ipynb"] = comparative_notebook(d1d2_configs)   # D1/D2 paper figures
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
