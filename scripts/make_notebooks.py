"""Generate the Colab notebooks of the final framework from one template (keeps them identical and thin).

    python scripts/make_notebooks.py          # rewrite notebooks/ ; tests check they are up to date

    notebooks/00_setup_and_stores.ipynb        verify and lock the four validated stores (CPU)
    notebooks/01_participant_qc.ipynb          participant QC of every store; category A decisions (CPU)
    notebooks/02_parcellation.ipynb            source-resolution analysis, feature reliability, atlas recommendation
    notebooks/03_graph_construction.ipynb      graph diagnostics + the pre-specified development-CV comparison (GPU)
    notebooks/Exp<k>_*.ipynb                   ONE experiment each, Exp 0-8, every inverse x window mode (GPU for 2-8)
    notebooks/99_results_report.ipynb          combined tables, LCMV vs wMNE, mode B vs A, difficult participants
    notebooks/README.md                        index, Colab links, execution order, Drive layout

Every notebook: Runtime -> Run all. The only cell meant to be edited is an experiment notebook's CONTEXTS list.
Scientific code lives in src/adgraph; the notebooks call scripts/final.py and scripts/build_stores.py only.
Earlier notebooks (D1/D2, D3, LX, WX, ES200, R1) are archived in legacy/notebooks/ (docs/final/inventory.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from adgraph import release
from adgraph.experiments.config import expand_variants, experiment_configs, load_experiment

REPO = Path(__file__).resolve().parents[1]
OWNER_REPO = "Samimj9/Source-space-EEG-graph-learning-for-Alzheimer-s-disease-using-multi-band-functional-connectivity"
BRANCH = "claude/unified-final-framework"
REPO_URL = f"https://github.com/{OWNER_REPO}.git"
COLAB = f"https://colab.research.google.com/github/{OWNER_REPO}/blob/{BRANCH}/notebooks"

SETUP = f'''# ===== Setup: nothing to edit. Mounts Drive, gets the code, installs the pinned environment. =====
UPDATE_CODE = False   # leave False. True only to move the release code pin deliberately to the newest commit
                      # (<outputs>/{release.PIN_FILE}; unfinished work is then recomputed).
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
FINAL = OUT + '/{release.ROOT}'
print('release outputs on Drive:', FINAL)
'''

HOWTO = f"""
**How to use:** *Runtime → Run all*. Allow Google Drive access when asked. If Colab disconnects, open the notebook
again and *Run all* again: finished work is kept on Google Drive and skipped. Everything is written to
`<outputs>/{release.ROOT}/` on Drive (never to the temporary Colab disk); the validated graph stores and all earlier
results are only read.
"""

HOLDOUT = (f"**Holdout policy.** {release.HOLDOUT_POLICY}")


def run_plan() -> dict:
    import yaml

    return yaml.safe_load((REPO / "configs" / "final" / "run_plan.yaml").read_text())


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


SHOW_LATEST = '''import glob
import pandas as pd
from IPython.display import Image, display
pd.set_option('display.max_rows', 400, 'display.width', 250, 'display.max_columns', 40)

def latest(pattern):
    found = sorted(glob.glob(pattern))
    return found[-1] if found else None'''


# ------------------------------------------------------------------------------------------------ 00 / 01 / 02 / 03
def setup_notebook() -> dict:
    return notebook([
        md(f"""# 00 · Setup and stores (run first)

Verifies the four validated source-graph stores the final framework reads (LCMV and wMNE, window modes B and A,
68 Desikan-Killiany ROIs) and records each one's content identity in the release lock
(`{release.LOCK_FILE}`): from then on every experiment refuses a store whose content changed.
Nothing is built or modified. CPU runtime.
""" + HOWTO),
        code(SETUP),
        md("## 1. Store verification (read-only)"),
        code('sh("python scripts/final.py stores")'),
        md("## 2. Lock the stores and run the synthetic test suite once for this commit"),
        code('''from adgraph.colab import run_tests_once
from adgraph.stores import open_store
for inverse in ('lcmv', 'wmne'):
    for mode in ('B', 'A'):
        store, rec = open_store(OUT, inverse, mode)
        print(f"{inverse} / mode {mode}: {rec['dir']} ({rec['n_windows']} windows, graph_run_id {rec['graph_run_id']})")
run_tests_once(OUT, '/content/repo')'''),
    ])


def qc_notebook() -> dict:
    return notebook([
        md(f"""# 01 · Participant quality control (before any classification)

For every store: label-blind participant summaries, **category A** technical corruption (blocks the experiments
until a decision is recorded), **category B** statistically unusual but valid EEG (robust modified z-scores against
the 52 development participants, multivariate MCD outliers; flagged and kept). Category C (difficult to classify)
follows after the experiments (notebook 99). **No participant is excluded by any rule**: a decision other than
`keep` stops the experiments, because an exclusion changes the frozen split and must be a logged protocol amendment.
Decisions go in `{release.ROOT}/qc/decisions.yaml` (format in `docs/final/qc.md`). CPU runtime.
""" + HOWTO),
        code(SETUP),
        md("## 1. QC of the four stores (computed once, then loaded)"),
        code('''UNRESOLVED = []
for inverse in ('lcmv', 'wmne'):
    for mode in ('B', 'A'):
        try:
            sh(f"python scripts/final.py qc --inverse {inverse} --mode {mode}")
        except RuntimeError:
            UNRESOLVED.append((inverse, mode))'''),
        md("## 2. Reports"),
        code(SHOW_LATEST + '''
from IPython.display import Markdown
for d in sorted(glob.glob(FINAL + '/qc/*_*_*/')):
    if os.path.exists(d + 'qc_report.md'):
        display(Markdown(open(d + 'qc_report.md').read()))
        if os.path.exists(d + 'pca.png'):
            display(Image(filename=d + 'pca.png'))
if UNRESOLVED:
    raise RuntimeError(f'category A findings need a decision for {UNRESOLVED}: see the reports above and '
                       'docs/final/qc.md; experiments refuse to start until then')'''),
    ])


def parcellation_notebook() -> dict:
    return notebook([
        md("""# 02 · Cortical parcellation for 19-channel EEG (source resolution; no labels)

1. **Source resolution per atlas** (68 Desikan-Killiany labels, 34 gyral groups, 18 regions, 12 lobes) for both
   inverses, from the fsaverage forward model with the stores' BEM and the stores' exact inverse settings: spatial
   rank of the 19-electrode leadfield, ROI-operator rank (forced redundancy), cross-talk, own-region dominance,
   leakage correlation. No EEG and no diagnosis is used.
2. **Atlas recommendation** by the pre-specified rule of `configs/final/selection.yaml` (`atlas_resolution`).
3. **Feature reliability** of the existing 68-ROI stores (development participants only, label-blind): split-half
   reliability of PSD / DE, node strength and edges; the rank of each participant's saved LCMV ROI operator.
   3b: whether coarser-atlas ROI signals could be recomputed from the stores' saved operators and ROI signals.

Building stores of a coarser atlas needs the author's approval (section 4 does nothing unless enabled). CPU runtime.
""" + HOWTO),
        code(SETUP),
        md("## 1-2. Source resolution and the atlas recommendation"),
        code('sh("python scripts/final.py parcellation")'),
        code(SHOW_LATEST + '''
import json
d = latest(FINAL + '/parcellation/resolution_*')
display(pd.read_csv(d + '/resolution_by_atlas.csv').round(3))
rec = json.load(open(d + '/recommendation.json'))
print('recommended atlas:', rec['recommended'], '|', rec['forward_model'])
for v in rec['verdicts']:
    print(v['atlas'], 'eligible' if v['eligible'] else 'not eligible', '; '.join(v['reasons']))'''),
        md("## 3. Feature reliability of the 68-ROI stores (development participants)"),
        code('''for inverse in ('lcmv', 'wmne'):
    sh(f"python scripts/final.py reliability --inverse {inverse} --mode B")'''),
        md("""## 3b. Can coarser-atlas ROI signals be rebuilt from the saved intermediates? (read-only)

Per development participant: rank of the saved ROI operator W, its response to a constant (average reference), the
relative residual of recovering the average-referenced sensor signals from the saved ROI signals (pinv(W)), and the
fraction of windows whose analytic-filter margins lie inside the saved pieces (only those could be recomputed
exactly without the recordings). Nothing is built; this only quantifies the shortcut (docs/final/parcellation.md §6)."""),
        code('''for inverse in ('lcmv', 'wmne'):
    sh(f"python scripts/final.py reconstruct-check --inverse {inverse} --mode B")'''),
        md("""## 4. Stores of a coarser atlas (ONLY after the author's approval)

Builds the recommended atlas's stores for both inverses with the unchanged pipeline (merged Desikan-Killiany labels,
`mean_flip` over their own sources), validates them and matches wMNE to LCMV. New store directories; the 68-ROI
stores are untouched. CPU, about 40 min (LCMV) + 30-45 min (wMNE) per atlas. Then set `release.PRIMARY['atlas']`
by a logged decision (docs/final/parcellation.md)."""),
        code('''BUILD_COARSE_STORES = False      # set True only after the author approved the coarser atlas
ATLAS = rec['recommended']
if BUILD_COARSE_STORES and ATLAS and ATLAS != 'dk68':
    for inverse in ('lcmv', 'wmne'):
        sh(f"python scripts/build_stores.py build --inverse {inverse} --atlas {ATLAS}")
        sh(f"python scripts/build_stores.py validate --inverse {inverse} --atlas {ATLAS}")
    sh(f"python scripts/build_stores.py match --atlas {ATLAS}")
else:
    print('no store built (BUILD_COARSE_STORES is False or dk68 recommended)')'''),
    ])


def graph_notebook() -> dict:
    import yaml

    rule = yaml.safe_load((REPO / "configs" / "final" / "selection.yaml").read_text())["graph_construction"]
    cons = [rule["default"]] + rule["candidates"] + rule["reported_only"]
    refs = [(exp, var) for exp, vs in rule["reference"].items() for var in vs]
    return notebook([
        md(f"""# 03 · Graph construction: diagnostics and the pre-specified comparison

1. **Diagnostics** (label-blind, development participants only): density, degree, isolated nodes, connected
   components, retained connectivity weight, within- vs between-participant edge stability, for every construction
   of `configs/final/graphs.yaml` (dense reference, union / mutual kNN with k = 10 / 20 / 30 % of the ROIs,
   proportional threshold, cortical contiguity, feature similarity).
2. **Comparison** (development CV only; the holdout is not run here): the reference models
   {', '.join(f'`{e}` `{v}`' for e, v in refs)} for every construction, both inverses, mode
   {rule['mode']}.
3. **Selection** by the rule of `configs/final/selection.yaml`: a construction replaces `dense` only if it is better
   by at least {rule['margin']} mean participant AUC AND in at least {int(rule['min_fraction_of_cells_better'] * 100)} %
   of the cells; otherwise `dense` is kept. The feature-similarity graph is reported, never selected. The decision is
   recorded once (`{release.ROOT}/decisions/graph_construction.json`) and used by Exp 2-8 (`--graph selected`).

{HOLDOUT}

GPU runtime (T4). Resumable.
""" + HOWTO),
        code(SETUP),
        md("## 1. Diagnostics"),
        code('''for inverse in ('lcmv', 'wmne'):
    sh(f"python scripts/final.py graphs --inverse {inverse} --mode B")'''),
        md("## 2. Comparison runs (development CV only)"),
        code(f'''CONSTRUCTIONS = {json.dumps(cons)}
REFERENCE = {json.dumps(refs)}
ERRORS = []
for graph in CONSTRUCTIONS:
    for inverse in {json.dumps(rule['inverses'])}:
        for exp, variant in REFERENCE:
            try:
                sh(f"python scripts/final.py run --exp {{exp.split('_')[0]}} --inverse {{inverse}} --mode {rule['mode']} "
                   f"--graph {{graph}} --variants {{variant}} --settings xu_dev_cv")
            except RuntimeError:
                ERRORS.append((graph, inverse, exp, variant))'''),
        md("## 3. Selection (development CV only)"),
        code('''sh("python scripts/final.py report")
if ERRORS:
    raise RuntimeError(f'units did not complete: {ERRORS}; completed units are saved. Run all again.')
sh("python scripts/final.py select")'''),
    ], gpu=True)


# ------------------------------------------------------------------------------------------------ experiments
EXP_TITLES = {
    "exp0": ("Exp0_sensor_psd_de", "Sensor-space PSD / DE with logistic regression and SVM", False),
    "exp1": ("Exp1_source_psd_de", "Source-space PSD / DE with logistic regression and SVM", False),
    "exp2": ("Exp2_conn_minimal", "Minimal connectivity-only GCN", True),
    "exp3": ("Exp3_single_band", "Single-band GCN with PSD node features", True),
    "exp4": ("Exp4_band_mean", "Multiband averaged-adjacency GCN", True),
    "exp5": ("Exp5_multibranch", "Frequency-aware multibranch GNN (mean / attention fusion)", True),
    "exp6": ("Exp6_psd_strength", "PSD / DE and connectivity-strength ablations", True),
    "exp7": ("Exp7_earm_gnn", "EARM-GNN with learned adaptive connectivity (exploratory)", True),
    "exp8": ("Exp8_xu_dual_gcn", "Xu-inspired dual-adjacency GCN", True),
}


def experiment_notebook(exp: str, cfg: Path) -> dict:
    name, title, gnn = EXP_TITLES[exp]
    v = expand_variants(load_experiment(cfg))
    sensor = exp == "exp0"
    plan = run_plan()["contexts"]["sensor" if sensor else "source"]
    ctx = "[" + ", ".join(f"('{i}', '{m}')" for i, m in plan) + "]"
    variants = "\n".join(f"- `{x['variant']}`" for x in v)
    protocol = ("Classical model: standardization fitted on the training windows, C chosen by participant-level inner "
                "CV (5 folds) inside each outer training set, refit on the whole training set; logistic regression "
                "scored by P(AD) (threshold 0.5), SVM by its decision margin (threshold 0)" if not gnn else
                "Fixed hyperparameters (protocol `gnn_fixed60_v1`): exactly 60 epochs, Adam lr 1e-3, weight decay "
                "1e-4, batch 32, participant- and class-balanced BCE, no early stopping, no search; node features "
                "z-scored with training statistics; score P(AD) (threshold 0.5)")
    return notebook([
        md(f"""# {exp.upper()[:1]}{exp[1:3]} {exp[3:]} · {title}

Config `configs/final/experiments/{cfg.name}` ({len(v)} variants; PLV and |ciPLV| are separate results, never merged):

{variants}

**Protocol:** frozen participant split of ds004504 (52 development / 13 holdout). `xu_dev_cv`: 5-fold
participant-grouped CV inside the development set; `xu_holdout`: fit on all 52 development participants, evaluate
once on the 13 holdout participants. {protocol}. Participant score = mean window score; participant-level metrics
(pooled out-of-fold ROC-AUC, mean fold AUC, balanced accuracy, sensitivity, specificity, confusion matrix, DeLong /
bootstrap / Wilson CIs) are primary. The store's participant QC must have no unresolved category A finding.
{"" if not gnn else "Graph construction: the recorded development-only decision of notebook 03 (`--graph selected`)."}

{HOLDOUT}

**Contexts:** {"sensor space (identical in the LCMV and wMNE stores), window modes B then A" if sensor else
"LCMV and wMNE, window mode B (all accepted windows, primary) then A (10 windows per participant, sensitivity)"}.
Edit `CONTEXTS` below only to run a subset; the order is the recommended order.
""" + HOWTO + (" Use a **GPU runtime** (T4)." if gnn else " CPU is sufficient.")),
        code(SETUP),
        md("## 1. Contexts to run (the only cell to edit, if at all)"),
        code(f'''EXP = '{exp}'
CONTEXTS = {ctx}     # (inverse, window mode), in this order
GRAPH = {"'selected'" if gnn else "None"}
ATLAS = 'dk68'          # the primary atlas of the release (release.PRIMARY); coarser atlases need their own stores
import torch
print('runtime:', ('GPU ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU only')'''),
        md("## 2. Status, then run (resumable; one experiment only)"),
        code('''ERRORS = {}
g = f" --graph {GRAPH}" if GRAPH else ""
for inverse, mode in CONTEXTS:
    inv = 'lcmv' if inverse == 'sensor' else inverse
    print(f'==== {EXP} | {inverse} | mode {mode}')
    sh(f"python scripts/final.py status --exp {EXP} --inverse {inv} --mode {mode} --atlas {ATLAS}{g}")
    try:
        sh(f"python scripts/final.py run --exp {EXP} --inverse {inv} --mode {mode} --atlas {ATLAS}{g}")
    except RuntimeError as exc:
        ERRORS[(inverse, mode)] = exc
        print(f'{inverse} / mode {mode}: some units failed; details after the results')'''),
        md("## 3. Results and diagnostics (saved outputs only; diagnostic figures are not paper figures)"),
        code(SHOW_LATEST + '''
for inverse, mode in CONTEXTS:
    inv = 'lcmv' if inverse == 'sensor' else inverse
    sh(f"python scripts/final.py report --exp {EXP} --inverse {inv} --mode {mode} --atlas {ATLAS}{g}")
    rep = latest(f"{FINAL}/results/{inverse}/mode_{mode}/{ATLAS}/*/*/_reports/report_*")
    if rep:
        print('====', inverse, 'mode', mode, rep)
        t = pd.read_csv(rep + '/results.csv')
        cols = ['variant', 'setting', 'auc', 'auc_ci_low', 'auc_ci_high', 'mean_fold_auc', 'min_fold_auc',
                'max_fold_auc', 'balanced_accuracy', 'sensitivity', 'specificity', 'tn', 'fp', 'fn', 'tp']
        display(t[[c for c in cols if c in t]].round(3))
        for png in sorted(glob.glob(rep + '/*.png')):
            display(Image(filename=png))
if ERRORS:
    raise RuntimeError(f'units did not complete in {sorted(ERRORS)}; completed units are saved. Run all again.')'''),
    ], gpu=gnn)


def report_notebook() -> dict:
    return notebook([
        md(f"""# 99 · Results report (saved outputs only; nothing is trained)

Combined participant-level table of every completed result, the matched LCMV vs wMNE and mode B vs A comparisons
(paired DeLong, descriptive, same participants), and the category C participants (difficult to classify across the
development-CV models; reported with their QC flags, never excluded).

{HOLDOUT}
""" + HOWTO),
        code(SETUP),
        code('''sh("python scripts/final.py report")
sh("python scripts/final.py difficult")'''),
        code(SHOW_LATEST + '''
d = latest(FINAL + '/reports/tables_*')
for f in ('all_results.csv', 'lcmv_vs_wmne.csv', 'mode_B_vs_A.csv'):
    if os.path.exists(f'{d}/{f}'):
        print('====', f)
        display(pd.read_csv(f'{d}/{f}').round(3))'''),
    ])


# ------------------------------------------------------------------------------------------------ index
def readme(entries: list[tuple[str, str, str]]) -> str:
    rows = "\n".join(f"| {i} | [{n}]({COLAB}/{n}) | {d} | {rt} |" for i, (n, d, rt) in enumerate(entries, 1))
    return f"""# Notebooks of the final framework (release {release.RELEASE})

Generated by `scripts/make_notebooks.py`; never edit by hand. Run them **in this order**, one at a time, each with
*Runtime → Run all* (details, configurations and expected run times: `FINAL_RUN_GUIDE.md`).

| # | Notebook | What it does | Runtime |
|---|---|---|---|
{rows}

All outputs go to `<outputs>/{release.ROOT}/` on Google Drive (layout in `FINAL_RUN_GUIDE.md`). The validated stores
(`D3_LCMV_XU/graphsLX*_corr1`, `D3_WMNE_XU/graphsWX*`) and every earlier result are only read. The notebooks of the
earlier suites are archived in `legacy/notebooks/` and are not part of this release.
"""


def main() -> None:
    out = REPO / "notebooks"
    out.mkdir(exist_ok=True)
    nbs = {"00_setup_and_stores.ipynb": (setup_notebook(), "verify and lock the four stores; test suite", "CPU"),
           "01_participant_qc.ipynb": (qc_notebook(), "participant QC; category A decisions", "CPU"),
           "02_parcellation.ipynb": (parcellation_notebook(), "source resolution per atlas; recommendation; "
                                                              "feature reliability", "CPU"),
           "03_graph_construction.ipynb": (graph_notebook(), "graph diagnostics; development-CV comparison; "
                                                             "selection", "GPU")}
    for exp, cfg in experiment_configs().items():
        name, title, gnn = EXP_TITLES[exp]
        nbs[f"{name}.ipynb"] = (experiment_notebook(exp, cfg), title, "GPU" if gnn else "CPU")
    nbs["99_results_report.ipynb"] = (report_notebook(), "combined tables, comparisons, difficult participants", "CPU")
    order = [s["notebook"] for s in run_plan()["steps"]]
    if sorted(order) != sorted(nbs):
        raise SystemExit(f"configs/final/run_plan.yaml and the generated notebooks differ: {set(order) ^ set(nbs)}")
    nbs = {n: nbs[n] for n in order}
    for name, (nb, _, _) in nbs.items():
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(readme([(n, d, rt) for n, (_, d, rt) in nbs.items()]))
    print(f"{len(nbs)} notebooks -> {out}")


if __name__ == "__main__":
    main()
