"""Generate the Colab notebooks of the final framework from one template (keeps them identical and thin).

    python scripts/make_notebooks.py          # rewrite notebooks/ ; tests check they are up to date

    notebooks/00_setup_and_stores.ipynb        verify and lock the four validated stores (CPU)
    notebooks/01_participant_qc.ipynb          participant QC of every store; category A decisions (CPU)
    notebooks/02_parcellation.ipynb            source resolution, atlas checks, lobes12 / r18 stores (authorized, validated,
                                               matched), QC
    notebooks/04_roi_68_vs_18.ipynb            controlled dk68 vs lobes12 vs r18 comparison (development CV); stops
    notebooks/03_graph_construction.ipynb      graph diagnostics + the development-CV comparison on that atlas (GPU)
    notebooks/Exp<k>_*.ipynb                   ONE experiment each, Exp 0-8, development CV (GPU for 2-8)
    notebooks/Exp8a_xu_dual_gcn_surface.ipynb  amendment A2: Exp 8 with a surface anatomical layer (own pin, namespace)
    notebooks/99_results_report.ipynb          freeze the final analysis, descriptive holdout, final tables (GPU)
    notebooks/README.md                        index, Colab links, execution order, Drive layout
    notebooks/cross_dataset/*.ipynb            the cross-dataset experiment D3 <-> D2 (own branch, namespace, pin)

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

from adgraph.colab import sh as _sh

def sh(cmd, cwd=None):  # every script runs with the file backend: Colab's inline backend breaks figures in scripts
    os.environ['MPLBACKEND'] = 'Agg'
    return _sh(cmd, cwd)

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


SHOW_LATEST = '''import glob, json
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


QC_BODY = """for inverse in ('lcmv', 'wmne'):
    for mode in ('B', 'A'):
        try:
            sh(f"python scripts/final.py qc --inverse {inverse} --mode {mode} --atlas {ATLAS}")
        except RuntimeError:
            UNRESOLVED.append((ATLAS, inverse, mode))
for d in sorted(glob.glob(FINAL + f'/qc/*_*_{ATLAS}/')):
    if os.path.exists(d + 'qc_report.md'):
        display(Markdown(open(d + 'qc_report.md').read()))"""


def parcellation_notebook() -> dict:
    from adgraph.experiments import roi_comparison as rc

    new = [a for a in rc.config()["atlases"] if a != rc.config()["reference"]]
    return notebook([
        md(f"""# 02 · Parcellation: source resolution, then the {' and '.join(new)} stores (CPU)

1. **Anatomy** of {' and '.join(new)} (no EEG): each hemisphere's 34 Desikan-Killiany labels partitioned, no
   cross-hemisphere region, identical left / right definitions, every region one contiguous piece of cortex, correct
   hemispheric placement, non-empty source sets.
2. **Analytic source resolution** per atlas, both inverses, with the stores' exact fsaverage BEM and inverse settings,
   and the pre-specified `atlas_resolution` rule (`configs/final/selection.yaml`, unchanged). Its verdict is
   authoritative and is never altered: with the exact BEM, lobes12 is rule-recommended and r18 fails the LCMV
   own-region dominance criterion (0.778 < 0.80).
3. **The 68-ROI stores' own operators** (real data, no labels; the corrected LCMV store reads the verified original
   checkpoints), feature reliability, reconstruction check.
4. **Authorization, then the new stores.** An atlas is reconstructed only if its anatomical checks passed AND it is
   rule-eligible (lobes12) OR the author's post-result approval of an exploratory atlas is recorded (r18: amendment
   A1, `configs/final/amendments.yaml`; set `EXPLORATORY_R18_APPROVAL = 'A1'` below to record it). The build refuses
   otherwise (`build_stores.py build`). Each store is built from the ORIGINAL recordings by the unchanged pipeline
   (merged Desikan-Killiany labels, `mean_flip` over their own sources; never averages of 68-ROI signals or PLV),
   LCMV and wMNE, window modes B and A; validated, wMNE matched to LCMV, and each matched window by window to its
   68-ROI store (participants, labels, windows, split, bands, sampling, sensor features).
5. **QC** of every new store (category A stops here), then the resolution and reliability of the operators built.

The 68-ROI stores are only read. No analysis atlas is selected here. Resumable (participant checkpoints). CPU,
about 1.5 h per atlas for the builds plus 30-60 min of diagnostics.
""" + HOWTO),
        code(SETUP),
        code(SHOW_LATEST),
        md("## 1. Anatomical checks"),
        code(f'''NEW_ATLASES = {json.dumps(new)}
ATLAS_OK = {{}}
for atlas in NEW_ATLASES:
    try:
        sh(f"python scripts/final.py atlas-check --atlas {{atlas}}")
        ATLAS_OK[atlas] = True
    except RuntimeError:
        ATLAS_OK[atlas] = False
print(ATLAS_OK)'''),
        md("## 2. Analytic source resolution and the rule's recommendation (stores' BEM; unchanged)"),
        code('sh("python scripts/final.py parcellation")'),
        code('''d = latest(FINAL + '/parcellation/resolution_*')
display(pd.read_csv(d + '/resolution_by_atlas.csv').round(3))
rec = json.load(open(d + '/recommendation.json'))
print('rule-recommended atlas:', rec['recommended'], '|', rec['forward_model'])
for v in rec['verdicts']:
    print(v['atlas'], 'eligible' if v['eligible'] else 'NOT eligible', '; '.join(v['reasons']))
if 'approx' in rec['forward_model'] or 'sphere' in rec['forward_model']:
    raise RuntimeError("the stores' BEM was not found: an approximate head model cannot authorize a reconstruction")'''),
        md("## 3. The 68-ROI stores' own operators, feature reliability, reconstruction check (read-only)"),
        code('''for inverse in ('lcmv', 'wmne'):
    sh(f"python scripts/final.py store-resolution --inverse {inverse} --atlas dk68")
    sh(f"python scripts/final.py reliability --inverse {inverse} --mode B")
    sh(f"python scripts/final.py reconstruct-check --inverse {inverse} --mode B")'''),
        md("""## 4. Authorization, then the new stores from the original recordings

`EXPLORATORY_R18_APPROVAL`: the author's post-result approval of the exploratory r18 arm. Leave `None` to build only
the rule-eligible atlas; set `'A1'` to record amendment A1 (once, in `FINAL_v1/decisions/exploratory_r18.json`; the
rule verdict is copied into it unchanged). Builds are resumable; `validation.json`, `match.json` and
`atlas_match.json` are written once. New directories next to the 68-ROI stores (outside `FINAL_v1/`), e.g.
`D3_LCMV_XU/graphsLX-lobes12_v1_<hash>`, `graphsLX10-lobes12_v1_<hash>`, `D3_WMNE_XU/graphsWX-r18_v1_<hash>`."""),
        code('''EXPLORATORY_R18_APPROVAL = None   # set to 'A1' to record the author's approval (amendment A1, 2026-10-19)
import shlex
if EXPLORATORY_R18_APPROVAL:
    sh(f"python scripts/final.py approve-exploratory --atlas r18 --amendment {shlex.quote(EXPLORATORY_R18_APPROVAL)}")
BUILD = []
for atlas in NEW_ATLASES:
    try:
        sh(f"python scripts/final.py authorize --atlas {atlas}")
        BUILD.append(atlas)
    except RuntimeError:
        print(f'{atlas}: not built (not authorized; see above)')
if not BUILD:
    raise RuntimeError('no atlas is authorized for reconstruction; see the messages above')
for atlas in BUILD:
    for inverse in ('lcmv', 'wmne'):
        sh(f"python scripts/build_stores.py build --inverse {inverse} --atlas {atlas}")
        sh(f"python scripts/build_stores.py validate --inverse {inverse} --atlas {atlas}")
    sh(f"python scripts/build_stores.py match --atlas {atlas}")
    for inverse in ('lcmv', 'wmne'):
        sh(f"python scripts/build_stores.py match-atlas --inverse {inverse} --atlas {atlas}")
        sh(f"python scripts/build_stores.py size --inverse {inverse} --atlas {atlas}")
    sh(f"python scripts/final.py stores --atlas {atlas}")   # all four stores verify (locked when QC opens them)'''),
        md("## 5. QC of the new stores, then the resolution and reliability of the operators actually built"),
        code('''UNRESOLVED = []
from IPython.display import Markdown
for ATLAS in BUILD:
''' + "\n".join("    " + ln for ln in QC_BODY.splitlines()) + '''
if UNRESOLVED:
    raise RuntimeError(f'category A findings need a decision for {UNRESOLVED}: see the reports above and '
                       'docs/final/qc.md; nothing proceeds until then')'''),
        code('''for atlas in BUILD:
    for inverse in ('lcmv', 'wmne'):
        sh(f"python scripts/final.py store-resolution --inverse {inverse} --atlas {atlas}")
        sh(f"python scripts/final.py reliability --inverse {inverse} --mode B --atlas {atlas}")
for inverse in ('lcmv', 'wmne'):
    for atlas in ['dk68'] + BUILD:
        d = latest(f"{FINAL}/parcellation/store_resolution_{inverse}_{atlas}_*")
        t = pd.read_csv(d + '/per_participant.csv')
        print(inverse, atlas, t[['operator_rank', 'redundant_signals', 'crosstalk_fraction_median',
                                 'rois_dominated_by_own_region', 'leakage_corr_abs_median']].median().round(3).to_dict())'''),
    ])


def roi_notebook() -> dict:
    from adgraph.experiments import roi_comparison as rc

    cfg = rc.config()
    runs = [(r["exp"], r["variant"], r["atlas"], r["inverse"], r["graph"]) for r in rc.runs(cfg)]
    models = "\n".join(f"- `{e}`: " + ", ".join(f"`{v}`" for v in vs) for e, vs in cfg["models"].items())
    roles = "\n".join(f"- **{a}**: {cfg['roles'][a]}" for a in cfg["atlases"])
    comps = ", ".join(f"`{a} - {b}`" for a, b in cfg["comparisons"])
    return notebook([
        md(f"""# 04 · Controlled atlas comparison: {' vs '.join(cfg['atlases'])} (development CV only; exploratory)

`configs/final/roi_comparison.yaml`.

{roles}

The same 52 development participants, the same five participant-disjoint folds (`xu_dev_cv`), window mode
{cfg['mode']}, the same training procedures and seeds, LCMV and wMNE, for:

{models}

GNNs on dense weighted PLV, exactly 60 epochs. **The holdout is not evaluated** (the code refuses it until the final
analysis is frozen). Paired differences on the same participants: {comps}.

The report keeps **source localization / spatial resolution** (label-free; notebook 02), **classification per
atlas** (participant-level pooled out-of-fold ROC-AUC with DeLong CI, balanced accuracy, F1, sensitivity,
specificity) and **paired differences** (paired DeLong, participant-bootstrap CIs) in separate sections. All
comparisons are exploratory and not corrected for multiplicity; a higher AUC alone does not make an atlas better,
and the summary never forces a winner. **This notebook stops after the report: no atlas is selected** (no
`decisions/atlas.json` is written); the next stage is your decision.

The dk68 runs are the same units the experiment notebooks reuse (nothing is computed twice). An atlas whose stores
are not authorized / built (e.g. r18 without the recorded approval) is skipped and its comparisons are reported as
incomplete. GPU runtime (T4). Resumable.
""" + HOWTO),
        code(SETUP),
        code(SHOW_LATEST),
        md("## 1. Stores that take part (read-only verification)"),
        code(f'''ATLASES = []
for atlas in {json.dumps(cfg['atlases'])}:
    try:
        sh(f"python scripts/final.py stores --atlas {{atlas}}")
        ATLASES.append(atlas)
    except RuntimeError:
        if atlas == '{cfg['reference']}':
            raise
        print(f'{{atlas}}: stores missing or not verified: skipped (its comparisons will be incomplete)')
print('atlases compared:', ATLASES)'''),
        md("## 2. Runs (development CV only; each unit is checkpointed)"),
        code(f'''RUNS = {json.dumps(runs)}      # (experiment, variant, atlas, inverse, graph)
ERRORS = []
for exp, variant, atlas, inverse, graph in RUNS:
    if atlas not in ATLASES:
        continue
    g = '' if graph == 'none' else f' --graph {{graph}}'
    try:
        sh(f"python scripts/final.py run --exp {{exp}} --inverse {{inverse}} --mode {cfg['mode']} --atlas {{atlas}}{{g}} "
           f"--variants {{variant}}")
    except RuntimeError:
        ERRORS.append((exp, variant, atlas, inverse))
if ERRORS:
    raise RuntimeError(f'units did not complete: {{ERRORS}}; completed units are saved. Run all again.')'''),
        md("## 3. The comparison (saved outputs only)"),
        code('''try:
    sh("python scripts/final.py roi-compare")
except RuntimeError:
    print('some comparisons are incomplete (see above); the complete ones are reported')
d = latest(FINAL + '/roi_comparison/compare_*')
from IPython.display import Markdown
if d:
    display(Markdown(open(d + '/report.md').read()))'''),
        md("""## Stop here

The comparison is complete for review. Nothing has been selected: no analysis atlas is recorded, notebook 03 and the
experiment notebooks are not started. The next stage is the author's explicit decision (FINAL_RUN_GUIDE.md §3.1)."""),
    ], gpu=True)


def graph_notebook() -> dict:
    import yaml

    rule = yaml.safe_load((REPO / "configs" / "final" / "selection.yaml").read_text())["graph_construction"]
    cons = [rule["default"]] + rule["candidates"] + rule["reported_only"]
    refs = [(exp, var) for exp, vs in rule["reference"].items() for var in vs]
    return notebook([
        md(f"""# 03 · Graph construction on the recorded atlas: diagnostics and the pre-specified comparison

Runs on the analysis atlas you recorded in notebook 04 (`--atlas selected`; it stops if none is recorded); k scales
with that atlas's ROI count (k = round(f (R - 1)): 7 / 13 / 20 for 68 ROIs, 2 / 3 / 5 for 18).

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
   recorded once (`{release.ROOT}/decisions/graph_construction.json`, with its atlas) and used by Exp 2-8
   (`--graph selected`). It is a pre-specified exploratory decision rule on development CV, not a test of
   statistical superiority: read the per-cell AUCs with their confidence intervals in the result tables.

{HOLDOUT}

GPU runtime (T4). Resumable.
""" + HOWTO),
        code(SETUP),
        md("## 1. Diagnostics"),
        code('''for inverse in ('lcmv', 'wmne'):
    sh(f"python scripts/final.py graphs --inverse {inverse} --mode B --atlas selected")'''),
        md("## 2. Comparison runs (development CV only)"),
        code(f'''CONSTRUCTIONS = {json.dumps(cons)}
REFERENCE = {json.dumps(refs)}
ERRORS = []
for graph in CONSTRUCTIONS:
    for inverse in {json.dumps(rule['inverses'])}:
        for exp, variant in REFERENCE:
            try:
                sh(f"python scripts/final.py run --exp {{exp.split('_')[0]}} --inverse {{inverse}} --mode {rule['mode']} "
                   f"--atlas selected --graph {{graph}} --variants {{variant}}")
            except RuntimeError:
                ERRORS.append((graph, inverse, exp, variant))'''),
        md("## 3. Selection (development CV only)"),
        code('''sh("python scripts/final.py report")
if ERRORS:
    raise RuntimeError(f'units did not complete: {ERRORS}; completed units are saved. Run all again.')
sh("python scripts/final.py select --atlas selected")'''),
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


AMENDED = {"exp8": """
**Amendment A2 (lobes12):** with the 12-lobe atlas this experiment's anatomical layer (`lobe_hemisphere`) is empty
and every unit stops before training. Use `Exp8a_xu_dual_gcn_surface` instead (cortical-surface anatomical layer, own
code pin and namespace; methodology log entry 60). This notebook is kept unchanged for the record.
"""}


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

**Protocol:** frozen participant split of ds004504 (52 development / 13 holdout). This notebook runs `xu_dev_cv`
only: 5-fold participant-grouped CV inside the development set. The holdout (`xu_holdout`: fit on all 52
development participants, evaluate once on the 13 holdout participants) is NOT run here: it is evaluated once, for
the frozen final analysis only, in notebook 99. {protocol}. Participant score = mean window score; participant-level metrics
(pooled out-of-fold ROC-AUC, mean fold AUC, balanced accuracy, sensitivity, specificity, confusion matrix, DeLong /
bootstrap / Wilson CIs) are primary. The store's participant QC must have no unresolved category A finding.
Atlas: the analysis atlas recorded in notebook 04 (`--atlas selected`{"; Exp 0 is sensor space" if sensor else ""}).
{"" if not gnn else "Graph construction: the recorded development-only decision of notebook 03 (`--graph selected`)."}

{HOLDOUT}

**Contexts:** {"sensor space (identical in the LCMV and wMNE stores), window modes B then A" if sensor else
"LCMV and wMNE, window mode B (all accepted windows, primary) then A (10 windows per participant, sensitivity)"}.
Edit `CONTEXTS` below only to run a subset; the order is the recommended order.
{AMENDED.get(exp, "")}""" + HOWTO + (" Use a **GPU runtime** (T4)." if gnn else " CPU is sufficient.")),
        code(SETUP),
        md("## 1. Contexts to run (the only cell to edit, if at all)"),
        code(f'''EXP = '{exp}'
CONTEXTS = {ctx}     # (inverse, window mode), in this order
GRAPH = {"'selected'" if gnn else "None"}
ATLAS = 'selected'      # the analysis atlas recorded in notebook 04 (FINAL_v1/decisions/atlas.json)
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
    atlas = 'dk68' if inverse == 'sensor' else json.load(open(FINAL + '/decisions/atlas.json'))['atlas']
    rep = latest(f"{FINAL}/results/{inverse}/mode_{mode}/{atlas}/*/*/_reports/report_*")
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


def amendment_setup(aid: str) -> str:
    """The setup cell of an amendment notebook: the amendment's own code pin (never the release pin)."""
    from adgraph.experiments import amendment

    s = SETUP.replace(
        f"UPDATE_CODE = False   # leave False. True only to move the release code pin deliberately to the newest commit\n"
        f"                      # (<outputs>/{release.PIN_FILE}; unfinished work is then recomputed).",
        f"UPDATE_CODE = False   # leave False. True only to move THIS AMENDMENT's code pin deliberately\n"
        f"                      # (<outputs>/{amendment.pin_file(aid)}; the release pin is never used or moved here).")
    s = s.replace("_run([sys.executable, 'scripts/colab_setup.py'] + (['--update-code'] if UPDATE_CODE else []))",
                  f"_run([sys.executable, 'scripts/colab_setup.py', '--amendment', '{aid}']"
                  " + (['--update-code'] if UPDATE_CODE else []))")
    return s + f"A2_ROOT = FINAL + '/amendments/{amendment.NAMESPACES[aid]}'\n"


def amendment_exp8_notebook() -> dict:
    from adgraph.experiments import amendment

    aid = "A2"
    cfg = amendment.experiment_configs(aid)["exp8a"]
    v = expand_variants(load_experiment(cfg))
    name = v[0]["experiment"]
    plan = run_plan()["contexts"]["source"]
    ctx = "[" + ", ".join(f"('{i}', '{m}')" for i, m in plan) + "]"
    variants = "\n".join(f"- `{x['variant']}`: layer 1 {x['model']['layer1']}, layer 2 {x['model']['layer2']}"
                         + (f" ({x['model']['anatomical_grouping']})" if 'anatomical' in (x['model']['layer1'],
                                                                                       x['model']['layer2']) else "")
                         + ("" if x['model']['band_specific'] else ", weights shared across bands") for x in v)
    return notebook([
        md(f"""# Exp8a · Xu-inspired dual-adjacency GCN with a cortical-surface anatomical layer (amendment A2)

**Why this notebook exists.** On the recorded analysis atlas (lobes12) the original Exp 8 anatomical layer
(`lobe_hemisphere`: regions of the same lobe and hemisphere) is the zero matrix, because the 12 regions ARE the
lobe x hemisphere groups; every original Exp 8 unit failed before training. Amendment A2 (post-design, pre-result;
methodology log entry 60) redefines layer 2 as the fixed **cortical-surface contiguity** of the atlas regions
(fsaverage mesh, within a hemisphere): a data- and label-independent anatomical-neighbourhood prior, not measured
white-matter connectivity. Layer 1 is unchanged: the window's frequency-specific PLV or |ciPLV|. This is an
adaptation inspired by Xu et al. (2025), not a reproduction.

Config `configs/final/amendments/{cfg.name}` (`{name}`, {len(v)} variants; `alt_bilateral_lobes` is an alternative
anatomical prior: each lobe linked to its contralateral homologue):

{variants}

**Isolation.** This notebook checks out the amendment's own code pin
(`<outputs>/{amendment.pin_file(aid)}`) and writes only to `<outputs>/{release.ROOT}/amendments/{amendment.NAMESPACES[aid]}/`.
The release pin, Exp 0-7, their results and the failed / partial units of the original Exp 8 are not touched or read
as checkpoints. Stores, QC reports and the atlas / graph decisions are read only. Same protocol (`gnn_fixed60_v1`),
folds and seeds; development CV only (the holdout is not run here).

{HOLDOUT}
""" + HOWTO + " Use a **GPU runtime** (T4)."),
        code(amendment_setup(aid)),
        md("""## 1. Numerical environment (exactly that of the completed Exp 7 units)

Exp 8a is compared with Exp 0-7, so it runs with the Python and library versions recorded in the code identity of a
completed Exp 7 unit (python, numpy, scipy, pandas, scikit-learn, torch). Only these versions are compared: the git
commit and source hash of Exp 8a are its own and are NOT expected to match Exp 7. If this Colab kernel differs, the
environment is restored automatically in `/content/restore<python digits>` (e.g. `/content/restore31315`; built with
`uv`, a few minutes, on the temporary Colab disk only) and verified; every `scripts/final.py` call below uses that
interpreter (`PY`). Leave `ACCEPT_ENVIRONMENT_DIFFERENCE = False`: a difference that cannot be removed stops here."""),
        code(f'''ACCEPT_ENVIRONMENT_DIFFERENCE = False   # True only after reviewing a reported difference (it is then recorded)
from adgraph import numerical_env
REFERENCE = numerical_env.reference_libraries(OUT, 'exp7_earm_gnn')   # recorded by a completed Exp 7 unit
PY = numerical_env.ensure_interpreter(REFERENCE, repo='/content/repo', base='/content',
                                      accept_difference=ACCEPT_ENVIRONMENT_DIFFERENCE, run=sh)
NOW = numerical_env.interpreter_versions(PY, REFERENCE)
DIFF = numerical_env.differences(REFERENCE, NOW)
assert ACCEPT_ENVIRONMENT_DIFFERENCE or not DIFF, f'numerical environment differs from Exp 7: {{DIFF}}'
print('scripts run with', PY, NOW)'''),
        md("## 2. Amendment check: pins, namespace, decisions, stores and QC (nothing is trained)"),
        code(f'''sh(f"{{PY}} scripts/final.py amendment-check --amendment {aid} --atlas selected"
   + (" --accept-environment-difference" if ACCEPT_ENVIRONMENT_DIFFERENCE else ""), cwd='/content/repo')

import json, os, subprocess, yaml
def _pin(path):
    return next(ln.strip() for ln in open(path) if ln.strip() and not ln.startswith('#'))
RECORD = yaml.safe_load(open('/content/repo/configs/final/amendments.yaml'))['{aid}']
MANIFEST = json.load(open(A2_ROOT + '/manifest.json'))
RELEASE_PIN_FILE, A2_PIN_FILE = FINAL + '/CODE_COMMIT.txt', A2_ROOT + '/CODE_COMMIT.txt'
HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='/content/repo', text=True).strip()
ATLAS_NAME = json.load(open(FINAL + '/decisions/atlas.json'))['atlas']
GRAPH_NAME = json.load(open(FINAL + '/decisions/graph_construction.json'))['selected']
assert MANIFEST['amendment'] == '{aid}' and RECORD['namespace'].endswith('{amendment.NAMESPACES[aid]}'), 'not amendment {aid}'
assert ATLAS_NAME == 'lobes12', f'recorded atlas is {{ATLAS_NAME}}, amendment {aid} is defined for lobes12'
assert GRAPH_NAME == 'dense', f'recorded graph construction is {{GRAPH_NAME}}, expected dense'
assert os.path.realpath(A2_PIN_FILE) != os.path.realpath(RELEASE_PIN_FILE), 'pins must be separate files'
assert _pin(RELEASE_PIN_FILE) == RECORD['base_release_pin'] == MANIFEST['base_release_pin'], 'release pin moved'
assert _pin(A2_PIN_FILE) == HEAD == MANIFEST['amendment_commit'], 'the code is not the amendment pin'
assert _pin(A2_PIN_FILE) != _pin(RELEASE_PIN_FILE), 'the amendment must not run at the release pin'
assert ACCEPT_ENVIRONMENT_DIFFERENCE or MANIFEST['environment_ok'], MANIFEST['library_differences']
print('release pin  ', _pin(RELEASE_PIN_FILE), RELEASE_PIN_FILE)
print('amendment pin', _pin(A2_PIN_FILE), A2_PIN_FILE)
print('atlas', ATLAS_NAME, '| graph', GRAPH_NAME, '| namespace', A2_ROOT)'''),
        md("## 3. The anatomical layer of the recorded atlas (fixed; from the atlas only)"),
        code('''import numpy as np, pandas as pd
from adgraph.models.xu_dual import anatomical_adjacency
from adgraph.source.atlas import region_names
names = region_names(ATLAS_NAME)
A = anatomical_adjacency(names, 'surface_contiguity')
print(ATLAS_NAME, len(names), 'regions,', int(A.sum() // 2), 'surface edges, symmetric', bool((A == A.T).all()))
display(pd.DataFrame(A.astype(int), index=names, columns=names))'''),
        md("## 4. Contexts, status, run (resumable; this amendment experiment only)"),
        code(f'''CONTEXTS = {ctx}     # (inverse, window mode), in this order
ERRORS = {{}}
for inverse, mode in CONTEXTS:
    print(f'==== exp8a | {{inverse}} | mode {{mode}}')
    sh(f"{{PY}} scripts/final.py status --amendment {aid} --exp exp8a --inverse {{inverse}} --mode {{mode}} "
       f"--atlas selected --graph selected", cwd='/content/repo')
    try:
        sh(f"{{PY}} scripts/final.py run --amendment {aid} --exp exp8a --inverse {{inverse}} --mode {{mode}} "
           f"--atlas selected --graph selected", cwd='/content/repo')
    except RuntimeError as exc:
        ERRORS[(inverse, mode)] = exc
        print(f'{{inverse}} / mode {{mode}}: some units failed; details after the results')'''),
        md("## 5. Results and diagnostics of THIS amendment only (saved outputs; diagnostic figures)"),
        code(SHOW_LATEST + f'''
EXPECTED_VARIANTS = {sorted(x["variant"] for x in v)!r}
assert len(EXPECTED_VARIANTS) == {len(v)}
for inverse, mode in CONTEXTS:
    sh(f"{{PY}} scripts/final.py report --amendment {aid} --exp exp8a --inverse {{inverse}} --mode {{mode}} "
       f"--atlas selected --graph selected", cwd='/content/repo')
    rep = latest(f"{{A2_ROOT}}/results/{{inverse}}/mode_{{mode}}/{{ATLAS_NAME}}/{{GRAPH_NAME}}/{name}/_reports/report_*")
    if (inverse, mode) in ERRORS:
        continue
    assert rep, f'no report of {name} for {{inverse}} / mode {{mode}} under {{A2_ROOT}}'
    print('====', inverse, 'mode', mode, rep)
    t = pd.read_csv(rep + '/results.csv')
    assert sorted(t['variant'].unique()) == EXPECTED_VARIANTS, sorted(t['variant'].unique())
    assert set(t['setting']) == {{'xu_dev_cv'}}, 'development CV only'
    cols = ['variant', 'setting', 'auc', 'auc_ci_low', 'auc_ci_high', 'mean_fold_auc', 'min_fold_auc',
            'max_fold_auc', 'balanced_accuracy', 'sensitivity', 'specificity', 'tn', 'fp', 'fn', 'tp']
    display(t[[c for c in cols if c in t]].round(3))
    for png in sorted(glob.glob(rep + '/*.png')):
        display(Image(filename=png))
assert _pin(RELEASE_PIN_FILE) == RECORD['base_release_pin'], 'release pin moved'
if ERRORS:
    raise RuntimeError(f'units did not complete in {{sorted(ERRORS)}}; completed units are saved. Run all again.')'''),
    ], gpu=True)


def report_notebook() -> dict:
    from adgraph.experiments.decisions import holdout_plan

    plan = holdout_plan()
    return notebook([
        md(f"""# 99 · Final analysis: development-CV report, freeze, descriptive holdout, final tables

1. **Development-CV report** of every completed result and the category C participants (difficult to classify
   across the development-CV models; reported with their QC flags, never excluded).
2. **Freeze** the final analysis (`configs/final/holdout.yaml`): every pre-specified variant of
   {', '.join(plan['experiments'])}, window mode {plan['mode']}, {' and '.join(plan['inverses'])}, on the recorded atlas
   and graph construction. It refuses while any of these units lacks a completed development-CV result or a decision
   is missing, and it is written once (`{release.ROOT}/decisions/final_analysis.json`, with the code commit).
   Exp 7 (exploratory), mode A, the atlas not chosen and the graph constructions not selected stay development-only.
3. **Holdout**: each frozen unit is fitted on the 52 development participants and evaluated once on the 13 holdout
   participants. The code refuses the holdout for any unit outside the frozen analysis. Descriptive only.
4. **Final tables**: all results, the matched LCMV vs wMNE and mode B vs A comparisons (paired DeLong, descriptive).

{HOLDOUT}

GPU runtime (T4) for step 3; resumable.
""" + HOWTO),
        code(SETUP),
        md("## 1. Development-CV report and difficult participants"),
        code('''sh("python scripts/final.py report")
sh("python scripts/final.py difficult")'''),
        md("## 2. Freeze the final analysis (once)"),
        code('sh("python scripts/final.py freeze")'),
        md("## 3. Holdout of the frozen units only (descriptive)"),
        code(f'''ERRORS = []
for exp in {json.dumps(plan['experiments'])}:
    try:
        sh(f"python scripts/final.py holdout --exp {{exp}}")
    except RuntimeError:
        ERRORS.append(exp)
if ERRORS:
    raise RuntimeError(f'holdout units did not complete for {{ERRORS}}; completed units are saved. Run all again.')'''),
        md("## 4. Final tables (holdout rows are descriptive only)"),
        code(SHOW_LATEST + '''
sh("python scripts/final.py report")
d = latest(FINAL + '/reports/tables_*')
for f in ('all_results.csv', 'lcmv_vs_wmne.csv', 'mode_B_vs_A.csv'):
    if os.path.exists(f'{d}/{f}'):
        print('====', f)
        display(pd.read_csv(f'{d}/{f}').round(3))'''),
    ], gpu=True)


# ------------------------------------------------------------------------------------------------ cross-dataset
CROSS_BRANCH = "claude/cross-dataset-d3-d2-19ch"
CROSS_COLAB = f"https://colab.research.google.com/github/{OWNER_REPO}/blob/{CROSS_BRANCH}/notebooks/cross_dataset"


def cross_setup() -> str:
    """Setup cell of the cross-dataset notebooks: their own branch and code pin (CROSS_D3_D2_19CH_v1/CODE_COMMIT.txt);
    the release pin and the amendment A2 pin are never used or moved."""
    from adgraph.crossdataset import namespace as ns

    s = SETUP.replace(f"BRANCH = '{BRANCH}'", f"BRANCH = '{CROSS_BRANCH}'")
    s = s.replace(
        f"UPDATE_CODE = False   # leave False. True only to move the release code pin deliberately to the newest commit\n"
        f"                      # (<outputs>/{release.PIN_FILE}; unfinished work is then recomputed).",
        f"UPDATE_CODE = False   # leave False. True only to move THIS EXPERIMENT's code pin deliberately\n"
        f"                      # (<outputs>/{ns.pin_file()}; the release and A2 pins are never used or moved here).")
    s = s.replace("_run([sys.executable, 'scripts/colab_setup.py'] + (['--update-code'] if UPDATE_CODE else []))",
                  "_run([sys.executable, 'scripts/colab_setup.py', '--namespace', 'cross']"
                  " + (['--update-code'] if UPDATE_CODE else []))")
    s = s.replace("print('release outputs on Drive:', FINAL)", "print('release outputs (READ ONLY here):', FINAL)")
    return s + f"CROSS = OUT + '/{ns.ROOT}'\nprint('cross-dataset outputs (the only place this notebook writes):', CROSS)\n"


CROSS_ENV = """## Numerical environment (the Python / library versions of the completed Exp 7 units)

The cross-dataset fits must be comparable with Exp 0-8a, so every script runs with the interpreter that reproduces the
versions recorded by a completed Exp 7 unit (python, numpy, scipy, pandas, scikit-learn, torch); if this kernel
differs, `/content/restore<python digits>` is built with `uv` and verified (temporary Colab disk only). Leave
`ACCEPT_ENVIRONMENT_DIFFERENCE = False`."""

CROSS_ENV_CODE = """ACCEPT_ENVIRONMENT_DIFFERENCE = False
from adgraph import numerical_env
REFERENCE = numerical_env.reference_libraries(OUT, 'exp7_earm_gnn')
PY = numerical_env.ensure_interpreter(REFERENCE, repo='/content/repo', base='/content',
                                      accept_difference=ACCEPT_ENVIRONMENT_DIFFERENCE, run=sh)
DIFF = numerical_env.differences(REFERENCE, numerical_env.interpreter_versions(PY, REFERENCE))
assert ACCEPT_ENVIRONMENT_DIFFERENCE or not DIFF, f'numerical environment differs from Exp 7: {DIFF}'
def cross(args):   # every step of this experiment: scripts/cross_dataset.py with the verified interpreter
    sh(f"{PY} scripts/cross_dataset.py {args}", cwd='/content/repo')"""

CROSS_HOWTO = """
**How to use:** *Runtime → Run all*. Allow Google Drive access when asked. If Colab disconnects, open the notebook
again and *Run all* again: finished work is kept on Google Drive and skipped. Everything this notebook writes goes to
`<outputs>/CROSS_D3_D2_19CH_v1/` on Drive (never to the temporary Colab disk, and never into `FINAL_v1/`, the
amendment A2 folder or the D3 stores, which are only read).
"""

CROSS_RULES = """**Rules of this experiment** (docs/cross_dataset/design.md): outputs only in `<outputs>/CROSS_D3_D2_19CH_v1/`;
the release (`FINAL_v1/`), amendment A2 (running Exp 8a), the D3 stores, their lock, pins and decisions are only
read. Source-only training; the target is BLINDED (labels masked) when it is scored; target labels are read only by
the evaluation in notebook 03, after the campaign is frozen. Nothing is selected with target results."""


def cross_audit_notebook() -> dict:
    return notebook([
        md(f"""# Cross-dataset 00 · Audit, matched-channel D2 stores, compatibility, freeze

{CROSS_RULES}

**Part A (no authorization needed; nothing is built):** the verified 19-electrode mapping (D2 EDF channel number ->
D3 label, T3=T7, T4=T8, T5=P7, T6=P8), the layout file hash, the shared head geometry, the D2 EDF headers (technical
fields only), the D3 stores (read only), the inherited atlas / graph decisions, and a recording-level inspection of
D2 (durations, selection, candidate windows; no source model, no graph).

**Part B (only with the author's written authorization):** the matched-channel D2 stores (LCMV and wMNE, modes B and
A, lobes12) from the 23 original resting-state EDFs, their validation, the compatibility report, the label-blind
shift diagnostics and D2 QC, and the harmonization manifest.

**Part C (author):** the frozen campaign (configs/cross_dataset/campaign.yaml) is recorded before any training.
""" + CROSS_HOWTO + " A CPU runtime is enough."),
        code(cross_setup()),
        md(CROSS_ENV),
        code(CROSS_ENV_CODE),
        md("## Part A1 · Mapping, layout, geometry, EDF headers, D3 stores, inherited decisions (read only)"),
        code(SHOW_LATEST + """
cross('audit')
display(pd.read_csv(CROSS + '/audit/channel_mapping_audit.csv'))
display(pd.read_csv(CROSS + '/audit/d2_edf_headers.csv'))
print('D2 EDF annotation texts (recording-condition evidence, decision D6):',
      json.load(open(CROSS + '/audit/audit.json'))['d2_edf_annotations'])
display(pd.read_csv(CROSS + '/audit/demographics.csv').round(1))          # age / sex by diagnosis (descriptive)
display(pd.read_csv(CROSS + '/audit/age_overlap_feasibility.csv'))
print(json.dumps(json.load(open(CROSS + '/audit/d2_filter_response.json'))['magnitude_db'], indent=1))"""),
        md("""## Part A2 · Recording-level inspection (label-blind; no source model, no graph)

Per D2 recording: duration, selection, candidate and accepted windows, amplitude, and line noise with what a 50 Hz
notch WOULD change (LCMV covariance input, window acceptance); the same line-noise audit of the D3 recordings for
comparison. Nothing is applied; decision D3 is taken from these numbers before any classification."""),
        code("""cross('inspect-d2')
cross('inspect-d3-line')
display(pd.read_csv(CROSS + '/audit/d2_inspection.csv').filter(regex='key|duration|windows|std|line|covariance|flip|status').round(3))"""),
        md("""## Part B · Matched-channel D2 stores (STOP HERE unless the author has authorized it in writing)

Set `AUTHORIZED_BY` to the author's name only after the written authorization; it is recorded once in
`CROSS_D3_D2_19CH_v1/authorizations.json`. A two-participant smoke build (`--limit 2`) runs and is validated first."""),
        code("""AUTHORIZED_BY = ''          # e.g. 'Sami (written authorization 2026-..-..)'; empty = stop here
if not AUTHORIZED_BY:
    print('Not authorized: the D2 stores are not built. Review Part A and docs/cross_dataset/design.md first.')
else:
    cross(f'authorize --stage build_d2 --by "{AUTHORIZED_BY}"')
    cross('build-d2 --inverse lcmv --limit 2')
    cross('validate-d2 --inverse lcmv --limit 2')
    for inverse in ('lcmv', 'wmne'):
        cross(f'build-d2 --inverse {inverse}')
        cross(f'validate-d2 --inverse {inverse}')
    for inverse in ('lcmv', 'wmne'):
        for mode in ('B', 'A'):
            cross(f'compat --inverse {inverse} --mode {mode}')"""),
        md("## Part C · Freeze the campaign (author; before any training)"),
        code("""FREEZE_CAMPAIGN = False     # True only after reviewing Part A / B outputs and the design document
if FREEZE_CAMPAIGN:
    cross('freeze')
else:
    print('campaign not frozen; nothing can be trained or evaluated')"""),
    ])


def cross_direction_notebook(direction: str) -> dict:
    src, tgt = direction.split("_to_")
    who = ("the 52 frozen D3 development participants (the 13 D3 holdout participants are never used)"
           if src == "D3" else "all 23 D2 participants; plus the source-only D2 cross-validation (5 folds x 1 repeat, "
                                  "the primary D3 protocol)")
    return notebook([
        md(f"""# Cross-dataset {'01' if src == 'D3' else '02'} · {src} -> {tgt} (zero-shot)

{CROSS_RULES}

Each unit fits a FRESH model of an unchanged Exp 0 / 1 / 3-7 / 8a variant (fixed 60-epoch protocol for the GNNs;
classical models: inner CV for C inside the source only) on {who}, then scores the BLINDED {tgt} windows; one mean
P(AD) per participant is saved, without labels. Completed units are checkpointed and skipped on restart. No target
metric is computed here (notebook 03).
""" + CROSS_HOWTO + " Use a **GPU runtime** (T4)."),
        code(cross_setup()),
        md(CROSS_ENV),
        code(CROSS_ENV_CODE),
        md("## Training authorization (author) and status"),
        code("""AUTHORIZED_BY = ''          # the author's name after the written authorization of real-data training
if AUTHORIZED_BY:
    cross(f'authorize --stage train --by "{AUTHORIZED_BY}"')
cross('status')"""),
        md("## Run (resumable; this direction only). `CONTEXTS` is the only cell meant to be edited."),
        code(f"""CONTEXTS = [('lcmv', 'B'), ('wmne', 'B'), ('sensor', 'B'), ('lcmv', 'A'), ('wmne', 'A'), ('sensor', 'A')]
ERRORS = {{}}
for inverse, mode in CONTEXTS:
    print(f'==== {direction} | {{inverse}} | mode {{mode}}')
    try:
        cross(f'run --direction {direction} --inverses {{inverse}} --modes {{mode}}')
    except RuntimeError as exc:
        ERRORS[(inverse, mode)] = exc
cross('status')
if ERRORS:
    raise RuntimeError(f'units failed in {{sorted(ERRORS)}}; completed units are saved. Run all again.')"""),
        md("""## Optional · Source-only stability analysis (separately authorized; never selects anything)

Training seeds 1-4 for every unit and, for D2 -> D3, the repeated D2 source CV (5 folds x 5 repeats). Leave empty
unless the author has authorized it in writing."""),
        code(f"""STABILITY_AUTHORIZED_BY = ''
if STABILITY_AUTHORIZED_BY:
    cross(f'authorize --stage stability --by "{{STABILITY_AUTHORIZED_BY}}"')
    for inverse, mode in CONTEXTS:
        cross(f'run --stage stability --direction {direction} --inverses {{inverse}} --modes {{mode}}')"""),
    ], gpu=True)


def cross_report_notebook() -> dict:
    return notebook([
        md(f"""# Cross-dataset 03 · Evaluation and comparison report (both directions)

{CROSS_RULES}

The evaluation reads the target labels for the units of the frozen campaign only, at its commit: participant ROC-AUC
(DeLong and participant bootstrap CIs), balanced accuracy, sensitivity, specificity (Wilson), F1, MCC, confusion
matrices, source-only CV, transfer degradation, D3 development / holdout strata (the 13 holdout participants were
inspected in earlier work: they are NOT an untouched holdout), paired exploratory comparisons (Holm), the age-only
baseline and the age analyses. Tables and figures are diagnostics from saved outputs.
""" + CROSS_HOWTO + " A CPU runtime is enough."),
        code(cross_setup()),
        md(CROSS_ENV),
        code(CROSS_ENV_CODE),
        md("## Evaluate (frozen units only) and report"),
        code(SHOW_LATEST + """
cross('status')
cross('evaluate')
cross('report')
rep = latest(CROSS + '/reports/primary_*')
print(open(rep + '/README.md').read())
for f in ('table1_transfer.csv', 'table2_d3_strata.csv', 'table3_paired.csv', 'table4_age.csv'):
    print('====', f)
    display(pd.read_csv(f'{rep}/{f}').round(3))
for png in sorted(glob.glob(rep + '/*.png')):
    display(Image(filename=png))"""),
    ])


def cross_readme(entries) -> str:
    rows = "\n".join(f"| {i} | [{n}]({CROSS_COLAB}/{n}) | {d} | {rt} |" for i, (n, d, rt) in enumerate(entries, 1))
    return f"""# Cross-dataset D3 <-> D2 notebooks (branch `{CROSS_BRANCH}`)

Generated by `scripts/make_notebooks.py`; never edit by hand. Design: `docs/cross_dataset/design.md`; exact steps:
`docs/cross_dataset/run_guide.md`. One notebook at a time, *Runtime → Run all*.

| # | Notebook | What it does | Runtime |
|---|---|---|---|
{rows}

Outputs only in `<outputs>/CROSS_D3_D2_19CH_v1/` (own code pin). The release (`FINAL_v1/`), amendment A2 and the D3
stores are read only. The D2 stores and real-data training need the author's recorded authorization.
"""


def cross_main() -> None:
    out = REPO / "notebooks" / "cross_dataset"
    out.mkdir(parents=True, exist_ok=True)
    nbs = {"00_CrossDataset_Audit_and_Build.ipynb": (cross_audit_notebook(), "audit; matched-channel D2 stores "
                                                     "(authorized); compatibility; freeze", "CPU"),
           "01_CrossDataset_D3_to_D2.ipynb": (cross_direction_notebook("D3_to_D2"), "D3 -> D2 source fits, blind D2 "
                                              "scores", "GPU"),
           "02_CrossDataset_D2_to_D3.ipynb": (cross_direction_notebook("D2_to_D3"), "D2 -> D3 source fits and D2 CV, "
                                              "blind D3 scores", "GPU"),
           "03_CrossDataset_Comparison_Report.ipynb": (cross_report_notebook(), "evaluation (frozen units), tables, "
                                                       "figures", "CPU")}
    for name, (nb, _, _) in nbs.items():
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(cross_readme([(n, d, rt) for n, (_, d, rt) in nbs.items()]))
    print(f"{len(nbs)} cross-dataset notebooks -> {out}")


# ------------------------------------------------------------------------------------------------ index
def readme(entries: list[tuple[str, str, str]]) -> str:
    rows = "\n".join(f"| {i} | [{n}]({COLAB}/{n}) | {d} | {rt} |" for i, (n, d, rt) in enumerate(entries, 1))
    return f"""# Notebooks of the final framework (release {release.RELEASE})

Generated by `scripts/make_notebooks.py`; never edit by hand. Run them **in this order**, one at a time, each with
*Runtime → Run all* (details, configurations and expected run times: `FINAL_RUN_GUIDE.md`).

| # | Notebook | What it does | Runtime |
|---|---|---|---|
{rows}

Notebook 04 comes before 03 on purpose: the graph-construction comparison runs once, on the atlas you record after
reviewing 04. Notebook 04 itself selects nothing and stops after the comparison.
Development CV only until notebook 99, which freezes the final analysis and then evaluates the holdout once,
descriptively, for the frozen units.

Outputs go to `<outputs>/{release.ROOT}/` on Google Drive, except the new lobes12 / r18 stores of notebook 02, which
are written next to the 68-ROI stores (`<outputs>/D3_LCMV_XU/graphsLX{{,10}}-<atlas>_*`,
`<outputs>/D3_WMNE_XU/graphsWX{{,10}}-<atlas>_*`; layout in `FINAL_RUN_GUIDE.md`). The validated 68-ROI
stores (`D3_LCMV_XU/graphsLX*_corr1`, `D3_WMNE_XU/graphsWX*`) and every earlier result are only read. The notebooks
of the earlier suites are archived in `legacy/notebooks/` and are not part of this release.
"""


def main() -> None:
    out = REPO / "notebooks"
    out.mkdir(exist_ok=True)
    nbs = {"00_setup_and_stores.ipynb": (setup_notebook(), "verify and lock the four stores; test suite", "CPU"),
           "01_participant_qc.ipynb": (qc_notebook(), "participant QC; category A decisions", "CPU"),
           "02_parcellation.ipynb": (parcellation_notebook(), "source resolution; lobes12 / r18 atlas checks; "
                                                              "authorized stores from the recordings, validated and "
                                                              "matched; their QC", "CPU"),
           "04_roi_68_vs_18.ipynb": (roi_notebook(), "controlled dk68 vs lobes12 vs r18 comparison (development CV; "
                                                     "exploratory); stops for your decision", "GPU"),
           "03_graph_construction.ipynb": (graph_notebook(), "graph diagnostics and development-CV comparison on the "
                                                             "recorded atlas; selection", "GPU")}
    for exp, cfg in experiment_configs().items():
        name, title, gnn = EXP_TITLES[exp]
        nbs[f"{name}.ipynb"] = (experiment_notebook(exp, cfg), title, "GPU" if gnn else "CPU")
    nbs["Exp8a_xu_dual_gcn_surface.ipynb"] = (amendment_exp8_notebook(), "Exp 8 with a cortical-surface anatomical "
                                              "layer (amendment A2; own namespace and code pin)", "GPU")
    nbs["99_results_report.ipynb"] = (report_notebook(), "development-CV report; freeze; descriptive holdout of the "
                                                         "frozen units; final tables", "GPU")
    order = [s["notebook"] for s in run_plan()["steps"]]
    if sorted(order) != sorted(nbs):
        raise SystemExit(f"configs/final/run_plan.yaml and the generated notebooks differ: {set(order) ^ set(nbs)}")
    nbs = {n: nbs[n] for n in order}
    for name, (nb, _, _) in nbs.items():
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(readme([(n, d, rt) for n, (_, d, rt) in nbs.items()]))
    print(f"{len(nbs)} notebooks -> {out}")
    cross_main()


if __name__ == "__main__":
    main()
