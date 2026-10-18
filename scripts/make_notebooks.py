"""Generate the Colab notebooks of the final framework from one template (keeps them identical and thin).

    python scripts/make_notebooks.py          # rewrite notebooks/ ; tests check they are up to date

    notebooks/00_setup_and_stores.ipynb        verify and lock the four validated stores (CPU)
    notebooks/01_participant_qc.ipynb          participant QC of every store; category A decisions (CPU)
    notebooks/02_parcellation.ipynb            source resolution, r18 atlas checks, r18 stores (validated, matched), QC
    notebooks/04_roi_68_vs_18.ipynb            controlled dk68 vs r18 comparison (development CV); atlas decision
    notebooks/03_graph_construction.ipynb      graph diagnostics + the development-CV comparison on that atlas (GPU)
    notebooks/Exp<k>_*.ipynb                   ONE experiment each, Exp 0-8, development CV (GPU for 2-8)
    notebooks/99_results_report.ipynb          freeze the final analysis, descriptive holdout, final tables (GPU)
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


QC_GATE = """UNRESOLVED = []
for inverse in ('lcmv', 'wmne'):
    for mode in ('B', 'A'):
        try:
            sh(f"python scripts/final.py qc --inverse {inverse} --mode {mode} --atlas {ATLAS}")
        except RuntimeError:
            UNRESOLVED.append((inverse, mode))
from IPython.display import Markdown
for d in sorted(glob.glob(FINAL + f'/qc/*_*_{ATLAS}/')):
    if os.path.exists(d + 'qc_report.md'):
        display(Markdown(open(d + 'qc_report.md').read()))
if UNRESOLVED:
    raise RuntimeError(f'category A findings need a decision for {ATLAS} {UNRESOLVED}: see the reports above and '
                       'docs/final/qc.md; experiments refuse to start until then')"""


def parcellation_notebook() -> dict:
    return notebook([
        md("""# 02 · Parcellation: source resolution, then the 18-region (r18) stores (CPU)

1. **Anatomy of the r18 atlas** (no EEG): each hemisphere's 34 Desikan-Killiany labels partitioned into 9 regions,
   no cross-hemisphere region, identical left / right definitions, every region one contiguous piece of cortex,
   correct hemispheric placement, non-empty source sets.
2. **Analytic source resolution** per atlas (68 / 34 / 18 / 12 regions), both inverses, with the stores' fsaverage
   BEM and inverse settings: leadfield rank, forced redundancy, cross-talk, own-region dominance, leakage.
3. **The stores' own operators** (real data, no labels): every development participant's saved 68-ROI operator
   (LCMV from the participant's own data covariance; wMNE) against the stores' cached forward model; feature
   reliability; whether coarse signals could be rebuilt from saved intermediates (they are not: §4 rebuilds).
4. **r18 stores** (approved by the author, 2026-10-18, conditional on steps 1-2): built from the ORIGINAL recordings
   by the unchanged store pipeline (merged Desikan-Killiany labels, `mean_flip` over their own sources; never averages
   of 68-ROI signals or PLV), LCMV and wMNE, window modes B and A; validated, wMNE matched to LCMV, and each matched
   window by window to its 68-ROI store (participants, labels, windows, split, sampling, sensor features).
5. **QC** of the four r18 stores, then the resolution of the r18 operators actually built and their reliability.

The 68-ROI stores are only read. r18 is NOT made the analysis atlas here: that is recorded by the author after
notebook 04. A rank-18 sensor space does not prove that 18 regions are independently or accurately localized; the
cross-talk and dominance numbers quantify how far they are. Resumable (participant checkpoints). CPU, about 2-3 h
in total (store builds ≈ 40 min LCMV + 30-45 min wMNE).
""" + HOWTO),
        code(SETUP),
        code(SHOW_LATEST + '''
import json'''),
        md("## 1. Anatomical checks of the r18 atlas"),
        code('''ATLAS_OK = True
try:
    sh("python scripts/final.py atlas-check --atlas r18")
except RuntimeError:
    ATLAS_OK = False'''),
        md("## 2. Analytic source resolution and the atlas recommendation (stores' BEM)"),
        code('sh("python scripts/final.py parcellation")'),
        code('''d = latest(FINAL + '/parcellation/resolution_*')
display(pd.read_csv(d + '/resolution_by_atlas.csv').round(3))
rec = json.load(open(d + '/recommendation.json'))
print('recommended atlas:', rec['recommended'], '|', rec['forward_model'])
for v in rec['verdicts']:
    print(v['atlas'], 'eligible' if v['eligible'] else 'not eligible', '; '.join(v['reasons']))
R18_ELIGIBLE = any(v['atlas'] == 'r18' and v['eligible'] for v in rec['verdicts'])
if 'approx' in rec['forward_model'] or 'sphere' in rec['forward_model']:
    print("NOTE: the stores' BEM was not found; this is an approximate head model (docs/final/parcellation.md)")'''),
        md("## 3. The 68-ROI stores' own operators, feature reliability, reconstruction check (read-only)"),
        code('''for inverse in ('lcmv', 'wmne'):
    sh(f"python scripts/final.py store-resolution --inverse {inverse} --atlas dk68")
    sh(f"python scripts/final.py reliability --inverse {inverse} --mode B")
    sh(f"python scripts/final.py reconstruct-check --inverse {inverse} --mode B")'''),
        md("""## 4. r18 stores from the original recordings (approved; runs only if steps 1-2 passed)

Each step is resumable: completed participants are skipped, and `validation.json`, `match.json` and
`atlas_match.json` are written once. New directories next to the 68-ROI stores (outside `FINAL_v1/`):
`D3_LCMV_XU/graphsLX-r18_v1_<hash>`, `graphsLX10-r18_v1_<hash>`, `D3_WMNE_XU/graphsWX-r18_v1_<hash>`,
`graphsWX10-r18_v1_<hash>`; their sizes are printed below."""),
        code('''BUILD_R18 = True   # approved by the author (2026-10-18), conditional on the checks above
if not (ATLAS_OK and R18_ELIGIBLE):
    raise RuntimeError(f"r18 not built: atlas checks passed = {ATLAS_OK}, r18 meets the resolution rule with the "
                       f"stores' BEM = {R18_ELIGIBLE}. The 68-ROI stores remain the analysis; report to the author.")
if BUILD_R18:
    for inverse in ('lcmv', 'wmne'):
        sh(f"python scripts/build_stores.py build --inverse {inverse} --atlas r18")
        sh(f"python scripts/build_stores.py validate --inverse {inverse} --atlas r18")
    sh("python scripts/build_stores.py match --atlas r18")
    for inverse in ('lcmv', 'wmne'):
        sh(f"python scripts/build_stores.py match-atlas --inverse {inverse} --atlas r18")
        sh(f"python scripts/build_stores.py size --inverse {inverse} --atlas r18")
    sh("python scripts/final.py stores --atlas r18")      # verification of the four r18 stores (locked when QC opens them)'''),
        md("## 5. QC of the r18 stores, then the resolution and reliability of the r18 operators actually built"),
        code("ATLAS = 'r18'\n" + QC_GATE),
        code('''for inverse in ('lcmv', 'wmne'):
    sh(f"python scripts/final.py store-resolution --inverse {inverse} --atlas r18")
    sh(f"python scripts/final.py reliability --inverse {inverse} --mode B --atlas r18")
for inverse in ('lcmv', 'wmne'):
    for atlas in ('dk68', 'r18'):
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
    return notebook([
        md(f"""# 04 · Controlled ROI comparison: 68 vs 18 regions (development CV only)

`configs/final/roi_comparison.yaml`. The same 52 development participants, the same five participant-disjoint
folds (`xu_dev_cv`), window mode {cfg['mode']}, the same training procedures and seeds, LCMV and wMNE, for:

{models}

GNNs on dense weighted PLV, exactly 60 epochs. **The holdout is not evaluated** (and is refused by the code until
the final analysis is frozen in notebook 99).

The report keeps two kinds of evidence apart: **source localization / spatial resolution** (label-free: forced
redundancy, cross-talk, own-region dominance of the operators actually built, feature reliability; notebook 02) and
**classification** (participant-level pooled out-of-fold ROC-AUC with DeLong CI, balanced accuracy, F1, sensitivity,
specificity, paired r18 - dk68 differences with paired DeLong and participant-bootstrap CIs). A higher AUC alone
does not make an atlas better, and the summary never forces a winner. You then record the analysis atlas (§4).

The dk68 runs are the same units the experiment notebooks reuse (nothing is computed twice). GPU runtime (T4).
Resumable.
""" + HOWTO),
        code(SETUP),
        code(SHOW_LATEST + '''
import json'''),
        md("## 1. Both atlases' stores verify (read-only)"),
        code('''for atlas in ('dk68', 'r18'):
    sh(f"python scripts/final.py stores --atlas {atlas}")'''),
        md("## 2. Runs (development CV only; each unit is checkpointed)"),
        code(f'''RUNS = {json.dumps(runs)}      # (experiment, variant, atlas, inverse, graph)
ERRORS = []
for exp, variant, atlas, inverse, graph in RUNS:
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
finally:
    d = latest(FINAL + '/roi_comparison/compare_*')
    from IPython.display import Markdown
    if d:
        display(Markdown(open(d + '/report.md').read()))'''),
        md("""## 4. Your decision: the analysis atlas (recorded once)

Review §3 (source-resolution evidence first, classification second). Then set `ATLAS_DECISION` to `'dk68'` or
`'r18'` with the scientific reason and run this cell. Nothing is recorded while it is `None`; notebook 03 and the
experiment notebooks stop until a decision exists. The atlas not chosen stays a sensitivity / reference analysis."""),
        code('''ATLAS_DECISION = None   # 'dk68' or 'r18'
REASON = ''             # e.g. 'r18: no forced redundancy, cross-talk 0.5 vs 0.85, reliability ...; AUC difference ...'
if ATLAS_DECISION:
    import shlex
    sh(f"python scripts/final.py choose-atlas --atlas {ATLAS_DECISION} --reason {shlex.quote(REASON)}")
else:
    print('no atlas recorded yet: set ATLAS_DECISION and REASON above after reviewing section 3')'''),
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
""" + HOWTO + (" Use a **GPU runtime** (T4)." if gnn else " CPU is sufficient.")),
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


# ------------------------------------------------------------------------------------------------ index
def readme(entries: list[tuple[str, str, str]]) -> str:
    rows = "\n".join(f"| {i} | [{n}]({COLAB}/{n}) | {d} | {rt} |" for i, (n, d, rt) in enumerate(entries, 1))
    return f"""# Notebooks of the final framework (release {release.RELEASE})

Generated by `scripts/make_notebooks.py`; never edit by hand. Run them **in this order**, one at a time, each with
*Runtime → Run all* (details, configurations and expected run times: `FINAL_RUN_GUIDE.md`).

| # | Notebook | What it does | Runtime |
|---|---|---|---|
{rows}

Notebook 04 comes before 03 on purpose: the graph-construction comparison runs once, on the atlas you record in 04.
Development CV only until notebook 99, which freezes the final analysis and then evaluates the holdout once,
descriptively, for the frozen units.

Outputs go to `<outputs>/{release.ROOT}/` on Google Drive, except the new 18-region stores of notebook 02, which are
written next to the 68-ROI stores (`<outputs>/D3_LCMV_XU/graphsLX-r18_*`, `graphsLX10-r18_*`,
`<outputs>/D3_WMNE_XU/graphsWX-r18_*`, `graphsWX10-r18_*`; layout in `FINAL_RUN_GUIDE.md`). The validated 68-ROI
stores (`D3_LCMV_XU/graphsLX*_corr1`, `D3_WMNE_XU/graphsWX*`) and every earlier result are only read. The notebooks
of the earlier suites are archived in `legacy/notebooks/` and are not part of this release.
"""


def main() -> None:
    out = REPO / "notebooks"
    out.mkdir(exist_ok=True)
    nbs = {"00_setup_and_stores.ipynb": (setup_notebook(), "verify and lock the four stores; test suite", "CPU"),
           "01_participant_qc.ipynb": (qc_notebook(), "participant QC; category A decisions", "CPU"),
           "02_parcellation.ipynb": (parcellation_notebook(), "source resolution; r18 atlas checks; r18 stores "
                                                              "from the recordings, validated and matched; their QC",
                                     "CPU"),
           "04_roi_68_vs_18.ipynb": (roi_notebook(), "controlled dk68 vs r18 comparison (development CV); "
                                                     "you record the analysis atlas", "GPU"),
           "03_graph_construction.ipynb": (graph_notebook(), "graph diagnostics and development-CV comparison on the "
                                                             "recorded atlas; selection", "GPU")}
    for exp, cfg in experiment_configs().items():
        name, title, gnn = EXP_TITLES[exp]
        nbs[f"{name}.ipynb"] = (experiment_notebook(exp, cfg), title, "GPU" if gnn else "CPU")
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


if __name__ == "__main__":
    main()
