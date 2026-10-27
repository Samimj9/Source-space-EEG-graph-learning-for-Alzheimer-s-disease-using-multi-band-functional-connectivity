"""Generate the Colab notebooks of amendment A3 (``notebooks_a3/``; tests check they are up to date).

    A3_01_development_report.ipynb   audit + consolidated Exp 0-8a development report + D3 -> D2 manifest (CPU)
    A3_02_freeze.ipynb               dry run of the dual freeze; writes only if WRITE_FREEZE = True AND A3 is approved (CPU)
    A3_03_holdout_exp0_6.ipynb       descriptive holdout of the frozen Exp 0-6 units at the RELEASE pin (GPU)
    A3_04_holdout_exp8a.ipynb        descriptive holdout of the frozen Exp 8a units at the A2 pin (GPU)
    A3_05_final_report.ipynb         combined final report: development, holdout (separate), comparisons (CPU)
    A3_06_single_model_holdout.ipynb separate single-model holdout (Exp 8a, wMNE, abl_anatomical_both) (GPU)

They live apart from ``notebooks/`` (the release notebooks, whose tests and pins are untouched). Notebook 99 of the
release can no longer complete a freeze (exp8 has no result on lobes12); these notebooks are its companion.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_notebooks", REPO / "scripts" / "make_notebooks.py")
_mn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mn)
md, code, notebook, SHOW_LATEST = _mn.md, _mn.code, _mn.notebook, _mn.SHOW_LATEST

OWNER_REPO = _mn.OWNER_REPO
A3_BRANCH = "claude/a3-final-analysis"
REPO_URL = _mn.REPO_URL
COLAB = f"https://colab.research.google.com/github/{OWNER_REPO}/blob/{A3_BRANCH}/notebooks_a3"

SETUP = f'''# ===== Setup: nothing to edit. Mounts Drive, gets the A3 code, installs the requirements. =====
UPDATE_CODE = False   # leave False. True only to move the A3 code pin (<outputs>/FINAL_v1/amendments/A3_final/CODE_COMMIT.txt)
BRANCH = '{A3_BRANCH}'
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
_run(['git', 'clone', '-q', '-b', BRANCH, REPO_URL, '/content/repo'])   # full history: the pins are ancestors
os.chdir('/content/repo')
_run([sys.executable, 'scripts/a3_setup.py'] + (['--update-code'] if UPDATE_CODE else []))
sys.path.insert(0, '/content/repo/src')

from adgraph.colab import sh as _sh

def sh(cmd, cwd='/content/repo'):  # every script runs with the file backend
    os.environ['MPLBACKEND'] = 'Agg'
    return _sh(cmd, cwd)

from adgraph.config import load_paths
PATHS = load_paths('/content/repo/configs/paths.yaml')
OUT = PATHS['processed_dir']
FINAL = OUT + '/FINAL_v1'
A3 = FINAL + '/amendments/A3_final'
print('A3 outputs on Drive:', A3)
'''

ENV = '''ACCEPT_ENVIRONMENT_DIFFERENCE = False   # leave False: a difference that cannot be removed must stop here
from adgraph import numerical_env
from adgraph.a3 import environment, inventory as a3_inventory
# The holdout fits are restored to the single environment of the Exp 8a (A2) results (A2 chose it to equal the Exp 7
# units'). Release units developed in a documented-compatible environment (Python patch version, torch cpu / CUDA build)
# are refit in it; `a3.py holdout` compares it with every unit's development environment, records the comparison and
# refuses an incompatible one.
FOUND, _ = a3_inventory.discover(OUT, settings=('xu_dev_cv',))
REFERENCE = environment.holdout_reference(FOUND)
PY = numerical_env.ensure_interpreter(REFERENCE, repo='/content/repo', base='/content',
                                      accept_difference=ACCEPT_ENVIRONMENT_DIFFERENCE, run=sh)
NOW = numerical_env.interpreter_versions(PY, REFERENCE)
DIFF = numerical_env.differences(REFERENCE, NOW)
assert ACCEPT_ENVIRONMENT_DIFFERENCE or not DIFF, f'numerical environment differs from the Exp 8a (A2) environment: {DIFF}'
print('holdout scripts run with', PY, NOW)'''

HOWTO = f"""
**How to use:** *Runtime → Run all*. Allow Google Drive access when asked. If Colab disconnects, open the notebook again
and *Run all* again: finished work is kept on Drive. Everything A3 writes goes to `<outputs>/FINAL_v1/amendments/A3_final/`
(plus, once, the two frozen records created by the freeze); no completed result, pin, QC decision or store is changed.

**Holdout policy.** {_mn.release.HOLDOUT_POLICY}
"""
LATEST = SHOW_LATEST + '''
def show_latest(kind):
    d = latest(A3 + f'/reports/{kind}_*')
    print(d)
    return d'''


def dev_report_nb() -> dict:
    return notebook([
        md("""# A3-01 · Development report, Exp 0-8a (audit, consolidated tables, comparisons, D3 → D2 manifest)

Reads the completed development-CV results of the release namespace (Exp 0-7, at the release pin) and of the amendment A2
namespace (Exp 8a, at the A2 pin); copies and changes nothing. Writes tables, comparisons with Holm correction, selection
checks, figures and an interpretation to a new folder `A3_final/reports/development_<time>/`; opens no holdout result.
""" + HOWTO),
        code(SETUP),
        md("## 1. Reproducibility audit (read only; stops if a check fails)"),
        code('sh("python scripts/a3.py audit")'),
        md("## 2. Consolidated report"),
        code('sh("python scripts/a3.py report")'),
        code(LATEST + '''
d = show_latest('development')
print(open(d + '/interpretation.md').read())
for f in ('development_results_primary.csv', 'family_summary.csv'):
    print('====', f)
    display(pd.read_csv(f'{d}/{f}').round(3))
for png in sorted(glob.glob(d + '/figures/*.png')):
    display(Image(filename=png))'''),
        md("## 3. D3 → D2 manifest (reads D3 development results only; starts nothing of D2)"),
        code('sh("python scripts/a3.py d2-manifest")'),
    ])


def freeze_nb() -> dict:
    return notebook([
        md("""# A3-02 · Dual-namespace freeze (dry run unless you switch it on)

Checks that every unit of the A3 plan (`configs/final/amendments/holdout_A3.yaml`: `holdout.yaml` with `exp8` → `exp8a`)
has a complete, checksum-verified development-CV result at its pin, then prints the inventory. With `WRITE_FREEZE = True`
**and** an approved A3 record in `configs/final/amendments.yaml` (status `approved`, `approved_by` filled, committed) it
creates, once: the release record (Exp 0-6, release pin), the A2 record (Exp 8a, A2 pin) and the A3 master record with
their SHA-256. Rerunning cannot change an existing freeze. Nothing is trained and no holdout is evaluated here.
""" + HOWTO),
        code(SETUP),
        md("## 0. Which write-once method does this Drive folder allow? (own temporary folder only)"),
        code('sh("python scripts/a3.py probe")'),
        md("## 1. Audit and dry run (writes nothing)"),
        code('''sh("python scripts/a3.py audit")
sh("python scripts/a3.py freeze")'''),
        md("## 2. Write the freeze (default off; needs the author's approval recorded in the repository)"),
        code('''WRITE_FREEZE = False   # True only after you reviewed the inventory above and approved amendment A3 in the repo
if WRITE_FREEZE:
    sh("python scripts/a3.py freeze --write")
else:
    print('WRITE_FREEZE is False: nothing was written')'''),
        md("## 3. Verify (only passes once a freeze exists)"),
        code('''import os
if os.path.exists(A3 + '/freeze.json'):
    sh("python scripts/a3.py verify")
else:
    print('no freeze yet')'''),
    ])


def holdout_nb(part: str, exps: list[str], title: str, pin_text: str, name: str) -> dict:
    return notebook([
        md(f"""# {name} · {title}

{pin_text} The pinned code runs in a separate git worktree (`/content/a3_{part}_<pin>`), in the numerical environment of the
completed units (restored automatically if this Colab differs). Each frozen unit is fitted on the 52 development
participants and evaluated once on the 13 holdout participants; the pinned code refuses any unit that is not in its
frozen record or any other commit, and A3 verifies the master freeze (checksums, pins, QC decisions, store lock, development
results) before and after every call. Resumable. GPU runtime (T4). Descriptive only.
""" + HOWTO),
        code(SETUP),
        md("## 1. Numerical environment (exactly that of the completed units)"),
        code(ENV),
        md("## 2. The freeze must verify (refused otherwise)"),
        code('sh("python scripts/a3.py verify")'),
        md(f"## 3. Holdout of the frozen units of {', '.join(exps)}"),
        code(f'''ERRORS = []
for exp in {json.dumps(exps)}:
    try:
        sh(f"python scripts/a3.py holdout --part {part} --exp {{exp}} --python {{PY}} --repo /content/repo --base /content")
    except RuntimeError:
        ERRORS.append(exp)
sh("python scripts/a3.py holdout-status")
if ERRORS:
    raise RuntimeError(f'holdout units did not complete for {{ERRORS}}; completed units are saved. Run all again.')'''),
    ], gpu=True)


ENV_SINGLE = '''ACCEPT_ENVIRONMENT_DIFFERENCE = False   # leave False: a difference that cannot be removed must stop here
from adgraph import numerical_env
from adgraph.a3 import single
# The environment recorded by THIS unit's development result (one small file; the ~320 other results are not scanned).
REFERENCE = single.dev_libraries(OUT)
PY = numerical_env.ensure_interpreter(REFERENCE, repo='/content/repo', base='/content',
                                      accept_difference=ACCEPT_ENVIRONMENT_DIFFERENCE, run=sh)
NOW = numerical_env.interpreter_versions(PY, REFERENCE)
DIFF = numerical_env.differences(REFERENCE, NOW)
assert ACCEPT_ENVIRONMENT_DIFFERENCE or not DIFF, f'numerical environment differs from the development environment of this unit: {DIFF}'
print('the holdout runs with', PY, NOW)'''


def single_nb() -> dict:
    return notebook([
        md("""# A3-06 · Single-model D3 holdout (selected after development CV)

**A separate, lightweight evaluation of ONE frozen configuration, not completion of the full A3 holdout protocol.**
Model: `exp8a_xu_dual_gcn_surface` / `plv__psd_de_strength__abl_anatomical_both`, wMNE, mode B, lobes12, dense
(`configs/final/amendments/a3_single_model.yaml`). It was chosen by the author after development CV to save computation;
the other 115 frozen configurations are **not** evaluated here, and the 13-participant holdout was inspected earlier, so the
result is descriptive only. The A3 master freeze, both frozen records, the 116-unit inventory and all development results
are only read.

It verifies that the exact unit is in the frozen A2 record and that its development result is intact (the other results
are not rescanned), restores the numerical environment of the A2 results and checks it against this unit's development
environment, then fits the unit once on the 52 development participants and evaluates the 13 holdout participants **at the
A2 pin** (`7fa470c`, unchanged model and protocol). Resumable: a completed unit is not recomputed. GPU runtime (T4).
""" + HOWTO),
        code(SETUP),
        md("## 1. Numerical environment (restored to the one recorded by this unit's development result)"),
        code(ENV_SINGLE),
        md("## 2. Membership in the frozen inventory and integrity checks (read only)"),
        code('sh("python scripts/a3.py single-verify")'),
        md("## 3. Holdout of this one unit (writes the selection manifest once; resumable)"),
        code('sh(f"python scripts/a3.py single-holdout --python {PY} --repo /content/repo --base /content")'),
        md("## 4. Report of this one model (does not need the other frozen units)"),
        code('sh("python scripts/a3.py single-report")'),
        code(LATEST + '''
d = latest(A3 + '/single_model/report_*')
print(d)
print(open(d + '/single_model_holdout_report.md').read())
display(pd.read_csv(d + '/predictions_participants.csv'))
for png in sorted(glob.glob(d + '/*.png')):
    display(Image(filename=png))'''),
    ], gpu=True)


def final_nb() -> dict:
    return notebook([
        md("""# A3-05 · Combined final report (development and holdout, kept apart)

Requires a verifying freeze and every frozen holdout unit completed. Writes a development report (all of Exp 0-8a,
comparisons, selection checks, interpretation) and a separate descriptive holdout report to new folders of `A3_final/reports/`.
The holdout report shows holdout metrics only, plus the development AUCs of the frozen records in a labelled comparison
table; it contains no test and selects nothing.
""" + HOWTO),
        code(SETUP),
        md("## 1. Freeze verification and holdout completeness"),
        code('''sh("python scripts/a3.py verify")
sh("python scripts/a3.py holdout-status")
from adgraph.a3 import holdout
st = holdout.status(OUT)
assert (st['holdout_state'] == 'completed').all(), st[st['holdout_state'] != 'completed']'''),
        md("## 2. Development report and holdout report"),
        code('''sh("python scripts/a3.py report")
sh("python scripts/a3.py holdout-report")'''),
        code(LATEST + '''
d = show_latest('development')
print(open(d + '/interpretation.md').read())
h = show_latest('holdout')
print(open(h + '/holdout_results.md').read())
for png in sorted(glob.glob(d + '/figures/*.png')) + sorted(glob.glob(h + '/*.png')):
    display(Image(filename=png))'''),
    ])


def readme(entries) -> str:
    rows = "\n".join(f"| {i} | [{n}]({COLAB}/{n}) | {d} | {rt} |" for i, (n, d, rt) in enumerate(entries, 1))
    return f"""# Notebooks of amendment A3 (final-analysis integration)

Generated by `scripts/make_a3_notebooks.py`; never edit by hand. Open them from branch `{A3_BRANCH}` and run **in this order**,
one at a time, each with *Runtime → Run all* (details: `docs/final/a3_colab_instructions.md`). Do not run steps 3-4 before
you have reviewed the freeze inventory and approved A3.

| # | Notebook | What it does | Runtime |
|---|---|---|---|
{rows}
"""


def main() -> None:
    out = REPO / "notebooks_a3"
    out.mkdir(exist_ok=True)
    rel_pin = "Release part: Exp 0-6 at the **release pin** (`FINAL_v1/CODE_COMMIT.txt`, `59a9ab7`), `scripts/final.py holdout --exp <exp>`."
    nbs = {"A3_01_development_report.ipynb": (dev_report_nb(), "audit; consolidated Exp 0-8a development report; "
                                              "D3 → D2 manifest", "CPU"),
           "A3_02_freeze.ipynb": (freeze_nb(), "dry run of the dual freeze (writes only with approval + flag)", "CPU"),
           "A3_03_holdout_exp0_6.ipynb": (holdout_nb("release", ["exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"],
                                                     "Descriptive holdout of the frozen Exp 0-6 units (release pin)",
                                                     rel_pin, "A3-03"), "descriptive holdout, Exp 0-6, at the release "
                                          "pin", "GPU"),
           "A3_04_holdout_exp8a.ipynb": (holdout_nb("A2", ["exp8a"], "Descriptive holdout of the frozen Exp 8a units "
                                                    "(A2 pin)", "Amendment part: Exp 8a at the **A2 pin** (`FINAL_v1/"
                                                    "amendments/A2_exp8_surface/CODE_COMMIT.txt`, `7fa470c`), "
                                                    "`scripts/final.py holdout --amendment A2 --exp exp8a`.", "A3-04"),
                                         "descriptive holdout, Exp 8a, at the A2 pin", "GPU"),
           "A3_05_final_report.ipynb": (final_nb(), "combined final report: development and holdout (separate)", "CPU"),
           "A3_06_single_model_holdout.ipynb": (single_nb(), "SEPARATE: holdout of ONE selected frozen model "
                                                "(Exp 8a, wMNE); not the full protocol", "GPU")}
    for name, (nb, _, _) in nbs.items():
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(readme([(n, d, rt) for n, (_, d, rt) in nbs.items()]))
    print(f"{len(nbs)} notebooks -> {out}")


if __name__ == "__main__":
    main()
