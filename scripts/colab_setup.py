"""One-step Colab setup, called by the first cell of every notebook (nothing to edit).

1. Locate the shared AD-LEB folder on the mounted Drive and choose the output folder
   (AD-LEB/processed if writable, else My Drive/AD-LEB-processed).
2. Pin the code: the first session records the commit in <output>/CODE_COMMIT.txt and every
   later session (any notebook) checks out that same commit. All results then come from one
   commit, and a newer push to GitHub never invalidates unfinished work. --update-code moves
   the pin to the newest commit deliberately (unfinished units are then recomputed).
   Two independent pins (methodology log entry 45): ``--family D1D2`` (default) uses <output>/CODE_COMMIT.txt
   and the D1/D2 checkpoints; ``--family D3`` uses <output>/D3_CODE_COMMIT.txt and the D3 checkpoints
   (graphsD3_v*, experiments_D3/) and never reads or writes CODE_COMMIT.txt. ``--no-pin`` (maintenance only,
   e.g. restoring a pin) uses the cloned commit and touches no pin.
3. Write configs/paths.yaml and install the pinned environment (requirements.txt).

Runs in a subprocess before the notebook imports adgraph, so the checked-out code is the code
that is imported.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from adgraph.colab import (choose_output_root, find_data_root, has_completed_work, resolve_code_pin,  # noqa: E402
                           sh, write_paths_yaml)

parser = argparse.ArgumentParser()
parser.add_argument("--update-code", action="store_true", help="move the code pin to the cloned commit")
parser.add_argument("--family", choices=["D1D2", "D3", "LX", "WX"], default="D1D2", help="which code pin to use")
parser.add_argument("--no-pin", action="store_true", help="use the cloned commit; read and write no pin")
parser.add_argument("--skip-install", action="store_true")
args = parser.parse_args()


def git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


data_root = find_data_root()
output_root = choose_output_root(data_root)
head = git("rev-parse", "HEAD")
PIN_FILES = {"D1D2": "CODE_COMMIT.txt", "D3": "D3_CODE_COMMIT.txt", "LX": "D3_LCMV_XU/CODE_COMMIT.txt",
             "WX": "D3_WMNE_XU/CODE_COMMIT.txt"}
pin_file = output_root / PIN_FILES[args.family]
# files the current notebooks call; an older pinned commit without them cannot serve these notebooks
REQUIRED = ["scripts/experiment.py", "scripts/build_graphs.py", "scripts/make_figures.py", "scripts/run_plan.py",
            "src/adgraph/colab.py"]
if args.family == "D3":                     # boundary-aware D3 processing (entry 45) is required from now on
    REQUIRED += ["src/adgraph/io/bids.py", "configs/pipeline_d3.yaml", "scripts/d3_data.py",
                 "src/adgraph/preprocessing/discontinuities.py"]
if args.family == "WX":                     # matched wMNE / Xu suite (entry 55), its own pin
    REQUIRED += ["src/adgraph/lcmvxu/wmne.py", "src/adgraph/lcmvxu/match.py", "configs/wmne_xu/pipeline.yaml",
                 "scripts/wx_build.py"]
if args.family == "LX":                     # D3 LCMV / Xu suite (entry 50)
    REQUIRED += ["src/adgraph/lcmvxu/build.py", "configs/lcmv_xu/pipeline.yaml", "scripts/lx_build.py"]
if args.no_pin:
    commit, needs_checkout = head, False
    print(f"code: cloned commit {commit} (maintenance mode, no code pin read or written)")
else:
    old_pin = pin_file.read_text().split()[0] if pin_file.exists() and pin_file.read_text().split() else None
    work = has_completed_work(output_root, args.family)
    commit, needs_checkout = resolve_code_pin(pin_file, head, update=args.update_code, has_work=work)
    flag = {"D3": "UPDATE_D3_CODE", "LX": "UPDATE_LX_CODE", "WX": "UPDATE_WX_CODE"}.get(args.family, "UPDATE_CODE")
    if old_pin and old_pin != commit:
        print(f"{args.family} code pin moved {old_pin[:10]} -> {commit[:10]} "
              + (f"({flag})" if args.update_code else "(no completed results were pinned to the old commit)"))
    if needs_checkout:
        missing = [f for f in REQUIRED
                   if subprocess.run(["git", "cat-file", "-e", f"{commit}:{f}"], cwd=REPO).returncode]
        if missing:
            raise SystemExit(f"The pinned code ({commit[:10]}, {pin_file}) is older than these notebooks (it lacks "
                             f"{missing}). Set {flag} = True in the first cell to use the newest code "
                             "(unfinished work is then recomputed; finished results stay on Drive).")
        sh(f"git checkout -q {commit}", cwd=REPO)
    print(f"code: commit {commit} (pinned in {pin_file})" + (f" - newer pushes are ignored until {flag}"
                                                              if needs_checkout else ""))
    for fam, name in PIN_FILES.items():     # the other family's pin is only reported, never changed
        other = output_root / name
        if fam != args.family and other.exists() and other.read_text().split():
            print(f"{fam} code pin (not used here, unchanged): {other.read_text().split()[0]} ({other.name})")

paths_file = write_paths_yaml(REPO, data_root, output_root)
print(f"data:    {data_root}\noutputs: {output_root}\npaths:   {paths_file}")
if not args.skip_install:
    sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=REPO)
    sh(f"{sys.executable} -m pip install -q --no-deps -e .", cwd=REPO)
print(f"setup complete: commit {commit}")
