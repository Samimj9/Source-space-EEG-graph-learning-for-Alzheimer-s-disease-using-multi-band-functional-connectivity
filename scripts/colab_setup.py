"""One-step Colab setup, called by the first cell of every final notebook (nothing to edit).

1. Locate the shared AD-LEB folder on the mounted Drive and choose the output folder
   (AD-LEB/processed if writable, else My Drive/AD-LEB-processed).
2. Pin the code with the ONE version system of the final framework (``adgraph.release``): the first session records
   the commit in ``<output>/FINAL_v1/CODE_COMMIT.txt`` and every later session (any final notebook) checks out that
   same commit, so all results of the release come from one commit and a newer push never changes unfinished work.
   ``--update-code`` moves the pin deliberately (unfinished units are then recomputed; finished ones are kept and an
   identity change is refused, never silently reused). The pins of the earlier work (D1/D2, D3, LX, WX, ES200, R1,
   Exp 8) are only reported: they belong to the old branch and are never read for checkout or changed here.
   ``--no-pin`` (maintenance only) uses the cloned commit and touches no pin.
   ``--amendment A2`` (amendment notebooks only, methodology log entry 60) uses the amendment's OWN pin
   ``FINAL_v1/amendments/<dir>/CODE_COMMIT.txt`` instead; the release pin is then only reported, never moved.
3. Write configs/paths.yaml and install the pinned environment (requirements.txt).

Runs in a subprocess before the notebook imports adgraph, so the checked-out code is the code that is imported.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from adgraph import release  # noqa: E402
from adgraph.colab import (choose_output_root, find_data_root, has_completed_work, resolve_code_pin,  # noqa: E402
                           sh, write_paths_yaml)

parser = argparse.ArgumentParser()
parser.add_argument("--update-code", action="store_true", help="move the release code pin to the cloned commit")
parser.add_argument("--no-pin", action="store_true", help="use the cloned commit; read and write no pin")
parser.add_argument("--skip-install", action="store_true")
parser.add_argument("--amendment", default=None, help="use the code pin of this amendment (e.g. A2), not the release's")
args = parser.parse_args()

# pins of the earlier work: reported, never changed (they stay valid for the old branch and its results)
HISTORICAL_PINS = ["CODE_COMMIT.txt", "D3_CODE_COMMIT.txt", "D3_LCMV_XU/CODE_COMMIT.txt",
                   "D3_WMNE_XU/CODE_COMMIT.txt", "D3_LCMV_XU/ES200_CODE_COMMIT.txt",
                   "D3_WMNE_XU/ES200_CODE_COMMIT.txt", "D3_LCMV_XU/R1_CODE_COMMIT.txt",
                   "D3_WMNE_XU/R1_CODE_COMMIT.txt"]
# files the final notebooks call; an older pinned commit without them cannot serve these notebooks
REQUIRED = ["scripts/final.py", "scripts/build_stores.py", "src/adgraph/release.py", "src/adgraph/stores.py",
            "configs/final/protocol.yaml", "configs/source_graphs.yaml", "src/adgraph/colab.py"]


def git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


data_root = find_data_root()
output_root = choose_output_root(data_root)
head = git("rev-parse", "HEAD")
pin_file = output_root / release.PIN_FILE
if args.amendment:                     # an amendment's own pin; the release pin is not read for checkout
    from adgraph.experiments import amendment

    release_pin = pin_file.read_text().split()[0] if pin_file.exists() and pin_file.read_text().split() else None
    print(f"release pin (unchanged, not used for this notebook): {release_pin}")
    pin_file = output_root / amendment.pin_file(args.amendment)
    REQUIRED += ["src/adgraph/experiments/amendment.py", "configs/final/amendments.yaml"]
if args.no_pin:
    commit, needs_checkout = head, False
    print(f"code: cloned commit {commit} (maintenance mode, no code pin read or written)")
else:
    old_pin = pin_file.read_text().split()[0] if pin_file.exists() and pin_file.read_text().split() else None
    work = (any(pin_file.parent.glob("**/*.done.json")) if args.amendment
            else has_completed_work(output_root, "FINAL"))
    commit, needs_checkout = resolve_code_pin(pin_file, head, update=args.update_code, has_work=work)
    if old_pin and old_pin != commit:
        print(f"release code pin moved {old_pin[:10]} -> {commit[:10]} "
              + ("(UPDATE_CODE)" if args.update_code else "(no completed results were pinned to the old commit)"))
    if needs_checkout:
        missing = [f for f in REQUIRED
                   if subprocess.run(["git", "cat-file", "-e", f"{commit}:{f}"], cwd=REPO).returncode]
        if missing:
            raise SystemExit(f"The pinned code ({commit[:10]}, {pin_file}) is older than these notebooks (it lacks "
                             f"{missing}). Set UPDATE_CODE = True in the first cell to use the newest code.")
        sh(f"git checkout -q {commit}", cwd=REPO)
    print(f"code: commit {commit} ({'amendment ' + args.amendment if args.amendment else 'release ' + release.RELEASE}"
          f", pinned in {pin_file})"
          + (" - newer pushes are ignored until UPDATE_CODE" if needs_checkout else ""))
    for name in HISTORICAL_PINS:
        other = output_root / name
        if other.exists() and other.read_text().split():
            print(f"historical pin (old branch, unchanged): {name} = {other.read_text().split()[0][:10]}")

paths_file = write_paths_yaml(REPO, data_root, output_root)
print(f"data:    {data_root}\noutputs: {output_root}\npaths:   {paths_file}")
if not args.skip_install:
    sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=REPO)
    sh(f"{sys.executable} -m pip install -q --no-deps -e .", cwd=REPO)
print(f"setup complete: commit {commit}")
