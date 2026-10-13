"""One-step Colab setup, called by the first cell of every notebook (nothing to edit).

1. Locate the shared AD-LEB folder on the mounted Drive and choose the output folder
   (AD-LEB/processed if writable, else My Drive/AD-LEB-processed).
2. Pin the code: the first session records the commit in <output>/CODE_COMMIT.txt and every
   later session (any notebook) checks out that same commit. All results then come from one
   commit, and a newer push to GitHub never invalidates unfinished work. --update-code moves
   the pin to the newest commit deliberately (unfinished units are then recomputed).
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

from adgraph.colab import (choose_output_root, find_data_root, resolve_code_pin, sh,  # noqa: E402
                           write_paths_yaml)

parser = argparse.ArgumentParser()
parser.add_argument("--update-code", action="store_true", help="move the code pin to the cloned commit")
parser.add_argument("--skip-install", action="store_true")
args = parser.parse_args()


def git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


data_root = find_data_root()
output_root = choose_output_root(data_root)
head = git("rev-parse", "HEAD")
pin_file = output_root / "CODE_COMMIT.txt"
commit, needs_checkout = resolve_code_pin(pin_file, head, update=args.update_code)
if needs_checkout:
    sh(f"git checkout -q {commit}", cwd=REPO)
print(f"code: commit {commit} (pinned in {pin_file})" + (" - newer pushes are ignored until UPDATE_CODE"
                                                          if needs_checkout else ""))

paths_file = write_paths_yaml(REPO, data_root, output_root)
print(f"data:    {data_root}\noutputs: {output_root}\npaths:   {paths_file}")
if not args.skip_install:
    sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=REPO)
    sh(f"{sys.executable} -m pip install -q --no-deps -e .", cwd=REPO)
print(f"setup complete: commit {commit}")
