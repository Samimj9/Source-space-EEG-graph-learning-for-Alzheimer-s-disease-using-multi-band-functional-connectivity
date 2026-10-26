"""One-step Colab setup of the A3 notebooks (``notebooks_a3/``; methodology log entry 61). Nothing to edit.

Locates AD-LEB, chooses the output folder (as ``colab_setup.py`` does), pins the A3 code in its OWN file
``<output>/FINAL_v1/amendments/A3_final/CODE_COMMIT.txt`` (moved freely until the freeze exists, afterwards only with
``--update-code``), writes ``configs/paths.yaml`` and installs the requirements. The release pin
(``FINAL_v1/CODE_COMMIT.txt``) and the A2 pin are only read and printed, never written; the pinned code of those two
runs in separate git worktrees made by ``scripts/a3.py holdout``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from adgraph import release  # noqa: E402
from adgraph.a3 import namespace as ns  # noqa: E402
from adgraph.colab import (choose_output_root, find_data_root, resolve_code_pin, sh,  # noqa: E402
                           write_paths_yaml)

parser = argparse.ArgumentParser()
parser.add_argument("--update-code", action="store_true", help="move the A3 code pin to the cloned commit")
parser.add_argument("--skip-install", action="store_true")
args = parser.parse_args()

REQUIRED = ["scripts/a3.py", "src/adgraph/a3/freeze.py", "configs/final/amendments/holdout_A3.yaml",
            "src/adgraph/colab.py"]


def git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


data_root = find_data_root()
output_root = choose_output_root(data_root)
head = git("rev-parse", "HEAD")
pin_file = ns.pin_path(output_root)
frozen = ns.master_path(output_root).exists()
old = ns.read_pin(pin_file)
commit, needs_checkout = resolve_code_pin(pin_file, head, update=args.update_code, has_work=frozen)
if old and old != commit:
    print(f"A3 code pin moved {old[:10]} -> {commit[:10]}")
if needs_checkout:
    missing = [f for f in REQUIRED if subprocess.run(["git", "cat-file", "-e", f"{commit}:{f}"], cwd=REPO).returncode]
    if missing:
        raise SystemExit(f"the pinned A3 code ({commit[:10]}) lacks {missing}; use UPDATE_CODE = True")
    sh(f"git checkout -q {commit}", cwd=REPO)
print(f"A3 code: commit {commit} (pinned in {pin_file})" + (" - newer pushes ignored until UPDATE_CODE" if needs_checkout
                                                              else ""))
rec = ns.record()
for name, f, want in (("release pin", ns.pin_file(output_root, "release"), rec["release_pin"]),
                      ("A2 pin", ns.pin_file(output_root, "A2"), rec["a2_pin"])):
    have = ns.read_pin(f)
    print(f"{name} (read only, never written here): {have}" + ("" if have == want else f"   !! expected {want}"))
paths_file = write_paths_yaml(REPO, data_root, output_root)
print(f"data:    {data_root}\noutputs: {output_root}\npaths:   {paths_file}\nfreeze:  "
      + ("EXISTS (a3.py verify checks it)" if frozen else "not written yet"))
if not args.skip_install:
    sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=REPO)
    sh(f"{sys.executable} -m pip install -q --no-deps -e .", cwd=REPO)
print(f"A3 setup complete: commit {commit}")
