"""Report the code pins on Drive and restore the D1/D2 pin (methodology log entry 45). Never touches results.

    python scripts/code_pin.py status
    python scripts/code_pin.py restore-d1d2            # the last D1/D2 pin that predates Dataset 3
    python scripts/code_pin.py restore-d1d2 --commit 1b6988aae58ea128cd8fe52b60b95a451db9871f

Pins: ``CODE_COMMIT.txt`` (D1/D2 graphs and experiments 0-6), ``D3_CODE_COMMIT.txt`` (Dataset 3),
``experiments/exp7_earm_gnn/CODE_COMMIT.txt`` (Experiment 7). ``status`` prints each pin with its history and the
commits recorded by the completion markers of each results tree. ``restore-d1d2`` sets the D1/D2 pin back to
``--commit``, or by default to the most recent pin in its history that does not contain the first Dataset 3
commit (FIRST_D3_COMMIT), i.e. the pin before a D3 session moved it. The old lines stay in the file as history.
It refuses if any D1/D2 unit was completed under the current pin's commit, because those units would then no
longer resume (nothing would be deleted either way).
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from adgraph.colab import resolve_code_pin  # noqa: E402
from adgraph.config import load_paths  # noqa: E402

PINS = {"D1/D2": "CODE_COMMIT.txt", "D3": "D3_CODE_COMMIT.txt", "Exp7": "experiments/exp7_earm_gnn/CODE_COMMIT.txt"}
FIRST_D3_COMMIT = "f1353ad7d48f18365cbaee4294d965e19b8e6ead"     # "Add Dataset 3: OpenNeuro ds004504 AD vs HC ..."
TREES = {"D1/D2": ["graphs_v*/subjects/*/subject.done.json", "experiments/**/*.done.json"],
         "D3": ["graphsD3_v*/subjects/*/subject.done.json", "experiments_D3/**/*.done.json"]}


def pinned(path: Path) -> str | None:
    text = path.read_text() if path.exists() else ""
    return next((ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")), None)


def marker_commits(out: Path, family: str, exclude_exp7: bool = False) -> Counter:
    c = Counter()
    for pattern in TREES[family]:
        for m in out.glob(pattern):
            if exclude_exp7 and "exp7_earm_gnn" in m.parts:
                continue                                  # Exp7 has its own pin
            try:
                c[json.loads(m.read_text()).get("git_commit") or "unknown"] += 1
            except (OSError, json.JSONDecodeError):
                c["unreadable"] += 1
    return c


def status(out: Path) -> None:
    for name, rel in PINS.items():
        p = out / rel
        print(f"\n{name} pin ({rel}): {pinned(p) or 'not set'}")
        if p.exists():
            print("  " + "\n  ".join(p.read_text().strip().splitlines()))
    for fam in TREES:
        c = marker_commits(out, fam)
        print(f"\ncompleted {fam} units by commit:", dict(c) if c else "none")


def contains_d3(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", FIRST_D3_COMMIT, commit], cwd=REPO).returncode == 0


def pre_d3_pin(pin: Path) -> str | None:
    """Newest pin (current, then history newest first) whose commit predates Dataset 3."""
    lines = pin.read_text().splitlines() if pin.exists() else []
    cands = [pinned(pin)] + [ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("# previous pin:")]
    return next((c for c in cands if c and not contains_d3(c)), None)


def restore_d1d2(out: Path, commit: str | None, force: bool) -> int:
    if commit is None:
        commit = pre_d3_pin(out / PINS["D1/D2"])
        if commit is None:
            print("no D1/D2 pin predating Dataset 3 in the pin history; pass --commit")
            return 1
    full = subprocess.run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=REPO, text=True,
                          capture_output=True)
    if full.returncode:
        print(f"{commit} is not a commit of this repository")
        return 1
    commit = full.stdout.strip()
    pin = out / PINS["D1/D2"]
    current = pinned(pin)
    if current == commit:
        print(f"D1/D2 pin already {commit}: unchanged")
        return 0
    used = marker_commits(out, "D1/D2", exclude_exp7=True).get(current, 0) if current else 0
    if used and not force:
        print(f"refused: {used} D1/D2 unit(s) were completed under the current pin {current}; restoring {commit[:10]} "
              "would stop them from resuming. Nothing changed.")
        return 1
    resolve_code_pin(pin, commit, update=True)
    lines = pin.read_text().splitlines()
    lines.insert(2, f"# restored by scripts/code_pin.py {datetime.now(timezone.utc).isoformat()} (was {current}; "
                    "D3 has its own pin D3_CODE_COMMIT.txt)")
    pin.write_text("\n".join(lines) + "\n")
    print(f"D1/D2 pin restored: {current} -> {commit}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "restore-d1d2"])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--commit", help="restore-d1d2: the commit to pin (default: last pin predating D3)")
    parser.add_argument("--force", action="store_true", help="restore even if units were completed under the pin")
    args = parser.parse_args()
    out = Path(load_paths(args.paths)["processed_dir"])
    if args.action == "status":
        status(out)
        sys.exit(0)
    code = restore_d1d2(out, args.commit, args.force)
    status(out)
    sys.exit(code)
