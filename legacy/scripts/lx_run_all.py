"""Optional master runner of the D3 LCMV / Xu suite: Exp 0-7 in order, mode A (tenfold, fast) then mode B.

    python scripts/lx_run_all.py [--modes A B] [--experiments 0 1 2 3 4 5 6 7]

Each experiment is run with scripts/experiment.py (resumable: completed units are skipped). A failing experiment
is reported and the next one continues; the exit code is 1 if any failed. Then scripts/lx_report.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = {"A": REPO / "configs/experiments/lx/tenfold", "B": REPO / "configs/experiments/lx"}
parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--modes", nargs="+", default=["A", "B"], choices=["A", "B"])
parser.add_argument("--experiments", nargs="+", type=int, default=list(range(8)))   # Exp 8: own notebook + pin
args = parser.parse_args()
failed = []
for mode in args.modes:
    for n in args.experiments:
        cfg = next(CONFIGS[mode].glob(f"lx*_exp{n}_*.yaml"), None)
        if cfg is None:                             # not in the checked-out commit
            print(f"mode {mode} Exp {n}: no config in this commit, skipped")
            continue
        print(f"\n===== mode {mode} · Exp {n} · {cfg.relative_to(REPO)} =====", flush=True)
        r = subprocess.run([sys.executable, "scripts/experiment.py", "run", "--config", str(cfg), "--paths", args.paths],
                           cwd=REPO)
        if r.returncode:
            failed.append(f"mode {mode} Exp {n}")
subprocess.run([sys.executable, "scripts/lx_report.py", "--paths", args.paths], cwd=REPO)
print("\nfailed: " + (", ".join(failed) if failed else "none"))
sys.exit(1 if failed else 0)
