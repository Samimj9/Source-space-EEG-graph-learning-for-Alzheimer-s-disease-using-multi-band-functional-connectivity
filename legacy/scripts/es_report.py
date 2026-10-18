"""Early-stopping summary of the 200-epoch / patience-10 experiments (methodology log entry 56); saved outputs only.

    python scripts/es_report.py --suite LX      # LCMV-Xu  (D3_LCMV_XU/experiments_*/lx*_es_*)
    python scripts/es_report.py --suite WX      # wMNE-Xu  (D3_WMNE_XU/experiments_*/wx*_es_*)

Reads hparams.csv of every early-stopping (variant, setting) directory (one row per outer fit: epochs run, best epoch,
best inner-validation loss, inner validation participants, loss curves) and writes a new
<suite>/reports/early_stopping_<time>/ (never overwritten): summary.csv (per mode x experiment x variant x setting:
number of fits, best / stopping epoch median and range, fits stopped before 200, best validation loss) and
curves_<mode>_<experiment>.png (train and inner-validation loss of every fit, best epoch marked).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adgraph.config import load_paths

parser = argparse.ArgumentParser()
parser.add_argument("--suite", choices=["LX", "WX"], required=True)
parser.add_argument("--paths", default="configs/paths.yaml")
args = parser.parse_args()
root = Path(load_paths(args.paths)["processed_dir"]) / {"LX": "D3_LCMV_XU", "WX": "D3_WMNE_XU"}[args.suite]
prefix = args.suite.lower()
rows, curves = [], {}
for mode, sub in (("A", "experiments_A_xu_tenfold"), ("B", "experiments_B_all_windows")):
    for hp_path in sorted((root / sub).glob(f"{prefix}*_es_*/*/*/hparams.csv")):
        d = hp_path.parent
        exp, variant, setting = d.parent.parent.name, d.parent.name, d.name
        hp = pd.read_csv(hp_path)
        if "best_epoch" not in hp:
            continue
        rows.append({"mode": mode, "experiment": exp, "variant": variant, "setting": setting, "n_fits": len(hp),
                     "complete": (d / "final.done.json").exists(),
                     "best_epoch_median": float(hp["best_epoch"].median()), "best_epoch_min": int(hp["best_epoch"].min()),
                     "best_epoch_max": int(hp["best_epoch"].max()),
                     "epochs_run_median": float(hp["epochs_run"].median()),
                     "n_stopped_before_max": int(hp["stopped_early"].astype(bool).sum()),
                     "best_val_loss_mean": float(hp["best_val_loss"].mean()),
                     "n_inner_val_participants": int(hp["n_inner_val_participants"].iloc[0])})
        for r in hp.itertuples():
            curves.setdefault((mode, exp), []).append((variant, setting, r.fold, json.loads(r.history_train_loss),
                                                       json.loads(r.history_val_loss), int(r.best_epoch)))
out = root / "reports" / f"early_stopping_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
table = pd.DataFrame(rows)
table.to_csv(out / "summary.csv", index=False)
if len(table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (mode, exp), cs in curves.items():
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for variant, setting, fold, trl, val, best in cs:
            ax[0].plot(np.arange(1, len(trl) + 1), trl, lw=0.6, alpha=0.6)
            ax[1].plot(np.arange(1, len(val) + 1), val, lw=0.6, alpha=0.6)
            ax[1].plot(best, val[best - 1], "k.", ms=3)
        ax[0].set(title="training loss", xlabel="epoch")
        ax[1].set(title="inner-validation loss (dot = best epoch)", xlabel="epoch")
        fig.suptitle(f"{exp}, mode {mode}: every outer fit ({len(cs)})")
        fig.tight_layout()
        fig.savefig(out / f"curves_{mode}_{exp}.png", dpi=120)
        plt.close(fig)
    print(table.round(3).to_string(index=False))
print(f"\n{len(table)} early-stopping units -> {out}")
