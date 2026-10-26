"""Discovery of completed experiment results across the release and the A2 namespace (read only).

A result counts as completed only if its ``final.done.json`` marker exists and every payload file still has the SHA-256
the marker recorded. Directories without a valid marker (the failed original Exp 8 units, partial runs) are listed
separately as NOT results. Nothing is copied, moved or written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..evaluation.reporting import summary_row
from ..experiments.report import KEY
from . import namespace as ns

FAMILIES = {
    "exp0": ("Exp0", "Sensor-space classical baseline"), "exp1": ("Exp1", "Source-space classical baseline"),
    "exp2": ("Exp2", "Minimal GCN (connectivity only)"), "exp3": ("Exp3", "Single-band GCN (PSD node features)"),
    "exp4": ("Exp4", "Band-averaged GCN"), "exp5": ("Exp5", "Multibranch GNN"),
    "exp6": ("Exp6", "PSD / strength / band ablations"), "exp7": ("Exp7", "EARM-GNN (exploratory)"),
    "exp8": ("Exp8", "Xu-inspired dual-adjacency GCN (original; failed on lobes12)"),
    "exp8a": ("Exp8a", "Xu-inspired dual-adjacency GCN, surface layer (amendment A2)"),
}
CLASSICAL = ("exp0", "exp1")
UNIT_KEY = ("experiment", "variant", "inverse", "mode", "atlas", "graph")


def exp_id(experiment: str) -> str:
    return experiment.split("_")[0]


def _marker_ok(d: Path) -> tuple[bool, str]:
    from ..checkpoint import file_sha256

    mp = d / "final.done.json"
    if not mp.exists():
        return False, "no completion marker"
    try:
        marker = json.loads(mp.read_text())
    except (OSError, json.JSONDecodeError):
        return False, "unreadable marker"
    for rel, sha in marker.get("files", {}).items():
        p = d / rel
        if not p.exists():
            return False, f"missing {rel}"
        if file_sha256(p) != sha:
            return False, f"checksum mismatch {rel}"
    return True, ""


def _dirs(results_root: Path):
    yield from sorted(p.parent for p in Path(results_root).glob("*/mode_*/*/*/*/*/*/_identity.json"))


def discover(processed, namespaces=ns.PARTS, settings=("xu_dev_cv",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(completed results, results that are not completed) over the given namespaces. ``settings`` limits what is even
    opened: the default reads development-CV directories only, so a development report or a freeze can never load a
    holdout metric (the holdout report passes ``("xu_holdout",)``)."""
    done, other = [], []
    for part in namespaces:
        rr = ns.results_root(processed, part)
        commit = ns.part_pin(part)
        for d in _dirs(rr):
            ident = json.loads((d / "_identity.json").read_text()).get("identity", {})
            code = ident.get("code") or {}
            ctx = ident.get("context") or {}
            if settings and ident.get("setting") not in settings:
                continue
            base = {"namespace": part, **{k: ctx.get(k) for k in ("inverse", "mode", "atlas", "graph")},
                    "experiment": ident.get("experiment"), "variant": ident.get("variant"),
                    "setting": ident.get("setting"), "git_commit": code.get("git_commit"),
                    "source_sha256": code.get("source_sha256"), "libraries": json.dumps(code.get("libraries")),
                    "graph_run_id": ident.get("graph_run_id"), "expected_commit": commit,
                    "commit_matches_pin": code.get("git_commit") == commit, "rel_dir": str(d.relative_to(processed))}
            ok, why = _marker_ok(d)
            if ok and (d / "metrics.json").exists():
                m = json.loads((d / "metrics.json").read_text())
                done.append({**base, **summary_row(m), "n_ad": m["n_ad"], "n_hc": m["n_hc"],
                             "threshold": m.get("threshold"), "score_type": m.get("score_type"),
                             "metrics_sha256": ns.sha256_file(d / "metrics.json"),
                             "predictions_sha256": ns.sha256_file(d / "predictions_participants.csv")})
            else:
                state, detail = "not completed", why
                st = d / "status.csv"
                if st.exists():
                    s = pd.read_csv(st)
                    col = "state" if "state" in s else s.columns[-1]
                    states = s[col].astype(str).value_counts().to_dict()
                    state = "failed" if states.get("failed") else "partial" if states.get("completed") else "not started"
                    detail = f"{why}; units {states}"
                other.append({**base, "state": state, "detail": detail})
    t = pd.DataFrame(done)
    if len(t):
        t["exp"] = t["experiment"].map(exp_id)
        t["family"] = t["exp"].map(lambda e: FAMILIES.get(e, (e, e))[0])
        t["family_title"] = t["exp"].map(lambda e: FAMILIES.get(e, (e, e))[1])
    else:
        t = pd.DataFrame(columns=["namespace", "exp"] + KEY + ["auc", "git_commit"])
    o = pd.DataFrame(other)
    if len(o):
        o["exp"] = o["experiment"].map(exp_id)
        o["family"] = o["exp"].map(lambda e: FAMILIES.get(e, (e, e))[0])
    return t, o


def analysis_role(row, atlas: str, graph: str) -> str:
    """confirmatory | amended | exploratory | sensitivity, for the development rows of the consolidated report."""
    if row["setting"] == "xu_holdout":
        return "holdout (descriptive)"
    exp, mode = row["exp"], row["mode"]
    classical = exp in CLASSICAL
    on_record = (row["atlas"] == ("dk68" if exp == "exp0" else atlas)
                 and (classical or row["graph"] == graph))
    if exp == "exp7":
        return "exploratory (Exp 7, designed after earlier results)"
    if mode != "B":
        return "sensitivity (window mode A)" + (", amended (A2)" if exp == "exp8a" else "")
    if not on_record:
        return "exploratory (atlas / graph comparison, not the recorded analysis)"
    if exp == "exp8a":
        return "amended (A2; post-design, pre-result)"
    return "confirmatory (pre-specified)"


def with_roles(table: pd.DataFrame, atlas: str, graph: str) -> pd.DataFrame:
    t = table.copy()
    if len(t):
        t["role"] = t.apply(lambda r: analysis_role(r, atlas, graph), axis=1)
    else:
        t["role"] = []
    return t
