"""Cross-dataset gate (DESIGN.md §7): checked on the graph run before S4 is allowed."""

from __future__ import annotations

from ..graphs.store import GraphStore


def gate_failures(store: GraphStore, tol: float = 1e-6) -> list[str]:
    """Reasons why cross-dataset testing is not allowed on this graph run (empty list = pass)."""
    m = store.manifest
    fails = []
    op = m.get("source_operator", {})
    if not op.get("shared_across_datasets", False):
        fails.append("source operator not shared across D1 and D2")
    err = op.get("verification_rel_error")
    if err is None or err > tol:
        fails.append(f"operator verification error {err} > {tol}")
    if op.get("channels") and op["channels"] != store.channels:
        fails.append("channel list in manifest differs from stored channels")
    datasets = set(store.windows["dataset"])
    if datasets != {"D1", "D2"}:
        fails.append(f"graph run does not contain both datasets: {sorted(datasets)}")
    subj = store.subjects
    if not subj.empty and "status" in subj:
        bad = subj.loc[subj["status"].astype(str).str.startswith("error"), "key"].tolist()
        if bad:
            fails.append(f"subjects failed in graph construction: {bad}")
    if not subj.empty and "median_std_uv" in subj:
        ok = subj[subj["status"] == "ok"] if "status" in subj else subj
        r = ok.groupby("dataset")["median_std_uv"].agg(["min", "max"])
        if {"D1", "D2"} <= set(r.index) and (r.loc["D1", "max"] < r.loc["D2", "min"]
                                             or r.loc["D2", "max"] < r.loc["D1", "min"]):
            fails.append("D1 and D2 amplitude ranges (median_std_uv) do not overlap: check units")
    if m.get("registry", {}).get("subset_limit"):
        fails.append("graph run was a --limit smoke test")
    return fails
