"""Participant registry, sensor geometry and store discovery for the source-graph stores (ds004504 only).

These helpers were part of the D1/D2 pipeline module (``legacy/src/adgraph/pipeline.py``) and are carried over
unchanged for Dataset 3, the only dataset of the final framework:

* ``build_registry``: recordings joined to participant metadata (AD and HC; FTD excluded by design);
* ``design_exclusions``: the FTD participants excluded by the pre-specified design rule;
* ``canonical_info``: the 19-channel sensor ``Info`` (electrode positions from the recordings, average reference);
* ``list_graph_runs`` / ``latest_complete_graph_run``: assembled stores of one name prefix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import progress
from ..config import common_channels
from ..graphs.store import GraphStore, require_complete_cohort
from ..io import chanlocs
from ..io.eeg import make_montage
from ..io.fingerprint import InputHashes


def canonical_info(cfg: dict, geometry: dict | None = None):
    """Sensor ``Info``: common channels, electrode positions (``geometry`` from the recordings, else the configured
    template montage), average-reference projection applied."""
    import mne

    info = mne.create_info(common_channels(cfg), cfg["preprocessing"]["sfreq"], "eeg")
    info.set_montage(chanlocs.montage(geometry) if geometry is not None else make_montage(cfg["channels"]["montage"]))
    raw = mne.io.RawArray(np.zeros((len(info["ch_names"]), 1)), info, verbose="error")
    raw.set_eeg_reference("average", projection=True, verbose="error")
    raw.apply_proj(verbose="error")
    return raw.info


def design_exclusions(paths: dict, cfg: dict | None = None) -> dict:
    """Recordings excluded by a pre-specified design rule (not by results): the ds004504 FTD participants."""
    from ..io import bids

    _, excluded = bids.load_participants_tsv(bids.participants_path(paths["dataset3"]["root"]))
    return {"D3": excluded.to_dict("records")}


def build_registry(paths: dict, limit: int | None = None, hashes: InputHashes | None = None, cfg: dict | None = None):
    """Recordings joined to participant metadata: every AD and HC participant of ds004504 as provided (FTD
    excluded by design). With ``hashes``, a ``sha256`` column holds each recording's content hash.

    Returns (registry, recordings, keys without metadata, keys without a recording)."""
    from ..io import bids

    root = paths["dataset3"]["root"]
    kept, excluded = bids.load_participants_tsv(bids.participants_path(root))
    bids.check_cohort(kept)
    r = bids.list_recordings(root)
    recs = r[~r["key"].isin(excluded["key"])].reset_index(drop=True)          # FTD recordings are never used
    meta = kept.reset_index(drop=True)
    registry = recs.merge(meta.drop(columns=["dataset", "subject_id"]), on="key", how="outer", indicator=True)
    no_meta = registry.loc[registry["_merge"] == "left_only", "key"].tolist()
    no_rec = registry.loc[registry["_merge"] == "right_only", "key"].tolist()
    registry = (registry[registry["_merge"] != "right_only"].drop(columns="_merge")
                .sort_values(["dataset", "subject_id"]).reset_index(drop=True))
    if limit:
        registry = registry.groupby("dataset", group_keys=False).head(limit).reset_index(drop=True)
    if hashes is not None:
        shas = []
        with progress.reporter(len(registry), "input SHA-256", unit="files", eta=False) as bar:
            for p in registry["path"]:
                bar.set(_file=Path(p).name)
                shas.append(hashes.sha256(p))
                bar.advance(timed=False)
        registry["sha256"] = shas
    return registry, recs, no_meta, no_rec


def list_graph_runs(processed_dir: str | Path, prefix: str) -> pd.DataFrame:
    """All store directories ``<prefix>_v*`` with their state (assembled or not, smoke test or full)."""
    rows = []
    for d in sorted(Path(processed_dir).glob(f"{prefix}_v*")):
        if not d.is_dir():
            continue
        man = d / "manifest.json"
        assembled = (d / "assembly.done.json").exists() and man.exists()
        m = json.loads(man.read_text()) if assembled else {}
        rows.append({"run_dir": str(d), "assembled": assembled, "limit_smoke_test": "_limit" in d.name,
                     "forced": "_force" in d.name, "assembled_at": m.get("assembled_at", ""),
                     "graph_run_id": m.get("graph_run_id", ""), "config_hash": m.get("config_hash", ""),
                     "version": int((m.get("config") or {}).get("version", 0) or 0)})
    return pd.DataFrame(rows, columns=["run_dir", "assembled", "limit_smoke_test", "forced", "assembled_at",
                                       "graph_run_id", "config_hash", "version"])


def latest_complete_graph_run(processed_dir: str | Path, prefix: str) -> Path:
    """Assembled full (non-smoke-test) store of the highest pipeline version, most recent first; raises if there
    is none (never falls back to an older or incomplete store)."""
    runs = list_graph_runs(processed_dir, prefix)
    runs = runs[runs["assembled"] & ~runs["limit_smoke_test"]].sort_values(["version", "assembled_at"])
    if runs.empty:
        raise FileNotFoundError(f"no assembled store {prefix}_v* in {processed_dir}")
    latest = Path(runs["run_dir"].iloc[-1])
    require_complete_cohort(GraphStore.load(latest, keys=()))
    return latest
