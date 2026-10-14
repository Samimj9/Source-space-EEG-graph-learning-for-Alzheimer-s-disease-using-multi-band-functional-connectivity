"""SHA-256 content fingerprints of input files (EDF recordings, participant sheets, layout).

Hashing ~2 GB of EDF over the Drive mount on every resume would be slow, so hashes are
cached in a JSON file keyed by absolute path and validated by ``(size, mtime_ns)``: a file
is re-hashed when either changes, or always with ``rehash=True`` (CLI ``--rehash-inputs``).
The cache only avoids re-reading unchanged files; identities use the content hash.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..checkpoint import atomic_write_json, file_sha256


class InputHashes:
    def __init__(self, cache_path: str | Path | None = None, rehash: bool = False):
        self.cache_path = Path(cache_path) if cache_path else None
        self.rehash = rehash
        self.n_hashed = 0                        # files actually read in this session
        self.cache: dict = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                self.cache = {}                  # unreadable cache: re-hash everything

    def sha256(self, path: str | Path) -> str:
        p = Path(path).absolute()
        st = p.stat()
        entry = self.cache.get(str(p))
        if (not self.rehash and entry and entry.get("size") == st.st_size
                and entry.get("mtime_ns") == st.st_mtime_ns):
            return entry["sha256"]
        sha = file_sha256(p)
        self.n_hashed += 1
        self.cache[str(p)] = {"sha256": sha, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                              "hashed_at": datetime.now(timezone.utc).isoformat()}
        if self.cache_path:
            atomic_write_json(self.cache_path, self.cache)   # saved per file: survives a disconnect
        return sha


def auxiliary_inputs(paths: dict, datasets=("D1", "D2"), layout: bool = True) -> dict[str, str]:
    """Non-EEG input files that are part of a run's provenance (those present in ``paths``)."""
    files = {}
    for ds_name in datasets:
        if ds_name == "D3":                       # OpenNeuro ds004504 (BIDS)
            root = Path(paths.get("dataset3", {}).get("root", ""))
            files["participants_D3"] = str(root / "participants.tsv")
            files["dataset_description_D3"] = str(root / "dataset_description.json")
            continue
        ds = paths.get(f"dataset{ds_name[1]}", {})
        if ds.get("participants"):
            files[f"participants_{ds_name}"] = ds["participants"]
    if layout and paths.get("electrode_layout"):
        files["electrode_layout"] = paths["electrode_layout"]
    return files
