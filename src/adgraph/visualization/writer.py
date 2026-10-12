"""Saving figures with their exact plotting data and a manifest; never overwriting.

For every figure ``<id>``:
    <id>.pdf / .svg / .png    (formats from the style; PNG at style['dpi'])
    <id>_data.csv             exact numbers drawn (long format)  [or <id>_data.json]
    <id>_meta.json            figure kind (descriptive / inferential), statistical unit,
                              display-only choices, source files (sha256)
and once per run ``figure_manifest.json`` (config, style, sources, timestamp, git commit).
Any existing target file raises ``FileExistsError``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..provenance import git_commit, versions

SYNTHETIC_MARK = "SYNTHETIC DATA – NOT RESULTS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FigureWriter:
    def __init__(self, out_dir: str | Path, style: dict, config: dict | None = None, synthetic: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.style, self.config, self.synthetic = style, config or {}, synthetic
        self.figures: list[dict] = []
        self.sources: dict[str, dict] = {}

    def _target(self, name: str) -> Path:
        p = self.out_dir / name
        if p.exists():
            raise FileExistsError(f"{p} exists; figures are never overwritten (use a new output directory)")
        return p

    def _register_sources(self, sources) -> list[str]:
        out = []
        for s in sources or []:
            p = Path(s)
            key = str(p)
            if key not in self.sources:
                self.sources[key] = ({"sha256": sha256(p), "bytes": p.stat().st_size} if p.is_file()
                                     else {"sha256": None, "note": "directory or in-memory source"})
            out.append(key)
        return out

    def save(self, fig, fig_id: str, data, meta: dict, sources=None) -> dict:
        targets = [self._target(f"{fig_id}.{fmt}") for fmt in self.style["formats"]]
        data_path = self._target(f"{fig_id}_data.csv" if isinstance(data, pd.DataFrame) else f"{fig_id}_data.json")
        meta_path = self._target(f"{fig_id}_meta.json")
        if self.synthetic:
            fig.text(0.5, 0.5, SYNTHETIC_MARK, ha="center", va="center", rotation=30, fontsize=14,
                     color="0.6", alpha=0.35, transform=fig.transFigure, zorder=1000)
        for t in targets:
            fig.savefig(t, dpi=self.style["dpi"] if t.suffix == ".png" else None)
        plt.close(fig)
        if isinstance(data, pd.DataFrame):
            data.to_csv(data_path, index=False)
        else:
            data_path.write_text(json.dumps(data, indent=2, default=float))
        record = {"id": fig_id, **meta, "synthetic": self.synthetic, "files": [t.name for t in targets],
                  "data_file": data_path.name, "sources": self._register_sources(sources)}
        meta_path.write_text(json.dumps(record, indent=2, default=str))
        self.figures.append(record)
        return record

    def write_manifest(self) -> Path:
        path = self._target("figure_manifest.json")
        path.write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
            "versions": versions(("numpy", "pandas", "matplotlib", "scipy", "sklearn")),
            "synthetic": self.synthetic, "config": self.config, "style": self.style,
            "sources": self.sources, "figures": self.figures,
        }, indent=2, default=str))
        return path
