"""On-disk schema of a graph-construction run (the single interface between
graph construction and the experiments).

A run directory contains

    graphs.npz     per-window arrays (index i <-> row i of windows.csv):
                     plv, ciplv            (n_windows, n_bands, n_rois, n_rois) float32
                                           symmetric, zero diagonal, in [0, 1]; ciplv = |ciPLV|
                     ciplv_signed          same shape, antisymmetric, in [-1, 1] (optional;
                                           sign = sign of the non-zero-lag phase relation
                                           for ROI order (j, k), not a direction of
                                           influence; never used as an adjacency)
                     psd_abs/psd_log/psd_rel            (n_windows, n_bands, n_rois) float32
                     sensor_psd_abs/_log/_rel           (n_windows, n_bands, n_channels) float32
                   and constants: bands, band_edges, roi_names, channels, W,
                   roi_coords (n_rois, 3) MNI mm label centroids (optional; visualization only)
    windows.csv    one row per window: identifiers + QC (see WINDOW_COLUMNS)
    subjects.csv   one row per subject: metadata + QC
    manifest.json  provenance (config, hash, commit, versions, operator check)
    roi_timeseries/<key>.npy   optional, continuous (n_rois, n_times) float32 per subject;
                   a window is ``ts[:, start_sample:stop_sample]``

PLV and ciPLV are kept as separate arrays and are never combined here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CONNECTIVITY = ("plv", "ciplv")            # adjacency-valid (non-negative, symmetric) measures
PSD_KINDS = ("abs", "log", "rel")
ARRAY_KEYS = CONNECTIVITY + ("ciplv_signed",) + tuple(f"psd_{k}" for k in PSD_KINDS) + tuple(f"sensor_psd_{k}" for k in PSD_KINDS)
WINDOW_COLUMNS = ("window_id", "key", "dataset", "subject_id", "label", "y", "window_index",
                  "start_sample", "stop_sample", "tmin", "tmax", "sample_weight")


def window_id(key: str, window_index: int) -> str:
    """Stable window identifier: subject key + index among all candidate windows (before rejection)."""
    return f"{key}-w{window_index:04d}"


def check_adjacency(a: np.ndarray, name: str = "adjacency", atol: float = 1e-5) -> None:
    """Adjacency contract: symmetric, zero diagonal, values in [0, 1]. a (..., R, R)."""
    if not np.allclose(a, np.swapaxes(a, -1, -2), atol=atol):
        raise ValueError(f"{name}: not symmetric")
    if np.abs(np.diagonal(a, axis1=-2, axis2=-1)).max() > atol:
        raise ValueError(f"{name}: non-zero diagonal")
    if a.min() < -atol or a.max() > 1 + atol:
        raise ValueError(f"{name}: values outside [0, 1] (signed ciPLV must not be used as an adjacency)")


@dataclass
class GraphStore:
    arrays: dict[str, np.ndarray]
    windows: pd.DataFrame
    bands: list[str]
    band_edges: np.ndarray
    roi_names: list[str]
    channels: list[str]
    subjects: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest: dict = field(default_factory=dict)
    root: Path | None = None
    roi_coords: np.ndarray | None = None

    def __post_init__(self):
        self.validate()

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def validate(self) -> None:
        missing = [c for c in WINDOW_COLUMNS if c not in self.windows.columns]
        if missing:
            raise ValueError(f"windows table lacks columns {missing}")
        if self.windows["window_id"].duplicated().any():
            raise ValueError("duplicate window_id")
        n, k, r, c = self.n_windows, len(self.bands), len(self.roi_names), len(self.channels)
        expected = {"plv": (n, k, r, r), "ciplv": (n, k, r, r), "ciplv_signed": (n, k, r, r)}
        expected.update({f"psd_{p}": (n, k, r) for p in PSD_KINDS})
        expected.update({f"sensor_psd_{p}": (n, k, c) for p in PSD_KINDS})
        for name, arr in self.arrays.items():
            if name in expected and arr.shape != expected[name]:
                raise ValueError(f"{name} has shape {arr.shape}, expected {expected[name]}")
        for name in CONNECTIVITY:
            if name in self.arrays and self.n_windows:
                check_adjacency(self.arrays[name], name)
        if "ciplv_signed" in self.arrays and self.n_windows:
            a = self.arrays["ciplv_signed"]
            if not np.allclose(a, -np.swapaxes(a, -1, -2), atol=1e-5) or np.abs(a).max() > 1 + 1e-5:
                raise ValueError("ciplv_signed must be antisymmetric with values in [-1, 1]")
        if self.roi_coords is not None and np.asarray(self.roi_coords).shape != (r, 3):
            raise ValueError(f"roi_coords shape {np.asarray(self.roi_coords).shape} != ({r}, 3)")
        if self.band_edges.shape != (k, 2):
            raise ValueError(f"band_edges shape {self.band_edges.shape} != ({k}, 2)")

    def band_index(self, names: list[str]) -> list[int]:
        unknown = [b for b in names if b not in self.bands]
        if unknown:
            raise KeyError(f"unknown bands {unknown}; available {self.bands}")
        return [self.bands.index(b) for b in names]

    def subset(self, idx: np.ndarray) -> "GraphStore":
        idx = np.asarray(idx)
        return GraphStore({k: v[idx] for k, v in self.arrays.items()}, self.windows.iloc[idx].reset_index(drop=True),
                          self.bands, self.band_edges, self.roi_names, self.channels, self.subjects, self.manifest,
                          self.root, self.roi_coords)

    def roi_timeseries(self, i: int) -> np.ndarray:
        """ROI time series of window ``i`` (requires a run saved with ``save_roi_timeseries``)."""
        if self.root is None:
            raise RuntimeError("store was not loaded from disk")
        row = self.windows.iloc[i]
        ts = np.load(self.root / "roi_timeseries" / f"{row['key']}.npy", mmap_mode="r")
        return np.asarray(ts[:, int(row["start_sample"]):int(row["stop_sample"])])

    # ---- persistence ----
    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "graphs.npz", **self.arrays, bands=np.array(self.bands),
                            band_edges=self.band_edges, roi_names=np.array(self.roi_names),
                            **({"roi_coords": np.asarray(self.roi_coords)} if self.roi_coords is not None else {}),
                            channels=np.array(self.channels))
        self.windows.to_csv(out_dir / "windows.csv", index=False)
        self.subjects.to_csv(out_dir / "subjects.csv", index=False)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        return out_dir

    @classmethod
    def load(cls, run_dir: str | Path, keys: tuple[str, ...] | None = None) -> "GraphStore":
        """Load a run. ``keys`` restricts which per-window arrays are read (saves memory)."""
        run_dir = Path(run_dir)
        npz = np.load(run_dir / "graphs.npz")
        wanted = [k for k in ARRAY_KEYS if k in npz.files and (keys is None or k in keys)]
        subjects_path = run_dir / "subjects.csv"
        manifest_path = run_dir / "manifest.json"
        return cls(
            arrays={k: npz[k] for k in wanted},
            windows=pd.read_csv(run_dir / "windows.csv"),
            bands=npz["bands"].tolist(), band_edges=npz["band_edges"], roi_names=npz["roi_names"].tolist(),
            channels=npz["channels"].tolist(),
            subjects=pd.read_csv(subjects_path) if subjects_path.exists() else pd.DataFrame(),
            manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {},
            root=run_dir,
            roi_coords=npz["roi_coords"] if "roi_coords" in npz.files else None,
        )
