"""Electrode layout (EEGLAB ``.loc``) and index-based EDF channel labels.

The AD-LEB recordings label their 32 channels by number only: Dataset 1 as ``EEG E01`` …
``EEG E32`` and Dataset 2 as ``Misc ExG1`` … ``Misc ExG32``. Both were recorded with the
same 32-electrode cap, whose order is given by ``AD-LEB/electrodes_positions_32.loc``
(EEGLAB polar format: ``index theta radius label``; labels padded with dots, CRLF lines).
Channel *k* of an EDF is the electrode with index *k* in that file.

``configs/pipeline.yaml`` (``channels.recorded``) holds the same order; ``verify_layout``
checks it against the ``.loc`` file at run time, so the mapping never relies on the config
alone. The original files are only read, never modified.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# "EEG E01", "E1", "Misc ExG1", "ExG 01", optionally with trailing dots / spaces
INDEXED_LABEL = re.compile(r"^\s*(?:EEG\s*)?(?:MISC\s*)?(?:EXG|E)\s*0*(\d+)\s*\.*\s*$", re.IGNORECASE)


def clean_label(label: str) -> str:
    """``'F7..'`` -> ``'F7'`` (EEGLAB pads labels with dots)."""
    return str(label).strip().rstrip(".").strip()


def parse_loc(path: str | Path) -> pd.DataFrame:
    """Rows ``index, theta, radius, label`` of an EEGLAB ``.loc`` file, validated:
    indices 1..N contiguous and unique, labels non-empty and unique."""
    rows = []
    for n, line in enumerate(Path(path).read_text(errors="strict").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}: line {n} is not 'index theta radius label': {line!r}")
        rows.append({"index": int(parts[0]), "theta": float(parts[1]), "radius": float(parts[2]),
                     "label": clean_label(parts[3])})
    df = pd.DataFrame(rows, columns=["index", "theta", "radius", "label"])
    if df.empty:
        raise ValueError(f"{path}: no electrodes")
    if sorted(df["index"]) != list(range(1, len(df) + 1)) or df["index"].duplicated().any():
        raise ValueError(f"{path}: electrode indices must be 1..{len(df)} without gaps or repeats, "
                         f"got {df['index'].tolist()}")
    if (df["label"] == "").any() or df["label"].str.lower().duplicated().any():
        raise ValueError(f"{path}: empty or repeated electrode labels {df['label'].tolist()}")
    return df.sort_values("index").reset_index(drop=True)


def verify_layout(path: str | Path, recorded: list[str]) -> list[str]:
    """Raise unless the ``.loc`` electrode order equals ``recorded`` (case-insensitive);
    returns the labels in index order."""
    labels = parse_loc(path)["label"].tolist()
    if [x.lower() for x in labels] != [x.lower() for x in recorded]:
        raise ValueError(f"electrode layout {path} order {labels} differs from channels.recorded {recorded}; "
                         "index-based channel labels cannot be mapped safely")
    return labels


def indexed_number(name: str) -> int | None:
    """Channel number of an index-only label (``'EEG E07'`` -> 7, ``'Misc ExG12'`` -> 12), else None."""
    m = INDEXED_LABEL.match(str(name))
    return int(m.group(1)) if m else None


def find_layout_file(data_root: str | Path) -> Path:
    """The 32-channel ``.loc`` in the AD-LEB folder (root first, then EEG subfolders; a
    browser-download copy such as ``electrodes_positions_32 (1).loc`` is accepted if it is the
    only one, or identical in content to the others)."""
    root = Path(data_root)
    exact = root / "electrodes_positions_32.loc"
    if exact.is_file():
        return exact
    found = sorted({*root.glob("electrodes_positions_32*.loc"), *root.glob("EEGs/electrodes_positions_32*.loc"),
                    *root.glob("EEGs/*/electrodes_positions_32*.loc")})
    if not found:
        raise FileNotFoundError(f"no electrodes_positions_32*.loc under {root} (root or EEGs/)")
    contents = {p.read_bytes() for p in found}
    if len(contents) > 1:
        raise ValueError(f"several different layout files: {[str(p) for p in found]}")
    return found[0]


def spatial_consistency(data: np.ndarray, positions: np.ndarray) -> float:
    """Spearman correlation between inter-electrode distance and |signal correlation|.

    Volume conduction makes nearby electrodes correlate more than distant ones, so a correct
    channel-to-position mapping gives a clearly negative value; a wrong (e.g. shifted or
    shuffled) mapping gives a value near zero. QC only; it changes nothing."""
    from scipy.stats import spearmanr

    c = np.abs(np.corrcoef(data))
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    iu = np.triu_indices(len(positions), 1)
    return float(spearmanr(d[iu], c[iu]).statistic)


def check_layout(paths: dict, recorded: list[str]) -> dict:
    """Locate and verify the electrode layout named in ``paths`` (``electrode_layout``; falls back
    to ``find_layout_file(drive_root)``). Raises if it is missing or its order differs from
    ``recorded``. Returns ``{"path", "sha256", "labels"}`` for logs and manifests."""
    import hashlib

    path = Path(paths["electrode_layout"]) if paths.get("electrode_layout") else None
    if path is None or not path.is_file():
        if not paths.get("drive_root"):
            raise FileNotFoundError(f"electrode layout not found: {path}")
        path = find_layout_file(paths["drive_root"])
    labels = verify_layout(path, recorded)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "labels": labels}
