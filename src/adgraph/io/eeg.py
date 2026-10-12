"""Discovery and loading of raw EDF recordings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .participants import subject_key

# File-name patterns of the resting-state recordings.
PATTERNS = {
    "D1": re.compile(r"^sub(\d+)\.edf$", re.IGNORECASE),
    "D2": re.compile(r"^(\d+)_\d+_1\.edf$", re.IGNORECASE),
}


def list_recordings(eeg_dir: str | Path, dataset: str) -> pd.DataFrame:
    """One row per resting-state recording: key, dataset, subject_id, path, size_bytes, mtime."""
    rows = []
    for p in sorted(Path(eeg_dir).iterdir()):
        m = PATTERNS[dataset].match(p.name)
        if m:
            sid = int(m.group(1))
            rows.append({"key": subject_key(dataset, sid), "dataset": dataset, "subject_id": sid,
                         "path": str(p), "size_bytes": p.stat().st_size, "mtime": p.stat().st_mtime})
    df = pd.DataFrame(rows, columns=["key", "dataset", "subject_id", "path", "size_bytes", "mtime"])
    if df["key"].duplicated().any():
        raise ValueError(f"{dataset}: several recordings for {df.loc[df['key'].duplicated(), 'key'].tolist()}")
    return df.sort_values("subject_id").reset_index(drop=True)


def normalize_channel_name(name: str, valid: list[str]) -> str | None:
    """Map an EDF label (e.g. ``'EEG FP1-REF'``, ``'Fp1.'``) to a canonical name, or None."""
    s = name.strip().rstrip(".").strip()
    s = re.sub(r"^(EEG|EXG)[\s_-]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_](REF|LE|AVG|AR)$", "", s, flags=re.IGNORECASE)
    s = s.replace(" ", "")
    lookup = {v.lower(): v for v in valid}
    return lookup.get(s.lower())


def channel_mapping(ch_names: list[str], valid: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return ``{raw_name: canonical}`` for recognised channels and the unrecognised names."""
    mapping, unknown = {}, []
    for ch in ch_names:
        canon = normalize_channel_name(ch, valid)
        if canon is None or canon in mapping.values():
            unknown.append(ch)
        else:
            mapping[ch] = canon
    return mapping, unknown


def load_raw(path: str | Path, dataset: str, cfg: dict):
    """Read an EDF, keep the configured EEG channels in canonical order, apply the unit scale.

    Returns ``(raw, info)`` where ``info`` records what was done for the manifest.
    Raises if any of the common (non-dropped) channels is missing.
    """
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    recorded = cfg["channels"]["recorded"]
    mapping, unknown = channel_mapping(raw.ch_names, recorded)
    raw.pick(list(mapping))
    raw.rename_channels(mapping)

    keep = [ch for ch in recorded if ch not in cfg["channels"]["drop"]]
    missing = [ch for ch in keep if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"{path}: missing channels {missing}; found {raw.ch_names}")
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.pick(keep)
    raw.reorder_channels(keep)

    scale = float(cfg["units"]["scale"][dataset])
    if scale != 1.0:
        raw.apply_function(lambda x: x * scale, picks="all", channel_wise=False)
    raw.set_montage(make_montage(cfg["channels"]["montage"]), match_case=False)

    info = {
        "orig_sfreq": float(raw.info["sfreq"]),
        "duration_s": float(raw.n_times / raw.info["sfreq"]),
        "unit_scale": scale,
        "dropped_channels": [c for c in mapping.values() if c not in keep],
        "unrecognised_channels": unknown,
    }
    return raw, info


def make_montage(name: str):
    """Built-in montage by name. MNE >= 1.13 renamed ``standard_1020`` to ``colin27_1020``
    (identical positions), so the new name is used when available."""
    import mne

    if name == "standard_1020" and "colin27_1020" in mne.channels.get_builtin_montages():
        name = "colin27_1020"
    return mne.channels.make_standard_montage(name)


def inspect_headers(recordings: pd.DataFrame, valid: list[str]) -> pd.DataFrame:
    """Header-only scan (no signal loaded): channel names, sampling rate, duration, units."""
    import mne

    rows = []
    for rec in recordings.itertuples():
        raw = mne.io.read_raw_edf(rec.path, preload=False, verbose="error")
        mapping, unknown = channel_mapping(raw.ch_names, valid)
        units = sorted({str(ch.get("unit_mul", "")) for ch in raw.info["chs"]})
        rows.append({
            "key": rec.key, "dataset": rec.dataset, "n_channels": len(raw.ch_names),
            "sfreq": raw.info["sfreq"], "duration_s": raw.n_times / raw.info["sfreq"],
            "missing_canonical": ",".join(c for c in valid if c not in mapping.values()),
            "unrecognised": ",".join(unknown), "raw_names": ",".join(raw.ch_names), "unit_mul": ",".join(units),
        })
    return pd.DataFrame(rows)


def robust_channel_std(data: np.ndarray) -> float:
    """Median over channels of the per-channel standard deviation."""
    return float(np.median(np.std(data, axis=1)))
