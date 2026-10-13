"""Discovery and loading of raw EDF recordings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .layout import indexed_number
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
    s = re.sub(r"[-_](REF|LE|AVG|AR|A1A2|A1|A2|M1M2|M1|M2)$", "", s, flags=re.IGNORECASE)   # reference suffix
    s = s.replace(" ", "")
    lookup = {v.lower(): v for v in valid}
    return lookup.get(s.lower())


class ChannelMappingError(ValueError):
    """The EDF channel inventory cannot be mapped to electrodes unambiguously."""


def channel_mapping(ch_names: list[str], valid: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return ``{raw_name: canonical}`` for recognised channels and the unrecognised names.

    Two label conventions are accepted, never mixed within one recording:

    * **named**: 10-20 labels (``'Fp1'``, ``'EEG FP1-REF'``, ``'Fp1.'``);
    * **indexed**: number-only labels, channel *k* = electrode *k* of the layout, where
      ``valid`` is the layout order (``channels.recorded``, verified against the ``.loc``
      file): D1 ``'EEG E01'``-``'EEG E32'``, D2 ``'Misc ExG1'``-``'Misc ExG32'``, also
      ``'E1'``, ``'ExG 01'``, trailing dots. Numbers outside 1..len(valid) are unrecognised.

    Raises ``ChannelMappingError`` for a mixed inventory or a repeated channel number."""
    named = {ch: normalize_channel_name(ch, valid) for ch in ch_names}
    numbers = {ch: indexed_number(ch) for ch in ch_names}
    if any(named.values()) and any(n is not None and named[ch] is None for ch, n in numbers.items()):
        raise ChannelMappingError(f"mixed channel naming (10-20 names and index-only labels): {list(ch_names)}")
    seen = [n for n in numbers.values() if n is not None and 1 <= n <= len(valid)]
    if len(seen) != len(set(seen)):
        dup = sorted({n for n in seen if seen.count(n) > 1})
        raise ChannelMappingError(f"channel number(s) {dup} occur more than once: {list(ch_names)}")
    mapping, unknown = {}, []
    for ch in ch_names:
        canon = named[ch]
        if canon is None and numbers[ch] is not None and 1 <= numbers[ch] <= len(valid):
            canon = valid[numbers[ch] - 1]
        if canon is None or canon in mapping.values():
            unknown.append(ch)
        else:
            mapping[ch] = canon
    return mapping, unknown


def mapping_mode(mapping: dict[str, str]) -> str:
    """'indexed' if the recognised channels use number-only labels, else 'named'."""
    return "indexed" if mapping and all(indexed_number(ch) is not None for ch in mapping) else "named"


def load_raw(path: str | Path, dataset: str, cfg: dict):
    """Read an EDF, keep the configured EEG channels in canonical order, apply the unit scale.

    Returns ``(raw, info)`` where ``info`` records what was done for the manifest.
    Raises if any of the common (non-dropped) channels is missing.
    """
    import mne

    # Header only first; only the channels that are used are then read into memory
    # (identical samples, without loading ECG/EOG/mastoids at the original sampling rate).
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
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
    raw.load_data(verbose="error")

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
        "channel_naming": mapping_mode(mapping),
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
        try:
            mapping, unknown = channel_mapping(raw.ch_names, valid)
            error = ""
        except ChannelMappingError as exc:     # reported per recording; the strict gate fails on it
            mapping, unknown, error = {}, list(raw.ch_names), str(exc)
        units = sorted({str(u) for u in (getattr(raw, "_orig_units", None) or {}).values()})   # header units
        rows.append({
            "key": rec.key, "dataset": rec.dataset, "n_channels": len(raw.ch_names),
            "sfreq": raw.info["sfreq"], "duration_s": raw.n_times / raw.info["sfreq"],
            "naming": mapping_mode(mapping) if mapping else "unmapped",
            "missing_canonical": ",".join(c for c in valid if c not in mapping.values()),
            "unrecognised": ",".join(unknown), "mapping_error": error,
            "mapping": ",".join(f"{k}->{v}" for k, v in mapping.items()),
            "raw_names": ",".join(raw.ch_names), "header_units": ",".join(units),
        })
    return pd.DataFrame(rows)


def robust_channel_std(data: np.ndarray) -> float:
    """Median over channels of the per-channel standard deviation."""
    return float(np.median(np.std(data, axis=1)))
