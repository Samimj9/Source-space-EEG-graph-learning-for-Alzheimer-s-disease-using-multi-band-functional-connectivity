"""Dataset 3: OpenNeuro ds004504 (BIDS), preprocessed derivatives, AD and HC only.

ds004504 (Miltiadous et al., 2023; doi:10.18112/openneuro.ds004504) contains 88 eyes-closed resting-state
recordings: Group A (Alzheimer's disease, 36), F (frontotemporal dementia, 23) and C (healthy controls, 29).
Dataset 3 uses ONLY groups A and C, from the provider's preprocessed derivatives
(``derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set``); FTD is excluded by design (pre-specified,
methodology log entry 44), never because of results. The subject key is ``D3-<participant number>``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .participants import COLUMNS, subject_key

DATASET = "D3"
GROUPS = {"A": "AD", "C": "HC"}             # provider group code -> label used in this project
EXCLUDED_GROUPS = {"F": "frontotemporal dementia (FTD): excluded by design"}
EXPECTED = {"AD": 36, "HC": 29}
DERIVATIVE = re.compile(r"^sub-(\d+)_task-eyesclosed_eeg\.set$")


def participants_path(root: str | Path) -> Path:
    return Path(root) / "participants.tsv"


def load_participants_tsv(path: str | Path, dataset: str = DATASET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(kept rows in the project schema, excluded rows: key + group + reason). Unknown groups raise."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    need = {"participant_id", "Group"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}, got {list(df.columns)}")
    rows, excluded = [], []
    for _, r in df.iterrows():
        m = re.fullmatch(r"sub-(\d+)", str(r["participant_id"]).strip())
        if not m:
            raise ValueError(f"{path}: unexpected participant_id {r['participant_id']!r}")
        sid, group = int(m.group(1)), str(r["Group"]).strip().upper()
        key = subject_key(dataset, sid)
        if group in EXCLUDED_GROUPS:
            excluded.append({"key": key, "group": group, "reason": EXCLUDED_GROUPS[group]})
            continue
        if group not in GROUPS:
            raise ValueError(f"{path}: {r['participant_id']} has unknown Group {r['Group']!r}")
        label = GROUPS[group]
        rows.append({"key": key, "dataset": dataset, "subject_id": sid, "label": label, "y": int(label == "AD"),
                     "age": pd.to_numeric(r.get("Age"), errors="coerce"),
                     "sex": str(r.get("Gender", "")).strip().upper() if pd.notna(r.get("Gender")) else "",
                     "mmse": pd.to_numeric(r.get("MMSE"), errors="coerce"), "acer": float("nan"), "comment": ""})
    kept = pd.DataFrame(rows, columns=COLUMNS)
    if kept["key"].duplicated().any():
        raise ValueError(f"{path}: repeated participant ids {kept.loc[kept['key'].duplicated(), 'key'].tolist()}")
    return kept, pd.DataFrame(excluded, columns=["key", "group", "reason"])


def check_cohort(kept: pd.DataFrame, expected: dict | None = None) -> None:
    expected = expected or EXPECTED
    counts = kept["label"].value_counts().to_dict()
    if {k: counts.get(k, 0) for k in expected} != expected:
        raise ValueError(f"ds004504 cohort is {counts}, expected {expected} (AD and HC of release 1.0.x)")


def list_recordings(root: str | Path, dataset: str = DATASET) -> pd.DataFrame:
    """Preprocessed derivative recordings: key, dataset, subject_id, path, size_bytes, mtime."""
    rows = []
    deriv = Path(root) / "derivatives"
    for p in sorted(deriv.glob("sub-*/eeg/*.set")) if deriv.is_dir() else []:
        m = DERIVATIVE.match(p.name)
        if m:
            sid = int(m.group(1))
            rows.append({"key": subject_key(dataset, sid), "dataset": dataset, "subject_id": sid, "path": str(p),
                         "size_bytes": p.stat().st_size, "mtime": p.stat().st_mtime})
    df = pd.DataFrame(rows, columns=["key", "dataset", "subject_id", "path", "size_bytes", "mtime"])
    if df["key"].duplicated().any():
        raise ValueError(f"{dataset}: several recordings for {df.loc[df['key'].duplicated(), 'key'].tolist()}")
    return df.sort_values("subject_id").reset_index(drop=True)
