"""Participant metadata for both datasets, normalized to one schema."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUMNS = ["key", "dataset", "subject_id", "label", "y", "age", "sex", "mmse", "acer", "comment"]


def subject_key(dataset: str, subject_id: int) -> str:
    return f"{dataset}-{int(subject_id)}"


def _find_col(df: pd.DataFrame, *prefixes: str) -> str | None:
    for col in df.columns:
        name = str(col).strip().lower()
        if any(name.startswith(p) for p in prefixes):
            return col
    return None


def _leading_int(value) -> int | None:
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def normalize_participants(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Map a raw participant sheet (D1 or D2 layout) onto ``COLUMNS``.

    D1 ids look like ``Sub3``; D2 ids look like ``3_1342019`` (subject_date).
    Either way the subject id is the leading integer.
    """
    id_col = _find_col(df, "id")
    group_col = _find_col(df, "group")
    if id_col is None or group_col is None:
        raise ValueError(f"{dataset}: participant sheet needs ID and Group columns, got {list(df.columns)}")
    age_col = _find_col(df, "age")
    sex_col = _find_col(df, "gender", "sex")
    mmse_col = _find_col(df, "mmse")
    acer_col = _find_col(df, "ace-r", "ace")
    comment_col = _find_col(df, "comment")

    rows = []
    for _, r in df.iterrows():
        sid = _leading_int(r[id_col])
        group = "" if pd.isna(r[group_col]) else str(r[group_col]).strip().upper()
        if sid is None or not group:
            continue  # blank or footer rows
        if group not in {"AD", "HC"}:
            raise ValueError(f"{dataset}: subject {r[id_col]!r} has unrecognised Group {r[group_col]!r} "
                             "(expected AD or HC); the label cannot be assigned")
        rows.append({
            "key": subject_key(dataset, sid),
            "dataset": dataset,
            "subject_id": sid,
            "label": group,
            "y": int(group == "AD"),
            "age": pd.to_numeric(r[age_col], errors="coerce") if age_col else float("nan"),
            "sex": str(r[sex_col]).strip().upper() if sex_col and not pd.isna(r[sex_col]) else "",
            "mmse": pd.to_numeric(r[mmse_col], errors="coerce") if mmse_col else float("nan"),
            "acer": pd.to_numeric(r[acer_col], errors="coerce") if acer_col else float("nan"),
            "comment": "" if comment_col is None or pd.isna(r[comment_col]) else str(r[comment_col]).strip(),
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    dup = out["key"].duplicated()
    if dup.any():
        raise ValueError(f"{dataset}: repeated subject ids in participant sheet: {out.loc[dup, 'key'].tolist()}")
    return out


def load_participants(path: str | Path, dataset: str) -> pd.DataFrame:
    return normalize_participants(pd.read_excel(path), dataset)
