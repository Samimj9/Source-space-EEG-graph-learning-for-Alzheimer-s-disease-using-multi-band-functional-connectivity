"""The matched 19-electrode representation of D2 and its head geometry (``configs/cross_dataset/channel_map_19.yaml``).

D2 EDFs carry channel numbers only (``Misc ExG1`` ... ``Misc ExG32``); EDF channel *k* is electrode *k* of
``AD-LEB/electrodes_positions_32.loc``. The mapping to the 19 D3 electrodes is an explicit, version-controlled table
and is never inferred (in particular never "the first 19 channels"). Everything is verified, and any failure raises
``MappingError`` immediately:

* ``load_table``      table structure: 19 rows = the D3 store channels in their order, unique EDF indices, the four
                      documented equivalences (T3=T7, T4=T8, T5=P7, T6=P8) and identity for all other electrodes;
* ``check_layout``    the layout file is the identified one (SHA-256) and its label at every table index is the
                      table's D2 label, with the table's polar coordinates;
* ``check_geometry``  the colin27 10-05 positions of the D2 labels equal those of the D3 labels and the table's
                      ``mni_mm``; every .loc direction agrees with its position within ``max_angle_deg``; the
                      angle-minimising assignment of .loc directions to positions is the table (no swap); odd
                      electrodes left, even right, midline electrodes on the midline;
* ``edf_picks``       the EDF channels to read, in canonical order: every one of the 19 numbers present exactly
                      once, labelled ``Misc ExG<k>``; the rest of the inventory consistent with the 32-electrode layout;
* ``harmonized_geometry``  the D2 head geometry = the established D3 geometry (positions of every D3 .set file,
                      MRI/MNI frame): one shared forward model; the D3 geometry must equal the table's positions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..io.layout import indexed_number, parse_loc

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "cross_dataset" / "channel_map_19.yaml"
POSITION_TOL_MM = 0.01


class MappingError(ValueError):
    """The 19-electrode mapping, the layout file, the geometry or an EDF inventory is not as verified."""


# ------------------------------------------------------------------------------------------------------- the table
def load_table(path: str | Path | None = None) -> dict:
    t = yaml.safe_load(Path(path or CONFIG).read_text())
    rows = t["channels"]
    order = t["canonical_order"]
    eq = t["equivalences"]
    d3 = [r["d3"] for r in rows]
    if d3 != order or len(order) != 19 or len(set(order)) != 19:
        raise MappingError(f"table rows {d3} are not the 19 canonical channels in canonical order {order}")
    idx = [int(r["edf_index"]) for r in rows]
    if len(set(idx)) != 19 or not all(1 <= i <= int(t["layout_n_electrodes"]) for i in idx):
        raise MappingError(f"EDF indices must be 19 distinct numbers in 1..{t['layout_n_electrodes']}: {idx}")
    for r in rows:
        if r["d2"] != eq.get(r["d3"], r["d3"]):
            raise MappingError(f"{r['d3']} is mapped to {r['d2']}, expected {eq.get(r['d3'], r['d3'])} (only "
                               f"{eq} differ by name)")
        if r["edf_label"] != t["d2_edf_label"].format(index=r["edf_index"]):
            raise MappingError(f"{r['d3']}: EDF label {r['edf_label']!r} does not name index {r['edf_index']}")
    if set(t["excluded_d2"]) & {r["d2"] for r in rows}:
        raise MappingError("an excluded D2 electrode is also mapped")
    if set(eq) != {"T3", "T4", "T5", "T6"} or eq != {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}:
        raise MappingError(f"unexpected equivalences {eq}")
    return t


def frame(table: dict) -> pd.DataFrame:
    """The table as a DataFrame (one row per canonical channel, canonical order)."""
    df = pd.DataFrame(table["channels"])
    df.insert(0, "canonical_position", range(len(df)))
    xyz = np.array(df.pop("mni_mm").tolist(), float)
    df["mni_x_mm"], df["mni_y_mm"], df["mni_z_mm"] = xyz.T
    return df


def table_hash(table: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(table, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------- layout
def check_layout(path: str | Path, table: dict) -> pd.DataFrame:
    """Verify the .loc file against the table; returns the parsed layout (index, theta, radius, label)."""
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != table["layout_sha256"]:
        raise MappingError(f"{path} has SHA-256 {sha}, not the verified layout {table['layout_sha256']}")
    loc = parse_loc(path)
    if len(loc) != int(table["layout_n_electrodes"]):
        raise MappingError(f"{path}: {len(loc)} electrodes, expected {table['layout_n_electrodes']}")
    by_index = loc.set_index("index")
    for r in table["channels"]:
        row = by_index.loc[int(r["edf_index"])]
        if row["label"].lower() != r["d2"].lower():
            raise MappingError(f"layout electrode {r['edf_index']} is {row['label']!r}, the table says {r['d2']!r}")
        if abs(row["theta"] - r["loc_theta_deg"]) > 1e-9 or abs(row["radius"] - r["loc_radius"]) > 1e-9:
            raise MappingError(f"{r['d2']}: layout polar ({row['theta']}, {row['radius']}) differs from the table")
    excluded = set(loc["label"]) - {r["d2"] for r in table["channels"]}
    if excluded != set(table["excluded_d2"]):
        raise MappingError(f"unmapped layout electrodes {sorted(excluded)} differ from excluded_d2")
    return loc


def loc_direction(theta_deg, radius) -> np.ndarray:
    """EEGLAB polar (theta from the nose, positive to the right; radius 0.5 = equator) -> unit vectors in RAS."""
    th, ph = np.radians(np.asarray(theta_deg, float)), np.radians(np.asarray(radius, float) * 180.0)
    return np.stack([np.sin(ph) * np.sin(th), np.sin(ph) * np.cos(th), np.cos(ph)], axis=-1)


def fit_sphere(pos: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere through points (n, 3): (centre, radius)."""
    a = np.c_[2 * pos, np.ones(len(pos))]
    sol = np.linalg.lstsq(a, (pos ** 2).sum(1), rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + c @ c))


# ------------------------------------------------------------------------------------------------------- geometry
def template_positions(labels: list[str], template: str = "colin27_1005") -> np.ndarray:
    """Positions (m, MRI/MNI frame) of ``labels`` in an MNE template montage."""
    import mne

    pos = mne.channels.make_standard_montage(template).get_positions()
    if pos["coord_frame"] != "mri":
        raise MappingError(f"{template} is not in the mri frame")
    lower = {k.lower(): v for k, v in pos["ch_pos"].items()}
    missing = [x for x in labels if x.lower() not in lower]
    if missing:
        raise MappingError(f"{template} has no position for {missing}")
    return np.array([lower[x.lower()] for x in labels], float)


def check_geometry(table: dict, positions_m: np.ndarray | None = None) -> pd.DataFrame:
    """Geometry audit of the mapping (raises on any failure). ``positions_m``: the positions the forward model uses
    (default: the table's ``mni_mm``). Returns one row per channel with the angle to its .loc direction."""
    rows = table["channels"]
    mni = np.array([r["mni_mm"] for r in rows], float) / 1000
    pos = mni if positions_m is None else np.asarray(positions_m, float)
    d3_t = template_positions([r["d3"] for r in rows], table["template"])
    d2_t = template_positions([r["d2"] for r in rows], table["template"])
    for name, other in (("D2 labels", d2_t), ("D3 labels", d3_t), ("forward-model positions", pos)):
        dev = np.linalg.norm(other - mni, axis=1).max() * 1000
        if dev > POSITION_TOL_MM:
            raise MappingError(f"{name}: {table['template']} positions deviate from the table by {dev:.3f} mm")
    centre, radius = fit_sphere(pos)
    vec = (pos - centre) / np.linalg.norm(pos - centre, axis=1, keepdims=True)
    loc = loc_direction([r["loc_theta_deg"] for r in rows], [r["loc_radius"] for r in rows])
    cos = np.clip(loc @ vec.T, -1, 1)
    ang = np.degrees(np.arccos(cos))
    own = np.diag(ang)
    from scipy.optimize import linear_sum_assignment

    _, assign = linear_sum_assignment(ang)
    problems = []
    if (own > table["max_angle_deg"]).any():
        problems.append(f"angles above {table['max_angle_deg']} deg: "
                        f"{[(r['d3'], round(a, 1)) for r, a in zip(rows, own) if a > table['max_angle_deg']]}")
    if not (assign == np.arange(len(rows))).all():
        problems.append(f"angle-minimising assignment differs from the table: "
                        f"{[(rows[i]['d2'], rows[j]['d3']) for i, j in enumerate(assign) if i != j]}")
    for r, p in zip(rows, pos):
        side = r["d3"][-1]
        if side in "13579" and p[0] >= 0 or side in "2468" and p[0] <= 0:
            problems.append(f"{r['d3']} at x = {p[0] * 1000:.1f} mm is on the wrong side")
        if side == "z" and abs(p[0]) > 0.005:
            problems.append(f"{r['d3']} at x = {p[0] * 1000:.1f} mm is not on the midline")
        th = r["loc_theta_deg"]
        if r["loc_radius"] > 0 and (side in "13579" and not th < 0 or side in "2468" and not th > 0):
            problems.append(f"{r['d2']}: .loc theta {th} on the wrong side")
    if problems:
        raise MappingError("; ".join(problems))
    nearest = ang.argmin(axis=0)
    return pd.DataFrame({"d3": [r["d3"] for r in rows], "d2": [r["d2"] for r in rows],
                         "edf_index": [r["edf_index"] for r in rows], "angle_to_loc_deg": own.round(2),
                         "nearest_loc_electrode": [rows[j]["d2"] for j in nearest],
                         "sphere_radius_mm": round(radius * 1000, 1)})


def template_geometry(table: dict) -> dict:
    """A geometry record (``io.chanlocs`` layout) with the table's colin27 positions, for synthetic runs and tests."""
    from ..io.chanlocs import geometry_hash

    g = {"source": "colin27_1005_table", "frame": table["frame"], "lookup": "colin27_1005",
         "channels": list(table["canonical_order"]),
         "pos_m": np.round(np.array([r["mni_mm"] for r in table["channels"]], float) / 1000, 7).tolist()}
    g["hash"] = geometry_hash(g)
    g["template_deviation_mm"] = 0.0
    g["file_fiducials"] = []
    return g


def harmonized_geometry(reference: dict, table: dict) -> dict:
    """The D2 geometry: exactly the positions of the established D3 geometry ``reference`` (same channels, same
    order, same frame), verified against the table and the .loc directions. Recorded as its own source so the
    provenance is explicit; the positions (and hence the forward model) are those of D3."""
    from ..io.chanlocs import geometry_hash

    if list(reference["channels"]) != list(table["canonical_order"]):
        raise MappingError(f"reference geometry channels {reference['channels']} != {table['canonical_order']}")
    if reference["frame"] != table["frame"]:
        raise MappingError(f"reference geometry frame {reference['frame']!r} != {table['frame']!r}")
    pos = np.asarray(reference["pos_m"], float)
    check_geometry(table, pos)
    g = {"source": "d3_reference_geometry", "frame": reference["frame"], "lookup": reference.get("lookup", ""),
         "channels": list(reference["channels"]), "pos_m": reference["pos_m"],
         "reference_hash": reference["hash"], "template_deviation_mm": reference.get("template_deviation_mm"),
         "file_fiducials": []}
    g["hash"] = geometry_hash(g)
    return g


# ------------------------------------------------------------------------------------------------------- EDF
def edf_picks(ch_names: list[str], table: dict) -> list[tuple[str, str]]:
    """(EDF channel name, canonical D3 name) of the 19 channels, in canonical order. The header must hold every
    layout number 1..32 exactly once as an index-only label; the 19 mapped ones must be ``Misc ExG<k>``."""
    numbers = [indexed_number(c) for c in ch_names]
    n_layout = int(table["layout_n_electrodes"])
    used = [n for n in numbers if n is not None]
    if sorted(used) != list(range(1, n_layout + 1)):
        dup = sorted({n for n in used if used.count(n) > 1})
        raise MappingError(f"EDF channel numbers are not 1..{n_layout} exactly once (repeated {dup}, missing "
                           f"{sorted(set(range(1, n_layout + 1)) - set(used))}): {list(ch_names)}")
    by_number = {n: c for c, n in zip(ch_names, numbers) if n is not None}
    out = []
    for r in table["channels"]:
        name = by_number[int(r["edf_index"])]
        if name.strip().lower() != r["edf_label"].lower():
            raise MappingError(f"EDF channel {r['edf_index']} is labelled {name!r}, expected {r['edf_label']!r}")
        out.append((name, r["d3"]))
    return out


def audit(table: dict, layout_path: str | Path | None = None, headers: pd.DataFrame | None = None,
          reference: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The channel-mapping audit: one row per canonical channel (mapping, EDF index, coordinates, angle) and a
    summary of every check. ``headers``: one row per D2 recording with ``key`` and ``raw_names`` (comma-separated
    EDF labels). Raises on the first failed check."""
    summary = {"table_hash": table_hash(table), "n_channels": len(table["channels"]),
               "equivalences": table["equivalences"], "canonical_order": table["canonical_order"]}
    if layout_path is not None:
        check_layout(layout_path, table)
        summary["layout"] = {"path": str(layout_path), "sha256": table["layout_sha256"], "verified": True}
    geo = check_geometry(table, None if reference is None else np.asarray(reference["pos_m"], float))
    summary["geometry"] = {"max_angle_deg": float(geo["angle_to_loc_deg"].max()), "assignment": "table",
                           "positions": "D3 reference geometry" if reference is not None else "table (colin27_1005)",
                           **({"reference_hash": reference["hash"]} if reference is not None else {})}
    if headers is not None:
        per = []
        for h in headers.itertuples():
            picks = edf_picks(str(h.raw_names).split(","), table)
            per.append({"key": h.key, "n_channels": len(str(h.raw_names).split(",")),
                        "picked": ",".join(p for p, _ in picks)})
        summary["recordings"] = {"n": len(per), "all_19_present_once": True,
                                 "identical_picks": len({p["picked"] for p in per}) <= 1}
    df = frame(table).merge(geo[["d3", "angle_to_loc_deg", "nearest_loc_electrode"]], on="d3")
    return df, summary
