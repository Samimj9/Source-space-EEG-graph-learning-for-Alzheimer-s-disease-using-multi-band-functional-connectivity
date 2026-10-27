"""Matched-channel D2 source-graph stores (``configs/cross_dataset/d2_store.yaml``; design section 4-5).

The configuration of a D2 store is the validated D3 store configuration of the same inverse on lobes12
(``sourcegraphs.config.resolve(inverse, 'lobes12')``) with only the documented D2 overrides (``resolve``). The
per-participant pipeline is the D3 one (``sourcegraphs.build.process_subject``: segments, five-minute selection,
adaptive rejection, LCMV from the participant's own accepted samples or the data-independent wMNE operator, analytic
FIR features, PLV / |ciPLV| / PSD / DE) with the D2 loader ``load_d2``:

1. EDF header only; the 19 channels of ``channel_map_19.yaml`` are selected BY NUMBER (``channels.edf_picks``), renamed
   to the D3 labels and put in the D3 store order; the 13 other channels are never read;
2. ``edge_crop_s`` discarded at each end (before any filtering);
3. 0.5-45 Hz zero-phase 4th-order Butterworth (the provider's documented D3 filter; order not documented by the
   provider) on the 19 channels; then the per-channel unit / amplitude checks (``check_units``);
3b. the acquisition-specific 50 Hz notch (``line_noise``: zero-phase FIR, firwin, stop band 49-51 Hz), D2 only;
4. ``settle_s`` discarded at each end (covers the band-pass and notch edge transients);
5. head geometry = the D3 positions (``channels.harmonized_geometry``): one forward model shared with D3;
6. then exactly the D3 path: per-segment resampling to 250 Hz, common average of the SAME 19 electrodes, ...

Stores: ``<processed>/CROSS_D3_D2_19CH_v1/stores/graphsXD2{L,W}{,10}_v1_<hash>`` (mode B / A), validated by
``validate`` (``validation.json``). The original EDFs are only read.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import progress
from ..checkpoint import StatusTable, atomic_write_csv, atomic_write_json, check_unit, code_identity, stable_hash, \
    supersede
from ..config import config_hash
from ..experiments.config import deep_merge
from ..provenance import git_commit
from . import channels, namespace

log = logging.getLogger(__name__)
SPEC = Path(__file__).resolve().parents[3] / "configs" / "cross_dataset" / "d2_store.yaml"
VERSION = 1
EXPECTED_D2 = {"n": 23, "AD": 10, "HC": 13}
MODES = {"B": 0, "A": 1}


class D2StoreError(RuntimeError):
    pass


class UnitError(D2StoreError):
    """A selected channel's unit or amplitude is not that of an EEG signal in volts (fail fast; nothing rescaled)."""


def spec(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or SPEC).read_text())


def resolve(inverse: str, table: dict | None = None, path: str | Path | None = None) -> dict:
    """The D2 store configuration of ``inverse``: the D3 lobes12 configuration + the D2 overrides only."""
    from ..sourcegraphs.config import resolve as resolve_d3

    s = spec(path)
    table = table or channels.load_table()
    base = resolve_d3(inverse, s["atlas"])
    over = copy.deepcopy(s["overrides"])
    coords = over["channels"].pop("coordinates")
    cfg = deep_merge(base, over)
    cfg["channels"]["recorded"] = list(table["canonical_order"])
    cfg["channels"]["drop"] = []
    cfg["channels"]["montage"] = coords["source"]
    cfg["channels"]["coordinates"] = coords                       # replaced, not merged (no EEGLAB lookups in D2)
    cfg["channels"]["channel_map_hash"] = channels.table_hash(table)
    cfg["units"]["scale"] = dict(over["units"]["scale"])          # D2 only
    cfg.pop("split", None)                                        # the ds004504 partition does not apply to D2
    cfg["output_subdir"] = f"{namespace.ROOT}/stores"
    cfg["store_names"] = list(s["store_names"][inverse])
    cfg["cross_dataset"] = {"base_d3_config_hash": config_hash(base), "spec_version": s["version"],
                            "dataset": "D2", "inverse": inverse, "atlas": s["atlas"]}
    return cfg


def store_dirs(processed, inverse: str, limit: int | None = None) -> tuple[Path, Path]:
    from ..sourcegraphs.build import run_dirs

    return run_dirs(processed, resolve(inverse), limit)


def store_dir(processed, inverse: str, mode: str, limit: int | None = None) -> Path:
    return store_dirs(processed, inverse, limit)[MODES[mode]]


# ---------------------------------------------------------------------------------------------- registry / headers
def registry(paths: dict, limit: int | None = None, hashes=None) -> pd.DataFrame:
    """The D2 resting-state recordings joined to the 2019 participant sheet (all 23 as provided)."""
    from ..io.eeg import list_recordings
    from ..io.participants import load_participants

    eeg_dir = Path(paths["dataset2"]["eeg_dir"])
    if eeg_dir.name != "RestingState":
        raise D2StoreError(f"{eeg_dir} is not the RestingState folder (PictureNaming / ButtonPress are never used)")
    recs = list_recordings(eeg_dir, "D2")
    meta = load_participants(paths["dataset2"]["participants"], "D2")
    reg = recs.merge(meta.drop(columns=["dataset", "subject_id"]), on="key", how="outer", indicator=True)
    no_meta = reg.loc[reg["_merge"] == "left_only", "key"].tolist()
    if no_meta:
        raise D2StoreError(f"recordings without a participant row: {no_meta}")
    reg = reg[reg["_merge"] == "both"].drop(columns="_merge").sort_values("subject_id").reset_index(drop=True)
    if reg["key"].duplicated().any() or reg["path"].duplicated().any():
        raise D2StoreError("duplicate participant or recording")
    if limit:
        reg = reg.head(limit).reset_index(drop=True)
    elif (len(reg), int(reg["y"].sum()), int((reg["y"] == 0).sum())) != tuple(EXPECTED_D2.values()):
        raise D2StoreError(f"D2 cohort is {len(reg)} ({int(reg['y'].sum())} AD), expected {EXPECTED_D2}")
    if hashes is not None:
        reg["sha256"] = [hashes.sha256(p) for p in reg["path"]]
    return reg


TECHNICAL = ("label", "sampling_frequency", "physical_dimension", "prefiltering", "transducer_type",
             "physical_min", "physical_max", "digital_min", "digital_max")


def header_audit(reg: pd.DataFrame) -> pd.DataFrame:
    """Technical EDF header fields of every recording (header only; no signal, no patient / recording identity
    fields): channel labels, rates, units, prefiltering, transducer, ranges, duration, and the distinct annotation
    texts (evidence on the recording condition, decision D6)."""
    import edfio

    rows = []
    for rec in reg.itertuples():
        edf = edfio.read_edf(rec.path, lazy_load_data=True)
        sig = edf.signals
        per = {k: sorted({str(getattr(s, k)) for s in sig}) for k in TECHNICAL if k != "label"}
        ann = sorted({str(a.text).strip() for a in edf.annotations if str(a.text).strip()})
        rows.append({"key": rec.key, "n_signals": len(sig), "raw_names": ",".join(s.label for s in sig),
                     "annotations": "|".join(ann)[:500],       # event texts only (e.g. eyes open / closed markers)
                     "duration_s": float(edf.duration),
                     "data_record_duration_s": float(edf.data_record_duration),
                     **{k: ";".join(v) for k, v in per.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- loader
def load_d2(path, dataset: str, cfg: dict, table: dict, geometry: dict):
    """(raw of the 19 canonical channels after crop, band-pass and settle crop, record); see the module doc."""
    import mne

    from ..io import chanlocs

    if dataset != "D2":
        raise D2StoreError(f"load_d2 reads D2 only, got {dataset}")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    sf = float(raw.info["sfreq"])
    picks = channels.edf_picks(raw.ch_names, table)
    n_bound = sum(str(d).strip().lower() == "boundary" for d in raw.annotations.description)
    if n_bound:
        raise D2StoreError(f"{Path(path).name}: {n_bound} boundary annotation(s); D2 recordings are continuous")
    idx = [raw.ch_names.index(name) for name, _ in picks]
    units = channel_units(raw, [name for name, _ in picks])
    bad_units = {n: u for n, u in units.items() if u not in cfg["units"]["physical_dimensions"]}
    if bad_units:
        raise UnitError(f"{Path(path).name}: unsupported EDF physical dimension(s) {bad_units}")
    x = raw.get_data(picks=idx) * float(cfg["units"]["scale"][dataset])
    if x.shape[0] != 19:
        raise D2StoreError(f"{Path(path).name}: read {x.shape[0]} channels")
    pp = cfg["preprocessing"]
    n_edge, n_settle = int(round(pp["edge_crop_s"] * sf)), int(round(pp["settle_s"] * sf))
    n_total = x.shape[1]
    if n_total <= 2 * (n_edge + n_settle):
        raise D2StoreError(f"{Path(path).name}: {n_total / sf:.1f} s is shorter than the crops")
    x = x[:, n_edge:n_total - n_edge]
    bp = pp["bandpass"]
    if (bp["method"], bp["ftype"], bp["phase"]) != ("iir", "butter", "zero"):
        raise D2StoreError(f"unsupported band-pass {bp}")
    x = mne.filter.filter_data(x, sf, bp["l_freq"], bp["h_freq"], method="iir",
                               iir_params={"order": int(bp["order"]), "ftype": "butter", "output": "sos"},
                               phase="zero", verbose="error")
    rstd = check_units(x, [d3 for _, d3 in picks], units, cfg, Path(path).name)
    x = remove_line_noise(x, sf, pp.get("line_noise"))
    x = x[:, n_settle:x.shape[1] - n_settle]
    info = mne.create_info(list(table["canonical_order"]), sf, "eeg")
    out = mne.io.RawArray(x, info, verbose="error")
    out.set_montage(chanlocs.montage(geometry))
    rec = {"orig_sfreq": sf, "duration_s": n_total / sf, "unit_scale": float(cfg["units"]["scale"][dataset]),
           "dropped_channels": [], "unrecognised_channels": [], "channel_naming": "indexed", "geometry": geometry,
           "x_edf_picks": json.dumps([name for name, _ in picks]), "x_original_samples": int(n_total),
           "x_edge_crop_samples": n_edge, "x_settle_samples": n_settle,
           "x_input_offset_samples": n_edge + n_settle, "x_bandpass": json.dumps(bp),
           "x_line_noise": json.dumps(pp.get("line_noise")),
           "x_channel_units": json.dumps({d3: units[name] for name, d3 in picks}),
           "x_mixed_units": len(set(units.values())) > 1,
           "x_channel_robust_std_uv": json.dumps({d3: round(v * 1e6, 4) for d3, v in rstd.items()})}
    return out, rec


def channel_units(raw, names: list[str]) -> dict:
    """EDF physical dimension of each named channel as MNE read it (MNE scales each channel by its own unit)."""
    orig = getattr(raw, "_orig_units", None) or {}
    return {n: str(orig.get(n, "")).replace("µ", "u") for n in names}


def robust_std(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=1)


def check_units(x: np.ndarray, names: list[str], units: dict, cfg: dict, rec: str) -> dict:
    """Fail fast unless every selected channel's band-passed robust std (volts) lies in ``channel_robust_std_v``:
    catches a header unit that does not match the stored values (factor 1e3 / 1e6) and dead channels."""
    lo, hi = cfg["units"]["channel_robust_std_v"]
    r = robust_std(x)
    low = {n: f"{v:.3g} V" for n, v in zip(names, r) if v < lo}
    high = {n: f"{v:.3g} V" for n, v in zip(names, r) if v > hi}
    if low or high:
        raise UnitError(f"{rec}: channel amplitude outside [{lo:g}, {hi:g}] V (robust std of the band-passed signal): "
                        + (f"no EEG in {low} (flat, or the signal is below one quantization step of the EDF range; "
                           "see lsb_uv / constant_fraction in inspect-d2-units); " if low else "")
                        + (f"too large in {high} (a header unit that does not match the stored values?); " if high else "")
                        + f"EDF units of the selected channels {sorted(set(units.values()))}. Nothing is rescaled, "
                        "interpolated or excluded automatically")
    return dict(zip(names, r.tolist()))


def remove_line_noise(x: np.ndarray, sf: float, ln: dict | None) -> np.ndarray:
    """The configured acquisition-specific line-noise notch (zero-phase FIR; deterministic). ``None``: unchanged."""
    import mne

    if not ln:
        return x
    if (ln["method"], ln["phase"]) != ("fir_notch", "zero"):
        raise D2StoreError(f"unsupported line-noise correction {ln}")
    y = mne.filter.notch_filter(x, sf, list(ln["freqs"]), notch_widths=float(ln["notch_width_hz"]),
                                trans_bandwidth=float(ln["trans_bandwidth_hz"]), method="fir", phase="zero",
                                fir_design="firwin", pad=ln.get("pad", "reflect_limited"), verbose="error")
    if not np.isfinite(y).all():
        raise D2StoreError("non-finite values after the line-noise notch")
    return y


def unit_audit(reg: pd.DataFrame, cfg: dict, table: dict) -> pd.DataFrame:
    """Every channel of every recording: EDF label, physical dimension, physical / digital range, the quantization step
    (``lsb_uv`` = physical range / digital range, in uV), whether it is one of the 19 selected channels, the fraction of
    raw samples equal to the most frequent value (``constant_fraction``), the number of distinct raw values, the plain
    and robust std MNE returns after the band-pass, and a ``finding``:
    ``ok`` | ``flat`` (>= 99 % of the stored samples identical: no recorded signal) | ``no_eeg_below_quantization``
    (varying samples, but a robust std below the minimum and below one quantization step) | ``too_small`` | ``too_large``
    (unit / scaling error) | ``unsupported_unit``. Reads the signals (no labels, no source model)."""
    import edfio
    import mne

    from ..io.layout import indexed_number

    rows = []
    lo, hi = cfg["units"]["channel_robust_std_v"]
    bp = cfg["preprocessing"]["bandpass"]
    sel = {r["edf_index"]: r["d3"] for r in table["channels"]}
    for rec in reg.itertuples():
        hdr = {s.label: s for s in edfio.read_edf(rec.path, lazy_load_data=True).signals}
        raw = mne.io.read_raw_edf(rec.path, preload=True, verbose="error")
        sf = float(raw.info["sfreq"])
        n = int(round(cfg["preprocessing"]["edge_crop_s"] * sf))
        raw_x = raw.get_data()[:, n:-n]
        x = mne.filter.filter_data(raw_x, sf, bp["l_freq"], bp["h_freq"], method="iir",
                                   iir_params={"order": int(bp["order"]), "ftype": "butter", "output": "sos"},
                                   phase="zero", verbose="error")
        r, sd = robust_std(x), x.std(axis=1)
        units = channel_units(raw, raw.ch_names)
        for i, name in enumerate(raw.ch_names):
            h = hdr.get(name)
            k = indexed_number(name)
            vals, counts = np.unique(raw_x[i], return_counts=True)
            scale = {"V": 1e6, "mV": 1e3, "uV": 1.0, "nV": 1e-3}.get(units[name])
            lsb = (float(h.physical_max - h.physical_min) / float(h.digital_max - h.digital_min) * scale
                   if h is not None and scale and h.digital_max != h.digital_min else float("nan"))
            if units[name] not in cfg["units"]["physical_dimensions"]:
                finding = "unsupported_unit"
            elif r[i] > hi:
                finding = "too_large"
            elif r[i] < lo:                       # no EEG: constant samples, or a signal finer than one step
                finding = ("flat" if counts.max() / counts.sum() >= 0.99 else
                           "no_eeg_below_quantization" if np.isfinite(lsb) and r[i] * 1e6 < lsb else "too_small")
            else:
                finding = "ok"
            rows.append({"key": rec.key, "edf_label": name, "edf_index": k, "selected_as": sel.get(k, ""),
                         "physical_dimension": units[name],
                         "physical_min": getattr(h, "physical_min", None), "physical_max": getattr(h, "physical_max", None),
                         "digital_min": getattr(h, "digital_min", None), "digital_max": getattr(h, "digital_max", None),
                         "lsb_uv": lsb, "constant_fraction": float(counts.max() / counts.sum()),
                         "n_unique_values": int(len(vals)), "std_uv": float(sd[i] * 1e6),
                         "robust_std_uv": float(r[i] * 1e6), "plausible": finding == "ok", "finding": finding})
        del raw, x, raw_x
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- build
def subject_identity(cfg: dict, rec, geometry_hash: str, table: dict) -> dict:
    from ..sourcegraphs.build import METADATA_FIELDS

    return {"kind": "xd2_subject", "version": VERSION, "config_hash": config_hash(cfg), "key": rec.key,
            "input": {"sha256": rec.sha256, "size_bytes": int(rec.size_bytes)},
            "metadata": stable_hash({k: getattr(rec, k, None) for k in METADATA_FIELDS}),
            "geometry": geometry_hash, "channel_map": channels.table_hash(table), "code": code_identity("cross_store")}


def contexts(cfg: dict, geometry: dict, subjects_dir, cache: Path) -> dict:
    """{geometry hash: build context} (labels, sensor info, forward model or wMNE operator); one geometry."""
    from ..source import atlas
    from ..source.inverse import fetch_fsaverage, load_labels
    from ..sourcegraphs import build, wmne
    from ..sourcegraphs.lcmv import forward, sensor_info

    fetch_fsaverage(subjects_dir)
    labels = atlas.merged_labels(load_labels(subjects_dir, cfg["source"]["parcellation"]), cfg["source"]["atlas"])
    if [lab.name for lab in labels] != atlas.region_names(cfg["source"]["atlas"]):
        raise D2StoreError("lobes12 regions differ from the D3 stores' ROI order")
    info = sensor_info(cfg, geometry)
    ctx = {"geometry": geometry, "info": info, "labels": labels, "roi_names": [lab.name for lab in labels],
           "channels": list(cfg["channels"]["recorded"])}
    if build.method(cfg) == "wmne":
        ctx["wmne"] = wmne.roi_operator(info, cfg, subjects_dir, cache, geometry["hash"], labels)
    else:
        ctx["fwd"] = forward(info, cfg, subjects_dir, cache, geometry["hash"])
    return {geometry["hash"]: ctx}


def run(paths: dict, processed, inverse: str, geometry: dict, subjects_dir, limit: int | None = None,
        allow_ephemeral: bool = False, table: dict | None = None, ctxs: dict | None = None,
        authorized: bool = False) -> tuple[Path, Path]:
    """Build (resumable) and assemble the D2 stores of ``inverse``. Any build from the real recordings (also a
    ``limit`` smoke build) needs the recorded ``build_d2`` authorization; ``authorized`` is for synthetic runs."""
    from ..io.fingerprint import InputHashes
    from ..sourcegraphs import build
    from ..storage import check_persistent

    table = table or channels.load_table()
    cfg = resolve(inverse, table)
    if not authorized:
        namespace.require_authorized(processed, "build_d2")
    if geometry["channels"] != cfg["channels"]["recorded"]:
        raise D2StoreError("geometry channels differ from the canonical order")
    run_dir, run10 = build.run_dirs(processed, cfg, limit)
    namespace.guard(processed, run_dir)
    namespace.guard(processed, run10)
    cache = namespace.sub(processed, "stores", "cache")
    check_persistent({"D2 store": run_dir, "fsaverage": subjects_dir}, allow_ephemeral=allow_ephemeral)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "run_identity.json").exists():
        atomic_write_json(run_dir / "run_identity.json", {"config_hash": config_hash(cfg), "config": cfg,
                                                         "git_commit": git_commit(),
                                                         "created_at": datetime.now(timezone.utc).isoformat()})
    reg = registry(paths, limit, InputHashes(cache / "input_sha256.json"))
    ctxs = ctxs or contexts(cfg, geometry, subjects_dir, cache)
    ctx = ctxs[geometry["hash"]]
    loader = partial(load_d2, table=table, geometry=geometry)
    table_st = StatusTable(run_dir / "status.csv")
    table_st.ensure(reg["key"].tolist())
    idents = {rec.key: subject_identity(cfg, rec, geometry["hash"], table) for rec in reg.itertuples()}
    failed = {}
    done0 = sum(check_unit(run_dir / "subjects" / k, "subject", i)[0] == "complete" for k, i in idents.items())
    with progress.reporter(len(reg), f"D2 {inverse} participants", initial=done0, unit="participants") as bar:
        for rec in reg.itertuples():
            unit = run_dir / "subjects" / rec.key
            state, _ = check_unit(unit, "subject", idents[rec.key])
            if state == "complete":
                table_st.set(rec.key, "completed", "validated existing checkpoint")
                continue
            if state in ("incompatible", "corrupt"):
                supersede(unit, run_dir)
            bar.set(subject=rec.key)
            table_st.set(rec.key, "running")
            t0 = time.perf_counter()
            try:
                out = build.process_subject(rec, cfg, ctx, loader)
            except Exception as exc:
                log.exception("%s failed", rec.key)
                failed[rec.key] = f"error: {exc}"
                atomic_write_json(unit / "failed.json", {"key": rec.key, "status": f"error: {exc}"})
                table_st.set(rec.key, "failed", str(exc)[:500])
                bar.advance()
                continue
            out[2]["seconds"] = round(time.perf_counter() - t0, 1)
            build.save_subject(unit, idents[rec.key], *out)
            table_st.set(rec.key, "completed")
            bar.advance()
    if failed:
        raise build.GraphRunIncomplete(f"{len(failed)} D2 participant(s) failed (not assembled): {failed}")
    extra = {"cross_dataset": {"dataset": "D2", "channel_map_hash": channels.table_hash(table),
                               "geometry": {k: geometry[k] for k in ("source", "frame", "hash")}
                               | {"reference_hash": geometry.get("reference_hash")},
                               "base_d3_config_hash": cfg["cross_dataset"]["base_d3_config_hash"],
                               "split": "none (all D2 participants; the ds004504 partition does not apply)"}}
    return build.assemble(run_dir, run10, reg, cfg, idents, {"D2": []}, limit, ctxs, split=False,
                          extra_manifest=extra, kind="xd2_assembly")


# ---------------------------------------------------------------------------------------------- validation
def validate(run_dir, expected: dict | None = EXPECTED_D2, check_arrays: bool = True) -> dict:
    """Checks of an assembled D2 store (boundaries, selection, windows, cohort, arrays, adjacency, names); writes
    ``validation.json`` into the store (inside the namespace)."""
    from ..graphs.store import CONNECTIVITY, check_adjacency
    from ..source.atlas import region_names
    from ..stores import XU_BANDS
    from ..sourcegraphs.validate import boundary_violations, selection_excess

    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    windows = pd.read_csv(run_dir / "windows.csv")
    subjects = pd.read_csv(run_dir / "subjects.csv")
    cfg = man["config"]
    seg = man["segmentation"]
    sf = float(cfg["preprocessing"]["sfreq"])
    checks = {}
    viol = boundary_violations(windows, subjects, seg["margin_samples"], sf)
    checks["no window crosses a boundary or its filter margin"] = (not viol, viol[:5])
    excess = selection_excess(windows, subjects, sf)
    checks["selected EEG at most the five-minute target"] = (not excess, excess[:5])
    checks["window length 10 s, step 1 s"] = (bool(((windows.stop_sample - windows.start_sample)
                                                     == seg["window_samples"]).all()), "")
    per = windows.groupby("key").size()
    checks["at most 291 windows per participant"] = (bool((per <= 291).all()), per.describe().round(1).to_dict())
    ok = subjects.loc[subjects.status == "ok"]
    checks["only D2 participants, unique keys and recordings"] = (
        bool((windows["dataset"] == "D2").all() and not subjects["key"].duplicated().any()
             and not subjects["path"].duplicated().any()), "")
    labels_ok = bool((windows.groupby("key")["y"].nunique() == 1).all()
                     and set(windows["label"]) <= {"AD", "HC"}
                     and ((windows["label"] == "AD") == (windows["y"] == 1)).all())
    checks["diagnosis labels consistent (AD = 1, HC = 0)"] = (labels_ok, "")
    if expected and not (man.get("registry") or {}).get("subset_limit"):
        got = (len(ok), int(ok["y"].sum()), int((ok["y"] == 0).sum()))
        checks[f"cohort {expected}"] = (got == tuple(expected.values()), got)
    n_col = "n_tenfold" if man.get("mode") == "A_xu_tenfold" else "n_windows_kept"
    mism = [k for k, n in per.items() if k not in set(ok["key"]) or int(ok.set_index("key").loc[k, n_col]) != n]
    checks[f"windows per participant equal subjects.csv {n_col}"] = (not mism, mism[:5])
    w = windows.groupby("key")["sample_weight"].sum()
    checks["sample weights sum to 1 per participant"] = (bool(np.allclose(w.to_numpy(), 1.0)), "")
    names = man["source_operator"]
    checks["sensor channels = the 19 canonical D3 channels in order"] = (
        names["channels"] == channels.load_table()["canonical_order"], names["channels"])
    checks["ROI names = lobes12 in the D3 store order"] = (names["roi_names"] == region_names("lobes12"), "")
    checks["five Xu bands in order"] = (list(cfg["bands"]) == XU_BANDS, list(cfg["bands"]))
    if check_arrays:
        n, r, k, c = len(windows), len(names["roi_names"]), len(cfg["bands"]), len(names["channels"])
        bad = []
        with np.load(run_dir / "graphs.npz") as z:
            for a in z.files:
                if a in ("bands", "band_edges", "roi_names", "channels", "roi_coords"):
                    continue
                v = z[a]
                want = (n, k, r, r) if a in ("plv", "ciplv") else (n, k, c) if a.startswith("sensor") else (n, k, r)
                if v.shape != want or not np.isfinite(v).all():
                    bad.append(f"{a}: shape {v.shape} (want {want}), finite {bool(np.isfinite(v).all())}")
                elif a in CONNECTIVITY:
                    try:
                        check_adjacency(v, a)
                    except ValueError as exc:
                        bad.append(f"{a}: {exc}")
                del v
        checks["arrays: 12 ROIs x 5 bands (19 sensors), finite; PLV / |ciPLV| valid adjacencies"] = (not bad, bad)
    out = {"run_dir": str(run_dir), "graph_run_id": man.get("graph_run_id"), "mode": man.get("mode"),
           "n_windows": int(len(windows)), "n_participants_ok": int(len(ok)),
           "n_participants_not_ok": int((subjects.status != "ok").sum()),
           "checks": {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()},
           "validated_at": datetime.now(timezone.utc).isoformat()}
    out["passed"] = all(c["passed"] for c in out["checks"].values())
    atomic_write_json(run_dir / "validation.json", out)
    return out


def open_d2(run_dir, keys: tuple | None = None):
    """The validated D2 store (refused unless ``validation.json`` passed for its current graph_run_id)."""
    from ..graphs.store import GraphStore

    run_dir = Path(run_dir)
    v = json.loads((run_dir / "validation.json").read_text()) if (run_dir / "validation.json").exists() else None
    gid = json.loads((run_dir / "manifest.json").read_text()).get("graph_run_id")
    if not v or not v.get("passed") or v.get("graph_run_id") != gid:
        raise D2StoreError(f"{run_dir.name} is not validated for graph run {gid} (crossdataset.d2store.validate)")
    return GraphStore.load(run_dir, keys=keys)
