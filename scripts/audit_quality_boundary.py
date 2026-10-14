"""Read-only dataset-wide QC audit: bad channels (Part 1) and connectivity boundary sensitivity (Part 2).

Plan, metrics and pre-specified decision criteria: docs/qc_boundary_audit.md (methodology log entry 40).

Runs against the production code at the pinned commit (default 7fd5a69) without changing it: it calls
the production ``load_raw``, ``preprocess`` and ``process_recording``, the cached source operator
(read-only) and ``band_analytic`` / ``phase_sync``. It writes ONLY under ``--out`` (a separate Drive
folder), one checkpointed unit per recording, so it resumes after a disconnect and never recomputes a
finished recording.

    python audit_quality_boundary.py estimate --out <dir>   # durations, runtime / memory / storage estimate
    python audit_quality_boundary.py part1    --out <dir>   # bad-channel QC, all 43 recordings
    python audit_quality_boundary.py report1  --out <dir>   # Part 1 tables
    python audit_quality_boundary.py part2    --out <dir>   # A (production) vs B (reflected edges)
    python audit_quality_boundary.py report2  --out <dir>   # Part 2 tables + figures
    python audit_quality_boundary.py status   --out <dir>

B replaces the 10 s outside the analysed interval by a mirror image of the adjacent analysed data
(same length, same windows, same band_analytic / phase_sync). A-B is sensitivity to boundary
handling, NOT proven artifact contamination; B is not ground truth (its own reflection changes the
first/last window, see the plan).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adgraph.checkpoint import atomic_savez, atomic_write_csv, atomic_write_json, check_unit, stable_hash, \
    write_marker
from adgraph.config import config_hash, load_paths, load_yaml
from adgraph.connectivity.plv import band_analytic, phase_sync, transition_kwargs
from adgraph.io.eeg import load_raw
from adgraph.memory import MemoryTracker
from adgraph.pipeline import (ANALYTIC_CHUNK_ROWS, build_registry, graph_run_dir, operator_identity,
                              process_recording)
from adgraph.preprocessing.clean import analysed_interval, preprocess
from adgraph.preprocessing.windows import reject_windows, window_samples, window_starts
from adgraph.provenance import git_commit, git_dirty, versions
from adgraph.storage import check_persistent

AUDIT_VERSION = "qc-boundary-audit-1"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SMOKE_LIMIT = 1                    # the smoke-test run whose stored outputs are used for verification


# ------------------------------------------------------------------------------------------- setup
def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["estimate", "part1", "report1", "part2", "report2", "status"])
    p.add_argument("--out", required=True, help="separate Drive folder for the audit (e.g. My Drive/AD-LEB-audit)")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--keys", nargs="*", help="only these recordings (default: all)")
    p.add_argument("--pin", default="7fd5a69", help="required production commit (prefix)")
    p.add_argument("--allow-other-commit", action="store_true")
    p.add_argument("--allow-ephemeral-storage", action="store_true")
    p.add_argument("--n-perm", type=int, default=20000, help="permutations for the stratified group tests")
    return p.parse_args()


def setup(args):
    commit = git_commit()
    if not commit.startswith(args.pin) and not args.allow_other_commit:
        raise SystemExit(f"production code is at {commit[:10]}, not the pinned {args.pin}; refusing "
                         "(the audit must use the code that built the graphs)")
    if git_dirty() and not args.allow_other_commit:
        raise SystemExit("the production working tree has uncommitted changes; refusing")
    check_persistent({"--out": args.out}, allow_ephemeral=args.allow_ephemeral_storage)
    paths, cfg = load_paths(args.paths), load_yaml(args.config)
    reg = build_registry(paths)[0]
    if args.keys:
        missing = sorted(set(args.keys) - set(reg["key"]))
        if missing:
            raise SystemExit(f"unknown keys {missing}")
        reg = reg[reg["key"].isin(args.keys)]
    out = Path(args.out) / f"qc_boundary_audit_{commit[:7]}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if not manifest.exists():
        atomic_write_json(manifest, {"audit_version": AUDIT_VERSION, "created_at": now(), "production_commit": commit,
                                     "config_hash": config_hash(cfg), "config": cfg, "versions": versions(),
                                     "plan": "docs/qc_boundary_audit.md", "read_only": True})
    return paths, cfg, reg.reset_index(drop=True), out, commit


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity(rec, part: str, cfg: dict, commit: str) -> dict:
    p = Path(rec.path)
    return {"audit": AUDIT_VERSION, "part": part, "key": rec.key, "file": p.name, "size_bytes": int(p.stat().st_size),
            "mtime": int(p.stat().st_mtime), "production_commit": commit, "config_hash": config_hash(cfg)}
    # AUDIT_VERSION (not the script hash) identifies the computation: edits to reporting code do not force a
    # recompute; bump AUDIT_VERSION when what a unit computes changes. Each summary records the script hash.


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def edge_metrics(x: np.ndarray, iv: slice) -> dict:
    """Edge-transient severity of the final preprocessed sensor signal (what the inverse sees):
    median-over-channels RMS in the discarded start/end segment over that of the analysed interval,
    and the largest |x| in the segment over the median robust std of the analysed interval."""
    rms = lambda s: np.sqrt((s ** 2).mean(axis=1))                                   # noqa: E731
    mid = np.median(rms(x[:, iv]))
    med = np.median(x[:, iv], axis=1, keepdims=True)
    rob = np.median(1.4826 * np.median(np.abs(x[:, iv] - med), axis=1))
    out = {}
    for side, seg in (("start", x[:, :iv.start]), ("end", x[:, iv.stop:])):
        out[f"edge_rms_ratio_{side}"] = float(np.median(rms(seg)) / mid) if seg.size else float("nan")
        out[f"edge_peak_ratio_{side}"] = float(np.abs(seg).max() / rob) if seg.size else float("nan")
    out["edge_rms_ratio_max"] = max(out["edge_rms_ratio_start"], out["edge_rms_ratio_end"])
    return out


def done(unit_dir: Path, ident: dict) -> bool:
    return check_unit(unit_dir, "unit", ident)[0] == "complete"


def records(out: Path, part: str) -> list[dict]:
    rows = []
    for f in sorted((out / part).glob("*/summary.json")):
        if (f.parent / "unit.done.json").exists():
            rows.append(json.loads(f.read_text()))
    return rows


# ------------------------------------------------------------------------------------------- estimate
def cmd_estimate(args, paths, cfg, reg, out, commit):
    import mne

    rows = []
    for rec in reg.itertuples():
        raw = mne.io.read_raw_edf(rec.path, preload=False, verbose="error")
        dur = raw.n_times / raw.info["sfreq"]
        n = int(dur * 250)
        rows.append({"key": rec.key, "dataset": rec.dataset, "label": rec.label, "sfreq": raw.info["sfreq"],
                     "duration_s": round(dur, 1), "size_mb": round(rec.size_bytes / 1e6, 1),
                     "candidate_windows": len(window_starts(n, 250.0, cfg["windowing"]["window_s"],
                                                            cfg["windowing"]["step_s"], cfg["windowing"]["edge_trim_s"])),
                     # peak working set: raw at the original rate (load + resample copy) and, in Part 2, the ROI
                     # signal, its reflected copy and one band's complex analytic signal (68 x n each)
                     "est_peak_mb_part1": round((30 * raw.n_times * 8 * 2 + 30 * n * 8 * 4) / 1e6),
                     "est_peak_mb_part2": round((30 * raw.n_times * 8 * 2 + 30 * n * 8 * 4 + 68 * n * (8 * 3 + 16 * 2)) / 1e6)})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(t.to_string(index=False))
    total_min = t["duration_s"].sum() / 60
    print(f"\n{len(t)} recordings, {total_min:.0f} min of EEG, {t['size_mb'].sum():.0f} MB of EDF, "
          f"{t['candidate_windows'].sum()} candidate windows; peak memory per recording (estimate) part 1 "
          f"{t['est_peak_mb_part1'].max()} MB, part 2 {t['est_peak_mb_part2'].max()} MB")
    sec = t["duration_s"].sum()
    print("runtime (measured on synthetic recordings, docs/qc_boundary_audit.md; Colab CPUs may be ~2x slower): "
          f"part 1 ~0.015-0.04 s per second of EEG (~{0.015 * sec / 60:.0f}-{0.04 * sec / 60:.0f} min), part 2 "
          f"~0.06-0.15 s per second of EEG (~{0.06 * sec / 60:.0f}-{0.15 * sec / 60:.0f} min), plus ~2-5 s of Drive "
          f"reading per recording and part; storage ~0.3 MB per recording (~{0.3 * len(t):.0f} MB) + reports ~2 MB")


# ------------------------------------------------------------------------------------------- part 1
def cmd_part1(args, paths, cfg, reg, out, commit):
    smoke = graph_run_dir(paths["processed_dir"], cfg, SMOKE_LIMIT) / "subjects"
    todo = [r for r in reg.itertuples() if not done(out / "part1" / r.key, identity(r, "part1", cfg, commit))]
    print(f"part 1: {len(reg) - len(todo)} recording(s) already complete, {len(todo)} to do")
    for i, rec in enumerate(todo, 1):
        unit = out / "part1" / rec.key
        t0 = time.perf_counter()
        base = {"key": rec.key, "dataset": rec.dataset, "subject_id": int(rec.subject_id), "label": rec.label,
                "file": Path(rec.path).name}
        with MemoryTracker() as m:
            raw, load_qc = load_raw(rec.path, rec.dataset, cfg)
            base.update({"orig_sfreq": load_qc["orig_sfreq"], "duration_s": load_qc["duration_s"]})
            try:
                rawp, qc = preprocess(raw, cfg)
            except ValueError as exc:                     # a QC finding, not a crash of the audit
                summary = {**base, "status": "preprocess_failed", "error": str(exc)}
                files = [atomic_write_json(unit / "summary.json", jsonable(summary))]
                write_marker(unit, "unit", identity(rec, "part1", cfg, commit), files)
                print(f"[{i}/{len(todo)}] {rec.key}: preprocess FAILED: {exc}")
                continue
            x, sf = rawp.get_data(), rawp.info["sfreq"]
            iv = analysed_interval(x.shape[1], sf, cfg)
            win, _ = window_samples(sf, cfg["windowing"]["window_s"], cfg["windowing"]["step_s"])
            starts = window_starts(x.shape[1], sf, cfg["windowing"]["window_s"], cfg["windowing"]["step_s"],
                                   cfg["windowing"]["edge_trim_s"])
            keep, thr, _ = reject_windows(x, starts, win, cfg["windowing"]["rejection"]["mad_k"],
                                          cfg["windowing"]["rejection"]["max_bad_channel_fraction"])
            summary = {**base, "status": "ok", "n_interpolated": qc["n_bad_channels"],
                       "interpolated": sorted(qc["bad_channels"]), "bad_reasons": qc["bad_channels"],
                       "whole_recording_bads": qc["bad_channels_whole_recording"],
                       "bad_fraction_warning": qc["bad_fraction_warning"],
                       "gross_robust_std_uv": qc["gross_robust_std_uv"], "median_std_uv": qc["median_std_uv"],
                       "median_robust_std_uv": qc["median_robust_std_uv"],
                       "whole_recording_std_uv": qc["whole_recording_std_uv"],
                       "n_windows_total": int(len(starts)), "n_windows_kept": int(keep.sum()),
                       "rejection_threshold_uv": thr * 1e6, **edge_metrics(x, iv)}
            ch = qc["channels"]
        sm = smoke / rec.key
        if (sm / "subject.done.json").exists() and (sm / "qc_channels.csv").exists():   # verify against stored QC
            st = pd.read_csv(sm / "qc_channels.csv")
            sj = json.loads((sm / "subject.json").read_text())
            num = [c for c in ch.columns if c in st.columns and pd.api.types.is_numeric_dtype(ch[c])]
            summary["smoke_check"] = {
                "bad_channels_equal": json.loads(sj["bad_channels"]) == qc["bad_channels"],
                "max_abs_diff_channel_table": float(np.nanmax(np.abs(st[num].to_numpy(float) - ch[num].to_numpy(float)))),
                "n_windows_kept_equal": int(sj["n_windows_kept"]) == int(keep.sum())}
        summary.update({"seconds": round(time.perf_counter() - t0, 1), "script_sha256": SCRIPT_SHA, **m.record()})
        files = [atomic_write_csv(unit / "channels.csv", ch), atomic_write_json(unit / "summary.json", jsonable(summary))]
        write_marker(unit, "unit", identity(rec, "part1", cfg, commit), files)
        print(f"[{i}/{len(todo)}] {rec.key}: {summary['n_interpolated']} interpolated {summary['interpolated']}, "
              f"edge RMS ratio start/end {summary['edge_rms_ratio_start']:.1f}/{summary['edge_rms_ratio_end']:.1f}, "
              f"{summary['seconds']} s, peak {summary.get('mem_peak_mb')} MB"
              + (f", smoke check {summary['smoke_check']}" if "smoke_check" in summary else ""), flush=True)
        del raw, rawp, x


# ------------------------------------------------------------------------------------------- stats helpers
def strat_perm(df: pd.DataFrame, col: str, n_perm: int, seed: int = 0) -> dict:
    """AD - HC difference in means of ``col`` within each dataset, averaged with dataset weights; two-sided
    p from permuting labels within dataset (subjects are the units)."""
    d = df.dropna(subset=[col])
    if d["label"].nunique() < 2:
        return {}
    rng = np.random.default_rng(seed)
    groups = [(g[col].to_numpy(float), (g["label"] == "AD").to_numpy()) for _, g in d.groupby("dataset")]
    groups = [(v, y) for v, y in groups if 0 < y.sum() < len(y)]
    if not groups:
        return {}
    w = np.array([len(v) for v, _ in groups], float) / sum(len(v) for v, _ in groups)

    def stat(ys):
        return float(sum(wi * (v[y].mean() - v[~y].mean()) for wi, (v, _), y in zip(w, groups, ys)))

    obs = stat([y for _, y in groups])
    null = np.array([stat([rng.permutation(y) for _, y in groups]) for _ in range(n_perm)])
    return {"AD_minus_HC_within_dataset": obs, "p_stratified_perm": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))}


def mwu(a, b) -> float:
    from scipy.stats import mannwhitneyu

    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else float("nan")


def report_dir(out: Path, name: str) -> Path:
    d = out / "reports" / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=False)
    return d


# ------------------------------------------------------------------------------------------- report 1
def neighbours(cfg) -> dict[str, set]:
    """Electrode neighbours: montage distance <= 1.5 x the median nearest-neighbour distance
    (2-6 neighbours per electrode on the 30-channel set; e.g. T7: F7, FC5, CP5, P7)."""
    from adgraph.config import common_channels
    from adgraph.io.eeg import make_montage

    used = common_channels(cfg)
    pos = {k.lower(): v for k, v in make_montage(cfg["channels"]["montage"]).get_positions()["ch_pos"].items()}
    X = np.array([pos[c.lower()] for c in used])
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nb = D <= 1.5 * np.median(D.min(axis=1))
    return {c: {used[j] for j in np.flatnonzero(nb[i])} for i, c in enumerate(used)}


def largest_cluster(chs: list[str], nbrs: dict[str, set]) -> int:
    """Size of the largest group of interpolated electrodes connected through neighbours."""
    left, best = set(chs), 0
    while left:
        stack, comp = [left.pop()], 1
        while stack:
            for n in nbrs[stack.pop()] & left:
                left.discard(n)
                stack.append(n)
                comp += 1
        best = max(best, comp)
    return best


def cmd_report1(args, paths, cfg, reg, out, commit):
    rows = records(out, "part1")
    print(f"part 1 units complete: {len(rows)} of {len(reg)}")
    if not rows:
        return
    rep = report_dir(out, "report1")
    s = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 120)
    failed = s[s["status"] != "ok"]
    if len(failed):
        print("\nrecordings that FAILED preprocessing (QC finding):")
        print(failed[["key", "label", "error"]].to_string(index=False))
    ok = s[s["status"] == "ok"].copy()
    cols = ["key", "dataset", "label", "duration_s", "n_interpolated", "interpolated", "largest_cluster",
            "nbr_interp_frac", "whole_recording_bads", "bad_fraction_warning", "median_std_uv", "gross_robust_std_uv", "edge_rms_ratio_start", "edge_rms_ratio_end",
            "n_windows_kept", "n_windows_total"]
    ok["whole_recording_bads"] = ok["whole_recording_bads"].apply(lambda d: sorted(d))
    nbrs = neighbours(cfg)
    ok["largest_cluster"] = ok["interpolated"].apply(lambda L: largest_cluster(L, nbrs))
    # spherical-spline interpolation relies on neighbours: fraction of the interpolated channels' neighbours
    # that were interpolated themselves
    ok["nbr_interp_frac"] = ok["interpolated"].apply(
        lambda L: float(np.mean([len(nbrs[c] & set(L)) / len(nbrs[c]) for c in L])) if L else 0.0)
    print("\n[1.1] interpolated channels per recording (analysed-interval detection; 'whole' = QC only):")
    print(ok[cols].round(2).to_string(index=False))
    atomic_write_csv(rep / "per_recording.csv", ok[cols + ["bad_reasons"]].astype(str))
    chans = cfg["channels"]["recorded"]
    used = [c for c in chans if c not in cfg["channels"]["drop"]]
    freq = []
    for ds, g in ok.groupby("dataset"):
        for c in used:
            hits = g["interpolated"].apply(lambda L: c in L)
            reasons = g.loc[hits, "bad_reasons"].apply(lambda d: d[c].split(" z=")[0].rsplit(" ", 1)[0] if c in d else "")
            freq.append({"dataset": ds, "channel": c, "n_flagged": int(hits.sum()), "n_recordings": len(g),
                         "pct": 100 * hits.mean(), "n_AD": int((hits & (g["label"] == "AD")).sum()),
                         "n_HC": int((hits & (g["label"] == "HC")).sum()),
                         "reasons": ",".join(sorted(set(r for r in reasons if r)))})
    f = pd.DataFrame(freq)
    wide = f.pivot(index="channel", columns="dataset", values="n_flagged").loc[used]
    wide["total"] = wide.sum(axis=1)
    print("\n[1.2] per-channel detection frequency (number of recordings in which the channel was interpolated):")
    print(wide.join(f.pivot(index="channel", columns="dataset", values="pct").add_suffix("_%").round(0))
          .sort_values("total", ascending=False).to_string())
    atomic_write_csv(rep / "channel_frequency.csv", f)
    print("\n[1.3] distribution of the number of interpolated channels per recording:")
    print(pd.crosstab(ok["n_interpolated"], ok["dataset"], margins=True).to_string())
    print("\n[1.4] AD vs HC (subjects are the units):")
    grp = ok.groupby(["dataset", "label"])["n_interpolated"].agg(["count", "median", "mean", "max",
                                                                  lambda v: int((v > 0).sum())])
    grp.columns = ["n_subjects", "median", "mean", "max", "n_with_any"]
    print(grp.round(2).to_string())
    tests = {ds: mwu(g.loc[g["label"] == "AD", "n_interpolated"], g.loc[g["label"] == "HC", "n_interpolated"])
             for ds, g in ok.groupby("dataset")}
    sp = strat_perm(ok, "n_interpolated", args.n_perm)
    print(f"Mann-Whitney p within dataset: {json.dumps({k: round(v, 4) for k, v in tests.items()})}; "
          f"stratified permutation (within dataset): {json.dumps({k: round(v, 4) for k, v in sp.items()})}")
    print("D1 vs D2 Mann-Whitney p (n_interpolated):",
          round(mwu(ok.loc[ok.dataset == "D1", "n_interpolated"], ok.loc[ok.dataset == "D2", "n_interpolated"]), 4))
    print("\n[1.4b] recordings meeting the pre-specified recording-level flags (docs/qc_boundary_audit.md, R1-R3):")
    flags = ok[(ok["n_interpolated"] > cfg["preprocessing"]["bad_channels"]["max_fraction_warn"] * len(used))
               | (ok["largest_cluster"] >= 3) | (ok["n_windows_kept"] < cfg["windowing"]["min_windows"])]
    print(flags[["key", "dataset", "label", "n_interpolated", "interpolated", "largest_cluster", "n_windows_kept"]]
          .to_string(index=False) if len(flags) else "none")
    print("largest interpolated cluster per recording:", pd.crosstab(ok["largest_cluster"], ok["dataset"]).to_dict())
    print("\n[1.5] systematic electrodes (pre-specified: flagged in >= 30 % of the recordings of a dataset):")
    sysm = f[f["pct"] >= 30]
    print(sysm.to_string(index=False) if len(sysm) else "none")
    if "D2-2" in set(ok["key"]):
        r = ok.set_index("key").loc["D2-2"]
        rank = (ok["n_interpolated"] >= r["n_interpolated"]).sum()
        print(f"D2-2: {r['n_interpolated']} interpolated {r['interpolated']}; {rank} of {len(ok)} recordings have "
              f">= that many; recordings sharing >= 2 of its channels: "
              + str([k for k, L in zip(ok["key"], ok["interpolated"]) if k != "D2-2"
                     and len(set(L) & set(r["interpolated"])) >= 2]))
    print("\n[1.6] edge-transient severity (final signal, discarded 10 s vs analysed interval, RMS ratio):")
    print(ok.groupby("dataset")[["edge_rms_ratio_start", "edge_rms_ratio_end"]].describe().round(2).T.to_string())
    checks = ok.dropna(subset=["smoke_check"]) if "smoke_check" in ok else ok.iloc[:0]
    if len(checks):
        print("\nverification against the stored smoke-test QC:")
        for k, c in zip(checks["key"], checks["smoke_check"]):
            print(f"  {k}: {c}")
    _fig_part1(ok, f, used, rep)
    print(f"\nsaved to {rep}")


def _fig_part1(ok, f, used, rep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    for ds, mk in (("D1", "o"), ("D2", "s")):
        g = f[f["dataset"] == ds].set_index("channel").loc[used]
        ax[0].plot(range(len(used)), g["pct"], marker=mk, label=ds)
    ax[0].set_xticks(range(len(used)), used, rotation=90, fontsize=8)
    ax[0].axhline(30, ls="--", color="k", lw=0.8, label="30 % (pre-specified)")
    ax[0].set_ylabel("% of recordings interpolated")
    ax[0].legend()
    lab = ok["dataset"] + " " + ok["label"]
    for i, (name, g) in enumerate(ok.groupby(lab)):
        ax[1].scatter(np.full(len(g), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(g)), g["n_interpolated"])
    ax[1].set_xticks(range(lab.nunique()), sorted(lab.unique()))
    ax[1].set_ylabel("interpolated channels per recording")
    fig.suptitle("Part 1 bad-channel audit (diagnostic, not a paper figure)")
    fig.tight_layout()
    fig.savefig(rep / "part1.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- part 2
def load_operator(paths, cfg):
    ident = operator_identity(cfg)
    name = f"operator_{stable_hash(ident)}"
    op_dir = Path(paths["cache_dir"]) / "operators"
    if check_unit(op_dir, name, ident)[0] != "complete":
        raise SystemExit("no valid cached source operator for the pinned commit (run the smoke test first); "
                         "the audit never builds or writes an operator")
    z = np.load(op_dir / f"{name}.npz")
    return z["W"], z["roi_names"].tolist()


def mirror_edges(x: np.ndarray, iv: slice) -> np.ndarray:
    y = x.copy()
    a, b = iv.start, iv.stop
    y[:, :a] = x[:, a + 1:2 * a + 1][:, ::-1]                    # x[a - k] = x[a + k]
    y[:, b:] = x[:, 2 * b - x.shape[1] - 1:b - 1][:, ::-1]       # x[b - 1 + k] = x[b - 1 - k]
    return y


def cmd_part2(args, paths, cfg, reg, out, commit):
    from mne.filter import filter_data

    W, roi_names = load_operator(paths, cfg)
    iu = np.triu_indices(W.shape[0], 1)
    bands = cfg["connectivity"]["bands"]
    tbw = cfg["connectivity"].get("transition_bandwidth_hz")
    smoke = graph_run_dir(paths["processed_dir"], cfg, SMOKE_LIMIT) / "subjects"
    todo = [r for r in reg.itertuples() if not done(out / "part2" / r.key, identity(r, "part2", cfg, commit))]
    print(f"part 2: {len(reg) - len(todo)} recording(s) already complete, {len(todo)} to do")
    for i, rec in enumerate(todo, 1):
        unit = out / "part2" / rec.key
        p1 = out / "part1" / rec.key / "summary.json"
        p1s = json.loads(p1.read_text()) if p1.exists() else None
        if p1s and p1s["status"] != "ok":
            summary = {"key": rec.key, "dataset": rec.dataset, "label": rec.label, "status": "skipped",
                       "reason": "preprocessing failed in part 1"}
            write_marker(unit, "unit", identity(rec, "part2", cfg, commit),
                         [atomic_write_json(unit / "summary.json", summary)])
            continue
        t0 = time.perf_counter()
        with MemoryTracker() as m:
            raw, _ = load_raw(rec.path, rec.dataset, cfg)
            rawp, qc = preprocess(raw, cfg)
            del raw
            A = process_recording(rawp, W, cfg)                          # production, unchanged
            x, sf = rawp.get_data(), rawp.info["sfreq"]
            del rawp
            iv = analysed_interval(x.shape[1], sf, cfg)
            win, step = window_samples(sf, cfg["windowing"]["window_s"], cfg["windowing"]["step_s"])
            starts = A["start_sample"]
            cand = A["window_index"]
            n_cand = int(A["n_windows_total"])
            dist = np.minimum(cand, n_cand - 1 - cand)                   # candidate steps from the nearest boundary
            roi = W @ x
            del x
            roi_b = mirror_edges(roi, iv)
            a = {"plv": A["plv"][:, :, iu[0], iu[1]], "ciplv": A["ciplv"][:, :, iu[0], iu[1]]}
            b = {"plv": np.zeros_like(a["plv"]), "ciplv": np.zeros_like(a["ciplv"])}
            band_edge = np.zeros((len(bands), 2))
            for bi, (lo, hi) in enumerate(bands.values()):
                an = band_analytic(roi_b, sf, lo, hi, chunk_rows=ANALYTIC_CHUNK_ROWS, transition_bandwidth_hz=tbw)
                for w, s in enumerate(starts):
                    pl, ci, _ = phase_sync(an[:, s:s + win])
                    b["plv"][w, bi], b["ciplv"][w, bi] = pl[iu], ci[iu]
                del an
                xf = filter_data(roi, sf, lo, hi, method="fir", phase="zero", fir_design="firwin", verbose="error",
                                 **transition_kwargs(lo, hi, tbw))       # band-limited edge severity (A's input)
                r = lambda s: np.sqrt((s ** 2).mean(axis=1))             # noqa: E731
                mid = r(xf[:, iv])
                band_edge[bi] = [np.median(r(xf[:, :iv.start]) / mid), np.median(r(xf[:, iv.stop:]) / mid)]
                del xf
            del roi, roi_b
        res = {"key": rec.key, "dataset": rec.dataset, "subject_id": int(rec.subject_id), "label": rec.label,
               "status": "ok", "n_windows": len(starts), "n_candidates": n_cand,
               "n_interpolated": qc["n_bad_channels"], "interpolated": sorted(qc["bad_channels"]),
               "band_edge_rms_ratio": {bn: {"start": band_edge[k, 0], "end": band_edge[k, 1]} for k, bn in enumerate(bands)}}
        if p1s:
            res["bad_channels_match_part1"] = sorted(p1s["interpolated"]) == sorted(qc["bad_channels"])
        st = smoke / rec.key / "arrays.npz"
        if st.exists() and (smoke / rec.key / "subject.done.json").exists():
            with np.load(st) as z:
                res["reproduces_stored"] = {"plv_max_abs_diff": float(np.abs(z["plv"] - A["plv"]).max()),
                                            "ciplv_max_abs_diff": float(np.abs(z["ciplv"] - A["ciplv"]).max())}
        arrays = {"window_start_s": starts / sf, "window_index": cand, "dist": dist,
                  "band_edge_rms_ratio": band_edge}
        for k in ("plv", "ciplv"):
            d = np.abs(a[k] - b[k])                                       # (windows, bands, pairs)
            arrays[f"{k}_win_band_median"] = np.median(d, axis=2)
            arrays[f"{k}_win_band_p95"] = np.percentile(d, 95, axis=2)
            arrays[f"{k}_win_band_max"] = d.max(axis=2)
            arrays[f"{k}_pair_mean_absdiff"] = d.mean(axis=0).astype(np.float32)          # (bands, pairs)
            arrays[f"{k}_subject_mean_diff"] = (a[k].mean(0) - b[k].mean(0)).astype(np.float32)
            arrays[f"{k}_subject_mean_A"] = a[k].mean(0).astype(np.float32)
            arrays[f"{k}_between_window_sd_A"] = a[k].std(0).astype(np.float32)
            stats = {}
            for name, sel in (("all", np.ones(len(dist), bool)), ("excl_boundary", dist >= 1),
                              ("excl_boundary2", dist >= 2)):
                dd = d[sel]
                sm = np.abs(a[k][sel].mean(0) - b[k][sel].mean(0)) if sel.any() else np.array([np.nan])
                stats[name] = {"n_windows": int(sel.sum()),
                               "median": float(np.median(dd)) if dd.size else None,
                               "p95": float(np.percentile(dd, 95)) if dd.size else None,
                               "max": float(dd.max()) if dd.size else None,
                               "subject_mean_max": float(np.nanmax(sm)),
                               "subject_mean_p95": float(np.nanpercentile(sm, 95))}
            stats["per_band"] = {bn: {"median": float(np.median(d[:, k2])), "p95": float(np.percentile(d[:, k2], 95)),
                                      "max": float(d[:, k2].max()),
                                      "subject_mean_max": float(np.abs(a[k][:, k2].mean(0) - b[k][:, k2].mean(0)).max()),
                                      "p95_excl_boundary": float(np.percentile(d[dist >= 1, k2], 95)) if (dist >= 1).any() else None}
                                 for k2, bn in enumerate(bands)}
            pos = {"first": 0, "middle": len(starts) // 2, "last": len(starts) - 1}
            stats["by_position"] = {p: {"t0_s": float(starts[w] / sf), "dist": int(dist[w]), "median": float(np.median(d[w])),
                                        "p95": float(np.percentile(d[w], 95)), "max": float(d[w].max())}
                                    for p, w in pos.items()}
            sm_all = np.abs(arrays[f"{k}_subject_mean_diff"]).ravel()
            top = np.argsort(sm_all)[::-1][:10]
            P = len(iu[0])
            stats["top_subject_mean_cells"] = [{"band": list(bands)[c // P], "roi_a": roi_names[iu[0][c % P]],
                                                "roi_b": roi_names[iu[1][c % P]], "abs_subject_mean_diff": float(sm_all[c]),
                                                "A_subject_mean": float(arrays[f"{k}_subject_mean_A"].ravel()[c])}
                                               for c in top]
            res[k] = stats
        res.update({"seconds": round(time.perf_counter() - t0, 1), "script_sha256": SCRIPT_SHA, **m.record()})
        files = [atomic_savez(unit / "arrays.npz", **arrays), atomic_write_json(unit / "summary.json", jsonable(res))]
        write_marker(unit, "unit", identity(rec, "part2", cfg, commit), files)
        print(f"[{i}/{len(todo)}] {rec.key}: PLV p95 {res['plv']['all']['p95']:.5f} max {res['plv']['all']['max']:.5f} "
              f"subject-mean max {res['plv']['all']['subject_mean_max']:.5f} | excl. boundary max "
              f"{res['plv']['excl_boundary']['max']}; {res['seconds']} s, peak {res.get('mem_peak_mb')} MB"
              + (f"; reproduces stored {res['reproduces_stored']}" if "reproduces_stored" in res else ""), flush=True)
        del A, a, b, arrays


# ------------------------------------------------------------------------------------------- report 2
def cmd_report2(args, paths, cfg, reg, out, commit):
    from scipy.stats import spearmanr

    rows = [r for r in records(out, "part2") if r.get("status") == "ok"]
    print(f"part 2 units complete (ok): {len(rows)} of {len(reg)}")
    if not rows:
        return
    rep = report_dir(out, "report2")
    p1 = {r["key"]: r for r in records(out, "part1")}
    bands = list(cfg["connectivity"]["bands"])
    pd.set_option("display.width", 230, "display.max_rows", 300, "display.max_colwidth", 100)
    flat = []
    for r in rows:
        e = p1.get(r["key"], {})
        row = {"key": r["key"], "dataset": r["dataset"], "label": r["label"], "n_windows": r["n_windows"],
               "n_interpolated": r["n_interpolated"], "edge_rms_ratio_max": e.get("edge_rms_ratio_max"),
               "band_edge_rms_max": max(max(v.values()) for v in r["band_edge_rms_ratio"].values())}
        for k in ("plv", "ciplv"):
            for sel in ("all", "excl_boundary", "excl_boundary2"):
                for st in ("median", "p95", "max", "subject_mean_max"):
                    row[f"{k}_{sel}_{st}"] = r[k][sel][st]
        flat.append(row)
    t = pd.DataFrame(flat)
    atomic_write_csv(rep / "per_recording.csv", t)
    show = ["key", "dataset", "label", "n_windows", "n_interpolated", "edge_rms_ratio_max", "band_edge_rms_max",
            "plv_all_median", "plv_all_p95", "plv_all_max", "plv_all_subject_mean_max", "plv_excl_boundary_max",
            "plv_excl_boundary_subject_mean_max", "ciplv_all_p95", "ciplv_all_max", "ciplv_all_subject_mean_max",
            "ciplv_excl_boundary_max"]
    print("\n[2.1] per recording, A (production) vs B (reflected edges); |d| over windows x bands x pairs:")
    print(t[show].round(5).sort_values("plv_all_subject_mean_max", ascending=False).to_string(index=False))
    checks = [(r["key"], r["reproduces_stored"]) for r in rows if "reproduces_stored" in r]
    print("\nreproduction of stored production graphs:", checks or "no stored units found")
    mism = [r["key"] for r in rows if r.get("bad_channels_match_part1") is False]
    print("bad channels differ between part 1 and part 2 runs:", mism or "none")
    print("\n[2.2] per band (median over recordings of each recording's statistic):")
    pb = []
    for r in rows:
        for k in ("plv", "ciplv"):
            for bn, v in r[k]["per_band"].items():
                pb.append({"metric": k, "band": bn, "key": r["key"], "dataset": r["dataset"], **v})
    pb = pd.DataFrame(pb)
    print(pb.groupby(["metric", "band"])[["median", "p95", "max", "subject_mean_max", "p95_excl_boundary"]]
          .median().round(5).to_string())
    worst_band = pb.loc[pb.groupby(["metric", "key"])["subject_mean_max"].idxmax()]
    print("\nmost sensitive band per recording (by subject-mean max), counts:")
    print(pd.crosstab([worst_band["metric"], worst_band["band"]], worst_band["dataset"]).to_string())
    atomic_write_csv(rep / "per_band.csv", pb)
    print("\n[2.3] by distance from the analysed-interval boundary (candidate-window steps; 0 = touches it):")
    prof = []
    for r in rows:
        with np.load(out / "part2" / r["key"] / "arrays.npz") as z:
            for k in ("plv", "ciplv"):
                mx = z[f"{k}_win_band_max"].max(axis=1)
                p95 = z[f"{k}_win_band_p95"].max(axis=1)
                for dd, m1, m2 in zip(z["dist"], mx, p95):
                    prof.append({"metric": k, "key": r["key"], "dist": int(min(dd, 6)), "max": m1, "p95": m2})
    prof = pd.DataFrame(prof)
    print(prof.groupby(["metric", "dist"])[["p95", "max"]].agg(["median", lambda v: np.percentile(v, 90), "max"])
          .round(5).rename(columns={"<lambda_0>": "p90"}).to_string())
    print("(dist 6 = 6 or more steps)")
    print("\n[2.4] position summary (first / middle / last accepted window), median and max over recordings:")
    pos = pd.DataFrame([{"metric": k, "position": p, **v} for r in rows for k in ("plv", "ciplv")
                        for p, v in r[k]["by_position"].items()])
    print(pos.groupby(["metric", "position"])[["median", "p95", "max"]].agg(["median", "max"]).round(5).to_string())
    print("\n[2.5] associations across recordings (Spearman; subjects are the units):")
    assoc = []
    for y in ("plv_all_subject_mean_max", "plv_excl_boundary_max", "plv_all_p95", "ciplv_all_subject_mean_max"):
        for xcol in ("edge_rms_ratio_max", "band_edge_rms_max", "n_interpolated"):
            for scope, g in [("all", t)] + list(t.groupby("dataset")):
                ok = g[[xcol, y]].dropna()
                if len(ok) > 3:
                    rr = spearmanr(ok[xcol], ok[y])
                    assoc.append({"y": y, "x": xcol, "scope": scope, "n": len(ok), "rho": rr.statistic, "p": rr.pvalue})
    print(pd.DataFrame(assoc).round(4).to_string(index=False))
    print("\n[2.6] D1 vs D2 and AD vs HC (per-recording statistics; no window is a unit):")
    grp = []
    for y in ("plv_all_p95", "plv_all_subject_mean_max", "plv_excl_boundary_max", "ciplv_all_subject_mean_max",
              "edge_rms_ratio_max", "n_interpolated"):
        sp = strat_perm(t, y, args.n_perm)
        grp.append({"statistic": y, "D1_median": t.loc[t.dataset == "D1", y].median(),
                    "D2_median": t.loc[t.dataset == "D2", y].median(),
                    "p_D1_vs_D2_MWU": mwu(t.loc[t.dataset == "D1", y], t.loc[t.dataset == "D2", y]),
                    "AD_median": t.loc[t.label == "AD", y].median(), "HC_median": t.loc[t.label == "HC", y].median(),
                    **{f"p_AD_vs_HC_MWU_{ds}": mwu(g.loc[g.label == "AD", y], g.loc[g.label == "HC", y])
                       for ds, g in t.groupby("dataset")}, **sp})
    print(pd.DataFrame(grp).round(5).to_string(index=False))
    print("\n[2.7] unusually sensitive recordings (pre-specified: robust z of log subject-mean max > 3.5, or "
          "subject-mean max > 0.01, or excl.-boundary max > 0.05):")
    lv = np.log10(t["plv_all_subject_mean_max"].clip(lower=1e-9))
    rz = (lv - lv.median()) / (1.4826 * np.median(np.abs(lv - lv.median())) or 1)
    t["robust_z"] = rz
    flagged = t[(rz > 3.5) | (t["plv_all_subject_mean_max"] > 0.01) | (t["plv_excl_boundary_max"] > 0.05)]
    print(flagged[["key", "dataset", "label", "robust_z", "plv_all_subject_mean_max", "plv_excl_boundary_max",
                   "n_interpolated", "edge_rms_ratio_max"]].round(4).to_string(index=False) if len(flagged) else "none")
    for key in flagged["key"]:
        r = next(x for x in rows if x["key"] == key)
        for k in ("plv", "ciplv"):
            wb = max(r[k]["per_band"].items(), key=lambda kv: kv[1]["subject_mean_max"])
            print(f"  {key} {k}: most affected band {wb[0]} (subject-mean max {wb[1]['subject_mean_max']:.4f}); "
                  f"band edge RMS ratios {json.dumps({b: {s: round(v, 2) for s, v in d.items()} for b, d in r['band_edge_rms_ratio'].items()})}")
            print("   top ROI pairs:", "; ".join(f"{c['band']} {c['roi_a']}-{c['roi_b']} {c['abs_subject_mean_diff']:.4f}"
                                             for c in r[k]["top_subject_mean_cells"][:5]))
    print("\n[2.8] ROI concentration across recordings (mean over recordings of the per-pair mean |d|, PLV):")
    acc = None
    for r in rows:
        with np.load(out / "part2" / r["key"] / "arrays.npz") as z:
            v = z["plv_pair_mean_absdiff"].astype(float)
        acc = v if acc is None else acc + v
    acc /= len(rows)
    W, roi_names = load_operator(paths, cfg)
    iu = np.triu_indices(W.shape[0], 1)
    node = np.zeros(W.shape[0])
    for bi in range(len(bands)):
        np.add.at(node, iu[0], acc[bi]); np.add.at(node, iu[1], acc[bi])
    node /= len(bands) * (W.shape[0] - 1)
    order = np.argsort(node)[::-1]
    print("ROIs with the largest mean |d| (median over ROIs %.2e): " % np.median(node)
          + ", ".join(f"{roi_names[i]} {node[i]:.2e}" for i in order[:10]))
    flat_cells = acc.ravel()
    o = np.argsort(flat_cells)[::-1]
    print(f"share of the total in the top 1 % / 5 % of (band, pair) cells: "
          f"{flat_cells[o[:len(o) // 100]].sum() / flat_cells.sum():.1%} / {flat_cells[o[:len(o) // 20]].sum() / flat_cells.sum():.1%}")
    print("\n[2.9] boundary effect relative to the between-subject spread (pre-specified primary criterion):")
    print("ratio = |subject-mean A - subject-mean B| / pooled within-dataset SD over subjects of the subject-mean A,"
          " per (band, pair); summarised per recording")
    mats, ds_of = {}, []
    for r in rows:
        with np.load(out / "part2" / r["key"] / "arrays.npz") as z:
            mats[r["key"]] = {k: (z[f"{k}_subject_mean_A"].astype(float), z[f"{k}_subject_mean_diff"].astype(float))
                              for k in ("plv", "ciplv")}
        ds_of.append(r["dataset"])
    rel = []
    for k in ("plv", "ciplv"):
        keys = [r["key"] for r in rows]
        Am = np.stack([mats[x][k][0] for x in keys])                       # (subjects, bands, pairs)
        var = [Am[np.array(ds_of) == ds].var(axis=0, ddof=1) for ds in sorted(set(ds_of))
               if (np.array(ds_of) == ds).sum() > 1]
        sd = np.sqrt(np.mean(var, axis=0)) if var else np.full(Am.shape[1:], np.nan)
        for x, r in zip(keys, rows):
            q = np.abs(mats[x][k][1]) / np.maximum(sd, 1e-6)
            rel.append({"metric": k, "key": x, "dataset": r["dataset"], "label": r["label"],
                        "ratio_median": np.median(q), "ratio_p95": np.percentile(q, 95), "ratio_max": q.max(),
                        "worst_band": bands[int(np.unravel_index(np.argmax(q), q.shape)[0])]})
    rel = pd.DataFrame(rel)
    atomic_write_csv(rep / "relative_to_between_subject_sd.csv", rel)
    print(rel.sort_values(["metric", "ratio_p95"], ascending=[True, False]).round(4).to_string(index=False))
    for k, g in rel.groupby("metric"):
        print(f"{k}: recordings with ratio p95 <= 0.10: {(g.ratio_p95 <= 0.10).mean():.0%}; > 0.25: "
              f"{(g.ratio_p95 > 0.25).sum()} {g.loc[g.ratio_p95 > 0.25, 'key'].tolist()}")
        for y in ("ratio_p95",):
            sp = strat_perm(g, y, args.n_perm)
            print(f"   D1 vs D2 p (MWU) {mwu(g.loc[g.dataset == 'D1', y], g.loc[g.dataset == 'D2', y]):.4f}; AD vs HC "
                  f"within dataset {json.dumps({kk: round(v, 5) for kk, v in sp.items()})}")
    _fig_part2(t, prof, rep)
    atomic_write_json(rep / "summary.json", jsonable({"n": len(rows), "association": assoc, "groups": grp,
                                                      "flagged": flagged["key"].tolist()}))
    print(f"\nsaved to {rep}")


def _fig_part2(t, prof, rep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for (ds, lab), g in t.groupby(["dataset", "label"]):
        col = "tab:blue" if ds == "D1" else "tab:orange"
        kw = {"marker": "o" if lab == "AD" else "^", "edgecolors": col, "s": 40, "label": f"{ds} {lab}",
              "facecolors": col if lab == "AD" else "none"}
        ax[0].scatter(g["edge_rms_ratio_max"], g["plv_excl_boundary_max"], **kw)
        ax[1].scatter(g["n_interpolated"], g["plv_all_subject_mean_max"], **kw)
    for a_, xl, yl in ((ax[0], "edge RMS ratio (max of start/end)", "PLV max |A-B|, boundary windows excluded"),
                       (ax[1], "interpolated channels", "PLV subject-mean max |A-B|")):
        a_.set_yscale("log")
        a_.set_xlabel(xl)
        a_.set_ylabel(yl)
        a_.legend(fontsize=7)
    ax[0].set_xscale("log")
    for k, g in prof.groupby("metric"):
        m = g.groupby("dist")["max"].median()
        ax[2].plot(m.index, m.values, marker="o", label=f"{k} median over recordings")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("window distance from the boundary (steps; 6 = 6+)")
    ax[2].set_ylabel("max |A-B| in the window")
    ax[2].legend(fontsize=8)
    fig.suptitle("Part 2 boundary sensitivity, A (production) vs B (reflected edges); diagnostic, B is not ground truth")
    fig.tight_layout()
    fig.savefig(rep / "part2.png", dpi=110)
    plt.close(fig)


def cmd_status(args, paths, cfg, reg, out, commit):
    for part in ("part1", "part2"):
        n = sum(done(out / part / r.key, identity(r, part, cfg, commit)) for r in reg.itertuples())
        print(f"{part}: {n} of {len(reg)} recordings complete ({out / part})")


if __name__ == "__main__":
    a = parse()
    ctx = setup(a)
    print(f"audit {AUDIT_VERSION} (script sha256 {SCRIPT_SHA[:12]}), production commit {ctx[4][:10]}, output {ctx[3]}")
    {"estimate": cmd_estimate, "part1": cmd_part1, "report1": cmd_report1, "part2": cmd_part2,
     "report2": cmd_report2, "status": cmd_status}[a.cmd](a, *ctx)
    sys.exit(0)
