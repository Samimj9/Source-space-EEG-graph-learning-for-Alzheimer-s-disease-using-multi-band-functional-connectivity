"""Read-only diagnostic of the channel-mapping QC statistic ``qc_spatial_rho`` (``io.layout.spatial_consistency``).

    python scripts/legacy/diagnose_spatial.py --key D1-1 [--config configs/pipeline.yaml]
    python scripts/legacy/diagnose_spatial.py --key D3-1 --config configs/pipeline_d3.yaml
    python scripts/legacy/diagnose_spatial.py --key D3-1 --config configs/pipeline_d3.yaml --path <file.set> --dataset D3
    python scripts/legacy/diagnose_spatial.py --simulate [--config ...]      # spherical head model, no EEG read

Recording mode repeats the pipeline's own loading and preprocessing of ONE recording (load_raw, edge crop,
discontinuity segments, preprocess) and evaluates the statistic on:
* the signal just BEFORE the average reference (recording reference: D1/D2 as recorded, D3 A1-A2), captured
  inside preprocess by wrapping ``set_eeg_reference`` in this process only;
* the signal AFTER the average reference (what the pipeline evaluates; must reproduce ``qc_spatial_rho``).
For each: Spearman rho of distance vs |r| (the pipeline statistic) and vs signed r; mean r / |r| and the
fraction of negative r for near / middle / far thirds of the electrode pairs; the nearest and farthest pairs;
the share of variance carried by the channel mean (common signal).

Simulation mode: random dipoles in a 3-shell spherical head (MNE), electrodes of the configured montage,
reference-free potentials vs average reference vs a mastoid-like / Cz reference.

Nothing in the pipeline, the graph runs or the pins is changed. Output: printed tables and
<processed>/diagnostics/spatial_<key|simulation>_<time>/spatial.json (recording mode with --paths).
"""

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adgraph.config import common_channels, load_paths, load_yaml


def pair_table(data: np.ndarray, pos: np.ndarray, names: list[str]) -> pd.DataFrame:
    r = np.corrcoef(data)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    iu = np.triu_indices(len(names), 1)
    return pd.DataFrame({"a": np.array(names)[iu[0]], "b": np.array(names)[iu[1]], "dist_mm": d[iu] * 1000,
                         "r": r[iu]})


def summary(data: np.ndarray, pos: np.ndarray, names: list[str]) -> dict:
    t = pair_table(data, pos, names)
    t["third"] = pd.qcut(t["dist_mm"], 3, labels=["near", "middle", "far"])
    g = t.groupby("third", observed=True).agg(dist_mm=("dist_mm", "median"), mean_r=("r", "mean"),
                                              mean_abs_r=("r", lambda x: np.abs(x).mean()),
                                              frac_negative=("r", lambda x: (x < 0).mean()))
    mean_sig = data.mean(axis=0)
    return {
        "rho_abs": float(spearmanr(t["dist_mm"], t["r"].abs()).statistic),     # the pipeline statistic
        "rho_signed": float(spearmanr(t["dist_mm"], t["r"]).statistic),
        "mean_r": float(t["r"].mean()),
        "common_signal_variance_share": float(mean_sig.var() / data.var(axis=1).mean()),
        "by_distance": g.round(3).reset_index().to_dict("records"),
        "nearest": t.nsmallest(5, "dist_mm").round(3).to_dict("records"),
        "farthest": t.nlargest(5, "dist_mm").round(3).to_dict("records"),
    }


def mapping_sensitivity(data: np.ndarray, pos: np.ndarray, names: list[str], n_perm: int = 1000, seed: int = 0) -> dict:
    """How each statistic separates the real channel-to-position mapping from wrong ones: positions shuffled
    across channels (permutation null) and one pair of electrodes from opposite sides swapped."""
    r = np.corrcoef(data)
    iu = np.triu_indices(len(names), 1)

    def stats(p):
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)[iu]
        return spearmanr(d, np.abs(r[iu])).statistic, spearmanr(d, r[iu]).statistic
    obs = stats(pos)
    rng = np.random.default_rng(seed)
    null = np.array([stats(pos[rng.permutation(len(names))]) for _ in range(n_perm)])
    swaps = {}
    for a, b in (("Fp1", "O2"), ("F7", "T6"), ("C3", "C4"), ("Fp1", "Fp2")):
        if a in names and b in names:
            q = pos.copy()
            i, j = names.index(a), names.index(b)
            q[[i, j]] = q[[j, i]]
            swaps[f"{a}<->{b}"] = [round(float(v), 3) for v in stats(q)]
    out = {"observed": {"rho_abs": float(obs[0]), "rho_signed": float(obs[1])},
           "shuffled_positions": {k: {"p05": float(np.percentile(null[:, i], 5)), "median": float(np.median(null[:, i])),
                                      "p95": float(np.percentile(null[:, i], 95)),
                                      "fraction_le_observed": float((null[:, i] <= obs[i]).mean())}
                                  for i, k in enumerate(("rho_abs", "rho_signed"))},
           "one_swap": swaps}
    print("\nmapping sensitivity (same signal; [rho_abs, rho_signed]):")
    print(f"  correct mapping: [{obs[0]:+.3f}, {obs[1]:+.3f}]")
    for k, v in out["shuffled_positions"].items():
        print(f"  {k} with positions shuffled ({n_perm}x): median {v['median']:+.3f}, 5-95 % [{v['p05']:+.3f}, "
              f"{v['p95']:+.3f}]; fraction of shuffles <= observed {v['fraction_le_observed']:.3f}")
    for k, v in swaps.items():
        print(f"  one swap {k}: [{v[0]:+.3f}, {v[1]:+.3f}]")
    return out


def show(name: str, s: dict) -> None:
    print(f"\n--- {name}: rho(distance, |r|) = {s['rho_abs']:+.3f}   rho(distance, r) = {s['rho_signed']:+.3f}   "
          f"mean r = {s['mean_r']:+.3f}   common-signal variance share = {s['common_signal_variance_share']:.3f}")
    print(pd.DataFrame(s["by_distance"]).to_string(index=False))
    print("nearest pairs: ", ", ".join(f"{p['a']}-{p['b']} {p['dist_mm']:.0f}mm r={p['r']:+.2f}" for p in s["nearest"]))
    print("farthest pairs:", ", ".join(f"{p['a']}-{p['b']} {p['dist_mm']:.0f}mm r={p['r']:+.2f}" for p in s["farthest"]))


def recording(args, cfg) -> dict:
    import mne

    from adgraph.io.layout import spatial_consistency
    from adgraph.legacy.pipeline import build_registry, load_raw
    from adgraph.preprocessing import discontinuities as disc
    from adgraph.preprocessing.boundary import boundary_aware, crop_edges
    from adgraph.preprocessing.clean import preprocess

    if args.path:
        path, dataset = args.path, args.dataset
    else:
        reg = build_registry(load_paths(args.paths), cfg=cfg)[0].set_index("key")
        path, dataset = reg.loc[args.key, "path"], reg.loc[args.key, "dataset"]
    # the pipeline's own steps (process_subject, up to the spatial QC)
    raw, _ = load_raw(path, dataset, cfg)
    segs, offset = None, 0
    if disc.enabled(cfg):
        cuts, _ = disc.boundary_cuts(raw)
        sf_in = raw.info["sfreq"]
    if boundary_aware(cfg):
        crop = crop_edges(raw, cfg)
        offset = int(round(crop["crop_start_s"] * cfg["preprocessing"]["sfreq"]))
        if disc.enabled(cfg):
            segs = disc.plan(raw, int(round(crop["crop_start_s"] * sf_in)), cuts, cfg)["usable"]
    captured = {}
    original = mne.io.BaseRaw.set_eeg_reference

    def capture(self, *a, **k):                      # this process only: keep the signal before referencing
        captured["data"] = self.get_data().copy()
        return original(self, *a, **k)
    mne.io.BaseRaw.set_eeg_reference = capture
    try:
        raw, _ = preprocess(raw, cfg, segments=segs, offset_new=offset)
    finally:
        mne.io.BaseRaw.set_eeg_reference = original
    names = raw.ch_names
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in range(len(names))])
    after = raw.get_data()
    out = {"key": args.key, "path": str(path), "channels": names,
           "positions_head_m": pos.round(5).tolist(),
           "pipeline_qc_spatial_rho": spatial_consistency(after, pos),
           "before_average_reference": summary(captured["data"], pos, names),
           "after_average_reference": summary(after, pos, names)}
    print(f"{args.key}: {len(names)} channels in order {names}")
    print(f"pipeline statistic reproduced: qc_spatial_rho = {out['pipeline_qc_spatial_rho']:+.6f}")
    show("BEFORE average reference (recording reference)", out["before_average_reference"])
    show("AFTER average reference (what the pipeline evaluates)", out["after_average_reference"])
    out["mapping_sensitivity_after_average_reference"] = mapping_sensitivity(after, pos, names)
    return out


def simulate(cfg, n_dipoles: int = 400, seed: int = 0) -> dict:
    """Random dipoles (positions inside the brain shell, random orientations, independent white time
    courses) in a 3-shell spherical head; potentials at the configured electrodes."""
    import mne

    from adgraph.io.eeg import make_montage

    names = common_channels(cfg)
    mont = make_montage("standard_1020")
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(mont, match_case=False)
    sphere = mne.make_sphere_model("auto", "auto", info, verbose="error")
    rng = np.random.default_rng(seed)
    r0 = sphere["r0"]
    # cortex-like depth: 70-90 % of the inner-skull radius
    rad = sphere["layers"][0]["rad"] * rng.uniform(0.70, 0.90, n_dipoles)
    u = rng.standard_normal((n_dipoles, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u[:, 2] = np.abs(u[:, 2])                                     # upper hemisphere (under the electrodes)
    pos = r0 + u * rad[:, None]
    ori = rng.standard_normal((n_dipoles, 3))
    ori /= np.linalg.norm(ori, axis=1, keepdims=True)
    dip = mne.Dipole(np.zeros(n_dipoles), pos, np.ones(n_dipoles) * 1e-8, ori, np.ones(n_dipoles))
    fwd = mne.make_forward_dipole(dip, sphere, info, verbose="error")[0]
    L = fwd["sol"]["data"]                                        # (n_channels, n_dipoles), reference-free
    x = L @ rng.standard_normal((n_dipoles, 20000))
    epos = np.array([info["chs"][i]["loc"][:3] for i in range(len(names))])
    ref = {"reference-free (infinity)": x, "average reference": x - x.mean(axis=0)}
    if "Cz" in names:
        ref["Cz reference (Cz row excluded)"] = None
    out = {}
    for k, v in ref.items():
        if v is None:
            keep = [i for i, n in enumerate(names) if n != "Cz"]
            v, p, n_ = (x - x[names.index("Cz")])[keep], epos[keep], [names[i] for i in keep]
        else:
            p, n_ = epos, names
        out[k] = summary(v, p, n_)
        show(f"simulation, {k}", out[k])
    out["mapping_sensitivity_average_reference"] = mapping_sensitivity(ref["average reference"], epos, names)
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--config", default="configs/pipeline.yaml")
parser.add_argument("--key", help="recording key, e.g. D1-1, D2-3, D3-1")
parser.add_argument("--path", help="read this file instead of the registry entry (with --dataset)")
parser.add_argument("--dataset", default=None)
parser.add_argument("--simulate", action="store_true")
args = parser.parse_args()
cfg = load_yaml(args.config)
pd.set_option("display.width", 200)
if args.simulate:
    res = {"simulation": simulate(copy.deepcopy(cfg)), "config": args.config}
    name = "simulation"
else:
    if not args.key:
        parser.error("--key is required (or --simulate)")
    res = recording(args, cfg)
    name = args.key
if Path(args.paths).exists():
    out = Path(load_paths(args.paths)["processed_dir"]) / "diagnostics" / \
        f"spatial_{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spatial.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nsaved: {out / 'spatial.json'}")
sys.exit(0)
