"""Experiment 7 (EARM-GNN) helpers for the Exp7 notebook. Training and evaluation themselves use
``scripts/experiment.py run|report --config configs/experiments/exp7_earm_gnn.yaml`` (same runner,
splits, checkpoints and result writers as every experiment).

    python scripts/exp7_earm.py inspect   # verify the graph run + features, build every variant, print layers,
                                         # parameter counts and tensor shapes, forward/backward smoke test
    python scripts/exp7_earm.py compare   # comparison tables + ROC / confusion / performance figures
                                         # (EARM-GNN vs GCN experiments vs PSD baselines), saved outputs only
    python scripts/exp7_earm.py diagnose  # short read-only D1 / D2 diagnostic (QC, features, predictions)

``--synthetic`` uses a synthetic graph store (tests only). Nothing here changes the graph run or any result.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adgraph.experiments.config import expand_variants, load_experiment, set_dotted
from adgraph.graphs.dataset import build_dataset, node_strength
from adgraph.graphs.store import GraphStore, check_adjacency

EXPECTED_GRAPH_RUN_ID = "6d75de9975f0d9ae"          # frozen v4 graph run graphs_v4_33faa4b89a5d
REQUIRED = ("plv", "ciplv", "psd_log", "psd_rel")

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["inspect", "compare", "diagnose"])
parser.add_argument("--config", default="configs/experiments/exp7_earm_gnn.yaml")
parser.add_argument("--paths", default="configs/paths.yaml")
parser.add_argument("--graphs", help="graph run (default: latest assembled full run in processed_dir)")
parser.add_argument("--results", help="experiments root (default: <processed_dir>/experiments)")
parser.add_argument("--out", help="output folder (default: inside the experiment / diagnostics folder)")
parser.add_argument("--expect-graph-run-id", default=EXPECTED_GRAPH_RUN_ID)
parser.add_argument("--synthetic", action="store_true", help="synthetic graph store (tests only)")
parser.add_argument("--allow-ephemeral-storage", action="store_true")
args = parser.parse_args()
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 80)
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
cfg = load_experiment(args.config)
EXP = cfg["experiment"]


def load_store():
    if args.synthetic:
        from adgraph.synthetic import make_synthetic_store

        return make_synthetic_store({"D1": (4, 4), "D2": (4, 5)}, windows_per_subject=(3, 4), psd_effect=0.6,
                                    conn_effect=0.6, seed=0), None
    from adgraph.config import load_paths
    from adgraph.pipeline import latest_complete_graph_run

    paths = load_paths(args.paths)
    run = Path(args.graphs) if args.graphs else latest_complete_graph_run(paths["processed_dir"])
    return GraphStore.load(run), paths


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)


def fail(msg):
    print(f"\nFAILED: {msg}", flush=True)
    sys.exit(1)


# ====================================================================== inspect
def inspect():
    import torch

    from adgraph.models.earm import parameter_table
    from adgraph.models.trainer import GNNModel

    store, paths = load_store()
    section("1. Frozen graph run")
    gid = store.manifest.get("graph_run_id", "synthetic")
    print(f"graph run: {store.root or 'synthetic'}\ngraph_run_id: {gid}   pipeline version: "
          f"{store.manifest.get('config', {}).get('version', 'n/a')}   commit: {store.manifest.get('git_commit', 'n/a')}")
    if not args.synthetic and args.expect_graph_run_id and gid != args.expect_graph_run_id:
        fail(f"graph_run_id {gid} is not the frozen v4 run {args.expect_graph_run_id}")
    w = store.windows
    print(f"windows: {len(w)}; subjects: {w['key'].nunique()}")
    print(w.groupby(["dataset", "label"]).agg(subjects=("key", "nunique"), windows=("key", "size")).to_string())

    section("2. Required graph features")
    checks = {}
    for k in REQUIRED:
        if k not in store.arrays:
            fail(f"array {k} missing from the graph run")
        a = store.arrays[k]
        finite = bool(np.isfinite(a).all())
        checks[k] = {"shape": list(a.shape), "finite": finite}
        if not finite:
            fail(f"{k} has non-finite values")
        if k in ("plv", "ciplv"):
            check_adjacency(a, k)
        print(f"{k:8s} shape {a.shape}  finite  " + ("symmetric, zero diagonal, in [0, 1]" if k in ("plv", "ciplv") else ""))
    if list(store.bands) != cfg["graph"]["bands"]:
        fail(f"graph-run bands {store.bands} differ from the config {cfg['graph']['bands']}")
    print("bands:", list(store.bands), "| ROIs:", store.arrays["plv"].shape[-1])

    section("3. Variants, model inputs and parameter counts")
    variants = expand_variants(cfg)
    rows, nets = [], {}
    for v in variants:
        data = build_dataset(store, v)
        x, adj = data.payload["x"], data.payload["adj"]
        if not np.allclose(x[..., data.info["feature_names"].index("strength")], node_strength(adj), atol=1e-5):
            fail("node strength is not the strength of the band's own EEG adjacency")
        m = GNNModel(v["model"], v["training"], seed=0)
        torch.manual_seed(0)
        net = m.build_net(x.shape[-1], x.shape[1], 0, n_rois=x.shape[2])
        nets[v["variant"]] = (net, data)
        rows.append({"variant": v["variant"], "x": tuple(x.shape), "adj": tuple(adj.shape),
                     "node_features": ",".join(data.info["feature_names"]),
                     "parameters": sum(p.numel() for p in net.parameters())})
    print(pd.DataFrame(rows).to_string(index=False))
    ref = expand_variants(load_experiment("configs/experiments/exp6_multiband_psd_strength_fixedhp.yaml"))[0]
    ref_net = GNNModel(ref["model"], ref["training"]).build_net(3, 4, 0, n_rois=68)
    print(f"reference: Exp6 GCN full (fixed protocol) has {sum(p.numel() for p in ref_net.parameters())} parameters")

    full_name = next(v["variant"] for v in variants if v["variant"].endswith("__full"))
    net, data = nets[full_name]
    section(f"4. Layer structure ({full_name})")
    print(net)
    print("\ntrainable parameters:")
    tab = pd.DataFrame(parameter_table(net), columns=["parameter", "shape", "count"])
    print(tab.to_string(index=False))
    print(f"TOTAL trainable parameters: {int(tab['count'].sum())}")

    section("5. Tensor shapes (one forward pass, batch of 8 windows)")
    shapes = []
    hooks = [mod.register_forward_hook(lambda mod_, inp, out, n=n: shapes.append(
        (n, tuple(inp[0].shape) if inp and hasattr(inp[0], "shape") else "-",
         tuple(out[0].shape if isinstance(out, tuple) else out.shape))))
        for n, mod in net.named_modules()
        if n and n != "gate" and (n.count(".") == 0 and n != "blocks" or re.fullmatch(r"blocks\.\d+", n))]
    xb = torch.from_numpy(np.ascontiguousarray(data.payload["x"][:8]))
    ab = torch.from_numpy(np.ascontiguousarray(data.payload["adj"][:8]))
    net.eval()
    with torch.no_grad():
        logits, bw = net(xb, ab, return_band_weights=True)
        g = net.gates()
        p = net.operator(ab)
    for h in hooks:
        h.remove()
    for n, i, o in shapes:
        print(f"{n:14s} in {str(i):22s} out {o}")
    print(f"edge gates G: {tuple(g.shape)} in ({float(g.min()):.3f}, {float(g.max()):.3f}), symmetric "
          f"{bool(torch.allclose(g, g.transpose(-1, -2)))}")
    print(f"propagation P: {tuple(p.shape)} finite {bool(torch.isfinite(p).all())}, symmetric "
          f"{bool(torch.allclose(p, p.transpose(-1, -2), atol=1e-6))}")
    print(f"logits {tuple(logits.shape)}; band weights {tuple(bw.shape)} (softmax over theta, alpha1, alpha2, beta)")

    section("6. Forward/backward smoke test (a few training steps; NOT an evaluation)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    v = set_dotted(next(x for x in variants if x["variant"] == full_name), "training.epochs", 2)
    keys = data.meta["key"].unique()[: max(4, len(data.meta["key"].unique()) // 4)]
    idx = np.flatnonzero(data.meta["key"].isin(keys).to_numpy())
    m = GNNModel(v["model"], v["training"], seed=0, device=dev)
    t0 = time.perf_counter()
    m.fit(data.view(idx, "train"))
    sec = time.perf_counter() - t0
    losses = [h["train_loss"] for h in m.history]
    print(f"{len(idx)} windows of {len(keys)} subjects, 2 epochs in {sec:.1f} s; train loss {losses}")
    if not np.isfinite(losses).all():
        fail("non-finite training loss")
    net = m.net.train()
    xt, at, _ = m._tensors(data.view(idx[:32], "train"))
    y = torch.from_numpy(data.meta["y"].to_numpy()[idx[:32]].astype(np.float32)).to(dev)
    net.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt.to(dev), at.to(dev)), y) + net.regularization()
    loss.backward()
    grads = {n: float(p.grad.norm()) for n, p in net.named_parameters() if n.startswith("gate.")}
    print("gradient norms of the edge gates:", {k: f"{val:.2e}" for k, val in grads.items()})
    if not grads or not all(np.isfinite(list(grads.values()))) or not any(val > 0 for val in grads.values()):
        fail("gradients do not reach the edge gates")
    per_window_epoch = sec / (2 * len(idx))
    print("(timing of this tiny smoke run includes start-up overhead; for a measured runtime estimate run "
          "`python scripts/experiment.py estimate --config configs/experiments/exp7_earm_gnn.yaml`)")
    print("\nSMOKE TEST PASSED")
    if not args.synthetic:
        from adgraph.storage import check_persistent

        out = Path(args.out or Path(args.results or f"{paths['processed_dir']}/experiments") / EXP / "_inspect")
        check_persistent({"inspect": out}, allow_ephemeral=args.allow_ephemeral_storage)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"inspect_{stamp}.json").write_text(json.dumps(
            {"graph_run_id": gid, "arrays": checks, "variants": rows, "parameters_full": int(tab["count"].sum()),
             "smoke_train_loss": losses, "gate_grad_norms": grads, "device": dev,
             "ms_per_window_epoch": per_window_epoch * 1e3}, indent=1, default=str))
        print(f"saved: {out / f'inspect_{stamp}.json'}")


# ====================================================================== compare / diagnose
def results_root(paths):
    return Path(args.results or f"{paths['processed_dir']}/experiments")


def compare():
    from adgraph.experiments.compare import write_comparison

    paths = None
    if not args.synthetic:
        from adgraph.config import load_paths

        paths = load_paths(args.paths)
    root = Path(args.results) if args.results else results_root(paths)
    out = Path(args.out) if args.out else root / EXP / "_comparison" / f"comparison_{stamp}"
    if not args.synthetic:
        from adgraph.storage import check_persistent

        check_persistent({"comparison": out}, allow_ephemeral=args.allow_ephemeral_storage)
    res = write_comparison(root, out, "configs/figures/style.yaml")
    section("Comparison: subject-level ROC AUC (mean over CV repeats; cross-dataset: one train/test)")
    print(res["pivot"].round(3).to_string() if len(res["pivot"]) else "no results yet")
    t = res["table"]
    if len(t):
        cols = ["model", "connectivity", "setting", "n_repeats", "roc_auc_mean", "roc_auc_sd", "roc_auc_ci95_low",
                "roc_auc_ci95_high", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean",
                "n_parameters", "fit_seconds_mean"]
        section("Full table")
        print(t[[c for c in cols if c in t]].round(3).to_string(index=False))
    d = res["differences"]
    if len(d):
        section("EARM-GNN full minus comparator, per shared CV repeat (DESCRIPTIVE, no significance test)")
        print(d[d["metric"] == "roc_auc"].drop(columns=["reference", "metric"]).round(3).to_string(index=False))
    print(f"\nfigures: {', '.join(res['figures']) or 'none'}")
    print(f"saved to {res['dir']}")


def diagnose():
    from adgraph.evaluation.dataset_shift import diagnose as run

    store, paths = load_store()
    root = Path(args.results) if args.results else (results_root(paths) if paths else None)
    out = Path(args.out) if args.out else Path(paths["processed_dir"]) / "diagnostics" / f"dataset_shift_{stamp}"
    if not args.synthetic:
        from adgraph.storage import check_persistent

        check_persistent({"diagnostic": out}, allow_ephemeral=args.allow_ephemeral_storage)
    res = run(store, root, out)
    print(res["report"])
    print(f"\nsaved to {res['dir']}")


{"inspect": inspect, "compare": compare, "diagnose": diagnose}[args.action]()
