"""Exp 8 (Xu-inspired dual-adjacency source-space GCN) inspection; no data read, nothing trained or saved.

    python scripts/lx_exp8_inspect.py [--config configs/experiments/lx/lx_exp8_xu_dual_gcn.yaml]

For every variant: the anatomical adjacency (groups, edges), the layer structure, the parameter count, and one
forward + backward pass on synthetic inputs of the real shapes (5 bands x 68 ROIs, the variant's node features).
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import torch

from adgraph.experiments.config import expand_variants, load_experiment
from adgraph.models.xu_dual import DK_LOBES, XuDualGCN, anatomical_adjacency, roi_groups

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/experiments/lx/lx_exp8_xu_dual_gcn.yaml")
args = parser.parse_args()
names = sorted(f"{r}-{h}" for regs in DK_LOBES.values() for r in regs for h in ("lh", "rh"))   # store order
for grouping in ("lobe_hemisphere", "lobe_bilateral"):
    a = anatomical_adjacency(names, grouping)
    print(f"anatomical adjacency ({grouping}): {dict(collections.Counter(roi_groups(names, grouping)))}; "
          f"{int(a.sum() // 2)} edges, symmetric {bool((a == a.T).all())}")
rng = np.random.default_rng(0)
for v in expand_variants(load_experiment(Path(args.config))):
    nf = v["node_features"]
    f = (len(nf.get("psd_kinds", [])) if "psd" in nf["components"] else 0) + ("strength" in nf["components"])
    m = {k: val for k, val in v["model"].items() if k not in ("architecture", "encoder")}
    net = XuDualGCN(in_dim=f, n_bands=len(v["graph"]["bands"]), n_rois=68, roi_names=names, **m)
    x = torch.from_numpy(rng.standard_normal((4, 5, 68, f)).astype(np.float32))
    adj = torch.from_numpy(np.abs(rng.uniform(0, 1, (4, 5, 68, 68))).astype(np.float32))
    adj = (adj + adj.transpose(-1, -2)) / 2 * (1 - torch.eye(68))
    out = net(x, adj)
    out.sum().backward()
    grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)
    print(f"{v['variant']:45s} F={f}  params={sum(p.numel() for p in net.parameters()):7d}  "
          f"logits {tuple(out.shape)} finite {bool(torch.isfinite(out).all())}  gradients ok {grads}  "
          f"[layer1 {m['layer1']}, layer2 {m['layer2']}, {m['anatomical_grouping']}, "
          f"{'band-specific' if m['band_specific'] else 'shared'} weights]")
print("\nlayers:", XuDualGCN(4, 5, 68, names, **{k: val for k, val in v["model"].items()
                                                  if k not in ("architecture", "encoder")}))
