"""Experiment configurations of the final framework: one YAML per experiment (``configs/final/experiments/``) and a
run context chosen at run time.

An experiment YAML may contain

    extends: ../gnn.yaml          # deep-merged underneath this file
    sweep:                        # cartesian product -> one variant per combination
      graph.connectivity: [plv, ciplv]
    variants:                     # named overrides, crossed with the sweep
      full: {}
      no_psd: {node_features.components: [strength]}
    hparams:                      # classical models only: grid searched by participant-level inner CV
      model.params.C: [0.01, 0.1, 1.0]

The run context (``with_context``) is not part of the YAML; it is given to every notebook / CLI call:

    inverse   lcmv | wmne      (Exp 0 always runs in sensor space: inverse ``sensor``)
    mode      B | A            (B: all accepted windows, the primary analysis; A: 10 windows per participant)
    atlas     dk68 | dk34_gyral | r18 | lobes12   (default dk68; coarser atlases need their own stores)
    graph     a named construction of ``configs/final/graphs.yaml`` (default ``dense``; GNN experiments only)

Each variant is a complete, self-contained config (``name`` = experiment + variant tags) and is evaluated and saved
separately; PLV and |ciPLV| results are never merged. ``validate`` enforces the final protocol: GNNs train for
exactly the protocol's fixed number of epochs with no early stopping and no hyperparameter search; the classical
models use their established participant-level inner CV for C.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import yaml

from ..source.atlas import ATLASES

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "final"
SETTING_NAMES = ("xu_dev_cv", "xu_holdout")
INVERSES = ("lcmv", "wmne", "sensor")
MODES = ("B", "A")
FIXED_EPOCHS = 60
PROTOCOL = "gnn_fixed60_v1"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, path: str, value) -> dict:
    out = copy.deepcopy(cfg)
    node = out
    *parents, leaf = path.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = copy.deepcopy(value)
    return out


def get_dotted(cfg: dict, path: str):
    node = cfg
    for p in path.split("."):
        node = node[p]
    return node


def load_experiment(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "extends" in cfg:
        base = load_experiment(path.parent / cfg.pop("extends"))
        cfg = deep_merge(base, cfg)
    return cfg


def experiment_configs(root: str | Path | None = None) -> dict[str, Path]:
    """{experiment id (exp0 ... exp8): config path} of the final framework."""
    root = Path(root or CONFIG_ROOT) / "experiments"
    out = {}
    for p in sorted(root.glob("exp*.yaml")):
        out[p.stem.split("_")[0]] = p
    return out


def graph_constructions(root: str | Path | None = None) -> dict[str, dict]:
    """Named graph constructions of ``configs/final/graphs.yaml``."""
    raw = yaml.safe_load((Path(root or CONFIG_ROOT) / "graphs.yaml").read_text())
    return raw["constructions"]


def _tag(value, path: str = "") -> str:
    if isinstance(value, bool):
        return f"{path.split('.')[-1]}={value}"
    if isinstance(value, list):
        return "+".join(map(str, value))
    return str(value)


def expand_variants(cfg: dict) -> list[dict]:
    sweep = cfg.get("sweep", {}) or {}
    named = cfg.get("variants", {}) or {"": {}}
    keys = list(sweep)
    out = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        for vname, overrides in named.items():
            v = {k: copy.deepcopy(x) for k, x in cfg.items() if k not in ("sweep", "variants")}
            tags = []
            for k, val in zip(keys, combo):
                v = set_dotted(v, k, val)
                tags.append(_tag(val, k))
            for k, val in (overrides or {}).items():
                v = set_dotted(v, k, val)
            if vname:
                tags.append(vname)
            v["variant"] = "__".join(tags) if tags else "default"
            v["name"] = f"{cfg['experiment']}__{v['variant']}"
            validate(v)
            out.append(v)
    names = [v["name"] for v in out]
    if len(set(names)) != len(names):
        raise ValueError("variant names are not unique")
    return out


def with_context(v: dict, inverse: str, mode: str, atlas: str = "dk68", graph: str | None = None,
                 root: str | Path | None = None) -> dict:
    """The variant with its run context and resolved graph construction (both part of its identity)."""
    if v["input"] == "classical" and v["features"]["source"] == "sensor":
        inverse = "sensor"                               # Exp 0: sensor space, independent of the inverse
        if atlas != "dk68":
            raise ValueError("Exp 0 (sensor space) has no atlas; use dk68")
    if inverse not in INVERSES or mode not in MODES or atlas not in ATLASES:
        raise ValueError(f"unknown context inverse={inverse!r} mode={mode!r} atlas={atlas!r}")
    if v["input"] == "classical":
        if graph not in (None, "none"):
            raise ValueError(f"{v['name']}: graph construction applies to GNN experiments only")
        gname, gcfg = "none", None
    else:
        gname = graph or "dense"
        cons = graph_constructions(root)
        if gname not in cons:
            raise ValueError(f"unknown graph construction {gname!r}; choose from {list(cons)}")
        gcfg = dict(cons[gname])
    out = copy.deepcopy(v)
    out["context"] = {"inverse": inverse, "mode": mode, "atlas": atlas, "graph": gname}
    out["graph_construction"] = gcfg
    if gcfg and gcfg.get("adjacency") == "feature_similarity":
        out["graph"] = {**out["graph"], "adjacency": "feature_similarity"}
        out["graph_construction"] = {k: x for k, x in gcfg.items() if k != "adjacency"}
    return out


def validate(v: dict) -> None:
    if v.get("target", "label") != "label":
        raise ValueError(f"{v.get('name')}: the only target is the diagnosis (AD = 1, HC = 0)")
    unknown = set(v.get("settings") or []) - set(SETTING_NAMES)
    if unknown or not v.get("settings"):
        raise ValueError(f"{v.get('name')}: settings must be a non-empty subset of {SETTING_NAMES}")
    if v["input"] == "graph":
        g = v["graph"]
        if g["connectivity"] not in ("plv", "ciplv"):
            raise ValueError("graph.connectivity must be plv or ciplv (never a mix)")
        if g["band_mode"] == "single" and len(g["bands"]) != 1:
            raise ValueError(f"{v.get('name')}: single-band mode with bands {g['bands']}")
        if g["band_mode"] in ("mean", "multi") and len(g["bands"]) < 1:
            raise ValueError("no bands selected")
        if hparam_grid(v) != [{}]:
            raise ValueError(f"{v.get('name')}: the fixed GNN protocol forbids hyperparameter search ({v['hparams']})")
        t = v["training"]
        if int(t.get("epochs", 0)) != FIXED_EPOCHS or (t.get("early_stopping") or {}).get("enabled"):
            raise ValueError(f"{v.get('name')}: GNNs train for exactly {FIXED_EPOCHS} epochs without early stopping")
        if (v.get("protocol") or {}).get("name") != PROTOCOL:
            raise ValueError(f"{v.get('name')}: GNN protocol must be {PROTOCOL}")
        m = v["model"]
        if (m.get("architecture", "multiband_gnn") == "multiband_gnn" and g["band_mode"] == "multi"
                and len(g["bands"]) > 1 and m.get("shared_encoder", True) and m.get("fusion") in ("mean", "attention")
                and m.get("band_identity", "none") == "none" and not m.get("band_invariant_baseline", False)):
            raise ValueError(f"{v.get('name')}: a shared-encoder multiband GNN with symmetric fusion is invariant to "
                             "the band order; set model.band_identity: onehot (or declare band_invariant_baseline)")
    elif v["input"] == "classical":
        if v["model"]["name"] not in ("logreg", "svm"):
            raise ValueError("classical model must be logreg or svm")
    else:
        raise ValueError(f"unknown input {v['input']!r}")


def hparam_grid(v: dict) -> list[dict]:
    """List of {dotted.path: value} combinations from ``hparams`` (one empty dict if none)."""
    grid = v.get("hparams", {}) or {}
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))] or [{}]


def apply_hparams(v: dict, params: dict) -> dict:
    for k, val in params.items():
        v = set_dotted(v, k, val)
    return v
