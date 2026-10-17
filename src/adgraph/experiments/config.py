"""Experiment configs: ``extends`` inheritance, dotted overrides and variant expansion.

An experiment YAML may contain

    extends: base.yaml            # deep-merged underneath this file
    sweep:                        # cartesian product -> one variant per combination
      graph.connectivity: [plv, ciplv]
    variants:                     # named overrides, crossed with the sweep
      full: {}
      no_psd: {node_features.components: [strength]}
    hparams:                      # grid searched by subject-level inner CV (not a variant)
      model.hidden: [32, 64]

A named variant may set its own ``settings`` (a subset of the experiment's), e.g. ablations that run
only in some settings; ``plan_units`` and ``scripts/run_experiment.py`` honour it.

Each variant is a complete, self-contained config (``name`` = experiment name +
variant tags) and is evaluated and saved separately, so PLV and ciPLV results are
never merged.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import yaml

SETTING_NAMES = ("within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1", "within_D3",
                 "xu_dev_cv", "xu_holdout")
# graph-run family an experiment reads -> (graph-run prefix under processed_dir, results root, allowed settings).
# None = D1/D2; D3 = wMNE D3 runs; LX / LX10 = D3 LCMV / Xu suite (entry 50), modes B (all windows) / A (tenfold).
FAMILIES = {
    None: ("graphs", "experiments", ("within_D1", "within_D2", "pooled", "cross_D1_to_D2", "cross_D2_to_D1")),
    "D3": ("graphsD3", "experiments_D3", ("within_D3",)),
    "LX": ("D3_LCMV_XU/graphsLX", "D3_LCMV_XU/experiments_B_all_windows", ("xu_dev_cv", "xu_holdout")),
    "LX10": ("D3_LCMV_XU/graphsLX10", "D3_LCMV_XU/experiments_A_xu_tenfold", ("xu_dev_cv", "xu_holdout")),
    # matched wMNE / Xu suite (entry 55): same protocol, wMNE instead of LCMV; and the historical wMNE graphs
    # (graphsD3_v4, 20 s windows, 4 bands) under the same frozen split (D3H, a separate historical comparison)
    "WX": ("D3_WMNE_XU/graphsWX", "D3_WMNE_XU/experiments_B_all_windows", ("xu_dev_cv", "xu_holdout")),
    "WX10": ("D3_WMNE_XU/graphsWX10", "D3_WMNE_XU/experiments_A_xu_tenfold", ("xu_dev_cv", "xu_holdout")),
    "D3H": ("graphsD3", "D3_WMNE_XU/experiments_H_historical_graphs", ("xu_dev_cv", "xu_holdout")),
}
# families evaluated with the frozen LCMV / Xu participant split (reference: the validated LX / LX10 stores)
XU_FAMILIES = ("LX", "LX10", "WX", "WX10", "D3H")
GRAPH_FAMILIES = tuple(FAMILIES)


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


def validate(v: dict) -> None:
    if v["input"] == "graph":
        g = v["graph"]
        if g["connectivity"] not in ("plv", "ciplv"):
            raise ValueError("graph.connectivity must be plv or ciplv (never a mix)")
        if g["band_mode"] == "single" and len(g["bands"]) != 1:
            raise ValueError(f"{v.get('name')}: single-band mode with bands {g['bands']}")
        if g["band_mode"] in ("mean", "multi") and len(g["bands"]) < 1:
            raise ValueError("no bands selected")
    elif v["input"] == "classical":
        if v["model"]["name"] not in ("logreg", "svm"):
            raise ValueError("classical model must be logreg or svm")
    else:
        raise ValueError(f"unknown input {v['input']!r}")
    if (v.get("protocol") or {}).get("hyperparameter_search") == "none" and len(hparam_grid(v)) != 1:
        raise ValueError(f"{v.get('name')}: protocol {v['protocol'].get('name')!r} forbids hyperparameter search, "
                         f"but the config defines a grid {v.get('hparams')}")
    if v.get("graph_family") not in GRAPH_FAMILIES:
        raise ValueError(f"{v.get('name')}: graph_family must be one of {GRAPH_FAMILIES}")
    d3 = {s for s in v.get("settings") or [] if s.endswith("D3")}
    if (v.get("graph_family") == "D3") != bool(d3) or (d3 and set(v["settings"]) != d3):
        raise ValueError(f"{v.get('name')}: D3 settings and graph_family D3 go together and are never mixed with D1/D2")
    allowed = FAMILIES[v.get("graph_family")][2]
    if set(v.get("settings") or []) - set(allowed):
        raise ValueError(f"{v.get('name')}: settings {v.get('settings')} not allowed for graph_family "
                         f"{v.get('graph_family')} ({allowed}); families are never mixed")
    unknown = set(v.get("settings") or []) - set(SETTING_NAMES)
    if unknown:
        raise ValueError(f"{v.get('name')}: unknown settings {sorted(unknown)}")


def hparam_grid(v: dict) -> list[dict]:
    """List of {dotted.path: value} combinations from ``hparams`` (empty dict if none)."""
    grid = v.get("hparams", {}) or {}
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))] or [{}]


def apply_hparams(v: dict, params: dict) -> dict:
    for k, val in params.items():
        v = set_dotted(v, k, val)
    return v
