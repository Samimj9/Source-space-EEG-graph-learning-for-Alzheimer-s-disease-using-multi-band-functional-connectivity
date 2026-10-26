"""Machine-readable manifest for the separate D3 <-> D2 project (amendment A3, objective F).

Reads D3 development-CV results only (``discover(settings=("xu_dev_cv",))``): no holdout metric and no D2 data or outcome
is opened, and nothing of D2 is processed here. The recommended source-only set follows the criteria of
``configs/final/amendments/a3_statistics.yaml`` (written before this code ran); every completed variant of Exp 3-8a stays
in the manifest as available for exploratory transfer analyses. The manifest does not say that a model WILL transfer;
the 68-ROI / lobes12 stores of the D3 analysis are not the 19-electrode inputs of that project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import inventory
from . import namespace as ns


def build(processed, atlas: str, graph: str, cfg: dict | None = None) -> dict:
    from .. import provenance

    cfg = cfg or ns.stats_config()
    d2 = cfg["d2_transfer"]
    t, _ = inventory.discover(processed, settings=("xu_dev_cv",))
    t = t[(t["mode"] == "B") & (t["atlas"] == atlas) & t["graph"].isin([graph, "none"]) & (t["inverse"] != "sensor")]
    fams = list(d2["families"]) + [d2["baseline_family"]]
    models, per_variant = [], []
    for r in t[t["experiment"].isin(fams)].to_dict("records"):
        man = json.loads((Path(processed) / r["rel_dir"] / "manifest.json").read_text())
        models.append({"family": r["family"], "experiment": r["experiment"], "variant": r["variant"],
                       "inverse": r["inverse"], "mode": r["mode"], "atlas": r["atlas"], "graph": r["graph"],
                       "namespace": r["namespace"], "git_commit": r["git_commit"],
                       "d3_dev_auc": r["auc"], "d3_dev_auc_ci": [r["auc_ci_low"], r["auc_ci_high"]],
                       "variant_config": man["variant_config"], "variant_config_hash": man["variant_config_hash"],
                       "training_config_hash": man["training_config_hash"], "result_dir": r["rel_dir"],
                       "available_for_exploratory_transfer": True})
    if models:
        df = pd.DataFrame(models)
        both = df.groupby(["experiment", "variant"]).filter(lambda g: set(g["inverse"]) >= {"lcmv", "wmne"})
        score = both.groupby(["experiment", "variant"])["d3_dev_auc"].mean().reset_index(name="d3_mean_dev_auc")
    else:
        score = pd.DataFrame(columns=["experiment", "variant", "d3_mean_dev_auc"])
    rec = []
    for fam in fams:
        s = score[score["experiment"] == fam].sort_values(["d3_mean_dev_auc", "variant"], ascending=[False, True])
        if len(s):
            r = s.iloc[0]
            rec.append({"experiment": fam, "variant": r["variant"], "d3_mean_dev_auc_over_inverses":
                        float(r["d3_mean_dev_auc"]), "role": "baseline" if fam == d2["baseline_family"] else "model",
                        "rule": "highest mean development AUC over LCMV and wMNE within the family (ties: name)",
                        "selection_bias_note": "chosen as the best of its family on the same D3 participants; its D3 "
                                               "AUC is optimistic, which does not matter for the transfer design"})
    sig = ns.sha256_json(sorted((m["experiment"], m["variant"], m["inverse"], m["d3_dev_auc"]) for m in models))
    return {"amendment": ns.AID, "kind": "D3 -> D2 manifest", "atlas": atlas, "graph": graph,
            "criteria": d2["criteria"], "input_requirement": d2["input_requirement"],
            "d3_input": {"settings_read": ["xu_dev_cv"], "holdout_used": False, "d2_outcomes_used": False,
                         "d2_processing_started": False, "input_results_sha256": sig},
            "recommended_source_only_set": rec, "models": models, "n_models": len(models),
            "a3_commit": provenance.git_commit(), "git_dirty": provenance.git_dirty()}


def write(processed, atlas: str, graph: str, cfg=None) -> Path:
    m = build(processed, atlas, graph, cfg)
    return ns.new_file(ns.root(processed) / "d2_manifest" / f"d2_manifest_{ns.stamp()}.json",
                       json.dumps(m, indent=2, default=str))
