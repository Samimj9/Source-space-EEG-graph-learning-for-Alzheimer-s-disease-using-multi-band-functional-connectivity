"""Single-model holdout of the A3 frozen analysis (methodology log entry 62; ``docs/final/a3_single_model.md``).

A separate, explicitly development-CV-selected evaluation of ONE frozen unit (``configs/final/amendments/
a3_single_model.yaml``), chosen by the author after development CV to save computation. It is NOT completion of the full
A3 holdout protocol: the other frozen units are not evaluated here, the master freeze, both frozen records, the 116-unit
plan and every result are read and never changed, and nothing is selected by holdout outcomes.

Lighter than the full orchestration, without weakening what this evaluation needs: the master freeze and both records are
verified (checksums, unit sets from configuration, pins, QC decisions, store lock), and the selected unit's development
result is verified against the frozen record (completion marker, payload checksums, metrics / prediction hashes, AUC).
Not re-read: the development results of the other units (a Drive scan of ~320 result folders); the full ``a3.py verify``
still does that. The unit runs at the A2 pin through ``scripts/a3_single_holdout.py`` (pinned code, same gates as
``amendment.holdout``), so model, protocol and configuration are the pinned, unchanged ones.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .. import release
from ..experiments.config import CONFIG_ROOT
from . import environment, freeze, holdout, inventory
from . import namespace as ns


class SingleRefused(RuntimeError):
    """The single-model evaluation is refused (nothing was run or written)."""


def selection() -> dict:
    return yaml.safe_load((Path(CONFIG_ROOT) / "amendments" / "a3_single_model.yaml").read_text())


def unit_of(sel: dict) -> dict:
    return dict(sel["unit"])


def key_of(u: dict) -> tuple:
    return tuple(str(u[k]) for k in inventory.UNIT_KEY)


def unit_dir(processed, sel: dict, setting: str) -> Path:
    u = unit_of(sel)
    return (ns.results_root(processed, sel["part"]) / u["inverse"] / f"mode_{u['mode']}" / u["atlas"] / u["graph"]
            / u["experiment"] / u["variant"] / setting)


def dev_libraries(processed, sel: dict | None = None) -> dict:
    """Numerical environment recorded by the selected unit's development result (one small file; no scan)."""
    sel = sel or selection()
    return json.loads((unit_dir(processed, sel, "xu_dev_cv") / "_identity.json").read_text())["identity"]["code"][
        "libraries"]


def holdout_state(processed, sel: dict) -> str:
    d = unit_dir(processed, sel, "xu_holdout")
    if (d / "final.done.json").exists():
        ok, why = inventory._marker_ok(d)
        return "completed" if ok else f"corrupt ({why})"
    return "partial" if d.exists() else "pending"


# ------------------------------------------------------------------------------------------------ membership, rank
def frozen_entry(processed, sel: dict) -> tuple[dict, dict]:
    """(A2 frozen record, the selected unit's development entry in it); refused unless the exact unit is frozen."""
    if sel["part"] != "A2" or sel["unit"]["exp"] != "exp8a":
        raise SingleRefused("this evaluation is defined for an Exp 8a unit of the A2 part")
    if ns.record()[sel["pin"]] != ns.part_pin(sel["part"]):
        raise SingleRefused("the selection's pin is not the A3 record's A2 pin")
    rec = ns.read_json(ns.record_path(processed, sel["part"]))
    if rec is None:
        raise SingleRefused(f"no frozen record {ns.record_path(processed, sel['part'])}; run the A3 freeze first")
    k = key_of(unit_of(sel))
    if k not in {freeze._key(u) for u in rec["units"]}:
        raise SingleRefused(f"{k} is not in the frozen A3 inventory (A2 record); refused")
    entry = next((e for e in rec["development_cv"] if tuple(str(e[x]) for x in inventory.UNIT_KEY) == k), None)
    if entry is None:
        raise SingleRefused(f"{k} has no development entry in the frozen record")
    if round(float(entry["auc"]), 3) != round(float(sel["development_auc_reported"]), 3):
        raise SingleRefused(f"the frozen development AUC of {k} is {entry['auc']:.4f}, not the reported "
                            f"{sel['development_auc_reported']}; this is not the model that was meant")
    return rec, entry


def gnn_ranking(processed, sel: dict) -> dict:
    """Rank of the selected unit among the frozen GNN units (Exp 2-6 and 8a, both records) by development AUC, computed
    from the frozen records only. Informational: the manifest never claims 'highest' unless this says so."""
    rows = []
    for p in ns.PARTS:
        r = ns.read_json(ns.record_path(processed, p))
        for e in r["development_cv"]:
            if e["experiment"].split("_")[0] in ("exp2", "exp3", "exp4", "exp5", "exp6", "exp8a"):
                rows.append((float(e["auc"]), tuple(str(e[x]) for x in inventory.UNIT_KEY)))
    rows.sort(key=lambda x: (-x[0], x[1]))
    k = key_of(unit_of(sel))
    rank = next((i + 1 for i, (_, kk) in enumerate(rows) if kk == k), None)
    return {"n_frozen_gnn_units": len(rows), "rank_by_development_auc": rank, "is_highest": rank == 1,
            "highest": {"auc": rows[0][0], "unit": list(rows[0][1])} if rows else None}


def verify_light(processed, sel: dict | None = None, restrict=None) -> dict:
    """Master freeze + records (without scanning the other development results) and the selected unit's own development
    result against the frozen record. Raises ``freeze.FreezeInvalid`` / ``SingleRefused``."""
    sel = sel or selection()
    master = freeze.verify(processed, restrict=restrict, check_development=False)["master"]
    rec, entry = frozen_entry(processed, sel)
    d = unit_dir(processed, sel, "xu_dev_cv")
    ok, why = inventory._marker_ok(d)
    if not ok:
        raise SingleRefused(f"the selected unit's development result is not complete ({d}): {why}")
    if (ns.sha256_file(d / "metrics.json") != entry["metrics_sha256"]
            or ns.sha256_file(d / "predictions_participants.csv") != entry["predictions_sha256"]):
        raise freeze.FreezeInvalid("the selected unit's development result changed since the freeze")
    ident = json.loads((d / "_identity.json").read_text())["identity"]
    if (ident.get("code") or {}).get("git_commit") != ns.part_pin(sel["part"]):
        raise SingleRefused("the selected unit's development result was not produced at the A2 pin")
    return {"master": master, "record": rec, "entry": entry, "dev_libraries": ident["code"]["libraries"],
            "dev_dir": str(d), "n_frozen_units": master["n_units"]}


# ------------------------------------------------------------------------------------------------ environment
def check_environment(dev_libraries: dict, python: str, env_probe=None) -> dict:
    """The interpreter must be identical or documented-compatible (a3/environment.py) with the environment of the
    selected unit's development result (read from that one result only)."""
    from .. import numerical_env

    probe = env_probe or numerical_env.interpreter_versions
    libs = probe(python, ("numpy", "scipy", "pandas", "sklearn", "torch"))
    res = environment.interpreter_check(libs, [json.dumps(dev_libraries)])
    res["interpreter"] = libs
    if res["level"] == "incompatible":
        raise SingleRefused(f"the interpreter's numerical environment is incompatible with the development environment "
                            f"of the selected unit: {res['differences']}")
    return res


# ------------------------------------------------------------------------------------------------ selection manifest
def manifest_path(processed, sel: dict) -> Path:
    return ns.root(processed) / "single_model" / f"selection_{sel['selection_id']}.json"


def existing_holdout_dirs(processed) -> list[str]:
    return sorted(str(d.relative_to(processed)) for p in ns.PARTS
                  for d in Path(ns.results_root(processed, p)).glob("*/mode_*/*/*/*/*/xu_holdout"))


def selection_manifest(processed, sel: dict, ver: dict) -> tuple[dict, bool]:
    """The separate selection manifest, written once (an existing identical one is returned; a different one refused)."""
    from .. import provenance

    path = manifest_path(processed, sel)
    master_sha = ns.sha256_file(ns.master_path(processed))
    rank = gnn_ranking(processed, sel)
    own = str(unit_dir(processed, sel, "xu_holdout").relative_to(processed))
    others = [d for d in existing_holdout_dirs(processed) if d != own]
    n_others = ver["n_frozen_units"] - 1
    statement = (
        f"This model was chosen by the author AFTER development cross-validation (development participant AUC "
        f"{ver['entry']['auc']:.4f}; rank {rank['rank_by_development_auc']} of {rank['n_frozen_gnn_units']} frozen GNN "
        f"units by development AUC{', the highest' if rank['is_highest'] else ''}), to reduce computation. "
        f"{n_others} other frozen configurations are not evaluated by this path. The 13-participant D3 holdout was inspected "
        f"in earlier work: this result is descriptive only, not external validation, and it informs no selection. This "
        f"is a separate single-model evaluation, not completion of the full A3 holdout protocol.")
    m = {"amendment": ns.AID, "kind": "single-model holdout selection", "selection_id": sel["selection_id"],
         "authorized_by": sel["authorized_by"], "unit": unit_of(sel), "part": sel["part"],
         "pin": ns.part_pin(sel["part"]), "basis": sel["basis"].strip(), "chosen_after_development_cv": True,
         "development_auc_frozen_record": float(ver["entry"]["auc"]),
         "development_auc_reported": sel["development_auc_reported"], "gnn_ranking": rank,
         "n_frozen_units_total": ver["n_frozen_units"], "n_other_frozen_units_not_evaluated": n_others,
         "other_holdout_results_present_at_selection": others, "holdout_policy": release.HOLDOUT_POLICY,
         "statement": statement, "master_sha256": master_sha,
         "records_sha256": {p: ns.sha256_file(ns.record_path(processed, p)) for p in ns.PARTS},
         "created_at": datetime.now(timezone.utc).isoformat(), "a3_commit": provenance.git_commit(),
         "not_changed": ["A3 master freeze", "release and A2 frozen records", "the 116-unit plan",
                         "development results", "pins"]}
    if path.exists():
        old = ns.read_json(path)
        same = (old["unit"] == m["unit"] and old["master_sha256"] == master_sha and old["pin"] == m["pin"]
                and old["selection_id"] == m["selection_id"])
        if not same:
            raise SingleRefused(f"a different selection manifest exists ({path}); it is never overwritten")
        return old, False
    ns.write_once(path, m)
    return m, True


# ------------------------------------------------------------------------------------------------ run
def run_single(processed, repo, paths_file, python: str, base="/content", restrict=None, runner=None, env_probe=None,
               a3_commit: str | None = None, allow_ephemeral: bool = False, sel: dict | None = None) -> dict:
    """Evaluate the selected unit's holdout once (resumable) at the A2 pin. Nothing else is run."""
    from .. import provenance

    sel = sel or selection()
    ver = verify_light(processed, sel, restrict)
    env_check = check_environment(ver["dev_libraries"], python, env_probe)
    manifest, created = selection_manifest(processed, sel, ver)
    state = holdout_state(processed, sel)
    started = datetime.now(timezone.utc).isoformat()
    u, pin = unit_of(sel), ns.part_pin(sel["part"])
    cmd, code, wt = None, 0, None
    if state == "completed":                               # nothing to recompute (and no Drive-heavy store opening)
        code = 0
    else:
        if state.startswith("corrupt"):
            raise SingleRefused(f"the unit's holdout result is corrupt: {state}; move it to _superseded/ deliberately")
        wt = holdout.ensure_worktree(repo, base, sel["part"], pin)
        if not holdout.imports_worktree(python, wt):
            raise SingleRefused(f"{python} does not import the package of {wt}; the pinned code must run")
        cmd = [python, str(Path(repo) / "scripts" / "a3_single_holdout.py"), "--processed", str(processed), "--exp",
               u["exp"], "--variant", u["variant"], "--inverse", u["inverse"], "--mode", u["mode"], "--atlas",
               u["atlas"], "--graph", u["graph"]] + (["--allow-ephemeral-storage"] if allow_ephemeral else [])
        runner = runner or (lambda c, cwd, env: subprocess.run(c, cwd=cwd, env=env).returncode)
        code = runner(cmd, str(wt), {**os.environ, "PYTHONPATH": str(wt / "src"), "MPLBACKEND": "Agg"})
    freeze.verify(processed, restrict=restrict, check_development=False)   # the call changed no frozen record
    after = holdout_state(processed, sel)
    man = {"amendment": ns.AID, "kind": "single-model holdout run", "selection_id": sel["selection_id"],
           "unit": u, "pin": pin, "state_before": state, "state_after": after, "exit_code": code,
           "command": cmd, "worktree": str(wt) if wt else None, "environment": env_check,
           "selection_manifest_created_now": created, "started": started,
           "finished": datetime.now(timezone.utc).isoformat(), "interpreter": python,
           "a3_commit": a3_commit or provenance.git_commit(),
           "master_sha256": ns.sha256_file(ns.master_path(processed))}
    ns.new_file(ns.root(processed) / "single_model" / "runs" / f"run_{sel['selection_id']}_{ns.stamp()}.json",
                json.dumps(man, indent=2, default=str))
    return man


# ------------------------------------------------------------------------------------------------ report
def report(processed, sel: dict | None = None, out_dir=None, restrict=None, make_figures: bool = True) -> dict:
    """Report of the selected unit's holdout only (needs that one unit completed, not the other frozen units).
    Reads three small folders (selection manifest, the unit's development and holdout results); writes a new folder."""
    sel = sel or selection()
    ver = verify_light(processed, sel, restrict)
    if holdout_state(processed, sel) != "completed":
        raise SingleRefused(f"the unit's holdout is not completed ({holdout_state(processed, sel)}); nothing to report")
    sm = ns.read_json(manifest_path(processed, sel))
    if sm is None:
        raise SingleRefused("no selection manifest; the evaluation was not run through `a3.py single-holdout`")
    d = unit_dir(processed, sel, "xu_holdout")
    m = json.loads((d / "metrics.json").read_text())
    pred = pd.read_csv(d / "predictions_participants.csv")
    out = Path(out_dir or ns.root(processed) / "single_model" / f"report_{ns.stamp()}")
    if out.exists():
        raise FileExistsError(f"{out} exists; reports are never overwritten")
    out.mkdir(parents=True)
    p, dl, bs = m["participant"], m["participant_auc_delong"], m["participant_bootstrap_ci95"]
    row = {"experiment": sel["unit"]["experiment"], "variant": sel["unit"]["variant"], "inverse": sel["unit"]["inverse"],
           "mode": sel["unit"]["mode"], "atlas": sel["unit"]["atlas"], "graph": sel["unit"]["graph"],
           "n_participants": m["n_participants"], "n_ad": m["n_ad"], "n_hc": m["n_hc"], "threshold": m["threshold"],
           "auc": p["roc_auc"], "auc_delong_low": dl["ci_low"], "auc_delong_high": dl["ci_high"],
           "auc_boot_low": bs["roc_auc"][0], "auc_boot_high": bs["roc_auc"][1],
           "balanced_accuracy": p["balanced_accuracy"], "ba_boot_low": bs["balanced_accuracy"][0],
           "ba_boot_high": bs["balanced_accuracy"][1], "sensitivity": p["sensitivity"],
           "sensitivity_wilson_low": m["sensitivity_wilson_ci95"][0], "sensitivity_wilson_high": m["sensitivity_wilson_ci95"][1],
           "specificity": p["specificity"], "specificity_wilson_low": m["specificity_wilson_ci95"][0],
           "specificity_wilson_high": m["specificity_wilson_ci95"][1], "f1": p["f1"], "mcc": p["mcc"],
           "accuracy": p["accuracy"], **m["confusion"], "window_auc_secondary": m["window"]["roc_auc"],
           "development_auc_frozen_record": float(ver["entry"]["auc"])}
    pd.DataFrame([row]).to_csv(out / "single_model_holdout_results.csv", index=False)
    cols = [c for c in ("key", "label", "y", "score", "n_windows", "predicted", "correct") if c in pred]
    pred[cols].to_csv(out / "predictions_participants.csv", index=False)
    c = m["confusion"]
    md = f"""# Single-model D3 holdout (descriptive; selected after development CV)

**{sel['unit']['experiment']} / {sel['unit']['variant']}** · {sel['unit']['inverse']} · mode {sel['unit']['mode']} · {sel['unit']['atlas']} · {sel['unit']['graph']} · A2 pin `{ns.part_pin(sel['part'])[:10]}`

> {sm['statement']}

This report does not require, and says nothing about, the other frozen configurations. {release.HOLDOUT_POLICY}

## Result on the {m['n_participants']} holdout participants ({m['n_ad']} AD, {m['n_hc']} HC); fitted once on the 52 development participants

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | {p['roc_auc']:.3f} | DeLong {dl['ci_low']:.3f}-{dl['ci_high']:.3f}; bootstrap {bs['roc_auc'][0]:.3f}-{bs['roc_auc'][1]:.3f} |
| Balanced accuracy (threshold {m['threshold']}) | {p['balanced_accuracy']:.3f} | bootstrap {bs['balanced_accuracy'][0]:.3f}-{bs['balanced_accuracy'][1]:.3f} |
| Sensitivity (AD) | {p['sensitivity']:.3f} | Wilson {m['sensitivity_wilson_ci95'][0]:.3f}-{m['sensitivity_wilson_ci95'][1]:.3f} |
| Specificity (HC) | {p['specificity']:.3f} | Wilson {m['specificity_wilson_ci95'][0]:.3f}-{m['specificity_wilson_ci95'][1]:.3f} |
| F1 | {p['f1']:.3f} | |
| MCC | {p['mcc']:.3f} | |

Confusion matrix (rows true, columns predicted; AD positive): TN {c['tn']}, FP {c['fp']}, FN {c['fn']}, TP {c['tp']}.

Development CV (frozen record, 52 participants, pooled out-of-fold): AUC {ver['entry']['auc']:.3f}. The two are on different
participants and sample sizes; the difference is descriptive and is not a test. With 13 participants every interval is wide,
and the development AUC of this model is the maximum of many configurations and therefore optimistic.
Participant-level predictions: `predictions_participants.csv`.
"""
    (out / "single_model_holdout_report.md").write_text(md)
    if make_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        if pred["y"].nunique() == 2:
            fpr, tpr, _ = roc_curve(pred["y"], pred["score"])
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
            ax.plot(fpr, tpr, c="#0072B2")
            ax.plot([0, 1], [0, 1], "k:", lw=.6)
            ax.set_xlabel("1 - specificity")
            ax.set_ylabel("sensitivity")
            ax.set_title(f"holdout ROC, n={m['n_participants']} (descriptive)", fontsize=8)
            fig.savefig(out / "roc_holdout.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    files = {f.name: ns.sha256_file(f) for f in sorted(out.iterdir())}
    ns.new_file(out / "manifest.json", json.dumps({
        "amendment": ns.AID, "kind": "single-model holdout report", "selection_id": sel["selection_id"],
        "selection_manifest_sha256": ns.sha256_file(manifest_path(processed, sel)),
        "master_sha256": ns.sha256_file(ns.master_path(processed)),
        "inputs": {"holdout_metrics_sha256": ns.sha256_file(d / "metrics.json"),
                   "holdout_predictions_sha256": ns.sha256_file(d / "predictions_participants.csv"),
                   "development_metrics_sha256": ver["entry"]["metrics_sha256"]},
        "reads": ["selection manifest", "frozen records", "this unit's development and holdout result folders"],
        "other_units_required": False, "outputs_sha256": files}, indent=2))
    return {"dir": str(out), "auc": p["roc_auc"], "n_participants": m["n_participants"]}
