"""Versioned amendments of the release that must not touch its completed results (methodology log entry 60).

An amendment (``configs/final/amendments.yaml``, e.g. A2: Exp 8 with a cortical-surface anatomical layer) runs its
own experiment configs in its OWN namespace with its OWN code pin:

    <processed>/FINAL_v1/amendments/<dir>/CODE_COMMIT.txt     the amendment's code commit (scripts/colab_setup.py
                                                              --amendment A2); FINAL_v1/CODE_COMMIT.txt is not read
                                                              for checkout and never written
    <processed>/FINAL_v1/amendments/<dir>/manifest.json       the amendment record, base release pin, amendment
                                                              commit, environment check (written once)
    <processed>/FINAL_v1/amendments/<dir>/results/...         results (same layout as FINAL_v1/results)
    <processed>/FINAL_v1/amendments/<dir>/reports/...         tables of this namespace only
    <processed>/FINAL_v1/amendments/<dir>/decisions/final_analysis.json
                                                              the frozen holdout units of this namespace; written
                                                              only by a later, separately recorded final-analysis
                                                              amendment (entry 60, plan A3), never by this code

Shared with the release, READ ONLY: the verified stores (``stores.open_store(lock=False)``: the release lock is
checked, never written), the participant QC reports (used only if already complete for this store and QC code; an
amendment never computes QC in the release namespace), the recorded atlas and graph-construction decisions. Results
of the release are never read as checkpoints: an amendment experiment has its own name (e.g. exp8a_...) and results
root, so a failed or partial unit of the original experiment can never be resumed or reused.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .. import release
from ..checkpoint import atomic_write_json, check_unit
from .config import CONFIG_ROOT, load_experiment

NAMESPACES = {"A2": "A2_exp8_surface"}
EXPERIMENTS = {"A2": {"exp8a": "amendments/exp8a_xu_dual_gcn_surface.yaml"}}


class AmendmentError(RuntimeError):
    pass


def record(aid: str) -> dict:
    from .decisions import amendments

    a = amendments().get(aid)
    if not a or aid not in NAMESPACES:
        raise AmendmentError(f"unknown amendment {aid!r}")
    return a


def root(processed, aid: str) -> Path:
    return release.root(processed) / "amendments" / NAMESPACES[aid]


def pin_file(aid: str) -> str:
    """Path of the amendment's code pin relative to the output root (``scripts/colab_setup.py --amendment``)."""
    return f"{release.ROOT}/amendments/{NAMESPACES[aid]}/CODE_COMMIT.txt"


def results_dir(processed, aid: str) -> Path:
    return root(processed, aid) / "results"


def reports_dir(processed, aid: str) -> Path:
    return root(processed, aid) / "reports"


def final_path(processed, aid: str) -> Path:
    """The namespace's frozen final-analysis record (same format as ``decisions.final_path``)."""
    return root(processed, aid) / "decisions" / "final_analysis.json"


def experiment_configs(aid: str) -> dict[str, Path]:
    """{experiment id (e.g. exp8a): config path} of the amendment."""
    record(aid)
    return {k: Path(CONFIG_ROOT) / v for k, v in EXPERIMENTS[aid].items()}


def reference_identity(processed, experiment: str = "exp7_earm_gnn") -> dict | None:
    """Code identity of a completed release unit (default: an Exp 7 unit), to compare library versions."""
    for f in sorted(release.results_dir(processed).glob(f"*/mode_*/*/*/{experiment}/*/*/_identity.json")):
        ident = json.loads(f.read_text()).get("identity", {})
        if ident.get("code"):
            return ident
    return None


def environment_check(processed, aid: str) -> dict:
    """Base release pin, this code's commit, and library versions vs a completed release unit (Exp 7)."""
    from ..provenance import git_commit, versions

    a = record(aid)
    pin = release.root(processed) / "CODE_COMMIT.txt"
    base = pin.read_text().split()[0] if pin.exists() and pin.read_text().split() else None
    ref = reference_identity(processed) or {}
    ref_lib = (ref.get("code") or {}).get("libraries") or {}
    now = versions(tuple(k for k in ref_lib if k != "python")) if ref_lib else versions()
    diff = {k: {"release": ref_lib.get(k), "now": now.get(k)} for k in ref_lib if ref_lib.get(k) != now.get(k)}
    return {"amendment": aid, "base_release_pin": base, "expected_base_release_pin": a.get("base_release_pin"),
            "base_pin_ok": base == a.get("base_release_pin"), "amendment_commit": git_commit(),
            "reference_unit_commit": (ref.get("code") or {}).get("git_commit"), "libraries_release": ref_lib,
            "libraries_now": now, "library_differences": diff, "environment_ok": bool(ref_lib) and not diff}


def initialize(processed, aid: str, accept_environment_difference: bool = False) -> dict:
    """Write the amendment manifest once (same base pin and environment required; a library difference must be
    accepted explicitly and is recorded). Returns the manifest."""
    chk = environment_check(processed, aid)
    if not chk["base_pin_ok"]:
        raise AmendmentError(f"the release pin is {chk['base_release_pin']}, the amendment was defined on "
                             f"{chk['expected_base_release_pin']}")
    if not chk["environment_ok"] and not accept_environment_difference:
        raise AmendmentError(f"numerical environment differs from the completed release units "
                             f"({chk['library_differences'] or 'no completed Exp 7 unit found'}); rerun with "
                             "--accept-environment-difference to record and accept it")
    path = root(processed, aid) / "manifest.json"
    if path.exists():
        man = json.loads(path.read_text())
        if man["amendment_commit"] != chk["amendment_commit"]:
            raise AmendmentError(f"the amendment namespace belongs to commit {man['amendment_commit']}; this code is "
                                 f"{chk['amendment_commit']} (its own pin, {pin_file(aid)}, decides which code runs)")
        return man
    man = {"amendment": aid, "record": record(aid), **chk, "environment_difference_accepted":
           bool(chk["library_differences"]) and accept_environment_difference,
           "created_at": datetime.now(timezone.utc).isoformat(), "release": release.release_manifest()}
    atomic_write_json(path, man)
    return man


def require_initialized(processed, aid: str) -> dict:
    from ..provenance import git_commit

    path = root(processed, aid) / "manifest.json"
    if not path.exists():
        raise AmendmentError(f"amendment {aid} is not initialized (scripts/final.py amendment-check --amendment {aid})")
    man = json.loads(path.read_text())
    if man["amendment_commit"] != git_commit():
        raise AmendmentError(f"amendment {aid} belongs to commit {man['amendment_commit']}, this code is "
                             f"{git_commit()}")
    return man


def open_context(processed, inverse: str, mode: str, atlas: str):
    """(store, verification record, QC report) for an amendment run, without writing anything in the release
    namespace: the store is verified against the release lock (not locked again) and the QC report must already be
    complete for this store and the current QC code (an amendment never computes QC)."""
    from ..qc.report import development_keys, identity, qc_dir, require_qc
    from ..stores import open_store, store_inverse

    store, rec = open_store(processed, inverse, mode, atlas, lock=False)
    inv = store_inverse(inverse)
    out = qc_dir(processed, inv, mode, atlas)
    if check_unit(out, "qc", identity(store, development_keys(store)))[0] != "complete":
        raise AmendmentError(f"no complete participant QC report for {inv}/{mode}/{atlas} with this QC code ({out}); "
                             "an amendment does not compute QC in the release namespace")
    return store, rec, require_qc(store, processed, inv, mode, atlas)


def run(processed, aid: str, exp: str, inverse: str, mode: str, atlas: str, graph: str | None, variants=None,
        allow_ephemeral: bool = False):
    """Development CV of an amendment experiment in its namespace (resumable)."""
    from .plan import run_experiment

    require_initialized(processed, aid)
    cfg = experiment_configs(aid)[exp]
    if load_experiment(cfg).get("amendment") != aid:
        raise AmendmentError(f"{cfg} does not belong to amendment {aid}")
    store, rec, qc = open_context(processed, inverse, mode, atlas)
    return run_experiment(cfg, processed, inverse, mode, atlas, graph, variants, ["xu_dev_cv"],
                          results_root=results_dir(processed, aid), store=store, store_record=rec, qc_report=qc,
                          allow_ephemeral=allow_ephemeral)


def holdout(processed, aid: str, exp: str, allow_ephemeral: bool = False) -> list:
    """Holdout of the namespace's FROZEN units only (descriptive, release.HOLDOUT_POLICY), at the commit that froze
    them. Refused unless ``final_path`` exists (written by a recorded final-analysis amendment, never here), lists
    units of ``exp`` and names this code's commit; ``decisions.require_frozen`` is applied with that record."""
    from ..provenance import git_commit
    from .plan import run_experiment

    import pandas as pd

    require_initialized(processed, aid)
    f = final_path(processed, aid)
    if not f.exists():
        raise AmendmentError(f"nothing is frozen in amendment {aid} ({f}); its holdout is evaluated only after a "
                             "recorded final-analysis amendment has frozen it (methodology log entry 60, plan A3)")
    d = json.loads(f.read_text())
    if d.get("git_commit") != git_commit():
        raise AmendmentError(f"{f} was frozen at commit {d.get('git_commit')}, this code is {git_commit()}")
    units = [u for u in d.get("units", []) if u["exp"] == exp]
    if not units:
        raise AmendmentError(f"{exp} has no frozen unit in amendment {aid} ({f})")
    cfg = experiment_configs(aid)[exp]
    out = []
    for (inv, mode, atlas, graph), g in pd.DataFrame(units).groupby(["inverse", "mode", "atlas", "graph"], sort=False):
        store, rec, qc = open_context(processed, inv, mode, atlas)
        out.append(run_experiment(cfg, processed, inv, mode, atlas, graph, sorted(g["variant"]), ["xu_holdout"],
                                  results_root=results_dir(processed, aid), store=store, store_record=rec,
                                  qc_report=qc, allow_ephemeral=allow_ephemeral, frozen_path=f))
    return out


def namespace_check(processed, aid: str, repo_head: str | None = None) -> dict:
    """Separation of the amendment namespace from the release (raises ``AmendmentError`` on any violation): the
    release pin is still the amendment's base pin, the amendment pin is its own file, names the amendment commit
    (and the checked-out code) and differs from the release pin, and the manifest belongs to this amendment."""
    from ..provenance import git_commit

    a = record(aid)
    rel_file = release.root(processed) / "CODE_COMMIT.txt"
    am_file = Path(processed) / pin_file(aid)

    def first(f: Path):
        return next((ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")),
                    None) if f.exists() else None

    rel, am = first(rel_file), first(am_file)
    man_file = root(processed, aid) / "manifest.json"
    man = json.loads(man_file.read_text()) if man_file.exists() else {}
    head = repo_head or git_commit()
    checks = {
        "release pin unchanged (= base_release_pin)": rel == a["base_release_pin"],
        "amendment pin is a separate file": am_file.resolve() != rel_file.resolve(),
        "amendment pin recorded": am is not None,
        "amendment pin differs from the release pin": am is not None and am != rel,
        "checked-out code is the amendment pin": am == head,
        "manifest belongs to this amendment": man.get("amendment") == aid,
        "manifest commit is the amendment pin": man.get("amendment_commit") == am,
        "manifest base pin is the release pin": man.get("base_release_pin") == rel,
    }
    out = {"release_pin": rel, "release_pin_file": str(rel_file), "amendment_pin": am,
           "amendment_pin_file": str(am_file), "head": head, "checks": checks}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise AmendmentError(f"amendment {aid} namespace check failed: {bad} ({out})")
    return out
