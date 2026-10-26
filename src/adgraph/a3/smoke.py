"""Synthetic end-to-end check of the A3 machinery (never real data): dev CV at the two (patched) pins, the dual freeze,
the pinned holdout paths, resume, the consolidated and holdout reports and the D2 manifest, on a few variants.

The two code pins are the real commit ids (strings only): ``provenance.git_commit`` is patched so each part's units
carry the commit of its pin, exactly as they do on Drive. Real stores, results and records are never involved: the root
is a temporary directory, and the freeze is restricted to the smoke variants (the record is marked ``synthetic``).
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd

from .. import provenance, release, storage
from ..checkpoint import atomic_write_json
from ..experiments import amendment, decisions
from ..experiments.config import experiment_configs
from ..experiments.plan import run_experiment
from . import audit, d2, freeze, holdout, inventory, report
from . import namespace as ns

SMOKE = {("exp0", "logreg__psd"), ("exp1", "logreg__psd"), ("exp2", "plv__delta"), ("exp2", "ciplv__delta"),
         ("exp8a", "plv__psd"), ("exp8a", "plv__psd_de_strength"),
         # the pinned CLI selects variants by whole name tokens, so naming plv__psd_de_strength also runs its four
         # ablation / alternative variants; the smoke freezes exactly what runs (the real plan freezes all ten)
         ("exp8a", "plv__psd_de_strength__abl_functional_both"), ("exp8a", "plv__psd_de_strength__abl_anatomical_both"),
         ("exp8a", "plv__psd_de_strength__alt_bilateral_lobes"), ("exp8a", "plv__psd_de_strength__abl_shared_weights")}


def restrict(u: dict) -> bool:
    return (u["exp"], u["variant"]) in SMOKE


@contextlib.contextmanager
def at_commit(box: dict, commit: str):
    box["commit"] = commit
    yield
    box["commit"] = None


_ORIGINAL_RECORD = ns.record


def approved_record() -> dict:
    """The A3 record as if approved; only ever patched in by the synthetic smoke / tests, never by the CLI."""
    return {**_ORIGINAL_RECORD(), "approved_by": "synthetic smoke (no real approval)",
            "status": "approved (synthetic)"}


class Smoke:
    """A temporary synthetic release root with development CV done for the smoke variants."""

    def __init__(self, root=None, epochs: int = 2):
        from ..models import trainer
        from ..qc.report import require_qc
        from ..stores import open_store
        from ..synthetic import write_release_stores

        storage.in_colab = lambda: False
        self.root = Path(root or tempfile.mkdtemp(prefix="a3_smoke_"))
        self.rec = ns.record()
        self.rp, self.ap = self.rec["release_pin"], self.rec["a2_pin"]
        self.box = {"commit": None}
        self._stack = contextlib.ExitStack()
        real = provenance.git_commit
        self._stack.enter_context(mock.patch.object(provenance, "git_commit",
                                                    lambda: self.box["commit"] or real()))
        init = trainer.GNNModel.__init__
        self._stack.enter_context(mock.patch.object(
            trainer.GNNModel, "__init__", lambda s, model, training, *a, **k: init(s, model, {**training, "epochs":
                                                                                              epochs}, *a, **k)))
        for a in ("dk68", "lobes12"):
            write_release_stores(self.root, n_windows=3, psd_effect=0.8, conn_effect=0.8, band_direction=0.5, atlas=a)
        for inv in ("lcmv", "wmne"):
            for mode in ("B", "A"):
                store, _ = open_store(self.root, inv, mode, "lobes12")
                require_qc(store, self.root, inv, mode, "lobes12")
        (self.root / release.PIN_FILE).write_text(self.rp + "\n")
        ns.pin_file(self.root, "A2").parent.mkdir(parents=True, exist_ok=True)
        ns.pin_file(self.root, "A2").write_text(self.ap + "\n")
        decisions.record_atlas(self.root, "lobes12", "synthetic smoke: the recorded atlas of amendment A3")
        atomic_write_json(decisions.graph_path(self.root), {"selected": "dense", "atlas": "lobes12", "reason": "kept"})
        ref = release.results_dir(self.root) / "lcmv/mode_B/lobes12/dense/exp7_earm_gnn/v/xu_dev_cv/_identity.json"
        ref.parent.mkdir(parents=True)
        ref.write_text(json.dumps({"identity": {"code": {"git_commit": self.rp, "libraries": provenance.versions()}}}))
        self.cfgs = experiment_configs()

    def close(self):
        self._stack.close()

    def development(self):
        """Development CV of the smoke variants: release part at the release pin, Exp 8a at the A2 pin."""
        with at_commit(self.box, self.rp):
            for inv in ("lcmv", "wmne"):
                for mode in ("B", "A"):
                    run_experiment(self.cfgs["exp1"], self.root, inv, mode, "lobes12", None, ["logreg__psd"],
                                   ["xu_dev_cv"])
                    run_experiment(self.cfgs["exp2"], self.root, inv, mode, "lobes12", "dense",
                                   ["plv__delta", "ciplv__delta"] if mode == "B" else ["plv__delta"], ["xu_dev_cv"])
            for mode in ("B", "A"):
                run_experiment(self.cfgs["exp0"], self.root, "lcmv", mode, "dk68", None, ["logreg__psd"],
                               ["xu_dev_cv"])
        with at_commit(self.box, self.ap):
            amendment.initialize(self.root, "A2")
            for inv in ("lcmv", "wmne"):
                amendment.run(self.root, "A2", "exp8a", inv, "B", "lobes12", "dense",
                              ["plv__psd", "plv__psd_de_strength"])

    def freeze(self, write: bool = True):
        with mock.patch.object(ns, "record", approved_record):
            return freeze.freeze(self.root, write=write, restrict=restrict, require_clean=False,
                                 strict_holdout_yaml=True)

    def holdout(self):
        """What the pinned CLIs do: ``final.py holdout --exp expN`` at the release pin, ``--amendment A2`` at A2's."""
        out = []
        with at_commit(self.box, self.rp):
            f = json.loads(decisions.final_path(self.root).read_text())
            for exp in sorted({u["exp"] for u in f["units"]}):
                units = [u for u in f["units"] if u["exp"] == exp]
                for (inv, mode, atlas, graph), g in pd.DataFrame(units).groupby(["inverse", "mode", "atlas", "graph"]):
                    out.append(run_experiment(self.cfgs[exp], self.root, "lcmv" if inv == "sensor" else inv, mode, atlas,
                                              None if graph == "none" else graph, sorted(g["variant"]),
                                              ["xu_holdout"]))
        with at_commit(self.box, self.ap):
            out += amendment.holdout(self.root, "A2", "exp8a")
        return out


def run(root=None, epochs: int = 2, link: bool = True) -> dict:
    """``link=False`` simulates a file system that refuses hard links (the freeze must then use claim + rename)."""
    import errno

    s = Smoke(root, epochs)
    if not link:
        s._stack.enter_context(mock.patch.object(ns.os, "link", side_effect=OSError(errno.EPERM, "not permitted")))
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))
    try:
        s.development()
        dry = s.freeze(write=False)
        chk("dry run passes and writes nothing", dry["ok"] and not ns.master_path(s.root).exists(),
            [c for c in dry["checks"] if not c["ok"]])
        res = s.freeze(write=True)
        chk("freeze written (two records + master)", len(res["written"]) == 3, res["written"])
        want = "link" if link else "claim+rename"
        chk(f"write-once method is {want}", set(res["write_methods"].values()) == {want}, res["write_methods"])
        again = s.freeze(write=True)
        chk("rerunning the freeze writes nothing", again["written"] == [], again["written"])
        freeze.verify(s.root, restrict=restrict)
        ts = s.holdout()
        chk("holdout units completed", all((t["state"] == "completed").all() for t in ts))
        st = holdout.status(s.root, restrict)
        chk("holdout status complete", (st["holdout_state"] == "completed").all(), st["holdout_state"].value_counts())
        before = {str(p): ns.sha256_file(p) for p in (s.root / "FINAL_v1").rglob("xu_holdout/*.csv")}
        s.holdout()
        chk("a second holdout call changes nothing (resume)",
            before == {str(p): ns.sha256_file(p) for p in (s.root / "FINAL_v1").rglob("xu_holdout/*.csv")})
        freeze.verify(s.root, restrict=restrict)
        cfg = {**ns.stats_config(), "n_bootstrap": 40, "n_permutation": 40}
        r = report.development_report(s.root, cfg=cfg)
        chk("development report", r["n_completed"] > 0 and not r["figures"] == [], r)
        h = report.holdout_report(s.root, restrict=restrict)
        chk("holdout report", h["completed_units"] == h["planned_units"], h)
        m = d2.write(s.root, "lobes12", "dense", cfg)
        chk("D2 manifest", m.exists(), m)
        a = audit.audit(s.root, write=False)
        chk("audit runs", "checks" in a)
    finally:
        s.close()
    ok = all(c[1] for c in checks)
    summary = "\n".join(f"[{'PASS' if o else 'FAIL'}] {n}" + (f": {d}" if d and not o else "") for n, o, d in checks)
    return {"ok": ok, "summary": summary + f"\nsynthetic root: {s.root}", "root": s.root, "checks": checks}
