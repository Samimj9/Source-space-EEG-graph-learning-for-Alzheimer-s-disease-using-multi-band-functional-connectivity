"""The single version system of the final framework.

One release name, one results root and ONE code pin replace the per-suite pins of the earlier work (D1/D2, D3,
LX, WX, ES200, R1; those pin files stay on Drive untouched and belong to the old branch):

    <processed>/FINAL_v1/CODE_COMMIT.txt     the commit every final notebook checks out (moved only deliberately,
                                             UPDATE_CODE = True in the setup cell)
    <processed>/FINAL_v1/stores.lock.json    graph_run_id of every store the release has verified and used
    <processed>/FINAL_v1/qc/...              participant QC reports
    <processed>/FINAL_v1/results/...         experiment outputs (results_dir)
    <processed>/FINAL_v1/reports/...         combined tables

Every result manifest records ``release_manifest()``: the release, the code commit, the source fingerprint and the
library versions. A number in the paper is reproducible from the release name, the commit and a committed config.
"""

from __future__ import annotations

from pathlib import Path

RELEASE = "final-1.0"
ROOT = "FINAL_v1"
PIN_FILE = f"{ROOT}/CODE_COMMIT.txt"
LOCK_FILE = f"{ROOT}/stores.lock.json"
MODES = {"B": "all accepted windows (primary analysis)", "A": "10 windows per participant (sensitivity analysis)"}
INVERSES = {"lcmv": "LCMV beamformer", "wmne": "depth-weighted minimum norm (wMNE)",
            "sensor": "sensor space (no inverse; Exp 0)"}
PRIMARY = {"mode": "B", "inverse": "lcmv", "connectivity": "plv", "atlas": "dk68", "graph": "dense"}
HOLDOUT_POLICY = ("The 13-participant holdout was inspected during earlier experiments (LX / WX suites), so it is "
                  "no longer an untouched test set. In this release it is evaluated once per final configuration and "
                  "reported descriptively; it is never used to select architectures, graph constructions, k, atlas "
                  "resolution, QC thresholds or any other setting (selection reads development CV results only).")


def root(processed: str | Path) -> Path:
    return Path(processed) / ROOT


def results_dir(processed: str | Path) -> Path:
    return root(processed) / "results"


def release_manifest() -> dict:
    from . import provenance
    from .checkpoint import CODE_SCOPES

    return {"release": RELEASE, "git_commit": provenance.git_commit(), "git_dirty": provenance.git_dirty(),
            "source_sha256": provenance.source_fingerprint(CODE_SCOPES["experiment"]),
            "versions": provenance.versions()}
