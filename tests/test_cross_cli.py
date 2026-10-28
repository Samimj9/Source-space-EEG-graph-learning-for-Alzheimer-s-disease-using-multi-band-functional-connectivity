"""The cross-dataset CLI authorization command (scripts/cross_dataset.py): ``authorize --authorization-stage
{build_d2,train,stability}`` records the author's authorization once; ``--stage {primary,stability}`` is the analysis
stage of run / status / evaluate / report and is never read as an authorization. Synthetic paths only; nothing is
built, trained, frozen or evaluated."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from adgraph.crossdataset import namespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cross_dataset.py"


@pytest.fixture
def paths(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    f = tmp_path / "paths.yaml"
    f.write_text(yaml.safe_dump({"drive_root": str(tmp_path), "processed_dir": str(processed),
                                 "cache_dir": str(tmp_path / "cache")}))
    return f, processed


def cli(paths_file, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args, "--paths", str(paths_file)], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)


def records(processed) -> dict:
    f = namespace.root(processed) / "authorizations.json"
    return json.loads(f.read_text()) if f.exists() else {}


@pytest.mark.parametrize("stage", ["build_d2", "train", "stability"])
def test_the_documented_authorization_commands_are_accepted(paths, stage):
    f, processed = paths
    r = cli(f, "authorize", "--authorization-stage", stage, "--by", "Author (written authorization)")
    assert r.returncode == 0, r.stderr
    rec = records(processed)
    assert list(rec) == [stage] and rec[stage]["by"] == "Author (written authorization)"
    assert rec[stage]["what"] == namespace.STAGES[stage] and rec[stage]["git_commit"]


def test_build_authorization_does_not_authorize_training(paths):
    f, processed = paths
    assert cli(f, "authorize", "--authorization-stage", "build_d2", "--by", "A").returncode == 0
    assert namespace.require_authorized(processed, "build_d2")["by"] == "A"
    for other in ("train", "stability"):
        with pytest.raises(namespace.NotAuthorized, match="--authorization-stage"):
            namespace.require_authorized(processed, other)


def test_authorizations_are_write_once(paths):
    f, processed = paths
    assert cli(f, "authorize", "--authorization-stage", "train", "--by", "First").returncode == 0
    first = records(processed)["train"]
    assert cli(f, "authorize", "--authorization-stage", "train", "--by", "Second").returncode == 0
    assert records(processed)["train"] == first                     # never rewritten


@pytest.mark.parametrize("args", [
    ["authorize", "--stage", "build_d2", "--by", "A"],              # the old, broken form: refused by argparse
    ["authorize", "--stage", "train", "--by", "A"],
    ["authorize", "--stage", "stability", "--by", "A"],             # an analysis stage, not an authorization
    ["authorize", "--by", "A"],                                     # no authorization stage
    ["authorize", "--authorization-stage", "primary", "--by", "A"],
    ["authorize", "--authorization-stage", "evaluate", "--by", "A"],
    ["status", "--stage", "build_d2"],                              # authorization stages are not analysis stages
    ["run", "--stage", "train"],
    ["run", "--authorization-stage", "train"],                      # only the authorize command takes it
    ["status", "--authorization-stage", "build_d2"],
])
def test_invalid_stages_are_refused_and_nothing_is_recorded(paths, args):
    f, processed = paths
    r = cli(f, *args)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "usage" in r.stderr and records(processed) == {}
    assert not namespace.root(processed).exists() or not (namespace.root(processed) / "authorizations.json").exists()


def test_an_authorization_names_who_gave_it(paths):
    f, processed = paths
    r = cli(f, "authorize", "--authorization-stage", "build_d2", "--by", "  ")
    assert r.returncode != 0 and "names who gave it" in r.stderr and records(processed) == {}


def test_the_analysis_stage_still_parses_for_status(paths):
    f, processed = paths
    for stage in ("primary", "stability"):
        r = cli(f, "status", "--stage", stage)
        assert r.returncode == 0, r.stderr                         # markers only; nothing is run or written
    assert records(processed) == {}


def test_the_restored_environment_installs_what_the_store_build_imports():
    """The restored Exp 7 interpreter (numerical_env) installs only requirements.txt: every package the D2 store
    build needs beyond the pinned numerics must be listed there (nibabel: mne.read_labels_from_annot)."""
    import re

    from adgraph import numerical_env

    reqs = {re.split(r"[<>=\s#]", ln.strip(), maxsplit=1)[0].lower()
            for ln in (ROOT / "requirements.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    assert {"mne", "nibabel", "edfio", "pymatreader", "openpyxl"} <= reqs
    cmds = numerical_env.restore_commands({"python": "3.13.5", "numpy": "2.0.0"}, "/tmp/x", ROOT)
    assert any("-r" in c and c[c.index("-r") + 1].endswith("requirements.txt") for c in cmds)
