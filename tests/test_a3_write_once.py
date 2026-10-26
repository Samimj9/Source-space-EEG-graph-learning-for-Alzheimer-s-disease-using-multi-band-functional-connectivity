"""Write-once publishing of the A3 frozen records and reports (``a3.namespace``), with and without hard links.

Hard links may be unavailable on a Colab-mounted Google Drive; the fallback (claim file + rename) must keep the same
guarantees: an existing target is refused and left untouched, an interrupted write never leaves a partial target, a stale
claim does not block a rerun, a fresh claim does."""

import errno
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from adgraph.a3 import namespace as ns
from adgraph.a3 import probe


def no_link():
    return mock.patch.object(ns.os, "link", side_effect=OSError(errno.EPERM, "Operation not permitted"))


@pytest.fixture(params=["link", "claim+rename"])
def mode(request):
    """Run a test once with hard links and once on a file system that refuses them."""
    if request.param == "link":
        yield "link"
    else:
        with no_link():
            yield "claim+rename"


def leftovers(d: Path):
    return sorted(p.name for p in d.iterdir() if p.name.startswith(".") or p.name.endswith(".claim"))


def test_creates_the_file_once_with_the_expected_method(tmp_path, mode):
    p = tmp_path / "d" / "rec.json"
    assert ns.write_once(p, {"a": 1}) == p
    assert json.loads(p.read_text()) == {"a": 1} and ns.METHODS[str(p)] == mode
    assert leftovers(p.parent) == []                                       # no temporary file, no claim


def test_an_existing_target_is_refused_and_untouched(tmp_path, mode):
    p = tmp_path / "rec.json"
    ns.write_once(p, {"v": 1})
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        ns.write_once(p, {"v": 2})
    with pytest.raises(FileExistsError):
        ns.new_file(p, "other")
    assert p.read_bytes() == before and leftovers(tmp_path) == []
    q = tmp_path / "hand_made.txt"
    q.write_text("someone else's file")
    with pytest.raises(FileExistsError):
        ns.new_file(q, "ours")
    assert q.read_text() == "someone else's file"


def test_a_target_appearing_during_the_write_is_never_overwritten(tmp_path):
    p = tmp_path / "rec.json"

    def link_that_loses_a_race(src, dst):
        Path(dst).write_text("the winner")
        raise FileExistsError(errno.EEXIST, "File exists", str(dst))
    with mock.patch.object(ns.os, "link", side_effect=link_that_loses_a_race), pytest.raises(FileExistsError):
        ns.write_once(p, {"v": 1})
    assert p.read_text() == "the winner" and leftovers(tmp_path) == []

    q = tmp_path / "rec2.json"

    def refuses_but_target_exists(src, dst):
        Path(dst).write_text("the winner")
        raise OSError(errno.EPERM, "Operation not permitted")
    with mock.patch.object(ns.os, "link", side_effect=refuses_but_target_exists), pytest.raises(FileExistsError):
        ns.write_once(q, {"v": 1})                                         # fallback re-checks under its claim
    assert q.read_text() == "the winner" and leftovers(tmp_path) == []


@pytest.mark.parametrize("fail_in", ["link", "replace"])
def test_an_interrupted_write_leaves_no_target_and_resumes(tmp_path, fail_in):
    p = tmp_path / "rec.json"
    if fail_in == "link":
        ctx = mock.patch.object(ns.os, "link", side_effect=KeyboardInterrupt)
    else:                                                                    # dies between claim and rename
        ctx = mock.patch.multiple(ns.os, link=mock.Mock(side_effect=OSError(errno.EPERM, "x")),
                                  replace=mock.Mock(side_effect=KeyboardInterrupt))
    with ctx, pytest.raises(KeyboardInterrupt):
        ns.write_once(p, {"v": 1})
    assert not p.exists()                                                    # never a partial record
    assert not any(n.endswith(".claim") for n in leftovers(tmp_path))        # the claim is released on the way out
    ns.write_once(p, {"v": 1})                                               # resume: a plain rerun works
    assert json.loads(p.read_text()) == {"v": 1} and leftovers(tmp_path) == []


def test_a_crash_that_leaves_a_claim_blocks_only_while_it_is_fresh(tmp_path):
    p = tmp_path / "rec.json"
    claim = tmp_path / "rec.json.claim"
    claim.write_text(json.dumps({"pid": 1, "created": 1000.0}))             # as if the process was killed (SIGKILL)
    with no_link(), mock.patch.object(ns.time, "time", return_value=1000.0 + 5):
        with pytest.raises(ns.WriteInProgress):
            ns.write_once(p, {"v": 1})
    assert not p.exists() and claim.exists()                                 # a fresh claim is respected, untouched
    with no_link(), mock.patch.object(ns.time, "time", return_value=1000.0 + ns.CLAIM_TTL_S + 1):
        ns.write_once(p, {"v": 1})                                           # stale: taken over, write completes
    assert json.loads(p.read_text()) == {"v": 1} and not claim.exists()
    assert [n for n in leftovers(tmp_path) if n.startswith(".")] == []


def test_a_corrupt_claim_file_is_treated_by_age(tmp_path):
    p = tmp_path / "rec.json"
    (tmp_path / "rec.json.claim").write_text("not json")
    old = 1.0
    os.utime(tmp_path / "rec.json.claim", (old, old))
    with no_link():
        ns.write_once(p, {"v": 1})
    assert p.exists()


def test_the_readback_detects_a_target_that_does_not_hold_the_content(tmp_path):
    p = tmp_path / "rec.json"
    real_replace = os.replace

    def corrupting_replace(src, dst):
        real_replace(src, dst)
        Path(dst).write_text("truncated by the file system")
    with no_link(), mock.patch.object(ns.os, "replace", corrupting_replace), pytest.raises(ns.WriteConflict):
        ns.write_once(p, {"v": 1})


def test_new_file_and_json_records_have_the_same_guarantees(tmp_path, mode):
    ns.new_file(tmp_path / "r" / "report.md", "# hi\n")
    assert (tmp_path / "r" / "report.md").read_text() == "# hi\n" and ns.METHODS[str(tmp_path / "r/report.md")] == mode
    with pytest.raises(FileExistsError):
        ns.new_file(tmp_path / "r" / "report.md", "# changed\n")


def test_other_os_errors_are_not_swallowed(tmp_path):
    with mock.patch.object(ns.os, "link", side_effect=OSError(errno.ENOSPC, "No space left on device")):
        with pytest.raises(OSError) as e:
            ns.write_once(tmp_path / "rec.json", {"v": 1})
    assert e.value.errno == errno.ENOSPC and not (tmp_path / "rec.json").exists()   # no silent fallback on real failures


def test_probe_reports_the_method_and_cleans_up(tmp_path, mode):
    r = probe.probe(tmp_path)
    assert r["ok"] and r["method"] == mode and r["existing_file_refused"] and r["exclusive_create"] == "enforced"
    assert (r["hard_link"] == "supported") == (mode == "link")
    assert list(tmp_path.iterdir()) == []                                    # only its own temporary folder, removed
