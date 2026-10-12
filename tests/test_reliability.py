"""Pre-real-run safeguards: persistent storage, code fingerprint, input content hashes,
pinned environment (synthetic data only)."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from adgraph import provenance, storage
from adgraph.checkpoint import CODE_SCOPES, IncompatibleCheckpoint, code_identity
from adgraph.experiments import runner
from adgraph.io import fingerprint
from adgraph.io.fingerprint import InputHashes
from adgraph.storage import EphemeralStorageError, check_persistent, storage_problem
from adgraph.synthetic import make_synthetic_store
from test_checkpoint import CFG, cohort, fit_counter, lr_variant  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]


# ======================================================================= 1. persistent storage
def test_storage_classification():
    drive = "/content/drive/MyDrive/AD-LEB/processed"
    assert storage_problem(drive, colab=True, mounted=True) is None
    assert "not mounted" in storage_problem(drive, colab=True, mounted=False)     # looks like Drive, is VM disk
    assert "ephemeral" in storage_problem("/content/smoke", colab=True, mounted=True)
    assert "ephemeral" in storage_problem("/content/repo/out", colab=False, mounted=True)
    assert storage_problem("/tmp/out", colab=True, mounted=True) is not None       # Colab VM disk outside Drive
    assert storage_problem("/tmp/out", colab=False, mounted=False) is None         # local machine


def test_check_persistent_raises_or_warns(caplog):
    locs = {"run": "/content/drive/MyDrive/x", "cache": "/content/cache"}
    with pytest.raises(EphemeralStorageError, match="cache = /content/cache"):
        check_persistent(locs, colab=True, mounted=True)
    problems = check_persistent(locs, allow_ephemeral=True, colab=True, mounted=True)
    assert len(problems) == 1 and "lost" in caplog.text
    assert check_persistent({"run": "/content/drive/MyDrive/x"}, colab=True, mounted=True) == []


def test_graph_run_refuses_ephemeral_storage(tmp_path, monkeypatch):
    import adgraph.pipeline as pl

    monkeypatch.setattr(storage, "in_colab", lambda: True)            # tmp_path = Colab VM disk
    with pytest.raises(EphemeralStorageError, match="graph run directory"):
        pl.run({}, CFG, tmp_path / "out", tmp_path / "sd")
    assert not (tmp_path / "out").exists()                            # refused before writing anything
    with pytest.raises(EphemeralStorageError, match="operator cache"):
        pl.run({}, CFG, tmp_path / "out", tmp_path / "sd", operator_cache="/content/ops",
               allow_ephemeral=False)


def test_experiment_refuses_ephemeral_storage_for_real_runs(tmp_path, monkeypatch):
    real = make_synthetic_store({"D1": (6, 6), "D2": (6, 6)}, windows_per_subject=(3, 3))
    real.manifest = {**real.manifest, "synthetic": False}            # stands in for a real graph run
    with pytest.raises(EphemeralStorageError):
        runner.run_variant(real, lr_variant(1), "within_D1", "/content/experiments")
    monkeypatch.setattr(storage, "in_colab", lambda: True)
    with pytest.raises(EphemeralStorageError):
        runner.run_variant(real, lr_variant(1), "within_D1", tmp_path)
    assert not any(tmp_path.iterdir())
    synth = make_synthetic_store({"D1": (6, 6), "D2": (6, 6)}, windows_per_subject=(3, 3))
    v = lr_variant(1)
    v["hparams"] = {}
    runner.run_variant(synth, v, "within_D1", tmp_path)                # synthetic framework tests are exempt


def test_build_graphs_cli_refuses_ephemeral_out_root(tmp_path):
    (tmp_path / "paths.yaml").write_text(f"drive_root: {tmp_path}\nprocessed_dir: {tmp_path}/p\n"
                                         f"cache_dir: {tmp_path}/c\n")
    r = subprocess.run([sys.executable, "scripts/build_graphs.py", "--paths", str(tmp_path / "paths.yaml"),
                        "--out-root", "/content/smoke"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "EphemeralStorageError" in r.stderr and "/content/smoke" in r.stderr


# ======================================================================= 2. code fingerprint
def test_source_fingerprint_tracks_edits_within_scope(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "models").mkdir(parents=True)
    (pkg / "visualization").mkdir()
    (pkg / "models" / "gnn.py").write_text("H = 32\n")
    (pkg / "visualization" / "plot.py").write_text("DPI = 300\n")
    monkeypatch.setattr(provenance, "PACKAGE_DIR", pkg)

    def fp():
        provenance.source_fingerprint.cache_clear()
        return provenance.source_fingerprint(("models",))

    a = fp()
    (pkg / "visualization" / "plot.py").write_text("DPI = 600\n")     # outside the scope
    assert fp() == a
    (pkg / "models" / "gnn.py").write_text("H = 64\n")                # uncommitted edit inside the scope
    assert fp() != a
    provenance.source_fingerprint.cache_clear()


def test_code_identity_contents():
    ident = code_identity("experiment")
    assert set(ident) == {"git_commit", "source_sha256", "libraries"}
    assert len(ident["source_sha256"]) == 64 and "torch" in ident["libraries"] and "mne" not in ident["libraries"]
    assert "visualization" not in " ".join(CODE_SCOPES["experiment"])   # figures never invalidate results


@pytest.mark.parametrize("change", ["source", "git_commit", "library"])
def test_code_change_invalidates_graph_units_without_bumping_code_versions(cohort, monkeypatch, change):
    pl, paths, counts, tmp = cohort
    out = tmp / "out"
    run_dir = pl.run(paths, CFG, out, tmp / "sd")
    old_id = pd.read_json(run_dir / "manifest.json", typ="series")["graph_run_id"]
    assert counts["build"] == 1 and len(counts["process"]) == 4
    if change == "source":
        real = provenance.source_fingerprint
        monkeypatch.setattr(provenance, "source_fingerprint", lambda parts: "edited" + real(parts)[6:])
    elif change == "git_commit":
        monkeypatch.setattr(provenance, "git_commit", lambda: "0" * 40)
    else:
        real_v = provenance.versions
        monkeypatch.setattr(provenance, "versions", lambda pk=("numpy",): {**real_v(pk), "numpy": "9.9.9"})
    st = pl.status(paths, CFG, out)["subjects"]
    assert st["state"].eq("incompatible").all() and st["changed"].str.contains("code").all()
    counts["process"].clear()
    pl.run(paths, CFG, out, tmp / "sd")                               # CODE_VERSIONS untouched
    assert sorted(counts["process"]) == ["D1-1", "D1-2", "D2-17", "D2-2"]   # all recomputed automatically
    assert counts["build"] == 2                                       # operator code changed too
    assert pd.read_json(run_dir / "manifest.json", typ="series")["graph_run_id"] != old_id
    assert (run_dir / "_superseded").exists()                         # old units kept, not overwritten


def test_code_change_is_never_silently_reused_in_experiments(tmp_path, monkeypatch, fit_counter):
    store = make_synthetic_store({"D1": (6, 6), "D2": (6, 6)}, windows_per_subject=(3, 3))
    v = lr_variant(1)
    first = runner.run_variant(store, v, "within_D1", tmp_path)
    real = provenance.source_fingerprint
    monkeypatch.setattr(provenance, "source_fingerprint", lambda parts: "edited" + real(parts)[6:])
    st = runner.experiment_status(store, v, "within_D1", tmp_path)
    assert st["state"] == "incompatible" and st["changed"] == "code"
    with pytest.raises(IncompatibleCheckpoint, match="code"):
        runner.run_variant(store, v, "within_D1", tmp_path)
    fits = fit_counter["fits"]
    again = runner.run_variant(store, v, "within_D1", tmp_path, force=True)    # explicit recomputation
    assert fit_counter["fits"] > fits and (tmp_path / "_superseded").exists()
    assert again["subject_level"]["all"]["roc_auc"]["mean"] == first["subject_level"]["all"]["roc_auc"]["mean"]


# ======================================================================= 3. input content hashes
def test_input_hash_cache(tmp_path, monkeypatch):
    f = tmp_path / "rec.edf"
    f.write_bytes(b"abcdef" * 1000)
    reads = []
    real = fingerprint.file_sha256
    monkeypatch.setattr(fingerprint, "file_sha256", lambda p: (reads.append(p), real(p))[1])
    cache = tmp_path / "cache" / "input_sha256.json"
    h = InputHashes(cache).sha256(f)
    assert h == real(f) and len(reads) == 1
    assert InputHashes(cache).sha256(f) == h and len(reads) == 1      # resume: served from the cache
    f.write_bytes(b"abcdeX" * 1000)                                   # content (and mtime) changed
    assert InputHashes(cache).sha256(f) != h and len(reads) == 2
    changed = InputHashes(cache).sha256(f)
    st = f.stat()
    f.write_bytes(b"abcdeY" * 1000)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))                  # same size + mtime: documented limit,
    assert InputHashes(cache).sha256(f) == changed and len(reads) == 2  # cache cannot see it ...
    assert InputHashes(cache, rehash=True).sha256(f) == real(f) and len(reads) == 3   # ... --rehash-inputs does
    (cache).write_text("{trunc")                                      # torn cache file -> re-hash, no crash
    assert InputHashes(cache).sha256(f) == real(f)


def _flip_last_byte(path):
    data = bytearray(Path(path).read_bytes())
    data[-1] ^= 0x01
    Path(path).write_bytes(bytes(data))


def test_input_content_change_recomputes_only_that_subject(cohort):
    pl, paths, counts, tmp = cohort
    out = tmp / "out"
    run_dir = pl.run(paths, CFG, out, tmp / "sd")
    man = pd.read_json(run_dir / "manifest.json", typ="series")
    assert all(len(r["sha256"]) == 64 for r in man["inputs"])
    assert set(man["auxiliary_inputs"]) == {"participants_D1", "participants_D2"}
    _flip_last_byte(Path(paths["dataset2"]["eeg_dir"]) / "17_952019_1.edf")   # same size, new content
    st = pl.status(paths, CFG, out)["subjects"].set_index("key")
    assert st.loc["D2-17", "state"] == "incompatible" and st.loc["D2-17", "changed"] == "input"
    assert st.drop("D2-17")["state"].eq("completed").all()
    counts["process"].clear()
    pl.run(paths, CFG, out, tmp / "sd")
    assert counts["process"] == ["D2-17"]


def test_moved_data_is_not_recomputed_but_metadata_change_is(cohort):
    pl, paths, counts, tmp = cohort
    out = tmp / "out"
    run_dir = pl.run(paths, CFG, out, tmp / "sd")
    old_id = pd.read_json(run_dir / "manifest.json", typ="series")["graph_run_id"]
    moved = tmp / "remounted"
    os.rename(tmp / "d1", moved)                                      # e.g. a different Drive mount path
    paths = {**paths, "dataset1": {**paths["dataset1"], "eeg_dir": str(moved)}}
    counts["process"].clear()
    pl.run(paths, CFG, out, tmp / "sd")
    assert counts["process"] == []                                    # identity is content, not path

    pd.DataFrame({"Id": ["2_842019", "17_952019"], "Age": [71, 80], "Gender": ["M", "F"], "Group": ["HC", "AD"],
                  "MMSE (/30)": [29, 11]}).to_excel(paths["dataset2"]["participants"], index=False)
    st = pl.status(paths, CFG, out)["subjects"].set_index("key")
    assert st.loc["D2-17", "changed"] == "metadata"
    pl.run(paths, CFG, out, tmp / "sd")
    assert counts["process"] == ["D2-17"]                             # only the corrected subject
    subj = pd.read_csv(run_dir / "subjects.csv").set_index("key")
    assert subj.loc["D2-17", "age"] == 80
    assert pd.read_json(run_dir / "manifest.json", typ="series")["graph_run_id"] != old_id


def test_input_hashes_are_cached_across_resumes(cohort, monkeypatch):
    pl, paths, counts, tmp = cohort
    reads = []
    real = fingerprint.file_sha256
    monkeypatch.setattr(fingerprint, "file_sha256", lambda p: (reads.append(Path(p).name), real(p))[1])
    pl.run(paths, CFG, tmp / "out", tmp / "sd", input_hash_cache=tmp / "cache" / "input_sha256.json")
    assert len(reads) == 6                                            # 4 EDF + 2 participant sheets
    reads.clear()
    pl.run(paths, CFG, tmp / "out", tmp / "sd", input_hash_cache=tmp / "cache" / "input_sha256.json")
    assert reads == []
    pl.run(paths, CFG, tmp / "out", tmp / "sd", input_hash_cache=tmp / "cache" / "input_sha256.json",
           rehash_inputs=True)
    assert len(reads) == 6


# ======================================================================= 4. pinned environment
def test_test_environment_matches_pins():
    pins = provenance.pinned_requirements()
    assert {"mne", "edfio"} <= set(pins)
    assert provenance.check_pinned_environment() == []
    import edfio  # noqa: F401  (hard requirement of the suite, never skipped)


def test_pin_mismatch_is_reported(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy>=1.0\nmne==0.0.1  # wrong\nno-such-package-xyz==1.0\n")
    problems = provenance.check_pinned_environment(req)
    assert len(problems) == 2 and any("not installed" in p for p in problems)
    with pytest.raises(RuntimeError, match="requirements.txt"):
        provenance.verify_environment(req)


def test_dependency_skip_fails_the_suite(tmp_path):
    (tmp_path / "test_a.py").write_text(
        "import pytest\n\ndef test_dep():\n    pytest.importorskip('no_such_package_xyz')\n\n"
        "def test_ok():\n    pass\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "tests") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "dependency_guard", "-p", "no:cacheprovider",
           "--rootdir", str(tmp_path), str(tmp_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert r.returncode == 1 and "dependency-related skips" in r.stdout
    (tmp_path / "test_a.py").write_text("import pytest\n\ndef test_x():\n    pytest.skip('not applicable')\n")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert r.returncode == 0                                          # ordinary skips are unaffected
