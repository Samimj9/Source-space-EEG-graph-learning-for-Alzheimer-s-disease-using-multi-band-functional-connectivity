"""Helpers that let the Colab notebooks run end to end without manual steps.

* ``find_data_root``: locate the shared ``AD-LEB`` folder on the mounted Drive.
* ``choose_output_root``: ``AD-LEB/processed`` if writable, else a folder in My Drive.
* ``write_paths_yaml``: create ``configs/paths.yaml`` for the scripts.
* ``check_inputs``: the expected recordings, sheets and labels are all there.
* ``sh``: run a command, stream its output into the notebook, raise on failure
  (``!cmd`` in a notebook neither shows subprocess output reliably nor stops "Run all").

Nothing here processes EEG.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

import yaml

DRIVE = Path("/content/drive")
DATA_FOLDER = "AD-LEB"
FALLBACK_OUTPUT = "AD-LEB-processed"     # in My Drive, only if AD-LEB/processed is read-only
# Snapshot of AD-LEB (docs/data_inventory.md): resting-state recordings and labels per dataset.
EXPECTED = {"D1": {"recordings": 20, "AD": 10, "HC": 10}, "D2": {"recordings": 23, "AD": 10, "HC": 13}}


def sh(cmd: str, cwd: str | Path | None = None) -> None:
    """Run a shell command, show its output live, raise if it fails (stops "Run all").

    Output is forwarded in raw chunks as it arrives (not line by line), so carriage-return
    progress bars redraw in place in the notebook instead of printing a new line per update."""
    import codecs
    import os

    print(f"$ {cmd}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = os.read(p.stdout.fileno(), 8192)
        if not chunk:
            break
        sys.stdout.write(decoder.decode(chunk))
        sys.stdout.flush()
    sys.stdout.write(decoder.decode(b"", final=True))
    sys.stdout.flush()
    if p.wait():
        raise RuntimeError(f"command failed (exit {p.returncode}): {cmd}")


def _is_data_root(p: Path) -> bool:
    return (p / "EEGs").is_dir() and (p / "Info participants").is_dir()


def find_data_root(drive: str | Path = DRIVE) -> Path:
    """The AD-LEB folder: My Drive shortcut first, then shortcut targets and shared drives."""
    drive = Path(drive)
    candidates = [drive / "MyDrive" / DATA_FOLDER, *sorted((drive / "MyDrive").glob(f"*/{DATA_FOLDER}")),
                  *sorted(drive.glob(f".shortcut-targets-by-id/*/{DATA_FOLDER}")),
                  *sorted(drive.glob(f"Shareddrives/*/{DATA_FOLDER}"))]
    for c in candidates:
        if _is_data_root(c):
            return c
    raise FileNotFoundError(
        f"could not find the shared '{DATA_FOLDER}' folder (with EEGs/ and 'Info participants/') under {drive}. "
        "In Google Drive: 'Shared with me' -> right-click AD-LEB -> Organize -> Add shortcut -> My Drive, "
        "then run this notebook again.")


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".write_test_{uuid.uuid4().hex[:8]}"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def choose_output_root(data_root: str | Path, drive: str | Path = DRIVE) -> Path:
    """``AD-LEB/processed`` when it can be written to; otherwise ``My Drive/AD-LEB-processed``.

    The choice depends only on folder permissions, so every notebook and every session
    picks the same location."""
    processed = Path(data_root) / "processed"
    if _writable(processed):
        return processed
    fallback = Path(drive) / "MyDrive" / FALLBACK_OUTPUT
    if not _writable(fallback):
        raise PermissionError(f"neither {processed} nor {fallback} is writable")
    print(f"note: {processed} is read-only for this account; outputs go to {fallback}")
    return fallback


def write_paths_yaml(repo: str | Path, data_root: str | Path, output_root: str | Path) -> Path:
    """configs/paths.yaml (gitignored) pointing at the located data and output folders.

    The electrode layout is located with ``find_layout_file`` (AD-LEB root first, then the EEG
    folders, also accepting a single ``electrodes_positions_32 (1).loc`` copy); if none is found
    the conventional root path is written and the header check reports it as missing."""
    from .io.layout import find_layout_file

    try:
        layout = str(find_layout_file(data_root))
    except (FileNotFoundError, ValueError) as e:
        print(f"warning: {e}")
        layout = "{drive_root}/electrodes_positions_32.loc"
    paths = {
        "drive_root": str(data_root),
        "dataset1": {"eeg_dir": "{drive_root}/EEGs/Dataset 1 (2017)",
                     "participants": "{drive_root}/Info participants/Info Participants_2017.xlsx"},
        "dataset2": {"eeg_dir": "{drive_root}/EEGs/Dataset 2 (2019)/RestingState",
                     "participants": "{drive_root}/Info participants/Info Participants_2019.xlsx"},
        "electrode_layout": layout,
        "processed_dir": str(output_root),
        "cache_dir": str(Path(output_root) / "cache"),
    }
    out = Path(repo) / "configs" / "paths.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# written by adgraph.colab.write_paths_yaml (not committed)\n"
                   + yaml.safe_dump(paths, sort_keys=False, allow_unicode=True))
    return out


def check_inputs(paths: dict, expected: dict = EXPECTED):
    """Raise unless every expected recording, sheet and label is present and matched.

    Returns the per-dataset summary table. Reads file names and the participant sheets only."""
    import pandas as pd

    from .pipeline import build_registry

    problems = []
    for key in ("participants",):
        for ds in ("dataset1", "dataset2"):
            if not Path(paths[ds][key]).is_file():
                problems.append(f"missing {paths[ds][key]}")
    for ds in ("dataset1", "dataset2"):
        if not Path(paths[ds]["eeg_dir"]).is_dir():
            problems.append(f"missing folder {paths[ds]['eeg_dir']}")
    if problems:
        raise FileNotFoundError("; ".join(problems))
    registry, _, no_meta, no_rec = build_registry(paths)
    rows = []
    for ds, exp in expected.items():
        sub = registry[registry["dataset"] == ds]
        counts = sub["label"].value_counts().to_dict()
        rows.append({"dataset": ds, "recordings": len(sub), "AD": counts.get("AD", 0), "HC": counts.get("HC", 0),
                     "expected": f"{exp['recordings']} ({exp['AD']} AD / {exp['HC']} HC)"})
        if len(sub) != exp["recordings"] or counts.get("AD", 0) != exp["AD"] or counts.get("HC", 0) != exp["HC"]:
            problems.append(f"{ds}: found {len(sub)} recordings ({counts}), expected {rows[-1]['expected']}")
    if no_meta:
        problems.append(f"recordings without a participant row: {no_meta}")
    if no_rec:
        problems.append(f"participant rows without a recording: {no_rec}")
    table = pd.DataFrame(rows)
    if problems:
        raise RuntimeError("input check failed (the data differ from docs/data_inventory.md):\n  "
                           + "\n  ".join(problems) + f"\n{table.to_string(index=False)}")
    return table


def run_tests_once(output_root: str | Path, repo: str | Path = ".") -> None:
    """Run the synthetic test suite unless it already passed for this exact code and environment.

    The pass is remembered in ``<output_root>/cache/tests_passed/<commit>_<env>.txt``, so a
    resumed session does not spend the test time again. Dependency skips fail the suite."""
    import hashlib
    import json

    from .provenance import check_pinned_environment, git_commit, git_dirty, versions

    env = hashlib.sha256(json.dumps(versions(("numpy", "scipy", "pandas", "mne", "sklearn", "torch", "edfio")),
                                    sort_keys=True).encode()).hexdigest()[:12]
    marker = Path(output_root) / "cache" / "tests_passed" / f"{git_commit()}_{env}.txt"
    if marker.exists() and not git_dirty():
        print(f"tests already passed for this code and environment ({marker.name}); skipped")
        return
    problems = check_pinned_environment()
    if problems:
        raise RuntimeError("environment does not match requirements.txt: " + "; ".join(problems))
    sh(f"{sys.executable} -m pytest -q -rs -p no:cacheprovider", cwd=repo)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("passed\n")


def has_completed_work(output_root: str | Path) -> bool:
    """True if any graph-subject or experiment checkpoint has been completed under ``output_root``
    (a pinned commit then protects results and must not move silently)."""
    root = Path(output_root)
    if any(root.glob("graphs_v*/subjects/*/subject.done.json")):
        return True
    return any(root.glob("experiments/*/*/*/units/*.done.json")) or any(
        root.glob("experiments/*/*/*/final.done.json"))


def resolve_code_pin(pin_file: str | Path, head: str, update: bool = False,
                     has_work: bool = True) -> tuple[str, bool]:
    """Commit every session must use, and whether it differs from the cloned ``head``.

    The first call records ``head`` in ``pin_file``; later calls return the recorded commit,
    so all sessions and notebooks run the same code. ``update`` moves the pin to ``head``
    (previous pins are kept as comments). A pin with no completed work behind it
    (``has_work`` False, e.g. left by a setup-only session) protects nothing and is moved to
    ``head`` automatically, so the first real run records the current commit."""
    from datetime import datetime, timezone

    pin_file = Path(pin_file)
    text = pin_file.read_text() if pin_file.exists() else ""
    pinned = next((ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")), None)
    if pinned and not update and (has_work or pinned == head):
        return pinned, pinned != head
    history = [ln if ln.startswith("#") else f"# previous pin: {ln}" for ln in text.splitlines() if ln.strip()]
    pin_file.parent.mkdir(parents=True, exist_ok=True)
    pin_file.write_text("\n".join([head, f"# pinned {datetime.now(timezone.utc).isoformat()}", *history]) + "\n")
    return head, False
