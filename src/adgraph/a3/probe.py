"""What does this file system support? (``scripts/a3.py probe``; safe: only a temporary ``_probe_*`` folder it creates and
removes itself, inside the A3 namespace.) Tells which write-once method the freeze will use here, in particular whether a
Colab-mounted Google Drive folder accepts hard links; the freeze works either way (``namespace._publish``)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import namespace as ns


def probe(directory) -> dict:
    d = Path(directory) / f"_probe_{ns.stamp()}"
    d.mkdir(parents=True)
    out: dict = {"directory": str(directory)}
    try:
        a = d / "a.txt"
        a.write_text("x")
        try:
            os.link(a, d / "b.txt")
            out["hard_link"] = "supported" if (d / "b.txt").read_text() == "x" else "BROKEN (content differs)"
        except OSError as exc:
            out["hard_link"] = f"unsupported (errno {exc.errno}: {exc.strerror})"
        claim = d / "c.claim"
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            out["exclusive_create"] = "NOT enforced (a second exclusive create succeeded)"
        except FileExistsError:
            out["exclusive_create"] = "enforced"
        (d / "t1").write_text("one")
        (d / "t2").write_text("two")
        os.replace(d / "t2", d / "t1")
        out["rename"] = "supported" if (d / "t1").read_text() == "two" and not (d / "t2").exists() else "BROKEN"
        ns.write_once(d / "rec.json", {"v": 1})
        out["method"] = ns.METHODS[str(d / "rec.json")]
        try:
            ns.write_once(d / "rec.json", {"v": 2})
            out["existing_file_refused"] = False
        except FileExistsError:
            out["existing_file_refused"] = ns.read_json(d / "rec.json") == {"v": 1}
        out["leftovers"] = sorted(p.name for p in d.iterdir() if p.name.startswith(".") or p.suffix == ".claim"
                                  and p.name != "c.claim")
    finally:
        if d.name.startswith("_probe_"):
            shutil.rmtree(d, ignore_errors=True)
    out["ok"] = (out.get("rename") == "supported" and out.get("existing_file_refused") is True
                 and out.get("exclusive_create") == "enforced" and not out.get("leftovers"))
    return out
