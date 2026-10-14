"""Download of OpenNeuro ds004504 (Dataset 3) to persistent storage: metadata + AD/HC derivatives only.

Files come from OpenNeuro's public S3 bucket over HTTPS. The download is resumable and verified: a file is
fetched to a temporary name and renamed only after its size matches the bucket listing and (for
single-part uploads, whose ETag is the MD5) its MD5 matches; existing complete files are skipped.
``download_manifest.json`` records, per file, the URL, size, ETag, MD5 / SHA-256 and the time, plus the
dataset release (``dataset_description.json`` DOI and the first line of ``CHANGES``). FTD recordings are
never downloaded.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from .bids import EXPECTED, check_cohort, load_participants_tsv

BASE_URL = "https://s3.amazonaws.com/openneuro.org"
DATASET_ID = "ds004504"
METADATA = ("participants.tsv", "participants.json", "dataset_description.json", "README", "CHANGES")


def _get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def list_objects(prefix: str, base_url: str = BASE_URL) -> dict[str, dict]:
    """S3 listing under ``prefix`` (key -> size, etag)."""
    out, token = {}, None
    while True:
        q = {"list-type": "2", "prefix": prefix, **({"continuation-token": token} if token else {})}
        root = ElementTree.fromstring(_get(f"{base_url}?{urllib.parse.urlencode(q)}"))
        ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        tag = (lambda t: f"s:{t}") if ns else (lambda t: t)
        for c in root.findall(tag("Contents"), ns):
            out[c.find(tag("Key"), ns).text] = {"size": int(c.find(tag("Size"), ns).text),
                                                "etag": c.find(tag("ETag"), ns).text.strip('"')}
        trunc = root.find(tag("IsTruncated"), ns)
        if trunc is None or trunc.text != "true":
            return out
        token = root.find(tag("NextContinuationToken"), ns).text


def _hashes(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def fetch(url: str, dest: Path, size: int | None = None, etag: str | None = None) -> dict:
    """Download ``url`` to ``dest`` unless a verified copy exists; returns the file record."""
    plain_md5 = bool(etag) and re.fullmatch(r"[0-9a-f]{32}", etag or "") is not None
    if dest.exists() and (size is None or dest.stat().st_size == size):
        md5, sha = _hashes(dest)
        if not plain_md5 or md5 == etag:
            return {"url": url, "path": str(dest), "size": dest.stat().st_size, "etag": etag, "md5": md5,
                    "sha256": sha, "status": "present"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(_get(url))
    md5, sha = _hashes(tmp)
    got = tmp.stat().st_size
    if size is not None and got != size:
        tmp.unlink()
        raise IOError(f"{url}: size {got} != listed {size}")
    if plain_md5 and md5 != etag:
        tmp.unlink()
        raise IOError(f"{url}: MD5 {md5} != ETag {etag}")
    tmp.replace(dest)
    return {"url": url, "path": str(dest), "size": dest.stat().st_size, "etag": etag, "md5": md5, "sha256": sha,
            "status": "downloaded", "downloaded_at": datetime.now(timezone.utc).isoformat()}


def download_ds004504(root: str | Path, base_url: str = BASE_URL, progress=None) -> dict:
    """Metadata, then the preprocessed derivative of every AD and HC participant (never FTD)."""
    root = Path(root)
    listing = list_objects(f"{DATASET_ID}/", base_url)
    files = []
    for name in METADATA:
        key = f"{DATASET_ID}/{name}"
        if key in listing:
            files.append(fetch(f"{base_url}/{key}", root / name, **listing[key]))
    kept, excluded = load_participants_tsv(root / "participants.tsv")
    check_cohort(kept)
    for i, sid in enumerate(kept["subject_id"]):
        sub = f"sub-{sid:03d}"
        key = f"{DATASET_ID}/derivatives/{sub}/eeg/{sub}_task-eyesclosed_eeg.set"
        if key not in listing:
            raise FileNotFoundError(f"{key} is not in the OpenNeuro bucket listing")
        if progress:
            progress(i, len(kept), sub)
        files.append(fetch(f"{base_url}/{key}", root / "derivatives" / sub / "eeg" / Path(key).name, **listing[key]))
    desc = json.loads((root / "dataset_description.json").read_text()) if (root / "dataset_description.json").exists() else {}
    changes = (root / "CHANGES").read_text().splitlines() if (root / "CHANGES").exists() else []
    manifest = {"dataset": DATASET_ID, "doi": desc.get("DatasetDOI"), "release": changes[0].strip() if changes else None,
                "license": desc.get("License"), "source": base_url, "created_at": datetime.now(timezone.utc).isoformat(),
                "cohort": {"used": kept["label"].value_counts().to_dict(), "expected": EXPECTED,
                           "excluded": excluded.to_dict("records")},
                "files": files}
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest
