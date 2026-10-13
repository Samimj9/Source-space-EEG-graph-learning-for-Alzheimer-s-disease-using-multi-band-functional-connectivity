"""Process memory monitoring for QC logs (Linux /proc; values are None elsewhere).

* ``rss_mb``: current resident memory of this process.
* ``peak_rss_mb``: peak resident memory (VmHWM). ``reset_peak`` resets it (Linux >= 4.0,
  ``/proc/self/clear_refs``), so a peak can be attributed to one subject.
* ``available_mb`` / ``total_mb``: system memory (what an OOM kill depends on).

``MemoryTracker`` wraps one unit of work and returns the numbers recorded in the QC tables.
Nothing here changes any computation.
"""

from __future__ import annotations

import gc
import resource
import sys
from pathlib import Path


def _status_kb(field: str) -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return float(line.split()[1])
    except OSError:
        pass
    return None


def _meminfo_kb(field: str) -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(field + ":"):
                return float(line.split()[1])
    except OSError:
        pass
    return None


def rss_mb() -> float | None:
    kb = _status_kb("VmRSS")
    return None if kb is None else kb / 1024


def peak_rss_mb() -> float | None:
    kb = _status_kb("VmHWM")
    if kb is None:                                   # not Linux: lifetime peak from getrusage
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    return kb / 1024


def reset_peak() -> bool:
    """Reset the peak-RSS counter to the current RSS; False if not supported."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


def available_mb() -> float | None:
    kb = _meminfo_kb("MemAvailable")
    return None if kb is None else kb / 1024


def total_mb() -> float | None:
    kb = _meminfo_kb("MemTotal")
    return None if kb is None else kb / 1024


class MemoryTracker:
    """``with MemoryTracker() as m: ...`` then ``m.record()`` -> dict of memory QC fields (MB).

    ``mem_peak_mb`` is the process peak during the block if the counter could be reset
    (``mem_peak_scope='unit'``), otherwise the process lifetime peak (``'process'``)."""

    def __enter__(self):
        gc.collect()
        self.scope = "unit" if reset_peak() else "process"
        self.before, self.avail_before = rss_mb(), available_mb()
        return self

    def __exit__(self, *exc):
        self.peak, self.avail_after = peak_rss_mb(), available_mb()
        gc.collect()
        self.after = rss_mb()
        return False

    def record(self) -> dict:
        r = lambda x: None if x is None else round(float(x), 1)  # noqa: E731
        return {"mem_rss_before_mb": r(self.before), "mem_peak_mb": r(self.peak), "mem_rss_after_mb": r(self.after),
                "mem_available_before_mb": r(self.avail_before), "mem_available_after_mb": r(self.avail_after),
                "mem_total_mb": r(total_mb()), "mem_peak_scope": self.scope}
