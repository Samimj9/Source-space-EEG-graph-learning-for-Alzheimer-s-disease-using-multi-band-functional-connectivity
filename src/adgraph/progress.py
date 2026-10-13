"""Single-line progress bars (tqdm) for long pipeline stages, safe for Colab "Run all".

One bar per stage, never nested: the bar counts the stage's main items (subjects, outer
folds + permutations, figures) and its suffix shows the current context (subject, dataset,
processing step; variant, setting, repeat/fold, inner-CV trial, epoch; figure id) and the
counts of completed / resumed / failed / skipped / pending items.

* **Checkpoint-aware.** Items recovered from checkpoints are counted in ``initial`` (they
  appear as done immediately and are never recomputed). They do not enter the timing.
* **Honest ETA.** tqdm's own rate-based ETA is not shown. An ETA is shown only from the
  mean duration of items processed *in this session* (``min_items`` of them), and only
  where items are comparable (``eta=True``); otherwise only elapsed time is shown.
* **Low overhead.** Library code calls ``set(...)`` / ``advance()``; both are no-ops when no
  bar is active, and the bar is redrawn at most every ``ADGRAPH_PROGRESS_INTERVAL`` seconds
  (default 2). ``ADGRAPH_PROGRESS=0`` disables all bars.

In a subprocess the bar redraws with carriage returns; ``adgraph.colab.sh`` forwards them
unchanged so the notebook shows one updating line.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

_ACTIVE: list["Reporter"] = []


def fmt_seconds(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


class Reporter:
    def __init__(self, total: int, desc: str, initial: int = 0, unit: str = "item", eta: bool = True,
                 min_items: int = 2, counts: dict | None = None):
        from tqdm import tqdm

        self.enabled = os.environ.get("ADGRAPH_PROGRESS", "1") != "0"
        self.total, self.n, self.eta, self.min_items = total, initial, eta, min_items
        self.fields: dict = {}
        self.counts: dict = dict(counts or {})
        self.durations: list[float] = []
        self._t0 = None
        self.interval = float(os.environ.get("ADGRAPH_PROGRESS_INTERVAL", 2.0))
        self._last_draw = 0.0
        self.bar = tqdm(total=total, initial=initial, desc=desc, unit=unit, file=sys.stderr,
                        mininterval=float(os.environ.get("ADGRAPH_PROGRESS_INTERVAL", 2.0)),
                        bar_format="{desc}: {n_fmt}/{total_fmt} {unit} |{bar:20}| [{elapsed}]{postfix}",
                        disable=not self.enabled, dynamic_ncols=False, ascii=True)
        self._render()

    # -- context and counts
    def set(self, **fields) -> None:
        for k, v in fields.items():
            if v is None:
                self.fields.pop(k, None)
            else:
                self.fields[k] = v
        self._render(refresh=False)

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n
        self._render(refresh=False)

    # -- items
    def start_item(self) -> None:
        self._t0 = time.perf_counter()

    def advance(self, n: int = 1, timed: bool = True) -> None:
        if timed and self._t0 is not None:
            self.durations.append((time.perf_counter() - self._t0) / max(n, 1))
        self._t0 = None
        self.n += n
        self.bar.update(n)
        self._render()

    def eta_seconds(self) -> float | None:
        if not self.eta or len(self.durations) < self.min_items or self.total is None:
            return None
        return (self.total - self.n) * sum(self.durations) / len(self.durations)

    def _render(self, refresh: bool = True) -> None:
        if not self.enabled:
            return
        parts = [f"{v}" if k.startswith("_") else f"{k} {v}" for k, v in self.fields.items()]
        if self.counts:
            parts.append(" ".join(f"{k}:{v}" for k, v in self.counts.items()))
        eta = self.eta_seconds()
        if eta is not None:
            parts.append(f"ETA ~{fmt_seconds(eta)} (from {len(self.durations)} processed)")
        now = time.monotonic()
        # context changes (stage, inner trial, epoch) between item completions are shown too, but the
        # line is redrawn at most every ``interval`` seconds (keeps output and overhead small)
        draw = refresh or now - self._last_draw >= self.interval
        self.bar.set_postfix_str(" | ".join(parts), refresh=draw)
        if draw:
            self._last_draw = now

    def close(self) -> None:
        self._render()
        self.bar.close()


@contextmanager
def reporter(total: int, desc: str, **kw):
    """``with reporter(n, "subjects", initial=k) as r: ...`` — the active bar for ``set``/``advance``."""
    r = Reporter(total, desc, **kw)
    _ACTIVE.append(r)
    try:
        yield r
    finally:
        _ACTIVE.remove(r)
        r.close()


def current() -> Reporter | None:
    return _ACTIVE[-1] if _ACTIVE else None


def set(**fields) -> None:  # noqa: A001 - module-level convenience, mirrors Reporter.set
    r = current()
    if r is not None:
        r.set(**fields)


def advance(n: int = 1, timed: bool = True) -> None:
    r = current()
    if r is not None:
        r.advance(n, timed)


def start_item() -> None:
    r = current()
    if r is not None:
        r.start_item()


def count(key: str, n: int = 1) -> None:
    r = current()
    if r is not None:
        r.count(key, n)


def log_handler_context():
    """Route ``logging`` through tqdm so log lines print above the bar instead of breaking it."""
    from tqdm.contrib.logging import logging_redirect_tqdm

    return logging_redirect_tqdm()
