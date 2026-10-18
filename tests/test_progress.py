"""Progress reporting: checkpoint-aware, honest ETA, one bar at a time, live in Colab (synthetic only)."""

import sys
from pathlib import Path

import pytest
import yaml

from adgraph import progress
from adgraph.colab import sh
from test_checkpoint import CFG, SimulatedCrash, cohort, fit_counter  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def bars(monkeypatch):
    """Record every progress bar: its totals, the fields it showed and its final counts."""
    made, depth = [], {"max": 0}

    class Spy(progress.Reporter):
        def __init__(self, total, desc, **kw):
            super().__init__(total, desc, **kw)
            self.desc, self.initial, self.seen = desc, kw.get("initial", 0), set()
            made.append(self)

        def set(self, **fields):
            self.seen |= {k for k, v in fields.items() if v is not None}
            super().set(**fields)

    real_reporter = progress.reporter

    def reporter(total, desc, **kw):
        cm = real_reporter(total, desc, **kw)
        depth["max"] = max(depth["max"], len(progress._ACTIVE) + 1)
        return cm

    monkeypatch.setattr(progress, "Reporter", Spy)
    monkeypatch.setattr(progress, "reporter", reporter)
    return made, depth


# ------------------------------------------------------------------ the reporter itself
def test_eta_only_from_items_processed_in_this_session():
    r = progress.Reporter(10, "t", initial=4)
    assert r.eta_seconds() is None
    r.advance(timed=False)                                  # a resumed item: not timed
    assert r.eta_seconds() is None
    r.start_item(); r.advance()
    assert r.eta_seconds() is None                          # one processed item is not enough
    r.durations = [2.0, 4.0]
    assert r.eta_seconds() == pytest.approx((10 - 6) * 3.0)
    r.close()
    no_eta = progress.Reporter(10, "t", eta=False)
    no_eta.durations = [1.0, 1.0, 1.0]
    assert no_eta.eta_seconds() is None
    no_eta.close()


def test_library_calls_are_noops_without_a_bar():
    progress.set(stage="x")
    progress.advance()
    progress.count("done")
    assert progress.current() is None


def test_sh_forwards_carriage_returns_live(capsys):
    sh(f"{sys.executable} -c \"import sys,time; [sys.stderr.write(f'\\rstep {{i}}') or sys.stderr.flush() "
       f"for i in range(3)]; print()\"")
    out = capsys.readouterr().out
    assert "\rstep 0\rstep 1\rstep 2" in out                 # redraws in place, not one line per update


# ------------------------------------------------------------------ graph construction
def test_graph_progress_is_checkpoint_aware(cohort, bars):
    pl, paths, counts, tmp = cohort
    made, depth = bars
    pl.run(paths, CFG, tmp / "out", tmp / "sd")
    subj = next(b for b in made if b.desc == "subjects")
    assert (subj.total, subj.initial, subj.n) == (4, 0, 4)
    assert subj.counts == {"done": 4, "resumed": 0, "failed": 0, "pending": 0}
    assert {"subject", "dataset", "stage"} <= subj.seen
    assert {"input SHA-256", "source operator", "assembly", "saving cohort"} <= {b.desc for b in made}
    assert depth["max"] == 1                                 # never nested

    made.clear()
    counts["process"].clear()
    pl.run(paths, CFG, tmp / "out", tmp / "sd")             # "Colab reconnect"
    subj = next(b for b in made if b.desc == "subjects")
    assert (subj.initial, subj.n) == (4, 4)                  # shown as done from the start
    assert subj.counts == {"done": 0, "resumed": 4, "failed": 0, "pending": 0}
    assert counts["process"] == []                           # and not recomputed
    assert subj.eta_seconds() is None                        # nothing processed -> no ETA invented


# ------------------------------------------------------------------ experiments
def _tiny_config(tmp_path, base, **extra):
    cfg = {"extends": str(REPO / "configs" / "experiments" / base),
           "evaluation": {"n_repeats": 2, "k": 3, "inner_k": 2, "n_bootstrap": 20,
                          "permutation": {"n": 2, "n_repeats": 1}}, **extra}
    p = tmp_path / base
    p.write_text(yaml.safe_dump(cfg))
    return str(p)






# ------------------------------------------------------------------ figures
