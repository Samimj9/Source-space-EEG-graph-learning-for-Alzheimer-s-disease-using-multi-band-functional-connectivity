import numpy as np

import pytest

from adgraph.config import load_yaml
from adgraph.preprocessing.windows import expected_n_windows, reject_windows, window_samples, window_starts

CFG = load_yaml("configs/pipeline.yaml")


def test_primary_segmentation_is_frozen_at_20s_50pct():
    w = CFG["windowing"]
    assert (w["window_s"], w["step_s"]) == (20.0, 10.0)
    assert window_samples(CFG["preprocessing"]["sfreq"], w["window_s"], w["step_s"]) == (5000, 2500)


def test_window_starts_respect_trim_and_step():
    starts = window_starts(n_samples=250 * 80, sfreq=250, window_s=20, step_s=10, edge_trim_s=10)
    assert starts.tolist() == [2500, 5000, 7500, 10000, 12500]  # 10-30, 20-40, ..., 50-70 s
    assert (starts[1:] - starts[:-1] == 2500).all()             # 50 % overlap
    assert starts[-1] + 5000 <= 250 * 80 - 10 * 250


@pytest.mark.parametrize("duration", [39.9, 40, 49.99, 50, 80, 300, 617.3])
def test_expected_n_windows_matches_starts(duration):
    n_samples = int(round(duration * 250))
    starts = window_starts(n_samples, 250, 20, 10, 10)
    assert len(starts) == expected_n_windows(n_samples / 250, 20, 10, 10)


def test_expected_n_windows_examples():
    assert expected_n_windows(300, 20, 10, 10) == 27       # 10 s trimmed at both ends
    assert expected_n_windows(300, 20, 10, 0) == 29        # without trimming
    assert expected_n_windows(39.9, 20, 10, 10) == 0


def test_window_samples_rejects_fractional():
    with pytest.raises(ValueError):
        window_samples(250, 20.001, 10)


def test_window_starts_short_recording():
    assert len(window_starts(250 * 39, 250, 20, 10, 10)) == 0


def test_reject_windows_flags_artifact_window():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((30, 10_000))
    data[:10, 2000:2100] += 50  # one third of channels hit by a large artifact
    starts = np.array([0, 2000, 4000, 6000])
    keep, thr, ptp = reject_windows(data, starts, 1000, mad_k=5, max_bad_channel_fraction=0.15)
    assert keep.tolist() == [True, False, True, True]
    assert ptp.shape == (4, 30) and np.isfinite(thr)
