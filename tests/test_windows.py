import numpy as np

from adgraph.preprocessing.windows import reject_windows, window_starts


def test_window_starts_respect_trim_and_step():
    starts = window_starts(n_samples=250 * 120, sfreq=250, window_s=40, step_s=20, edge_trim_s=10)
    assert starts.tolist() == [2500, 7500, 12500, 17500]  # 10-50, 30-70, 50-90, 70-110 s
    assert starts[-1] + 40 * 250 <= 250 * 120 - 10 * 250


def test_window_starts_short_recording():
    assert len(window_starts(250 * 50, 250, 40, 20, 10)) == 0


def test_reject_windows_flags_artifact_window():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((30, 10_000))
    data[:10, 2000:2100] += 50  # one third of channels hit by a large artifact
    starts = np.array([0, 2000, 4000, 6000])
    keep, thr, ptp = reject_windows(data, starts, 1000, mad_k=5, max_bad_channel_fraction=0.15)
    assert keep.tolist() == [True, False, True, True]
    assert ptp.shape == (4, 30) and np.isfinite(thr)
