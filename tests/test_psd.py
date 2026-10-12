import numpy as np
import pytest

from adgraph.config import load_yaml
from adgraph.features.psd import band_powers, compute_psd, window_band_powers

CFG = load_yaml("configs/pipeline.yaml")
BANDS = CFG["connectivity"]["bands"]
SF = 250


def _cfg(method):
    return {**CFG["psd"], "method": method}


@pytest.mark.parametrize("method", ["welch", "multitaper", "periodogram"])
def test_psd_integrates_to_variance(method):
    x = np.random.default_rng(0).standard_normal((3, 20 * SF))
    freqs, psd = compute_psd(x, SF, _cfg(method))
    total = psd.sum(-1) * (freqs[1] - freqs[0])
    assert np.allclose(total, x.var(axis=1), rtol=0.15)


@pytest.mark.parametrize("method", ["welch", "multitaper", "periodogram"])
def test_band_power_locates_oscillation(method):
    t = np.arange(20 * SF) / SF
    rng = np.random.default_rng(1)
    x = np.stack([np.sin(2 * np.pi * f * t) + 0.1 * rng.standard_normal(t.size) for f in (6, 9, 11.5, 20)])
    bp = window_band_powers(x, SF, BANDS, _cfg(method))
    assert bp["abs"].shape == bp["log"].shape == bp["rel"].shape == (4, 4)    # (n_bands, n_signals)
    assert (np.argmax(bp["abs"], axis=0) == np.arange(4)).all()                # signal i peaks in band i
    assert np.allclose(bp["log"], np.log10(bp["abs"]))
    assert (bp["rel"].max(axis=0) > 0.8).all()
    # sine amplitude 1 -> power 0.5, nearly all inside its band
    assert np.allclose(bp["abs"].max(axis=0), 0.5, rtol=0.15)


def test_bands_are_half_open_and_disjoint():
    freqs = np.arange(0, 50.5, 0.5)
    psd = np.ones((1, freqs.size))
    out = band_powers(freqs, psd, BANDS, [1.0, 45.0])
    assert np.allclose(out["abs"][0], [4.0, 2.0, 3.0, 17.0])   # band widths in Hz
    assert np.isclose(out["rel"][0].sum(), 26 / 44)  # 4-30 Hz of 1-45 Hz


def test_unknown_estimator():
    with pytest.raises(ValueError):
        compute_psd(np.zeros((1, 100)), SF, {"method": "fft"})


def test_relative_range_inside_preserved_passband():
    """The relative-power denominator must only cover frequencies the preprocessing filter preserves."""
    import mne

    from adgraph.features.psd import check_relative_range

    pp = CFG["preprocessing"]
    lo, hi = CFG["psd"]["relative_range"]
    check_relative_range([lo, hi], pp["l_freq"], pp["h_freq"])
    h = mne.filter.create_filter(None, pp["sfreq"], pp["l_freq"], pp["h_freq"], method="fir",
                                 fir_design="firwin", verbose="error")
    f = np.fft.rfftfreq(2 ** 18, 1 / pp["sfreq"])
    gain_db = 20 * np.log10(np.abs(np.fft.rfft(h, 2 ** 18)))
    inside = (f >= lo) & (f <= hi)
    assert gain_db[inside].min() > -0.1                      # flat (< 0.1 dB loss) over the whole denominator
    assert gain_db[np.argmin(abs(f - 56.25))] < -40           # stopband well above 45 Hz


@pytest.mark.parametrize("rng_,lo,hi", [([1.0, 50.0], 0.1, 45.0), ([0.05, 45.0], 0.1, 45.0), ([1.0, 45.0], 0.1, 40.0)])
def test_relative_range_outside_passband_is_rejected(rng_, lo, hi):
    from adgraph.features.psd import check_relative_range

    with pytest.raises(ValueError, match="passband"):
        check_relative_range(rng_, lo, hi)
