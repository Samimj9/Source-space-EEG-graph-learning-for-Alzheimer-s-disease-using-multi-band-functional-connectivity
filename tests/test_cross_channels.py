"""Cross-dataset D3 <-> D2: the matched 19-electrode mapping, the EDF channel-index mapping, the electrode geometry and
the D2 loader (design section 3). Synthetic only: the layout text is the real AD-LEB layout file (its SHA-256 is
checked); the EDFs are generated."""

import copy
import hashlib

import numpy as np
import pytest
import yaml

from adgraph.crossdataset import channels
from adgraph.crossdataset.channels import MappingError

from test_channel_mapping import LOC

D3_ORDER = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz",
            "Cz", "Pz"]
D2_LAYOUT = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
             "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"]
D2_NAMES = [f"Misc ExG{k}" for k in range(1, 33)]


@pytest.fixture
def table():
    return channels.load_table()


@pytest.fixture
def loc(tmp_path):
    p = tmp_path / "electrodes_positions_32.loc"
    p.write_bytes(LOC.encode())
    return p


def _write(tmp_path, t):
    p = tmp_path / "map.yaml"
    p.write_text(yaml.safe_dump(t, sort_keys=False))
    return p


# ------------------------------------------------------------------------------------------- 1. the 19-channel map
def test_the_table_is_the_d3_store_order_with_the_four_documented_equivalences(table):
    from adgraph.sourcegraphs.config import resolve

    assert table["canonical_order"] == D3_ORDER == resolve("lcmv", "lobes12")["channels"]["recorded"]
    d2 = {r["d3"]: r["d2"] for r in table["channels"]}
    assert {k: v for k, v in d2.items() if k != v} == {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    assert set(d2.values()) | set(table["excluded_d2"]) == set(D2_LAYOUT) and len(table["excluded_d2"]) == 13
    for r in table["channels"]:                                   # index = position in the 32-electrode layout
        assert D2_LAYOUT[r["edf_index"] - 1] == r["d2"] and r["edf_label"] == f"Misc ExG{r['edf_index']}"
    assert [r["edf_index"] for r in table["channels"]] != list(range(1, 20))     # never "the first 19 channels"


@pytest.mark.parametrize("edit,match", [
    (lambda t: t["channels"][12].update(d2="T3"), "mapped to"),                      # T3 must be T7 in D2
    (lambda t: t["channels"][14].update(d2="T5"), "mapped to"),
    (lambda t: (t["channels"][0].update(edf_index=3, edf_label="Misc ExG3")), "distinct"),   # duplicate index
    (lambda t: t["channels"].reverse(), "canonical order"),
    (lambda t: t["channels"].pop(), "canonical order"),
    (lambda t: t["channels"][2].update(edf_label="Misc ExG6"), "does not name"),
    (lambda t: t["equivalences"].update(T3="T8"), "mapped to|equivalences"),
])
def test_a_wrong_table_is_refused(tmp_path, table, edit, match):
    t = copy.deepcopy(table)
    edit(t)
    with pytest.raises(MappingError, match=match):
        channels.load_table(_write(tmp_path, t))


def test_the_layout_file_is_verified_by_hash_and_content(tmp_path, table, loc):
    df = channels.check_layout(loc, table)
    assert hashlib.sha256(loc.read_bytes()).hexdigest() == table["layout_sha256"] and len(df) == 32
    bad = tmp_path / "other.loc"
    bad.write_bytes(LOC.replace("14 -90 0.500 T7..", "14 -90 0.500 T3..").encode())
    with pytest.raises(MappingError, match="SHA-256"):
        channels.check_layout(bad, table)
    t = copy.deepcopy(table)
    t["layout_sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
    with pytest.raises(MappingError, match="is 'T3'"):
        channels.check_layout(bad, t)                             # content checked even when the hash is accepted


# ------------------------------------------------------------------------------------------- 2. EDF index mapping
def test_edf_picks_select_by_number_in_canonical_order(table):
    picks = channels.edf_picks(D2_NAMES, table)
    assert [d3 for _, d3 in picks] == D3_ORDER
    assert dict(picks)["Misc ExG14"] == "T3" and dict(picks)["Misc ExG18"] == "T4"
    assert dict(picks)["Misc ExG24"] == "T5" and dict(picks)["Misc ExG28"] == "T6"
    shuffled = list(np.random.default_rng(0).permutation(D2_NAMES))          # header order is irrelevant
    assert channels.edf_picks(shuffled, table) == picks


@pytest.mark.parametrize("names,match", [
    (D2_NAMES[:-1], "not 1..32"),                                            # a channel missing
    (D2_NAMES + ["Misc ExG5"], "not 1..32"),                                 # a channel twice
    ([n.replace("Misc ExG", "EEG E") for n in D2_NAMES], "expected 'Misc ExG"),   # another device's labels
    (D2_NAMES[:-1] + ["Misc ExG33"], "not 1..32"),
])
def test_a_wrong_edf_inventory_is_refused(table, names, match):
    with pytest.raises(MappingError, match=match):
        channels.edf_picks(names, table)


def _write_edf(path, order, sf=1024, seconds=60, scale=None, annotate=False):
    """EDF whose channel number k carries the same 10 uV signal times k (identity readable from the amplitude)."""
    import edfio

    rng = np.random.default_rng(1)
    base = rng.standard_normal(sf * seconds) * 10.0
    sigs = [edfio.EdfSignal(base * int(n.split("ExG")[1]) * (scale or 1.0), sampling_frequency=sf, label=n,
                            physical_dimension="uV", physical_range=(-20000, 20000)) for n in order]
    edf = edfio.Edf(sigs)
    if annotate:
        edf.set_annotations([edfio.EdfAnnotation(10.0, None, "boundary")])
    edf.write(path)


def test_load_d2_reads_exactly_the_mapped_channels_by_number(tmp_path, table):
    from adgraph.crossdataset import d2store

    order = list(np.random.default_rng(2).permutation(D2_NAMES))             # numbers written in shuffled order
    _write_edf(tmp_path / "1_0_1.edf", order)
    cfg = d2store.resolve("lcmv", table)
    raw, rec = d2store.load_d2(tmp_path / "1_0_1.edf", "D2", cfg, table, channels.template_geometry(table))
    assert raw.ch_names == D3_ORDER and raw.info["sfreq"] == 1024
    x = raw.get_data()
    idx = np.array([r["edf_index"] for r in table["channels"]], float)
    ratio = x.std(axis=1) / x[0].std()                                       # channel 0 = Fp1 = number 1
    assert np.allclose(ratio, idx / idx[0], rtol=1e-3)
    n = 60 * 1024
    assert rec["x_edge_crop_samples"] == 10 * 1024 and rec["x_settle_samples"] == 5 * 1024
    assert x.shape[1] == n - 2 * (10 + 5) * 1024
    expected = np.median(idx) * 10e-6 * np.sqrt((45 - 0.5) / 512)          # volts; white noise kept in 0.5-45 Hz
    assert abs(np.median(np.std(x, axis=1)) - expected) / expected < 0.1


def test_load_d2_refuses_discontinuities_and_short_recordings(tmp_path, table):
    from adgraph.crossdataset import d2store

    cfg = d2store.resolve("lcmv", table)
    g = channels.template_geometry(table)
    _write_edf(tmp_path / "a.edf", D2_NAMES, annotate=True)
    with pytest.raises(d2store.D2StoreError, match="boundary"):
        d2store.load_d2(tmp_path / "a.edf", "D2", cfg, table, g)
    _write_edf(tmp_path / "b.edf", D2_NAMES, seconds=20)
    with pytest.raises(d2store.D2StoreError, match="shorter"):
        d2store.load_d2(tmp_path / "b.edf", "D2", cfg, table, g)


# ------------------------------------------------------------------------------------------- 3. geometry
def test_geometry_agrees_with_the_layout_and_the_d3_template(table):
    geo = channels.check_geometry(table)
    assert (geo["angle_to_loc_deg"] <= table["max_angle_deg"]).all() and (geo["d2"] == geo["nearest_loc_electrode"]).all()
    pos = channels.template_positions([r["d2"] for r in table["channels"]])
    assert np.allclose(pos, channels.template_positions(D3_ORDER))           # T7 == T3 etc. in colin27 10-05
    left = [i for i, c in enumerate(D3_ORDER) if c[-1] in "13579"]
    right = [i for i, c in enumerate(D3_ORDER) if c[-1] in "2468"]
    assert (pos[left, 0] < 0).all() and (pos[right, 0] > 0).all()


def test_swapped_or_moved_positions_are_refused(table):
    pos = np.array([r["mni_mm"] for r in table["channels"]], float) / 1000
    swap = pos.copy()
    swap[[12, 13]] = swap[[13, 12]]                                          # T3 <-> T4 (left / right flip)
    with pytest.raises(MappingError):
        channels.check_geometry(table, swap)
    moved = pos.copy()
    moved[0] += [0, 0, 0.01]                                                 # 10 mm off the template
    with pytest.raises(MappingError, match="deviate"):
        channels.check_geometry(table, moved)
    mirrored = pos * [-1, 1, 1]                                              # wrong axis convention
    with pytest.raises(MappingError):
        channels.check_geometry(table, mirrored)


def test_the_d2_geometry_is_the_d3_geometry(table):
    ref = {**channels.template_geometry(table), "source": "eeglab_chanlocs", "lookup": "standard_1005.elc"}
    from adgraph.io.chanlocs import geometry_hash

    ref["hash"] = geometry_hash(ref)
    g = channels.harmonized_geometry(ref, table)
    assert g["pos_m"] == ref["pos_m"] and g["channels"] == D3_ORDER and g["frame"] == "mri"
    assert g["reference_hash"] == ref["hash"] and g["source"] == "d3_reference_geometry"
    with pytest.raises(MappingError, match="channels"):
        channels.harmonized_geometry({**ref, "channels": D3_ORDER[::-1]}, table)
    with pytest.raises(MappingError, match="frame"):
        channels.harmonized_geometry({**ref, "frame": "head"}, table)


def test_the_mapping_audit(table, loc):
    df, summary = channels.audit(table, loc, headers=__import__("pandas").DataFrame(
        {"key": ["D2-2", "D2-3"], "raw_names": [",".join(D2_NAMES)] * 2}))
    assert list(df["d3"]) == D3_ORDER and summary["layout"]["verified"]
    assert summary["recordings"] == {"n": 2, "all_19_present_once": True, "identical_picks": True}
    assert summary["geometry"]["max_angle_deg"] < 15



# ------------------------------------------------------------------------------------------- 4. selection before CAR
def test_unselected_channels_never_reach_the_reference_or_the_source_model(tmp_path, table):
    """A huge signal on the 13 unselected channels changes nothing: selection happens on the EDF header, before the
    band-pass, the common average (over the 19 channels only) and the inverse."""
    import edfio

    from adgraph.crossdataset import d2store
    from adgraph.sourcegraphs import recording

    rng = np.random.default_rng(3)
    sf, n = 1024, 1024 * 60
    base = rng.standard_normal((32, n)) * 10.0 + rng.standard_normal(n) * 5.0
    loud = base.copy()
    mapped = {r["edf_index"] - 1 for r in table["channels"]}
    for k in set(range(32)) - mapped:
        loud[k] = rng.standard_normal(n) * 5000.0                             # 5 mV on Fpz, FC*, M1/M2, CP*, POz, Oz
    for name, data in (("a.edf", base), ("b.edf", loud)):
        edfio.Edf([edfio.EdfSignal(x, sampling_frequency=sf, label=f"Misc ExG{k + 1}", physical_dimension="uV",
                                   physical_range=(-30000, 30000)) for k, x in enumerate(data)]).write(tmp_path / name)
    cfg = d2store.resolve("lcmv", table)
    g = channels.template_geometry(table)
    ra, _ = d2store.load_d2(tmp_path / "a.edf", "D2", cfg, table, g)
    rb, _ = d2store.load_d2(tmp_path / "b.edf", "D2", cfg, table, g)
    assert np.allclose(ra.get_data(), rb.get_data(), atol=1e-12)
    segs, _ = recording.segments(rb, cfg)
    assert segs[0].data.shape[0] == 19 and np.allclose(segs[0].data.mean(axis=0), 0, atol=1e-12)   # CAR of 19
    sa, _ = recording.segments(ra, cfg)
    assert np.allclose(sa[0].data, segs[0].data, atol=1e-12)


# ------------------------------------------------------------------------------------------- D3 / D4: signal audits
def test_the_d2_filter_response_is_verified(table):
    from adgraph.crossdataset import d2store, signal_audit

    fr = signal_audit.filter_response(d2store.resolve("lcmv", table))
    m = fr["magnitude_db"]
    assert abs(m[0.5] + 6.02) < 0.05 and abs(m[45.0] + 6.02) < 0.05           # zero phase: -6 dB at both edges
    assert all(abs(m[f]) < 0.1 for f in (2.0, 4.0, 8.0, 10.0, 13.0, 20.0))    # flat inside delta-beta
    assert m[50.0] < -10 and m[0.25] < -40 and fr["settle_covers_impulse"]


def test_line_noise_and_notch_sensitivity_are_measured_not_applied():
    from adgraph.crossdataset import signal_audit

    rng = np.random.default_rng(0)
    sf, t = 250.0, np.arange(250 * 60) / 250.0
    clean = rng.standard_normal((19, t.size)) * 1e-5
    noisy = clean + 2e-5 * np.sin(2 * np.pi * 50 * t)
    a, b = signal_audit.line_noise(clean, sf), signal_audit.line_noise(noisy, sf)
    assert b["line_ratio_db"] > a["line_ratio_db"] + 10 and b["line_variance_fraction"] > a["line_variance_fraction"]
    assert np.abs(signal_audit.notch(noisy, sf) - noisy).max() > 0                # computed on a copy only


def test_the_header_audit_lists_annotation_texts(tmp_path):
    import edfio
    import pandas as pd

    from adgraph.crossdataset import d2store

    p = tmp_path / "2_842019_1.edf"
    edf = edfio.Edf([edfio.EdfSignal(np.zeros(1024 * 10), sampling_frequency=1024, label=n, physical_dimension="uV",
                                     physical_range=(-100, 100)) for n in D2_NAMES])
    edf.set_annotations([edfio.EdfAnnotation(1.0, None, "Eyes closed"), edfio.EdfAnnotation(5.0, None, "EO")])
    edf.write(p)
    h = d2store.header_audit(pd.DataFrame({"key": ["D2-2"], "path": [str(p)]}))
    assert h["annotations"].iloc[0] == "EO|Eyes closed" and h["n_signals"].iloc[0] == 32
    assert "patient" not in " ".join(h.columns)
