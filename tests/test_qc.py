"""Participant quality control before classification (adgraph.qc; docs/final/qc.md). Synthetic stores only.

Category A (technical corruption) is detected and blocks the experiments until a decision is recorded; category B
(statistically unusual, valid) is flagged with thresholds fitted on reference / training participants only; nobody
is excluded or reweighted by any rule; the diagnosis never enters the summaries."""

import numpy as np
import pandas as pd
import pytest
import yaml

from adgraph.qc import fold as fold_qc
from adgraph.qc.difficult import difficult_participants
from adgraph.qc.participants import modified_z, participant_summaries, robust_flags, technical_problems
from adgraph.qc.report import QCUnresolved, decisions_path, development_keys, require_qc
from adgraph.synthetic import make_store


def test_clean_store_has_no_category_a(d3_store):
    assert technical_problems(d3_store).empty


@pytest.mark.parametrize("damage,check", [("nan_psd", "non-finite"), ("bad_plv", "valid range"),
                                          ("flat", "constant over all windows"), ("count", "window count")])
def test_category_a_findings_are_detected(damage, check):
    st = make_store(4, seed=1)
    w = np.flatnonzero(st.windows["key"].to_numpy() == "D3-5")
    if damage == "nan_psd":
        st.arrays["psd_log"][w[0], 1, 3] = np.nan
    elif damage == "bad_plv":
        st.arrays["plv"][w[0], 0, 2, 3] = 1.5
        st.arrays["plv"][w[0], 0, 3, 2] = 1.5
    elif damage == "flat":
        st.arrays["psd_rel"][w, 2, 7] = 0.1
    else:
        st.subjects.loc[st.subjects["key"] == "D3-5", "n_windows_kept"] = 99
    t = technical_problems(st)
    assert (t["key"] == "D3-5").any() and t["check"].str.contains(check).any()
    assert set(t["key"]) == {"D3-5"}


def test_summaries_are_label_blind(d3_store):
    a = participant_summaries(d3_store)
    flipped = d3_store.subset(np.arange(d3_store.n_windows))
    flipped.windows = flipped.windows.assign(y=1 - flipped.windows["y"])
    pd.testing.assert_frame_equal(a, participant_summaries(flipped))
    assert "y" not in a.columns


def test_outlier_flags_use_the_reference_participants_only(d3_store):
    summ = participant_summaries(d3_store)
    dev = development_keys(d3_store)
    holdout = [k for k in summ.index if k not in dev]
    flags, long = robust_flags(summ, dev)
    moved = summ.copy()
    moved.loc[holdout] = moved.loc[holdout] * 50 + 1e3                    # wildly different holdout values ...
    z_ref = modified_z(summ, dev)
    z_moved = modified_z(moved, dev)
    pd.testing.assert_frame_equal(z_ref.loc[dev], z_moved.loc[dev])      # ... leave the reference statistics alone
    assert flags.index.tolist() == summ.index.tolist()                     # every participant reported, none removed


def test_an_extreme_participant_is_flagged_and_kept():
    st = make_store(4, seed=2)
    w = np.flatnonzero(st.windows["key"].to_numpy() == "D3-7")
    st.arrays["psd_log"][w] += 4.0                                         # 10^4 more power: unusual, still valid
    summ = participant_summaries(st)
    flags, long = robust_flags(summ, development_keys(st))
    assert flags.loc["D3-7", "category_B"] and (long["key"] == "D3-7").any()
    assert len(flags) == 65 and technical_problems(st).empty               # category B, not A; nobody excluded


def test_fold_qc_thresholds_come_from_the_training_participants():
    rng = np.random.default_rng(0)
    keys = [f"D3-{i}" for i in range(1, 21)]
    s = pd.DataFrame(rng.standard_normal((20, 4)), index=keys, columns=list("abcd"))
    train, test = keys[:15], keys[15:]
    s.loc["D3-20", "a"] = 40.0
    t = fold_qc.fold_qc(s, train, test)
    assert t.set_index("key").loc["D3-20", "flagged"]
    s2 = s.copy()
    s2.loc[test] = s2.loc[test] * 100
    pd.testing.assert_frame_equal(fold_qc.fold_qc(s, train, test).query("side == 'train'"),
                                  fold_qc.fold_qc(s2, train, test).query("side == 'train'"))
    with pytest.raises(RuntimeError):
        fold_qc.fold_qc(s, keys[:16], keys[15:])                            # overlapping participants


def _gate_store(tmp_path, damaged: bool):
    from adgraph.synthetic import save_store
    from adgraph.graphs.store import GraphStore

    st = make_store(3, seed=3)
    if damaged:
        st.arrays["psd_log"][0, 0, 0] = np.nan
    d = save_store(st, tmp_path / "store")
    return GraphStore.load(d)


def test_qc_gate_blocks_unresolved_category_a_and_refuses_exclusions(tmp_path):
    store = _gate_store(tmp_path, damaged=True)
    key = store.windows["key"].iloc[0]
    with pytest.raises(QCUnresolved, match="without a decision"):
        require_qc(store, tmp_path, "lcmv", "B", "dk68")
    p = decisions_path(tmp_path)
    p.write_text(yaml.safe_dump({key: {"decision": "exclude", "reason": "test"}}))
    with pytest.raises(QCUnresolved, match="protocol amendment"):
        require_qc(store, tmp_path, "lcmv", "B", "dk68")
    p.write_text(yaml.safe_dump({key: {"decision": "keep", "reason": "isolated value, checked"}}))
    rep = require_qc(store, tmp_path, "lcmv", "B", "dk68")
    assert rep["summary"]["category_A_participants"] == [key] and rep["summary"]["n_participants"] == 65


def test_qc_report_is_computed_once_and_reused(tmp_path):
    store = _gate_store(tmp_path, damaged=False)
    a = require_qc(store, tmp_path, "lcmv", "B", "dk68")
    stamp = a["summary"]["computed_at"]
    assert require_qc(store, tmp_path, "lcmv", "B", "dk68")["summary"]["computed_at"] == stamp


def test_difficult_participants_are_reported_not_excluded():
    keys = [f"D3-{i}" for i in range(1, 11)]
    y = [1] * 5 + [0] * 5
    a = pd.DataFrame({"key": keys, "y": y, "score": [0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.2, 0.3, 0.4, 0.9]})
    b = a.assign(score=[0.8, 0.9, 0.6, 0.7, 0.3, 0.2, 0.1, 0.3, 0.2, 0.8])
    d = difficult_participants([("m1", a, 0.5), ("m2", b, 0.5)]).set_index("key")
    assert d.loc["D3-5", "difficult"] and d.loc["D3-10", "difficult"] and not d.loc["D3-1", "difficult"]
    assert len(d) == 10
