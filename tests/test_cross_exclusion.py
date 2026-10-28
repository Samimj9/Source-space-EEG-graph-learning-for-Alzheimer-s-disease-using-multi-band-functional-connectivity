"""Amendment X1 (configs/cross_dataset/amendments.yaml; docs/cross_dataset/design.md 12.8): D2-19 is excluded from the
D2 analysis cohort only. The 23-recording input inventory and its audits are preserved, the analysis cohort is
22 (9 AD / 13 HC) and verified against the participant metadata, the evidence is re-checked from the unit audit, and
the amendment is hashed reproducibly into the store configuration. Synthetic only (no EEG)."""

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml

from adgraph.crossdataset import channels, d2store

AD_IDS = [1, 4, 5, 7, 9, 12, 15, 19, 21, 22]            # 10 AD (with 19), 13 HC: the provided D2 composition


def inventory(ad_ids=AD_IDS, n=23) -> pd.DataFrame:
    ids = list(range(1, n + 1))
    return pd.DataFrame({"key": [f"D2-{i}" for i in ids], "subject_id": ids,
                         "y": [int(i in ad_ids) for i in ids], "path": [f"/x/{i}_1_1.edf" for i in ids]})


def test_the_amendment_record_is_complete_and_consistent():
    doc = yaml.safe_load(d2store.AMENDMENTS.read_text())["amendments"]["X1"]
    assert doc["status"] == "approved" and doc["decided_on"] == "2026-10-28" and doc["dataset"] == "D2"
    assert "exceeds" in doc["exceeds_preregistered_rule"].lower() and "fewer than 5 clean windows" in \
        doc["exceeds_preregistered_rule"]
    assert "pre-build" in doc["timing"] and "label-blind" in doc["basis"]
    assert {"D3_to_D2", "D2_to_D3", "d2_source_cv", "d2_stores"} <= set(doc["applies_to"])
    assert {"d2_input_inventory", "d2_unit_audit"} <= set(doc["not_applied_to"])
    spec = d2store.cohort_spec()
    assert spec["inventory"] == {"n": 23, "AD": 10, "HC": 13} and spec["analysis"] == {"n": 22, "AD": 9, "HC": 13}
    assert d2store.EXPECTED_INVENTORY == spec["inventory"] and d2store.EXPECTED_D2 == spec["analysis"]
    (e,) = spec["excluded"]
    assert (e["key"], e["subject_id"], e["diagnosis"]) == ("D2-19", 19, "AD")
    table = {r["d3"]: r for r in channels.load_table()["channels"]}
    assert [x["d3"] for x in e["electrodes"]] == ["T4", "T5", "T6", "O1"]
    for x in e["electrodes"]:                                  # the recorded EDF channels ARE the matched electrodes
        assert (x["d2"], x["edf_index"], x["edf_label"]) == (table[x["d3"]]["d2"], table[x["d3"]]["edf_index"],
                                                             table[x["d3"]]["edf_label"])
    assert "30.5 uV" in e["evidence"] and "constant" in e["evidence"]


def test_split_keeps_the_inventory_and_removes_only_the_excluded_participant():
    inv = inventory()
    snapshot = inv.copy()
    out, ex = d2store.split_cohort(inv)
    pd.testing.assert_frame_equal(inv, snapshot)                       # the input is not modified
    assert len(out) == 23 and out["key"].tolist() == inv["key"].tolist()
    assert out.loc[out["key"] == "D2-19", "cohort"].item() == "excluded (X1)"
    ana = out[out["cohort"] == "analysis"]
    assert d2store.counts(ana) == {"n": 22, "AD": 9, "HC": 13} and "D2-19" not in set(ana["key"])
    assert ex.to_dict("records")[0] | {"reason": ""} == {
        "key": "D2-19", "amendment": "X1", "diagnosis": "AD", "electrodes": "T4,T5,T6,O1",
        "edf_channels": "Misc ExG18,Misc ExG24,Misc ExG28,Misc ExG30", "reason": "", "decided_on": "2026-10-28"}


@pytest.mark.parametrize("change, match", [
    (lambda t: t[t["key"] != "D2-19"], "not in the inventory"),                       # the recording must exist
    (lambda t: t.assign(y=np.where(t["key"] == "D2-19", 0, t["y"])), "diagnosis differs"),   # sheet says HC
    (lambda t: t.assign(y=np.where(t["key"] == "D2-2", 1, t["y"])), "analysis cohort"),     # counts differ
])
def test_the_cohort_is_verified_against_the_participant_metadata(change, match):
    with pytest.raises(d2store.D2StoreError, match=match):
        d2store.split_cohort(change(inventory()))


def test_a_partial_input_folder_still_excludes_d2_19():
    out, ex = d2store.split_cohort(inventory().iloc[15:20], strict=False)
    assert out.loc[out["key"] == "D2-19", "cohort"].item() == "excluded (X1)" and len(ex) == 1
    out, ex = d2store.split_cohort(inventory().iloc[:5], strict=False)
    assert (out["cohort"] == "analysis").all() and ex.empty


def test_assert_no_excluded():
    d2store.assert_no_excluded(["D2-1", "D2-20", "D3-19"], "ok")
    with pytest.raises(d2store.D2StoreError, match="D2-19"):
        d2store.assert_no_excluded(["D2-1", "D2-19"], "fold")


def test_the_amendment_is_hashed_reproducibly_into_the_store_configuration(tmp_path, monkeypatch):
    from adgraph.config import config_hash

    a, b = d2store.cohort_spec(), d2store.cohort_spec()
    assert a == b and a["hash"] == b["hash"]
    for inv in ("lcmv", "wmne"):
        cfg = d2store.resolve(inv)
        assert cfg["cohort"] == {"amendments": ["X1"], "hash": a["hash"], "inventory": a["inventory"],
                                 "analysis": a["analysis"], "excluded": ["D2-19"]}
        assert config_hash(cfg) == config_hash(d2store.resolve(inv))
    # a different exclusion is a different cohort -> a different store configuration (never silently reused)
    doc = yaml.safe_load(d2store.AMENDMENTS.read_text())
    other = copy.deepcopy(doc)
    other["amendments"]["X1"]["excluded"][0]["electrodes"] = other["amendments"]["X1"]["excluded"][0]["electrodes"][:3]
    f = tmp_path / "amendments.yaml"
    f.write_text(yaml.safe_dump(other))
    assert d2store.cohort_spec(f)["hash"] != a["hash"]
    bad = copy.deepcopy(doc)
    bad["amendments"]["X1"]["analysis_cohort"] = {"n": 22, "AD": 10, "HC": 12}
    f.write_text(yaml.safe_dump(bad))
    with pytest.raises(d2store.D2StoreError, match="not the inventory"):
        d2store.cohort_spec(f)
    bad = copy.deepcopy(doc)
    bad["amendments"]["X1"]["status"] = "proposed"
    f.write_text(yaml.safe_dump(bad))
    with pytest.raises(d2store.D2StoreError, match="approved"):
        d2store.cohort_spec(f)


def _units(flat=("T4", "T5", "T6", "O1"), finding="flat", extra=None):
    rows = []
    for key in ("D2-18", "D2-19"):
        for r in channels.load_table()["channels"]:
            f = finding if key == "D2-19" and r["d3"] in flat else "ok"
            rows.append({"key": key, "selected_as": r["d3"], "finding": f})
        rows.append({"key": key, "selected_as": "", "finding": "flat" if key == "D2-19" else "ok"})  # CP1
    if extra:
        rows = [{**r, "finding": extra[1]} if r["key"] == "D2-19" and r["selected_as"] == extra[0] else r
                for r in rows]
    return pd.DataFrame(rows)


def test_the_unit_audit_must_still_show_the_recorded_evidence():
    ev = d2store.exclusion_evidence(_units())
    assert ev["evidence_confirmed"].all() and ev["audited_electrodes_without_eeg"].item() == "O1,T4,T5,T6"
    assert d2store.exclusion_evidence(_units(finding="no_eeg_below_quantization"))["evidence_confirmed"].all()
    assert not d2store.exclusion_evidence(_units(flat=("T4", "T5", "T6")))["evidence_confirmed"].any()
    assert not d2store.exclusion_evidence(_units(extra=("Cz", "too_small")))["evidence_confirmed"].any()
    assert not d2store.exclusion_evidence(_units(finding="too_large"))["evidence_confirmed"].any()
    assert not d2store.exclusion_evidence(_units().query("key != 'D2-19'"))["evidence_confirmed"].any()


def test_every_other_gate_is_unchanged():
    """No QC threshold, unit rule, rejection or preprocessing parameter changes with the amendment."""
    spec = d2store.spec()
    assert spec["overrides"]["units"]["channel_robust_std_v"] == [1.0e-7, 1.0e-3]
    assert spec["overrides"]["units"]["physical_dimensions"] == ["V", "mV", "uV", "nV"]
    from adgraph.sourcegraphs.config import resolve

    for inv in ("lcmv", "wmne"):
        d2, d3 = d2store.resolve(inv), resolve(inv, "lobes12")
        for k in ("windowing", "filters", "features", "source", "bands", "psd"):
            assert d2[k] == d3[k]
        assert "interpolat" not in str(d2["preprocessing"]).lower()


def test_the_raw_recording_is_only_read(tmp_path):
    """The real loader on a D2-19-like EDF: the participant still fails the unchanged unit gate (no EEG in T4, T5,
    T6, O1) - the amendment removes it from the cohort, it does not bypass the gate - and the file is unchanged."""
    import edfio

    rng = np.random.default_rng(0)
    data = rng.standard_normal((32, 1024 * 40)) * 10.0
    data[[17, 23, 27, 29]] = 0.0
    f = tmp_path / "19_842019_1.edf"
    edfio.Edf([edfio.EdfSignal(x, sampling_frequency=1024, label=f"Misc ExG{k + 1}", physical_dimension="uV",
                               physical_range=(-20000, 20000)) for k, x in enumerate(data)]).write(f)
    before = hashlib.sha256(f.read_bytes()).hexdigest()
    table = channels.load_table()
    with pytest.raises(d2store.UnitError, match="T4.*T5.*T6.*O1|O1.*T4"):
        d2store.load_d2(f, "D2", d2store.resolve("lcmv"), table, channels.template_geometry(table))
    assert hashlib.sha256(f.read_bytes()).hexdigest() == before
