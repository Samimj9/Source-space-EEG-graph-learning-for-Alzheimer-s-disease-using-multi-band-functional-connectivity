import numpy as np
import pandas as pd
import pytest

from adgraph.config import common_channels, config_hash, load_yaml
from adgraph.io.eeg import channel_mapping, normalize_channel_name
from adgraph.io.participants import normalize_participants
from adgraph.preprocessing.clean import detect_bad_channels

CFG = load_yaml("configs/pipeline.yaml")
VALID = CFG["channels"]["recorded"]


@pytest.mark.parametrize("raw,canon", [("Fp1", "Fp1"), ("FP1", "Fp1"), ("Fp1.", "Fp1"), ("EEG FP1-REF", "Fp1"),
                                       ("POz", "POz"), ("POZ.", "POz"), ("M1", "M1"), ("ECG", None), ("EOG1", None)])
def test_normalize_channel_name(raw, canon):
    assert normalize_channel_name(raw, VALID) == canon


def test_channel_mapping_rejects_duplicates():
    mapping, unknown = channel_mapping(["Fp1", "FP1", "Cz", "Status"], VALID)
    assert mapping == {"Fp1": "Fp1", "Cz": "Cz"} and unknown == ["FP1", "Status"]


def test_common_channels_harmonized():
    chans = common_channels(CFG)
    assert len(chans) == 30 and "M1" not in chans and "M2" not in chans


def test_config_hash_order_independent():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})


def test_normalize_participants_both_layouts():
    d1 = pd.DataFrame({"ID": ["Sub1", "Sub2", None], "Age": [72, 68, None], "Gender": ["M", "F", None],
                       "Group": ["AD", "HC", None], "MMSE": [21, 27, None], "Comments": [None, "2-3 bad channels", None]})
    d2 = pd.DataFrame({"Id": ["2_842019", "17_952019"], "Age": [71, 79], "Gender": ["M", "F"], "Group": ["HC", "AD"],
                       "ACE-R score (/100)": [88, 39], "MMSE (/30)": [29, 11], "Comments": [None, "1-2 bad channels"]})
    a, b = normalize_participants(d1, "D1"), normalize_participants(d2, "D2")
    assert a["key"].tolist() == ["D1-1", "D1-2"] and a["y"].tolist() == [1, 0]
    assert b["key"].tolist() == ["D2-2", "D2-17"] and b["acer"].tolist() == [88, 39] and b["mmse"].tolist() == [29, 11]


def test_detect_bad_channels():
    rng = np.random.default_rng(0)
    common = rng.standard_normal(5000)
    data = (common + 0.3 * rng.standard_normal((10, 5000))) * 1e-5
    data[3] = 0.0                                  # flat
    data[5] *= 200                                 # huge amplitude
    data[7] = rng.standard_normal(5000) * 1e-5     # uncorrelated
    bads = detect_bad_channels(data, [f"c{i}" for i in range(10)], CFG["preprocessing"]["bad_channels"])
    assert set(bads) == {"c3", "c5", "c7"}
