"""Regression tests for issues found in the pre-real-data audit (synthetic data only)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adgraph.experiments import runner
from adgraph.experiments.config import expand_variants, load_experiment, set_dotted
from adgraph.experiments.plan import variant_matches
from adgraph.graphs.dataset import build_dataset
from adgraph.io.eeg import normalize_channel_name
from adgraph.io.participants import normalize_participants
from adgraph.legacy.synthetic import make_synthetic_store
from test_checkpoint import CFG, cohort  # noqa: F401  (fixture)

EXP = Path("configs/experiments")


def _variant(path, name, **over):
    v = next(x for x in expand_variants(load_experiment(EXP / path)) if x["variant"] == name)
    for k, val in over.items():
        v = set_dotted(v, k, val)
    return v


# ----------------------------------------------------------------- experiments








def test_gnn_loss_gives_each_batch_weight_by_its_windows():
    """Loss = sum(w * l) / batch size, with weights of mean 1 over the training set: a batch of
    low-weight windows (a subject with many windows) contributes less, as one vote per subject requires."""
    import torch
    from torch.nn import functional as F

    logits, y = torch.zeros(4), torch.tensor([1.0, 1.0, 0.0, 0.0])
    w_low, w_high = torch.full((4,), 0.5), torch.full((4,), 1.5)
    loss = lambda w: (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / 4  # noqa: E731
    assert loss(w_high) == pytest.approx(3 * loss(w_low))
    src = Path("src/adgraph/models/trainer.py").read_text()
    assert ".sum() / len(idx)" in src and "/ w[idx].sum()" not in src




# ----------------------------------------------------------------- inputs
def test_unrecognised_group_is_an_error_but_footers_are_skipped():
    ok = normalize_participants(pd.DataFrame({"ID": ["Sub1", "Sub2", "Total", None, "Sub3"],
                                              "Group": ["AD", "hc", None, None, None],
                                              "Gender": ["F", None, None, None, None]}), "D1")
    assert ok["key"].tolist() == ["D1-1", "D1-2"] and ok["sex"].tolist() == ["F", ""]
    with pytest.raises(ValueError, match="unrecognised Group"):
        normalize_participants(pd.DataFrame({"ID": ["Sub1", "Sub2"], "Group": ["AD", "Control"]}), "D1")


@pytest.mark.parametrize("label", ["Fp1", "EEG Fp1-REF", "FP1-A1", "Fp1-M2", "fp1_A1A2", "Fp1."])
def test_reference_suffixes_are_recognised(label):
    assert normalize_channel_name(label, ["Fp1", "M1", "M2"]) == "Fp1"
    assert normalize_channel_name("M1", ["Fp1", "M1", "M2"]) == "M1"


def test_graph_run_refuses_recordings_without_label(cohort):
    pl, paths, counts, tmp = cohort
    shutil.copy(Path(paths["dataset1"]["eeg_dir"]) / "sub1.edf", Path(paths["dataset1"]["eeg_dir"]) / "sub7.edf")
    with pytest.raises(ValueError, match=r"no AD/HC label.*D1-7"):
        pl.run(paths, CFG, tmp / "out", tmp / "sd")
    assert counts["process"] == []                                  # refused before any processing


# ----------------------------------------------------------------- figures
@pytest.fixture(scope="module")
def results(synthetic_results):
    return synthetic_results                           # shared session fixture (conftest)


def _paper():
    import yaml

    return yaml.safe_load(Path("configs/figures/paper.yaml").read_text())
