"""Source-resolution metrics and the atlas rule (source.resolution; docs/final/parcellation.md), on synthetic
leadfields and operators (no fsaverage, no EEG)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from adgraph.source import atlas
from adgraph.source.resolution import (recommend_atlas, resolution_metrics, spatial_rank, split_half_reliability,
                                       _avg_ref)


def test_average_reference_removes_one_spatial_dimension():
    rng = np.random.default_rng(0)
    g = _avg_ref(19) @ rng.standard_normal((19, 2000))
    r = spatial_rank(g)
    assert r["numerical_rank"] == 18 and 1 <= r["effective_rank_entropy"] <= 18
    assert r["n_sv_95pct_energy"] <= r["n_sv_99pct_energy"] <= 18


def test_perfect_resolution_has_no_crosstalk():
    rng = np.random.default_rng(1)
    g = rng.standard_normal((12, 12))
    m = resolution_metrics(np.linalg.inv(g), g)                           # W G = I: every ROI sees only itself
    assert m["crosstalk_fraction_median"] == pytest.approx(0, abs=1e-10) and m["rois_dominated_by_own_region"] == 1
    assert m["redundant_signals"] == 0 and m["operator_rank"] == 12


def test_more_regions_than_channels_forces_redundancy_and_leakage():
    rng = np.random.default_rng(2)
    g = _avg_ref(19) @ rng.standard_normal((19, 68))
    W = np.linalg.pinv(g)                                                 # best linear estimate, 68 ROIs
    m = resolution_metrics(W, g)
    assert m["operator_rank"] == 18 and m["redundant_signals"] == 50
    assert m["crosstalk_fraction_median"] > 0.5
    assert m["participation_ratio"] < 18


def _table(rows):
    return pd.DataFrame([{"atlas": a, "inverse": inv, "n_regions": atlas.N_REGIONS[a], "redundant_signals": red,
                          "crosstalk_fraction_median": ct, "rois_dominated_by_own_region": own}
                         for a, inv, red, ct, own in rows])


RULE = yaml.safe_load(open("configs/final/selection.yaml"))["atlas_resolution"]


def test_the_finest_admissible_atlas_is_recommended_for_both_inverses():
    rows = []
    for inv in ("lcmv", "wmne"):
        rows += [("dk68", inv, 50, 0.85, 0.55), ("dk34_gyral", inv, 16, 0.73, 0.68), ("r18", inv, 0, 0.5, 0.89),
                 ("lobes12", inv, 0, 0.4, 1.0)]
    rec = recommend_atlas(_table(rows), RULE)
    assert rec["recommended"] == "r18"
    v = {x["atlas"]: x for x in rec["verdicts"]}
    assert not v["dk68"]["eligible"] and any("linear combinations" in r for r in v["dk68"]["reasons"])
    rows = [r if not (r[0] == "r18" and r[1] == "wmne") else ("r18", "wmne", 0, 0.65, 0.89) for r in rows]
    assert recommend_atlas(_table(rows), RULE)["recommended"] == "lobes12"     # one inverse failing is enough
    assert recommend_atlas(_table([]), RULE)["recommended"] is None


def test_split_half_reliability():
    rng = np.random.default_rng(3)
    keys = np.repeat([f"D3-{i}" for i in range(40)], 10)
    trait = np.repeat(rng.standard_normal(40), 10)
    stable = trait + 0.1 * rng.standard_normal(400)
    noise = rng.standard_normal(400)
    rel = split_half_reliability(pd.DataFrame({"stable": stable, "noise": noise}), keys)
    assert rel["stable"] > 0.95 and abs(rel["noise"]) < 0.6


def test_atlases_partition_the_desikan_killiany_labels_per_hemisphere():
    for a in atlas.ATLASES:
        m = atlas.aggregation_matrix(a)
        assert m.shape == (atlas.N_REGIONS[a], 68) and (m.sum(0) == 1).all()
        names = atlas.region_names("dk68")
        for i, r in enumerate(atlas.region_names(a)):
            assert all(names[j].endswith(r[-3:]) for j in np.flatnonzero(m[i]))    # never across hemispheres
