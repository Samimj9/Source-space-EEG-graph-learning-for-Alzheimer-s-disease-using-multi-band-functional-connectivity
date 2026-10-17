"""Every LCMV / Xu experiment config (Exp 0-8, modes A and B) schedules both of its settings; a setting that no
scheduling pass knows fails loudly instead of running nothing (methodology log entry 54)."""

from pathlib import Path

import pytest

from adgraph.experiments import plan
from adgraph.experiments.config import load_experiment

LX = sorted(Path("configs/experiments/lx").rglob("*.yaml"))


@pytest.mark.parametrize("config", LX, ids=lambda p: p.stem)
def test_lx_configs_schedule_dev_cv_and_holdout(config):
    entries = plan.experiment_entries(str(config))
    units = plan.plan_units(entries)
    cfg = load_experiment(config)
    assert {s for _, _, _, s in units} == {"xu_dev_cv", "xu_holdout"}
    assert len(units) == 2 * len({u[2]["name"] for u in units}) > 0


def test_lx_exp0_has_four_configurations():
    units = plan.plan_units(plan.experiment_entries("configs/experiments/lx/lx_exp0_sensor_psd_de.yaml"))
    assert sorted({u[2]["variant"] for u in units}) == ["logreg__de", "logreg__psd", "svm__de", "svm__psd"]


def test_unscheduled_setting_fails_loudly(monkeypatch):
    monkeypatch.setattr(plan, "SETTING_PASSES", plan.SETTING_PASSES[:3])     # the old passes, without the Xu ones
    with pytest.raises(plan.EmptySchedule, match="xu_dev_cv"):
        plan.experiment_entries("configs/experiments/lx/lx_exp0_sensor_psd_de.yaml")
    with pytest.raises(plan.EmptySchedule):
        plan.plan_units([])
