"""Check that each oscillator experiment makes exactly its stated change."""

from configgle import PartialConfig

import torch

from priml.baselines.etth1.experiments import exp000 as etth1_exp000
from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.experiments import exp000, exp001, exp002, exp_smoke
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection
from priml.testing.golden import assert_pprint_golden


def test_config_is_printable_without_running_a_simulator() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_only_experiments_name_a_run() -> None:
    assert ForecastStudy.Config().experiment_name == ""
    assert exp000().experiment_name == "exp000"


def test_no_experiment_pins_a_seed() -> None:
    assert ForecastStudy.Config().seed is None
    for factory in (exp000, exp001, exp002, exp_smoke):
        assert factory().seed is None


def test_exp000_changes_only_the_starting_rate() -> None:
    cfg = exp000()
    assert isinstance(cfg.method, DLinearForecast.Config)
    assert cfg.method.step.optimizer == PartialConfig(torch.optim.Adam, lr=1e-3)
    cfg.method.step.optimizer = DLinearForecast.Config().step.optimizer
    assert cfg.method == DLinearForecast.Config()


def test_pretraining_fork_changes_only_the_data_allocation() -> None:
    before = exp000()
    after = exp001()
    assert isinstance(before.method, DLinearForecast.Config)
    assert isinstance(after.method, DLinearForecast.Config)
    assert after.method.pretrain_updates == before.method.updates // 2
    after.method.pretrain_updates = before.method.pretrain_updates
    after.experiment_name = before.experiment_name
    assert after == before


def test_ridge_fork_changes_only_the_method() -> None:
    before = exp000()
    after = exp002()
    assert after.method == RidgeCorrection.Config()
    after.method = before.method
    after.experiment_name = before.experiment_name
    assert after == before


def test_smoke_shrinks_only_sizes() -> None:
    cfg = exp_smoke()
    base = exp000()
    assert isinstance(cfg.method, DLinearForecast.Config)
    assert isinstance(base.method, DLinearForecast.Config)
    assert cfg.method.updates == 2
    cfg.method.updates = base.method.updates
    assert cfg.method.step == base.method.step


def test_new_geometry_does_not_change_etth1() -> None:
    old = etth1_exp000().copy_tree().finalize()
    cfg = exp000().copy_tree().finalize()
    assert isinstance(cfg.method, DLinearForecast.Config)
    assert cfg.method.step.model.seq_len == cfg.data.seq_len
    assert cfg.method.step.model.pred_len == cfg.data.pred_len
    assert cfg.method.step.model.channels == 2
    assert cfg.method.step.train_budget_steps == cfg.method.updates
    assert old.step.model.seq_len == 336
    assert old.step.model.pred_len == 96
    assert old.step.model.channels == 7


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
