"""Check that each oscillator experiment makes exactly its stated change."""

from configgle import PartialConfig
from configgle.launch import resolve_config

import pytest
import torch

from priml.baselines.etth1.experiments import exp000 as etth1_exp000
from priml.baselines.oscillator import experiments
from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.calibration import CalibratedSimulator
from priml.baselines.oscillator.delayed import (
    DelayedCalibration,
    DelayedData,
    DelayedOscillator,
    DelayedStudy,
    SignedHistory,
)
from priml.baselines.oscillator.discovery import SparseSimulator
from priml.baselines.oscillator.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp_smoke,
)
from priml.baselines.oscillator.forces import CorrectedOscillator
from priml.baselines.oscillator.nonlinear import (
    HistoryAblation,
    KernelCorrection,
    NeuralForecast,
    PolynomialKernel,
)
from priml.baselines.oscillator.suite import BenchmarkSuite
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection
from priml.model.init import unit_fan_in_uniform
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.testing.golden import assert_pprint_golden


def test_config_is_printable_without_running_a_simulator() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_each_experiment_names_itself_and_pins_no_seed() -> None:
    assert ForecastStudy.Config().experiment_name == ""
    assert ForecastStudy.Config().seed is None
    names = [name for name in vars(experiments) if name.startswith("exp")]
    assert set(names) == {f"exp{index:03d}" for index in range(41)} | {"exp_smoke"}
    for name in names:
        cfg = resolve_config(f"{experiments.__name__}.{name}")
        if name == "exp040":
            assert isinstance(cfg, BenchmarkSuite.Config)
            assert all(recipe.study.seed is None for recipe in cfg.recipes)
            continue
        assert isinstance(cfg, ForecastStudy.Config)
        assert cfg.experiment_name == name
        assert cfg.seed is None


def test_exp000_changes_only_the_starting_rate() -> None:
    cfg = exp000()
    assert isinstance(cfg.method, DLinearForecast.Config)
    assert cfg.method.step.optimizer == PartialConfig(torch.optim.Adam, lr=1e-3)
    cfg.method.step.optimizer = DLinearForecast.Config().step.optimizer
    assert cfg.method == DLinearForecast.Config()


@pytest.mark.parametrize("number", [32, 34, 35, 36, 38])
def test_transfer_controls_change_only_the_forecaster(number: int) -> None:
    parent = experiments.exp035() if number == 36 else experiments.exp031()
    cfg = resolve_config(f"{experiments.__name__}.exp{number:03d}")
    assert isinstance(cfg, ForecastStudy.Config)
    assert cfg.method != parent.method
    cfg.method = parent.method
    cfg.experiment_name = parent.experiment_name
    assert cfg == parent


def test_transfer_training_budget_changes_only_the_update_limit() -> None:
    before = experiments.exp030()
    after = experiments.exp031()
    assert isinstance(before.method, DLinearForecast.Config)
    assert isinstance(after.method, DLinearForecast.Config)
    assert after.method.updates == 2400
    after.method.updates = before.method.updates
    after.experiment_name = before.experiment_name
    assert after == before


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


def test_calibration_fork_changes_only_the_method() -> None:
    before = exp002()
    after = exp003()
    assert after.method == CalibratedSimulator.Config()
    after.method = before.method
    after.experiment_name = before.experiment_name
    assert after == before


def test_drag_fork_changes_only_the_reference() -> None:
    before = exp003()
    after = exp004()
    assert isinstance(after.data.reference, CorrectedOscillator.Config)
    assert len(after.data.reference.corrections) == 1
    assert after.data.reference.corrections[0].coefficient == 0.35
    after.data.reference = before.data.reference
    after.experiment_name = before.experiment_name
    assert after == before


@pytest.mark.parametrize("number", [5, 7, 8])
def test_harder_controls_change_only_the_method(number: int) -> None:
    before = exp004()
    after = {5: exp005, 7: exp007, 8: exp008}[number]()
    expected = {
        5: SparseSimulator.Config(),
        7: exp000().method,
        8: RidgeCorrection.Config(),
    }[number]
    assert after.method == expected
    after.method = before.method
    after.experiment_name = before.experiment_name
    assert after == before


def test_ablation_removes_only_the_quadratic_drag_candidate() -> None:
    before, after = exp005(), exp006()
    assert isinstance(before.method, SparseSimulator.Config)
    assert isinstance(after.method, SparseSimulator.Config)
    assert [term.make().name for term in after.method.terms] == [
        term.make().name
        for term in before.method.terms
        if term.make().name != "-velocity*abs(velocity)^1"
    ]
    after.method.terms = before.method.terms
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


@pytest.mark.parametrize("number", [10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 24])
def test_delayed_controls_change_only_the_method(number: int) -> None:
    parents = {
        10: (experiments.exp009, experiments.exp010),
        11: (experiments.exp009, experiments.exp011),
        12: (experiments.exp009, experiments.exp012),
        13: (experiments.exp010, experiments.exp013),
        14: (experiments.exp009, experiments.exp014),
        15: (experiments.exp009, experiments.exp015),
        18: (experiments.exp015, experiments.exp018),
        20: (experiments.exp019, experiments.exp020),
        21: (experiments.exp019, experiments.exp021),
        22: (experiments.exp019, experiments.exp022),
        23: (experiments.exp019, experiments.exp023),
        24: (experiments.exp019, experiments.exp024),
    }
    expected = {
        10: RidgeCorrection.Config(),
        11: CalibratedSimulator.Config(),
        12: SparseSimulator.Config(),
        13: KernelCorrection.Config(),
        14: NeuralForecast.Config(),
        15: _known_equation(experiments.exp009()),
        18: HistoryAblation.Config(
            method=_known_equation(experiments.exp009()),
            visible_steps=32,
        ),
        20: RidgeCorrection.Config(),
        21: KernelCorrection.Config(),
        22: SparseSimulator.Config(),
        23: NeuralForecast.Config(),
        24: _known_equation(experiments.exp019()),
    }[number]
    parent, child = parents[number]
    before, after = parent(), child()
    assert after.method == expected
    after.method = before.method
    after.experiment_name = before.experiment_name
    assert after == before


@pytest.mark.parametrize("number", [16, 25])
def test_longer_training_fork_changes_only_update_budget(number: int) -> None:
    parent, child = {
        16: (experiments.exp014, experiments.exp016),
        25: (experiments.exp023, experiments.exp025),
    }[number]
    before, after = parent(), child()
    assert isinstance(before.method, NeuralForecast.Config)
    assert isinstance(after.method, NeuralForecast.Config)
    assert after.method.updates == before.method.updates * 10
    after.method.updates = before.method.updates
    after.experiment_name = before.experiment_name
    assert after == before


@pytest.mark.parametrize("number", [17, 27])
def test_more_data_fork_changes_only_the_reference_sample_counts(number: int) -> None:
    parent, child = {
        17: (experiments.exp016, experiments.exp017),
        27: (experiments.exp025, experiments.exp027),
    }[number]
    before, after = parent(), child()
    assert after.data.train.count == before.data.train.count * 8
    assert after.data.validation.count == before.data.validation.count * 8
    after.data.train.count = before.data.train.count
    after.data.validation.count = before.data.validation.count
    after.experiment_name = before.experiment_name
    assert after == before


def test_four_way_task_changes_only_data_and_keeps_independent_tuple_counts() -> None:
    before, after = experiments.exp009(), experiments.exp019()
    assert isinstance(after.data, DelayedData.Config)
    assert isinstance(after.data.history, SignedHistory.Config)
    groups = 1 << len(after.data.history.pulses)
    assert after.data.train.count // groups == before.data.train.count // 4
    assert after.data.validation.count // groups == before.data.validation.count // 4
    after.data = before.data
    after.experiment_name = before.experiment_name
    assert after == before


def test_quartic_control_changes_only_the_kernel_degree() -> None:
    before, after = experiments.exp021(), experiments.exp026()
    assert isinstance(before.method, KernelCorrection.Config)
    assert isinstance(after.method, KernelCorrection.Config)
    assert isinstance(before.method.kernel, PolynomialKernel.Config)
    assert isinstance(after.method.kernel, PolynomialKernel.Config)
    assert after.method.kernel.degree == 4
    after.method.kernel.degree = before.method.kernel.degree
    after.experiment_name = before.experiment_name
    assert after == before


@pytest.mark.parametrize("number", [28, 29])
def test_gate_capacity_controls_change_only_the_block(number: int) -> None:
    before = experiments.exp025()
    after = {28: experiments.exp028, 29: experiments.exp029}[number]()
    assert isinstance(before.method, NeuralForecast.Config)
    assert isinstance(after.method, NeuralForecast.Config)
    biased = before.method.step.model.block.copy_tree()
    assert isinstance(biased, SwiGLU.Config)
    biased.bias = True
    expected = {
        28: biased,
        29: Sequential.Config(
            elements=[
                SwiGLU.Config(
                    channels_hidden=64,
                    channels_out=64,
                    init_weight_out=unit_fan_in_uniform,
                ),
                SwiGLU.Config(channels_hidden=64),
            ],
        ),
    }[number]
    assert after.method.step.model.block == expected
    after.method.step.model.block = before.method.step.model.block
    after.experiment_name = before.experiment_name
    assert after == before


def _known_equation(task: DelayedStudy.Config) -> DelayedCalibration.Config:
    """Return the privileged control fitted to ``task``'s own delay equation."""
    assert isinstance(task.data.reference, DelayedOscillator.Config)
    simulator = task.data.reference.copy_tree()
    simulator.gain = 0.0
    return DelayedCalibration.Config(simulator=simulator)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
