"""Freeze the complete public recipe and its printable configuration tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import ast
import inspect
import itertools
import math

from priml.baselines.motor_imagery import experiments
from priml.baselines.motor_imagery.campaign import MotorImageryCampaign
from priml.baselines.motor_imagery.data import (
    NUM_CLASSES,
    Preprocessing,
    WindowData,
)
from priml.baselines.motor_imagery.experiments import exp000, exp_smoke
from priml.baselines.motor_imagery.folds import FixedCohort, GroupedFolds
from priml.baselines.motor_imagery.metric import EEGAccuracy
from priml.baselines.motor_imagery.model import (
    AttentionEEGNet,
    MultiScaleEEGNet,
    ShallowConvNet,
)
from priml.baselines.motor_imagery.train_loop import MotorImageryLoop
from priml.baselines.motor_imagery.train_step import (
    MotorImageryStep,
    cross_entropy_off_target,
)
from priml.loss.simple_loss import SimpleLoss
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.runtime import CudnnSingleProcess
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from functools import partial

    from torch import Tensor


def test_exp000_is_the_full_public_ensemble_and_validation_recipe() -> None:
    cfg = exp000().copy_tree().finalize()
    assert isinstance(cfg.folds, GroupedFolds.Config)
    assert cfg.folds.count == 5
    assert cfg.validation.max_epochs == 18
    assert cfg.ensemble_loop.max_epochs == 16
    assert cfg.seed == 42
    assert cfg.expected_submission_rows == 360
    assert cfg.validation.dataset.batch_size == 64
    assert cfg.ensemble_loop.dataset.batch_size == 64
    assert cfg.validation.dataset.preprocessing.window_size == 500
    assert cfg.validation.dataset.preprocessing.window_step == 125
    assert cfg.budget.seconds == float("inf")
    assert [type(model) for model in cfg.ensemble] == [
        MultiScaleEEGNet.Config,
        ShallowConvNet.Config,
        AttentionEEGNet.Config,
    ]
    assert isinstance(cfg.validation.step.model, MultiScaleEEGNet.Config)


def test_exp000_declares_both_loops_with_their_own_horizons() -> None:
    cfg = exp000().copy_tree().finalize()
    for loop, epochs in ((cfg.validation, 18), (cfg.ensemble_loop, 16)):
        schedule = loop.step.lr_schedule
        assert isinstance(schedule, RecursiveCosine.Config)
        assert loop.max_epochs == loop.step.train_budget_epochs == schedule.T_max
        assert loop.max_epochs == epochs
        assert loop.num_steps_eval == math.inf
        assert loop.checkpointer is None
        assert loop.study_name == "motor_imagery"
        assert loop.experiment_name == "exp000"
        assert str(loop.working_dir) == "/opt/scratch/runs/motor_imagery/exp000/fits"
    assert cfg.validation.eval_every_epoch
    assert not cfg.ensemble_loop.eval_every_epoch
    assert isinstance(cfg.validation.metrics_eval[""], EEGAccuracy.Config)
    assert not cfg.ensemble_loop.metrics_eval
    assert cfg.validation.runtime == cfg.ensemble_loop.runtime
    assert isinstance(cfg.validation.runtime, CudnnSingleProcess.Config)
    assert cfg.validation.runtime.cudnn_deterministic
    assert cfg.validation.runtime.cudnn_benchmark is False
    assert (
        str(cfg.validation.dataset.working_dir) == "/opt/scratch/datasets/motor-imagery"
    )


def test_exp000_states_the_recipe_that_baseline_class_defaults_leave_neutral() -> None:
    cfg = exp000()
    step = MotorImageryStep.Config()
    assert MotorImageryLoop.Config().max_epochs == math.inf
    assert step.train_budget_epochs == math.inf
    assert step.gradient_clip_norm == math.inf
    assert Preprocessing.Config().window_size == 0
    assert WindowData.Config().batch_size == 1
    assert MotorImageryCampaign.Config().ensemble == []
    assert cfg.validation.step.gradient_clip_norm == 1.0
    assert cfg.validation.step.compile is None
    assert cfg.validation.dataset.preprocessing.window_size == 500


def test_exp006_retains_the_complete_historical_five_minute_candidate() -> None:
    cfg = experiments.exp006().finalized()
    assert cfg.experiment_name == "exp006"
    assert str(cfg.working_dir) == "/opt/scratch/runs/motor_imagery/exp006"
    assert isinstance(cfg.folds, FixedCohort.Config)
    assert cfg.folds.subjects == ["S002", "S006", "S019"]
    assert cfg.validation.dataset.preprocessing.window_step == 250
    for model in [cfg.validation.step.model, *cfg.ensemble]:
        assert isinstance(model, MultiScaleEEGNet.Config)
        assert model.channels_hidden == 4
        assert model.channels_projection == 8
    assert cfg.budget.seconds == 300
    assert cfg.validation.max_epochs == cfg.ensemble_loop.max_epochs == 3
    assert len(cfg.ensemble) == 1
    assert (
        cfg.validation.dataset.preprocessing.band_hz
        == exp000().validation.dataset.preprocessing.band_hz
    )


def test_exp001_changes_only_the_fixed_holdout_protocol() -> None:
    parent, fork = exp000(), experiments.exp001()
    parent.experiment_name = fork.experiment_name
    parent.folds = fork.folds
    assert parent.udiff(fork, mode="fields", finalize=False, color=False) == ""


def test_exp002_changes_only_complete_fit_duration() -> None:
    parent, fork = experiments.exp001(), experiments.exp002()
    parent.experiment_name = fork.experiment_name
    for loop in (parent.validation, parent.ensemble_loop):
        loop.max_epochs = loop.step.train_budget_epochs = 3
        schedule = loop.step.lr_schedule
        assert isinstance(schedule, RecursiveCosine.Config)
        schedule.T_max = 3
    assert parent.udiff(fork, mode="fields", finalize=False, color=False) == ""


def test_exp003_changes_only_temporal_stride() -> None:
    parent, fork = experiments.exp002(), experiments.exp003()
    parent.experiment_name = fork.experiment_name
    parent.validation.dataset.preprocessing.window_step = 250
    assert parent.udiff(fork, mode="fields", finalize=False, color=False) == ""


def test_exp004_changes_only_ensemble_membership() -> None:
    parent, fork = experiments.exp003(), experiments.exp004()
    parent.experiment_name = fork.experiment_name
    parent.ensemble = parent.ensemble[:1]
    assert parent.udiff(fork, mode="fields", finalize=False, color=False) == ""


def test_exp005_changes_only_the_coupled_model_widths() -> None:
    parent, fork = experiments.exp004(), experiments.exp005()
    parent.experiment_name = fork.experiment_name
    for model in [parent.validation.step.model, *parent.ensemble]:
        assert isinstance(model, MultiScaleEEGNet.Config)
        model.channels_hidden = 4
        model.channels_projection = 8
    assert parent.udiff(fork, mode="fields", finalize=False, color=False) == ""


def test_exp006_changes_only_the_outer_allowance() -> None:
    parent, fork = experiments.exp005(), experiments.exp006()
    parent.experiment_name = fork.experiment_name
    parent.budget.seconds = 300
    assert parent.udiff(fork, mode="fields", finalize=False, color=False) == ""


def test_every_fork_calls_its_named_parent() -> None:
    recipes = [
        exp000,
        experiments.exp001,
        experiments.exp002,
        experiments.exp003,
        experiments.exp004,
        experiments.exp005,
        experiments.exp006,
    ]
    for parent, fork in itertools.pairwise(recipes):
        tree = ast.parse(inspect.getsource(fork))
        assert (
            sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == parent.__name__
                for node in ast.walk(tree)
            )
            == 1
        )


def test_smoke_retains_every_architecture_and_all_numerical_choices() -> None:
    cfg = exp_smoke().copy_tree().finalize()
    assert isinstance(cfg.folds, GroupedFolds.Config)
    assert cfg.folds.count == 2
    assert cfg.validation.max_epochs == cfg.ensemble_loop.max_epochs == 3
    assert len(cfg.ensemble) == 3
    for model in [cfg.validation.step.model, *cfg.ensemble]:
        assert isinstance(
            model,
            (MultiScaleEEGNet.Config, ShallowConvNet.Config, AttentionEEGNet.Config),
        )
        assert model.num_samples == 100
        assert model.channels_hidden == 2
    parent = exp000().validation.step
    assert cfg.validation.step.optimizer == parent.optimizer
    loss = cfg.validation.step.loss
    assert isinstance(loss, SimpleLoss.Config)
    objective = cast("partial[Tensor]", loss.loss_fn)
    assert objective.func is cross_entropy_off_target
    assert objective.keywords == {"smoothing": 0.05}
    assert loss.channels_out == NUM_CLASSES
    assert "Not a result" in (exp_smoke.__doc__ or "")


def test_exp000_complete_config_matches_printable_golden() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
