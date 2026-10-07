"""Tests for the Craftax experiment configs.

Two kinds of assertion, and the distinction matters:

* The fields ``exp000`` pins are checked by value. It is the control every
  fork is measured against, so a change to it invalidates published results;
  the test exists to make that change deliberate rather than incidental.
* Each fork is checked as a DELTA -- exactly which fields differ from its
  parent. That is what keeps "one named change per experiment" true as a
  property of the code rather than a claim in a docstring.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from importlib import util
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import re

from configgle import InlineConfig

import pytest

from priml.baselines.craftax import experiments
from priml.baselines.craftax.data import CraftaxRollouts
from priml.baselines.craftax.experiments import (
    CraftaxGTrXLTrainLoop,
    CraftaxPQNTrainLoop,
    CraftaxRNNTrainLoop,
    CraftaxTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
    exp011,
    exp013,
    exp_smoke,
)
from priml.baselines.craftax.gtrxl_train_step import CraftaxGTrXLTrainStep
from priml.baselines.craftax.metric import CraftaxScore
from priml.baselines.craftax.model import ActorCritic
from priml.baselines.craftax.pqn_train_step import CraftaxPQNTrainStep
from priml.baselines.craftax.rnn_train_step import CraftaxRNNTrainStep
from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.lib.absent import ABSENT
from priml.runtime import SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.parallelism import NoParallel
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from priml.testing.experiments import ExperimentFactory


type _CraftaxLoop = (
    CraftaxTrainLoop | CraftaxGTrXLTrainLoop | CraftaxRNNTrainLoop | CraftaxPQNTrainLoop
)

ALL_EXPERIMENTS: Final[list[ExperimentFactory[_CraftaxLoop]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp011,
    exp013,
    exp_smoke,
]

PUBLISHED_EXPERIMENTS: Final[list[ExperimentFactory[_CraftaxLoop]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp011,
    exp013,
]


def shrink(config: _CraftaxLoop) -> _CraftaxLoop:
    """Narrow ``config`` to a size a CPU test can run, preserving its recipe.

    Only SIZE changes -- workers, rollout, widths, horizon. Objective,
    coefficients, and schedule stay exactly as the experiment set them, so
    what runs here is the published recipe at minimum scale.
    """
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.device = "cpu"
    config.step.parallelism = NoParallel.Config(device="cpu")
    config.step.env.device = "cpu"
    config.step.env.num_envs = 2
    config.step.env.optimistic_reset_ratio = 1
    # A 3x3 view, not the benchmark's 9x11. The observation is one one-hot
    # vector per visible tile, so the window sets the input width -- 798 floats
    # against 8,268 -- and every layer, gradient, and optimizer moment scales
    # with it. What is under test is that the RECIPE runs end to end, which a
    # smaller window exercises identically.
    config.step.env.view = (3, 5)
    config.step.compile = None
    config.step.num_minibatches = 1
    config.step.num_epochs = 1
    config.step.total_train_steps = 2

    model = config.step.model
    model.channels_in = 8
    if isinstance(config.step, CraftaxGTrXLTrainStep.Config):
        config.step.rollout_steps = 2
        config.step.gradient_window = 1
        config.step.model.embed_dim = 8
        config.step.model.num_heads = 2
        config.step.model.num_layers = 1
        config.step.model.qkv_dim = 8
        config.step.model.memory_length = 2
    elif isinstance(
        config.step,
        CraftaxRNNTrainStep.Config | CraftaxPQNTrainStep.Config,
    ):
        # Neither recurrence has depth to shrink: the state is its width.
        config.step.rollout_steps = 2
    else:
        assert type(config.step) is CraftaxTrainStep.Config
        train_step = config.step
        train_step.rollout_steps = 2
        assert isinstance(train_step.model, ActorCritic.Config)
        train_step.model.num_layers = 1

    config.base_dir = None
    config.dataset.updates_per_epoch = 2
    config.max_steps = 2
    config.max_epochs = 1
    config.num_steps_eval = 2
    config.checkpointer = None
    config.tracker = None
    score = config.metrics_eval["craftax"]
    assert isinstance(score, CraftaxScore.Config)
    score.num_envs = 2
    score.steps = 2
    score.device = "cpu"
    # The metric plays its own episodes, so its window has to match the one
    # the policy trained on -- a different width is an input the network
    # cannot read.
    score.view = config.step.env.view
    return config


def test_experiment_binds_the_craftax_step_and_dataset() -> None:
    """The two slots carry their concrete types straight from the defaults.

    That binding is what lets a factory write ``cfg.step.model`` without an
    ``isinstance`` narrow. Drop the field defaults and the slots silently
    revert to the base loop's ``TrainStep`` and ``DummyDataset``, which
    type-checks and then fails at run time.
    """
    cfg = CraftaxTrainLoop()
    assert isinstance(cfg.step, CraftaxTrainStep.Config)
    assert isinstance(cfg.dataset, CraftaxRollouts.Config)

    recurrent = CraftaxGTrXLTrainLoop()
    assert isinstance(recurrent.step, CraftaxGTrXLTrainStep.Config)
    assert isinstance(recurrent.dataset, CraftaxRollouts.Config)


def test_experiment_makes_a_train_loop() -> None:
    assert CraftaxTrainLoop.parent_class is TrainLoop
    assert CraftaxGTrXLTrainLoop.parent_class is TrainLoop


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_finalizes(factory: ExperimentFactory[_CraftaxLoop]) -> None:
    assert factory().copy_tree().finalize() is not None


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_name_matches_the_factory(
    factory: ExperimentFactory[_CraftaxLoop],
) -> None:
    # The launcher derives run identity, and therefore the output directory,
    # from this field; a mismatch silently writes one run over another.
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_the_network_is_sized_from_the_environment(
    factory: ExperimentFactory[_CraftaxLoop],
) -> None:
    # An experiment must not have to remember the observation width; the
    # environment renders it and ``finalize`` propagates it.
    step = factory().step.copy_tree().finalize()
    assert step.model.observation_size == 8_268
    assert step.model.num_actions == 43


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_the_schedule_horizon_matches_the_step_budget(
    factory: ExperimentFactory[_CraftaxLoop],
) -> None:
    # A horizon shorter than the run leaves the tail at zero learning rate;
    # longer, and the run stops before the schedule anneals.
    cfg = factory()
    assert cfg.step.total_train_steps == cfg.max_steps


@pytest.mark.parametrize(
    "factory",
    PUBLISHED_EXPERIMENTS,
    ids=[f.__name__ for f in PUBLISHED_EXPERIMENTS],
)
def test_every_published_experiment_resets_optimistically(
    factory: ExperimentFactory[_CraftaxLoop],
) -> None:
    # The reference baseline sets this on every run; it is a throughput
    # treatment, so a fork that dropped it would be slower without saying so.
    assert factory().step.env.optimistic_reset_ratio == 16


@pytest.mark.parametrize(
    "factory",
    PUBLISHED_EXPERIMENTS,
    ids=[f.__name__ for f in PUBLISHED_EXPERIMENTS],
)
def test_every_published_experiment_scores_identically(
    factory: ExperimentFactory[_CraftaxLoop],
) -> None:
    # Two runs are comparable only when their evaluation geometry matches, so
    # a fork that changed it would be reporting on a different benchmark.
    score = factory().metrics_eval["craftax"]
    assert isinstance(score, CraftaxScore.Config)
    assert (score.num_envs, score.steps, score.seed) == (64, 10_000, 42)


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
@pytest.mark.compute_training
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[_CraftaxLoop],
) -> None:
    """Each published recipe runs end to end, shrunk to test size."""
    loop = shrink(factory()).make()
    loop.train()
    assert loop.step.global_step == 2


@pytest.mark.compute_training
def test_metric_only_evaluation_runs_no_discarded_trainer_rollout() -> None:
    loop = shrink(exp000()).make()

    assert isinstance(loop.step, CraftaxTrainStep)
    with patch.object(loop.step, "collect", wraps=loop.step.collect) as collect:
        metrics = loop.eval()

    assert collect.call_count == 0
    assert "craftax_episodes" in metrics


def test_exp000_pins_the_baseline_recipe() -> None:
    """exp000 is frozen: a change here invalidates every measured comparison."""
    cfg = exp000()
    assert cfg.step.env.num_envs == 256
    assert cfg.step.rollout_steps == 16
    assert cfg.step.num_epochs == 4
    assert cfg.step.num_minibatches == 8
    assert cfg.step.learning_rate == 3e-4
    assert cfg.step.anneal_learning_rate is True
    assert cfg.step.discount == 0.99
    assert cfg.step.trace_decay == 0.8
    assert cfg.step.clip_epsilon == 0.2
    assert cfg.step.entropy_coefficient == 0.01
    assert cfg.step.value_coefficient == 0.5
    assert cfg.step.max_grad_norm == 1.0
    assert cfg.step.env.optimistic_reset_ratio == 16
    assert cfg.step.model.channels_in == 512
    assert cfg.step.model.num_layers == 3
    assert cfg.step.seed == cfg.step.env.seed == 42


def test_exp000_spends_one_million_interactions() -> None:
    # The benchmark compares interaction budgets, not update counts, so the
    # conversion is what makes the run comparable at all.
    cfg = exp000()
    spent = cfg.max_steps * cfg.step.env.num_envs * cfg.step.rollout_steps
    assert spent == 999_424
    assert cfg.step.env.seed == 42
    assert cfg.step.seed == 42


def test_exp000_pins_every_recipe_field() -> None:
    cfg = exp000()
    score = cfg.metrics_eval["craftax"]
    assert isinstance(score, CraftaxScore.Config)
    assert {
        "study_name": cfg.study_name,
        "experiment_name": cfg.experiment_name,
        "step.env.num_envs": cfg.step.env.num_envs,
        "step.env.seed": cfg.step.env.seed,
        "step.env.optimistic_reset_ratio": cfg.step.env.optimistic_reset_ratio,
        "step.rollout_steps": cfg.step.rollout_steps,
        "step.num_epochs": cfg.step.num_epochs,
        "step.num_minibatches": cfg.step.num_minibatches,
        "step.learning_rate": cfg.step.learning_rate,
        "step.anneal_learning_rate": cfg.step.anneal_learning_rate,
        "step.discount": cfg.step.discount,
        "step.trace_decay": cfg.step.trace_decay,
        "step.clip_epsilon": cfg.step.clip_epsilon,
        "step.entropy_coefficient": cfg.step.entropy_coefficient,
        "step.value_coefficient": cfg.step.value_coefficient,
        "step.max_grad_norm": cfg.step.max_grad_norm,
        "step.seed": cfg.step.seed,
        "step.model.channels_in": cfg.step.model.channels_in,
        "step.model.num_layers": cfg.step.model.num_layers,
        "max_steps": cfg.max_steps,
        "step.total_train_steps": cfg.step.total_train_steps,
        "dataset.updates_per_epoch": cfg.dataset.updates_per_epoch,
        "num_steps_eval": cfg.num_steps_eval,
        "metrics_eval.craftax.num_envs": score.num_envs,
        "metrics_eval.craftax.steps": score.steps,
        "metrics_eval.craftax.seed": score.seed,
    } == {
        "study_name": "craftax",
        "experiment_name": "exp000",
        "step.env.num_envs": 256,
        "step.env.seed": 42,
        "step.env.optimistic_reset_ratio": 16,
        "step.rollout_steps": 16,
        "step.num_epochs": 4,
        "step.num_minibatches": 8,
        "step.learning_rate": 3e-4,
        "step.anneal_learning_rate": True,
        "step.discount": 0.99,
        "step.trace_decay": 0.8,
        "step.clip_epsilon": 0.2,
        "step.entropy_coefficient": 0.01,
        "step.value_coefficient": 0.5,
        "step.max_grad_norm": 1.0,
        "step.seed": 42,
        "step.model.channels_in": 512,
        "step.model.num_layers": 3,
        "max_steps": 244,
        "step.total_train_steps": 244,
        "dataset.updates_per_epoch": 244,
        "num_steps_eval": 244,
        "metrics_eval.craftax.num_envs": 64,
        "metrics_eval.craftax.steps": 10_000,
        "metrics_eval.craftax.seed": 42,
    }


def test_exp002_pins_every_recipe_field() -> None:
    cfg = exp002()
    assert {
        "study_name": cfg.study_name,
        "experiment_name": cfg.experiment_name,
        "step.env.num_envs": cfg.step.env.num_envs,
        "step.env.seed": cfg.step.env.seed,
        "step.env.optimistic_reset_ratio": cfg.step.env.optimistic_reset_ratio,
        "step.rollout_steps": cfg.step.rollout_steps,
        "step.num_epochs": cfg.step.num_epochs,
        "step.num_minibatches": cfg.step.num_minibatches,
        "step.learning_rate": cfg.step.learning_rate,
        "step.anneal_learning_rate": cfg.step.anneal_learning_rate,
        "step.discount": cfg.step.discount,
        "step.trace_decay": cfg.step.trace_decay,
        "step.clip_epsilon": cfg.step.clip_epsilon,
        "step.entropy_coefficient": cfg.step.entropy_coefficient,
        "step.value_coefficient": cfg.step.value_coefficient,
        "step.max_grad_norm": cfg.step.max_grad_norm,
        "step.seed": cfg.step.seed,
        "step.model.channels_in": cfg.step.model.channels_in,
        "max_steps": cfg.max_steps,
        "step.total_train_steps": cfg.step.total_train_steps,
        "dataset.updates_per_epoch": cfg.dataset.updates_per_epoch,
        "num_steps_eval": cfg.num_steps_eval,
        "runtime": cfg.runtime,
    } == {
        "study_name": "craftax",
        "experiment_name": "exp002",
        "step.env.num_envs": 1_024,
        "step.env.seed": 42,
        "step.env.optimistic_reset_ratio": 16,
        "step.rollout_steps": 64,
        "step.num_epochs": 4,
        "step.num_minibatches": 8,
        "step.learning_rate": 2e-4,
        "step.anneal_learning_rate": True,
        "step.discount": 0.99,
        "step.trace_decay": 0.8,
        "step.clip_epsilon": 0.2,
        "step.entropy_coefficient": 0.01,
        "step.value_coefficient": 0.5,
        "step.max_grad_norm": 1.0,
        "step.seed": 42,
        "step.model.channels_in": 512,
        "max_steps": 15_258,
        "step.total_train_steps": 15_258,
        "dataset.updates_per_epoch": 15_258,
        "num_steps_eval": 15_258,
        "runtime": SingleProcess.Config(),
    }


def test_exp003_pins_every_recipe_field() -> None:
    cfg = exp003()
    assert {
        "study_name": cfg.study_name,
        "experiment_name": cfg.experiment_name,
        "step.env.num_envs": cfg.step.env.num_envs,
        "step.env.seed": cfg.step.env.seed,
        "step.env.optimistic_reset_ratio": cfg.step.env.optimistic_reset_ratio,
        "step.rollout_steps": cfg.step.rollout_steps,
        "step.num_epochs": cfg.step.num_epochs,
        "step.num_minibatches": cfg.step.num_minibatches,
        "step.learning_rate": cfg.step.learning_rate,
        "step.anneal_learning_rate": cfg.step.anneal_learning_rate,
        "step.discount": cfg.step.discount,
        "step.trace_decay": cfg.step.trace_decay,
        "step.epsilon_start": cfg.step.epsilon_start,
        "step.epsilon_finish": cfg.step.epsilon_finish,
        "step.epsilon_decay_fraction": cfg.step.epsilon_decay_fraction,
        "step.max_grad_norm": cfg.step.max_grad_norm,
        "step.seed": cfg.step.seed,
        "step.model.channels_in": cfg.step.model.channels_in,
        "max_steps": cfg.max_steps,
        "step.total_train_steps": cfg.step.total_train_steps,
        "dataset.updates_per_epoch": cfg.dataset.updates_per_epoch,
        "num_steps_eval": cfg.num_steps_eval,
        "runtime": cfg.runtime,
    } == {
        "study_name": "craftax",
        "experiment_name": "exp003",
        "step.env.num_envs": 1_024,
        "step.env.seed": 42,
        "step.env.optimistic_reset_ratio": 16,
        "step.rollout_steps": 128,
        "step.num_epochs": 4,
        "step.num_minibatches": 4,
        "step.learning_rate": 3e-4,
        "step.anneal_learning_rate": True,
        "step.discount": 0.99,
        "step.trace_decay": 0.5,
        "step.epsilon_start": 1.0,
        "step.epsilon_finish": 0.005,
        "step.epsilon_decay_fraction": 0.1,
        "step.max_grad_norm": 0.5,
        "step.seed": 42,
        "step.model.channels_in": 512,
        "max_steps": 7_629,
        "step.total_train_steps": 7_629,
        "dataset.updates_per_epoch": 7_629,
        "num_steps_eval": 7_629,
        "runtime": SingleProcess.Config(),
    }


def test_exp011_pins_every_recipe_field() -> None:
    cfg = exp011()
    assert {
        "experiment_name": cfg.experiment_name,
        "step.env.num_envs": cfg.step.env.num_envs,
        "step.rollout_steps": cfg.step.rollout_steps,
        "step.learning_rate": cfg.step.learning_rate,
        "max_steps": cfg.max_steps,
        "step.total_train_steps": cfg.step.total_train_steps,
        "dataset.updates_per_epoch": cfg.dataset.updates_per_epoch,
        "num_steps_eval": cfg.num_steps_eval,
    } == {
        "experiment_name": "exp011",
        "step.env.num_envs": 1_024,
        "step.rollout_steps": 64,
        "step.learning_rate": 2e-4,
        "max_steps": 1_525,
        "step.total_train_steps": 1_525,
        "dataset.updates_per_epoch": 1_525,
        "num_steps_eval": 1_525,
    }


def test_exp013_pins_every_recipe_field() -> None:
    cfg = exp013()
    assert {
        "study_name": cfg.study_name,
        "experiment_name": cfg.experiment_name,
        "step.env.num_envs": cfg.step.env.num_envs,
        "step.env.seed": cfg.step.env.seed,
        "step.env.optimistic_reset_ratio": cfg.step.env.optimistic_reset_ratio,
        "step.rollout_steps": cfg.step.rollout_steps,
        "step.gradient_window": cfg.step.gradient_window,
        "step.num_epochs": cfg.step.num_epochs,
        "step.num_minibatches": cfg.step.num_minibatches,
        "step.learning_rate": cfg.step.learning_rate,
        "step.anneal_learning_rate": cfg.step.anneal_learning_rate,
        "step.discount": cfg.step.discount,
        "step.trace_decay": cfg.step.trace_decay,
        "step.clip_epsilon": cfg.step.clip_epsilon,
        "step.entropy_coefficient": cfg.step.entropy_coefficient,
        "step.value_coefficient": cfg.step.value_coefficient,
        "step.max_grad_norm": cfg.step.max_grad_norm,
        "step.seed": cfg.step.seed,
        "step.model.embed_dim": cfg.step.model.embed_dim,
        "step.model.num_heads": cfg.step.model.num_heads,
        "step.model.num_layers": cfg.step.model.num_layers,
        "step.model.qkv_dim": cfg.step.model.qkv_dim,
        "step.model.channels_in": cfg.step.model.channels_in,
        "step.model.memory_length": cfg.step.model.memory_length,
        "step.model.gating_bias": cfg.step.model.gating_bias,
        "max_steps": cfg.max_steps,
        "step.total_train_steps": cfg.step.total_train_steps,
        "dataset.updates_per_epoch": cfg.dataset.updates_per_epoch,
        "num_steps_eval": cfg.num_steps_eval,
        "runtime": cfg.runtime,
    } == {
        "study_name": "craftax",
        "experiment_name": "exp013",
        "step.env.num_envs": 1_024,
        "step.env.seed": 42,
        "step.env.optimistic_reset_ratio": 16,
        "step.rollout_steps": 128,
        "step.gradient_window": 64,
        "step.num_epochs": 4,
        "step.num_minibatches": 8,
        "step.learning_rate": 2e-4,
        "step.anneal_learning_rate": True,
        "step.discount": 0.999,
        "step.trace_decay": 0.8,
        "step.clip_epsilon": 0.2,
        "step.entropy_coefficient": 0.002,
        "step.value_coefficient": 0.5,
        "step.max_grad_norm": 1.0,
        "step.seed": 42,
        "step.model.embed_dim": 256,
        "step.model.num_heads": 8,
        "step.model.num_layers": 2,
        "step.model.qkv_dim": 256,
        "step.model.channels_in": 256,
        "step.model.memory_length": 128,
        "step.model.gating_bias": 2.0,
        "max_steps": 7_629,
        "step.total_train_steps": 7_629,
        "dataset.updates_per_epoch": 7_629,
        "num_steps_eval": 7_629,
        "runtime": SingleProcess.Config(),
    }


def test_exp_smoke_pins_every_recipe_field() -> None:
    cfg = exp_smoke()
    score = cfg.metrics_eval["craftax"]
    assert isinstance(score, CraftaxScore.Config)
    assert {
        "experiment_name": cfg.experiment_name,
        "step.env.num_envs": cfg.step.env.num_envs,
        "step.rollout_steps": cfg.step.rollout_steps,
        "step.num_minibatches": cfg.step.num_minibatches,
        "step.model.channels_in": cfg.step.model.channels_in,
        "step.model.num_layers": cfg.step.model.num_layers,
        "max_steps": cfg.max_steps,
        "step.total_train_steps": cfg.step.total_train_steps,
        "dataset.updates_per_epoch": cfg.dataset.updates_per_epoch,
        "num_steps_eval": cfg.num_steps_eval,
        "metrics_eval.craftax.num_envs": score.num_envs,
        "metrics_eval.craftax.steps": score.steps,
    } == {
        "experiment_name": "exp_smoke",
        "step.env.num_envs": 8,
        "step.rollout_steps": 4,
        "step.num_minibatches": 2,
        "step.model.channels_in": 32,
        "step.model.num_layers": 1,
        "max_steps": 4,
        "step.total_train_steps": 4,
        "dataset.updates_per_epoch": 4,
        "num_steps_eval": 4,
        "metrics_eval.craftax.num_envs": 4,
        "metrics_eval.craftax.steps": 64,
    }


def test_exp001_spends_one_billion_interactions() -> None:
    cfg = exp001()
    spent = cfg.max_steps * cfg.step.env.num_envs * cfg.step.rollout_steps
    assert spent == 999_948_288
    assert (cfg.step.env.num_envs, cfg.step.rollout_steps) == (1_024, 64)
    assert cfg.step.learning_rate == 2e-4
    assert cfg.dataset.updates_per_epoch == cfg.max_steps
    assert cfg.num_steps_eval == cfg.max_steps


def test_exp011_spends_one_hundred_million_interactions() -> None:
    cfg = exp011()
    spent = cfg.max_steps * cfg.step.env.num_envs * cfg.step.rollout_steps
    assert 99_000_000 <= spent <= 100_000_000


def test_exp002_spends_one_billion_interactions() -> None:
    cfg = exp002()
    spent = cfg.max_steps * cfg.step.env.num_envs * cfg.step.rollout_steps
    assert 999_000_000 <= spent <= 1_000_000_000


def test_exp002_changes_only_the_policy_class() -> None:
    # A GRU is the cheapest memory there is, so nothing else may move: any
    # gain has to be attributable to remembering, not to a re-tune.
    parent, child = exp001(), exp002()
    assert child.step.env.num_envs == parent.step.env.num_envs
    assert child.step.rollout_steps == parent.step.rollout_steps
    assert child.step.learning_rate == parent.step.learning_rate
    assert child.step.discount == parent.step.discount
    assert child.step.entropy_coefficient == parent.step.entropy_coefficient
    assert child.max_steps == parent.max_steps


def test_exp003_spends_one_billion_interactions() -> None:
    cfg = exp003()
    spent = cfg.max_steps * cfg.step.env.num_envs * cfg.step.rollout_steps
    assert 999_000_000 <= spent <= 1_000_000_000


def test_exp003_explores_by_schedule_rather_than_entropy() -> None:
    # The defining difference from every other experiment here: a Q-learner
    # has no entropy bonus, so the schedule is the whole of its exploration.
    cfg = exp003()
    assert cfg.step.epsilon_start == 1.0
    assert cfg.step.epsilon_finish == 0.005
    assert cfg.step.epsilon_decay_fraction == 0.1


def test_exp013_spends_one_billion_interactions() -> None:
    cfg = exp013()
    spent = cfg.max_steps * cfg.step.env.num_envs * cfg.step.rollout_steps
    assert 999_000_000 <= spent <= 1_000_000_000


def test_exp001_changes_only_the_geometry_and_budget() -> None:
    # The learning rate moves with the batch: four times the workers and four
    # times the rollout is sixteen times the data per update, so keeping
    # 3e-4 would not be the same recipe at a larger size.
    assert _deltas(exp000(), exp001()) == {
        "experiment_name",
        "step.env.num_envs",
        "step.rollout_steps",
        "step.learning_rate",
        "step.total_train_steps",
        "max_steps",
        "num_steps_eval",
        "dataset.updates_per_epoch",
    }


def test_exp011_changes_only_the_budget() -> None:
    # Annealing is budget-relative, so the horizon has to move with the
    # budget or the screen would measure a differently-scheduled run.
    assert _deltas(exp001(), exp011()) == {
        "experiment_name",
        "step.total_train_steps",
        "max_steps",
        "num_steps_eval",
        "dataset.updates_per_epoch",
    }


def test_exp013_changes_the_architecture_and_what_travels_with_it() -> None:
    # Memory is the change; the rollout length, discount, and entropy weight
    # are the settings that only make sense alongside it, so they move as one
    # treatment rather than four.
    deltas = _deltas(exp001(), exp013())
    assert "step" in deltas
    assert {
        "step.rollout_steps",
        "step.discount",
        "step.entropy_coefficient",
    } <= deltas
    # Untouched: the objective itself, so any difference is attributable to
    # the policy class rather than to a re-tuned loss.
    assert not deltas & {
        "step.clip_epsilon",
        "step.value_coefficient",
        "step.max_grad_norm",
        "step.num_epochs",
        "step.num_minibatches",
        "step.learning_rate",
        "step.trace_decay",
        "step.env.num_envs",
    }


@pytest.mark.parametrize(
    ("factory", "treatment"),
    [
        (exp002, set[str]()),
        (
            exp003,
            {
                # Q-learning's recipe: rollout, minibatches, optimizer, and
                # bootstrap target travel together, as the docstring states.
                "step.rollout_steps",
                "step.num_minibatches",
                "step.learning_rate",
                "step.trace_decay",
                "step.max_grad_norm",
            },
        ),
        (
            exp013,
            {
                # Memory and what only makes sense alongside it.
                "step.rollout_steps",
                "step.discount",
                "step.entropy_coefficient",
                "step.model.channels_in",
            },
        ),
    ],
    ids=["exp002", "exp003", "exp013"],
)
def test_cross_architecture_forks_keep_every_shared_parent_setting(
    factory: ExperimentFactory[_CraftaxLoop],
    treatment: set[str],
) -> None:
    """A fork that swaps the step class keeps every setting both steps share.

    Only the documented treatment, the new step class, and the budget it
    implies may differ from exp001; anything else is a re-tune smuggled in
    with the architecture.
    """
    parent = _flatten(exp001())
    child = _flatten(factory())
    budget = {
        "max_steps",
        "step.total_train_steps",
        "num_steps_eval",
        "dataset.updates_per_epoch",
    }
    moved = {
        name
        for name in parent.keys() & child.keys()
        if parent[name] != child[name]
        and not name.startswith("step.model.")
        and name not in {"experiment_name", "step", "step.model"}
    }
    assert moved <= treatment | budget
    assert {name for name in treatment if parent[name] == child[name]} == set()


class _RetunedExp001:
    """exp001 with every shared setting moved, to see which forks follow."""

    def __init__(self, parent: ExperimentFactory[CraftaxTrainLoop]) -> None:
        self.parent = parent

    def __call__(self) -> CraftaxTrainLoop:
        cfg = self.parent()
        cfg.study_name = "craftax-retuned"
        cfg.step.env.seed = 7
        cfg.step.env.optimistic_reset_ratio = 8
        cfg.step.num_epochs = 3
        cfg.step.anneal_learning_rate = False
        cfg.step.clip_epsilon = 0.3
        cfg.step.value_coefficient = 0.25
        cfg.step.seed = 7
        score = cfg.metrics_eval["craftax"]
        assert isinstance(score, CraftaxScore.Config)
        score.seed = 7
        return cfg


@pytest.mark.parametrize(
    "factory",
    [exp002, exp003, exp013],
    ids=["exp002", "exp003", "exp013"],
)
def test_cross_architecture_forks_inherit_parent_settings(
    factory: ExperimentFactory[_CraftaxLoop],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A setting exp001 owns reaches each fork without being retyped there."""
    monkeypatch.setattr(experiments, "exp001", _RetunedExp001(experiments.exp001))
    parent = _flatten(experiments.exp001())
    child = _flatten(factory())
    shared = {
        "study_name",
        "step.env.seed",
        "step.env.optimistic_reset_ratio",
        "step.num_epochs",
        "step.anneal_learning_rate",
        "step.clip_epsilon",
        "step.value_coefficient",
        "step.seed",
        "runtime",
    }
    assert {
        name for name in shared if name in child and child[name] != parent[name]
    } == set()
    score = factory().metrics_eval["craftax"]
    assert isinstance(score, CraftaxScore.Config)
    assert score.seed == 7


def test_exp013_windows_divide_its_rollout() -> None:
    # A ragged final window would receive gradients over a shorter context
    # than every other one.
    cfg = exp013()
    assert cfg.step.rollout_steps % cfg.step.gradient_window == 0


def test_forks_do_not_mutate_their_parent() -> None:
    # Each factory rebuilds its parent, so a fork's mutations must not leak
    # into a config the caller built earlier.
    baseline = exp000()
    before = _flatten(baseline)
    _ = exp013()
    _ = exp011()
    assert _flatten(baseline) == before


def test_smoke_is_marked_as_carrying_no_result() -> None:
    assert "Not a result" in (exp_smoke.__doc__ or "")


def test_module_docstring_lists_every_published_experiment() -> None:
    documented = experiments.__doc__ or ""
    for factory in PUBLISHED_EXPERIMENTS:
        assert factory.__name__ in documented


def test_the_launch_command_names_importable_targets() -> None:
    match = re.search(r"python -m (\S+) (\S+)", experiments.__doc__ or "")
    assert match is not None
    module, factory = match.group(1, 2)
    assert util.find_spec(str(module)) is not None
    owner, _, name = str(factory).rpartition(".")
    assert owner == experiments.__name__
    assert name in {published.__name__ for published in PUBLISHED_EXPERIMENTS}


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for child, parent in (
        (exp001, exp000),
        (exp002, exp001),
        (exp003, exp001),
        (exp011, exp001),
        (exp013, exp001),
    ):
        summary = (child.__doc__ or "").splitlines()[0]
        assert parent.__name__ in summary, child.__name__


def test_the_renamed_forks_record_what_they_dropped() -> None:
    """A name carried over from the JAX study must not imply a comparison.

    ``exp011`` and ``exp013`` there also selected a scanned evaluator and an
    adaptive reset pool -- XLA compilation treatments with no torch analogue.
    A receipt that quietly omitted them would read as a like-for-like result.
    """
    for factory in (exp011, exp013):
        assert "Not carried over from the JAX" in (factory.__doc__ or ""), (
            factory.__name__
        )


# A field holding a nested Config is descended into, so a change buried in
# ``step.model`` is reported at the leaf that actually moved -- unless the two sides
# hold DIFFERENT Config classes, in which case the swap itself is the single change and
# its fields are not comparable.
def _deltas(parent: object, child: object) -> set[str]:
    """Return the dotted names of the fields that differ between two configs."""
    flat_parent, flat_child = _flatten(parent), _flatten(child)
    return {
        name
        for name in flat_parent.keys() | flat_child.keys()
        if flat_parent.get(name, ABSENT) != flat_child.get(name, ABSENT)
    }


def _flatten(config: object, prefix: str = "") -> dict[str, object]:
    """Return a dotted-name to value map, descending into nested Configs."""
    if not is_dataclass(config):
        raise TypeError(f"Expected a dataclass, got {type(config).__name__}.")
    flat: dict[str, object] = {}
    for field in fields(config):
        name = field.name
        value: object = getattr(config, name)  # pyright: ignore[reportAny] -- Dataclass fields are dynamically accessed by name.
        dotted = f"{prefix}{name}"
        if isinstance(value, InlineConfig):
            flat[dotted] = repr(cast(InlineConfig[object], value))
        elif isinstance(value, list):
            flat[dotted] = [repr(item) for item in cast(list[object], value)]
        elif is_dataclass(value):
            # Record the class itself, so swapping in a different Config
            # registers as one change rather than a diff of every field.
            flat[dotted] = type(value)
            flat.update(_flatten(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def test_exp000_matches_its_golden_config() -> None:
    """Pin the WHOLE finalized ``exp000``: the control every fork is measured against."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
