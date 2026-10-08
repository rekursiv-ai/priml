"""Tests for the Craftax experiment configs."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import inspect
import math
import re
import threading
import warnings

from configgle import InlineConfig, PartialConfig
from torch.nn import functional

import pytest
import torch

from priml.baselines.craftax import experiments
from priml.baselines.craftax.env import CraftaxEnv, FreshWorlds, WorldPool
from priml.baselines.craftax.experiments import (
    CraftaxTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp100,
    exp101,
    exp102,
    exp103,
    exp104,
    exp105,
    exp106,
    exp107,
    exp108,
    exp109,
    exp110,
    exp111,
    exp112,
    exp113,
    exp_smoke,
)
from priml.baselines.craftax.game.state import (
    ACTION_OBS_SIZE,
    SYMBOLIC_OBS_SIZE,
)
from priml.baselines.craftax.learners.gtrxl_train_step import TrajectoryWindows
from priml.baselines.craftax.learners.imitation import BranchImitation
from priml.baselines.craftax.learners.pqn_train_step import CraftaxPQNTrainLoop
from priml.baselines.craftax.learners.rnn_update import ShuffledTrajectories
from priml.baselines.craftax.learners.update import ShuffledTransitions
from priml.baselines.craftax.lib.adam import ClippedAdam
from priml.baselines.craftax.model import (
    FeasibilityLoss,
    MinGRUPolicy,
    NoEncoder,
)
from priml.baselines.craftax.policies.actor_critic import ActorCritic
from priml.baselines.craftax.policies.encoder import BoardEncoder, ConvNextTrunk
from priml.baselines.craftax.policies.gtrxl import GTrXLPolicy
from priml.baselines.craftax.policies.pqn import (
    EpsilonGreedy,
    GreedySampler,
    PreviousAction,
)
from priml.baselines.craftax.policies.rnn import ActorCriticRNN
from priml.baselines.craftax.rollout import (
    PhiloxSampler,
    TorchPhiloxSampler,
)
from priml.baselines.craftax.testing import (
    assert_golden,
    digest,
    fp32,
    host_agnostic_pipeline,
    multi_hot_embedding,
    portable_uniform,
)
from priml.baselines.craftax.train_step import (
    AgentWindows,
    CraftaxTrainStep,
    ProgressSchedule,
    minibatch_count,
)
from priml.baselines.craftax.world_model.context import ContextReplay
from priml.baselines.craftax.world_model.feature import (
    DonorHistory,
    Flash4CacheAttention,
    FreshWindow,
    InitialWeights,
    MaskedCacheAttention,
    Refill,
    Sliding,
    TrainedWeights,
    WorldModelFeature,
)
from priml.lib.absent import ABSENT
from priml.loss.policy_gradient import TorchPPO
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.math.schedules import linear, warmup
from priml.model.attention.flash4 import Flash4Varlen
from priml.model.attention.kernel import SdpaVarlen
from priml.model.init import fan_in_truncated_normal, kaiming_uniform
from priml.model.min_gru import TorchScan, TritonScan
from priml.optimizers.fused_muon import FusedMuon
from priml.runtime import SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.checkpointer import Checkpointer
from priml.train.tracker import (
    AsyncTracker,
    FileTracker,
    TrackerList,
    WandbTracker,
)


if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from priml.baselines.craftax.model import Policy
    from priml.testing.experiments import ExperimentFactory


_WORLD_MODEL_EXP001: Final = "priml.baselines.craftax.world_model.experiments.exp001"
_ORACLE_CHECKPOINTS: Final = Path(
    "/opt/scratch/datasets/craftax/world-model-oracle/v1/checkpoints",
)

_PORTED: Final[
    tuple[ExperimentFactory[CraftaxTrainLoop | CraftaxPQNTrainLoop], ...]
] = (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp100,
    exp101,
    exp102,
    exp103,
    exp104,
    exp105,
    exp106,
    exp107,
    exp108,
    exp109,
    exp110,
    exp111,
    exp112,
    exp113,
    exp_smoke,
)
"""Every experiment with a recipe, each built and printed below."""


def test_every_experiment_is_ported() -> None:
    """Every factory the module defines is in the list the build tests take."""
    defined = {
        name
        for name, function in inspect.getmembers(experiments, inspect.isfunction)
        if name.startswith("exp") and function.__module__ == experiments.__name__
    }
    assert {factory.__name__ for factory in _PORTED} == defined


def test_the_readme_links_every_experiment_to_its_factory() -> None:
    """Every factory has a results row, linked to the module that defines it.

    The links name no line: the export re-sorts the imports into separate
    third-party and first-party blocks, so the public module's line numbers
    differ from this one's.
    """
    readme = (
        Path(experiments.__file__).with_name("README.md").read_text(encoding="utf-8")
    )
    linked = set(re.findall(r"\[`(exp\d+)`\]\(experiments\.py\)", readme))
    assert linked == {
        factory.__name__ for factory in _PORTED if factory is not exp_smoke
    }
    assert "experiments.py#L" not in readme


@pytest.mark.parametrize("factory", _PORTED)
def test_every_experiment_is_named_for_its_factory(
    factory: ExperimentFactory[CraftaxTrainLoop | CraftaxPQNTrainLoop],
) -> None:
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    "factory",
    [
        exp000,
        exp001,
        exp002,
        exp003,
        exp004,
        exp005,
        exp007,
        exp008,
        exp100,
        exp101,
        exp102,
        exp103,
        exp104,
        exp105,
        exp109,
        exp110,
        exp111,
        exp112,
        exp113,
        exp_smoke,
    ],
)
def test_every_experiment_finalizes(
    factory: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    config = factory().copy_tree().finalize()
    assert config.experiment_name == factory.__name__
    assert config.max_steps == config.step.train_budget_steps


@pytest.mark.parametrize("factory", [CraftaxTrainLoop, *_PORTED])
def test_every_config_finalizes_and_prints_without_a_warning(
    factory: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    """The class defaults too: their budget is unset, and nothing derives from it.

    On the base the dataset's cadence was ``int(step.train_budget_steps)``
    under a -1 sentinel, so finalizing the defaults raised ``OverflowError``
    and ``pprint`` fell back, with three warnings, to the unfinalized tree.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factory().copy_tree().finalize()
        factory().pformat()


@pytest.mark.parametrize("factory", [exp000, exp002, exp003])
def test_a_run_keeps_its_newest_two_checkpoints_of_one_per_200_epochs(
    factory: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    checkpointer = factory().checkpointer
    assert isinstance(checkpointer, Checkpointer.Config)
    assert (checkpointer.save_every, checkpointer.keep_last_n) == (200, 2)


def test_exp000_runs_pufferlibs_epochs() -> None:
    config = exp000()
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert config.max_steps == 6_662
    assert config.max_steps * transitions == 3_492_806_656


def test_exp000_states_craftax_ini_over_the_class_defaults() -> None:
    """exp000 states the recipe: each ``craftax.ini`` value differs from its default.

    The class defaults are PufferLib's ``default.ini``.
    """
    assert _deltas(CraftaxTrainLoop(), exp000()) >= {
        "step.checkpoint",
        "step.optimizer.lr",
        "step.optimizer.momentum",
        "step.optimizer.max_grad_norm",
        "step.learner.objective.discount",
        "step.learner.objective.trace_decay",
        "step.learner.objective.clip_epsilon",
        "step.learner.objective.value_clip_epsilon",
        "step.learner.objective.value_coefficient",
        "step.learner.objective.entropy_coefficient",
        "step.learner.minibatch_size",
        "step.learner.replay_ratio",
        "step.schedule",
        "step.train_budget_steps",
    }


def test_exp000_keeps_pufferlibs_bf16_where_the_class_defaults_keep_fp32() -> None:
    """Carry, decoder output, stored rollout values: fp32 by default, bf16 for its bits."""
    default = CraftaxTrainLoop()
    config = exp000()
    assert _deltas(default, config) >= {
        "step.model.state_dtype",
        "step.model.output_dtype",
        "step.rollout.dtype",
    }
    for loop, dtype in ((default, torch.float32), (config, torch.bfloat16)):
        model = loop.step.model
        assert isinstance(model, MinGRUPolicy.Config)
        assert (
            model.state_dtype == model.output_dtype == loop.step.rollout.dtype == dtype
        )
        assert model.dtype == torch.bfloat16


def test_exp001_changes_only_the_budget() -> None:
    assert exp001().max_steps == 476
    assert _deltas(exp000(), exp001()) == {
        "experiment_name",
        "max_steps",
        "step.train_budget_steps",
    }


def test_exp002_changes_the_setup_and_what_its_observation_needs() -> None:
    # The six options are one treatment (the Hypothesis says why); the dense
    # first stage and the policy's own init travel with the symbolic view. A
    # swapped slot also lists the fields its new class does not share.
    assert _deltas(exp000(), exp002()) == {
        "experiment_name",
        "step.env.rules.original_reward",
        "step.env.rules.end_on_boss_defeat",
        "step.env.rules.collapse_sleep",
        "step.env.rules.action_mask",
        "step.env.rules.symbolic_observation",
        "step.env.restart",
        "step.env.restart.num_worlds",
        "step.model.embedding",
        "step.model.embedding.channels_in",
        "step.model.embedding.channels_out",
        "step.model.embedding.offsets",
        "step.model.embedding.num_cells",
        "step.model.embedding.num_scalars",
        "step.model.embedding.dtype",
        "step.model.embedding.init_weight",
        "step.model.embedding.observation_size",
        "step.checkpoint",
    }


def test_every_experiment_seeds_torchs_generator_as_the_module_states() -> None:
    """PufferLib's 73, inherited from exp000; exp003's reference seed, 42, in its forks."""
    for factory in (
        exp000,
        exp001,
        exp002,
        exp100,
        exp101,
        exp102,
        exp103,
        exp104,
        exp105,
        exp106,
        exp107,
        exp108,
        exp109,
        exp110,
        exp111,
        exp112,
        exp113,
        exp_smoke,
    ):
        assert factory().seed == 73, factory.__name__
    for factory in (exp003, exp004, exp005, exp007):
        config = factory()
        learner = config.step.learner
        assert isinstance(learner, ShuffledTransitions.Config)
        sampler = config.step.sampler
        assert isinstance(sampler, PhiloxSampler.Config)
        seeds = {config.seed, config.step.env.seed, sampler.seed, learner.seed}
        assert seeds == {42}, factory.__name__


def test_exp100_changes_only_where_the_weights_come_from() -> None:
    """No checkpoint, so the policy draws its own init under exp000's seed."""
    assert _deltas(exp000(), exp100()) == {"experiment_name", "step.checkpoint"}
    assert exp100().step.checkpoint is None


def test_exp101_swaps_every_bit_pin_for_the_default_and_keeps_the_recipe() -> None:
    """The compat classes become their parents with the same fields; fp32 precision."""
    config, defaults = exp101(), CraftaxTrainLoop().step
    parent = exp100()
    assert type(config.step.optimizer) is FusedMuon.Config
    assert type(config.step.sampler) is PhiloxSampler.Config
    learner, parent_learner = config.step.learner, parent.step.learner
    assert isinstance(learner, AgentWindows.Config)
    assert isinstance(parent_learner, AgentWindows.Config)
    objective, parent_objective = learner.objective, parent_learner.objective
    assert type(objective) is TritonPPO.Config
    assert isinstance(objective, TritonPPO.Config)
    assert isinstance(parent_objective, TritonPPO.Config)
    assert objective.discount == parent_objective.discount
    optimizer, parent_optimizer = config.step.optimizer, parent.step.optimizer
    assert isinstance(optimizer, FusedMuon.Config)
    assert isinstance(parent_optimizer, FusedMuon.Config)
    assert optimizer.lr == parent_optimizer.lr
    model = config.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert type(model.block.scan) is TritonScan.Config
    assert (
        model.state_dtype
        == model.output_dtype
        == config.step.rollout.dtype
        == torch.float32
    )
    assert config.step.schedule == defaults.schedule
    assert config.step.checkpoint is None


def test_exp_smoke_changes_only_sizes_budget_and_sinks() -> None:
    assert _deltas(exp000(), exp_smoke()) == {
        "experiment_name",
        "step.checkpoint",
        "step.model.channels_hidden",
        "step.model.num_layers",
        "step.model.embedding.channels_out",
        "step.env.num_envs",
        "step.env.num_buffers",
        "step.env.restart.num_worlds",
        "step.rollout.horizon",
        "step.learner.minibatch_size",
        "step.train_budget_steps",
        "step.evaluation.num_episodes",
        "max_steps",
        "tracker",
        "tracker.trackers",
        "tracker.capture_prefix",
        "tracker.working_dir",
    }
    config = exp_smoke()
    step = config.step
    model = step.model
    assert isinstance(model, MinGRUPolicy.Config)
    pool = step.env.restart
    assert isinstance(pool, WorldPool.Config)
    learner = step.learner
    assert isinstance(learner, AgentWindows.Config)
    assert (
        model.channels_hidden,
        model.num_layers,
        multi_hot_embedding(model).channels_out,
        step.env.num_envs,
        step.env.num_buffers,
        pool.num_worlds,
        step.rollout.horizon,
        learner.minibatch_size,
        config.max_steps,
        step.evaluation.num_episodes,
    ) == (8, 1, 2, 16, 2, 64, 16, 128, 4, 1)
    # A file, not W&B: a smoke run on a fresh machine logs nowhere it needs a login.
    assert config.tracker == FileTracker.Config()


def test_exp002_evaluates_by_the_rules_it_trains_by() -> None:
    config = exp002().copy_tree().finalize()
    assert config.step.evaluation.env == config.step.env
    assert config.step.env.rules.original_reward
    assert isinstance(config.step.env.restart, FreshWorlds.Config)


def test_an_unset_evaluation_plays_the_training_sampler_and_horizon() -> None:
    step = exp000().step.copy_tree().finalize()
    assert step.evaluation.sampler == step.sampler
    assert step.evaluation.rollout is not None
    assert step.evaluation.rollout.horizon == step.rollout.horizon
    assert step.evaluation.rollout.num_slots == 1


def test_exp002_policy_reads_the_symbolic_observation() -> None:
    model = exp002().copy_tree().finalize().step.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert model.proj_in.channels_in == SYMBOLIC_OBS_SIZE


def test_exp_smoke_is_narrow_and_reads_no_file() -> None:
    """exp_smoke cuts every axis that costs time, and needs no checkpoint (EXP-2)."""
    step = exp_smoke().step.copy_tree().finalize()
    assert isinstance(step.model, MinGRUPolicy.Config)
    assert step.model.channels_hidden <= 8
    assert step.model.num_layers == 1
    assert step.checkpoint is None


@pytest.mark.compute_training
def test_exp_smoke_trains_and_leaves_none_of_its_steps_threads_behind(
    tmp_path: Path,
) -> None:
    """exp_smoke runs end to end on the CPU, its evaluation included.

    The loop then closes the step: its rollout, worker and env threads stop
    (LIFE-1).
    """
    before = set(threading.enumerate())
    config = exp_smoke()
    config.base_dir = tmp_path
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.device = "cpu"
    # The kernels need CUDA; their torch forms run the same recurrence, rule and
    # streams.
    assert isinstance(config.step.model, MinGRUPolicy.Config)
    config.step.model.block.scan = TorchScan.Config()
    windows = config.step.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.objective = TorchPPO.Config().update(
        windows.objective,
        skip_missing=True,
    )
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    loop = config.make()
    loop.train()
    assert loop.step.global_step == config.max_steps
    leftover = [thread for thread in threading.enumerate() if thread not in before]
    assert not leftover, leftover


def test_exp003_swaps_in_craftax_baselines_learner_and_geometry() -> None:
    """The recipe is one treatment; a swapped slot counts once, its fields its own."""
    swapped = {
        "step.model": ActorCritic.Config,
        "step.optimizer": PartialConfig,
        "step.schedule": ProgressSchedule.Config,
        "step.learner": ShuffledTransitions.Config,
    }
    deltas = {
        name
        for name in _deltas(exp002(), exp003())
        if not name.startswith(tuple(f"{slot}." for slot in swapped))
    }
    assert deltas == {
        "experiment_name",
        "seed",
        "runtime.float32_matmul_precision",
        *swapped,
        "step.reward_clip",
        "step.env.num_envs",
        "step.env.num_buffers",
        "step.env.seed",
        "step.sampler.seed",
        "step.rollout.horizon",
        "step.rollout.num_slots",
        "step.rollout.bootstrap",
        "step.rollout.dtype",
        "step.train_budget_steps",
        "max_steps",
    }
    config = exp003()
    for path, kind in swapped.items():
        assert isinstance(_at(config, path), kind), path
    optimizer = config.step.optimizer.make()
    assert isinstance(optimizer, partial)
    assert optimizer.func is ClippedAdam
    schedule = config.step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    curve = schedule.curve.make()
    assert isinstance(curve, partial)
    assert curve.func is linear


def test_exp003_runs_the_references_budget_and_minibatches() -> None:
    """15,258 updates of 1,024 x 64, each four passes of eight minibatches of 8,192."""
    config = exp003().copy_tree().finalize()
    step = config.step
    transitions = step.env.num_envs * step.rollout.horizon
    assert config.max_steps == 15_258
    assert config.max_steps * transitions == 999_948_288
    learner = step.learner
    assert isinstance(learner, ShuffledTransitions.Config)
    assert (learner.num_passes, learner.num_minibatches) == (4, 8)
    assert transitions // learner.num_minibatches == 8_192
    assert step.reward_clip == math.inf
    assert isinstance(config.runtime, SingleProcess.Config)
    assert config.runtime.float32_matmul_precision == "high"
    # Nothing learns during the evaluation, so it plays without a bootstrap row.
    assert step.evaluation.rollout is not None
    assert (step.evaluation.rollout.num_slots, step.evaluation.rollout.bootstrap) == (
        1,
        False,
    )


@pytest.mark.compute_training
def test_exp003_at_test_size_trains_and_evaluates_on_the_cpu(tmp_path: Path) -> None:
    """exp003's recipe on a network of 8 and 8 environments, two epochs and an eval."""
    config = exp003()
    config.base_dir = tmp_path
    config.tracker = FileTracker.Config()
    # The matmul precision is process-wide and means nothing on a CPU; left set,
    # it would outlive this test in its worker.
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.float32_matmul_precision = None
    config.runtime.device = "cpu"
    model = config.step.model
    assert isinstance(model, ActorCritic.Config)
    model.channels_hidden = 8
    model.num_layers = 1
    config.step.env.num_envs = 8
    config.step.env.num_buffers = 2
    config.step.env.threads_per_buffer = 1
    config.step.rollout.horizon = 8
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    loop = config.make()
    loop.train()
    assert loop.step.global_step == 2


def test_exp004_takes_the_1m_geometry_rate_and_budget_and_nothing_else() -> None:
    """256 x 16 steps, Adam at 3e-4, 244 updates: one treatment, the rest exp003's."""
    assert _deltas(exp003(), exp004()) == {
        "experiment_name",
        "step.env.num_envs",
        "step.rollout.horizon",
        "step.optimizer",
        "step.train_budget_steps",
        "max_steps",
    }
    config = exp004().copy_tree().finalize()
    step = config.step
    transitions = step.env.num_envs * step.rollout.horizon
    assert (config.max_steps, config.max_steps * transitions) == (244, 999_424)
    learner = step.learner
    assert isinstance(learner, ShuffledTransitions.Config)
    assert (learner.num_passes, transitions // learner.num_minibatches) == (4, 512)
    optimizer = step.optimizer.make()
    assert isinstance(optimizer, partial)
    assert optimizer.func is ClippedAdam
    assert optimizer.keywords == {"lr": 3e-4, "eps": 1e-5, "max_grad_norm": 1.0}
    # The evaluation plays the training geometry, as exp003's does.
    assert step.evaluation.env is not None
    assert step.evaluation.rollout is not None
    assert (step.evaluation.env.num_envs, step.evaluation.rollout.horizon) == (256, 16)


def test_exp007_changes_only_the_budget_and_the_schedules_horizon() -> None:
    """1,525 updates of 1,024 x 64; the anneal spans them, not exp003's 15,258."""
    assert _deltas(exp003(), exp007()) == {
        "experiment_name",
        "step.train_budget_steps",
        "max_steps",
    }
    config = exp007()
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert (config.max_steps, config.max_steps * transitions) == (1_525, 99_942_400)
    assert config.step.train_budget_steps == config.max_steps


@pytest.mark.compute_training
def test_exp007s_first_epochs_match_their_golden() -> None:
    """exp007's learner at test size from portable weights, three epochs, frozen.

    Its rate and linear anneal, Adam behind the global clip, four passes of
    eight shuffled minibatches and exp002's rules in fresh worlds, on 4
    environments of 2 buffers, rollouts of 6 and towers of 8 in 2 layers.
    Each epoch's rate, its mean loss terms and every weight. The test's budget
    replaces exp007's, so these are exp003's bits too; exp004 runs the same
    code at another rate, which its delta test pins.
    """
    with host_agnostic_pipeline():
        step = _tiny_baseline_step(exp007()).make()
        try:
            _fill_portable(step.model)
            lines: list[str] = []
            while step.global_step < step.total_steps:
                result = step.train_step()
                prefix = f"epoch {step.global_step:04d}"
                metrics = result.get("metrics", {})
                lines.append(f"{prefix} learning_rate {fp32(metrics['learning_rate'])}")
                lines += [
                    f"{prefix} {term} {fp32(value)}"
                    for term, value in zip(
                        TorchPPO.Config.LOSS_NAMES,
                        result["model"],
                        strict=True,
                    )
                ]
                lines += [
                    f"{prefix} parameter {parameter} {digest(value)}"
                    for parameter, value in step.model.named_parameters()
                ]
        finally:
            step.close()
    assert_golden(test_file=__file__, name="exp007_tiny", lines=lines)


def test_exp006_keeps_exp003s_setup_around_the_q_learner() -> None:
    """Outside the step only the name and the budget move; inside, exp003's env and streams."""
    parent, config = exp003(), exp006()
    outside = {
        name
        for name in _deltas(parent, config)
        if name != "step" and not name.startswith("step.")
    }
    assert outside == {"experiment_name", "max_steps"}
    assert _flatten(config.step.env) == _flatten(parent.step.env)
    sampler = config.step.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    assert _flatten(sampler.sampler) == _flatten(parent.step.sampler)
    assert config.step.seed == 42


def test_exp006_runs_the_references_budget_and_recipe() -> None:
    """7,629 updates of 1,024 x 128, each four passes of four minibatches of 256."""
    config = exp006().copy_tree().finalize()
    step = config.step
    transitions = step.env.num_envs * step.rollout.horizon
    assert config.max_steps == step.train_budget_steps == 7_629
    assert config.max_steps * transitions == 999_948_288
    assert (step.rollout.horizon, step.rollout.num_slots) == (128, 1)
    assert (step.num_epochs, step.num_minibatches) == (4, 4)
    assert step.env.num_envs // step.num_minibatches == 256
    assert (step.discount, step.trace_decay, step.max_grad_norm) == (0.99, 0.5, 0.5)
    assert step.model.channels_hidden == 512
    assert step.model.observation_size == SYMBOLIC_OBS_SIZE
    sampler = step.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    assert (sampler.start, sampler.finish, sampler.decay_fraction) == (1.0, 0.005, 0.1)
    assert (sampler.steps_per_update, sampler.total_updates) == (128, 7_629)
    optimizer = step.optimizer.make()
    assert isinstance(optimizer, partial)
    assert (optimizer.func, optimizer.keywords) == (torch.optim.RAdam, {"lr": 3e-4})
    schedule = step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    curve = schedule.curve.make()
    assert isinstance(curve, partial)
    assert curve.func is linear
    assert isinstance(step.feature, PreviousAction.Config)
    assert isinstance(step.evaluation.sampler, GreedySampler.Config)
    assert isinstance(config.runtime, SingleProcess.Config)
    assert config.runtime.float32_matmul_precision == "high"


@pytest.mark.compute_training
def test_exp006_at_test_size_trains_and_evaluates_on_the_cpu(tmp_path: Path) -> None:
    """exp006's recipe on a network of 5 and 8 environments, two updates and an eval."""
    config = exp006()
    config.base_dir = tmp_path
    config.tracker = FileTracker.Config()
    # The matmul precision is process-wide and means nothing on a CPU; left set,
    # it would outlive this test in its worker.
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.float32_matmul_precision = None
    config.runtime.device = "cpu"
    config.step.model.channels_hidden = 5
    env = config.step.env
    assert isinstance(env, CraftaxEnv.Config)
    env.num_envs = 8
    env.num_buffers = 2
    env.threads_per_buffer = 1
    config.step.rollout.horizon = 4
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    sampler = config.step.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    sampler.sampler = TorchPhiloxSampler.Config().update(
        sampler.sampler,
        skip_missing=True,
    )
    loop = config.make()
    loop.train()
    assert loop.step.global_step == 2


def test_exp005_swaps_in_the_gru_and_whole_trajectories_and_nothing_else() -> None:
    """The policy and the minibatches are one treatment; the learner keeps exp003's fields."""
    swapped = {
        "step.model": ActorCriticRNN.Config,
        "step.learner": ShuffledTrajectories.Config,
    }
    assert {
        name
        for name in _deltas(exp003(), exp005())
        if not name.startswith("step.model.")
    } == {"experiment_name", *swapped}
    config = exp005().copy_tree().finalize()
    for path, kind in swapped.items():
        assert type(_at(config, path)) is kind, path
    model = config.step.model
    assert isinstance(model, ActorCriticRNN.Config)
    assert (model.observation_size, model.channels_hidden, model.num_layers) == (
        SYMBOLIC_OBS_SIZE,
        512,
        2,
    )
    learner = config.step.learner
    assert isinstance(learner, ShuffledTrajectories.Config)
    # 1,024 agents: eight minibatches of 128 whole trajectories of 64 steps.
    assert config.step.env.num_envs // learner.num_minibatches == 128
    assert config.step.rollout.horizon == 64
    assert (config.step.rollout.num_slots, config.step.rollout.bootstrap) == (1, True)


@pytest.mark.compute_training
def test_exp005_at_test_size_trains_and_evaluates_on_the_cpu(tmp_path: Path) -> None:
    """exp005's recipe on a GRU of 8 and 16 environments, two epochs and an eval.

    Sixteen, so its eight minibatches each replay two agents' trajectories.
    """
    config = exp005()
    config.base_dir = tmp_path
    config.tracker = FileTracker.Config()
    # The matmul precision is process-wide and means nothing on a CPU; left set,
    # it would outlive this test in its worker.
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.float32_matmul_precision = None
    config.runtime.device = "cpu"
    model = config.step.model
    assert isinstance(model, ActorCriticRNN.Config)
    model.channels_hidden = 8
    model.num_layers = 1
    config.step.env.num_envs = 16
    config.step.env.num_buffers = 2
    config.step.env.threads_per_buffer = 1
    config.step.rollout.horizon = 8
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    loop = config.make()
    loop.train()
    assert loop.step.global_step == 2


def test_exp008_swaps_in_the_gtrxl_policy_its_windows_and_longer_rollouts() -> None:
    """The recipe is one treatment; a swapped slot counts once, its fields its own."""
    swapped = {
        "step.model": GTrXLPolicy.Config,
        "step.learner": TrajectoryWindows.Config,
    }
    deltas = {
        name
        for name in _deltas(exp003(), exp008())
        if not name.startswith(tuple(f"{slot}." for slot in swapped))
    }
    assert deltas == {
        "experiment_name",
        *swapped,
        "step.rollout.horizon",
        "step.train_budget_steps",
        "max_steps",
    }
    config = exp008()
    for path, kind in swapped.items():
        assert isinstance(_at(config, path), kind), path


def test_exp008_runs_the_references_geometry_and_budget() -> None:
    """transformerXL_PPO_JAX's network and learner, 7,629 updates of 1,024 x 128."""
    config = exp008().copy_tree().finalize()
    step = config.step
    model = step.model
    assert isinstance(model, GTrXLPolicy.Config)
    assert (model.observation_size, model.num_actions) == (SYMBOLIC_OBS_SIZE, 43)
    assert (model.channels_hidden, model.num_layers, model.memory_length) == (
        256,
        2,
        128,
    )
    # The tiny bfb goldens draw their own weights, so only this pins the init.
    init = model.proj_in.init_weight
    assert isinstance(init, partial)
    assert (init.func, init.keywords) == (
        fan_in_truncated_normal,
        {"variance_correction": True},
    )
    block = model.block
    assert (block.heads, block.channels_head, block.gating_bias) == (8, 32, 2.0)
    decoder = model.decoder
    assert (decoder.channels_hidden, decoder.num_layers) == (256, 2)
    assert decoder.activation is torch.relu
    learner = step.learner
    assert isinstance(learner, TrajectoryWindows.Config)
    assert (learner.num_passes, learner.num_minibatches, learner.window) == (4, 8, 64)
    assert (learner.discount, learner.trace_decay, learner.clip_epsilon) == (
        0.999,
        0.8,
        0.2,
    )
    assert (learner.value_coefficient, learner.entropy_coefficient) == (0.5, 0.002)
    sampler = step.sampler
    assert isinstance(sampler, PhiloxSampler.Config)
    assert {config.seed, step.env.seed, sampler.seed, learner.seed} == {42}
    transitions = step.env.num_envs * step.rollout.horizon
    assert (step.env.num_envs, step.rollout.horizon) == (1_024, 128)
    assert config.max_steps == 7_629
    assert config.max_steps * transitions == 999_948_288
    assert (step.rollout.num_slots, step.rollout.bootstrap) == (1, True)
    assert step.rollout.dtype == model.dtype == torch.float32
    assert step.evaluation.rollout is not None
    assert step.evaluation.rollout.horizon == 128


@pytest.mark.compute_training
def test_exp008_at_test_size_trains_and_evaluates_on_the_cpu(tmp_path: Path) -> None:
    """exp008's recipe at width 4 over 8 environments, two epochs and an eval."""
    config = exp008()
    config.base_dir = tmp_path
    config.tracker = FileTracker.Config()
    # The matmul precision is process-wide and means nothing on a CPU; left set,
    # it would outlive this test in its worker.
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.float32_matmul_precision = None
    config.runtime.device = "cpu"
    model = config.step.model
    assert isinstance(model, GTrXLPolicy.Config)
    model.channels_hidden = 4
    model.memory_length = 3
    model.block.heads = 2
    model.block.channels_head = 3
    model.decoder.channels_hidden = 5
    learner = config.step.learner
    assert isinstance(learner, TrajectoryWindows.Config)
    learner.num_passes = 2
    learner.num_minibatches = 2
    learner.window = 2
    config.step.env.num_envs = 8
    config.step.env.num_buffers = 2
    config.step.env.threads_per_buffer = 1
    config.step.rollout.horizon = 4
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    loop = config.make()
    loop.train()
    assert loop.step.global_step == 2


def test_exp102_changes_the_policy_the_rule_the_rate_and_the_budget() -> None:
    """One bundled fork: the model's three new stages, the recipe's coefficients."""
    model_stages = (
        "step.model.embedding",
        "step.model.injection",
        "step.model.auxiliary",
    )
    assert {
        name
        for name in _deltas(exp101(), exp102())
        if not name.startswith(tuple(f"{stage}." for stage in model_stages))
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        "step.env.rules.previous_action",
        *model_stages,
        "step.learner.objective.value_coefficient",
        "step.learner.objective.value_clip_epsilon",
        "step.optimizer.lr",
        "step.schedule.curve",
        "step.train_budget_steps",
        "max_steps",
        "num_steps_eval",
    }
    config = exp102().copy_tree().finalize()
    assert config.step.env.observation_size == ACTION_OBS_SIZE
    assert config.step.model.observation_size == ACTION_OBS_SIZE
    optimizer = config.step.optimizer
    assert isinstance(optimizer, FusedMuon.Config)
    parent = exp000().step.optimizer
    assert isinstance(parent, FusedMuon.Config)
    assert torch.tensor(optimizer.lr, dtype=torch.float32) == (
        torch.tensor(parent.lr, dtype=torch.float32) / 4
    )
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert config.max_steps * transitions == 19_999_490_048
    assert config.num_steps_eval * transitions <= 1_000_000_000
    schedule = config.step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    curve = schedule.curve.make()
    assert curve(0.0) == 0.0
    assert curve(250_000_000 / 20_000_000_000) == 1.0
    assert curve(1.0) == 0.0


def test_exp102_trains_20b_transitions_evaluating_every_1b_at_any_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both budgets are transitions over the epoch's: exact at one transition an epoch."""

    def one_transition_an_epoch() -> CraftaxTrainLoop:
        cfg = exp101()
        cfg.step.env.num_envs = cfg.step.rollout.horizon = 1
        return cfg

    monkeypatch.setattr(experiments, "exp101", one_transition_an_epoch)
    cfg = experiments.exp102()
    assert (cfg.max_steps, cfg.num_steps_eval) == (20_000_000_000, 1_000_000_000)


def test_exp103_adds_practice_imitation_the_cap_div8_and_the_win() -> None:
    assert _deltas(exp102(), exp103()) == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        "step.env.rules.end_on_boss_defeat",
        "step.env.stall_cap",
        "step.env.stall_cap.stall_limit",
        "step.env.stall_cap.uncapped_fraction",
        "step.env.practice",
        "step.env.practice.entries_per_level",
        "step.env.practice.entries_per_world",
        "step.env.practice.fraction",
        "step.env.practice.level_width",
        "step.env.practice.num_donors",
        "step.env.practice.num_levels",
        "step.env.practice.reach_decay",
        "step.env.practice.seed",
        "step.learner.auxiliary",
        "step.learner.auxiliary.capacity",
        "step.learner.auxiliary.coefficient",
        "step.learner.auxiliary.rows",
        "step.reward_scale",
        "step.reward_clip",
    }
    config = exp103().copy_tree().finalize()
    assert (config.step.reward_scale, config.step.reward_clip) == (0.125, math.inf)


def test_exp103_evaluates_uncapped_without_practice_ending_on_the_win() -> None:
    config = exp103().copy_tree().finalize()
    evaluation = config.step.evaluation.env
    assert evaluation is not None
    assert evaluation.stall_cap is None
    assert evaluation.practice is None
    assert evaluation.rules == config.step.env.rules
    assert evaluation.rules.end_on_boss_defeat
    assert evaluation.rules.previous_action


def test_exp109_trains_in_fresh_worlds_and_evaluates_as_exp103_does() -> None:
    """Only training's world source changes; the evaluation keeps exp103's pool."""
    restart, evaluation = "step.env.restart", "step.evaluation.env"
    assert {
        name
        for name in _deltas(exp103(), exp109())
        if not name.startswith((f"{restart}.", f"{evaluation}."))
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        restart,
        evaluation,
    }
    config = exp109().copy_tree().finalize()
    assert isinstance(config.step.env.restart, FreshWorlds.Config)
    assert config.step.env.practice is not None
    assert config.step.env.stall_cap is not None
    parent = exp103().copy_tree().finalize()
    assert config.step.evaluation.env == parent.step.evaluation.env


def test_exp104_doubles_the_width_and_the_window_and_keeps_the_rollout() -> None:
    assert _deltas(exp102(), exp104()) == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        "step.model.channels_hidden",
        "step.env.num_envs",
        "step.rollout.horizon",
    }
    config, parent = exp104().copy_tree().finalize(), exp102().copy_tree().finalize()
    model = config.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert (model.proj_in.channels_in, model.proj_in.channels_out) == (1_635, 2_048)
    assert model.num_layers == 4
    assert model.block.channels_hidden == model.proj_out.channels_in == 2_048
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert transitions == parent.step.env.num_envs * parent.step.rollout.horizon
    windows = config.step.learner
    assert isinstance(windows, AgentWindows.Config)
    assert windows.minibatch_size // config.step.rollout.horizon == 64
    count = minibatch_count(
        agents=config.step.env.num_envs,
        horizon=config.step.rollout.horizon,
        minibatch_size=windows.minibatch_size,
        replay_ratio=windows.replay_ratio,
    )
    assert count == 18
    assert config.max_steps * transitions == 19_999_490_048
    evaluation = config.step.evaluation
    assert evaluation.env is not None
    assert evaluation.rollout is not None
    assert (evaluation.env.num_envs, evaluation.rollout.horizon) == (1_024, 512)


def test_exp105_swaps_the_boards_blocks_for_one_convnext_trunk() -> None:
    board = "step.model.embedding.board"
    assert {
        name
        for name in _deltas(exp104(), exp105())
        if not name.startswith(f"{board}.block.")
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        f"{board}.block",
        f"{board}.num_layers",
    }
    model = exp105().copy_tree().finalize().step.model
    assert isinstance(model, MinGRUPolicy.Config)
    encoder = model.embedding
    assert isinstance(encoder, BoardEncoder.Config)
    trunk = encoder.board.block
    assert isinstance(trunk, ConvNextTrunk.Config)
    assert (encoder.board.num_layers, trunk.channels_in, trunk.channels_hidden) == (
        1,
        16,
        64,
    )
    assert (trunk.num_layers, trunk.block.expansion, trunk.block.layer_scale) == (
        2,
        4,
        1e-6,
    )
    assert trunk.block.depthwise.kernel_size == 5
    assert trunk.block.norm.eps == 1e-6
    assert trunk.block.mlp.activation is functional.gelu
    # The c2 bfb golden draws its own weights, so only this pins the trunk's init.
    mlp = trunk.block.mlp
    assert [
        layer.init_weight
        for layer in (trunk.stem, trunk.block.depthwise, mlp.proj_in, mlp.proj_out)
    ] == [kaiming_uniform] * 4
    assert trunk.proj_out.init_weight is torch.nn.init.zeros_


def test_exp106_trains_250m_inside_the_20b_warmup_evaluating_every_50m() -> None:
    assert _deltas(exp105(), exp106()) == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        "max_steps",
        "num_steps_eval",
        "step.schedule.curve",
    }
    config = exp106().copy_tree().finalize()
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert config.max_steps == 476
    epochs = int(config.max_steps)
    assert epochs * transitions == 249_561_088
    assert config.step.train_budget_steps == exp105().step.train_budget_steps
    evaluations = [
        step for step in range(1, epochs) if step % config.num_steps_eval == 0
    ]
    # The first epoch past each 50M, then the final evaluation at 249.6M.
    assert [step * transitions for step in evaluations] == [
        50_331_648,
        100_663_296,
        150_994_944,
        201_326_592,
    ]
    schedule = config.step.schedule
    assert isinstance(schedule, ProgressSchedule.Config)
    assert schedule.curve == PartialConfig(warmup, fraction=0.0125)
    curve = schedule.curve.make()
    total = config.step.train_budget_steps
    rates = [curve(step / total) for step in range(epochs)]
    # A linear ramp from zero, still rising at the run's last epoch.
    assert rates[0] == 0.0
    assert all(
        math.isclose(rate, step * rates[1], rel_tol=1e-12)
        for step, rate in enumerate(rates)
    )
    assert rates[-1] < 1.0
    assert curve(250_000_000 / 20_000_000_000) == 1.0


def test_exp107_adds_the_frozen_feature_and_its_zero_projection() -> None:
    feature_path, proj_path = "step.feature", "step.model.proj_feature"
    assert {
        name
        for name in _deltas(exp106(), exp107())
        if not name.startswith((f"{feature_path}.", f"{proj_path}."))
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        feature_path,
        proj_path,
    }
    config = exp107().copy_tree().finalize()
    assert config.step.feature == _frozen_feature(
        TrainedWeights.Config(
            experiment=_WORLD_MODEL_EXP001,
            checkpoint=_ORACLE_CHECKPOINTS / "exp001-s0" / "step_00001525.pt",
            overrides=[],
        ),
    )
    model = config.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    proj = model.proj_feature
    assert proj is not None
    assert (proj.channels_in, proj.channels_out, proj.bias) == (1_152, 2_048, False)
    assert proj.init_weight is torch.nn.init.zeros_


def test_exp108_swaps_only_the_feature_weights_for_a_seeded_init() -> None:
    weights = "step.feature.weights"
    assert {
        name
        for name in _deltas(exp107(), exp108())
        if not name.startswith(f"{weights}.")
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        weights,
    }
    assert exp108().copy_tree().finalize().step.feature == _frozen_feature(
        InitialWeights.Config(experiment=_WORLD_MODEL_EXP001, overrides=[], seed=0),
    )


def test_exp110_reads_a_frozen_early_world_model_alone_and_evaluates_as_its_reference() -> (
    None
):
    """The encoder's slots, the feature, and the evaluation's rows and cadence change."""
    slots = (
        "step.model.embedding",
        "step.model.injection",
        "step.model.proj_feature",
        "step.feature",
        "step.evaluation.env",
    )
    assert {
        name
        for name in _deltas(exp103(), exp110())
        if not name.startswith(tuple(f"{slot}." for slot in slots))
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        *slots,
        "num_steps_eval",
    }
    config = exp110().copy_tree().finalize()
    model = config.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert isinstance(model.embedding, NoEncoder.Config)
    assert model.observation_size == config.step.env.observation_size == ACTION_OBS_SIZE
    assert model.injection is None
    assert isinstance(model.auxiliary, FeasibilityLoss.Config)
    proj = model.proj_feature
    assert proj is not None
    assert (proj.channels_in, proj.channels_out, proj.bias) == (1_152, 1_024, False)
    assert proj.init_weight is kaiming_uniform
    # Frozen: ``joint`` is False until exp112 gives the learner the feature to train.
    assert config.step.feature == _frozen_feature(
        TrainedWeights.Config(
            experiment=_WORLD_MODEL_EXP001,
            checkpoint=_ORACLE_CHECKPOINTS / "early-fit-s73" / "step_00013135.pt",
            overrides=[],
        ),
    )


def test_exp110_evaluates_on_1024_environments_every_250m_and_at_the_end() -> None:
    """The first epoch past each 250M, at 20B and at the 1B the docstring overrides."""
    config = exp110().copy_tree().finalize()
    evaluation = config.step.evaluation.env
    assert evaluation is not None
    assert (evaluation.num_envs, evaluation.num_buffers) == (1_024, 4)
    assert evaluation.stall_cap is None
    assert evaluation.practice is None
    assert evaluation.rules == config.step.env.rules
    assert evaluation.restart == config.step.env.restart
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert (config.num_steps_eval - 1) * transitions < 250_000_000
    assert config.num_steps_eval * transitions == 250_085_376
    assert config.max_steps == config.step.train_budget_steps == exp103().max_steps
    one_billion = 1_000_000_000 // transitions
    assert (one_billion, one_billion * transitions) == (1_907, 999_817_216)
    evaluations = [
        step for step in range(1, one_billion) if step % config.num_steps_eval == 0
    ]
    assert [step * transitions for step in evaluations] == [
        250_085_376,
        500_170_752,
        750_256_128,
    ]


def test_exp110s_policy_builds_its_trunk_heads_and_projection_alone() -> None:
    """At width 8: no encoder weight, the feature's projection last."""
    model_config = exp110().step.model
    assert isinstance(model_config, MinGRUPolicy.Config)
    model_config.channels_hidden = 8
    model = model_config.make()
    shapes = [(name, tuple(weight.shape)) for name, weight in model.named_parameters()]
    assert shapes == [
        ("proj_out.weight", (44, 8)),
        *((f"blocks.{index}.proj_gates.weight", (24, 8)) for index in range(4)),
        ("auxiliary.proj_out.weight", (43, 8)),
        ("proj_feature.weight", (8, 1_152)),
    ]


def test_exp111_swaps_only_the_world_models_checkpoint() -> None:
    assert _deltas(exp110(), exp111()) == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        "step.feature.weights.checkpoint",
    }
    feature = exp111().step.feature
    assert isinstance(feature, WorldModelFeature.Config)
    assert isinstance(feature.weights, TrainedWeights.Config)
    assert feature.weights.checkpoint == (
        _ORACLE_CHECKPOINTS / "mature-fit-s73" / "step_00012738.pt"
    )


@pytest.mark.parametrize(("parent", "child"), [(exp110, exp112), (exp111, exp113)])
def test_the_joint_arms_train_their_frozen_parents_world_model(
    parent: ExperimentFactory[CraftaxTrainLoop],
    child: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    """One change: the learner trains the feature, compiled, on FA4; it turns joint."""
    training = "step.feature_training"
    assert {
        name
        for name in _deltas(parent(), child())
        if not name.startswith(f"{training}.")
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        training,
    }
    final = child().copy_tree().finalize().step
    assert isinstance(final.feature, WorldModelFeature.Config)
    assert final.feature.joint
    assert final.feature_training == ContextReplay.Config(
        attention=Flash4Varlen.Config(),
        bin_tokens=4_096,
        pass_tokens=16_384,
        frames_per_batch=512,
        compile=PartialConfig(torch.compile, fullgraph=True, dynamic=False),
    )


@pytest.mark.compute_training
def test_a_sole_feature_step_on_exact_windows_learns_from_branches_and_evaluates() -> (
    None
):
    """exp110's recipe on exact windows: width 8, 8 environments, the smoke world model.

    Practice saves donor histories in the step graph, every row of the second
    epoch's slot is marked a branch, and the imitation archives the features
    the actor stored for those rows; then a 4-environment evaluation plays.
    """
    step = _tiny_sole_feature_step(exp110()).make()
    try:
        step.train_step()
        slot = step.rollout.slots[step.ready]
        assert slot.features is not None
        slot.branch_starts.fill_(1)
        stored = slot.features.transpose(0, 1).clone()
        metrics = step.train_step().get("metrics", {})
        assert float(metrics["imitation/branches"]) == 2.0
        assert step.rollout.histories is not None
        archived = step.learner.state_dict()["features"]
        assert archived.shape == (2, *stored.shape[1:])
        for branch in archived:
            assert any(torch.equal(branch, row) for row in stored[:2]), "not a row's"
        evaluator = step.make_evaluator()
        try:
            assert evaluator.play().rollouts >= 1
        finally:
            evaluator.close()
    finally:
        step.close()


@pytest.mark.compute_training
def test_a_joint_step_on_exact_windows_trains_beside_practice_and_imitation() -> None:
    """exp112's recipe at the same test size, the replay masked and eager.

    The boot slot is the starting weights' own, so the first epoch's replay
    matches what the actor read; the world model trains, practice saves
    donor histories, the imitation archives two branches' stored features,
    and an evaluation plays with the trained weights published.
    """
    config = exp112()
    step_config = _tiny_sole_feature_step(config)
    replay = step_config.feature_training
    assert isinstance(replay, ContextReplay.Config)
    replay.attention = SdpaVarlen.Config()
    replay.compile = None
    replay.bin_tokens = 1
    # The last epoch starts no rollout: a third makes the second rebuild.
    step_config.train_budget_steps = 3
    step = step_config.make()
    try:
        assert step.joint is not None
        before = {name: w.clone() for name, w in step.joint.weights().items()}
        first = step.train_step().get("metrics", {})
        # The stored features are bf16, the policy's dtype: rounding alone.
        assert float(first["joint/feature_gap"]) < 0.01
        slot = step.rollout.slots[step.ready]
        slot.branch_starts.fill_(1)
        second = step.train_step().get("metrics", {})
        assert float(second["imitation/branches"]) == 2.0
        assert float(second["rebuild_seconds"]) > 0
        # exp102's one-cycle schedule starts at rate 0, so the first epoch
        # moves nothing.
        assert not all(
            torch.equal(weight, before[name])
            for name, weight in step.joint.weights().items()
        )
        assert step.rollout.histories is not None
        evaluator = step.make_evaluator()
        try:
            assert evaluator.play().rollouts >= 1
        finally:
            evaluator.close()
    finally:
        step.close()


@pytest.mark.parametrize(
    ("factory", "group"),
    [
        (exp000, ""),
        (exp102, "exp102"),
        (exp103, "exp103"),
        (exp104, "exp104"),
        (exp105, "exp105"),
        (exp106, "exp106"),
        (exp107, "exp107"),
        (exp108, "exp108"),
        (exp109, "exp109"),
        (exp110, "exp110"),
        (exp111, "exp111"),
        (exp112, "exp112"),
        (exp113, "exp113"),
    ],
)
def test_the_dashboard_names_the_run_after_the_experiment_and_groups_its_seeds(
    factory: ExperimentFactory[CraftaxTrainLoop],
    group: str,
) -> None:
    for name in (factory.__name__, f"{factory.__name__}_s74"):
        config = factory().copy_tree()
        config.experiment_name = name
        config = config.finalize()
        assert isinstance(config.tracker, TrackerList.Config)
        wrapper = config.tracker.trackers["wandb"]
        assert isinstance(wrapper, AsyncTracker.Config)
        dashboard = wrapper.tracker
        assert isinstance(dashboard, WandbTracker.Config)
        assert (dashboard.name, dashboard.group) == (name, group)


def test_the_q_learning_loops_defaults_finalize_and_print_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CraftaxPQNTrainLoop().copy_tree().finalize()
        CraftaxPQNTrainLoop().pformat()


# Only the two reference baselines and the headline recipe carry a config golden.
# Every other fork is pinned by its delta test against a parent that has one, and
# each stack's class defaults by its tiny bit-for-bit golden or a stated config.
def test_exp000_matches_its_golden_config() -> None:
    """Pin the whole finalized ``exp000`` as readable text, every default shown."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp003_matches_its_golden_config() -> None:
    """Pin the whole finalized ``exp003`` as readable text, every default shown."""
    assert_pprint_golden(test_file=__file__, name="exp003", config=exp003())


def test_exp103_matches_its_golden_config() -> None:
    """Pin the whole finalized ``exp103`` as readable text, every default shown."""
    assert_pprint_golden(test_file=__file__, name="exp103", config=exp103())


def _frozen_feature(
    weights: TrainedWeights.Config | InitialWeights.Config,
) -> WorldModelFeature.Config:
    """Return the 1xx recipes' frozen world-model feature, every field stated."""
    return WorldModelFeature.Config(
        weights=weights,
        history=Refill.Config(t_max=1_024, keep=256),
        practice=FreshWindow.Config(),
        joint=False,
        layers=20,
        hook_interval=128,
        attention=Flash4CacheAttention.Config(),
        reprefill_rows=128,
        dtype=torch.bfloat16,
        compile=PartialConfig(
            torch.compile,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        ),
    )


# Width 8 in one layer; 8 environments in 2 buffers of 8 pool worlds, 2 donors and an
# archive of 4 levels of 2; imitation of rows 0-1 into an archive of 2; windows of 2
# agents over a horizon of 8; the smoke world model, float32, on exact windows of 4
# decisions with donor histories, its blocks 2 steps; the torch forms of the kernels; an
# evaluation of 4 environments and 1 episode.
def _tiny_sole_feature_step(config: CraftaxTrainLoop) -> CraftaxTrainStep.Config:
    """Shrink a sole-feature recipe's step to the CPU and put it on exact windows."""
    step = config.step
    step.parallelism.device = "cpu"
    model = step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.channels_hidden = 8
    model.num_layers = 1
    model.block.scan = TorchScan.Config()
    proj = model.proj_feature
    assert proj is not None
    proj.channels_in = 36  # The smoke world model's width.
    env = step.env
    env.num_envs = 8
    env.num_buffers = 2
    env.threads_per_buffer = 1
    pool = env.restart
    assert isinstance(pool, WorldPool.Config)
    pool.num_worlds = 8
    practice = env.practice
    assert practice is not None
    practice.num_donors = 2
    practice.num_levels = 4
    practice.entries_per_level = 2
    evaluation = step.evaluation.env
    assert evaluation is not None
    evaluation.num_envs = 4
    evaluation.num_buffers = 2
    evaluation.threads_per_buffer = 1
    evaluation.restart = pool.copy_tree()
    step.evaluation.num_episodes = 1
    windows = step.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.objective = TorchPPO.Config().update(windows.objective, skip_missing=True)
    windows.minibatch_size = 16
    imitation = windows.auxiliary
    assert isinstance(imitation, BranchImitation.Config)
    imitation.rows = imitation.capacity = 2
    step.sampler = TorchPhiloxSampler.Config().update(step.sampler, skip_missing=True)
    step.rollout.horizon = 8
    step.train_budget_steps = 2
    feature = step.feature
    assert isinstance(feature, WorldModelFeature.Config)
    weights = feature.weights = InitialWeights.Config()
    weights.experiment = "priml.baselines.craftax.world_model.experiments.exp_smoke"
    history = feature.history = Sliding.Config()
    history.decisions = 4
    feature.practice = DonorHistory.Config()
    feature.layers = 1
    feature.hook_interval = 2
    feature.attention = MaskedCacheAttention.Config()
    feature.compile = None
    feature.dtype = torch.float32
    return step


# 4 environments of 2 buffers by 6 steps: 24 transitions, so each of the recipe's eight
# minibatches holds 3, and none of the sizes that meet ties another.
def _tiny_baseline_step(config: CraftaxTrainLoop) -> CraftaxTrainStep.Config:
    """Shrink a Craftax_Baselines PPO recipe's step to the CPU, its recipe unchanged."""
    step = config.step
    step.parallelism.device = "cpu"
    model = step.model
    assert isinstance(model, ActorCritic.Config)
    model.channels_hidden = 8
    model.num_layers = 2
    step.env.num_envs = 4
    step.env.num_buffers = 2
    step.env.threads_per_buffer = 1
    step.rollout.horizon = 6
    step.sampler = TorchPhiloxSampler.Config().update(step.sampler, skip_missing=True)
    step.train_budget_steps = 3
    return step


# The towers' orthogonal init draws ``randn``, whose bits differ by host (SLEEF's
# ``log`` under AVX2 against libm's), so the golden starts from portable draws instead.
def _fill_portable(model: Policy) -> None:
    """Overwrite every weight with ``U(+-1/sqrt(fan_in))`` drawn the same on any host."""
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            parameter.copy_(
                portable_uniform(
                    *parameter.shape,
                    bound=parameter.shape[-1] ** -0.5,
                    generator=generator,
                ),
            )


def _at(config: object, path: str) -> object:
    """Return the value at a dotted path of nested configs."""
    for name in path.split("."):
        config = cast("object", getattr(config, name))
    return config


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
    flat: dict[str, object] = {}
    for field in fields(cast("DataclassInstance", config)):
        value = cast("object", getattr(config, field.name))
        dotted = f"{prefix}{field.name}"
        if isinstance(value, InlineConfig):
            flat[dotted] = repr(cast("InlineConfig[object]", value))
        elif is_dataclass(value) and not isinstance(value, type):
            # The class itself, so swapping in a different Config registers
            # as one change rather than a diff of every field.
            flat[dotted] = type(value)
            flat.update(_flatten(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = repr(value)
    return flat


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
