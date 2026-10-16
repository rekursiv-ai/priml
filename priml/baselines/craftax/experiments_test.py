"""Tests for the Craftax experiment configs."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import inspect
import math
import re
import sys
import threading
import warnings

from configgle import PartialConfig
from torch.nn import functional

import pytest
import torch

from priml.baselines.craftax import experiments
from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.env import (
    BossFightReward,
    CraftaxEnv,
    FreshWorlds,
    WorldPool,
)
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
    exp009,
    exp010,
    exp011,
    exp012,
    exp013,
    exp014,
    exp015,
    exp_smoke,
)
from priml.baselines.craftax.game.state import (
    ACTION_OBS_SIZE,
    SYMBOLIC_OBS_SIZE,
)
from priml.baselines.craftax.learners.gtrxl_train_step import TrajectoryWindows
from priml.baselines.craftax.learners.imitation import BranchImitation
from priml.baselines.craftax.learners.pqn_train_step import (
    CraftaxPQNTrainLoop,
    CraftaxPQNTrainStep,
)
from priml.baselines.craftax.learners.rnn_update import ShuffledTrajectories
from priml.baselines.craftax.learners.update import ShuffledTransitions
from priml.baselines.craftax.lib.adam import ClippedAdam
from priml.baselines.craftax.model import MinGRUPolicy
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
    config_deltas,
    config_fields,
    digest,
    fp32,
    host_agnostic_pipeline,
    multi_hot_embedding,
    optimizer_state,
    portable_uniform,
    tiny_board_policy,
)
from priml.baselines.craftax.train_step import (
    AgentWindows,
    CraftaxTrainStep,
    ProgressSchedule,
    load_masters,
    minibatch_count,
)
from priml.loss.policy_gradient import TorchPPO
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.math.schedules import linear
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
    from priml.baselines.craftax.model import Policy
    from priml.testing.experiments import ExperimentFactory


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
    exp009,
    exp010,
    exp011,
    exp012,
    exp013,
    exp014,
    exp015,
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
        exp006,
        exp007,
        exp008,
        exp009,
        exp010,
        exp011,
        exp012,
        exp014,
        exp015,
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


@pytest.mark.parametrize("factory", [exp000, exp009, exp010])
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
    assert config_deltas(CraftaxTrainLoop(), exp000()) >= {
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
    assert config_deltas(default, config) >= {
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


def test_exp005_changes_only_the_budget() -> None:
    assert exp005().max_steps == 476
    assert config_deltas(exp000(), exp005()) == {
        "experiment_name",
        "max_steps",
        "step.train_budget_steps",
    }


def test_exp009_changes_the_setup_and_what_its_observation_needs() -> None:
    # The six options are one treatment (the Hypothesis says why); the dense
    # first stage travels with the symbolic view. A swapped slot also lists the
    # fields its new class does not share.
    assert config_deltas(exp000(), exp009()) == {
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
    }


def test_every_experiment_seeds_torchs_generator_as_the_module_states() -> None:
    """PufferLib's 73, inherited from exp000; exp010's reference seed, 42, in its forks."""
    for factory in (
        exp000,
        exp001,
        exp002,
        exp003,
        exp004,
        exp005,
        exp006,
        exp007,
        exp008,
        exp009,
        exp_smoke,
    ):
        assert factory().seed == 73, factory.__name__
    for factory in (exp010, exp011, exp012, exp014):
        config = factory()
        learner = config.step.learner
        assert isinstance(learner, ShuffledTransitions.Config)
        sampler = config.step.sampler
        assert isinstance(sampler, PhiloxSampler.Config)
        seeds = {config.seed, config.step.env.seed, sampler.seed, learner.seed}
        assert seeds == {42}, factory.__name__


def test_exp000_draws_its_own_init_and_reads_no_file() -> None:
    """No checkpoint: torch's generator draws the policy's init under seed 73."""
    config = exp000()
    assert (config.step.checkpoint, config.seed) == (None, 73)


def test_exp001_swaps_every_bit_pin_for_the_default_and_keeps_the_recipe() -> None:
    """The compat classes become their parents with the same fields; fp32 precision."""
    config, defaults = exp001(), CraftaxTrainLoop().step
    parent = exp000()
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
    assert config_deltas(exp000(), exp_smoke()) == {
        "experiment_name",
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


def test_exp009_evaluates_by_the_rules_it_trains_by() -> None:
    config = exp009().copy_tree().finalize()
    assert config.step.evaluation.env == config.step.env
    assert config.step.env.rules.original_reward
    assert isinstance(config.step.env.restart, FreshWorlds.Config)


def test_an_unset_evaluation_plays_the_training_sampler_and_horizon() -> None:
    step = exp000().step.copy_tree().finalize()
    assert step.evaluation.sampler == step.sampler
    assert step.evaluation.rollout is not None
    assert step.evaluation.rollout.horizon == step.rollout.horizon
    assert step.evaluation.rollout.num_slots == 1


def test_exp009_policy_reads_the_symbolic_observation() -> None:
    model = exp009().copy_tree().finalize().step.model
    assert isinstance(model, MinGRUPolicy.Config)
    assert model.proj_in.channels_in == SYMBOLIC_OBS_SIZE


def test_exp_smoke_is_narrow_and_reads_no_file() -> None:
    """exp_smoke cuts every axis that costs time, and needs no checkpoint (EXP-2)."""
    step = exp_smoke().step.copy_tree().finalize()
    assert isinstance(step.model, MinGRUPolicy.Config)
    assert step.model.channels_hidden <= 8
    assert step.model.num_layers == 1
    assert step.checkpoint is None


def test_exp_smoke_trains_and_leaves_none_of_its_steps_threads_behind(
    tmp_path: Path,
) -> None:
    """exp_smoke's loop builds on the CPU and its step trains an epoch.

    The loop then closes the step: its rollout, worker and env threads stop
    (LIFE-1). Rollouts of 2 steps, not 16, in one window, and a pool of one
    world, not 64, whose generation took 43 ms on x86: every other part of an
    epoch is the tiny step's, its golden's and its unit tests'.
    """
    before = set(threading.enumerate())
    config = exp_smoke()
    config.base_dir = tmp_path
    pool = config.step.env.restart
    assert isinstance(pool, WorldPool.Config)
    pool.num_worlds = 1
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
    config.step.rollout.horizon = 2
    windows.minibatch_size = 2 * config.step.env.num_envs
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    loop = config.make()
    try:
        loop.step.train_step()
        assert loop.step.global_step == 1
    finally:
        loop.close()
    leftover = [thread for thread in threading.enumerate() if thread not in before]
    assert not leftover, leftover


def test_exp010_swaps_in_craftax_baselines_learner_and_geometry() -> None:
    """The recipe is one treatment; a swapped slot counts once, its fields its own."""
    swapped = {
        "step.model": ActorCritic.Config,
        "step.optimizer": PartialConfig,
        "step.schedule": ProgressSchedule.Config,
        "step.learner": ShuffledTransitions.Config,
    }
    deltas = {
        name
        for name in config_deltas(exp009(), exp010())
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
    config = exp010()
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


def test_exp010_runs_the_references_budget_and_minibatches() -> None:
    """15,258 updates of 1,024 x 64, each four passes of eight minibatches of 8,192."""
    config = exp010().copy_tree().finalize()
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


def test_exp010_at_test_size_builds_its_loop_and_evaluates_on_the_cpu(
    tmp_path: Path,
) -> None:
    """exp010's recipe on a network of 8 and 4 environments: its loop and an eval.

    Its training is the golden below's.
    """
    config = exp010()
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
    config.step.env.num_envs = 4
    config.step.env.num_buffers = 2
    config.step.env.threads_per_buffer = 1
    config.step.rollout.horizon = 2
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    _builds_and_evaluates(config)


def test_exp011_takes_the_1m_geometry_rate_and_budget_and_nothing_else() -> None:
    """256 x 16 steps, Adam at 3e-4, 244 updates: one treatment, the rest exp010's."""
    assert config_deltas(exp010(), exp011()) == {
        "experiment_name",
        "step.env.num_envs",
        "step.rollout.horizon",
        "step.optimizer",
        "step.train_budget_steps",
        "max_steps",
    }
    config = exp011().copy_tree().finalize()
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
    # The evaluation plays the training geometry, as exp010's does.
    assert step.evaluation.env is not None
    assert step.evaluation.rollout is not None
    assert (step.evaluation.env.num_envs, step.evaluation.rollout.horizon) == (256, 16)


def test_exp014_changes_only_the_budget_and_the_schedules_horizon() -> None:
    """1,525 updates of 1,024 x 64; the anneal spans them, not exp010's 15,258."""
    assert config_deltas(exp010(), exp014()) == {
        "experiment_name",
        "step.train_budget_steps",
        "max_steps",
    }
    config = exp014()
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert (config.max_steps, config.max_steps * transitions) == (1_525, 99_942_400)
    assert config.step.train_budget_steps == config.max_steps


def test_exp010s_epoch_matches_its_golden() -> None:
    """exp010's learner at test size from portable weights, an epoch, frozen.

    Its rate, Adam behind the global clip, shuffled minibatches and exp009's
    rules in fresh worlds, on 4 environments of 2 buffers, rollouts of 3 and
    towers of one layer of 8: the epoch's rate, its mean loss terms and every
    weight. One pass of two minibatches, not four of eight, as each update
    costs 30 ms in host-agnostic numerics on x86; the loop is the passes'
    own. exp011 and exp014 run the same code at another rate and budget,
    which their delta tests pin. Built natively, as the portable draws
    overwrite every weight the build draws: in host-agnostic numerics the
    build cost a tenth of the test.
    """
    step = _tiny_baseline_step(exp010()).make()
    with host_agnostic_pipeline():
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
    assert_golden(test_file=__file__, name="exp010_tiny", lines=lines)


def test_exp013_keeps_exp010s_setup_around_the_q_learner() -> None:
    """Outside the step only the name and the budget move; inside, exp010's env and streams."""
    parent, config = exp010(), exp013()
    outside = {
        name
        for name in config_deltas(parent, config)
        if name != "step" and not name.startswith("step.")
    }
    assert outside == {"experiment_name", "max_steps"}
    assert config_fields(config.step.env) == config_fields(parent.step.env)
    sampler = config.step.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    assert config_fields(sampler.sampler) == config_fields(parent.step.sampler)
    assert config.step.seed == 42


def test_exp013_runs_the_references_budget_and_recipe() -> None:
    """7,629 updates of 1,024 x 128, each four passes of four minibatches of 256."""
    config = exp013().copy_tree().finalize()
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


def test_exp013_at_test_size_builds_its_loop_and_evaluates_on_the_cpu(
    tmp_path: Path,
) -> None:
    """exp013's recipe on a network of 5 and 4 environments: its loop and an eval."""
    config = exp013()
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
    env.num_envs = 4
    env.num_buffers = 2
    env.threads_per_buffer = 1
    config.step.rollout.horizon = 2
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    sampler = config.step.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    sampler.sampler = TorchPhiloxSampler.Config().update(
        sampler.sampler,
        skip_missing=True,
    )
    _builds_and_evaluates(config)


def test_exp012_swaps_in_the_gru_and_whole_trajectories_and_nothing_else() -> None:
    """The policy and the minibatches are one treatment; the learner keeps exp010's fields."""
    swapped = {
        "step.model": ActorCriticRNN.Config,
        "step.learner": ShuffledTrajectories.Config,
    }
    assert {
        name
        for name in config_deltas(exp010(), exp012())
        if not name.startswith("step.model.")
    } == {"experiment_name", *swapped}
    config = exp012().copy_tree().finalize()
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


def test_exp012_at_test_size_builds_its_loop_and_evaluates_on_the_cpu(
    tmp_path: Path,
) -> None:
    """exp012's recipe on a GRU of 8 and 4 environments: its loop and an eval.

    Two minibatches, not eight, so each would replay two agents' trajectories.
    """
    config = exp012()
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
    learner = config.step.learner
    assert isinstance(learner, ShuffledTrajectories.Config)
    learner.num_minibatches = 2
    config.step.env.num_envs = 4
    config.step.env.num_buffers = 2
    config.step.env.threads_per_buffer = 1
    config.step.rollout.horizon = 2
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    _builds_and_evaluates(config)


def test_exp015_swaps_in_the_gtrxl_policy_its_windows_and_longer_rollouts() -> None:
    """The recipe is one treatment; a swapped slot counts once, its fields its own."""
    swapped = {
        "step.model": GTrXLPolicy.Config,
        "step.learner": TrajectoryWindows.Config,
    }
    deltas = {
        name
        for name in config_deltas(exp010(), exp015())
        if not name.startswith(tuple(f"{slot}." for slot in swapped))
    }
    assert deltas == {
        "experiment_name",
        *swapped,
        "step.rollout.horizon",
        "step.train_budget_steps",
        "max_steps",
    }
    config = exp015()
    for path, kind in swapped.items():
        assert isinstance(_at(config, path), kind), path


def test_exp015_runs_the_references_geometry_and_budget() -> None:
    """transformerXL_PPO_JAX's network and learner, 7,629 updates of 1,024 x 128."""
    config = exp015().copy_tree().finalize()
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


def test_exp015_at_test_size_builds_its_loop_and_evaluates_on_the_cpu(
    tmp_path: Path,
) -> None:
    """exp015's recipe at width 4 over 4 environments: its loop and an eval."""
    config = exp015()
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
    config.step.env.num_envs = 4
    config.step.env.num_buffers = 2
    config.step.env.threads_per_buffer = 1
    config.step.rollout.horizon = 2
    config.step.evaluation.num_episodes = 1
    config.max_steps = config.step.train_budget_steps = 2
    config.step.sampler = TorchPhiloxSampler.Config().update(
        config.step.sampler,
        skip_missing=True,
    )
    _builds_and_evaluates(config)


def test_exp002_changes_the_policy_the_rule_the_rate_and_the_budget() -> None:
    """One bundled fork: the model's three new stages, the recipe's coefficients."""
    model_stages = (
        "step.model.embedding",
        "step.model.injection",
        "step.model.auxiliary",
    )
    assert {
        name
        for name in config_deltas(exp001(), exp002())
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
    config = exp002().copy_tree().finalize()
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


def test_exp002_trains_20b_transitions_evaluating_every_1b_at_any_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both budgets are transitions over the epoch's: exact at one transition an epoch."""

    def one_transition_an_epoch() -> CraftaxTrainLoop:
        cfg = exp001()
        cfg.step.env.num_envs = cfg.step.rollout.horizon = 1
        return cfg

    monkeypatch.setattr(experiments, "exp001", one_transition_an_epoch)
    cfg = experiments.exp002()
    assert (cfg.max_steps, cfg.num_steps_eval) == (20_000_000_000, 1_000_000_000)


def test_exp003_adds_practice_imitation_the_cap_div8_and_the_win() -> None:
    assert config_deltas(exp002(), exp003()) == {
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
    config = exp003().copy_tree().finalize()
    assert (config.step.reward_scale, config.step.reward_clip) == (0.125, math.inf)


def test_exp003_evaluates_uncapped_without_practice_ending_on_the_win() -> None:
    config = exp003().copy_tree().finalize()
    evaluation = config.step.evaluation.env
    assert evaluation is not None
    assert evaluation.stall_cap is None
    assert evaluation.practice is None
    assert evaluation.rules == config.step.env.rules
    assert evaluation.rules.end_on_boss_defeat
    assert evaluation.rules.previous_action


def test_exp008_trains_in_fresh_worlds_and_evaluates_as_exp003_does() -> None:
    """Only training's world source changes; the evaluation keeps exp003's pool."""
    restart, evaluation = "step.env.restart", "step.evaluation.env"
    assert {
        name
        for name in config_deltas(exp003(), exp008())
        if not name.startswith((f"{restart}.", f"{evaluation}."))
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        restart,
        evaluation,
    }
    config = exp008().copy_tree().finalize()
    assert isinstance(config.step.env.restart, FreshWorlds.Config)
    assert config.step.env.practice is not None
    assert config.step.env.stall_cap is not None
    parent = exp003().copy_tree().finalize()
    assert config.step.evaluation.env == parent.step.evaluation.env


def test_exp006_doubles_the_width_and_the_window_and_keeps_the_rollout() -> None:
    assert config_deltas(exp002(), exp006()) == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        "step.model.channels_hidden",
        "step.env.num_envs",
        "step.rollout.horizon",
    }
    config, parent = exp006().copy_tree().finalize(), exp002().copy_tree().finalize()
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


def test_exp007_swaps_the_boards_blocks_for_one_convnext_trunk() -> None:
    board = "step.model.embedding.board"
    assert {
        name
        for name in config_deltas(exp006(), exp007())
        if not name.startswith(f"{board}.block.")
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        f"{board}.block",
        f"{board}.num_layers",
    }
    model = exp007().copy_tree().finalize().step.model
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


def test_base_dir_moves_the_steps_inputs(tmp_path: Path) -> None:
    """One override relocates a step's checkpoint."""
    config = exp000()
    config.base_dir = tmp_path
    config.step.checkpoint = "/runs/craftax/init.pt"
    assert config.copy_tree().finalize().step.checkpoint == (
        tmp_path / "runs/craftax/init.pt"
    )


def test_exp004_fine_tunes_exp003s_final_run_for_the_boss_fight() -> None:
    """The run's masters, the fight's reward, a longer horizon, a tenth of the rate."""
    parent, config = exp003(), exp004()
    assert {
        name
        for name in config_deltas(parent, config)
        if not name.startswith("step.env.boss_fight_reward.")
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        "step.checkpoint",
        "step.env.boss_fight_reward",
        "step.learner.objective.discount",
        "step.optimizer.lr",
        "step.schedule.curve",
        "max_steps",
        "step.train_budget_steps",
    }
    final = config.copy_tree().finalize()
    assert final.step.checkpoint == Path(
        "/opt/scratch/runs/craftax/exp003/checkpoints/step_00038146.pt",
    )
    assert final.step.env.boss_fight_reward == BossFightReward.Config(hit=1.0, kill=1.5)
    # The evaluation scores the game's reward alone.
    evaluation = final.step.evaluation.env
    assert evaluation is not None
    assert evaluation.boss_fight_reward is None
    optimizer, parent_optimizer = config.step.optimizer, parent.step.optimizer
    assert isinstance(optimizer, FusedMuon.Config)
    assert isinstance(parent_optimizer, FusedMuon.Config)
    # A tenth of the parent's peak, as the arm's command wrote it.
    assert parent_optimizer.lr == 0.001182217733003199
    assert optimizer.lr == 0.0001182217733003199
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert (config.max_steps, config.max_steps * transitions) == (5_722, 2_999_975_936)


def test_exp004_reads_the_file_exp003s_checkpointer_writes_last(tmp_path: Path) -> None:
    """exp003's run ends on a save at its last epoch; exp004 reads that file.

    Both beneath one ``base_dir``, as one override puts every job of a chain.
    """
    parent, child = exp003(), exp004()
    parent.base_dir = child.base_dir = tmp_path
    parent, child = parent.copy_tree().finalize(), child.copy_tree().finalize()
    checkpointer = parent.checkpointer
    assert isinstance(checkpointer, Checkpointer.Config)
    written = Path(checkpointer.working_dir) / checkpointer.filename.format(
        step=parent.max_steps,
    )
    assert child.step.checkpoint == written
    assert written == tmp_path / "runs/craftax/exp003/checkpoints/step_00038146.pt"


def test_exp004_starts_from_the_masters_of_an_exp003_steps_checkpoint(
    tmp_path: Path,
) -> None:
    """exp003's step at test size, saved as the training loop saves it, starts exp004.

    The masters carry over exactly; the momentum does not, as the fine-tune's
    optimizer starts afresh. The parent's momentum is set to ones, as a run's
    is after training, so a carried buffer would show; its masters are drawn
    at random and loaded, which leaves its optimizer holding them. exp004's
    model and optimizer load the file as its step's constructor does; the
    parent's kernels run eagerly, on tiny worlds.
    """
    generator = torch.Generator().manual_seed(0)
    config = _tiny_recipe_step(exp003())
    masters = {
        name: torch.randn(parameter.shape, generator=generator)
        for name, parameter in config.model.make().named_parameters()
    }
    torch.save(masters, tmp_path / "masters.pt")
    config.checkpoint = tmp_path / "masters.pt"
    with eager(world=tiny_world):
        parent = config.make()
        try:
            state = optimizer_state(parent.model, parent.optimizer, "momentum_buffer")
            for buffer in state.values():
                buffer.fill_(1.0)
            held = optimizer_state(parent.model, parent.optimizer, "master_weight")
            torch.save({"step": parent.state_dict()}, tmp_path / "run.pt")
        finally:
            parent.close()
    config = _tiny_recipe_step(exp004())
    model = config.model.make()
    optimizer = config.optimizer.make()(model.parameters())
    load_masters(model, optimizer, tmp_path / "run.pt")
    loaded = optimizer_state(model, optimizer, "master_weight")
    assert loaded.keys() == held.keys()
    for name, master in loaded.items():
        assert torch.equal(master, held[name]), name
    momentum = optimizer_state(model, optimizer, "momentum_buffer")
    assert not any(buffer.any() for buffer in momentum.values())


@pytest.mark.parametrize(
    ("factory", "group"),
    [
        (exp000, ""),
        (exp002, "exp002"),
        (exp003, "exp003"),
        (exp004, "exp004"),
        (exp006, "exp006"),
        (exp007, "exp007"),
        (exp008, "exp008"),
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


def test_exp010_matches_its_golden_config() -> None:
    """Pin the whole finalized ``exp010`` as readable text, every default shown."""
    assert_pprint_golden(test_file=__file__, name="exp010", config=exp010())


def test_exp003_matches_its_golden_config() -> None:
    """Pin the whole finalized ``exp003`` as readable text, every default shown."""
    assert_pprint_golden(test_file=__file__, name="exp003", config=exp003())


# Training a recipe at test size is its learner's golden's: exp010's here, exp012's in
# ``rnn_update_test``, exp013's in ``pqn_train_step_test``, exp015's in
# ``gtrxl_train_step_test``.
def _builds_and_evaluates(config: CraftaxTrainLoop | CraftaxPQNTrainLoop) -> None:
    """Build a recipe's loop, play one rollout of its evaluation, then close it."""
    loop = config.make()
    try:
        step = loop.step
        assert isinstance(step, CraftaxTrainStep | CraftaxPQNTrainStep)
        evaluator = step.make_evaluator()
        try:
            evaluator.collect()
            assert evaluator.gameplay_seconds > 0
        finally:
            evaluator.close()
    finally:
        loop.close()


# exp003's recipe at test size: the board policy at width 8, 4 environments in 2
# buffers of 8 pool worlds, no practice or stall cap, imitation of 2 rows, windows of
# every agent over a horizon of 2, and the torch forms of the kernels.
def _tiny_recipe_step(config: CraftaxTrainLoop) -> CraftaxTrainStep.Config:
    """Shrink a board recipe's step to the CPU."""
    step = config.step
    step.parallelism.device = "cpu"
    env = step.env
    env.num_envs = 4
    env.num_buffers = 2
    env.threads_per_buffer = 1
    pool = env.restart
    assert isinstance(pool, WorldPool.Config)
    pool.num_worlds = 8
    env.practice = env.stall_cap = None
    model = step.model = tiny_board_policy()
    model.block.scan = TorchScan.Config()
    windows = step.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.objective = TorchPPO.Config().update(windows.objective, skip_missing=True)
    windows.minibatch_size = 8
    imitation = windows.auxiliary
    assert isinstance(imitation, BranchImitation.Config)
    imitation.rows = imitation.capacity = 2
    step.sampler = TorchPhiloxSampler.Config().update(step.sampler, skip_missing=True)
    step.rollout.horizon = 2
    step.train_budget_steps = 2
    return step


# 4 environments of 2 buffers by 3 steps: 12 transitions in two minibatches of 6, and
# none of the sizes that meet ties another. One pass, not the recipe's four of eight
# minibatches, and towers of one layer, not three: each update costs 30 ms in
# host-agnostic numerics on x86; ``actor_critic_test`` stacks the layers. Two buffers:
# host-agnostic numerics store a one-buffer bootstrap rollout's log-probabilities as
# zeros.
def _tiny_baseline_step(config: CraftaxTrainLoop) -> CraftaxTrainStep.Config:
    """Shrink a Craftax_Baselines PPO recipe's step to the CPU and one epoch of one pass."""
    step = config.step
    step.parallelism.device = "cpu"
    model = step.model
    assert isinstance(model, ActorCritic.Config)
    model.channels_hidden = 8
    model.num_layers = 1
    step.env.num_envs = 4
    step.env.num_buffers = 2
    step.env.threads_per_buffer = 1
    step.rollout.horizon = 3
    learner = step.learner
    assert isinstance(learner, ShuffledTransitions.Config)
    learner.num_passes = 1
    learner.num_minibatches = 2
    step.sampler = TorchPhiloxSampler.Config().update(step.sampler, skip_missing=True)
    step.train_budget_steps = 1
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


def test_exp000_exp010_input_diff() -> None:
    result = exp000().udiff(exp010(), finalize=False, n=0, width=120, color=False)
    assert re.search(r"^-\s*[│\s]*experiment_name='exp000',$", result, re.MULTILINE)
    assert re.search(r"^\+\s*[│\s]*experiment_name='exp010',$", result, re.MULTILINE)
    assert re.search(r"^-.*\bhorizon=256[,)]", result, re.MULTILINE)
    assert re.search(r"^\+.*\bhorizon=64[,)]", result, re.MULTILINE)
    sys.stdout.write(result + "\n")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
