"""Tests for the Craftax world-model experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import inspect
import math
import re
import warnings

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.craftax import experiments, experiments_world_model
from priml.baselines.craftax.env import WorldPool
from priml.baselines.craftax.experiments_world_model import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
)
from priml.baselines.craftax.learners.imitation import BranchImitation
from priml.baselines.craftax.model import MinGRUPolicy, NoEncoder
from priml.baselines.craftax.rollout import TorchPhiloxSampler
from priml.baselines.craftax.testing import config_deltas
from priml.baselines.craftax.train_step import (
    AgentWindows,
    CraftaxTrainStep,
    ProgressSchedule,
)
from priml.baselines.craftax.world_model.context import ContextReplay
from priml.baselines.craftax.world_model.feature import (
    Flash4CacheAttention,
    FreshWindow,
    InitialWeights,
    MaskedCacheAttention,
    Refill,
    Sliding,
    TrainedWeights,
    WorldModelFeature,
)
from priml.loss.policy_gradient import TorchPPO
from priml.math.schedules import warmup
from priml.model.attention.kernel import SdpaVarlen
from priml.model.linear import Linear
from priml.model.min_gru import TorchScan
from priml.train.tracker import AsyncTracker, TrackerList, WandbTracker


if TYPE_CHECKING:
    from priml.baselines.craftax.experiments import CraftaxTrainLoop
    from priml.testing.experiments import ExperimentFactory


_WORLD_MODEL_EXP001: Final = "priml.baselines.craftax.world_model.experiments.exp001"

_BUILT: Final[tuple[ExperimentFactory[CraftaxTrainLoop], ...]] = (
    exp000,
    exp001,
    exp002,
)
"""Experiments whose inputs a step here produces, each built and printed below."""

_NOT_REPRODUCIBLE: Final[tuple[ExperimentFactory[CraftaxTrainLoop], ...]] = (
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
)
"""Experiments whose world model no experiment trains: each factory raises."""


def test_every_experiment_is_built_or_waits_for_its_producer() -> None:
    """Every factory the module defines is in one of the lists the tests take."""
    defined = {
        name
        for name, function in inspect.getmembers(
            experiments_world_model,
            inspect.isfunction,
        )
        if name.startswith("exp")
        and function.__module__ == experiments_world_model.__name__
    }
    assert {factory.__name__ for factory in (*_BUILT, *_NOT_REPRODUCIBLE)} == defined


def test_the_readme_links_every_experiment_to_its_factory() -> None:
    """Every factory has a results row, linked to the module that defines it."""
    readme = (
        Path(experiments_world_model.__file__)
        .with_name("README.md")
        .read_text(encoding="utf-8")
    )
    linked = set(re.findall(r"\[`(exp\d+)`\]\(experiments_world_model\.py\)", readme))
    assert linked == {factory.__name__ for factory in (*_BUILT, *_NOT_REPRODUCIBLE)}
    assert "experiments_world_model.py#L" not in readme


@pytest.mark.parametrize("factory", _NOT_REPRODUCIBLE)
def test_an_experiment_without_its_world_model_raises_its_todo(
    factory: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    """Each names the producer it waits for, its own or its parent's."""
    with pytest.raises(NotImplementedError, match="TODO"):
        factory()
    assert "Not yet reproducible" in (factory.__doc__ or "")


@pytest.mark.parametrize("factory", _BUILT)
def test_every_experiment_is_named_for_its_factory_and_keeps_the_recipes_seed(
    factory: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    config = factory()
    assert (config.experiment_name, config.seed) == (factory.__name__, 73)


@pytest.mark.parametrize("factory", _BUILT)
def test_every_config_finalizes_and_prints_without_a_warning(
    factory: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factory().copy_tree().finalize()
        factory().pformat()


@pytest.mark.parametrize(
    ("factory", "recipe"),
    [
        (exp000, experiments.exp000),
        (exp001, experiments.exp001),
        (exp002, experiments.exp002),
    ],
)
def test_a_run_shares_its_name_but_not_its_directory_or_board_with_a_recipe(
    factory: ExperimentFactory[CraftaxTrainLoop],
    recipe: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    """A recipe of ``experiments`` has each name; under its study, a run would resume it.

    A run resumes the newest checkpoint in its working directory, which its
    study and name decide, so a launch here would continue the recipe's run.
    """
    ours, theirs = factory().copy_tree().finalize(), recipe().copy_tree().finalize()
    assert ours.experiment_name == theirs.experiment_name
    assert ours.working_dir != theirs.working_dir
    assert ours.working_dir == Path(
        f"/opt/scratch/runs/craftax-world-model-rl/{factory.__name__}",
    )
    projects = [_dashboard(config).project for config in (ours, theirs)]
    assert projects == ["craftax-world-model-rl", "craftax"]


@pytest.mark.parametrize("factory", _BUILT)
def test_the_dashboard_names_the_run_after_the_experiment_and_groups_its_seeds(
    factory: ExperimentFactory[CraftaxTrainLoop],
) -> None:
    for name in (factory.__name__, f"{factory.__name__}_s74"):
        config = factory().copy_tree()
        config.experiment_name = name
        dashboard = _dashboard(config.finalize())
        assert (dashboard.name, dashboard.group) == (name, factory.__name__)


def test_exp000_trains_250m_inside_the_20b_warmup_evaluating_every_50m() -> None:
    assert config_deltas(experiments.exp007(), exp000()) == {
        "study_name",
        "experiment_name",
        # The W&B project and group, inside the tracker list's dict.
        "tracker.trackers",
        "max_steps",
        "num_steps_eval",
        "step.schedule.curve",
    }
    config = exp000().copy_tree().finalize()
    transitions = config.step.env.num_envs * config.step.rollout.horizon
    assert config.max_steps == 476
    epochs = int(config.max_steps)
    assert epochs * transitions == 249_561_088
    assert (
        config.step.train_budget_steps == experiments.exp007().step.train_budget_steps
    )
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


def test_base_dir_moves_the_weights_exp001_reads(tmp_path: Path) -> None:
    config = exp001()
    config.base_dir = tmp_path
    feature = config.copy_tree().finalize().step.feature
    assert isinstance(feature, WorldModelFeature.Config)
    assert isinstance(feature.weights, TrainedWeights.Config)
    assert feature.weights.checkpoint == (
        tmp_path / "runs/craftax-world-model/exp001/checkpoints/step_00001525.pt"
    )


def test_exp001_adds_the_frozen_feature_and_its_zero_projection() -> None:
    feature_path, proj_path = "step.feature", "step.model.proj_feature"
    assert {
        name
        for name in config_deltas(exp000(), exp001())
        if not name.startswith((f"{feature_path}.", f"{proj_path}."))
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        feature_path,
        proj_path,
    }
    config = exp001().copy_tree().finalize()
    assert config.step.feature == _frozen_feature(
        TrainedWeights.Config(
            experiment=_WORLD_MODEL_EXP001,
            # ``world_model.experiments.exp001``'s run, its final step: 50M
            # decisions / 32,768.
            checkpoint=Path(
                "/opt/scratch/runs/craftax-world-model/exp001/checkpoints/"
                "step_00001525.pt",
            ),
            overrides=[],
        ),
    )
    model = config.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    proj = model.proj_feature
    assert proj is not None
    assert (proj.channels_in, proj.channels_out, proj.bias) == (1_152, 2_048, False)
    assert proj.init_weight is torch.nn.init.zeros_


def test_exp002_swaps_only_the_feature_weights_for_a_seeded_init() -> None:
    weights = "step.feature.weights"
    assert {
        name
        for name in config_deltas(exp001(), exp002())
        if not name.startswith(f"{weights}.")
    } == {
        "experiment_name",
        # The W&B group, inside the tracker list's dict.
        "tracker.trackers",
        weights,
    }
    assert exp002().copy_tree().finalize().step.feature == _frozen_feature(
        InitialWeights.Config(experiment=_WORLD_MODEL_EXP001, overrides=[], seed=0),
    )


def test_a_sole_feature_step_archives_the_features_its_actor_stored_for_branches() -> (
    None
):
    """The sole-feature recipe on exact windows: width 8, 4 environments, the smoke model.

    Every row of the boot rollout's slot is marked a branch, as practice marks
    a restored row, and the imitation archives the features the actor stored
    for those rows. Practice's own part, the donor histories the step graph
    saves, is ``rollout_test``'s; an evaluation with a feature,
    ``evaluation_test``'s.
    """
    step = _tiny_sole_feature_step(_sole_feature_recipe()).make()
    try:
        step._boot()
        slot = step.rollout.slots[step.ready]
        assert slot.features is not None
        slot.branch_starts.fill_(1)
        stored = slot.features.transpose(0, 1).clone()
        _, metrics = step.learner(step, step._learner_rollout(slot))
        assert float(metrics["imitation/branches"]) == 2.0
        archived = step.learner.state_dict()["features"]
        assert archived.shape == (2, *stored.shape[1:])
        for branch in archived:
            assert any(torch.equal(branch, row) for row in stored[:2]), "not a row's"
    finally:
        step.close()


def test_a_joint_step_on_exact_windows_trains_beside_imitation() -> None:
    """The joint sole-feature recipe at the same test size, the replay masked and eager.

    The boot slot is the starting weights' own, so its replay matches what the
    actor read; the world model trains and the imitation archives two
    branches' stored features. The rebuild of the actor's histories and the
    evaluation of the trained weights are ``train_step_test``'s.
    """
    step_config = _tiny_sole_feature_step(_sole_feature_recipe())
    replay = step_config.feature_training = ContextReplay.Config()
    replay.attention = SdpaVarlen.Config()
    replay.compile = None
    replay.bin_tokens = 1
    # The default micro-batches, padded full, would be nearly all the replay's work.
    replay.pass_tokens = 26
    replay.frames_per_batch = 5
    step = step_config.make()
    try:
        assert step.joint is not None
        before = {name: w.clone() for name, w in step.joint.weights().items()}
        step._boot()
        slot = step.rollout.slots[step.ready]
        slot.branch_starts.fill_(1)
        _, metrics = step.learner(step, step._learner_rollout(slot))
        # The stored features are bf16, the policy's dtype: rounding alone.
        assert float(metrics["joint/feature_gap"]) < 0.01
        assert float(metrics["imitation/branches"]) == 2.0
        assert not all(
            torch.equal(weight, before[name])
            for name, weight in step.joint.weights().items()
        )
    finally:
        step.close()


def _dashboard(config: CraftaxTrainLoop) -> WandbTracker.Config:
    """Return a finalized loop's W&B tracker, inside its list and async wrapper."""
    assert isinstance(config.tracker, TrackerList.Config)
    wrapper = config.tracker.trackers["wandb"]
    assert isinstance(wrapper, AsyncTracker.Config)
    dashboard = wrapper.tracker
    assert isinstance(dashboard, WandbTracker.Config)
    return dashboard


def _frozen_feature(
    weights: TrainedWeights.Config | InitialWeights.Config,
) -> WorldModelFeature.Config:
    """Return exp001's and exp002's frozen world-model feature, every field stated."""
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


# exp003 waits for its world model's producer, so the pipeline's tests build its policy
# and feature here; :func:`_tiny_sole_feature_step` gives the feature the smoke model's
# weights.
def _sole_feature_recipe() -> CraftaxTrainLoop:
    """Return experiments.exp003 reading a world-model feature alone, as exp003 does."""
    config = experiments.exp003()
    model = config.step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.embedding = NoEncoder.Config()
    model.injection = None
    model.proj_feature = Linear.Config()
    config.step.feature = WorldModelFeature.Config()
    return config


# Width 8 in one layer; 4 environments in 2 buffers of 8 pool worlds, without practice,
# the stall cap or the previous action: the feature reads the rollout's own actions,
# and the environment's kernels then type as :func:`tiny_env`'s, which ``kernel_cache``
# compiles, where the recipe's options compile their own for seconds; imitation of rows
# 0-1 into an archive of 2; windows of every agent over a horizon of 2; the smoke world
# model, float32, on exact windows of 4 decisions, its blocks 2 steps; the torch forms
# of the kernels. No test builds its evaluation.
def _tiny_sole_feature_step(config: CraftaxTrainLoop) -> CraftaxTrainStep.Config:
    """Shrink a sole-feature recipe's step to the CPU and put it on exact windows."""
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
    env.rules.previous_action = False
    model = step.model
    assert isinstance(model, MinGRUPolicy.Config)
    model.channels_hidden = 8
    model.num_layers = 1
    model.block.scan = TorchScan.Config()
    encoder = model.embedding
    assert isinstance(encoder, NoEncoder.Config)
    encoder.observation_size = env.observation_size
    proj = model.proj_feature
    assert proj is not None
    proj.channels_in = 36  # The smoke world model's width.
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
    feature = step.feature
    assert isinstance(feature, WorldModelFeature.Config)
    weights = feature.weights = InitialWeights.Config()
    weights.experiment = "priml.baselines.craftax.world_model.experiments.exp_smoke"
    history = feature.history = Sliding.Config()
    history.decisions = 4
    feature.layers = 1
    feature.hook_interval = 2
    feature.attention = MaskedCacheAttention.Config()
    feature.compile = None
    feature.dtype = torch.float32
    return step


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
