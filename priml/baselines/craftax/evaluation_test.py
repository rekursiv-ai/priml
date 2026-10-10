"""Tests for the evaluation trainer: fresh, isolated, deterministic, and its stop.

Goldens freeze whole evaluations of the tiny pipeline, played and then
scored, the same bits on every CPU: exp000's recipe from portable seed-73
weights, over three rollouts and over one, and exp002's from its own init.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from priml.baselines.craftax.env import BossFightReward, StallCap
from priml.baselines.craftax.evaluation import Evaluation
from priml.baselines.craftax.game.state import LOG_DTYPE
from priml.baselines.craftax.learners.practice import FrontierPractice
from priml.baselines.craftax.lib.arrays import typed
from priml.baselines.craftax.metric import CraftaxScore
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.rollout import Rollout, TorchPhiloxSampler
from priml.baselines.craftax.testing import (
    assert_golden,
    digest,
    fill_portable,
    fp32,
    host_agnostic_pipeline,
    smoke_feature,
    tiny_env,
    tiny_exp002_step,
    tiny_policy,
    tiny_train_step,
)
from priml.model.linear import Linear


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.craftax.model import Policy
    from priml.baselines.craftax.world_model.feature import (
        FeatureEngine,
        WorldModelFeature,
    )


_HORIZON = 3


def _config() -> Evaluation.Config:
    config = Evaluation.Config()
    config.env = tiny_env()
    config.sampler = TorchPhiloxSampler.Config()
    rollout = config.rollout = Rollout.Config()
    rollout.num_slots = 1
    rollout.horizon = _HORIZON
    return config


def _evaluation(policy: MinGRUPolicy) -> Evaluation:
    return Evaluation(_config(), policy=policy, device=torch.device("cpu"))


@pytest.fixture
def policy() -> MinGRUPolicy:
    torch.manual_seed(0)
    return tiny_policy().make()


def test_an_evaluation_is_an_evaluator(policy: MinGRUPolicy) -> None:
    evaluation = _evaluation(policy)
    evaluation.close()


def test_an_evaluation_leaves_the_training_environment_untouched(
    policy: MinGRUPolicy,
) -> None:
    training = tiny_env().make()
    training.reset()
    training.step_buffer(0)
    # Through uint8: a structured copy moves the fields but not the padding
    # between them, so its bytes would differ from the original's.
    states, rngs = training.states.view(np.uint8).copy(), training.rngs.copy()

    evaluation = _evaluation(policy)
    evaluation.reset()
    evaluation.collect()

    assert np.array_equal(training.states.view(np.uint8), states)
    assert np.array_equal(training.rngs, rngs)
    evaluation.close()
    training.close()


def test_fresh_evaluations_replay_identically(policy: MinGRUPolicy) -> None:
    first, second = _evaluation(policy), _evaluation(policy)
    for evaluation in (first, second):
        evaluation.reset()
        evaluation.collect()

    assert first.env.states.tobytes() == second.env.states.tobytes()
    assert first.logs.tobytes() == second.logs.tobytes()
    first.close()
    second.close()


def test_collect_steps_every_environment_one_horizon(policy: MinGRUPolicy) -> None:
    evaluation = _evaluation(policy)
    evaluation.reset()
    evaluation.collect()

    assert np.equal(evaluation.env.stats["steps"], _HORIZON).all()
    assert evaluation.gameplay_seconds > 0
    evaluation.close()


def test_reset_wipes_the_logs_and_the_clock(policy: MinGRUPolicy) -> None:
    evaluation = _evaluation(policy)
    evaluation.collect()
    evaluation.env.stats["log"]["n"] = 5.0
    evaluation.env.stats["log"]["perf"] = 1.0

    evaluation.reset()

    assert evaluation.logs.tobytes() == np.zeros(4, dtype=LOG_DTYPE).tobytes()
    assert evaluation.gameplay_seconds == 0.0
    evaluation.close()


def test_a_reset_evaluation_plays_as_a_fresh_one(policy: MinGRUPolicy) -> None:
    """``reset`` restarts the environments, streams and carry, not only the logs."""
    reused, fresh = _evaluation(policy), _evaluation(policy)
    reused.collect()
    reused.reset()
    assert np.equal(reused.env.stats["steps"], 0).all()
    for evaluation in (reused, fresh):
        evaluation.collect()
    assert reused.env.states.tobytes() == fresh.env.states.tobytes()
    assert reused.logs.tobytes() == fresh.logs.tobytes()
    assert torch.equal(
        reused._rollout.slots[0].actions,
        fresh._rollout.slots[0].actions,
    )
    reused.close()
    fresh.close()


def test_play_stops_at_the_first_rollout_reaching_the_episode_count(
    policy: MinGRUPolicy,
) -> None:
    """Whole rollouts from a fresh start, stopping as soon as the count is reached."""
    config = _config()
    env = config.env = tiny_env()
    env.rules.max_timesteps = 4
    probe = Evaluation(config, policy=policy, device=torch.device("cpu"))
    try:
        probe.reset()
        probe.collect()
        after_one = int(np.sum(typed(probe.logs["n"], np.float32)))
        probe.collect()
        after_two = int(np.sum(typed(probe.logs["n"], np.float32)))
    finally:
        probe.close()
    assert after_one < after_two
    config.num_episodes = after_two
    evaluation = Evaluation(config, policy=policy, device=torch.device("cpu"))
    try:
        evaluation.collect()
        played = evaluation.play()
        again = evaluation.play()
    finally:
        evaluation.close()
    assert played.rollouts == 2
    assert int(np.sum(typed(played.logs["n"], np.float32))) == after_two
    assert played.gameplay_seconds > 0
    # Each play starts afresh, whatever ran before it.
    assert again.rollouts == played.rollouts
    assert again.logs.tobytes() == played.logs.tobytes()


def test_an_evaluation_needs_every_part_set(policy: MinGRUPolicy) -> None:
    """Only a train step's finalize fills the unset parts; built alone, it refuses."""
    with pytest.raises(ValueError, match="env, sampler and rollout"):
        Evaluation(Evaluation.Config(), policy=policy, device=torch.device("cpu"))


@pytest.mark.parametrize("option", ["stall_cap", "practice", "boss_fight_reward"])
def test_an_evaluation_refuses_a_training_only_option(option: str) -> None:
    """Scored episodes are natural and uncapped, as the recipe's evaluator plays them."""
    config = _config()
    assert config.env is not None
    Evaluation.check(config)
    if option == "stall_cap":
        config.env.stall_cap = StallCap.Config()
    elif option == "practice":
        practice = config.env.practice = FrontierPractice.Config()
        practice.num_donors = 2
    else:
        config.env.boss_fight_reward = BossFightReward.Config()
    with pytest.raises(ValueError, match="training-only"):
        Evaluation.check(config)


def test_with_a_feature_an_evaluation_holds_one_history_per_environment_of_its_own() -> (
    None
):
    """Its engines' rows are its own env's, whatever training's, and closing frees them.

    So a recipe whose training caches fill the device evaluates on fewer
    environments: here 2, in 2 buffers, against :func:`tiny_env`'s 4.
    """
    source = _RecordingFeature(smoke_feature().make())
    config = _config()
    assert config.env is not None
    config.env.num_envs = 2
    policy_config = tiny_policy()
    proj = policy_config.proj_feature = Linear.Config()
    proj.channels_in = source.width
    torch.manual_seed(0)
    evaluation = Evaluation(
        config,
        policy=policy_config.make(),
        device=torch.device("cpu"),
        feature=source,
    )
    try:
        evaluation.collect()
        assert all(engine.keys.numel() for engine in source.engines)
    finally:
        evaluation.close()
    assert [len(engine.length) for engine in source.engines] == [1, 1]
    assert not any(engine.keys.numel() for engine in source.engines)


def test_exp000s_evaluation_from_portable_weights_matches_its_golden() -> None:
    """The tiny pipeline's evaluation from portable seed-73 weights, frozen on every host.

    exp000's recipe in its torch forms, its evaluation config as the step's
    finalize fills it, with rollouts of 2 steps and episodes cut at 4 ticks so
    they finish: whole rollouts until 4 episodes finish, two of them, so the
    second plays on from the first's carries and worlds.
    """
    config = tiny_train_step()
    config.rollout.horizon = 2
    config = config.copy_tree().finalize()
    with host_agnostic_pipeline():
        policy = config.model.make()
        assert isinstance(policy, MinGRUPolicy)
        fill_portable(policy, seed=73)
        lines = _score_entries(config.evaluation, policy, episodes=4, ticks=4)
    assert_golden(test_file=__file__, name="evaluation_tiny", lines=lines)


def test_exp002s_short_evaluation_matches_its_golden() -> None:
    """exp002's evaluation at test size, of the policy's own init under torch seed 0.

    Frozen on every host. From scratch, so the golden moves with the init's
    draws: their order, their distributions, or torch's generator. One
    rollout of 2 steps, every episode cut at 2 ticks.
    """
    config = tiny_exp002_step()
    config.rollout.horizon = 2
    config = config.copy_tree().finalize()
    with host_agnostic_pipeline():
        torch.manual_seed(0)
        policy = config.model.make()
        lines = ["# from-scratch: the policy's own init under torch seed 0"]
        lines += _score_entries(config.evaluation, policy, episodes=4, ticks=2)
    assert_golden(
        test_file=__file__,
        name="evaluation_exp002_tiny_short",
        lines=lines,
    )


class _RecordingFeature:
    """A world model's feature source that keeps every engine it makes."""

    def __init__(self, source: WorldModelFeature) -> None:
        self.source = source
        self.width = source.width
        self.hook_interval = source.hook_interval
        self.joint = source.joint
        self.context_decisions = source.context_decisions
        self.engines: list[FeatureEngine] = []

    def make_engine(self, *, rows: int, device: torch.device) -> FeatureEngine:
        """Make the source's engine and keep it."""
        engine = self.source.make_engine(rows=rows, device=device)
        self.engines.append(engine)
        return engine

    def history_archive(
        self,
        entries: int,
        *,
        device: torch.device,
    ) -> dict[str, Tensor] | None:
        """Return the source's archive of practice histories."""
        return self.source.history_archive(entries, device=device)


# Every metric but the wall time, as fp32; then every environment's log.
def _score_entries(
    config: Evaluation.Config,
    policy: Policy,
    *,
    episodes: int,
    ticks: int,
) -> list[str]:
    """Play and score ``policy`` as training's final evaluation does; digest the result."""
    assert config.env is not None
    config.env.rules.max_timesteps = ticks
    config.num_episodes = episodes
    evaluation = Evaluation(config, policy=policy, device=torch.device("cpu"))
    try:
        played = evaluation.play()
    finally:
        evaluation.close()
    score = CraftaxScore.Config().make()
    score.update(played=played)
    lines: list[str] = []
    for key, value in score.compute().items():
        assert isinstance(value, float)
        if key != "gameplay_seconds":
            lines += [f"{key} {fp32(value)}"]
    return [*lines, f"logs {digest(played.logs)}"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
