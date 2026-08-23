"""Tests for the batched, auto-resetting environment."""

from __future__ import annotations

from unittest import mock

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.game import constants, observation, world_gen
from priml.baselines.craftax.game.state import EnvState
from priml.data.environment import BatchedEnvironmentProtocol


pytestmark = pytest.mark.compute_large_fixture


def _env(
    num_envs: int = 4,
    seed: int = 0,
    reset_ratio: int = 1,
    view: tuple[int, int] = (9, 11),
) -> CraftaxEnv:
    config = CraftaxEnv.Config()
    config.view = view
    config.num_envs = num_envs
    config.device = "cpu"
    config.seed = seed
    # One world per worker by default: most tests here assert on WHICH world a
    # restarted worker got, and sharing would make that ambiguous.
    config.optimistic_reset_ratio = reset_ratio
    return config.make()


def _actions(env: CraftaxEnv, count: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, env.num_actions, (count,), generator=generator)


def test_it_satisfies_the_environment_protocol() -> None:
    assert isinstance(_env(), BatchedEnvironmentProtocol)


def test_it_declares_the_published_geometry() -> None:
    env = _env()
    assert env.num_actions == 43
    assert env.observation_size == 8_268
    assert env.reward_ceiling == 226.0


def test_reset_returns_one_observation_per_worker() -> None:
    env = _env()
    rendered = env.reset()
    assert rendered.shape == (4, observation.observation_size())
    assert bool(torch.isfinite(rendered).all())


def test_reset_can_change_the_batch_size() -> None:
    env = _env()
    assert env.reset(7).shape == (7, observation.observation_size())


def test_stepping_returns_a_full_transition() -> None:
    env = _env()
    env.reset()
    transition = env.step(_actions(env, 4))
    assert transition.observation.shape == (4, observation.observation_size())
    assert transition.reward.shape == (4,)
    assert transition.done.shape == (4,)
    assert transition.done.dtype == torch.bool
    assert len(transition.info) == len(constants.Achievement)


def test_reading_the_world_before_reset_is_refused() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        _ = _env().state


def test_the_same_seed_replays_the_same_episode() -> None:
    def rollout() -> list[float]:
        env = _env(seed=5)
        env.reset()
        return [
            float(env.step(_actions(env, 4, index)).reward.sum()) for index in range(12)
        ]

    assert rollout() == rollout()


def test_different_seeds_give_different_worlds() -> None:
    assert not torch.equal(_env(seed=1).reset(), _env(seed=2).reset())


def test_a_finished_worker_restarts_without_disturbing_the_others() -> None:
    # This is what lets a rollout stay rectangular: the batch never shrinks
    # and the surviving workers keep their episodes.
    env = _env()
    env.reset()
    env.state.player_health[1] = 0.0
    env.state.timestep[:] = 25
    survivor = env.state.map[0].clone()

    transition = env.step(_actions(env, 4))

    assert transition.done.tolist() == [False, True, False, False]
    assert int(env.state.timestep[1]) == 0
    assert int(env.state.timestep[0]) == 26
    assert torch.equal(env.state.map[0], survivor)


def test_a_restarted_worker_gets_a_fresh_world() -> None:
    env = _env()
    env.reset()
    doomed = env.state.map[1].clone()
    env.state.player_health[1] = 0.0
    env.step(_actions(env, 4))
    assert not torch.equal(env.state.map[1], doomed)


def test_the_observation_after_a_restart_is_the_new_episode() -> None:
    # The terminal observation is deliberately not visible: the learner sees
    # where the next episode begins.
    env = _env()
    env.reset()
    env.state.player_health[1] = 0.0
    transition = env.step(_actions(env, 4))
    assert torch.equal(transition.observation[1], observation.render(env.state)[1])


def test_achievements_are_reported_only_when_an_episode_ends() -> None:
    env = _env()
    env.reset()
    env.state.achievements[:, int(constants.Achievement.COLLECT_WOOD)] = True

    quiet = env.step(_actions(env, 4))
    assert float(quiet.info["Achievements/collect_wood"].sum()) == 0.0

    env.state.achievements[:, int(constants.Achievement.COLLECT_WOOD)] = True
    env.state.player_health[0] = 0.0
    ending = env.step(_actions(env, 4))
    assert ending.info["Achievements/collect_wood"].tolist() == [100.0, 0.0, 0.0, 0.0]


def test_an_episode_ends_when_the_step_limit_is_reached() -> None:
    env = _env()
    env.reset()
    env.state.timestep[:] = constants.MAX_TIMESTEPS - 1
    assert env.step(_actions(env, 4)).done.all()


def test_a_long_rollout_stays_finite_and_rectangular() -> None:
    """Many steps in sequence keep the shape and stay numerically sane.

    A small view, because what is under test is that the rollout does not
    drift -- the batch never ragged, no value ever NaN. Neither property is a
    function of how many tiles the player can see, and the full 9x11 window
    makes every one of these steps render 8,268 floats to check that.
    """
    env = _env(view=(3, 3))
    env.reset()
    width = observation.observation_size((3, 3))
    for index in range(24):
        transition = env.step(_actions(env, 4, index))
        assert transition.observation.shape == (4, width)
        assert bool(torch.isfinite(transition.observation).all())
        assert bool(torch.isfinite(transition.reward).all())


def test_a_checkpoint_resumes_the_identical_episode() -> None:
    env = _env(seed=9)
    env.reset()
    for index in range(5):
        env.step(_actions(env, 4, index))
    saved = {
        key: value.clone() if isinstance(value, Tensor) else value
        for key, value in env.state_dict().items()
    }
    expected = [
        float(env.step(_actions(env, 4, 100 + i)).reward.sum()) for i in range(4)
    ]

    resumed = _env(seed=9)
    resumed.load_state_dict(saved)
    actual = [
        float(resumed.step(_actions(resumed, 4, 100 + i)).reward.sum())
        for i in range(4)
    ]

    assert actual == expected


def test_a_checkpoint_taken_before_reset_restores_cleanly() -> None:
    env = _env()
    saved = env.state_dict()
    restored = _env()
    restored.load_state_dict(saved)
    assert restored.reset().shape == (4, observation.observation_size())


def _worlds(env: CraftaxEnv) -> int:
    """Count how many distinct worlds the batch currently holds."""
    return len({tuple(env.state.map[i].flatten()[:32].tolist()) for i in range(4)})


def test_optimistic_reset_shares_one_world_across_several_workers() -> None:
    """The throughput treatment: generate few worlds, deal them to many.

    Generating a world is the most expensive thing this environment does and
    a step that ends no episode throws every generated world away. The ratio
    is how many workers one fresh world serves.
    """
    env = _env(reset_ratio=4)
    env.reset()
    env.state.player_health[:] = 0.0
    env.step(_actions(env, 4))
    assert _worlds(env) == 1


def test_a_ratio_of_one_gives_every_worker_its_own_world() -> None:
    # The correlation the ratio buys is opt-out, not mandatory.
    env = _env(reset_ratio=1)
    env.reset()
    env.state.player_health[:] = 0.0
    env.step(_actions(env, 4))
    assert _worlds(env) == 4


def test_optimistic_reset_still_restarts_every_finished_worker() -> None:
    # Sharing worlds must not mean skipping a restart: the point is cheapness,
    # not fewer resets.
    env = _env(reset_ratio=4)
    env.reset()
    env.state.timestep[:] = 40
    env.state.player_health[:] = 0.0
    env.step(_actions(env, 4))
    assert env.state.timestep.tolist() == [0, 0, 0, 0]


def test_optimistic_reset_leaves_living_workers_alone() -> None:
    env = _env(reset_ratio=4)
    env.reset()
    survivor = env.state.map[0].clone()
    env.state.player_health[1] = 0.0
    transition = env.step(_actions(env, 4))
    assert transition.done.tolist() == [False, True, False, False]
    assert torch.equal(env.state.map[0], survivor)


def test_only_as_many_worlds_are_generated_as_are_needed() -> None:
    """One finished worker costs one world, not a whole pool.

    Generation scales with batch size -- 25 ms for one world, 182 ms for
    sixty-four -- so paying the pool's full price on a step that ended a
    single episode is the waste this avoids. The reference must pick a static
    shape and compile it; an eager port can just count.
    """
    env = _env(num_envs=4, reset_ratio=2)
    env.reset()
    generated: list[int] = []
    original = world_gen.generate_world

    def spy(
        *,
        num_envs: int,
        generator: torch.Generator | None = None,
        device: torch.device,
    ) -> EnvState:
        generated.append(num_envs)
        return original(num_envs=num_envs, generator=generator, device=device)

    env.state.player_health[:] = 9.0
    env.state.player_health[0] = 0.0
    with mock.patch.object(world_gen, "generate_world", spy):
        env.step(_actions(env, 4))
    assert generated == [1]


def test_the_pool_caps_how_many_worlds_one_step_generates() -> None:
    # The ratio is a ceiling: sixteen workers finishing together must not
    # generate sixteen worlds when the ratio allows two.
    env = _env(num_envs=4, reset_ratio=2)
    env.reset()
    generated: list[int] = []
    original = world_gen.generate_world

    def spy(
        *,
        num_envs: int,
        generator: torch.Generator | None = None,
        device: torch.device,
    ) -> EnvState:
        generated.append(num_envs)
        return original(num_envs=num_envs, generator=generator, device=device)

    env.state.player_health[:] = 0.0
    with mock.patch.object(world_gen, "generate_world", spy):
        env.step(_actions(env, 4))
    assert generated == [2]


def test_a_degenerate_reset_ratio_is_refused() -> None:
    config = CraftaxEnv.Config()
    config.optimistic_reset_ratio = 0
    with pytest.raises(ValueError, match="positive"):
        config.make()


def test_an_empty_batch_is_refused() -> None:
    config = CraftaxEnv.Config()
    config.num_envs = 0
    with pytest.raises(ValueError, match="positive"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
