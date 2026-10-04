"""Tests for the batched, auto-resetting environment."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, cast
from unittest import mock

import copy

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.conftest import generated_world
from priml.baselines.craftax.env import CraftaxEnv, _achievement_info, _Stepper
from priml.baselines.craftax.game import constants, observation, world_gen
from priml.baselines.craftax.game.state import EnvState, empty_state
from priml.baselines.craftax.restart import (
    RestartFromReserve,
    RestartOnDemand,
)
from priml.data.environment import BatchedEnvironmentProtocol


if TYPE_CHECKING:
    from collections.abc import Callable

    from configgle import Makeable

    from priml.baselines.craftax.restart import RestartPolicy


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
    # On demand, because these tests count the worlds each step generates.
    config.restart = RestartOnDemand.Config()
    return config.make()


def test_a_restored_state_replays_the_same_transition() -> None:
    config = CraftaxEnv.Config()
    config.num_envs = 2
    config.device = "cpu"
    config.view = (3, 5)
    config.optimistic_reset_ratio = 1
    env = config.make()
    observation = env.reset()
    assert observation.shape == (2, env.observation_size)
    state = copy.deepcopy(env.state_dict())
    assert set(state) == {"generator", "num_envs", "state", "pool", "restart"}
    assert len(state["state"]) > 0
    assert "pool" in state
    assert "restart" in state
    assert len(state["pool"]) > 0
    assert state["restart"] == {"cursor": None}
    actions = torch.tensor([3, 5])
    transition = env.step(actions)
    assert transition.observation.shape == observation.shape
    assert transition.reward.shape == (2,)
    env.load_state_dict(state)
    replay = env.step(actions)
    assert torch.equal(replay.observation, transition.observation)
    assert torch.equal(replay.reward, transition.reward)


def _actions(env: CraftaxEnv, count: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, env.num_actions, (count,), generator=generator)


@pytest.mark.compute_large_fixture
def test_it_satisfies_the_environment_protocol() -> None:
    assert isinstance(_env(), BatchedEnvironmentProtocol)


@pytest.mark.compute_large_fixture
def test_it_declares_the_published_geometry() -> None:
    env = _env()
    assert env.num_actions == 43
    assert env.observation_size == 8_268
    assert env.reward_ceiling == 226.0


@pytest.mark.compute_large_fixture
def test_reset_returns_one_observation_per_worker() -> None:
    env = _env()
    rendered = env.reset()
    assert rendered.shape == (4, observation.observation_size())
    assert bool(torch.isfinite(rendered).all())


@pytest.mark.compute_large_fixture
def test_reset_can_change_the_batch_size() -> None:
    env = _env()
    assert env.reset(7).shape == (7, observation.observation_size())


@pytest.mark.compute_large_fixture
def test_stepping_returns_a_full_transition() -> None:
    env = _env()
    env.reset()
    transition = env.step(_actions(env, 4))
    assert transition.observation.shape == (4, observation.observation_size())
    assert transition.reward.shape == (4,)
    assert transition.done.shape == (4,)
    assert transition.done.dtype == torch.bool
    assert len(transition.info) == len(constants.Achievement)


def test_environment_passes_configured_device_to_generator() -> None:
    config = CraftaxEnv.Config()
    config.num_envs = 2
    config.device = "cpu"
    config.seed = 17
    original_generator = torch.Generator
    devices: list[object] = []
    device_names: list[str] = []

    def generator(*, device: torch.device) -> torch.Generator:
        devices.append(device)
        return original_generator(device=device)

    def resolve(device_name: str) -> torch.device:
        device_names.append(device_name)
        return torch.device("cpu")

    with (
        mock.patch.object(torch, "Generator", generator),
        mock.patch("priml.baselines.craftax.env.get_device", resolve),
    ):
        env = CraftaxEnv(config)

    assert device_names == ["cpu"]
    assert devices == [torch.device("cpu")]
    assert env._generator.initial_seed() == 17


def test_reading_the_world_before_reset_is_refused() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"^CraftaxEnv must reset before it can be read$",
    ):
        _ = _env().state


def test_achievement_info_names_and_selects_each_column() -> None:
    values = torch.arange(3 * len(constants.Achievement)).reshape(
        3,
        len(constants.Achievement),
    )
    info = _achievement_info(values)
    assert list(info) == [
        f"Achievements/{achievement.name.lower()}"
        for achievement in constants.Achievement
    ]
    assert all(
        torch.equal(info[f"Achievements/{achievement.name.lower()}"], values[:, index])
        for index, achievement in enumerate(constants.Achievement)
    )


@pytest.mark.compute_large_fixture
def test_the_same_seed_replays_the_same_episode() -> None:
    def rollout() -> list[float]:
        env = _env(seed=5)
        env.reset()
        return [
            float(env.step(_actions(env, 4, index)).reward.sum()) for index in range(12)
        ]

    assert rollout() == rollout()


@pytest.mark.compute_large_fixture
def test_different_seeds_give_different_worlds() -> None:
    assert not torch.equal(_env(seed=1).reset(), _env(seed=2).reset())


@pytest.mark.compute_large_fixture
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


@pytest.mark.compute_large_fixture
def test_a_restarted_worker_gets_a_fresh_world() -> None:
    env = _env()
    env.reset()
    doomed = env.state.map[1].clone()
    env.state.player_health[1] = 0.0
    env.step(_actions(env, 4))
    assert not torch.equal(env.state.map[1], doomed)


@pytest.mark.compute_large_fixture
def test_the_observation_after_a_restart_is_the_new_episode() -> None:
    # The terminal observation is deliberately not visible: the learner sees
    # where the next episode begins.
    env = _env()
    env.reset()
    env.state.player_health[1] = 0.0
    transition = env.step(_actions(env, 4))
    assert torch.equal(transition.observation[1], observation.render(env.state)[1])


@pytest.mark.compute_large_fixture
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


@pytest.mark.compute_large_fixture
def test_an_episode_ends_when_the_step_limit_is_reached() -> None:
    env = _env()
    env.reset()
    env.state.timestep[:] = constants.MAX_TIMESTEPS - 1
    assert env.step(_actions(env, 4)).done.all()


@pytest.mark.compute_large_fixture
def test_a_long_rollout_stays_finite_and_rectangular() -> None:
    """Many steps in sequence keep the shape and stay numerically sane.

    A small view, because what is under test is that the rollout does not
    drift -- the batch never ragged, no value ever NaN. Neither property is a
    function of how many tiles the player can see, and the full 9x11 window
    makes every one of these steps render 8,268 floats to check that.
    """
    env = _env(view=(3, 5))
    env.reset()
    width = observation.observation_size((3, 5))
    for index in range(24):
        transition = env.step(_actions(env, 4, index))
        assert transition.observation.shape == (4, width)
        assert bool(torch.isfinite(transition.observation).all())
        assert bool(torch.isfinite(transition.reward).all())


@pytest.mark.compute_large_fixture
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


def test_transition_preserves_terminal_state_and_achievement_info() -> None:
    env = _env(num_envs=4, view=(3, 5))
    worlds = {count: generated_world(num_envs=count) for count in (2, 4)}

    generated_counts: list[int] = []

    def generate_world(
        *,
        num_envs: int,
        generator: torch.Generator | None = None,
        device: torch.device,
    ) -> EnvState:
        del generator, device
        generated_counts.append(num_envs)
        return worlds[num_envs]

    with mock.patch.object(world_gen, "generate_world", generate_world):
        env.reset()
        env.state.achievements[:, int(constants.Achievement.COLLECT_WOOD)] = True
        env.state.player_health[:2] = 0.0
        transition = env.step(_actions(env, 4))
        assert transition.done.tolist() == [True, True, False, False]
        assert transition.info["Achievements/collect_wood"].tolist() == [
            100.0,
            100.0,
            0.0,
            0.0,
        ]
        assert transition.terminal_state.achievements[
            :,
            int(constants.Achievement.COLLECT_WOOD),
        ].tolist() == [True, True, True, True]
        assert transition.terminal_state.player_health[:2].le(0.0).all()
        assert torch.equal(env.state.map[:2], worlds[2].map)
        assert torch.equal(env.state.map[2:], transition.terminal_state.map[2:])
        cached_procedure = env._live_stepper()._generate_by_count[2]
        env.state.player_health[:2] = 0.0
        repeated = env.step(_actions(env, 4))

    assert generated_counts == [4, 2, 2]
    assert env._live_stepper()._generate_by_count[2] is cached_procedure
    assert repeated.done.tolist() == [True, True, False, False]


def test_a_checkpoint_taken_before_reset_restores_cleanly() -> None:
    env = _env()
    saved = env.state_dict()
    assert set(saved) == {"generator", "num_envs", "state"}
    assert saved["state"] == {}
    restored = _env()
    restored.load_state_dict(saved)
    assert restored.reset().shape == (4, observation.observation_size())


@pytest.mark.parametrize("missing", ["pool", "restart"])
def test_checkpoint_missing_one_restart_field_falls_back_to_spent_pool(
    missing: str,
) -> None:
    env = _env(num_envs=2, view=(3, 5))
    env.reset()
    saved = copy.deepcopy(dict(env.state_dict()))
    del saved[missing]

    restored = _env(num_envs=2, view=(3, 5))
    world = generated_world(num_envs=2)
    with mock.patch.object(world_gen, "generate_world", return_value=world):
        restored.load_state_dict(saved)

    assert restored.state.num_envs == 2
    restored_state = restored.state_dict()
    assert "restart" in restored_state
    assert restored_state["restart"] == {"cursor": None}


def _worlds(env: CraftaxEnv) -> int:
    """Count how many distinct worlds the batch currently holds."""
    return len({tuple(env.state.map[i].flatten()[:32].tolist()) for i in range(4)})


@pytest.mark.compute_large_fixture
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


@pytest.mark.compute_large_fixture
def test_a_ratio_of_one_gives_every_worker_its_own_world() -> None:
    # The correlation the ratio buys is opt-out, not mandatory.
    env = _env(reset_ratio=1)
    env.reset()
    env.state.player_health[:] = 0.0
    env.step(_actions(env, 4))
    assert _worlds(env) == 4


@pytest.mark.compute_large_fixture
def test_optimistic_reset_still_restarts_every_finished_worker() -> None:
    # Sharing worlds must not mean skipping a restart: the point is cheapness,
    # not fewer resets.
    env = _env(reset_ratio=4)
    env.reset()
    env.state.timestep[:] = 40
    env.state.player_health[:] = 0.0
    env.step(_actions(env, 4))
    assert env.state.timestep.tolist() == [0, 0, 0, 0]


@pytest.mark.compute_large_fixture
def test_optimistic_reset_leaves_living_workers_alone() -> None:
    env = _env(reset_ratio=4)
    env.reset()
    survivor = env.state.map[0].clone()
    env.state.player_health[1] = 0.0
    transition = env.step(_actions(env, 4))
    assert transition.done.tolist() == [False, True, False, False]
    assert torch.equal(env.state.map[0], survivor)


@pytest.mark.compute_large_fixture
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


@pytest.mark.compute_large_fixture
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


@pytest.mark.compute_large_fixture
def test_a_degenerate_reset_ratio_is_refused() -> None:
    config = CraftaxEnv.Config()
    config.optimistic_reset_ratio = 0
    with pytest.raises(ValueError, match="positive"):
        config.make()


def test_reset_uses_the_current_batch_and_environment_generator() -> None:
    env = _env(num_envs=3, view=(3, 5))
    worlds = {num_envs: generated_world(num_envs=num_envs) for num_envs in (3, 5)}
    calls: list[tuple[int, torch.Generator | None, torch.device]] = []

    def generate_world(
        *,
        num_envs: int,
        generator: torch.Generator | None = None,
        device: torch.device,
    ) -> EnvState:
        calls.append((num_envs, generator, device))
        return worlds[num_envs]

    with mock.patch.object(world_gen, "generate_world", generate_world):
        initial = env.reset()
        resized = env.reset(5)

    assert initial.shape == (3, observation.observation_size((3, 5)))
    assert resized.shape == (5, observation.observation_size((3, 5)))
    assert calls == [
        (3, env._generator, torch.device("cpu")),
        (5, env._generator, torch.device("cpu")),
    ]
    assert env.state.num_envs == 5


def test_an_empty_batch_is_refused() -> None:
    config = CraftaxEnv.Config()
    config.num_envs = 0
    with pytest.raises(ValueError, match="positive"):
        config.make()


def test_an_empty_view_is_refused() -> None:
    config = CraftaxEnv.Config()
    config.view = (0, 11)
    with pytest.raises(
        ValueError,
        match=r"^view must be positive in both dimensions$",
    ):
        config.make()


def test_single_environment_batch_is_valid_before_reset() -> None:
    config = CraftaxEnv.Config()
    config.num_envs = 1
    config.device = "cpu"
    assert config.make()._num_envs == 1


def test_single_tile_view_dimension_is_valid_before_reset() -> None:
    config = CraftaxEnv.Config()
    config.view = (1, 2)
    config.device = "cpu"
    assert config.make().observation_size == observation.observation_size((1, 2))


def test_every_nonpositive_batch_dimension_is_refused() -> None:
    for num_envs in (-1, 0):
        config = CraftaxEnv.Config()
        config.num_envs = num_envs
        with pytest.raises(ValueError, match=r"^num_envs must be positive$"):
            config.make()


def test_every_nonpositive_reset_ratio_is_refused() -> None:
    for ratio in (-1, 0):
        config = CraftaxEnv.Config()
        config.optimistic_reset_ratio = ratio
        with pytest.raises(
            ValueError,
            match=r"^optimistic_reset_ratio must be positive$",
        ):
            config.make()


@pytest.mark.compute_large_fixture
def test_a_reserve_generates_a_whole_pool_then_deals_from_it() -> None:
    env = _reserve_env()
    env.reset()
    spy = _GenerationSpy()
    env.state.player_health[0] = 0.0
    with mock.patch.object(world_gen, "generate_world", spy):
        env.step(_actions(env, 4))
    first = env.state.map[0].clone()
    env.state.player_health[1] = 0.0
    with mock.patch.object(world_gen, "generate_world", spy):
        env.step(_actions(env, 4, 1))
    assert spy.generated == [4]
    # The second worker took the NEXT world of the pool, not the first again.
    assert not torch.equal(env.state.map[1], first)
    assert int(env.state.timestep[1]) == 0


@pytest.mark.compute_large_fixture
def test_a_reserve_refills_when_too_few_worlds_remain() -> None:
    env = _reserve_env()
    env.reset()
    spy = _GenerationSpy()
    env.state.player_health[:3] = 0.0
    with mock.patch.object(world_gen, "generate_world", spy):
        env.step(_actions(env, 4))
    env.state.player_health[1:3] = 0.0
    with mock.patch.object(world_gen, "generate_world", spy):
        env.step(_actions(env, 4, 1))
    assert spy.generated == [4, 4]


@pytest.mark.compute_large_fixture
def test_a_checkpoint_resumes_the_identical_episode_from_a_reserve() -> None:
    # The pool's unused worlds and where dealing stopped are part of the world:
    # a resumed run must hand the next finished worker the same fresh world.
    env = _reserve_env(seed=9)
    env.reset()
    env.state.player_health[0] = 0.0
    env.step(_actions(env, 4))
    saved = copy.deepcopy(env.state_dict())
    env.state.player_health[2] = 0.0
    expected = env.step(_actions(env, 4, 1)).observation

    resumed = _reserve_env(seed=9)
    resumed.load_state_dict(saved)
    resumed.state.player_health[2] = 0.0
    assert torch.equal(resumed.step(_actions(resumed, 4, 1)).observation, expected)


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("restart", [RestartOnDemand.Config, RestartFromReserve.Config])
@pytest.mark.compute_large_fixture
def test_cuda_graphs_step_bit_for_bit_like_eager(
    restart: Callable[[], Makeable[RestartPolicy]],
) -> None:
    envs: list[CraftaxEnv] = []
    for graphs in (False, True):
        config = CraftaxEnv.Config()
        config.num_envs = 16
        config.device = "cuda"
        config.optimistic_reset_ratio = 4
        config.restart = restart()
        config.cuda_graphs = graphs
        envs.append(config.make())
    eager, graphed = envs
    assert torch.equal(eager.reset(), graphed.reset())
    generator = torch.Generator(device="cuda").manual_seed(1)
    for index in range(24):
        actions = torch.randint(0, 43, (16,), generator=generator, device="cuda")
        for env in envs:
            env.state.player_health[index % 16] = 0.0
            env.state.player_health[(5 * index) % 16] = 0.0
        a, b = eager.step(actions), graphed.step(actions)
        assert torch.equal(a.observation, b.observation), index
        assert torch.equal(a.reward, b.reward), index
        assert torch.equal(a.done, b.done), index
    for name, value in eager.state.state_dict().items():
        assert torch.equal(value, graphed.state.state_dict()[name]), name


@pytest.mark.parametrize("restart", [RestartOnDemand.Config, RestartFromReserve.Config])
def test_a_step_writes_only_into_the_memory_a_graph_replays(
    restart: Callable[[], Makeable[RestartPolicy]],
) -> None:
    """Capture's first rule, checked without a GPU: no step rebinds a buffer.

    A replayed graph addresses the memory it was captured against, so a buffer
    the step REBINDS -- the world, the pool, a reward or deal scalar -- goes
    unseen by every replay. ``test_cuda_graphs_step_bit_for_bit_like_eager``
    sees that only on a GPU; here every tensor the step machinery holds must
    keep its address across steps that restart, generate, and deal.
    """
    config = CraftaxEnv.Config()
    config.view = (3, 5)
    config.num_envs = 4
    config.device = "cpu"
    config.optimistic_reset_ratio = 2
    config.restart = restart()
    env = config.make()
    # Generated before the patch, which would otherwise answer the cache's own call.
    generated_world()
    with mock.patch.object(world_gen, "generate_world", _cached_world):
        env.reset()
        stepper = env._live_stepper()
        addresses = _tensor_addresses(stepper)
        for index in range(2):
            env.state.player_health[index] = 0.0
            assert bool(env.step(_actions(env, 4, index)).done[index])
    assert _tensor_addresses(stepper) == addresses


def test_stepper_allocates_exact_cpu_buffers() -> None:
    world = generated_world(num_envs=3)
    original_zeros, original_ones = torch.zeros, torch.ones
    allocations: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def zeros(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        recorded: dict[str, object] = {}
        if dtype is not None:
            recorded["dtype"] = dtype
        if device is not None:
            recorded["device"] = device
        allocations.append(("zeros", (size,), recorded))
        return original_zeros(size, dtype=dtype, device=device)

    def ones(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        recorded: dict[str, object] = {}
        if dtype is not None:
            recorded["dtype"] = dtype
        if device is not None:
            recorded["device"] = device
        allocations.append(("ones", (size,), recorded))
        return original_ones(size, dtype=dtype, device=device)

    with (
        mock.patch.object(torch, "zeros", zeros),
        mock.patch.object(torch, "ones", ones),
    ):
        stepper = _Stepper(
            world,
            generator=torch.Generator().manual_seed(2),
            pool_size=2,
            restart=RestartOnDemand.Config().make(),
            view=(3, 5),
            cuda_graphs=False,
        )

    action_allocation = max(
        index
        for index, allocation in enumerate(allocations)
        if allocation == ("zeros", (3,), {"dtype": torch.int64, "device": world.device})
    )
    assert allocations[action_allocation : action_allocation + 7] == [
        ("zeros", (3,), {"dtype": torch.int64, "device": world.device}),
        ("zeros", (3,), {"device": world.device}),
        ("zeros", (3,), {"dtype": torch.bool, "device": world.device}),
        (
            "zeros",
            ((3, len(constants.Achievement)),),
            {"device": world.device},
        ),
        ("zeros", ((),), {"dtype": torch.int64, "device": world.device}),
        ("zeros", ((),), {"dtype": torch.int64, "device": world.device}),
        ("ones", ((),), {"dtype": torch.int64, "device": world.device}),
    ]
    assert stepper._generator.initial_seed() == 2
    assert stepper._cuda_graphs is False
    assert isinstance(stepper._advance, MethodType)
    assert stepper._advance.__self__ is stepper

    def procedure() -> None:
        pass

    generator = torch.Generator().manual_seed(3)
    stepper._cuda_graphs = True
    with mock.patch(
        "priml.baselines.craftax.env.CudaGraphed",
        return_value=procedure,
    ) as cuda_graphed:
        assert stepper._procedure(procedure, (generator,)) is procedure
    cuda_graphed.assert_called_once_with(procedure, generators=(generator,))
    stepper._cuda_graphs = False
    assert stepper._procedure(procedure, ()) is procedure


def test_reward_ceiling_is_the_published_achievement_total() -> None:
    env = _env(num_envs=1, view=(3, 5))

    assert env.reward_ceiling == 226.0


@pytest.mark.parametrize(("enabled", "expected"), [(True, True), (False, False)])
def test_cuda_graphs_require_the_exact_cuda_device_type(
    enabled: bool,
    expected: bool,
) -> None:
    config = CraftaxEnv.Config()
    config.num_envs = 1
    config.view = (3, 5)
    config.device = "cuda"
    config.cuda_graphs = enabled
    generator = torch.Generator()

    with (
        mock.patch(
            "priml.baselines.craftax.env.get_device",
            return_value=torch.device("cuda"),
        ),
        mock.patch.object(torch, "Generator", return_value=generator) as factory,
    ):
        env = CraftaxEnv(config)

    factory.assert_called_once_with(device=torch.device("cuda"))
    assert env._device == torch.device("cuda")
    assert env._cuda_graphs is expected


def test_loading_an_empty_checkpoint_clears_an_initialized_stepper() -> None:
    env = _env(num_envs=1, view=(3, 5))
    checkpoint = env.state_dict()
    env.reset()

    env.load_state_dict(checkpoint)

    with pytest.raises(
        RuntimeError,
        match=r"^CraftaxEnv must reset before it can be read$",
    ):
        _ = env.state


def test_reserve_checkpoint_restores_existing_stepper_and_pool_exactly() -> None:
    source = _reserve_env(seed=9)
    source.reset()
    source.state.player_health[0] = 0.0
    source.step(_actions(source, 4, seed=1))
    checkpoint = copy.deepcopy(source.state_dict())
    assert "restart" in checkpoint
    assert "pool" in checkpoint
    assert checkpoint["restart"] == {"cursor": 1}

    target = _reserve_env(seed=10)
    target.reset()
    with mock.patch.object(
        world_gen,
        "generate_world",
        side_effect=AssertionError("same-sized restore must keep live buffers"),
    ):
        target.load_state_dict(checkpoint)

    restored = target.state_dict()
    assert "restart" in restored
    assert "pool" in restored
    assert restored["num_envs"] == checkpoint["num_envs"]
    assert torch.equal(restored["generator"], checkpoint["generator"])
    assert restored["restart"] == checkpoint["restart"]
    assert restored["state"].keys() == checkpoint["state"].keys()
    assert all(
        torch.equal(restored["state"][key], checkpoint["state"][key])
        for key in checkpoint["state"]
    )
    assert restored["pool"].keys() == checkpoint["pool"].keys()
    assert all(
        torch.equal(restored["pool"][key], checkpoint["pool"][key])
        for key in checkpoint["pool"]
    )


@pytest.mark.parametrize("missing", ["pool", "restart"])
def test_incomplete_reserve_checkpoint_resets_its_cursor(missing: str) -> None:
    source = _reserve_env(seed=9)
    source.reset()
    source.state.player_health[0] = 0.0
    source.step(_actions(source, 4, seed=1))
    checkpoint: dict[str, object] = copy.deepcopy(dict(source.state_dict()))
    del checkpoint[missing]

    restored = _reserve_env(seed=10)
    restored.load_state_dict(checkpoint)
    state = restored.state_dict()

    assert "restart" in state
    assert state["restart"] == {"cursor": None}


def test_reset_preserves_generator_and_uses_one_world_pool_minimum() -> None:
    env = _env(num_envs=1, reset_ratio=16, view=(3, 5))

    env.reset()

    stepper = env._live_stepper()
    assert stepper._generator is env._generator
    assert stepper.pool.num_envs == 1
    assert stepper._cuda_graphs is False


def test_stepper_allocations_and_capture_keep_their_devices_and_generators() -> None:
    world = generated_world(num_envs=2)
    generator = torch.Generator().manual_seed(17)
    allocations: list[tuple[int, torch.device | None]] = []
    captured: list[tuple[torch.Generator, ...] | None] = []

    def allocate_state(
        *,
        num_envs: int,
        device: torch.device | None,
    ) -> EnvState:
        allocations.append((num_envs, device))
        return empty_state(
            num_envs=num_envs,
            device=device if device is not None else world.device,
        )

    def capture(
        procedure: Callable[[], None],
        *,
        generators: tuple[torch.Generator, ...] | None,
    ) -> Callable[[], None]:
        captured.append(generators)
        return procedure

    with (
        mock.patch("priml.baselines.craftax.env.empty_state", allocate_state),
        mock.patch("priml.baselines.craftax.env.CudaGraphed", capture),
    ):
        _Stepper(
            world,
            generator=generator,
            pool_size=2,
            restart=RestartOnDemand.Config().make(),
            view=(3, 5),
            cuda_graphs=True,
        )

    assert allocations == [(2, world.device), (2, world.device)]
    assert captured == [(generator,), ()]


def test_advance_eager_passes_the_environment_generator() -> None:
    world = generated_world(num_envs=2)
    generator = torch.Generator().manual_seed(19)
    stepper = _Stepper(
        world,
        generator=generator,
        pool_size=1,
        restart=RestartOnDemand.Config().make(),
        view=(3, 5),
        cuda_graphs=False,
    )
    used_generators: list[torch.Generator | None] = []

    def advance(
        state: EnvState,
        actions: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[EnvState, Tensor]:
        del actions
        used_generators.append(generator)
        return state, torch.zeros(world.num_envs)

    with (
        mock.patch("priml.baselines.craftax.game.step.step", advance),
        mock.patch(
            "priml.baselines.craftax.game.step.is_done",
            return_value=torch.zeros(world.num_envs, dtype=torch.bool),
        ),
    ):
        stepper._advance_eager()

    assert used_generators == [generator]


def test_commit_eager_deals_wrapped_pool_rows_in_done_order() -> None:
    world = generated_world(num_envs=4)
    reached = generated_world(num_envs=4, seed=1)
    pool = generated_world(num_envs=3, seed=2)
    stepper = _Stepper(
        world,
        generator=torch.Generator().manual_seed(23),
        pool_size=3,
        restart=RestartOnDemand.Config().make(),
        view=(3, 5),
        cuda_graphs=False,
    )
    stepper.state.timestep.copy_(torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    reached.timestep.copy_(torch.tensor([20, 21, 22, 23], dtype=torch.int32))
    pool.timestep.copy_(torch.tensor([100, 101, 102], dtype=torch.int32))
    stepper.reached.copy_(reached)
    stepper.pool.copy_(pool)
    stepper._done.copy_(torch.tensor([True, False, True, True]))
    stepper._deal_offset.fill_(2)
    stepper._deal_modulus.fill_(3)

    stepper._commit_eager()

    assert stepper.state.timestep.tolist() == [102, 21, 100, 101]


def test_boolean_prefix_sum_matches_the_explicit_integer_cast() -> None:
    done = torch.tensor([True, False, True, True])

    explicit = done.to(torch.int64).cumsum(0)
    implicit = done.cumsum(0)

    assert explicit.dtype is torch.int64
    assert implicit.dtype is torch.int64
    assert torch.equal(implicit, explicit)


def test_generate_caches_graph_and_preserves_generator_and_device() -> None:
    world = generated_world(num_envs=2)
    generator = torch.Generator().manual_seed(29)
    stepper = _Stepper(
        world,
        generator=generator,
        pool_size=2,
        restart=RestartOnDemand.Config().make(),
        view=(3, 5),
        cuda_graphs=False,
    )
    stepper._cuda_graphs = True
    fresh = generated_world(num_envs=1, seed=31)
    captured: list[tuple[torch.Generator, ...] | None] = []
    generated: list[tuple[int, torch.Generator | None, torch.device | None]] = []

    def capture(
        procedure: Callable[[], None],
        *,
        generators: tuple[torch.Generator, ...] | None,
    ) -> Callable[[], None]:
        captured.append(generators)
        return procedure

    def generate_world(
        *,
        num_envs: int,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> EnvState:
        generated.append((num_envs, generator, device))
        return fresh

    with (
        mock.patch("priml.baselines.craftax.env.CudaGraphed", capture),
        mock.patch.object(world_gen, "generate_world", generate_world),
    ):
        stepper._generate(1)
        cached = stepper._generate_by_count[1]
        stepper._generate(1)

    assert captured == [(generator,)]
    assert stepper._generate_by_count[1] is cached
    assert generated == [(1, generator, world.device), (1, generator, world.device)]


def _cached_world(
    *,
    num_envs: int,
    generator: torch.Generator | None = None,
    device: torch.device,
) -> EnvState:
    """Stand in for world generation with copies of one world generated once."""
    del generator, device
    return generated_world().take(torch.zeros(num_envs, dtype=torch.int64))


def _tensor_addresses(holder: object) -> dict[str, int]:
    """Map every tensor ``holder`` keeps, directly or in a world, to its address."""
    addresses: dict[str, int] = {}
    for name, value in cast("dict[str, object]", vars(holder)).items():
        if isinstance(value, Tensor):
            addresses[name] = value.data_ptr()
        elif isinstance(value, EnvState):
            for field, tensor in value.state_dict().items():
                addresses[f"{name}.{field}"] = tensor.data_ptr()
    return addresses


def _reserve_env(*, seed: int = 0) -> CraftaxEnv:
    """Four workers sharing a pool of four worlds dealt from a reserve."""
    config = CraftaxEnv.Config()
    config.view = (3, 5)
    config.num_envs = 4
    config.device = "cpu"
    config.seed = seed
    config.optimistic_reset_ratio = 1
    config.restart = RestartFromReserve.Config()
    return config.make()


class _GenerationSpy:
    """Records the batch size of every world generation, then performs it."""

    def __init__(self) -> None:
        self.generated: list[int] = []
        self._generate = world_gen.generate_world

    def __call__(
        self,
        *,
        num_envs: int,
        generator: torch.Generator | None = None,
        device: torch.device,
    ) -> EnvState:
        self.generated.append(num_envs)
        return self._generate(num_envs=num_envs, generator=generator, device=device)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
