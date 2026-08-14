"""Tests for the assembled game step."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax.game import constants, step, world_gen
from priml.baselines.craftax.game.constants import (
    Achievement,
    Action,
    BlockType,
    ItemType,
)
from priml.baselines.craftax.game.state import EnvState, empty_state


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    state.player_direction[:] = int(Action.RIGHT)
    state.map[:] = int(BlockType.GRASS)
    state.player_health[:] = 9.0
    for meter in ("player_food", "player_drink", "player_energy", "player_mana"):
        getattr(state, meter)[:] = 9
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    state.potion_mapping[:] = torch.arange(6, dtype=torch.int32)
    return state


def _act(action: Action, num_envs: int = 2) -> torch.Tensor:
    return torch.full((num_envs,), int(action), dtype=torch.int32)


def _seed(value: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(value)


def test_a_step_advances_time_and_daylight() -> None:
    state, _ = step.step(_state(), _act(Action.NOOP), generator=_seed())
    assert state.timestep.tolist() == [1, 1]
    assert 0.0 <= float(state.light_level[0]) <= 1.0


def test_an_achievement_pays_its_reward_once() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.TREE)
    state, first = step.step(state, _act(Action.DO), generator=_seed())
    assert state.achievements[:, int(Achievement.COLLECT_WOOD)].tolist() == [True, True]
    assert float(first[0]) >= 1.0

    state.map[:, 0, 10, 11] = int(BlockType.TREE)
    _, again = step.step(state, _act(Action.DO), generator=_seed())
    # The second tree is wood, but the achievement has already been paid.
    assert float(again[0]) < float(first[0])


def test_losing_health_costs_a_little_reward() -> None:
    # A starving player is one tick from the health threshold, so this step
    # takes the point and the reward reflects it.
    state = _state()
    state.player_health[:] = 5.0
    state.player_food[:] = 0
    state.player_drink[:] = 0
    state.player_recover[:] = -15.0
    _, reward = step.step(state, _act(Action.NOOP), generator=_seed())
    assert float(reward[0]) == pytest.approx(-0.1)


def test_a_sleeping_player_cannot_act() -> None:
    # Sleeping is a commitment: the player must be woken before acting again.
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.TREE)
    state.is_sleeping[:] = True
    state.player_energy[:] = 3
    state, _ = step.step(state, _act(Action.DO), generator=_seed())
    assert state.inventory.wood.tolist() == [0, 0]
    assert state.map[0, 0, 10, 11].item() == int(BlockType.TREE)


def test_an_episode_ends_when_the_player_dies() -> None:
    state = _state()
    state.player_health[:] = 0.0
    assert step.is_done(state).tolist() == [True, True]


def test_an_episode_ends_when_the_boss_falls() -> None:
    state = _state()
    state.boss_progress[:] = constants.NUM_LEVELS - 1
    assert step.is_done(state).tolist() == [True, True]


def test_an_episode_ends_at_the_step_limit() -> None:
    state = _state()
    state.timestep[:] = constants.MAX_TIMESTEPS
    assert step.is_done(state).tolist() == [True, True]


def test_a_healthy_early_episode_is_not_done() -> None:
    assert step.is_done(_state()).tolist() == [False, False]


def test_descending_needs_a_cleared_floor() -> None:
    blocked = _state()
    blocked.item_map[:, 0, 10, 10] = int(ItemType.LADDER_DOWN)
    blocked.up_ladders[:, 1] = torch.tensor([5, 5], dtype=torch.int32)
    held, _ = step.step(blocked, _act(Action.DESCEND), generator=_seed())
    assert held.player_level.tolist() == [0, 0]

    cleared = _state()
    cleared.item_map[:, 0, 10, 10] = int(ItemType.LADDER_DOWN)
    cleared.up_ladders[:, 1] = torch.tensor([5, 5], dtype=torch.int32)
    cleared.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    descended, _ = step.step(cleared, _act(Action.DESCEND), generator=_seed())
    assert descended.player_level.tolist() == [1, 1]
    assert descended.player_position[0].tolist() == [5, 5]


def test_arriving_on_a_new_floor_pays_experience_once() -> None:
    state = _state()
    state.item_map[:, 0, 10, 10] = int(ItemType.LADDER_DOWN)
    state.up_ladders[:, 1] = torch.tensor([5, 5], dtype=torch.int32)
    state.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    state, reward = step.step(state, _act(Action.DESCEND), generator=_seed())
    assert state.player_xp.tolist() == [1, 1]
    assert state.achievements[:, int(Achievement.ENTER_DUNGEON)].tolist() == [
        True,
        True,
    ]
    assert float(reward[0]) >= 3.0


def test_ascending_returns_to_the_floor_above() -> None:
    state = _state()
    state.player_level[:] = 1
    state.item_map[:, 1, 10, 10] = int(ItemType.LADDER_UP)
    state.down_ladders[:, 0] = torch.tensor([7, 7], dtype=torch.int32)
    state, _ = step.step(state, _act(Action.ASCEND), generator=_seed())
    assert state.player_level.tolist() == [0, 0]
    assert state.player_position[0].tolist() == [7, 7]


def test_holding_a_tool_unlocks_its_achievement_however_it_arrived() -> None:
    # A pickaxe can be crafted or looted; checking the inventory covers both.
    state = _state()
    state.inventory.pickaxe[:] = 4
    state, _ = step.step(state, _act(Action.NOOP), generator=_seed())
    for achievement in (
        Achievement.MAKE_WOOD_PICKAXE,
        Achievement.MAKE_STONE_PICKAXE,
        Achievement.MAKE_IRON_PICKAXE,
        Achievement.MAKE_DIAMOND_PICKAXE,
    ):
        assert state.achievements[:, int(achievement)].tolist() == [True, True]


def test_the_boss_countdown_runs_only_on_the_boss_floor() -> None:
    elsewhere = _state()
    elsewhere.boss_timesteps_to_spawn_this_round[:] = 5
    held, _ = step.step(elsewhere, _act(Action.NOOP), generator=_seed())
    assert held.boss_timesteps_to_spawn_this_round.tolist() == [5, 5]

    fighting = _state()
    fighting.player_level[:] = constants.NUM_LEVELS - 1
    fighting.boss_timesteps_to_spawn_this_round[:] = 5
    counted, _ = step.step(fighting, _act(Action.NOOP), generator=_seed())
    assert counted.boss_timesteps_to_spawn_this_round.tolist() == [4, 4]


def test_a_generated_world_survives_a_long_random_rollout() -> None:
    # The whole game, driven by every action, must not raise or leave the map.
    generator = _seed(3)
    state = world_gen.generate_world(
        num_envs=4,
        generator=generator,
        device=torch.device("cpu"),
    )
    total = torch.zeros(4)
    for _ in range(120):
        action = torch.randint(
            0,
            len(constants.Action),
            (4,),
            generator=generator,
        )
        state, reward = step.step(state, action, generator=generator)
        total += reward

    assert state.timestep.tolist() == [120] * 4
    assert int(state.player_position.min()) >= 0
    assert int(state.player_position[:, 0].max()) < constants.MAP_SIZE[0]
    assert int(state.player_level.min()) >= 0
    assert int(state.player_level.max()) < constants.NUM_LEVELS
    assert torch.isfinite(total).all()
    assert float(state.player_health.min()) >= 0.0


def test_each_environment_follows_its_own_action() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.TREE)
    action = torch.tensor([int(Action.DO), int(Action.NOOP)], dtype=torch.int32)
    state, _ = step.step(state, action, generator=_seed())
    assert state.inventory.wood.tolist() == [1, 0]


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
