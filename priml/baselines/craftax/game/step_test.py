"""Tests for the assembled game step."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import Mock, patch

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import (
    abilities,
    constants,
    crafting,
    interact,
    mechanics,
    mobs,
    step,
    survival,
)
from priml.baselines.craftax.game.constants import (
    Achievement,
    Action,
    BlockType,
    ItemType,
)
from priml.baselines.craftax.game.indexing import batch_rows
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


def _act(action: Action, num_envs: int = 2) -> Tensor:
    return torch.full((num_envs,), int(action), dtype=torch.int32)


def _seed(value: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(value)


def test_a_step_advances_time_and_daylight() -> None:
    state, _ = step.step(_state(), _act(Action.NOOP), generator=_seed())
    assert state.timestep.tolist() == [1, 1]
    assert 0.0 <= float(state.light_level[0]) <= 1.0


def test_step_forwards_effective_actions_and_generator_to_each_stage() -> None:
    action = torch.tensor([int(value) for value in constants.Action], dtype=torch.int32)
    state = _state(num_envs=action.numel())
    state.is_sleeping[0] = True
    effective_action = action.clone()
    effective_action[0] = int(Action.NOOP)
    generator = _seed(13)
    stages = (
        (step, "change_floor"),
        (crafting, "craft"),
        (interact, "interact"),
        (crafting, "place"),
        (abilities, "shoot_arrow"),
        (abilities, "cast_spell"),
        (abilities, "drink_potion"),
        (abilities, "read_book"),
        (abilities, "enchant"),
        (step, "_advance_boss"),
        (abilities, "level_up"),
        (survival, "move_player"),
        (mobs, "update_mobs"),
        (mobs, "spawn_mobs"),
        (abilities, "grow_plants"),
        (survival, "update_intrinsics"),
        (mechanics, "clip_meters"),
        (step, "_unlock_from_inventory"),
        (step, "_reward"),
    )
    calls: dict[str, Mock] = {}
    with ExitStack() as stack:
        for module, name in stages:
            result = torch.zeros(state.num_envs) if name == "_reward" else state
            calls[name] = stack.enter_context(
                patch.object(module, name, return_value=result),
            )
        step.step(state, action, generator=generator)

    for name in (
        "craft",
        "place",
        "shoot_arrow",
        "cast_spell",
        "drink_potion",
        "read_book",
        "enchant",
        "level_up",
        "move_player",
        "update_intrinsics",
    ):
        assert calls[name].call_args.args[0] is state
        forwarded_action = calls[name].call_args.args[1]
        assert isinstance(forwarded_action, Tensor)
        assert torch.equal(forwarded_action, effective_action), name
    for name in ("interact", "read_book", "enchant", "update_mobs", "spawn_mobs"):
        assert calls[name].call_args.kwargs["generator"] is generator, name
    doing = calls["interact"].call_args.kwargs["doing"]
    assert isinstance(doing, Tensor)
    assert torch.equal(doing, effective_action == int(Action.DO))


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


def test_a_sleeping_player_cannot_gather() -> None:
    # Sleeping is a commitment: the player must be woken before acting again.
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.TREE)
    state.is_sleeping[:] = True
    state.player_energy[:] = 3
    state, _ = step.step(state, _act(Action.DO), generator=_seed())
    assert state.inventory.wood.tolist() == [0, 0]
    assert state.map[0, 0, 10, 11].item() == int(BlockType.TREE)


def test_a_sleeping_player_cannot_move() -> None:
    state = _state()
    state.is_sleeping[:] = True
    state.player_energy[:] = 3
    state, _ = step.step(state, _act(Action.RIGHT), generator=_seed())
    assert state.player_position.tolist() == [[10, 10], [10, 10]]


def test_step_crafts_and_places_with_the_selected_action() -> None:
    state = _state()
    state.map[:, 0, 9, 10] = int(BlockType.CRAFTING_TABLE)
    state.inventory.wood[:] = 2
    crafted, _ = step.step(state, _act(Action.MAKE_WOOD_SWORD), generator=_seed())
    assert crafted.inventory.sword.tolist() == [1, 1]
    assert crafted.inventory.wood.tolist() == [1, 1]
    assert crafted.achievements[:, int(Achievement.MAKE_WOOD_SWORD)].tolist() == [
        True,
        True,
    ]

    state = _state()
    state.inventory.stone[:] = 1
    placed, _ = step.step(state, _act(Action.PLACE_STONE), generator=_seed())
    assert placed.map[:, 0, 10, 11].tolist() == [
        int(BlockType.STONE),
        int(BlockType.STONE),
    ]
    assert placed.inventory.stone.tolist() == [0, 0]
    assert placed.achievements[:, int(Achievement.PLACE_STONE)].tolist() == [
        True,
        True,
    ]


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
    state = _state()
    state.player_health[:] = 2
    assert step.is_done(state).tolist() == [False, False]


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


def test_floor_transitions_stop_at_both_world_boundaries() -> None:
    top = _state()
    top.player_level[:] = constants.NUM_LEVELS - 1
    top.item_map[:, -1, 10, 10] = int(ItemType.LADDER_DOWN)
    top.monsters_killed[:, -1] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    held, _ = step.step(top, _act(Action.DESCEND), generator=_seed())
    assert held.player_level.tolist() == [constants.NUM_LEVELS - 1] * 2

    ground = _state()
    ground.item_map[:, 0, 10, 10] = int(ItemType.LADDER_UP)
    held, _ = step.step(ground, _act(Action.ASCEND), generator=_seed())
    assert held.player_level.tolist() == [0, 0]


def test_iron_and_diamond_inventory_thresholds_are_inclusive() -> None:
    state = _state()
    state.inventory.pickaxe[:] = torch.tensor([3, 4])
    state.inventory.sword[:] = torch.tensor([3, 4])

    unlocked = step._unlock_from_inventory(state)

    for iron, diamond in (
        (Achievement.MAKE_IRON_PICKAXE, Achievement.MAKE_DIAMOND_PICKAXE),
        (Achievement.MAKE_IRON_SWORD, Achievement.MAKE_DIAMOND_SWORD),
    ):
        assert unlocked.achievements[:, int(iron)].tolist() == [True, True]
        assert unlocked.achievements[:, int(diamond)].tolist() == [False, True]


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


def test_inventory_achievement_thresholds_and_rewards() -> None:
    state = _state()
    state.inventory.wood[:] = 1
    state.inventory.stone[:] = 1
    state.inventory.coal[:] = 1
    state.inventory.iron[:] = 1
    state.inventory.diamond[:] = 1
    state.inventory.ruby[:] = 1
    state.inventory.sapphire[:] = 1
    state.inventory.sapling[:] = 1
    state.inventory.bow[:] = 1
    state.inventory.arrows[:] = 1
    state.inventory.torches[:] = 1
    state.inventory.pickaxe[:] = 1
    state.inventory.sword[:] = 1
    state.inventory.pickaxe[1] = 2
    state.inventory.sword[1] = 2
    state, reward = step.step(state, _act(Action.NOOP), generator=_seed())
    expected = (
        Achievement.COLLECT_WOOD,
        Achievement.COLLECT_STONE,
        Achievement.COLLECT_COAL,
        Achievement.COLLECT_IRON,
        Achievement.COLLECT_DIAMOND,
        Achievement.COLLECT_RUBY,
        Achievement.COLLECT_SAPPHIRE,
        Achievement.COLLECT_SAPLING,
        Achievement.FIND_BOW,
        Achievement.MAKE_ARROW,
        Achievement.MAKE_TORCH,
        Achievement.MAKE_WOOD_PICKAXE,
        Achievement.MAKE_WOOD_SWORD,
    )
    for achievement in expected:
        assert state.achievements[:, int(achievement)].tolist() == [True, True]
    assert state.achievements[:, int(Achievement.MAKE_STONE_PICKAXE)].tolist() == [
        False,
        True,
    ]
    assert state.achievements[:, int(Achievement.MAKE_STONE_SWORD)].tolist() == [
        False,
        True,
    ]
    assert reward[0].item() == 19.0
    assert reward[1].item() == 21.0


def test_step_helpers_use_the_state_device_for_tensor_creation() -> None:
    state = _state()
    with patch.object(torch, "full", wraps=torch.full) as full:
        step._advance_boss(state)
    assert full.call_args is not None
    assert full.call_args.kwargs == {"device": state.device}

    with (
        patch.object(torch, "full", wraps=torch.full) as full,
        patch.object(mechanics, "unlock_achievement", return_value=state),
    ):
        step._unlock_from_inventory(state)
    assert full.call_count == 19
    assert all(call.kwargs == {"device": state.device} for call in full.call_args_list)


def test_reward_lookup_uses_the_state_device_and_exact_achievement_delta() -> None:
    state = _state()
    state.achievements[:, int(Achievement.COLLECT_WOOD)] = True
    before = torch.zeros_like(state.achievements)
    health_before = state.player_health.clone()

    with patch.object(constants, "on_device", wraps=constants.on_device) as on_device:
        reward = step._reward(
            state,
            unlocked_before=before,
            health_before=health_before,
        )

    assert on_device.call_args is not None
    assert on_device.call_args.args[0] is constants.ACHIEVEMENT_REWARD
    assert on_device.call_args.args[1] == state.device
    expected = float(constants.ACHIEVEMENT_REWARD[int(Achievement.COLLECT_WOOD)])
    assert reward.tolist() == [expected, expected]


def test_descending_from_the_penultimate_floor_uses_last_floor_ladder() -> None:
    state = _state()
    previous = constants.NUM_LEVELS - 2
    final = constants.NUM_LEVELS - 1
    state.player_level[:] = previous
    state.item_map[:, previous, 10, 10] = int(ItemType.LADDER_DOWN)
    state.monsters_killed[:, previous] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    state.up_ladders[:, final] = torch.tensor([[5, 7], [8, 9]], dtype=torch.int32)

    with (
        patch.object(step, "batch_rows", wraps=batch_rows) as rows,
        patch.object(constants, "on_device", wraps=constants.on_device) as on_device,
    ):
        descended = step.change_floor(state, _act(Action.DESCEND))

    assert descended.player_level.tolist() == [final, final]
    assert descended.player_position.tolist() == [[5, 7], [8, 9]]
    assert rows.call_args is not None
    assert rows.call_args.args == (state.num_envs, state.device)
    assert on_device.call_args is not None
    assert on_device.call_args.args[1] == state.device


def test_a_new_floor_pays_experience_and_reward_only_on_first_arrival() -> None:
    state = _state()
    state.item_map[:, 0, 10, 10] = int(ItemType.LADDER_DOWN)
    state.up_ladders[:, 1] = torch.tensor([[5, 7], [8, 9]], dtype=torch.int32)
    state.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    first, reward = step.step(state, _act(Action.DESCEND), generator=_seed())
    assert first.player_position.tolist() == [[5, 7], [8, 9]]
    assert first.player_xp.tolist() == [1, 1]
    assert reward.tolist() == [3.0, 3.0]
    first.achievements[:, int(Achievement.ENTER_DUNGEON)] = True
    first.player_health[0] = 2
    second, second_reward = step.step(
        first,
        _act(Action.DESCEND),
        generator=_seed(),
    )
    assert second.player_xp.tolist() == [1, 1]
    assert second_reward.tolist() == [0.0, 0.0]


def test_is_done_uses_zero_health_and_inclusive_time_limit() -> None:
    state = _state()
    state.player_health[:] = 1
    state.timestep[:] = constants.MAX_TIMESTEPS - 1
    assert step.is_done(state).tolist() == [False, False]
    state.player_health[0] = 0
    state.timestep[1] = constants.MAX_TIMESTEPS
    assert step.is_done(state).tolist() == [True, True]


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


def test_a_world_survives_every_action() -> None:
    """The whole game, driven by every action, must not raise or leave the map."""
    generator = _seed(3)
    state = _state(num_envs=4)
    total = torch.zeros(4)
    dealt = torch.arange(len(constants.Action))
    # Padded to a whole number of steps, wrapping onto NOOP.
    steps = -(-len(constants.Action) // 4)
    dealt = torch.cat(
        [dealt, torch.zeros(steps * 4 - dealt.numel(), dtype=dealt.dtype)],
    )
    seen: set[int] = set()
    for action in dealt.reshape(steps, 4):
        seen.update(int(value) for value in action)
        state, reward = step.step(state, action, generator=generator)
        total += reward

    assert seen == {int(action) for action in constants.Action}
    assert state.timestep.tolist() == [steps] * 4
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
    from priml.lib.testing.main import test_main

    test_main(__file__)
