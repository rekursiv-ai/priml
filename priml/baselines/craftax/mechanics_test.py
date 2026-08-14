"""Tests for the shared game rules."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax import constants, mechanics
from priml.baselines.craftax.constants import BlockType
from priml.baselines.craftax.state import EnvState, empty_state


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    state.player_strength[:] = 1
    state.player_dexterity[:] = 1
    state.player_intelligence[:] = 1
    state.map[:] = int(BlockType.GRASS)
    return state


def test_meter_caps_rise_with_their_attribute() -> None:
    state = _state()
    assert mechanics.max_health(state).tolist() == [9, 9]
    state.player_strength[:] = 5
    assert mechanics.max_health(state).tolist() == [13, 13]
    state.player_dexterity[:] = 5
    assert mechanics.max_food(state).tolist() == [17, 17]
    state.player_intelligence[:] = 5
    assert mechanics.max_mana(state).tolist() == [21, 21]


def test_bare_hands_do_less_damage_than_a_sword() -> None:
    state = _state()
    unarmed = mechanics.player_damage(state)[:, 0]
    state.inventory.sword[:] = 4
    armed = mechanics.player_damage(state)[:, 0]
    assert unarmed.tolist() == [1.0, 1.0]
    assert armed.tolist() == [8.0, 8.0]


def test_strength_doubles_physical_damage_at_the_cap() -> None:
    state = _state()
    state.inventory.sword[:] = 2
    state.player_strength[:] = 5
    assert mechanics.player_damage(state)[:, 0].tolist() == [6.0, 6.0]


def test_an_enchanted_sword_adds_its_element() -> None:
    state = _state()
    state.inventory.sword[:] = 2
    state.sword_enchantment[:] = 1
    damage = mechanics.player_damage(state)
    assert damage[:, 1].tolist() == [1.5, 1.5]
    assert damage[:, 2].tolist() == [0.0, 0.0]


def test_armour_reduces_incoming_damage() -> None:
    state = _state()
    incoming = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    bare = mechanics.damage_to_player(state, incoming)
    state.inventory.armour[:] = 2
    armoured = mechanics.damage_to_player(state, incoming)
    assert bare.tolist() == [10.0, 10.0]
    # Four pieces at two points each block eight tenths of the blow.
    assert armoured.tolist() == pytest.approx([2.0, 2.0])


def test_the_boss_floor_amplifies_incoming_damage() -> None:
    state = _state()
    incoming = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    state.player_level[:] = constants.NUM_LEVELS - 1
    assert mechanics.damage_to_player(state, incoming).tolist() == pytest.approx(
        [15.0, 15.0],
    )


def test_the_boss_is_shielded_while_its_summons_live() -> None:
    state = _state()
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.boss_timesteps_to_spawn_this_round[:] = 0
    assert mechanics.is_boss_vulnerable(state).tolist() == [True, True]
    state.melee_mobs.mask[:, constants.NUM_LEVELS - 1, 0] = True
    assert mechanics.is_boss_vulnerable(state).tolist() == [False, False]


def test_the_boss_is_shielded_between_waves() -> None:
    state = _state()
    state.player_level[:] = constants.NUM_LEVELS - 1
    state.boss_timesteps_to_spawn_this_round[:] = 3
    assert mechanics.is_boss_vulnerable(state).tolist() == [False, False]


def test_walking_is_refused_into_stone_and_allowed_onto_grass() -> None:
    state = _state()
    state.map[:, 0, 5, 5] = int(BlockType.STONE)
    never_collides = torch.zeros(2, 3, dtype=torch.bool)
    assert mechanics.can_walk_on(
        state,
        torch.tensor([[5, 5], [6, 6]]),
        never_collides,
    ).tolist() == [False, True]


def test_walking_off_the_map_is_refused_rather_than_wrapping() -> None:
    # The indexing helpers wrap a negative coordinate, so bounds must be
    # checked here or stepping off the top edge teleports to the bottom.
    state = _state()
    never_collides = torch.zeros(2, 3, dtype=torch.bool)
    assert mechanics.can_walk_on(
        state,
        torch.tensor([[-1, 5], [5, -1]]),
        never_collides,
    ).tolist() == [False, False]


def test_a_land_creature_will_not_enter_water() -> None:
    state = _state()
    state.map[:, 0, 5, 5] = int(BlockType.WATER)
    land = torch.tensor([[False, True, True], [False, True, True]])
    swims = torch.tensor([[True, False, True], [True, False, True]])
    target = torch.tensor([[5, 5], [5, 5]])
    assert mechanics.can_walk_on(state, target, land).tolist() == [False, False]
    assert mechanics.can_walk_on(state, target, swims).tolist() == [True, True]


def test_a_tile_holding_a_creature_is_occupied() -> None:
    state = _state()
    state.mob_map[:, 0, 5, 5] = True
    never_collides = torch.zeros(2, 3, dtype=torch.bool)
    assert mechanics.can_walk_on(
        state,
        torch.tensor([[5, 5], [6, 6]]),
        never_collides,
    ).tolist() == [False, True]


def test_the_player_blocks_their_own_tile() -> None:
    state = _state()
    assert mechanics.is_occupied(state, state.player_position).tolist() == [True, True]


def test_adjacency_finds_a_neighbouring_block_but_not_a_distant_one() -> None:
    state = _state()
    state.map[:, 0, 10, 11] = int(BlockType.CRAFTING_TABLE)
    assert mechanics.is_near_block(state, int(BlockType.CRAFTING_TABLE)).tolist() == [
        True,
        True,
    ]
    assert mechanics.is_near_block(state, int(BlockType.FURNACE)).tolist() == [
        False,
        False,
    ]


def test_adjacency_ignores_the_tile_underfoot() -> None:
    # A table the player stands on is not usable; it must be beside them.
    state = _state()
    state.map[:, 0, 10, 10] = int(BlockType.CRAFTING_TABLE)
    assert mechanics.is_near_block(state, int(BlockType.CRAFTING_TABLE)).tolist() == [
        False,
        False,
    ]


def test_clipping_holds_meters_and_stocks_in_range() -> None:
    state = _state()
    state.player_health[:] = 99.0
    state.player_food[:] = -5
    state.inventory.wood[:] = 500
    clipped = mechanics.clip_meters(state)
    assert clipped.player_health.tolist() == [9.0, 9.0]
    assert clipped.player_food.tolist() == [0, 0]
    assert clipped.inventory.wood.tolist() == [99, 99]


def test_unlocking_an_achievement_touches_only_the_named_one() -> None:
    state = _state()
    updated = mechanics.unlock_achievement(
        state,
        torch.tensor([int(constants.Achievement.COLLECT_WOOD)] * 2),
        torch.tensor([True, False]),
    )
    assert updated[0, int(constants.Achievement.COLLECT_WOOD)]
    assert not updated[1, int(constants.Achievement.COLLECT_WOOD)]
    assert not updated[:, int(constants.Achievement.PLACE_TABLE)].any()


def test_attacking_reduces_health_and_kills_at_zero() -> None:
    state = _state()
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 3.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)

    mobs, killed, struck, _ = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[5, 5], [5, 5]]),
        damage=torch.tensor([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        mob_class=1,
        can_unlock=torch.tensor([True, True]),
    )

    assert struck.tolist() == [True, True]
    assert killed.tolist() == [False, True]
    assert mobs.health[0, 0, 0].item() == pytest.approx(2.0)


def test_attacking_an_empty_tile_does_nothing() -> None:
    state = _state()
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 3.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)

    mobs, killed, struck, _ = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[9, 9], [9, 9]]),
        damage=torch.tensor([[5.0, 0.0, 0.0]] * 2),
        mob_class=1,
        can_unlock=torch.tensor([True, True]),
    )

    assert struck.tolist() == [False, False]
    assert killed.tolist() == [False, False]
    assert mobs.health[0, 0, 0].item() == pytest.approx(3.0)


def test_a_kill_unlocks_the_species_achievement() -> None:
    state = _state()
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 1.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([5, 5], dtype=torch.int32)

    _, _, _, achievements = mechanics.attack_mob_class(
        state,
        state.melee_mobs,
        position=torch.tensor([[5, 5], [5, 5]]),
        damage=torch.tensor([[5.0, 0.0, 0.0]] * 2),
        mob_class=1,
        can_unlock=torch.tensor([True, False]),
    )

    zombie = int(constants.Achievement.DEFEAT_ZOMBIE)
    assert achievements[0, zombie]
    assert not achievements[1, zombie]


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
