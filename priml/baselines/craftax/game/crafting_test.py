"""Tests for crafting and block placement."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import crafting
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
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    return state


def _with_table(state: EnvState) -> EnvState:
    state.map[:, 0, 9, 10] = int(BlockType.CRAFTING_TABLE)
    return state


def _with_furnace(state: EnvState) -> EnvState:
    state.map[:, 0, 11, 10] = int(BlockType.FURNACE)
    return state


def _act(action: Action, num_envs: int = 2) -> Tensor:
    return torch.full((num_envs,), int(action), dtype=torch.int32)


def test_a_wood_pickaxe_costs_wood_and_needs_a_table() -> None:
    away = _state()
    away.inventory.wood[:] = 5
    assert crafting.craft(
        away, _act(Action.MAKE_WOOD_PICKAXE)
    ).inventory.pickaxe.tolist() == [
        0,
        0,
    ]

    at_table = _with_table(_state())
    at_table.inventory.wood[:] = 5
    made = crafting.craft(at_table, _act(Action.MAKE_WOOD_PICKAXE))
    assert made.inventory.pickaxe.tolist() == [1, 1]
    assert made.inventory.wood.tolist() == [4, 4]
    assert made.achievements[:, int(Achievement.MAKE_WOOD_PICKAXE)].tolist() == [
        True,
        True,
    ]


def test_crafting_without_the_materials_changes_nothing() -> None:
    state = _with_table(_state())
    made = crafting.craft(state, _act(Action.MAKE_WOOD_PICKAXE))
    assert made.inventory.pickaxe.tolist() == [0, 0]
    assert made.inventory.wood.tolist() == [0, 0]


def test_an_iron_pickaxe_needs_both_stations() -> None:
    # This is the recipe that forces the player to build a workshop rather
    # than carry one block around.
    at_table = _with_table(_state())
    for material in ("wood", "stone", "iron", "coal"):
        getattr(at_table.inventory, material)[:] = 3
    assert crafting.craft(
        at_table, _act(Action.MAKE_IRON_PICKAXE)
    ).inventory.pickaxe.tolist() == [
        0,
        0,
    ]

    both = _with_furnace(_with_table(_state()))
    for material in ("wood", "stone", "iron", "coal"):
        getattr(both.inventory, material)[:] = 3
    made = crafting.craft(both, _act(Action.MAKE_IRON_PICKAXE))
    assert made.inventory.pickaxe.tolist() == [3, 3]
    assert made.inventory.iron.tolist() == [2, 2]
    assert made.inventory.coal.tolist() == [2, 2]


def test_a_tier_already_held_cannot_be_recrafted() -> None:
    # Otherwise a diamond pickaxe could be spent back down to wood.
    state = _with_table(_state())
    state.inventory.wood[:] = 5
    state.inventory.pickaxe[:] = 3
    made = crafting.craft(state, _act(Action.MAKE_WOOD_PICKAXE))
    assert made.inventory.pickaxe.tolist() == [3, 3]
    assert made.inventory.wood.tolist() == [5, 5]


@pytest.mark.parametrize(
    ("action", "tier"),
    [
        (Action.MAKE_WOOD_SWORD, 1),
        (Action.MAKE_STONE_SWORD, 2),
        (Action.MAKE_DIAMOND_SWORD, 4),
    ],
)
def test_swords_reach_their_tier(action: Action, tier: int) -> None:
    state = _with_table(_state())
    for material in ("wood", "stone", "diamond"):
        getattr(state.inventory, material)[:] = 5
    assert crafting.craft(state, _act(action)).inventory.sword.tolist() == [tier, tier]


def test_arrows_and_torches_come_in_batches() -> None:
    state = _with_table(_state())
    state.inventory.wood[:] = 5
    state.inventory.stone[:] = 5
    state.inventory.coal[:] = 5
    arrows = crafting.craft(state, _act(Action.MAKE_ARROW))
    assert arrows.inventory.arrows.tolist() == [2, 2]

    torches = crafting.craft(_with_table(_state()), _act(Action.MAKE_TORCH))
    torches.inventory.wood[:] = 5
    torches.inventory.coal[:] = 5
    torches = crafting.craft(torches, _act(Action.MAKE_TORCH))
    assert torches.inventory.torches.tolist() == [4, 4]


def test_armour_fills_one_slot_at_a_time() -> None:
    state = _with_furnace(_with_table(_state()))
    state.inventory.iron[:] = 30
    state.inventory.coal[:] = 30
    for expected in (1, 2, 3, 4):
        state = crafting.craft(state, _act(Action.MAKE_IRON_ARMOUR))
        assert int((state.inventory.armour >= 1).sum(-1)[0]) == expected
    assert state.achievements[:, int(Achievement.MAKE_IRON_ARMOUR)].tolist() == [
        True,
        True,
    ]


def test_placing_stone_spends_it_and_writes_the_block() -> None:
    state = _state()
    state.inventory.stone[:] = 3
    placed = crafting.place(state, _act(Action.PLACE_STONE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.STONE)
    assert placed.inventory.stone.tolist() == [2, 2]
    assert placed.achievements[:, int(Achievement.PLACE_STONE)].tolist() == [True, True]


def test_placing_needs_the_material() -> None:
    placed = crafting.place(_state(), _act(Action.PLACE_STONE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.GRASS)


def test_a_block_cannot_be_placed_on_stone() -> None:
    # Only loose ground accepts a placement.
    state = _state()
    state.inventory.stone[:] = 3
    state.map[:, 0, 10, 11] = int(BlockType.STONE)
    placed = crafting.place(state, _act(Action.PLACE_TABLE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.STONE)


def test_a_block_cannot_be_placed_on_a_creature() -> None:
    state = _state()
    state.inventory.stone[:] = 3
    state.mob_map[:, 0, 10, 11] = True
    placed = crafting.place(state, _act(Action.PLACE_STONE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.GRASS)


def test_sowing_a_sapling_records_a_growing_plant() -> None:
    state = _state()
    state.inventory.sapling[:] = 1
    placed = crafting.place(state, _act(Action.PLACE_PLANT))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.PLANT)
    assert placed.growing_plants_mask[:, 0].tolist() == [True, True]
    assert placed.growing_plants_positions[0, 0].tolist() == [10, 11]
    assert placed.achievements[:, int(Achievement.PLACE_PLANT)].tolist() == [True, True]


def test_a_torch_lights_its_surroundings() -> None:
    state = _state()
    state.inventory.torches[:] = 2
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert placed.item_map[0, 0, 10, 11].item() == int(ItemType.TORCH)
    assert placed.inventory.torches.tolist() == [1, 1]
    assert float(placed.light_map[0, 0, 10, 11]) == pytest.approx(1.0)
    assert float(placed.light_map[0, 0, 10, 13]) > 0.0
    assert placed.achievements[:, int(Achievement.PLACE_TORCH)].tolist() == [True, True]


def test_a_torch_never_dims_an_already_bright_tile() -> None:
    state = _state()
    state.inventory.torches[:] = 1
    state.light_map[:] = 1.0
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert float(placed.light_map.min()) == pytest.approx(1.0)


def test_only_the_acting_environments_craft() -> None:
    state = _with_table(_state())
    state.inventory.wood[:] = 5
    action = torch.tensor(
        [int(Action.MAKE_WOOD_PICKAXE), int(Action.NOOP)],
        dtype=torch.int32,
    )
    made = crafting.craft(state, action)
    assert made.inventory.pickaxe.tolist() == [1, 0]


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
