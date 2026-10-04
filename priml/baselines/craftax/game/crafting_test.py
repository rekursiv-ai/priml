"""Tests for crafting and block placement."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import constants, crafting
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
        away,
        _act(Action.MAKE_WOOD_PICKAXE),
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


def test_crafting_accepts_exactly_the_required_materials() -> None:
    state = _with_table(_state())
    state.inventory.wood[:] = 1
    made = crafting.craft(state, _act(Action.MAKE_WOOD_PICKAXE))
    assert made.inventory.pickaxe.tolist() == [1, 1]
    assert made.inventory.wood.tolist() == [0, 0]


def test_an_iron_pickaxe_needs_both_stations() -> None:
    # This is the recipe that forces the player to build a workshop rather
    # than carry one block around.
    at_table = _with_table(_state())
    for material in ("wood", "stone", "iron", "coal"):
        getattr(at_table.inventory, material)[:] = 3
    assert crafting.craft(
        at_table,
        _act(Action.MAKE_IRON_PICKAXE),
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

    same_tier = _with_table(_state())
    same_tier.inventory.wood[:] = 5
    same_tier.inventory.pickaxe[:] = 1
    unchanged = crafting.craft(same_tier, _act(Action.MAKE_WOOD_PICKAXE))
    assert unchanged.inventory.pickaxe.tolist() == [1, 1]
    assert unchanged.inventory.wood.tolist() == [5, 5]


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


def test_arrow_and_torch_recipes_stop_at_the_reference_stock_cap() -> None:
    # Upstream game_logic.py:758-785 refuses stock output at 99.
    arrows = _with_table(_state())
    arrows.inventory.wood[:] = 1
    arrows.inventory.stone[:] = 1
    arrows.inventory.arrows[:] = 99
    crafted_arrows = crafting.craft(arrows, _act(Action.MAKE_ARROW))
    assert crafted_arrows.inventory.arrows.tolist() == [99, 99]
    assert crafted_arrows.inventory.wood.tolist() == [1, 1]
    assert crafted_arrows.inventory.stone.tolist() == [1, 1]

    torches = _with_table(_state())
    torches.inventory.wood[:] = 1
    torches.inventory.coal[:] = 1
    torches.inventory.torches[:] = 99
    crafted_torches = crafting.craft(torches, _act(Action.MAKE_TORCH))
    assert crafted_torches.inventory.torches.tolist() == [99, 99]
    assert crafted_torches.inventory.wood.tolist() == [1, 1]
    assert crafted_torches.inventory.coal.tolist() == [1, 1]


def test_armour_fills_one_slot_at_a_time() -> None:
    state = _with_table(_state())
    state.inventory.iron[:] = 12
    state.inventory.coal[:] = 12
    unchanged = crafting.craft(state, _act(Action.MAKE_IRON_ARMOUR))
    assert unchanged.inventory.armour.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert unchanged.inventory.iron.tolist() == [12, 12]
    assert unchanged.inventory.coal.tolist() == [12, 12]
    assert unchanged.achievements[:, int(Achievement.MAKE_IRON_ARMOUR)].tolist() == [
        False,
        False,
    ]

    state = _with_furnace(state)
    for slot in range(4):
        state = crafting.craft(state, _act(Action.MAKE_IRON_ARMOUR))
        assert state.inventory.armour.tolist() == [
            [1 if index <= slot else 0 for index in range(4)],
            [1 if index <= slot else 0 for index in range(4)],
        ]
        assert state.inventory.iron.tolist() == [
            12 - 3 * (slot + 1),
            12 - 3 * (slot + 1),
        ]
        assert state.inventory.coal.tolist() == [
            12 - 3 * (slot + 1),
            12 - 3 * (slot + 1),
        ]
    assert state.achievements[:, int(Achievement.MAKE_IRON_ARMOUR)].tolist() == [
        True,
        True,
    ]


def test_iron_armour_requires_both_stations_and_only_fills_each_available_slot() -> (
    None
):
    state = _with_furnace(_state())
    state.inventory.iron[:] = 6
    state.inventory.coal[:] = 6
    without_table = crafting.craft(state, _act(Action.MAKE_IRON_ARMOUR))
    assert without_table.inventory.armour.tolist() == [[0, 0, 0, 0]] * 2
    assert without_table.inventory.iron.tolist() == [6, 6]
    assert without_table.inventory.coal.tolist() == [6, 6]

    state = _with_table(state)
    state.inventory.armour[:] = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]])
    made = crafting.craft(state, _act(Action.MAKE_IRON_ARMOUR))
    assert made.inventory.armour.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert made.inventory.iron.tolist() == [3, 3]
    assert made.inventory.coal.tolist() == [3, 3]

    full = _with_table(_with_furnace(_state()))
    full.inventory.armour[:] = 1
    full.inventory.iron[:] = 3
    full.inventory.coal[:] = 3
    unchanged = crafting.craft(full, _act(Action.MAKE_IRON_ARMOUR))
    assert unchanged.inventory.armour.tolist() == [[1, 1, 1, 1]] * 2
    assert unchanged.inventory.iron.tolist() == [3, 3]
    assert unchanged.inventory.coal.tolist() == [3, 3]


def test_armour_upgrade_availability_is_per_environment() -> None:
    state = _with_table(_with_furnace(_state()))
    state.inventory.armour[0] = 1
    state.inventory.armour[1] = torch.tensor([1, 1, 0, 1])
    state.inventory.iron[:] = 3
    state.inventory.coal[:] = 3

    made = crafting.craft(state, _act(Action.MAKE_IRON_ARMOUR))

    assert made.inventory.armour.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert made.inventory.iron.tolist() == [3, 0]
    assert made.inventory.coal.tolist() == [3, 0]


def test_diamond_armour_uses_diamonds_and_fills_each_slot_to_tier_two() -> None:
    state = _with_table(_state())
    state.inventory.diamond[:] = 3
    made = crafting.craft(state, _act(Action.MAKE_DIAMOND_ARMOUR))
    assert made.inventory.armour.tolist() == [[2, 0, 0, 0], [2, 0, 0, 0]]
    assert made.inventory.diamond.tolist() == [0, 0]
    assert made.achievements[:, int(Achievement.MAKE_DIAMOND_ARMOUR)].tolist() == [
        True,
        True,
    ]

    insufficient = _with_table(_state())
    insufficient.inventory.diamond[:] = 2
    unchanged = crafting.craft(insufficient, _act(Action.MAKE_DIAMOND_ARMOUR))
    assert unchanged.inventory.armour.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert unchanged.inventory.diamond.tolist() == [2, 2]


def test_placing_a_table_costs_two_wood() -> None:
    # Upstream game_logic.py:841,859 charges two wood for a table.
    state = _state()
    state.inventory.wood[:] = 1
    unchanged = crafting.place(state, _act(Action.PLACE_TABLE))
    assert unchanged.map[0, 0, 10, 11].item() == int(BlockType.GRASS)
    assert unchanged.inventory.wood.tolist() == [1, 1]

    state.inventory.wood[:] = 2
    placed = crafting.place(state, _act(Action.PLACE_TABLE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.CRAFTING_TABLE)
    assert placed.inventory.wood.tolist() == [0, 0]


def test_placing_stone_spends_it_and_writes_the_block() -> None:
    # Upstream game_logic.py:893-918 places stone on nonsolid ground.
    state = _state()
    state.inventory.stone[:] = 3
    placed = crafting.place(state, _act(Action.PLACE_STONE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.STONE)
    assert placed.inventory.stone.tolist() == [2, 2]
    assert placed.achievements[:, int(Achievement.PLACE_STONE)].tolist() == [True, True]


def test_placing_needs_the_material() -> None:
    placed = crafting.place(_state(), _act(Action.PLACE_STONE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.GRASS)


def test_blocks_can_be_placed_on_water() -> None:
    # Upstream game_logic.py:841-859,867-918 allows each block on water.
    for action, material, amount, block in (
        (Action.PLACE_STONE, "stone", 1, BlockType.STONE),
        (Action.PLACE_TABLE, "wood", 2, BlockType.CRAFTING_TABLE),
        (Action.PLACE_FURNACE, "stone", 1, BlockType.FURNACE),
    ):
        state = _state()
        getattr(state.inventory, material)[:] = amount
        state.map[:, 0, 10, 11] = int(BlockType.WATER)
        placed = crafting.place(state, _act(action))
        assert placed.map[0, 0, 10, 11].item() == int(block)
        stock = placed.inventory.wood if material == "wood" else placed.inventory.stone
        assert stock.tolist() == [0, 0]


def test_a_plant_can_only_be_placed_on_grass() -> None:
    # Upstream game_logic.py:995-1004 requires GRASS for saplings.
    state = _state()
    state.inventory.sapling[:] = 1
    state.map[:, 0, 10, 11] = int(BlockType.PATH)
    placed = crafting.place(state, _act(Action.PLACE_PLANT))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.PATH)
    assert placed.inventory.sapling.tolist() == [1, 1]
    assert placed.growing_plants_mask[:, 0].tolist() == [False, False]


def test_a_block_cannot_be_placed_over_an_item_or_creature() -> None:
    # Upstream game_logic.py:831-837,1037-1047 rejects occupied targets.
    item_state = _state()
    item_state.inventory.stone[:] = 1
    item_state.item_map[:, 0, 10, 11] = int(ItemType.TORCH)
    item_result = crafting.place(item_state, _act(Action.PLACE_STONE))
    assert item_result.map[0, 0, 10, 11].item() == int(BlockType.GRASS)
    assert item_result.inventory.stone.tolist() == [1, 1]

    mob_state = _state()
    mob_state.inventory.stone[:] = 1
    mob_state.mob_map[:, 0, 10, 11] = True
    mob_result = crafting.place(mob_state, _act(Action.PLACE_STONE))
    assert mob_result.map[0, 0, 10, 11].item() == int(BlockType.GRASS)
    assert mob_result.inventory.stone.tolist() == [1, 1]


def test_a_torch_cannot_be_placed_on_an_occupied_or_invalid_tile() -> None:
    # Upstream game_logic.py:928-942 checks eligible ground and an empty item tile.
    state = _state()
    state.inventory.torches[:] = 1
    state.item_map[:, 0, 10, 11] = int(ItemType.TORCH)
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert placed.inventory.torches.tolist() == [1, 1]
    assert placed.light_map.count_nonzero().item() == 0


def test_a_block_cannot_be_placed_on_stone() -> None:
    # Upstream game_logic.py:833-837 rejects solid target blocks.
    state = _state()
    state.inventory.stone[:] = 3
    state.map[:, 0, 10, 11] = int(BlockType.STONE)
    placed = crafting.place(state, _act(Action.PLACE_TABLE))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.STONE)


def test_sowing_a_sapling_records_a_growing_plant() -> None:
    state = _state()
    state.inventory.sapling[:] = 1
    placed = crafting.place(state, _act(Action.PLACE_PLANT))
    assert placed.map[0, 0, 10, 11].item() == int(BlockType.PLANT)
    assert placed.growing_plants_mask[:, 0].tolist() == [True, True]
    assert placed.growing_plants_positions[0, 0].tolist() == [10, 11]
    assert placed.achievements[:, int(Achievement.PLACE_PLANT)].tolist() == [True, True]


def test_planting_with_no_free_growth_slot_does_not_overwrite_a_plant() -> None:
    state = _state()
    state.inventory.sapling[:] = 1
    state.growing_plants_mask[:] = True
    state.growing_plants_positions[:] = 7
    planted = crafting.place(state, _act(Action.PLACE_PLANT))
    assert planted.map[0, 0, 10, 11].item() == int(BlockType.PLANT)
    assert planted.growing_plants_mask.all().item()
    assert planted.growing_plants_positions.tolist() == [
        [[7, 7]] * state.growing_plants_positions.shape[1],
        [[7, 7]] * state.growing_plants_positions.shape[1],
    ]
    assert planted.growing_plants_age.count_nonzero().item() == 0


def test_saplings_use_each_environments_first_free_slot() -> None:
    state = _state()
    state.inventory.sapling[:] = 1
    state.growing_plants_mask[0, 0] = True
    state.growing_plants_mask[1, 1] = True
    state.growing_plants_positions[0, 0] = torch.tensor([4, 5])
    state.growing_plants_positions[1, 1] = torch.tensor([6, 7])

    planted = crafting.place(state, _act(Action.PLACE_PLANT))

    assert planted.growing_plants_mask[:, :2].tolist() == [
        [True, True],
        [True, True],
    ]
    assert planted.growing_plants_positions[:, :2].tolist() == [
        [[4, 5], [10, 11]],
        [[10, 11], [6, 7]],
    ]


def test_sapling_without_a_free_slot_preserves_only_that_environment() -> None:
    state = _state(num_envs=3)
    state.inventory.sapling[:] = 1
    state.growing_plants_mask[0] = True
    state.growing_plants_positions[0] = 7
    state.growing_plants_mask[1, 0] = True
    state.growing_plants_positions[1, 0] = torch.tensor([4, 5])

    planted = crafting.place(state, _act(Action.PLACE_PLANT, num_envs=3))

    assert planted.growing_plants_mask[:, :2].tolist() == [
        [True, True],
        [True, True],
        [True, False],
    ]
    assert planted.growing_plants_mask.sum(-1).tolist() == [10, 2, 1]
    assert planted.growing_plants_positions[0].tolist() == [[7, 7]] * 10
    assert planted.growing_plants_positions[:, :2].tolist() == [
        [[7, 7], [7, 7]],
        [[4, 5], [10, 11]],
        [[10, 11], [0, 0]],
    ]


def test_a_torch_lights_its_surroundings() -> None:
    state = _state()
    state.inventory.torches[:] = 2
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert placed.item_map[0, 0, 10, 11].item() == int(ItemType.TORCH)
    assert placed.inventory.torches.tolist() == [1, 1]
    assert float(placed.light_map[0, 0, 10, 11]) == pytest.approx(1.0)
    assert float(placed.light_map[0, 0, 10, 13]) == pytest.approx(0.6000000238418579)
    assert placed.achievements[:, int(Achievement.PLACE_TORCH)].tolist() == [True, True]


def test_torch_does_not_place_on_unsupported_or_occupied_tiles() -> None:
    for block, occupied in (
        (BlockType.STONE, False),
        (BlockType.GRASS, True),
    ):
        state = _state()
        state.inventory.torches[:] = 1
        state.map[:, 0, 10, 11] = int(block)
        state.mob_map[:, 0, 10, 11] = occupied
        placed = crafting.place(state, _act(Action.PLACE_TORCH))
        assert placed.inventory.torches.tolist() == [1, 1]
        assert placed.item_map[0, 0, 10, 11].item() == int(ItemType.NONE)
        assert placed.light_map.count_nonzero().item() == 0
        assert placed.achievements[:, int(Achievement.PLACE_TORCH)].tolist() == [
            False,
            False,
        ]


def test_a_bottom_edge_torch_preserves_the_upstream_glow_value() -> None:
    # Upstream game_logic.py:956-987 pads the map and writes the full glow patch.
    state = _state()
    state.player_position[0] = torch.tensor([47, 10], dtype=torch.int32)
    state.inventory.torches[:] = 1
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert float(placed.light_map[0, 0, 47, 7]) == 0.19999998807907104


@pytest.mark.parametrize(
    ("position", "direction"),
    [((47, 10), Action.DOWN), ((10, 47), Action.RIGHT), ((0, 10), Action.UP)],
)
def test_facing_off_the_map_places_no_torch(
    position: tuple[int, int],
    direction: Action,
) -> None:
    state = _state()
    state.player_position[0] = torch.tensor(position, dtype=torch.int32)
    state.player_direction[0] = int(direction)
    state.inventory.torches[:] = 1
    before = state.light_map.clone()
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert int(placed.inventory.torches[0]) == 1
    assert torch.equal(placed.light_map[0], before[0])


def test_torch_glow_matches_the_upstream_float_bits() -> None:
    # Upstream constants.py:592-594 builds the glow; game_logic.py:971 adds it.
    state = _state()
    state.inventory.torches[:] = 1
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert float(placed.light_map[0, 0, 11, 12]) == 0.717157244682312
    assert float(placed.light_map[0, 0, 11, 13]) == 0.5527863502502441
    assert float(placed.light_map[0, 0, 12, 13]) == 0.4343145489692688
    assert float(placed.light_map[0, 0, 10, 14]) == 0.3999999761581421
    assert float(placed.light_map[0, 0, 12, 14]) == 0.2788897156715393
    assert float(placed.light_map[0, 0, 11, 15]) == 0.17537885904312134
    assert float(placed.light_map[0, 0, 12, 15]) == 0.10557276010513306


def test_a_torch_adds_light_to_an_already_lit_tile() -> None:
    # Upstream game_logic.py:971 adds the torch glow, then clips at one.
    state = _state()
    state.inventory.torches[:] = 1
    state.light_map[:] = 0.5
    placed = crafting.place(state, _act(Action.PLACE_TORCH))
    assert float(placed.light_map[0, 0, 10, 12]) == 1.0
    assert float(placed.light_map.min()) == 0.5


def test_crafting_and_placement_allocate_on_the_state_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_devices: list[torch.device | None] = []
    original_arange = torch.arange
    expected_full_values = {
        int(Achievement.MAKE_WOOD_PICKAXE),
        int(Achievement.MAKE_IRON_ARMOUR),
        int(BlockType.STONE),
        int(Achievement.PLACE_STONE),
        int(ItemType.TORCH),
        int(Achievement.PLACE_TORCH),
    }
    actions = [
        _act(Action.MAKE_WOOD_PICKAXE),
        _act(Action.PLACE_STONE),
        _act(Action.PLACE_TORCH),
    ]

    def record_full(
        size: int | tuple[int, ...],
        fill_value: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if fill_value in expected_full_values:
            requested_devices.append(device)
        result = torch.empty(size, dtype=dtype, device=device)
        return result.fill_(fill_value)

    def record_arange(
        end: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if end == 9:
            requested_devices.append(device)
        return original_arange(end, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "full", record_full)
    monkeypatch.setattr(torch, "arange", record_arange)
    state = _with_table(_state())
    state.inventory.wood[:] = 1
    crafting.craft(state, actions[0])

    armour = _with_furnace(_with_table(_state()))
    armour.inventory.iron[:] = 3
    armour.inventory.coal[:] = 3
    crafting.craft(armour, _act(Action.MAKE_IRON_ARMOUR))

    state = _state()
    state.inventory.stone[:] = 1
    crafting.place(state, actions[1])

    state = _state()
    state.inventory.torches[:] = 1
    crafting.place(state, actions[2])

    assert len(requested_devices) >= 6
    assert requested_devices == [state.device] * len(requested_devices)


def test_device_sensitive_helpers_receive_the_state_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[torch.device | None] = []
    original_on_device = constants.on_device

    def record_batch_rows(num_envs: int, device: torch.device) -> Tensor:
        devices.append(device)
        return batch_rows(num_envs, device)

    def record_on_device(table: Tensor, device: torch.device) -> Tensor:
        tracked_tables = (
            constants.DIRECTIONS,
            constants.SOLID_BLOCK,
            constants.CAN_PLACE_ITEM_ON,
            constants.TORCH_LIGHT_MAP,
        )
        if any(table is tracked for tracked in tracked_tables):
            devices.append(device)
        return original_on_device(table, device)

    monkeypatch.setattr(crafting, "batch_rows", record_batch_rows)
    monkeypatch.setattr(constants, "on_device", record_on_device)

    armour = _with_furnace(_with_table(_state()))
    armour.inventory.iron[:] = 3
    armour.inventory.coal[:] = 3
    crafting.craft(armour, _act(Action.MAKE_IRON_ARMOUR))

    stone = _state()
    stone.inventory.stone[:] = 1
    crafting.place(stone, _act(Action.PLACE_STONE))

    plant = _state()
    plant.inventory.sapling[:] = 1
    crafting.place(plant, _act(Action.PLACE_PLANT))

    torch_state = _state()
    torch_state.inventory.torches[:] = 1
    crafting.place(torch_state, _act(Action.PLACE_TORCH))

    assert devices
    assert None not in devices
    assert all(device == torch.device("cpu") for device in devices)


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
    from priml.lib.testing.main import test_main

    test_main(__file__)
