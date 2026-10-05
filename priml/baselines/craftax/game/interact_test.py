"""Tests for the interact action."""

from __future__ import annotations

from typing import Literal

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import constants, indexing, interact
from priml.baselines.craftax.game.constants import (
    Achievement,
    Action,
    BlockType,
    ItemType,
)
from priml.baselines.craftax.game.state import EnvState, Inventory, empty_state


def _state(num_envs: int = 2) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    # Facing right, so the tile under test is always [10, 11].
    state.player_direction[:] = int(Action.RIGHT)
    state.map[:] = int(BlockType.GRASS)
    state.player_health[:] = 9.0
    for meter in ("player_food", "player_drink", "player_energy", "player_mana"):
        getattr(state, meter)[:] = 9
    state.player_dexterity[:] = 1
    state.player_strength[:] = 1
    state.player_intelligence[:] = 1
    return state


def _facing(state: EnvState, block: BlockType) -> EnvState:
    state.map[:, 0, 10, 11] = int(block)
    return state


def _all(num_envs: int = 2) -> Tensor:
    return torch.ones(num_envs, dtype=torch.bool)


def _quiet() -> torch.Generator:
    # Seeded so the sapling draw never fires and cannot confound a test.
    return torch.Generator().manual_seed(0)


def _ore(
    inventory: Inventory,
    name: Literal["coal", "iron", "diamond", "sapphire", "ruby"],
) -> Tensor:
    match name:
        case "coal":
            return inventory.coal
        case "iron":
            return inventory.iron
        case "diamond":
            return inventory.diamond
        case "sapphire":
            return inventory.sapphire
        case "ruby":
            return inventory.ruby


def test_item_at_reads_the_current_floor_and_ladder_checks_kind() -> None:
    state = _state()
    state.item_map[:, 0, 10, 11] = int(ItemType.LADDER_DOWN)
    item = interact.item_at(state, torch.tensor([[10, 11], [10, 11]]))
    assert item.tolist() == [int(ItemType.LADDER_DOWN)] * 2
    assert interact.is_ladder(item, ItemType.LADDER_DOWN).tolist() == [True, True]
    assert interact.is_ladder(item, ItemType.LADDER_UP).tolist() == [False, False]


def test_chopping_a_tree_yields_wood_and_leaves_grass() -> None:
    state = interact.interact(
        _facing(_state(), BlockType.TREE),
        doing=_all(),
        generator=_quiet(),
    )
    assert state.inventory.wood.tolist() == [1, 1]
    assert state.map[0, 0, 10, 11].item() == int(BlockType.GRASS)
    assert state.achievements[:, int(Achievement.COLLECT_WOOD)].tolist() == [True, True]


def test_stone_needs_a_pickaxe() -> None:
    # This gate is the game's whole progression spine.
    bare = interact.interact(
        _facing(_state(), BlockType.STONE),
        doing=_all(),
        generator=_quiet(),
    )
    assert bare.inventory.stone.tolist() == [0, 0]
    assert bare.map[0, 0, 10, 11].item() == int(BlockType.STONE)

    equipped = _facing(_state(), BlockType.STONE)
    equipped.inventory.pickaxe[:] = 1
    mined = interact.interact(equipped, doing=_all(), generator=_quiet())
    assert mined.inventory.stone.tolist() == [1, 1]
    assert mined.map[0, 0, 10, 11].item() == int(BlockType.PATH)


@pytest.mark.parametrize(
    ("block", "resource", "tier"),
    [
        (BlockType.COAL, "coal", 1),
        (BlockType.IRON, "iron", 2),
        (BlockType.DIAMOND, "diamond", 3),
        (BlockType.SAPPHIRE, "sapphire", 4),
        (BlockType.RUBY, "ruby", 4),
    ],
)
def test_each_ore_needs_its_own_pickaxe_tier(
    block: BlockType,
    resource: Literal["coal", "iron", "diamond", "sapphire", "ruby"],
    tier: int,
) -> None:
    too_weak = _facing(_state(), block)
    too_weak.inventory.pickaxe[:] = tier - 1
    blocked = interact.interact(too_weak, doing=_all(), generator=_quiet())
    assert _ore(blocked.inventory, resource).tolist() == [0, 0]

    ready = _facing(_state(), block)
    ready.inventory.pickaxe[:] = tier
    mined = interact.interact(ready, doing=_all(), generator=_quiet())
    assert _ore(mined.inventory, resource).tolist() == [1, 1]


@pytest.mark.parametrize(
    ("block", "leaves"),
    [
        (BlockType.FIRE_TREE, BlockType.FIRE_GRASS),
        (BlockType.ICE_SHRUB, BlockType.ICE_GRASS),
    ],
)
def test_other_trees_yield_wood(
    block: BlockType,
    leaves: BlockType,
) -> None:
    state = interact.interact(
        _facing(_state(), block),
        doing=_all(),
        generator=_quiet(),
    )

    assert state.inventory.wood.tolist() == [1, 1]
    assert state.map[:, 0, 10, 11].tolist() == [int(leaves)] * 2


def test_stalagmites_need_stone_pickaxes() -> None:
    state = _facing(_state(), BlockType.STALAGMITE)
    state.inventory.pickaxe[:] = 1

    state = interact.interact(state, doing=_all(), generator=_quiet())

    assert state.inventory.stone.tolist() == [1, 1]
    assert state.map[:, 0, 10, 11].tolist() == [int(BlockType.PATH)] * 2


def test_drinking_water_fills_the_meter_and_resets_thirst() -> None:
    state = _facing(_state(), BlockType.WATER)
    state.player_drink[:] = 3
    state.player_thirst[:] = 15.0
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert state.player_drink.tolist() == [4, 4]
    assert state.player_thirst.tolist() == [0.0, 0.0]
    assert state.achievements[:, int(Achievement.COLLECT_DRINK)].tolist() == [
        True,
        True,
    ]


def test_drinking_from_a_fountain_is_capped_at_the_meter_maximum() -> None:
    state = _facing(_state(), BlockType.WATER)
    state.map[1, 0, 10, 11] = int(BlockType.FOUNTAIN)
    state.player_drink[:] = torch.tensor([8, 9])
    state.player_thirst[:] = 15.0

    state = interact.interact(state, doing=_all(), generator=_quiet())

    assert state.player_drink.tolist() == [9, 9]
    assert state.player_thirst.tolist() == [0.0, 0.0]
    assert state.achievements[:, int(Achievement.COLLECT_DRINK)].tolist() == [
        True,
        True,
    ]


def test_eating_a_ripe_plant_feeds_and_leaves_it_growing() -> None:
    state = _facing(_state(), BlockType.RIPE_PLANT)
    state.player_food[:] = 3
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert state.player_food.tolist() == [7, 7]
    assert state.map[0, 0, 10, 11].item() == int(BlockType.PLANT)
    assert state.achievements[:, int(Achievement.EAT_PLANT)].tolist() == [True, True]


def test_eating_resets_only_the_faced_plant_growth() -> None:
    state = _facing(_state(), BlockType.RIPE_PLANT)
    target = torch.tensor([10, 11], dtype=torch.int32)
    state.growing_plants_positions[:] = torch.tensor([3, 4], dtype=torch.int32)
    state.growing_plants_positions[0, 0] = target
    state.growing_plants_age[:] = 9
    state.growing_plants_age[0, 0] = 17
    state.player_food[:] = 3
    state.player_hunger[:] = torch.tensor([2.5, 3.5])

    state = interact.interact(
        state,
        doing=torch.tensor([True, False]),
        generator=_quiet(),
    )

    assert state.growing_plants_age[:, 0].tolist() == [0, 9]
    assert state.growing_plants_age[0, 1].item() == 9
    assert state.player_hunger.tolist() == [0.0, 3.5]
    assert state.player_food.tolist() == [7, 3]
    assert state.map[:, 0, 10, 11].tolist() == [
        int(BlockType.PLANT),
        int(BlockType.RIPE_PLANT),
    ]


def test_eating_restarts_the_first_matching_slot_or_else_slot_zero() -> None:
    """Upstream restarts one slot, the argmax of the matches (game_logic.py:12-22).

    A ripe plant no slot tracks restarts slot 0, and of two slots on one tile
    only the first restarts. Play keeps one crop per tile; this pins the rule.
    """
    state = _facing(_state(), BlockType.RIPE_PLANT)
    state.growing_plants_mask[:, :3] = True
    state.growing_plants_positions[:, :3] = torch.tensor(
        [[5, 5], [6, 6], [6, 6]],
        dtype=torch.int32,
    )
    state.growing_plants_positions[1, 1:3] = torch.tensor([10, 11], dtype=torch.int32)
    state.growing_plants_age[:, :3] = torch.tensor([300, 400, 500])
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert state.growing_plants_age[:, :3].tolist() == [[0, 400, 500], [300, 0, 500]]


def test_opening_a_chest_records_it_and_clears_the_tile() -> None:
    state = interact.interact(
        _facing(_state(), BlockType.CHEST),
        doing=_all(),
        generator=_quiet(),
    )
    assert state.map[0, 0, 10, 11].item() == int(BlockType.PATH)
    assert state.chests_opened[:, 0].tolist() == [True, True]
    assert state.achievements[:, int(Achievement.OPEN_CHEST)].tolist() == [True, True]
    assert state.inventory.torches.tolist() == [7, 7]
    assert state.inventory.coal.tolist() == [0, 0]
    assert state.inventory.iron.tolist() == [0, 0]
    assert state.inventory.diamond.tolist() == [1, 0]
    assert state.inventory.sapphire.tolist() == [0, 0]
    assert state.inventory.ruby.tolist() == [0, 0]
    assert state.inventory.arrows.tolist() == [0, 0]
    assert state.inventory.pickaxe.tolist() == [0, 0]
    assert state.inventory.sword.tolist() == [0, 0]
    assert state.inventory.potions.tolist() == [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    assert state.inventory.bow.tolist() == [0, 0]
    assert state.inventory.books.tolist() == [0, 0]


def test_chest_loot_values_are_seeded() -> None:
    state = interact.interact(
        _facing(_state(num_envs=32), BlockType.CHEST),
        doing=_all(32),
        generator=torch.Generator().manual_seed(0),
    )

    assert state.inventory.torches.tolist() == [
        7,
        4,
        6,
        4,
        4,
        0,
        0,
        0,
        6,
        4,
        4,
        5,
        7,
        4,
        0,
        0,
        6,
        7,
        0,
        0,
        0,
        7,
        0,
        5,
        7,
        6,
        7,
        7,
        0,
        0,
        7,
        0,
    ]
    assert state.inventory.coal.tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        3,
        0,
        0,
        3,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        3,
        0,
        2,
    ]
    assert state.inventory.iron.tolist() == [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        2,
        1,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
    ]
    assert state.inventory.diamond.tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
    ]
    assert state.inventory.sapphire.tolist() == [
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert state.inventory.ruby.tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    ]
    assert state.inventory.arrows.tolist() == [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        4,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
    ]
    assert state.inventory.pickaxe.tolist() == [0] * 10 + [3] + [0] * 21
    assert state.inventory.sword.tolist() == [0] * 21 + [3] + [0] * 8 + [3, 0]
    assert torch.nonzero(state.inventory.potions, as_tuple=False).tolist() == [
        [0, 1],
        [2, 2],
        [3, 2],
        [4, 3],
        [5, 3],
        [11, 4],
        [13, 1],
        [14, 5],
        [18, 4],
        [21, 2],
        [23, 4],
        [25, 3],
        [26, 2],
        [29, 5],
        [30, 0],
        [31, 4],
    ]
    assert state.inventory.potions[state.inventory.potions != 0].tolist() == [
        2,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
    ]
    assert state.inventory.bow.tolist() == [0] * 32
    assert state.inventory.books.tolist() == [0] * 32


def test_chest_drop_chance_boundaries_are_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _facing(_state(num_envs=1), BlockType.CHEST)
    boundaries = iter((0.5, 0.6, 0.6, 0.5, 0.5, 0.25, 0.2))
    original_full = torch.full
    draws = 0

    def rand(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        del generator
        nonlocal draws
        draws += 1
        return original_full((size,), next(boundaries), device=device)

    monkeypatch.setattr(torch, "rand", rand)
    opened = interact.interact(state, doing=_all(1), generator=_quiet())

    assert draws == 7
    assert opened.map[0, 0, 10, 11].item() == int(BlockType.PATH)
    assert opened.inventory.torches.tolist() == [0]
    assert opened.inventory.coal.tolist() == [0]
    assert opened.inventory.iron.tolist() == [0]
    assert opened.inventory.diamond.tolist() == [0]
    assert opened.inventory.sapphire.tolist() == [0]
    assert opened.inventory.ruby.tolist() == [0]
    assert opened.inventory.arrows.tolist() == [0]
    assert opened.inventory.pickaxe.tolist() == [0]
    assert opened.inventory.sword.tolist() == [0]
    assert opened.inventory.potions.tolist() == [[0, 0, 0, 0, 0, 0]]


def test_chest_loot_adds_to_existing_inventory() -> None:
    state = _facing(_state(num_envs=32), BlockType.CHEST)
    state.inventory.torches[:] = 10
    state.inventory.arrows[:] = 5

    opened = interact.interact(
        state,
        doing=_all(32),
        generator=torch.Generator().manual_seed(0),
    )

    assert opened.inventory.torches[:6].tolist() == [17, 14, 16, 14, 14, 10]
    assert opened.inventory.arrows[:4].tolist() == [5, 5, 6, 5]
    assert all(
        value.dtype == torch.int32
        for value in (
            opened.inventory.torches,
            opened.inventory.coal,
            opened.inventory.iron,
            opened.inventory.diamond,
            opened.inventory.sapphire,
            opened.inventory.ruby,
            opened.inventory.arrows,
            opened.inventory.pickaxe,
            opened.inventory.sword,
            opened.inventory.potions,
            opened.inventory.books,
        )
    )


def test_book_loot_adds_to_existing_inventory() -> None:
    state = _state(num_envs=1)
    state.player_level[:] = 3
    state.map[:, 3, 10, 11] = int(BlockType.CHEST)
    state.inventory.books[:] = 2

    opened = interact.interact(state, doing=_all(1), generator=_quiet())

    assert opened.inventory.books.tolist() == [3]


def test_chest_loot_is_masked_by_doing() -> None:
    state = interact.interact(
        _facing(_state(), BlockType.CHEST),
        doing=torch.tensor([True, False]),
        generator=_quiet(),
    )

    assert state.map[:, 0, 10, 11].tolist() == [
        int(BlockType.PATH),
        int(BlockType.CHEST),
    ]
    assert state.chests_opened[:, 0].tolist() == [True, False]
    assert state.achievements[:, int(Achievement.OPEN_CHEST)].tolist() == [True, False]
    assert state.inventory.coal[1].item() == 0
    assert state.inventory.iron[1].item() == 0
    assert state.inventory.diamond[1].item() == 0
    assert state.inventory.sapphire[1].item() == 0
    assert state.inventory.ruby[1].item() == 0
    assert state.inventory.torches[1].item() == 0
    assert state.inventory.arrows[1].item() == 0
    assert state.inventory.potions[1].tolist() == [0, 0, 0, 0, 0, 0]


def test_tensor_factories_use_the_state_device(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _facing(_state(), BlockType.CHEST)
    device = state.device
    factory_devices: list[torch.device | str | None] = []
    original_full = torch.full
    original_ones = torch.ones
    original_zeros = torch.zeros
    original_rand = torch.rand
    original_randint = torch.randint
    original_batch_rows = indexing.batch_rows
    original_on_device = constants.on_device

    def full(
        size: tuple[int, ...],
        fill_value: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_devices.append(device)
        return original_full(size, fill_value, device=device)

    def ones(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_devices.append(device)
        assert dtype in (torch.int32, torch.bool)
        return original_ones(size, dtype=dtype, device=device)

    def zeros(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_devices.append(device)
        assert dtype == torch.bool
        return original_zeros(size, dtype=dtype, device=device)

    def rand(
        *size: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_devices.append(device)
        return original_rand(*size, generator=generator, device=device)

    def randint(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        factory_devices.append(device)
        return original_randint(
            low,
            high,
            size,
            generator=generator,
            device=device,
        )

    table_devices: list[torch.device] = []

    def on_device(table: Tensor, target: torch.device) -> Tensor:
        table_devices.append(target)
        return original_on_device(table, target)

    row_devices: list[torch.device] = []

    def batch_rows(envs: int, target: torch.device) -> Tensor:
        row_devices.append(target)
        return original_batch_rows(envs, target)

    doing = _all()
    monkeypatch.setattr(torch, "full", full)
    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(interact, "batch_rows", batch_rows)
    monkeypatch.setattr(constants, "on_device", on_device)

    interact.interact(state, doing=doing, generator=_quiet())

    assert factory_devices
    assert all(target == device for target in factory_devices)
    assert table_devices
    assert all(target == device for target in table_devices)
    assert row_devices
    assert all(target == device for target in row_devices)


def test_first_chest_grants_a_bow_on_floor_one() -> None:
    state = _state()
    state.player_level[:] = 1
    state.map[:, 1, 10, 11] = int(BlockType.CHEST)
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert state.inventory.bow.tolist() == [1, 1]


def test_only_a_floor_three_first_chest_grants_books() -> None:
    state = _state()
    state.player_level[:] = 3
    state.map[:, 3, 10, 11] = int(BlockType.CHEST)
    state.chests_opened[0, 3] = True

    state = interact.interact(
        state,
        doing=torch.tensor([False, True]),
        generator=_quiet(),
    )

    assert state.inventory.books.tolist() == [0, 1]
    assert state.chests_opened[:, 3].tolist() == [True, True]
    assert state.map[:, 3, 10, 11].tolist() == [
        int(BlockType.CHEST),
        int(BlockType.PATH),
    ]


@pytest.mark.parametrize(("floor", "books"), [(4, [1, 1]), (5, [0, 0])])
def test_books_drop_only_on_floors_four_and_not_five(
    floor: int,
    books: list[int],
) -> None:
    state = _state()
    state.player_level[:] = floor
    state.map[:, floor, 10, 11] = int(BlockType.CHEST)

    opened = interact.interact(state, doing=_all(), generator=_quiet())

    assert opened.inventory.books.tolist() == books
    assert opened.chests_opened[:, floor].tolist() == [True, True]


def test_an_unopened_chest_does_not_reward_books_without_interacting() -> None:
    state = _state(num_envs=1)
    state.player_level[:] = 3
    state.map[:, 3, 10, 11] = int(BlockType.CHEST)

    unchanged = interact.interact(
        state,
        doing=torch.tensor([False]),
        generator=_quiet(),
    )

    assert unchanged.inventory.books.tolist() == [0]
    assert unchanged.chests_opened[0, 3].item() is False
    assert unchanged.map[0, 3, 10, 11].item() == int(BlockType.CHEST)


def test_a_creature_takes_the_blow_instead_of_the_block() -> None:
    # Otherwise the player would mine through whatever is attacking them.
    state = _facing(_state(), BlockType.TREE)
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 5.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)
    state.mob_map[:, 0, 10, 11] = True
    state.monsters_killed[:, 0] = torch.tensor([3, 5])

    state = interact.interact(state, doing=_all(), generator=_quiet())

    assert state.melee_mobs.health[0, 0, 0].item() == pytest.approx(4.0)
    assert state.inventory.wood.tolist() == [0, 0]
    assert state.map[0, 0, 10, 11].item() == int(BlockType.TREE)
    assert state.mob_map[:, 0, 10, 11].tolist() == [True, True]
    assert state.monsters_killed[:, 0].tolist() == [3, 5]


def test_a_ranged_creature_takes_the_blow() -> None:
    state = _facing(_state(), BlockType.TREE)
    state.ranged_mobs.mask[:, 0, 0] = True
    state.ranged_mobs.health[:, 0, 0] = 2.0
    state.ranged_mobs.position[:, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)

    state = interact.interact(state, doing=_all(), generator=_quiet())

    assert state.ranged_mobs.health[:, 0, 0].tolist() == [1.0, 1.0]
    assert state.inventory.wood.tolist() == [0, 0]
    assert state.map[:, 0, 10, 11].tolist() == [int(BlockType.TREE)] * 2


def test_killing_a_cow_feeds_the_player() -> None:
    state = _state()
    state.passive_mobs.mask[:, 0, 0] = True
    state.passive_mobs.health[:, 0, 0] = 0.5
    state.passive_mobs.position[:, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)
    state.player_food[:] = 2

    state = interact.interact(state, doing=_all(), generator=_quiet())

    assert state.player_food.tolist() == [8, 8]
    assert state.achievements[:, int(Achievement.EAT_COW)].tolist() == [True, True]


def test_killing_a_monster_counts_toward_clearing_the_floor() -> None:
    state = _state()
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 0.5
    state.melee_mobs.position[:, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)

    state = interact.interact(state, doing=_all(), generator=_quiet())

    assert state.monsters_killed[:, 0].tolist() == [1, 1]


def test_a_non_do_action_does_not_kill_the_faced_mob() -> None:
    state = _state(num_envs=1)
    state.melee_mobs.mask[0, 0, 0] = True
    state.melee_mobs.health[0, 0, 0] = 0.5
    state.melee_mobs.position[0, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)
    state.mob_map[0, 0, 10, 11] = True

    state = interact.interact(state, doing=torch.tensor([False]), generator=_quiet())

    # Upstream restores old_state when the action is not DO (game_logic.py:509-515).
    assert state.melee_mobs.mask[0, 0, 0]
    assert state.melee_mobs.health[0, 0, 0].item() == 0.5
    assert state.mob_map[0, 0, 10, 11]
    assert state.monsters_killed[0, 0].item() == 0
    assert not state.achievements[0, int(Achievement.DEFEAT_ZOMBIE)]


def test_a_do_kill_clears_the_mob_occupancy_tile() -> None:
    state = _state(num_envs=1)
    state.melee_mobs.mask[0, 0, 0] = True
    state.melee_mobs.health[0, 0, 0] = 0.5
    state.melee_mobs.position[0, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)
    state.mob_map[0, 0, 10, 11] = True

    state = interact.interact(state, doing=torch.tensor([True]), generator=_quiet())

    # Upstream clears the killed mob tile (game_logic_utils.py:165-175).
    assert not state.melee_mobs.mask[0, 0, 0]
    assert not state.mob_map[0, 0, 10, 11]
    assert state.monsters_killed[0, 0].item() == 1
    assert state.achievements[0, int(Achievement.DEFEAT_ZOMBIE)]


def test_not_interacting_leaves_the_world_alone() -> None:
    state = interact.interact(
        _facing(_state(), BlockType.TREE),
        doing=torch.tensor([True, False]),
        generator=_quiet(),
    )
    assert state.inventory.wood.tolist() == [1, 0]
    assert state.map[1, 0, 10, 11].item() == int(BlockType.TREE)


def test_not_interacting_does_not_drink_from_the_faced_tile() -> None:
    state = _facing(_state(num_envs=1), BlockType.WATER)
    state.player_drink[:] = 2
    state.player_thirst[:] = 15

    unchanged = interact.interact(
        state,
        doing=torch.tensor([False]),
        generator=_quiet(),
    )

    assert unchanged.player_drink.tolist() == [2]
    assert unchanged.player_thirst.tolist() == [15.0]
    assert unchanged.achievements[0, int(Achievement.COLLECT_DRINK)].item() is False


def test_interacting_off_the_map_does_nothing() -> None:
    state = _state()
    state.player_position[:] = torch.tensor([0, 0], dtype=torch.int32)
    state.player_direction[:] = int(Action.UP)
    before = state.map.clone()
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert torch.equal(state.map, before)


def _on_boss_floor(state: EnvState) -> EnvState:
    """Put the player on the final floor, facing the necromancer."""
    boss_floor = constants.NUM_LEVELS - 1
    state.player_level[:] = boss_floor
    state.map[:, boss_floor, 10, 11] = int(BlockType.NECROMANCER)
    return state


def test_the_boss_takes_damage_only_when_exposed() -> None:
    shielded = _on_boss_floor(_state())
    shielded.boss_timesteps_to_spawn_this_round[:] = 3
    guarded = interact.interact(shielded, doing=_all(), generator=_quiet())
    assert guarded.boss_progress.tolist() == [0, 0]

    exposed = _on_boss_floor(_state())
    exposed.boss_timesteps_to_spawn_this_round[:] = 0
    wounded = interact.interact(exposed, doing=_all(), generator=_quiet())
    assert wounded.boss_progress.tolist() == [1, 1]
    # Each wound summons the next wave.
    assert (
        wounded.boss_timesteps_to_spawn_this_round.tolist()
        == [
            constants.BOSS_FIGHT_SPAWN_TURNS,
        ]
        * 2
    )
    assert wounded.achievements[:, int(Achievement.DAMAGE_NECROMANCER)].tolist() == [
        True,
        True,
    ]


def test_a_chest_seen_off_the_map_still_marks_the_floor_opened() -> None:
    """Upstream sets the chest flag outside its in-bounds gate (game_logic.py:431, 467).

    Facing up from row 0 reads row 47, as JAX wraps a negative index. Nothing is
    looted and the chest stays, but the floor's first-chest book is spent.
    """
    state = _state()
    state.player_level[:] = 4
    state.player_position[:] = torch.tensor([0, 7], dtype=torch.int32)
    state.player_direction[:] = int(Action.UP)
    state.map[0, 4, 47, 7] = int(BlockType.CHEST)
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert state.chests_opened[:, 4].tolist() == [True, False]
    assert state.map[0, 4, 47, 7].item() == int(BlockType.CHEST)
    assert state.inventory.books.tolist() == [0, 0]
    assert not state.achievements[:, int(Achievement.OPEN_CHEST)].any()


def test_a_necromancer_seen_off_the_map_is_still_wounded() -> None:
    """Upstream advances the fight outside its in-bounds gate (game_logic.py:442-457).

    Facing left from column 0 reads column 47. The wave timer restarts, but the
    achievement, which the gate does cover, is not unlocked.
    """
    boss_floor = constants.NUM_LEVELS - 1
    state = _state()
    state.player_level[:] = boss_floor
    state.player_position[:] = torch.tensor([20, 0], dtype=torch.int32)
    state.player_direction[:] = int(Action.LEFT)
    state.map[0, boss_floor, 20, 47] = int(BlockType.NECROMANCER)
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert state.boss_progress.tolist() == [1, 0]
    assert state.boss_timesteps_to_spawn_this_round.tolist() == [
        constants.BOSS_FIGHT_SPAWN_TURNS,
        0,
    ]
    assert not state.achievements[:, int(Achievement.DAMAGE_NECROMANCER)].any()


def test_grass_sometimes_yields_a_sapling() -> None:
    # One in ten, so a large batch must contain both outcomes.
    state = empty_state(num_envs=256, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([10, 10], dtype=torch.int32)
    state.player_direction[:] = int(Action.RIGHT)
    state.map[:] = int(BlockType.GRASS)
    state.player_dexterity[:] = 1

    state = interact.interact(
        state,
        doing=torch.ones(256, dtype=torch.bool),
        generator=torch.Generator().manual_seed(1),
    )

    collected = int(state.inventory.sapling.sum())
    assert 5 < collected < 60


def test_sapling_chance_boundary_is_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _facing(_state(num_envs=1), BlockType.GRASS)
    original_full = torch.full

    def rand(
        size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        del generator
        return original_full((size,), 0.1, device=device)

    monkeypatch.setattr(torch, "rand", rand)
    gathered = interact.interact(state, doing=_all(1), generator=_quiet())

    assert gathered.inventory.sapling.tolist() == [0]
    assert gathered.achievements[0, int(Achievement.COLLECT_SAPLING)].item() is False


def test_a_crafting_table_can_be_reclaimed_but_yields_nothing() -> None:
    state = interact.interact(
        _facing(_state(), BlockType.CRAFTING_TABLE),
        doing=_all(),
        generator=_quiet(),
    )
    assert state.map[0, 0, 10, 11].item() == int(BlockType.PATH)
    assert state.inventory.wood.tolist() == [0, 0]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
