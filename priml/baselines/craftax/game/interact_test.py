"""Tests for the interact action."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import constants, interact
from priml.baselines.craftax.game.constants import Achievement, Action, BlockType
from priml.baselines.craftax.game.state import EnvState, empty_state


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
    resource: str,
    tier: int,
) -> None:
    too_weak = _facing(_state(), block)
    too_weak.inventory.pickaxe[:] = tier - 1
    blocked = interact.interact(too_weak, doing=_all(), generator=_quiet())
    assert getattr(blocked.inventory, resource).tolist() == [0, 0]

    ready = _facing(_state(), block)
    ready.inventory.pickaxe[:] = tier
    mined = interact.interact(ready, doing=_all(), generator=_quiet())
    assert getattr(mined.inventory, resource).tolist() == [1, 1]


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


def test_eating_a_ripe_plant_feeds_and_leaves_it_growing() -> None:
    state = _facing(_state(), BlockType.RIPE_PLANT)
    state.player_food[:] = 3
    state = interact.interact(state, doing=_all(), generator=_quiet())
    assert state.player_food.tolist() == [7, 7]
    assert state.map[0, 0, 10, 11].item() == int(BlockType.PLANT)
    assert state.achievements[:, int(Achievement.EAT_PLANT)].tolist() == [True, True]


def test_opening_a_chest_records_it_and_clears_the_tile() -> None:
    state = interact.interact(
        _facing(_state(), BlockType.CHEST),
        doing=_all(),
        generator=_quiet(),
    )
    assert state.map[0, 0, 10, 11].item() == int(BlockType.PATH)
    assert state.chests_opened[:, 0].tolist() == [True, True]
    assert state.achievements[:, int(Achievement.OPEN_CHEST)].tolist() == [True, True]


def test_a_creature_takes_the_blow_instead_of_the_block() -> None:
    # Otherwise the player would mine through whatever is attacking them.
    state = _facing(_state(), BlockType.TREE)
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.health[:, 0, 0] = 5.0
    state.melee_mobs.position[:, 0, 0] = torch.tensor([10, 11], dtype=torch.int32)

    state = interact.interact(state, doing=_all(), generator=_quiet())

    assert state.melee_mobs.health[0, 0, 0].item() == pytest.approx(4.0)
    assert state.inventory.wood.tolist() == [0, 0]
    assert state.map[0, 0, 10, 11].item() == int(BlockType.TREE)


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


def test_not_interacting_leaves_the_world_alone() -> None:
    state = interact.interact(
        _facing(_state(), BlockType.TREE),
        doing=torch.tensor([True, False]),
        generator=_quiet(),
    )
    assert state.inventory.wood.tolist() == [1, 0]
    assert state.map[1, 0, 10, 11].item() == int(BlockType.TREE)


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
