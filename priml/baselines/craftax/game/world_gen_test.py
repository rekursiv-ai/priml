"""Tests for procedural world generation.

Generation is random, so these assert the invariants a playable world must
have rather than particular tiles: the player can stand where they spawn, the
floors connect, and each floor uses its own materials.
"""

from __future__ import annotations

import numpy as np
import torch

from priml.baselines.craftax.conftest import reference, requires_craftax
from priml.baselines.craftax.game import constants, world_config
from priml.baselines.craftax.game.constants import BlockType, ItemType
from priml.baselines.craftax.game.world_gen import (
    daylight,
    generate_dungeon,
    generate_smooth_world,
    generate_world,
)


_DEVICE = torch.device("cpu")


def _world(num_envs: int = 2, seed: int = 0):
    return generate_world(
        num_envs=num_envs,
        generator=torch.Generator().manual_seed(seed),
        device=_DEVICE,
    )


def test_world_has_every_floor_at_the_declared_size() -> None:
    state = _world()
    rows, columns = constants.MAP_SIZE
    assert state.map.shape == (2, constants.NUM_LEVELS, rows, columns)
    assert state.timestep.tolist() == [0, 0]


def test_the_player_starts_at_the_center_on_solid_footing() -> None:
    state = _world()
    rows, columns = constants.MAP_SIZE
    assert state.player_position[0].tolist() == [rows // 2, columns // 2]
    spawn = state.map[:, 0, rows // 2, columns // 2]
    assert (spawn == int(world_config.OVERWORLD.player_spawn)).all()


def test_the_player_starts_alive_and_supplied() -> None:
    state = _world()
    assert state.player_health.tolist() == [9.0, 9.0]
    assert state.player_food.tolist() == [9, 9]
    assert state.inventory.wood.tolist() == [0, 0]
    assert not state.achievements.any()


def test_the_surface_ladder_starts_open() -> None:
    # There is nothing to kill on the surface, so requiring the usual clearing
    # count would seal the world shut on the first floor.
    state = _world()
    assert (
        state.monsters_killed[:, 0] >= constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    ).all()
    assert (state.monsters_killed[:, 1] == 0).all()


def test_potion_effects_are_shuffled_independently_per_environment() -> None:
    state = _world(num_envs=8, seed=3)
    for row in state.potion_mapping:
        assert sorted(row.tolist()) == list(range(6))
    assert len({tuple(row.tolist()) for row in state.potion_mapping}) > 1


def test_every_floor_has_the_ladders_its_recipe_declares() -> None:
    state = _world()
    for level, config in enumerate(world_config.LEVEL_CONFIGS):
        items = state.item_map[:, level]
        wants_down = getattr(config, "ladder_down", True)
        wants_up = getattr(config, "ladder_up", True)
        assert (items == int(ItemType.LADDER_DOWN)).any() == wants_down, level
        assert (items == int(ItemType.LADDER_UP)).any() == wants_up, level


def test_each_environment_gets_a_different_world() -> None:
    state = _world(num_envs=2, seed=5)
    assert not torch.equal(state.map[0], state.map[1])


def test_the_same_seed_reproduces_the_same_world() -> None:
    assert torch.equal(_world(seed=11).map, _world(seed=11).map)


def test_the_overworld_grows_the_blocks_its_recipe_names() -> None:
    blocks, _, light, _, _ = generate_smooth_world(
        num_envs=4,
        config=world_config.OVERWORLD,
        player_position=torch.tensor([24, 24]),
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )
    present = set(blocks.unique().tolist())
    assert int(BlockType.GRASS) in present
    assert int(BlockType.STONE) in present
    assert int(BlockType.TREE) in present
    # The surface is fully lit; only the caves need torches.
    assert float(light.min()) > 0.0


def test_water_and_mountains_keep_clear_of_the_spawn() -> None:
    # A player walled in or dropped in the sea cannot play, so generation
    # suppresses both near the center.
    blocks, _, _, _, _ = generate_smooth_world(
        num_envs=8,
        config=world_config.OVERWORLD,
        player_position=torch.tensor([24, 24]),
        generator=torch.Generator().manual_seed(1),
        device=_DEVICE,
    )
    around = blocks[:, 23:26, 23:26]
    assert not (around == int(BlockType.WATER)).all()


def test_a_dungeon_is_rooms_joined_by_corridors() -> None:
    blocks, items, _, _, _ = generate_dungeon(
        num_envs=4,
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )
    present = set(blocks.unique().tolist())
    assert int(BlockType.PATH) in present
    assert int(BlockType.WALL) in present
    assert int(BlockType.CHEST) in present
    # Torches mark the room corners, which is what makes a dungeon navigable.
    assert (items == int(ItemType.TORCH)).any()


def test_dungeon_walls_out_of_sight_read_as_darkness() -> None:
    blocks, _, _, _, _ = generate_dungeon(
        num_envs=2,
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(2),
        device=_DEVICE,
    )
    assert (blocks == int(BlockType.DARKNESS)).any()


def test_the_sewers_use_their_own_materials() -> None:
    blocks, _, _, _, _ = generate_dungeon(
        num_envs=4,
        config=world_config.SEWERS,
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )
    present = set(blocks.unique().tolist())
    assert int(BlockType.ENCHANTMENT_TABLE_ICE) in present
    assert int(BlockType.WATER) in present


def test_daylight_runs_from_dark_to_bright_and_back() -> None:
    steps = torch.arange(0, constants.DAY_LENGTH)
    light = daylight(steps)
    assert float(light.min()) >= 0.0
    assert float(light.max()) <= 1.0
    # A full cycle must contain both a bright noon and a dark night.
    assert float(light.max()) > 0.9
    assert float(light.min()) < 0.1


def test_daylight_repeats_every_day() -> None:
    early = daylight(torch.arange(0, 50))
    later = daylight(torch.arange(0, 50) + constants.DAY_LENGTH)
    assert torch.allclose(early, later, atol=1e-6)


@requires_craftax
def test_daylight_matches_the_reference() -> None:
    calculate_light_level = reference(
        "craftax_classic.game_logic",
    ).calculate_light_level
    params = reference("craftax_classic.envs.craftax_state").EnvParams()
    steps = torch.arange(0, 600, 7)
    expected = np.array(
        [float(calculate_light_level(int(step), params)) for step in steps],
    )
    assert np.allclose(daylight(steps).numpy(), expected, atol=1e-6)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
