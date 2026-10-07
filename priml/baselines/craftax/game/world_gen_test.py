"""Tests for procedural world generation.

Generation is random, so these assert the invariants a playable world must
have rather than particular tiles: the player can stand where they spawn, the
floors connect, and each floor uses its own materials.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import hashlib

import pytest
import torch

from priml.baselines.craftax.conftest import generated_world
from priml.baselines.craftax.game import constants, world_config, world_gen
from priml.baselines.craftax.game.constants import BlockType, ItemType
from priml.baselines.craftax.game.world_gen import (
    _brighten_around,
    _carve_corridor,
    _distance_from,
    _place_room_features,
    _sample_tile,
    daylight,
    generate_dungeon,
    generate_smooth_world,
    generate_world,
)
from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import EnvState
    from priml.baselines.craftax.game.world_config import SmoothWorldConfig


def _world(num_envs: int = 2, seed: int = 0) -> EnvState:
    return generated_world(num_envs=num_envs, seed=seed)


def _smooth_blocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    water_noise: float,
    mountain_noise: float,
    ridge_noise: float,
    tree_noise: float,
    config: SmoothWorldConfig = world_config.OVERWORLD,
    random_values: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    distance: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    noises = iter((water_noise, mountain_noise, ridge_noise, tree_noise))
    random_draws = iter(random_values)

    def fixed_noise(
        *,
        num_envs: int,
        shape: tuple[int, int],
        resolution: tuple[int, int],
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: int = 2,
        generator: torch.Generator | None = None,
        device: torch.device,
    ) -> torch.Tensor:
        del resolution, octaves, persistence, lacunarity, generator
        return torch.full((num_envs, *shape), next(noises), device=device, dtype=dtype)

    def fixed_distance(
        position: torch.Tensor,
        shape: tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        del position
        return torch.full(shape, distance, device=device, dtype=dtype)

    def fixed_rand(
        size: int | tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        del generator
        shape = (size,) if isinstance(size, int) else size
        return torch.full(shape, next(random_draws), device=device)

    monkeypatch.setattr(world_gen, "fractal_noise", fixed_noise)
    monkeypatch.setattr(world_gen, "_distance_from", fixed_distance)
    monkeypatch.setattr(torch, "rand", fixed_rand)
    blocks, _, _, _, _ = generate_smooth_world(
        num_envs=2,
        config=config,
        player_position=torch.tensor([24, 24]),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    return blocks


def test_corridors_include_both_corners_and_replace_only_walls() -> None:
    wall, path = int(BlockType.WALL), int(BlockType.PATH)
    blocks = torch.full((2, 3, 4), wall, dtype=torch.int32)
    blocks[0, 1, 0] = path
    blocks[0, 0, 2] = int(BlockType.CHEST)
    blocks[1, 2, 1] = int(BlockType.WATER)

    result = _carve_corridor(
        blocks,
        source=torch.tensor([[0, 1], [2, 3]]),
        sink=torch.tensor([[2, 3], [0, 1]]),
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [
                [wall, path, int(BlockType.CHEST), path],
                [path, wall, wall, path],
                [wall, wall, wall, path],
            ],
            [
                [wall, path, wall, wall],
                [wall, path, wall, wall],
                [wall, int(BlockType.WATER), path, path],
            ],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(result, expected)


def test_corridor_coordinate_ranges_use_the_requested_device() -> None:
    device = torch.device("cpu")
    blocks = torch.full((2, 3, 4), int(BlockType.WALL), dtype=torch.int32)
    with patch("torch.arange", wraps=torch.arange) as arange:
        _carve_corridor(
            blocks,
            source=torch.tensor([[0, 1], [2, 3]]),
            sink=torch.tensor([[2, 3], [0, 1]]),
            device=device,
        )
    assert len(arange.call_args_list) == 2
    assert all(call.kwargs["device"] == device for call in arange.call_args_list)


def test_world_factories_use_the_requested_device() -> None:
    device = torch.device("cpu")
    with (
        patch("torch.zeros", wraps=torch.zeros) as zeros,
        patch("torch.randperm", wraps=torch.randperm) as randperm,
        patch(
            "priml.baselines.craftax.game.world_gen.constants.on_device",
            wraps=constants.on_device,
        ) as on_device,
    ):
        generate_world(
            num_envs=2,
            generator=torch.Generator().manual_seed(23),
            device=device,
        )
    calls = [*zeros.call_args_list, *randperm.call_args_list]
    assert calls
    assert all(call.kwargs["device"] == device for call in calls)
    assert on_device.call_args_list
    assert all(call.args[1] == device for call in on_device.call_args_list)
    assert zeros.call_args_list[-1].kwargs["dtype"] == torch.int32


def test_distance_from_uses_both_axes_and_the_requested_device() -> None:
    device = torch.device("cpu")
    with patch("torch.arange", wraps=torch.arange) as arange:
        actual = _distance_from(
            torch.tensor([1, 3]),
            (3, 5),
            device=device,
        )
    expected = torch.tensor(
        [
            [10**0.5, 5**0.5, 2**0.5, 1.0, 2**0.5],
            [3.0, 2.0, 1.0, 0.0, 1.0],
            [10**0.5, 5**0.5, 2**0.5, 1.0, 2**0.5],
        ],
    )
    assert torch.equal(actual, expected)
    assert len(arange.call_args_list) == 2
    assert all(call.kwargs["device"] == device for call in arange.call_args_list)


def test_world_has_every_floor_at_the_declared_size() -> None:
    state = _world()
    rows, columns = constants.MAP_SIZE
    assert state.map.shape == (2, constants.NUM_LEVELS, rows, columns)
    assert state.timestep.tolist() == [0, 0]


def test_seeded_world_generation_matches_its_exact_state() -> None:
    state = generate_world(
        num_envs=2,
        generator=torch.Generator().manual_seed(23),
        device=torch.device("cpu"),
    )
    digest = hashlib.sha256()
    for name, value in state.state_dict().items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    assert (
        digest.hexdigest()
        == "b0a9b9899293b6e4bad2df926a32c8a7719c14b4c32f48f2c074e448f5ae6d31"
    )
    block_types, counts = state.map.unique(return_counts=True)
    assert dict(
        zip(
            (BlockType(int(block_type)).name for block_type in block_types),
            (int(count) for count in counts),
            strict=True,
        ),
    ) == {
        "GRASS": 2403,
        "WATER": 3121,
        "STONE": 6622,
        "TREE": 280,
        "PATH": 8215,
        "COAL": 244,
        "IRON": 89,
        "DIAMOND": 33,
        "SAND": 362,
        "LAVA": 1942,
        "WALL": 5655,
        "DARKNESS": 7930,
        "WALL_MOSS": 593,
        "STALAGMITE": 447,
        "SAPPHIRE": 58,
        "RUBY": 44,
        "CHEST": 48,
        "FOUNTAIN": 14,
        "FIRE_GRASS": 1750,
        "ICE_GRASS": 881,
        "FIRE_TREE": 239,
        "ICE_SHRUB": 443,
        "ENCHANTMENT_TABLE_FIRE": 2,
        "ENCHANTMENT_TABLE_ICE": 2,
        "NECROMANCER": 2,
        "GRAVE": 14,
        "GRAVE2": 20,
        "GRAVE3": 19,
    }
    assert state.player_position.tolist() == [[24, 24], [24, 24]]


def test_seeded_smooth_world_matches_its_exact_output() -> None:
    output = generate_smooth_world(
        num_envs=2,
        config=world_config.OVERWORLD,
        player_position=torch.tensor([24, 24]),
        generator=torch.Generator().manual_seed(29),
        device=torch.device("cpu"),
    )
    digest = hashlib.sha256()
    for value in output:
        digest.update(value.numpy().tobytes())
    assert (
        digest.hexdigest()
        == "6cd3c5ada7df1965aef8a72d822d862bf6c90c0d045c34f0700279de3743a84a"
    )


def test_smooth_noise_resolutions_follow_map_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(constants, "MAP_SIZE", (32, 48))
    resolutions: list[tuple[int, int]] = []

    def fixed_noise(
        *,
        num_envs: int,
        shape: tuple[int, int],
        resolution: tuple[int, int],
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: int = 2,
        generator: torch.Generator | None = None,
        device: torch.device,
    ) -> torch.Tensor:
        del octaves, persistence, lacunarity, generator
        resolutions.append(resolution)
        if len(resolutions) == 3:
            raise RuntimeError("ridge resolution recorded")
        return torch.zeros((num_envs, *shape), device=device)

    monkeypatch.setattr(world_gen, "fractal_noise", fixed_noise)
    with pytest.raises(RuntimeError, match="ridge resolution recorded"):
        generate_smooth_world(
            num_envs=2,
            config=world_config.OVERWORLD,
            player_position=torch.tensor([16, 24]),
            generator=torch.Generator().manual_seed(0),
            device=torch.device("cpu"),
        )
    assert resolutions == [(2, 3), (2, 3), (4, 24)]


def test_smooth_water_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        world_config.OVERWORLD,
        water_threshold=0.5,
        sand_threshold=0.2,
    )
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=1.5,
        mountain_noise=0.0,
        ridge_noise=0.0,
        tree_noise=0.0,
        config=config,
    )
    assert blocks[0, 1, 1] == int(BlockType.SAND)


def test_smooth_sand_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        world_config.OVERWORLD,
        water_threshold=0.9,
        sand_threshold=0.5,
    )
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=1.5,
        mountain_noise=0.0,
        ridge_noise=0.0,
        tree_noise=0.0,
        config=config,
    )
    assert blocks[0, 1, 1] == int(BlockType.GRASS)


def test_mountain_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        world_config.OVERWORLD,
        player_proximity_map_water_strength=1.0,
        player_proximity_map_water_max=2.0,
        player_proximity_map_mountain_strength=1.0,
        player_proximity_map_mountain_max=2.0,
    )
    distance = 0.9999999999999999
    mountain = (
        torch.tensor(0.65, dtype=torch.float64)
        + 0.05
        + torch.tensor(distance, dtype=torch.float64)
        - 1.0
    )
    assert mountain.item() == 0.7
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.0,
        mountain_noise=0.65,
        ridge_noise=0.0,
        tree_noise=0.0,
        config=config,
        distance=distance,
        dtype=torch.float64,
    )
    assert blocks[0, 1, 1] == int(BlockType.GRASS)


def test_ridge_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.0,
        mountain_noise=1.9,
        ridge_noise=0.8,
        tree_noise=0.0,
        random_values=(0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.STONE)


def test_inner_mountain_water_cutoff_is_strict_at_float_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    water_level = torch.tensor(1.4)
    water = water_level - 1.0
    assert water < torch.tensor(0.4)
    config = replace(
        world_config.OVERWORLD,
        water_threshold=10.0,
        sand_threshold=10.0,
    )
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=1.4,
        mountain_noise=1.9,
        ridge_noise=0.0,
        tree_noise=0.0,
        config=config,
        random_values=(0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.STONE)


def test_inner_mountain_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        world_config.OVERWORLD,
        player_proximity_map_water_strength=1.0,
        player_proximity_map_water_max=2.0,
        player_proximity_map_mountain_strength=1.0,
        player_proximity_map_mountain_max=2.0,
        water_threshold=10.0,
        sand_threshold=10.0,
    )
    mountain = torch.tensor(0.8) + 0.05 + torch.tensor(1.0) - 1.0
    assert mountain == torch.tensor(0.85)
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.5,
        mountain_noise=0.8,
        ridge_noise=0.0,
        tree_noise=0.0,
        config=config,
        distance=1.0,
        random_values=(0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.STONE)


def test_lava_mountain_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        world_config.OVERWORLD,
        player_proximity_map_water_strength=1.0,
        player_proximity_map_water_max=2.0,
        player_proximity_map_mountain_strength=1.0,
        player_proximity_map_mountain_max=2.0,
        water_threshold=10.0,
        sand_threshold=10.0,
    )
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.0,
        mountain_noise=0.8,
        ridge_noise=0.0,
        tree_noise=0.8,
        config=config,
        distance=1.0,
        random_values=(0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.STONE)


def test_tree_noise_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.0,
        mountain_noise=0.0,
        ridge_noise=0.0,
        tree_noise=0.5,
        random_values=(0.9, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.GRASS)


def test_tree_uniform_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.0,
        mountain_noise=0.0,
        ridge_noise=0.0,
        tree_noise=0.6,
        random_values=(0.8, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.GRASS)


def test_ore_probability_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(world_config.OVERWORLD, ore_chances=(0.5, 0.0, 0.0, 0.0, 0.0))
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.0,
        mountain_noise=1.9,
        ridge_noise=0.0,
        tree_noise=0.0,
        config=config,
        random_values=(0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.STONE)


def test_lava_tree_threshold_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _smooth_blocks(
        monkeypatch,
        water_noise=0.0,
        mountain_noise=1.9,
        ridge_noise=0.0,
        tree_noise=0.7,
        random_values=(0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert blocks[0, 1, 1] == int(BlockType.STONE)


def test_smooth_world_factories_use_the_requested_device() -> None:
    device = torch.device("cpu")
    with (
        patch("torch.arange", wraps=torch.arange) as arange,
        patch("torch.rand", wraps=torch.rand) as rand,
        patch("torch.full", wraps=torch.full) as full,
        patch("torch.zeros", wraps=torch.zeros) as zeros,
        patch(
            "priml.baselines.craftax.game.world_gen.constants.on_device",
            wraps=constants.on_device,
        ) as on_device,
        patch(
            "priml.baselines.craftax.game.world_gen._sample_tile",
            wraps=_sample_tile,
        ) as sample_tile,
    ):
        generate_smooth_world(
            num_envs=2,
            config=world_config.GNOMISH_MINES,
            player_position=torch.tensor([24, 24]),
            generator=torch.Generator().manual_seed(29),
            device=device,
        )
    calls = [
        *arange.call_args_list,
        *rand.call_args_list,
        *full.call_args_list,
        *zeros.call_args_list,
    ]
    assert calls
    assert all(call.kwargs["device"] == device for call in calls)
    assert on_device.call_args_list
    assert all(call.args[1] == device for call in on_device.call_args_list)
    assert sample_tile.call_args_list
    assert all(call.kwargs["device"] == device for call in sample_tile.call_args_list)
    assert zeros.call_args_list[-1].kwargs["dtype"] == torch.int32


def test_seeded_dungeon_matches_its_exact_output() -> None:
    output = generate_dungeon(
        num_envs=2,
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(31),
        device=torch.device("cpu"),
    )
    digest = hashlib.sha256()
    for value in output:
        digest.update(value.numpy().tobytes())
    assert (
        digest.hexdigest()
        == "97a38512ed44e458e1a9141a08b4fc2c556c7a58c8f4e773c0ef03ac2d4f9221"
    )


def test_seeded_dungeon_matches_a_non_square_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(constants, "MAP_SIZE", (48, 64))
    output = generate_dungeon(
        num_envs=2,
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(47),
        device=torch.device("cpu"),
    )
    digest = hashlib.sha256()
    for value in output:
        digest.update(value.numpy().tobytes())
    assert (
        digest.hexdigest()
        == "ea3a8e12d07a6a28fe7965ab4f5cbf7088ea60549a94059cdf438e6936899118"
    )


def test_dungeon_speckles_exclude_random_values_at_the_cutoff() -> None:
    original_rand = torch.rand

    def fixed_speckles(
        size: int | tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if isinstance(size, tuple) and len(size) == 3:
            return torch.full(size, 0.1, device=device)
        return original_rand(size, generator=generator, device=device)

    with patch(
        "torch.rand",
        side_effect=fixed_speckles,
    ):
        blocks, _, _, _, _ = generate_dungeon(
            num_envs=2,
            config=world_config.DUNGEON,
            generator=torch.Generator().manual_seed(31),
            device=torch.device("cpu"),
        )
    assert not (blocks == int(BlockType.WALL_MOSS)).any()


def test_dungeon_factories_use_the_requested_device_and_dtypes() -> None:
    device = torch.device("cpu")
    with (
        patch("torch.arange", wraps=torch.arange) as arange,
        patch("torch.rand", wraps=torch.rand) as rand,
        patch("torch.randint", wraps=torch.randint) as randint,
        patch("torch.multinomial", wraps=torch.multinomial) as multinomial,
        patch("torch.full", wraps=torch.full) as full,
        patch("torch.zeros", wraps=torch.zeros) as zeros,
        patch("torch.ones", wraps=torch.ones) as ones,
        patch("torch.empty", wraps=torch.empty) as empty,
        patch(
            "priml.baselines.craftax.game.world_gen.constants.on_device",
            wraps=constants.on_device,
        ) as on_device,
        patch(
            "priml.baselines.craftax.game.world_gen._sample_tile",
            wraps=_sample_tile,
        ) as sample_tile,
    ):
        generate_dungeon(
            num_envs=2,
            config=world_config.DUNGEON,
            generator=torch.Generator().manual_seed(31),
            device=device,
        )
    calls = [
        *arange.call_args_list,
        *rand.call_args_list,
        *randint.call_args_list,
        *full.call_args_list,
        *zeros.call_args_list,
        *ones.call_args_list,
        *empty.call_args_list,
    ]
    assert calls
    assert all(call.kwargs["device"] == device for call in calls)
    assert on_device.call_args_list
    assert all(call.args[1] == device for call in on_device.call_args_list)
    assert sample_tile.call_args_list
    assert all(call.kwargs["device"] == device for call in sample_tile.call_args_list)
    assert zeros.call_args_list[0].kwargs["dtype"] == torch.int32
    assert full.call_args_list[0].kwargs["dtype"] == torch.int32
    assert empty.call_args is not None
    assert empty.call_args.kwargs["dtype"] == torch.int64
    assert multinomial.call_args_list
    assert all(call.args[1] == 1 for call in multinomial.call_args_list)


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


def test_empty_projectile_directions_match_reference_defaults() -> None:
    state = _world()
    assert (state.mob_projectile_directions == 1).all()
    assert (state.player_projectile_directions == 1).all()


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
        assert sorted(from_plain(row.tolist(), list[int])) == list(range(6))
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
        device=torch.device("cpu"),
    )
    present = set(from_plain(blocks.flatten().tolist(), list[int]))
    assert int(BlockType.GRASS) in present
    assert int(BlockType.STONE) in present
    assert int(BlockType.TREE) in present
    # The surface is fully lit; only the caves need torches.
    assert float(light.min()) > 0.0


def test_empty_tile_weights_select_the_origin() -> None:
    position = _sample_tile(
        torch.zeros((2, 4)),
        (2, 2),
        generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
    )
    assert position.tolist() == [[0, 0], [0, 0]]


def test_sample_tile_keeps_empty_environments_independent() -> None:
    weights = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
    )
    position = _sample_tile(
        weights,
        (2, 3),
        generator=torch.Generator().manual_seed(59),
        device=torch.device("cpu"),
    )
    assert position.tolist() == [[0, 0], [1, 1]]


def test_sample_tile_moves_its_result_to_the_requested_device() -> None:
    device = torch.device("meta")
    position = _sample_tile(
        torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        (2, 2),
        generator=torch.Generator().manual_seed(53),
        device=device,
    )
    assert position.device == device
    assert position.dtype == torch.int32
    assert position.shape == (2, 2)


def test_graveyard_sampled_stone_and_ladder_light_quirks() -> None:
    blocks, items, light, _, _ = generate_smooth_world(
        num_envs=1,
        config=world_config.GRAVEYARD,
        player_position=torch.tensor([24, 24]),
        generator=torch.Generator().manual_seed(9),
        device=torch.device("cpu"),
    )
    # Upstream writes STONE with always_diamond=False (world_gen.py:483-504).
    # The candidate is still lit without an ascent (world_gen.py:549-555).
    actual = (
        int((blocks == int(BlockType.STONE)).sum()),
        bool((light > 0).any()),
        bool((items == int(ItemType.LADDER_UP)).any()),
    )
    assert actual == (1, True, False)


def test_water_and_mountains_keep_clear_of_the_spawn() -> None:
    # A player walled in or dropped in the sea cannot play, so generation
    # suppresses both near the center.
    blocks, _, _, _, _ = generate_smooth_world(
        num_envs=8,
        config=world_config.OVERWORLD,
        player_position=torch.tensor([24, 24]),
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    around = blocks[:, 23:26, 23:26]
    assert not (around == int(BlockType.WATER)).all()


def test_a_dungeon_is_rooms_joined_by_corridors() -> None:
    blocks, items, _, _, _ = generate_dungeon(
        num_envs=4,
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    present = set(from_plain(blocks.flatten().tolist(), list[int]))
    assert int(BlockType.PATH) in present
    assert int(BlockType.WALL) in present
    assert int(BlockType.CHEST) in present
    # Torches mark the room corners, which is what makes a dungeon navigable.
    assert (items == int(ItemType.TORCH)).any()


def test_room_features_match_a_seeded_hand_built_room() -> None:
    # _place_room_features tests one room with the production batch axis.
    blocks = torch.full((1, 5, 7), int(BlockType.PATH), dtype=torch.int32)
    result = _place_room_features(
        blocks,
        corner=torch.tensor([[1, 1]]),
        size=torch.tensor([[3, 5]]),
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(37),
        device=torch.device("cpu"),
    )
    expected = blocks.clone()
    expected[0, 2, 3] = int(BlockType.CHEST)
    assert torch.equal(result, expected)


def test_room_feature_factories_use_the_requested_device() -> None:
    # _place_room_features tests one room with the production batch axis.
    blocks = torch.full((1, 5, 7), int(BlockType.PATH), dtype=torch.int32)
    device = torch.device("cpu")
    with (
        patch("torch.rand", wraps=torch.rand) as rand,
        patch(
            "torch.full",
            wraps=torch.full,
        ) as full,
    ):
        _place_room_features(
            blocks,
            corner=torch.tensor([[1, 1]]),
            size=torch.tensor([[3, 5]]),
            config=world_config.DUNGEON,
            generator=torch.Generator().manual_seed(37),
            device=device,
        )
    assert len(rand.call_args_list) == 3
    assert all(call.kwargs["device"] == device for call in rand.call_args_list)
    assert len(full.call_args_list) == 2
    assert all(call.kwargs["device"] == device for call in full.call_args_list)


def test_room_features_excludes_a_fountain_at_the_exact_cutoff() -> None:
    # _place_room_features tests one room with the production batch axis.
    blocks = torch.full((1, 5, 7), int(BlockType.PATH), dtype=torch.int32)
    original_rand = torch.rand

    def fixed_fountain_cutoff(
        size: int | tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if isinstance(size, int) and size == 1:
            return torch.full((size,), 0.5, device=device)
        return original_rand(size, generator=generator, device=device)

    with patch(
        "torch.rand",
        side_effect=fixed_fountain_cutoff,
    ):
        result = _place_room_features(
            blocks,
            corner=torch.tensor([[1, 1]]),
            size=torch.tensor([[3, 5]]),
            config=world_config.DUNGEON,
            generator=torch.Generator().manual_seed(37),
            device=torch.device("cpu"),
        )
    expected = blocks.clone()
    expected[0, 2, 3] = int(BlockType.CHEST)
    assert torch.equal(result, expected)


def test_brighten_around_matches_two_hand_built_positions() -> None:
    light = torch.zeros((2, 13, 15))  # _brighten_around operates on batched maps.
    positions = torch.tensor([[4, 6], [10, 12]])
    expected = light.clone()
    expected[0, :9, 2:11] = constants.TORCH_LIGHT_MAP
    expected[1, 4:, 6:] = constants.TORCH_LIGHT_MAP
    result = _brighten_around(light, positions, ambient=0.0)
    assert torch.equal(result, expected)


def test_brighten_around_factories_use_the_light_device() -> None:
    light = torch.zeros((2, 13, 15))  # _brighten_around operates on batched maps.
    device = torch.device("cpu")
    with (
        patch("torch.arange", wraps=torch.arange) as arange,
        patch(
            "priml.baselines.craftax.game.world_gen.constants.on_device",
            wraps=constants.on_device,
        ) as on_device,
    ):
        _brighten_around(light, torch.tensor([[4, 6], [10, 12]]), ambient=0.0)
    assert len(arange.call_args_list) == 2
    assert all(call.kwargs["device"] == device for call in arange.call_args_list)
    assert on_device.call_args is not None
    assert on_device.call_args.args[1] == device


def test_ladder_light_matches_negative_update_indices_on_a_rectangular_map() -> None:
    light = torch.zeros((1, 13, 15))  # _brighten_around tests one batched map.
    actual = _brighten_around(light, torch.tensor([[0, 0]]), ambient=0.0)
    expected = torch.zeros_like(light)
    expected[0, -9:, -9:] = constants.TORCH_LIGHT_MAP
    assert torch.equal(actual, expected)


def test_brighten_around_preserves_wrapping_on_unequal_axes() -> None:
    rows, columns = 64, 48
    positions = [(0, 0), (3, 4), (4, 3), (24, 24), (63, 47)]
    light = torch.zeros((len(positions), rows, columns))
    expected = light.clone()
    for env, (row, column) in enumerate(positions):
        row_start = row - 4
        row_start = row_start + rows if row_start < 0 else row_start
        row_start = min(row_start, rows - 9)
        column_start = column - 4
        column_start = column_start + columns if column_start < 0 else column_start
        column_start = min(column_start, columns - 9)
        expected[
            env,
            row_start : row_start + 9,
            column_start : column_start + 9,
        ] = constants.TORCH_LIGHT_MAP

    assert torch.equal(
        _brighten_around(light, torch.tensor(positions), ambient=0.0),
        expected,
    )


def test_fixed_dungeon_seed_pins_room_layout() -> None:
    _, items, _, _, _ = generate_dungeon(
        num_envs=1,
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(431),
        device=torch.device("cpu"),
    )
    corners = torch.nonzero(items[0] == int(ItemType.TORCH), as_tuple=False)
    assert corners[:4].tolist() == [[0, 4], [0, 10], [7, 4], [7, 10]]


def test_dungeon_walls_out_of_sight_read_as_darkness() -> None:
    blocks, _, _, _, _ = generate_dungeon(
        num_envs=2,
        config=world_config.DUNGEON,
        generator=torch.Generator().manual_seed(2),
        device=torch.device("cpu"),
    )
    assert (blocks == int(BlockType.DARKNESS)).any()


def test_the_sewers_use_their_own_materials() -> None:
    blocks, _, _, _, _ = generate_dungeon(
        num_envs=4,
        config=world_config.SEWERS,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    present = set(from_plain(blocks.flatten().tolist(), list[int]))
    assert int(BlockType.ENCHANTMENT_TABLE_ICE) in present
    assert int(BlockType.WATER) in present


def test_daylight_starts_one_third_into_the_cycle() -> None:
    steps = torch.tensor([17, 149, 299])
    phase = (steps.float() * (1 / constants.DAY_LENGTH)) % 1.0 + 0.3
    expected = 1.0 - (torch.pi * phase).cos().abs() ** 3
    assert torch.allclose(daylight(steps), expected, rtol=0.0, atol=1e-7)


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


def test_daylight_pins_large_timestep_values() -> None:
    steps = torch.tensor([301, 599, 600, 601, 899, 1201, 10_000])
    assert daylight(steps).view(torch.int32).tolist() == [
        1_062_091_936,
        1_061_797_309,
        1_061_946_187,
        1_062_091_936,
        1_061_797_309,
        1_062_091_947,
        1_064_224_244,
    ]


@pytest.mark.gpu_torch_cuda
def test_daylight_matches_compiled_reference_exactly() -> None:
    assert torch.cuda.is_available()
    # Captured from upstream calculate_light_level on an RTX 5090, compiled as
    # Craftax's training compiles it: jit over vmap with EnvParams closed over.
    expected = torch.tensor(
        [
            1_061_946_187,
            1_062_091_936,
            1_063_970_034,
            1_056_379_794,
            1_061_797_309,
            1_061_946_187,
            1_064_988_890,
            1_051_553_488,
            1_064_301_888,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    steps = torch.tensor((0, 1, 17, 149, 299, 300, 33, 1061, 99_999), device="cuda")
    assert torch.equal(daylight(steps).view(torch.int32), expected)


@pytest.mark.parametrize(
    "position",
    [(0, 0), (0, 47), (47, 0), (47, 47), (3, 44), (4, 4), (24, 24), (43, 43)],
)
def test_ladder_light_writes_one_torch_patch_wherever_the_ladder_is(
    position: tuple[int, int],
) -> None:
    # The patch starts four tiles up and left of the ladder; a start off the
    # top or left edge wraps, and every start is pulled back so all 81 tiles
    # stay on the map.
    rows, columns = constants.MAP_SIZE
    light = torch.rand((1, rows, columns), generator=torch.Generator().manual_seed(0))
    starts: list[int] = []
    for coordinate, extent in zip(position, (rows, columns), strict=True):
        start = coordinate - 4
        start = start + extent if start < 0 else start
        starts.append(min(start, extent - 9))
    expected = light.clone()
    expected[0, starts[0] : starts[0] + 9, starts[1] : starts[1] + 9] = (
        constants.TORCH_LIGHT_MAP * (1 - 0.25) + 0.25
    )
    actual = _brighten_around(light, torch.tensor([position]), ambient=0.25)
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
