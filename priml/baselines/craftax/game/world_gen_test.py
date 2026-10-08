"""Tests for world generation: the noise, the floor recipes, and the worlds they build.

The noise is checked against a scalar float32 transcription of the C, and the
recipes against ``constants.h``'s tables. Generation is random, so most world
tests assert what every world must satisfy -- the player stands on the spawn
block, the ladders sit on items, the potion shuffle is a permutation, the
spawn bitsets describe the map -- and the bit-exact checks of whole worlds are
the parity tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from priml.baselines.craftax.game.jit import cosf, jit, sinf
from priml.baselines.craftax.game.rng import F32_PER_DRAW
from priml.baselines.craftax.game.rules import daylight_numba
from priml.baselines.craftax.game.state import (
    BOSS_SPAWN_TURNS,
    NOISE_PI2,
    NOISE_SQRT2,
    STATE_DTYPE,
    Action,
    BlockType,
    ItemType,
    env_state,
    new_states,
)
from priml.baselines.craftax.game.testing import (
    FMA_MNEMONIC,
    kernel_inspection,
)
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_CONFIG_DTYPE,
    DUNGEON_FLOOR_ORDER,
    DUNGEON_LEVEL_CONFIGS,
    OCTAVES,
    SMOOTH_CONFIG_DTYPE,
    SMOOTH_FLOOR_ORDER,
    SMOOTH_LEVEL_CONFIGS,
    build_pool_numba,
    generate_fractal_numba,
    generate_world_numba,
)
from priml.baselines.craftax.lib.arrays import typed
from priml.baselines.craftax.scripts import mint_goldens


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.craftax.game.world_gen import SmoothGenConfig


_LAND = (BlockType.GRASS, BlockType.PATH, BlockType.FIRE_GRASS, BlockType.ICE_GRASS)
_GRAVES = (BlockType.GRAVE, BlockType.GRAVE2, BlockType.GRAVE3)


@jit
def _cos_sin(angle: np.float32) -> tuple[np.float32, np.float32]:
    # The kernel computes both of an angle, which LLVM merges into one sincos
    # call; the transcription must take its trig from that same call, since
    # Darwin's sincos and its separate cosf differ by an ulp on rare inputs.
    # The trig golden in jit_test.py is what pins the libm itself.
    return np.float32(cosf(angle)), np.float32(sinf(angle))


# Returns the field and the rng state after the octave's one draw.
def _reference_fractal(
    seed: int,
    rows: int,
    cols: int,
    res_rows: int,
    res_cols: int,
) -> tuple[NDArray[np.float32], int]:
    """One octave of ``generate_fractal_numba`` in float32 scalars and the platform libm."""
    draws = mint_goldens.glibc_rand_r(np.array([seed], np.uint32), 1)
    angle_seed = draws.item(0, 0)
    # The octave's one rand_r is three LCG steps of the stream.
    next_state = seed
    for _ in range(3):
        next_state = (next_state * 1_103_515_245 + 12_345) & 0xFFFF_FFFF

    def angle_at(index: int) -> np.float32:
        local = (angle_seed ^ ((index * 747_796_405) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
        local = (local + 2_891_336_453) & 0xFFFF_FFFF
        draw = mint_goldens.glibc_rand_r(np.array([local], np.uint32), 1).item(0, 0)
        return np.float32(NOISE_PI2 * (np.float32(draw) * F32_PER_DRAW))

    one = np.float32(1.0)
    out = np.zeros((rows, cols), dtype=np.float32)
    cell_rows = rows // res_rows
    cell_cols = cols // res_cols
    width = res_cols + 1
    for row in range(rows):
        grad_row = row // cell_rows
        local_row = np.float32(
            np.float32(row - grad_row * cell_rows) / np.float32(cell_rows),
        )
        interp_row = np.float32(
            local_row * local_row * local_row
            * (local_row * (local_row * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0)),
        )  # fmt: skip
        for col in range(cols):
            grad_col = col // cell_cols
            local_col = np.float32(
                np.float32(col - grad_col * cell_cols) / np.float32(cell_cols),
            )
            interp_col = np.float32(
                local_col * local_col * local_col
                * (local_col * (local_col * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0)),
            )  # fmt: skip
            gx: dict[tuple[int, int], np.float32] = {}
            gy: dict[tuple[int, int], np.float32] = {}
            for dr in range(2):
                for dc in range(2):
                    angle = angle_at((grad_row + dr) * width + (grad_col + dc))
                    cos, sin = _cos_sin(angle)
                    gx[dr, dc] = np.float32(cos)
                    gy[dr, dc] = np.float32(sin)
            n00 = np.float32(local_row * gx[0, 0] + local_col * gy[0, 0])
            n10 = np.float32((local_row - one) * gx[1, 0] + local_col * gy[1, 0])
            n01 = np.float32(local_row * gx[0, 1] + (local_col - one) * gy[0, 1])
            n11 = np.float32(
                (local_row - one) * gx[1, 1] + (local_col - one) * gy[1, 1],
            )
            n0 = np.float32(n00 * (one - interp_row) + interp_row * n10)
            n1 = np.float32(n01 * (one - interp_row) + interp_row * n11)
            out[row, col] = np.float32(
                np.float32(out.item(row, col))
                + one * NOISE_SQRT2 * ((one - interp_col) * n0 + interp_col * n1),
            )
    low = out.min()
    high = out.max()
    scale = np.float32(high - low)
    return ((out - low) / scale).astype(np.float32), next_state


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(("res_rows", "res_cols"), [(3, 3), (6, 24), (12, 12)])
def test_one_octave_matches_the_scalar_transcription(
    res_rows: int,
    res_cols: int,
) -> None:
    rng = np.array([2024], dtype=np.uint32)
    # ``generate_fractal_numba`` fills one floor's field; Craftax's floors are 48 x 48.
    out = np.empty((48, 48), dtype=np.float32)
    generate_fractal_numba(rng, res_rows, res_cols, out)
    expected, next_state = _reference_fractal(2024, 48, 48, res_rows, res_cols)
    assert np.array_equal(out.view(np.uint32), expected.view(np.uint32))
    assert rng.item(0) == next_state


@pytest.mark.compute_large_fixture
def test_the_field_spans_the_unit_interval_and_depends_on_the_seed() -> None:
    # ``generate_fractal_numba`` fills one floor's field; Craftax's floors are 48 x 48.
    first = np.empty((48, 48), dtype=np.float32)
    second = np.empty((48, 48), dtype=np.float32)
    generate_fractal_numba(np.array([1], np.uint32), 3, 3, first)
    generate_fractal_numba(np.array([2], np.uint32), 3, 3, second)
    assert first.min() == 0.0
    assert first.max() == 1.0
    assert not np.array_equal(first, second)


@pytest.mark.compute_large_fixture
def test_the_octaves_consume_exactly_one_draw_each() -> None:
    # ``generate_fractal_numba`` fills one floor's field; Craftax's floors are 48 x 48.
    out = np.empty((48, 48), dtype=np.float32)
    rng = np.array([7], dtype=np.uint32)
    generate_fractal_numba(rng, 3, 3, out)
    state = 7
    for _ in range(3 * OCTAVES):
        state = (state * 1_103_515_245 + 12_345) & 0xFFFF_FFFF
    assert rng.item(0) == state


@pytest.mark.compute_large_fixture
def test_noise_kernel_emits_no_fma_and_no_float64() -> None:
    # ``generate_fractal_numba`` fills one floor's field; Craftax's floors are 48 x 48.
    out = np.empty((48, 48), dtype=np.float32)
    generate_fractal_numba(np.array([1], np.uint32), 3, 3, out)
    assembly, body = kernel_inspection(generate_fractal_numba)
    assert not FMA_MNEMONIC.search(assembly)
    assert "double" not in body
    assert "fpext" not in body


def test_every_floor_is_generated_exactly_once() -> None:
    floors = sorted([*SMOOTH_FLOOR_ORDER, *DUNGEON_FLOOR_ORDER])
    assert floors == list(range(9))
    assert SMOOTH_LEVEL_CONFIGS.shape == (6,)
    assert DUNGEON_LEVEL_CONFIGS.shape == (3,)
    assert SMOOTH_LEVEL_CONFIGS.dtype == SMOOTH_CONFIG_DTYPE
    assert DUNGEON_LEVEL_CONFIGS.dtype == DUNGEON_CONFIG_DTYPE


def test_overworld_row_matches_constants_h() -> None:
    overworld = _config(0)
    assert overworld.default_block == BlockType.GRASS
    assert list(overworld.ores) == [
        BlockType.COAL,
        BlockType.IRON,
        BlockType.DIAMOND,
        BlockType.OUT_OF_BOUNDS,
        BlockType.OUT_OF_BOUNDS,
    ]
    assert list(overworld.ore_chances) == [
        np.float32(0.03),
        np.float32(0.02),
        np.float32(0.001),
        0.0,
        0.0,
    ]
    assert (overworld.ladder_up, overworld.ladder_down) == (0, 1)
    assert overworld.default_light == 1.0
    assert overworld.tree_threshold_uniform == np.float32(0.8)


def test_graveyard_row_matches_constants_h() -> None:
    graveyard = _config(5)
    assert list(graveyard.ore_requirement_blocks) == [
        BlockType.WALL,
        BlockType.GRAVE,
        BlockType.GRAVE,
        BlockType.WALL,
        BlockType.WALL,
    ]
    assert list(graveyard.ores[:3]) == [
        BlockType.WALL_MOSS,
        BlockType.GRAVE2,
        BlockType.GRAVE3,
    ]
    assert list(graveyard.ore_chances[:3]) == [
        np.float32(0.1),
        np.float32(0.333),
        np.float32(0.5),
    ]
    assert graveyard.player_spawn == BlockType.NECROMANCER
    assert (graveyard.ladder_up, graveyard.ladder_down) == (0, 0)
    assert graveyard.mountain_max == 10.0
    assert graveyard.tree_threshold_uniform == np.float32(0.95)
    assert graveyard.tree_threshold_perlin == -1.0


def test_float_fields_are_float32_and_flags_are_bytes() -> None:
    for name in ("water_strength", "ore_chances", "default_light"):
        assert SMOOTH_LEVEL_CONFIGS[name].dtype == np.float32
    assert SMOOTH_LEVEL_CONFIGS["ladder_up"].dtype == np.uint8


def test_dungeon_rows_match_constants_h() -> None:
    assert DUNGEON_LEVEL_CONFIGS["special_block"].tolist() == [
        BlockType.PATH,
        BlockType.ENCHANTMENT_TABLE_ICE,
        BlockType.ENCHANTMENT_TABLE_FIRE,
    ]
    assert DUNGEON_LEVEL_CONFIGS["fountain_block"].tolist() == [
        BlockType.FOUNTAIN,
        BlockType.WATER,
        BlockType.FOUNTAIN,
    ]
    assert DUNGEON_LEVEL_CONFIGS["rare_path_replacement_block"].tolist() == [
        BlockType.PATH,
        BlockType.WATER,
        BlockType.PATH,
    ]


def _world(seed: int) -> NDArray[np.void]:
    states = new_states(1)
    generate_world_numba(
        env_state(states, 0),
        np.array([seed], dtype=np.uint32),
        SMOOTH_LEVEL_CONFIGS,
        DUNGEON_LEVEL_CONFIGS,
    )
    return states


def _config(index: int) -> SmoothGenConfig:
    return cast("SmoothGenConfig", SMOOTH_LEVEL_CONFIGS.view(np.recarray)[index])


def _bits(rows: NDArray[np.uint64]) -> NDArray[np.bool]:
    """Expand ``uint64 [levels, rows]`` bit rows into ``bool [levels, rows, 64]``."""
    shifts = np.arange(64, dtype=np.uint64)
    return ((rows[..., None] >> shifts) & np.uint64(1)).astype(bool)


@pytest.mark.compute_large_fixture
def test_the_player_starts_at_the_center_on_the_spawn_block() -> None:
    states = _world(0)
    state = env_state(states, 0)
    assert list(state.player_position) == [24, 24]
    assert state.player_level == 0
    assert state.player_direction == Action.UP
    assert state.map[0, 24, 24] == BlockType.GRASS
    assert state.map[8, 24, 24] == BlockType.NECROMANCER
    assert state.player_health == 9.0
    assert (state.player_food, state.player_drink, state.player_energy) == (
        9,
        9,
        9,
    )
    assert state.player_mana == 9
    assert list(state.monsters_killed) == [10, *[0] * 8]
    assert state.boss_timestep_to_spawn_this_round == BOSS_SPAWN_TURNS
    assert state.timestep == 0


@pytest.mark.compute_large_fixture
def test_ladders_sit_on_the_ladder_items_where_the_floor_has_them() -> None:
    states = _world(3)
    state = env_state(states, 0)
    for i, level in enumerate(SMOOTH_FLOOR_ORDER):
        config = _config(i)
        down = state.down_ladders[level]
        up = state.up_ladders[level]
        assert state.item_map[level, down[0], down[1]] == (
            ItemType.LADDER_DOWN if config.ladder_down else ItemType.NONE
        )
        assert state.item_map[level, up[0], up[1]] == (
            ItemType.LADDER_UP if config.ladder_up else ItemType.NONE
        )
    for level in DUNGEON_FLOOR_ORDER:
        down = state.down_ladders[level]
        up = state.up_ladders[level]
        assert state.item_map[level, down[0], down[1]] == ItemType.LADDER_DOWN
        assert state.item_map[level, up[0], up[1]] == ItemType.LADDER_UP
        assert state.map[level, down[0], down[1]] == BlockType.PATH


@pytest.mark.compute_large_fixture
def test_dungeons_are_fully_lit_and_walled_by_darkness() -> None:
    states = _world(11)
    for level in DUNGEON_FLOOR_ORDER:
        assert np.equal(typed(states["light_map"], np.uint8)[0, level, ...], 255).all()
        blocks = set(map(int, np.unique(typed(states["map"], np.uint8)[0, level, ...])))
        assert BlockType.DARKNESS in blocks
        assert BlockType.PATH in blocks
        assert BlockType.CHEST in blocks
        assert (
            np.count_nonzero(
                np.equal(
                    typed(states["item_map"], np.uint8)[0, level, ...],
                    ItemType.TORCH,
                ),
            )
            >= 8
        )


@pytest.mark.compute_large_fixture
def test_spawn_bitsets_describe_the_map() -> None:
    states = _world(5)
    grid = typed(states["map"], np.uint8)[0, ...]
    assert np.array_equal(
        _bits(typed(states["spawn_land"], np.uint64)[0, ...])[..., :48],
        np.isin(grid, _LAND),
    )
    assert np.array_equal(
        _bits(typed(states["spawn_grave"], np.uint64)[0, ...])[..., :48],
        np.isin(grid, _GRAVES),
    )
    assert np.array_equal(
        _bits(typed(states["spawn_water"], np.uint64)[0, ...])[..., :48],
        np.equal(grid, BlockType.WATER),
    )
    assert not _bits(typed(states["spawn_land"], np.uint64)[0, ...])[..., 48:].any()
    assert not typed(states["mob_bits"], np.uint64).any()


@pytest.mark.compute_large_fixture
def test_mob_slots_and_projectiles_start_as_the_c_initializes_them() -> None:
    states = _world(9)
    for kind in ("melee_mobs", "passive_mobs", "mob_projectiles", "player_projectiles"):
        assert np.equal(typed(states[kind]["health"], np.float32), 1.0).all(), kind
        assert not typed(states[kind]["mask"], np.uint8).any(), kind
    ranged = typed(states["ranged_mobs"]["health"], np.float32)
    assert np.equal(ranged[0, :, :2], 1.0).all()
    assert np.equal(ranged[0, :, 2], 0.0).all()
    assert np.equal(typed(states["mob_projectile_dirs"], np.int32), 1).all()
    assert np.equal(typed(states["player_projectile_directions"], np.int32), 1).all()


@pytest.mark.compute_large_fixture
def test_the_potion_shuffle_is_a_permutation_and_the_first_light_is_daylight_zero() -> (
    None
):
    states = _world(21)
    state = env_state(states, 0)
    assert sorted(state.potion_mapping) == list(range(6))
    assert list(state.learned_spells) == [0, 0]
    assert state.light_level == daylight_numba(0)


@pytest.mark.compute_large_fixture
def test_the_same_seed_reproduces_the_same_world_and_seeds_differ() -> None:
    assert _world(5).tobytes() == _world(5).tobytes()
    assert _world(5).tobytes() != _world(6).tobytes()


@pytest.mark.compute_large_fixture
def test_the_parallel_pool_equals_worlds_generated_one_at_a_time() -> None:
    pool = new_states(6)
    build_pool_numba(
        pool,
        pool.view(np.uint8).reshape(6, STATE_DTYPE.itemsize),
        SMOOTH_LEVEL_CONFIGS,
        DUNGEON_LEVEL_CONFIGS,
        100,
    )
    for k in range(6):
        assert pool[k : k + 1].tobytes() == _world(100 + k).tobytes(), k


@pytest.mark.compute_large_fixture
def test_the_pool_zeroes_a_dirty_row_first() -> None:
    pool = new_states(2)
    pool["timestep"] = 99
    pool["achievements"] = 1
    build_pool_numba(
        pool,
        pool.view(np.uint8).reshape(2, STATE_DTYPE.itemsize),
        SMOOTH_LEVEL_CONFIGS,
        DUNGEON_LEVEL_CONFIGS,
        0,
    )
    assert pool[1:2].tobytes() == _world(1).tobytes()


@pytest.mark.compute_large_fixture
def test_world_gen_kernels_emit_no_fma_and_no_float64() -> None:
    _world(1)
    assembly, body = kernel_inspection(generate_world_numba)
    assert not FMA_MNEMONIC.search(assembly)
    assert "double" not in body
    assert "fpext" not in body


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
