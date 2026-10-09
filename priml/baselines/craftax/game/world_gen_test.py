"""Tests for world generation: the noise, the floor recipes, and the stages that build a world.

The noise is checked against a scalar float32 transcription of the C, and the
recipes against ``constants.h``'s tables. Each stage runs as its Python source
(``craftax.eager``) on a field or a floor: a whole world takes seconds
as Python, so the invariants of whole compiled worlds -- every floor's spawn
block, its ladders, the spawn bitsets, a pool equal to worlds generated one
at a time -- are checked on the worlds ``env_test``'s goldens generate, and
the bit-exact checks of whole worlds are those goldens and the parity tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from priml.baselines.craftax.eager import eager
from priml.baselines.craftax.game import world_gen
from priml.baselines.craftax.game.jit import cosf, jit, sinf
from priml.baselines.craftax.game.rng import F32_PER_DRAW, rand_r_numba
from priml.baselines.craftax.game.rules import daylight_numba
from priml.baselines.craftax.game.state import (
    BOSS_SPAWN_TURNS,
    MAP_SIZE,
    NOISE_PI2,
    NOISE_SQRT2,
    NUM_BLOCK_TYPES,
    NUM_LEVELS,
    STATE_DTYPE,
    Action,
    BlockType,
    ItemType,
    env_state,
    new_states,
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
)
from priml.baselines.craftax.lib.arrays import typed
from priml.baselines.craftax.scripts import mint_goldens


if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from priml.baselines.craftax.game.state import Array1, EnvState, Records
    from priml.baselines.craftax.game.world_gen import (
        DungeonConfig,
        SmoothGenConfig,
    )


@pytest.fixture(autouse=True)
def kernels(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run generation as Python: a compile costs seconds, one floor milliseconds."""
    with eager(monkeypatch=monkeypatch):
        yield


@jit
def _cos_sin(angle: np.float32) -> tuple[np.float32, np.float32]:
    # The kernel's trig, so the transcription's angles go through the same calls:
    # the platform libm's, as Python here (``craftax.eager``). The trig golden in
    # jit_test.py is what pins the libm itself.
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
    width = res_cols + 1
    gradients = {
        index: _cos_sin(_lattice_angle(angle_seed, index))
        for index in range((res_rows + 1) * width)
    }
    one = np.float32(1.0)
    out = np.zeros((rows, cols), dtype=np.float32)
    cell_rows = rows // res_rows
    cell_cols = cols // res_cols
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
                    cos, sin = gradients[(grad_row + dr) * width + (grad_col + dc)]
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


def _lattice_angle(angle_seed: int, index: int) -> np.float32:
    """Return the gradient angle at lattice ``index``: ``rng_f32_at`` transcribed."""
    local = (angle_seed ^ ((index * 747_796_405) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    local = (local + 2_891_336_453) & 0xFFFF_FFFF
    draw = mint_goldens.glibc_rand_r(np.array([local], np.uint32), 1).item(0, 0)
    return np.float32(NOISE_PI2 * (np.float32(draw) * F32_PER_DRAW))


# A lattice cell two rows by three columns: every interpolation weight a cell has
# that is neither 0 nor 1 appears, at a fraction of a 48 x 48 floor's cost.
@pytest.mark.parametrize(("res_rows", "res_cols"), [(3, 3), (6, 24), (12, 12)])
def test_one_octave_matches_the_scalar_transcription(
    res_rows: int,
    res_cols: int,
) -> None:
    rng = np.array([2024], dtype=np.uint32)
    rows, cols = 2 * res_rows, 3 * res_cols
    out = np.empty((rows, cols), dtype=np.float32)
    generate_fractal_numba(rng, res_rows, res_cols, out)
    expected, next_state = _reference_fractal(2024, rows, cols, res_rows, res_cols)
    assert np.array_equal(out.view(np.uint32), expected.view(np.uint32))
    assert rng.item(0) == next_state


def test_the_field_spans_the_unit_interval_and_depends_on_the_seed() -> None:
    first = np.empty((6, 9), dtype=np.float32)
    second = np.empty((6, 9), dtype=np.float32)
    generate_fractal_numba(np.array([1], np.uint32), 3, 3, first)
    generate_fractal_numba(np.array([2], np.uint32), 3, 3, second)
    assert first.min() == 0.0
    assert first.max() == 1.0
    assert not np.array_equal(first, second)


def test_the_octaves_consume_exactly_one_draw_each() -> None:
    out = np.empty((6, 9), dtype=np.float32)
    rng = np.array([7], dtype=np.uint32)
    generate_fractal_numba(rng, 3, 3, out)
    state = 7
    for _ in range(3 * OCTAVES):
        state = (state * 1_103_515_245 + 12_345) & 0xFFFF_FFFF
    assert rng.item(0) == state


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
    assert overworld.player_spawn == BlockType.GRASS
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


def test_a_dungeon_floor_is_lit_walled_by_darkness_and_laddered_on_its_paths() -> None:
    states = new_states(1)
    level = DUNGEON_FLOOR_ORDER[0]
    world_gen._generate_dungeon_level_numba(
        env_state(states, 0),
        np.array([11], dtype=np.uint32),
        level,
        _dungeon_config(0),
    )
    state = env_state(states, 0)
    assert np.equal(typed(states["light_map"], np.uint8)[0, level, ...], 255).all()
    grid = typed(states["map"], np.uint8)[0, level, ...]
    counts = np.bincount(grid.ravel(), minlength=NUM_BLOCK_TYPES)
    for block in (BlockType.DARKNESS, BlockType.PATH, BlockType.CHEST):
        assert counts[block] > 0, block
    torches = np.equal(
        typed(states["item_map"], np.uint8)[0, level, ...],
        ItemType.TORCH,
    )
    assert np.count_nonzero(torches) >= 8
    down = state.down_ladders[level]
    up = state.up_ladders[level]
    assert state.item_map[level, down[0], down[1]] == ItemType.LADDER_DOWN
    assert state.item_map[level, up[0], up[1]] == ItemType.LADDER_UP
    assert state.map[level, down[0], down[1]] == BlockType.PATH
    # Every other floor is untouched.
    others = [floor for floor in range(NUM_LEVELS) if floor != level]
    assert not typed(states["map"], np.uint8)[0, others, ...].any()


def test_the_player_creatures_and_potions_start_as_the_c_initializes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refreshed: list[tuple[int, int, int]] = []
    monkeypatch.setattr(world_gen, "refresh_spawn_cell_numba", _recorder(refreshed))
    states = new_states(1)
    states["spawn_land"] = np.uint64(0xFFFF)
    state = env_state(states, 0)
    rng = np.array([21], dtype=np.uint32)
    world_gen._init_mobs_and_player_numba(state, rng)
    assert list(state.player_position) == [MAP_SIZE // 2, MAP_SIZE // 2]
    assert state.player_level == 0
    assert state.player_direction == Action.UP
    assert state.player_health == 9.0
    meters = (state.player_food, state.player_drink, state.player_energy)
    assert meters == (9, 9, 9)
    assert state.player_mana == 9
    attributes = (
        state.player_dexterity,
        state.player_strength,
        state.player_intelligence,
    )
    assert attributes == (1, 1, 1)
    assert list(state.monsters_killed) == [10, *[0] * 8]
    assert state.boss_timestep_to_spawn_this_round == BOSS_SPAWN_TURNS
    assert state.timestep == 0
    assert state.light_level == daylight_numba(0)
    # The potion shuffle: five draws, a permutation.
    assert sorted(state.potion_mapping) == list(range(6))
    assert list(state.potion_mapping) != list(range(6))
    assert list(state.learned_spells) == [0, 0]
    drawn = np.array([21], dtype=np.uint32)
    for _ in range(5):
        rand_r_numba(drawn)
    assert rng[0] == drawn[0]
    for kind in ("melee_mobs", "passive_mobs", "mob_projectiles", "player_projectiles"):
        assert np.equal(typed(states[kind]["health"], np.float32), 1.0).all(), kind
        assert not typed(states[kind]["mask"], np.uint8).any(), kind
    ranged = typed(states["ranged_mobs"]["health"], np.float32)
    assert np.equal(ranged[0, :, :2], 1.0).all()
    assert np.equal(ranged[0, :, 2], 0.0).all()
    assert np.equal(typed(states["mob_projectile_dirs"], np.int32), 1).all()
    assert np.equal(typed(states["player_projectile_directions"], np.int32), 1).all()
    # The bitsets are cleared, then every cell of every floor refreshed, in order.
    assert not typed(states["spawn_land"], np.uint64).any()
    assert refreshed == [
        (level, row, col)
        for level in range(NUM_LEVELS)
        for row in range(MAP_SIZE)
        for col in range(MAP_SIZE)
    ]


def test_the_pool_generates_world_k_from_seed_k_into_a_zeroed_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each row is zeroed, as the C's ``calloc``, then generated from its own seed.

    The generator stands in as a record of its stream's seed and of what the
    row held, so the pool's side is checked here; generation itself is the
    stages' tests' and ``env_test``'s goldens', which also hold the compiled,
    parallel pool equal to worlds generated one at a time.
    """
    monkeypatch.setattr(world_gen, "generate_world_numba", _stamp_world)
    pool = new_states(3)
    pool["timestep"] = 99
    pool["achievements"] = 1
    build_pool_numba(
        pool,
        pool.view(np.uint8).reshape(3, STATE_DTYPE.itemsize),
        SMOOTH_LEVEL_CONFIGS,
        DUNGEON_LEVEL_CONFIGS,
        100,
    )
    assert typed(pool["timestep"], np.int32).tolist() == [100, 101, 102]
    assert not typed(pool["achievements"], np.int32).any()
    rest = new_states(1)
    for k in range(3):
        rest["timestep"] = 100 + k
        assert pool[k : k + 1].tobytes() == rest.tobytes(), k


def _config(index: int) -> SmoothGenConfig:
    return cast("SmoothGenConfig", SMOOTH_LEVEL_CONFIGS.view(np.recarray)[index])


def _dungeon_config(index: int) -> DungeonConfig:
    return cast("DungeonConfig", DUNGEON_LEVEL_CONFIGS.view(np.recarray)[index])


def _recorder(cells: list[tuple[int, int, int]]) -> object:
    """Return a stand-in for ``refresh_spawn_cell_numba`` that records each cell."""

    def refresh(state: EnvState, level: int, row: int, col: int) -> None:
        del state
        cells.append((level, row, col))

    return refresh


# Any byte the row still held from before would survive into the pool, where the test
# finds it.
def _stamp_world(
    state: EnvState,
    rng: Array1[np.uint32],
    smooth_configs: Records[SmoothGenConfig],
    dungeon_configs: Records[DungeonConfig],
) -> None:
    """Stand in for ``generate_world_numba``: write the stream's seed as the clock."""
    assert smooth_configs.shape == SMOOTH_LEVEL_CONFIGS.shape
    assert dungeon_configs.shape == DUNGEON_LEVEL_CONFIGS.shape
    state.timestep = int(rng[0])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
