"""Procedural generation of the nine floors, one world at a time: noise, recipes, builders.

Two procedures build the world. Open floors -- the surface, the mines, the
elemental realms, the graveyard -- are grown from terrain noise: sea above one
threshold, coast between two, mountains above a second field, then ores and
trees sprinkled where the rules allow. Dungeons are built instead: eight
rooms scattered over a chunk grid, corridors cut between them, torches in
the corners, and a chest and an optional fountain inside.

Terrain is not drawn from independent per-tile randomness -- that would give
static, not landscape. Gradient noise assigns a random angle to each point of
a coarse lattice and interpolates between the four surrounding corners, so
nearby tiles are correlated and the result has continents and mountain ranges
at the lattice's scale. :func:`generate_fractal_numba` draws one ``rand_r`` per
octave for the lattice seed and derives each corner's angle from that seed and
the corner's index with ``rng_f32_at_numba``, so the angles depend on the cell and
not on the raster order in which the loop visits them. Every float32 operation
keeps C's order: a reordered sum or a fused multiply-add changes the low bits,
and the thresholds that turn noise into sea, coast and mountain are exact
comparisons. The C ``generate_fractal`` takes the octave count, persistence and
lacunarity as arguments; every call in the game passes ``1``, ``0.5`` and
``2``, so here they are constants and the octave loop is kept as written.

Each floor supplies its procedure with its own blocks and thresholds. That is
what makes the ice realm read as ice and the fire realm as fire while both run
the same code: only the values differ. The recipes are records of a structured
dtype rather than dataclasses because a Numba kernel reads them as
``config.water_strength``, exactly as the C reads ``config->water_strength``;
a dataclass instance cannot enter a kernel. The ``SmoothGenConfig`` and
``DungeonConfig`` protocols give a kernel's ``config`` argument its field
names for the type checker. The values are ``constants.h``'s tables, row for
row.

Every draw comes from the world's own ``rand_r`` stream in the C's order, so
pool world ``k`` here is byte for byte the C env's ``levels[k]``. The pool is
built once at construction, in parallel over worlds, because a fresh world
costs about 1.4 ms and an episode end copies one instead. Within a world the
builders are scalar Numba loops over numpy arrays, tile by tile in the C's
draw order (the package docstring has the measured case against batching).

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h`` and
        ``ocean/craftax/constants.h``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

from numba import prange

import numpy as np

from priml.baselines.craftax.game.jit import (
    clampf_numba,
    clampi_numba,
    cosf,
    jit,
    jit_parallel,
    powf,
    sinf,
    sqrtf,
    unliteral,
)
from priml.baselines.craftax.game.rng import (
    choice_valid_numba,
    rand_r_numba,
    rng_f32_at_numba,
    rng_f32_numba,
    rng_int_numba,
)
from priml.baselines.craftax.game.rules import refresh_spawn_cell_numba
from priml.baselines.craftax.game.state import (
    BOSS_SPAWN_TURNS,
    DUNGEON_CHUNK_SIZE,
    DUNGEON_MAX_ROOM_SIZE,
    DUNGEON_MIN_ROOM_SIZE,
    DUNGEON_ROOM_COUNT,
    MAP_CELLS,
    MAP_SIZE,
    MAX_MELEE_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PLAYER_PROJECTILES,
    MAX_RANGED_MOBS,
    NOISE_PI2,
    NOISE_SQRT2,
    NUM_LEVELS,
    NUM_POTIONS,
    PI,
    Action,
    BlockType,
    ItemType,
)


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import (
        Array1,
        Array2,
        EnvState,
        Records,
    )


OCTAVES: Final = 1
PERSISTENCE: Final = np.float32(0.5)
"""Amplitude ratio between octaves."""

LACUNARITY: Final = 2
"""Frequency ratio between octaves."""


class SmoothGenConfig(Protocol):
    """A ``SMOOTH_CONFIG_DTYPE`` record: a floor grown from terrain noise.

    Attributes:
      default_block: What a tile is when no other rule claims it.
      sea_block: Fills the low ground above ``water_threshold``.
      coast_block: Rings the sea, between the two thresholds.
      mountain_block: Fills the high ground.
      path_block: Carved through the mountains where the path noise is high.
      inner_mountain_block: Fills the wet high ground inside the mountains.
      ore_requirement_blocks: int32 [5]: which block each ore may replace.
      ores: int32 [5]: the five ores placed on this floor.
      ore_chances: float32 [5]: probability of each ore per eligible tile.
      tree_requirement_block: Which block a tree may grow on.
      tree: The tree-like block for this floor.
      lava: The hazard block; the ice realm uses water.
      player_spawn: Forced under the player so they never spawn inside rock.
      valid_ladder: Which block a ladder may be placed on.
      ladder_up: Whether this floor has an ascent (uint8).
      ladder_down: Whether this floor has a descent (uint8).
      water_strength: How far from the player the sea is suppressed.
      water_max: Cap on that suppression.
      mountain_strength: The same, for mountains.
      mountain_max: Cap on that suppression.
      default_light: Ambient light; zero means the floor is dark.
      water_threshold: Noise level above which a tile becomes sea.
      sand_threshold: Noise level above which a tile becomes coast.
      tree_threshold_uniform: Per-tile draw a tree must beat.
      tree_threshold_perlin: Noise level a tree needs.

    """

    default_block: int
    sea_block: int
    coast_block: int
    mountain_block: int
    path_block: int
    inner_mountain_block: int
    ore_requirement_blocks: Array1[int]
    ores: Array1[int]
    ore_chances: Array1[np.float32]
    tree_requirement_block: int
    tree: int
    lava: int
    player_spawn: int
    valid_ladder: int
    ladder_up: int
    ladder_down: int
    water_strength: np.float32
    water_max: np.float32
    mountain_strength: np.float32
    mountain_max: np.float32
    default_light: np.float32
    water_threshold: np.float32
    sand_threshold: np.float32
    tree_threshold_uniform: np.float32
    tree_threshold_perlin: np.float32


class DungeonConfig(Protocol):
    """A ``DUNGEON_CONFIG_DTYPE`` record: a floor built from rooms and corridors.

    Attributes:
      special_block: Placed once, two tiles into the first room.
      fountain_block: What a room's optional fountain is made of.
      rare_path_replacement_block: Replaces a tenth of the bare path tiles.

    """

    special_block: int
    fountain_block: int
    rare_path_replacement_block: int


SMOOTH_CONFIG_DTYPE: Final = np.dtype(
    [
        ("default_block", np.int32),
        ("sea_block", np.int32),
        ("coast_block", np.int32),
        ("mountain_block", np.int32),
        ("path_block", np.int32),
        ("inner_mountain_block", np.int32),
        ("ore_requirement_blocks", np.int32, (5,)),
        ("ores", np.int32, (5,)),
        ("ore_chances", np.float32, (5,)),
        ("tree_requirement_block", np.int32),
        ("tree", np.int32),
        ("lava", np.int32),
        ("player_spawn", np.int32),
        ("valid_ladder", np.int32),
        ("ladder_up", np.uint8),
        ("ladder_down", np.uint8),
        ("water_strength", np.float32),
        ("water_max", np.float32),
        ("mountain_strength", np.float32),
        ("mountain_max", np.float32),
        ("default_light", np.float32),
        ("water_threshold", np.float32),
        ("sand_threshold", np.float32),
        ("tree_threshold_uniform", np.float32),
        ("tree_threshold_perlin", np.float32),
    ],
)

DUNGEON_CONFIG_DTYPE: Final = np.dtype(
    [
        ("special_block", np.int32),
        ("fountain_block", np.int32),
        ("rare_path_replacement_block", np.int32),
    ],
)

SMOOTH_FLOOR_ORDER: Final = cast(
    "Array1[int]",
    np.array([0, 2, 5, 6, 7, 8], dtype=np.int64),
)
"""The floor each ``SMOOTH_LEVEL_CONFIGS`` row generates, in generation order."""

DUNGEON_FLOOR_ORDER: Final = cast("Array1[int]", np.array([1, 3, 4], dtype=np.int64))
"""The floor each ``DUNGEON_LEVEL_CONFIGS`` row generates, in generation order."""

_STONE_ORES: Final = (
    BlockType.COAL,
    BlockType.IRON,
    BlockType.DIAMOND,
    BlockType.SAPPHIRE,
    BlockType.RUBY,
)

SMOOTH_LEVEL_CONFIGS: Final = np.array(
    [
        # Overworld. The last two ore slots are unused: an impossible ore with
        # a zero chance.
        (
            BlockType.GRASS, BlockType.WATER, BlockType.SAND, BlockType.STONE, BlockType.PATH, BlockType.PATH,
            (BlockType.STONE,) * 5,
            (BlockType.COAL, BlockType.IRON, BlockType.DIAMOND, BlockType.OUT_OF_BOUNDS, BlockType.OUT_OF_BOUNDS),
            (0.03, 0.02, 0.001, 0.0, 0.0),
            BlockType.GRASS, BlockType.TREE, BlockType.LAVA, BlockType.GRASS, BlockType.PATH,
            0, 1, 5.0, 1.0, 5.0, 1.0, 1.0, 0.7, 0.6, 0.8, 0.5,
        ),
        # Gnomish mines.
        (
            BlockType.PATH, BlockType.WATER, BlockType.PATH, BlockType.STONE, BlockType.STONE, BlockType.STONE,
            (BlockType.STONE,) * 5,
            _STONE_ORES,
            (0.04, 0.02, 0.005, 0.0025, 0.0025),
            BlockType.PATH, BlockType.STALAGMITE, BlockType.LAVA, BlockType.PATH, BlockType.PATH,
            1, 1, 5.0, 1.0, 17.0, 1.5, 0.0, 0.7, 0.6, 0.8, 0.5,
        ),
        # Troll mines.
        (
            BlockType.PATH, BlockType.WATER, BlockType.PATH, BlockType.STONE, BlockType.STONE, BlockType.STONE,
            (BlockType.STONE,) * 5,
            _STONE_ORES,
            (0.04, 0.03, 0.01, 0.01, 0.01),
            BlockType.PATH, BlockType.STALAGMITE, BlockType.LAVA, BlockType.PATH, BlockType.PATH,
            1, 1, 5.0, 1.0, 17.0, 1.5, 0.0, 0.7, 0.6, 0.8, 0.5,
        ),
        # Fire realm.
        (
            BlockType.FIRE_GRASS, BlockType.LAVA, BlockType.SAND, BlockType.STONE, BlockType.STONE, BlockType.STONE,
            (BlockType.STONE,) * 5,
            _STONE_ORES,
            (0.05, 0.0, 0.0, 0.0, 0.025),
            BlockType.FIRE_GRASS, BlockType.FIRE_TREE, BlockType.LAVA, BlockType.FIRE_GRASS, BlockType.FIRE_GRASS,
            1, 1, 5.0, 1.0, 5.0, 1.0, 1.0, 0.5, 0.6, 0.8, 0.5,
        ),
        # Ice realm.
        (
            BlockType.ICE_GRASS, BlockType.WATER, BlockType.ICE_GRASS, BlockType.STONE, BlockType.STONE, BlockType.STONE,
            (BlockType.STONE,) * 5,
            _STONE_ORES,
            (0.0, 0.0, 0.005, 0.02, 0.0),
            BlockType.ICE_GRASS, BlockType.ICE_SHRUB, BlockType.WATER, BlockType.ICE_GRASS, BlockType.ICE_GRASS,
            1, 1, 5.0, 1.0, 17.0, 1.5, 0.0, 0.5, 0.6, 0.4, 0.5,
        ),
        # Graveyard: the boss arena, walled in with graves for ore.
        (
            BlockType.PATH, BlockType.PATH, BlockType.PATH, BlockType.WALL, BlockType.WALL, BlockType.WALL,
            (BlockType.WALL, BlockType.GRAVE, BlockType.GRAVE, BlockType.WALL, BlockType.WALL),
            (BlockType.WALL_MOSS, BlockType.GRAVE2, BlockType.GRAVE3, BlockType.SAPPHIRE, BlockType.RUBY),
            (0.1, 0.333, 0.5, 0.0, 0.0),
            BlockType.PATH, BlockType.GRAVE, BlockType.WALL, BlockType.NECROMANCER, BlockType.PATH,
            0, 0, 5.0, 1.0, 10.0, 10.0, 0.0, 0.7, 0.6, 0.95, -1.0,
        ),
    ],
    dtype=SMOOTH_CONFIG_DTYPE,
)  # fmt: skip
"""``constants.h``'s ``SMOOTH_LEVEL_CONFIGS``, one record per ``SMOOTH_FLOOR_ORDER`` entry."""

DUNGEON_LEVEL_CONFIGS: Final = np.array(
    [
        (BlockType.PATH, BlockType.FOUNTAIN, BlockType.PATH),
        (BlockType.ENCHANTMENT_TABLE_ICE, BlockType.WATER, BlockType.WATER),
        (BlockType.ENCHANTMENT_TABLE_FIRE, BlockType.FOUNTAIN, BlockType.PATH),
    ],
    dtype=DUNGEON_CONFIG_DTYPE,
)
"""``constants.h``'s ``DUNGEON_LEVEL_CONFIGS``, one record per ``DUNGEON_FLOOR_ORDER`` entry."""


_LAVA_LIGHT_KERNEL: Final = cast(
    "Array2[np.float32]",
    np.array(
        [[0.2, 0.7, 0.2], [0.7, 1.0, 0.7], [0.2, 0.7, 0.2]],
        dtype=np.float32,
    ),
)
"""How much light a lava tile sheds on itself and its eight neighbours."""


@jit
def generate_world_numba(
    state: EnvState,
    rng: Array1[np.uint32],
    smooth_configs: Records[SmoothGenConfig],
    dungeon_configs: Records[DungeonConfig],
) -> None:
    """Fill a zeroed ``state`` with a new world drawn from ``rng``.

    The C ``memset``s the state first; a caller passes a zeroed record (the
    pool builder zeroes each row), which is what keeps this kernel free of a
    byte view of the record.

    Args:
      state: A zeroed world record, filled in place.
      rng: The world's stream, advanced by every draw.
      smooth_configs: ``SMOOTH_LEVEL_CONFIGS``.
      dungeon_configs: ``DUNGEON_LEVEL_CONFIGS``.

    """
    for i in range(6):
        _generate_smooth_level_numba(
            state,
            rng,
            SMOOTH_FLOOR_ORDER[i],
            smooth_configs[i],
        )
    for i in range(3):
        _generate_dungeon_level_numba(
            state,
            rng,
            DUNGEON_FLOOR_ORDER[i],
            dungeon_configs[i],
        )
    _init_mobs_and_player_numba(state, rng)


@jit_parallel
def build_pool_numba(
    pool: Records[EnvState],
    pool_bytes: Array2[int],
    smooth_configs: Records[SmoothGenConfig],
    dungeon_configs: Records[DungeonConfig],
    first_seed: int,
) -> None:
    """Generate world ``k`` from seed ``first_seed + k`` into every row of ``pool``.

    Runs in parallel over worlds. The C env's pool is seeds
    ``0..num_worlds-1``.

    Args:
      pool: ``STATE_DTYPE [num_worlds]``, overwritten.
      pool_bytes: ``pool`` viewed as ``uint8 [num_worlds, 80248]``, used to
        zero each row as the C's ``calloc`` does.
      smooth_configs: ``SMOOTH_LEVEL_CONFIGS``.
      dungeon_configs: ``DUNGEON_LEVEL_CONFIGS``.
      first_seed: The seed of world 0.

    """
    for k in prange(pool.shape[0]):
        pool_bytes[k, :] = 0
        rng = np.empty(1, dtype=np.uint32)
        rng[0] = first_seed + k
        generate_world_numba(pool[k], rng, smooth_configs, dungeon_configs)


@jit
def generate_fractal_numba(
    rng: Array1[np.uint32],
    res_rows: int,
    res_cols: int,
    out: Array2[np.float32],
) -> None:
    """Sum the octaves of gradient noise into ``out`` and rescale it to ``[0, 1]``.

    Each octave draws one ``rand_r`` from ``rng`` for its lattice seed. The
    rescaling is per field, so every floor is thresholded against the same
    absolute values.

    Args:
      rng: The stream; one draw per octave.
      res_rows: Lattice cells down the field at the first octave; divides the
        field's rows.
      res_cols: Lattice cells across the field at the first octave; divides
        the field's columns.
      out: The field, float32 ``[rows, cols]``, overwritten.

    """
    rows = out.shape[0]
    cols = out.shape[1]
    for row in range(rows):
        for col in range(cols):
            out[row, col] = np.float32(0.0)
    frequency = 1
    amplitude = np.float32(1.0)
    for _ in range(OCTAVES):
        angle_seed = np.uint64(rand_r_numba(rng))
        cell_rows = rows // (frequency * res_rows)
        cell_cols = cols // (frequency * res_cols)
        width = frequency * res_cols + 1
        for row in range(rows):
            grad_row = row // cell_rows
            local_row = np.float32(row - grad_row * cell_rows) / np.float32(cell_rows)
            interp_row = (
                local_row
                * local_row
                * local_row
                * (
                    local_row * (local_row * np.float32(6.0) - np.float32(15.0))
                    + np.float32(10.0)
                )
            )
            for col in range(cols):
                grad_col = col // cell_cols
                local_col = np.float32(col - grad_col * cell_cols) / np.float32(
                    cell_cols,
                )
                interp_col = (
                    local_col
                    * local_col
                    * local_col
                    * (
                        local_col * (local_col * np.float32(6.0) - np.float32(15.0))
                        + np.float32(10.0)
                    )
                )
                corner = grad_row * width + grad_col
                angle_00 = NOISE_PI2 * rng_f32_at_numba(angle_seed, np.uint64(corner))
                angle_01 = NOISE_PI2 * rng_f32_at_numba(
                    angle_seed,
                    np.uint64(corner + 1),
                )
                angle_10 = NOISE_PI2 * rng_f32_at_numba(
                    angle_seed,
                    np.uint64(corner + width),
                )
                angle_11 = NOISE_PI2 * rng_f32_at_numba(
                    angle_seed,
                    np.uint64(corner + width + 1),
                )
                gx_00 = cosf(angle_00)
                gy_00 = sinf(angle_00)
                gx_01 = cosf(angle_01)
                gy_01 = sinf(angle_01)
                gx_10 = cosf(angle_10)
                gy_10 = sinf(angle_10)
                gx_11 = cosf(angle_11)
                gy_11 = sinf(angle_11)
                n00 = local_row * gx_00 + local_col * gy_00
                n10 = (local_row - np.float32(1.0)) * gx_10 + local_col * gy_10
                n01 = local_row * gx_01 + (local_col - np.float32(1.0)) * gy_01
                n11 = (local_row - np.float32(1.0)) * gx_11 + (
                    local_col - np.float32(1.0)
                ) * gy_11
                n0 = n00 * (np.float32(1.0) - interp_row) + interp_row * n10
                n1 = n01 * (np.float32(1.0) - interp_row) + interp_row * n11
                out[row, col] += (
                    amplitude
                    * NOISE_SQRT2
                    * ((np.float32(1.0) - interp_col) * n0 + interp_col * n1)
                )
        frequency *= LACUNARITY
        amplitude *= PERSISTENCE
    min_value = out[0, 0]
    max_value = out[0, 0]
    for row in range(rows):
        for col in range(cols):
            min_value = min(min_value, out[row, col])
            max_value = max(max_value, out[row, col])
    scale = max_value - min_value
    for row in range(rows):
        for col in range(cols):
            out[row, col] = (out[row, col] - min_value) / scale


@jit
def _generate_smooth_level_numba(
    state: EnvState,
    rng: Array1[np.uint32],
    level: int,
    config: SmoothGenConfig,
) -> None:
    """Grow one floor from four noise fields."""
    size = unliteral(MAP_SIZE)
    player_row = size // 2
    player_col = size // 2
    water = _float_grid_numba(size)
    mountain = _float_grid_numba(size)
    path_x = _float_grid_numba(size)
    tree_noise = _float_grid_numba(size)
    lava_map = np.empty((size, size), dtype=np.bool_)
    light_acc = _float_grid_numba(size)

    generate_fractal_numba(rng, 3, 3, water)
    generate_fractal_numba(rng, 3, 3, mountain)
    generate_fractal_numba(rng, 6, 24, path_x)
    tree_seed = np.uint64(rand_r_numba(rng))
    generate_fractal_numba(rng, 12, 12, tree_noise)

    for row in range(size):
        dr = abs(row - player_row)
        for col in range(size):
            dc = abs(col - player_col)
            distance = sqrtf(np.float32(dr * dr + dc * dc))
            proximity_water = clampf_numba(
                distance / config.water_strength,
                np.float32(0.0),
                config.water_max,
            )
            proximity_mountain = clampf_numba(
                distance / config.mountain_strength,
                np.float32(0.0),
                config.mountain_max,
            )
            idx = row * size + col

            water[row, col] = water[row, col] + proximity_water - np.float32(1.0)
            block = (
                config.sea_block
                if water[row, col] > config.water_threshold
                else config.default_block
            )
            if water[row, col] > config.sand_threshold and block != config.sea_block:
                block = config.coast_block

            mountain[row, col] = (
                mountain[row, col]
                + np.float32(0.05)
                + proximity_mountain
                - np.float32(1.0)
            )
            if mountain[row, col] > np.float32(0.7):
                block = config.mountain_block
            if mountain[row, col] > np.float32(0.7) and path_x[row, col] > np.float32(
                0.8,
            ):
                block = config.path_block
            if mountain[row, col] > np.float32(0.7) and path_x[col, row] > np.float32(
                0.8,
            ):
                block = config.path_block
            if mountain[row, col] > np.float32(0.85) and water[row, col] > np.float32(
                0.4,
            ):
                block = config.inner_mountain_block
            if (
                tree_noise[row, col] > config.tree_threshold_perlin
                and rng_f32_at_numba(tree_seed, np.uint64(idx))
                > config.tree_threshold_uniform
                and block == config.tree_requirement_block
            ):
                block = config.tree

            state.map[level, row, col] = block
            state.item_map[level, row, col] = ItemType.NONE
            light_acc[row, col] = config.default_light

    for ore_index in range(5):
        ore_seed = np.uint64(rand_r_numba(rng))
        for row in range(size):
            for col in range(size):
                idx = row * size + col
                if (
                    state.map[level, row, col]
                    == config.ore_requirement_blocks[ore_index]
                    and rng_f32_at_numba(ore_seed, np.uint64(idx))
                    < config.ore_chances[ore_index]
                ):
                    state.map[level, row, col] = config.ores[ore_index]

    for row in range(size):
        for col in range(size):
            lava_map[row, col] = mountain[row, col] > np.float32(0.85) and tree_noise[
                row,
                col,
            ] > np.float32(0.7)
            if lava_map[row, col]:
                state.map[level, row, col] = config.lava

    valid = np.empty(MAP_CELLS, dtype=np.bool_)
    for row in range(size):
        for col in range(size):
            valid[row * size + col] = state.map[level, row, col] == BlockType.STONE
    # The C writes stone over a stone tile: a no-op that still consumes a draw.
    diamond_index = choice_valid_numba(rng, valid, unliteral(MAP_CELLS))
    state.map[level, diamond_index // size, diamond_index % size] = BlockType.STONE
    state.map[level, player_row, player_col] = config.player_spawn

    for row in range(size):
        for col in range(size):
            valid[row * size + col] = state.map[level, row, col] == config.valid_ladder

    ladder_down_index = choice_valid_numba(rng, valid, unliteral(MAP_CELLS))
    state.down_ladders[level, 0] = ladder_down_index // size
    state.down_ladders[level, 1] = ladder_down_index % size
    if config.ladder_down:
        state.item_map[
            level,
            state.down_ladders[level, 0],
            state.down_ladders[level, 1],
        ] = ItemType.LADDER_DOWN

    ladder_up_index = choice_valid_numba(rng, valid, unliteral(MAP_CELLS))
    r = ladder_up_index // size
    c = ladder_up_index % size
    state.up_ladders[level, 0] = r
    state.up_ladders[level, 1] = c
    light_row = r - 4
    light_col = c - 4
    if light_row < 0:
        light_row += size
    if light_col < 0:
        light_col += size
    light_row = clampi_numba(light_row, 0, size - 9)
    light_col = clampi_numba(light_col, 0, size - 9)
    for lr in range(9):
        for lc in range(9):
            torch_light = clampf_numba(
                np.float32(1.0)
                - sqrtf(np.float32((lr - 4) * (lr - 4) + (lc - 4) * (lc - 4)))
                / np.float32(5.0),
                np.float32(0.0),
                np.float32(1.0),
            )
            light = (
                torch_light * (np.float32(1.0) - config.default_light)
                + config.default_light
            )
            light_acc[light_row + lr, light_col + lc] = light
    if config.lava == BlockType.LAVA:
        for row in range(size):
            for col in range(size):
                add = np.float32(0.0)
                for kr in range(3):
                    src_row = row + kr - 1
                    if src_row < 0 or src_row >= size:
                        continue
                    for kc in range(3):
                        src_col = col + kc - 1
                        if src_col < 0 or src_col >= size:
                            continue
                        if lava_map[src_row, src_col]:
                            add += _LAVA_LIGHT_KERNEL[kr, kc]
                light_acc[row, col] = clampf_numba(
                    light_acc[row, col] + add,
                    np.float32(0.0),
                    np.float32(1.0),
                )
    for row in range(size):
        for col in range(size):
            light = clampf_numba(light_acc[row, col], np.float32(0.0), np.float32(1.0))
            state.light_map[level, row, col] = np.uint8(light * np.float32(255.0))
    if config.ladder_up:
        state.item_map[level, r, c] = ItemType.LADDER_UP


@jit
def _generate_dungeon_level_numba(
    state: EnvState,
    rng: Array1[np.uint32],
    level: int,
    config: DungeonConfig,
) -> None:
    """Build one floor from rooms and corridors."""
    size = unliteral(MAP_SIZE)
    chunk_size = unliteral(DUNGEON_CHUNK_SIZE)
    world_chunk_height = size // chunk_size
    num_rooms = unliteral(DUNGEON_ROOM_COUNT)
    min_room_size = unliteral(DUNGEON_MIN_ROOM_SIZE)
    max_room_size = unliteral(DUNGEON_MAX_ROOM_SIZE)
    padded_size = size + 2 * max_room_size

    padded_map = np.empty((padded_size, padded_size), dtype=np.int32)
    padded_item = np.empty((padded_size, padded_size), dtype=np.int32)
    room_occupancy = np.ones(9, dtype=np.bool_)
    room_sizes = _int_pairs_numba(num_rooms)
    room_positions = _int_pairs_numba(num_rooms)

    for row in range(padded_size):
        for col in range(padded_size):
            inner = (
                row >= max_room_size
                and row < max_room_size + size
                and col >= max_room_size
                and col < max_room_size + size
            )
            padded_map[row, col] = BlockType.WALL if inner else 0
            padded_item[row, col] = ItemType.NONE

    for room in range(num_rooms):
        room_sizes[room, 0] = rng_int_numba(rng, min_room_size, max_room_size)
        room_sizes[room, 1] = rng_int_numba(rng, min_room_size, max_room_size)

    for room_index in range(num_rooms):
        room_chunk = choice_valid_numba(rng, room_occupancy, 9)
        room_occupancy[room_chunk] = False
        room_row = (room_chunk % world_chunk_height) * chunk_size + max_room_size
        room_col = (room_chunk // world_chunk_height) * chunk_size + max_room_size
        room_row += rng_int_numba(rng, 0, chunk_size - min_room_size)
        room_col += rng_int_numba(rng, 0, chunk_size - min_room_size)
        room_positions[room_index, 0] = room_row
        room_positions[room_index, 1] = room_col

        for row in range(max_room_size):
            for col in range(max_room_size):
                if row < room_sizes[room_index, 0] and col < room_sizes[room_index, 1]:
                    padded_map[room_row + row, room_col + col] = BlockType.PATH

        padded_item[room_row, room_col] = ItemType.TORCH
        padded_item[room_row + room_sizes[room_index, 0] - 1, room_col] = ItemType.TORCH
        padded_item[room_row, room_col + room_sizes[room_index, 1] - 1] = ItemType.TORCH
        padded_item[
            room_row + room_sizes[room_index, 0] - 1,
            room_col + room_sizes[room_index, 1] - 1,
        ] = ItemType.TORCH

        chest_row = rng_int_numba(rng, 1, room_sizes[room_index, 0] - 1)
        chest_col = rng_int_numba(rng, 1, room_sizes[room_index, 1] - 1)
        padded_map[room_row + chest_row, room_col + chest_col] = BlockType.CHEST

        fountain_row = rng_int_numba(rng, 1, room_sizes[room_index, 0] - 1)
        fountain_col = rng_int_numba(rng, 1, room_sizes[room_index, 1] - 1)
        if rng_f32_numba(rng) > np.float32(0.5):
            padded_map[room_row + fountain_row, room_col + fountain_col] = (
                config.fountain_block
            )

    included_rooms = np.zeros(num_rooms, dtype=np.bool_)
    included_rooms[num_rooms - 1] = True
    for path_index in range(num_rooms):
        source_row = room_positions[path_index, 0]
        source_col = room_positions[path_index, 1]
        sink_index = choice_valid_numba(rng, included_rooms, num_rooms)
        sink_row = room_positions[sink_index, 0]
        sink_col = room_positions[sink_index, 1]

        horizontal_distance = sink_col - source_col
        horizontal_sign = int(horizontal_distance > 0) - int(horizontal_distance < 0)
        if horizontal_sign != 0:
            abs_distance = abs(horizontal_distance)
            for col in range(padded_size):
                path_index_col = (col - source_col) * horizontal_sign
                if (
                    path_index_col >= 0
                    and path_index_col <= abs_distance
                    and padded_map[source_row, col] == BlockType.WALL
                ):
                    padded_map[source_row, col] = BlockType.PATH
        vertical_distance = sink_row - source_row
        vertical_sign = int(vertical_distance > 0) - int(vertical_distance < 0)
        if vertical_sign != 0:
            abs_distance = abs(vertical_distance)
            for row in range(padded_size):
                path_index_row = (row - source_row) * vertical_sign
                if (
                    path_index_row >= 0
                    and path_index_row <= abs_distance
                    and padded_map[row, sink_col] == BlockType.WALL
                ):
                    padded_map[row, sink_col] = BlockType.PATH
        included_rooms[path_index] = True

    padded_map[room_positions[0, 0] + 2, room_positions[0, 1] + 2] = (
        config.special_block
    )

    for row in range(size):
        for col in range(size):
            state.map[level, row, col] = padded_map[
                row + max_room_size,
                col + max_room_size,
            ]
            state.item_map[level, row, col] = padded_item[
                row + max_room_size,
                col + max_room_size,
            ]

    adjacent_path = np.empty((size, size), dtype=np.bool_)
    for row in range(size):
        for col in range(size):
            adjacent = state.map[level, row, col] != BlockType.WALL
            adjacent = adjacent or (
                row > 0 and state.map[level, row - 1, col] != BlockType.WALL
            )
            adjacent = adjacent or (
                row + 1 < size and state.map[level, row + 1, col] != BlockType.WALL
            )
            adjacent = adjacent or (
                col > 0 and state.map[level, row, col - 1] != BlockType.WALL
            )
            adjacent = adjacent or (
                col + 1 < size and state.map[level, row, col + 1] != BlockType.WALL
            )
            adjacent_path[row, col] = adjacent

    rare_seed = np.uint64(rand_r_numba(rng))
    for row in range(size):
        for col in range(size):
            idx = row * size + col
            rare = rng_f32_at_numba(rare_seed, np.uint64(idx)) < np.float32(0.1)
            wall_map = BlockType.WALL_MOSS if rare else BlockType.WALL
            rare_path = (
                rare
                and state.map[level, row, col] == BlockType.PATH
                and state.item_map[level, row, col] == ItemType.NONE
            )
            path_map = (
                config.rare_path_replacement_block
                if rare_path
                else state.map[level, row, col]
            )
            is_wall_map = (
                state.map[level, row, col] == BlockType.WALL and adjacent_path[row, col]
            )
            if not adjacent_path[row, col]:
                state.map[level, row, col] = BlockType.DARKNESS
            elif is_wall_map:
                state.map[level, row, col] = wall_map
            else:
                state.map[level, row, col] = path_map
            state.light_map[level, row, col] = 255

    valid = np.empty(MAP_CELLS, dtype=np.bool_)
    for row in range(size):
        for col in range(size):
            valid[row * size + col] = state.map[level, row, col] == BlockType.PATH
    ladder_down_index = choice_valid_numba(rng, valid, unliteral(MAP_CELLS))
    r = ladder_down_index // size
    c = ladder_down_index % size
    state.down_ladders[level, 0] = r
    state.down_ladders[level, 1] = c
    state.item_map[level, r, c] = ItemType.LADDER_DOWN

    ladder_up_index = choice_valid_numba(rng, valid, unliteral(MAP_CELLS))
    r = ladder_up_index // size
    c = ladder_up_index % size
    state.up_ladders[level, 0] = r
    state.up_ladders[level, 1] = c
    state.item_map[level, r, c] = ItemType.LADDER_UP


@jit
def _init_mobs_and_player_numba(state: EnvState, rng: Array1[np.uint32]) -> None:
    """Set the mob slots, the potion shuffle, the player and the spawn bitsets."""
    for level in range(NUM_LEVELS):
        for i in range(MAX_MELEE_MOBS):
            state.melee_mobs[level].health[i] = np.float32(1.0)
            state.passive_mobs[level].health[i] = np.float32(1.0)
            state.mob_projectiles[level].health[i] = np.float32(1.0)
            state.player_projectiles[level].health[i] = np.float32(1.0)
        for i in range(MAX_RANGED_MOBS):
            state.ranged_mobs[level].health[i] = np.float32(1.0)
        for projectile in range(MAX_MOB_PROJECTILES):
            state.mob_projectile_dirs[level, projectile, 0] = 1
            state.mob_projectile_dirs[level, projectile, 1] = 1
        for projectile in range(MAX_PLAYER_PROJECTILES):
            state.player_projectile_directions[level, projectile, 0] = 1
            state.player_projectile_directions[level, projectile, 1] = 1

    for i in range(NUM_POTIONS):
        state.potion_mapping[i] = i
    for i in range(5, 0, -1):
        j = rng_int_numba(rng, 0, i + 1)
        # rng_int returns i + 1 on the 1.0 edge; at i == 5 the C then swaps
        # potion_mapping[6], which is learned_spells[0] in the struct.
        tmp = state.potion_mapping[i]
        if j == NUM_POTIONS:
            state.potion_mapping[i] = state.learned_spells[0]
            state.learned_spells[0] = tmp
        else:
            state.potion_mapping[i] = state.potion_mapping[j]
            state.potion_mapping[j] = tmp

    state.monsters_killed[0] = 10
    state.player_position[0] = MAP_SIZE // 2
    state.player_position[1] = MAP_SIZE // 2
    state.player_level = 0
    state.player_direction = Action.UP
    state.player_health = np.float32(9.0)
    state.player_food = 9
    state.player_drink = 9
    state.player_energy = 9
    state.player_mana = 9
    state.player_dexterity = 1
    state.player_strength = 1
    state.player_intelligence = 1
    state.boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS
    cosine = cosf(PI * np.float32(0.3))
    state.light_level = np.float32(1.0) - powf(abs(cosine), np.float32(3.0))
    for level in range(NUM_LEVELS):
        for row in range(MAP_SIZE):
            state.spawn_land[level, row] = 0
            state.spawn_grave[level, row] = 0
            state.spawn_water[level, row] = 0
    for level in range(NUM_LEVELS):
        for row in range(MAP_SIZE):
            for col in range(MAP_SIZE):
                refresh_spawn_cell_numba(state, level, row, col)


@jit
def _float_grid_numba(size: int) -> Array2[np.float32]:
    """Return an uninitialized float32 ``[size, size]`` field, typed for its reads."""
    return np.empty((size, size), dtype=np.float32)


@jit
def _int_pairs_numba(count: int) -> Array2[int]:
    """Return an uninitialized int64 ``[count, 2]`` table, typed for its reads."""
    return np.empty((count, 2), dtype=np.int64)
