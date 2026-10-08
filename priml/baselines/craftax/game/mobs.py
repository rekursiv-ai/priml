"""Creatures: how they move, shoot, fly and spawn.

Two phases of the step. ``update_mobs_numba`` moves every slot of the current
floor in a fixed order -- three melee, three passive, two ranged, then
the creatures' projectiles, then the player's -- and every slot draws from
the stream whether or not it is live, so dead slots move too (nowhere) and
their draws are consumed. ``spawn_mobs_numba`` rolls one draw per creature class
and, for each that succeeds, picks a free tile in the annulus around the
player from the spawn bitsets.

Two things the C does are kept deliberately. The type of a floor's next
creature is written into the first free slot every step, even when no spawn
follows and even into slot 0 when none is free. And a spawn-index draw of
``n`` -- possible when ``rng_f32_numba`` returns 1.0 -- reads past the candidate
list in the C; this takes candidate ``n - 1`` and reports the event, so a
comparator can drop the environment from that step on.

Both phases run per environment and per slot: most draws here are
conditional and each spawn's candidate list has its own length, which a
batch would pay for as masked work over every environment (the package
docstring has the measurements).

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h``, pin
        ``6ffa5b10``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import numpy as np

from priml.baselines.craftax.game.jit import (
    clampi_numba,
    jit,
    popcount,
    trailing_zeros,
    unliteral,
)
from priml.baselines.craftax.game.rng import (
    choose_weighted_pair_numba,
    rng_f32_numba,
    rng_int_numba,
)
from priml.baselines.craftax.game.rules import (
    damage_mob_at_numba,
    damage_to_mob_numba,
    damage_to_player_numba,
    fighting_boss_numba,
    find_mob_at_numba,
    is_solid_block_numba,
    mob_at_numba,
    mob_damage_vector_numba,
    mobs_for_class_numba,
    move_mob_occupancy_numba,
    set_block_numba,
    set_mob_bit_numba,
    spawn_projectile_numba,
    valid_typed_mob_position_numba,
)
from priml.baselines.craftax.game.state import (
    MAP_SIZE,
    MAX_MELEE_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PASSIVE_MOBS,
    MAX_PLAYER_PROJECTILES,
    MAX_RANGED_MOBS,
    MOB_DESPAWN_DISTANCE,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    NUM_LEVELS,
    NUM_MOB_TYPES,
    Achievement,
    BlockType,
    MobType,
    ProjectileType,
)


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import (
        Array1,
        Array2,
        Array3,
        EnvState,
        Mobs,
    )


FLOOR_MOB_TYPES: Final = cast(
    "Array2[int]",
    np.array(
        [[0, 0, 0], [2, 2, 2], [1, 1, 1], [2, 3, 3], [2, 4, 4], [1, 5, 5], [1, 6, 6], [1, 7, 7], [0, 0, 0]],
        dtype=np.int64,
    ),
)  # fmt: skip
"""``[level, mob_class]``: the species each floor spawns."""

SPAWN_CHANCES: Final = cast(
    "Array2[np.float32]",
    np.array(
        [
            [0.1, 0.02, 0.05, 0.1],
            [0.1, 0.06, 0.05, 0.0],
            [0.1, 0.06, 0.05, 0.0],
            [0.1, 0.06, 0.05, 0.0],
            [0.1, 0.06, 0.05, 0.0],
            [0.1, 0.06, 0.05, 0.0],
            [0.1, 0.06, 0.05, 0.0],
            [0.0, 0.06, 0.05, 0.0],
            [0.1, 0.06, 0.05, 0.0],
        ],
        dtype=np.float32,
    ),
)
"""``[level, (passive, melee, ranged, melee at night)]`` per-step spawn odds."""

RANGED_PROJECTILES: Final = cast(
    "Array1[int]",
    np.array(
        [
            ProjectileType.ARROW,
            ProjectileType.ARROW,
            ProjectileType.FIREBALL,
            ProjectileType.DAGGER,
            ProjectileType.ARROW2,
            ProjectileType.SLIMEBALL,
            ProjectileType.FIREBALL2,
            ProjectileType.ICEBALL2,
        ],
        dtype=np.int64,
    ),
)
"""What each ranged species shoots."""

PASSIVE_HEALTH: Final = cast(
    "Array1[np.float32]",
    np.array([3, 4, 6, 8, 0, 0, 0, 0], dtype=np.float32),
)
MELEE_HEALTH: Final = cast(
    "Array1[np.float32]",
    np.array([5, 7, 9, 11, 12, 20, 20, 24], dtype=np.float32),
)
RANGED_HEALTH: Final = cast(
    "Array1[np.float32]",
    np.array([3, 5, 6, 8, 12, 4, 14, 16], dtype=np.float32),
)

SPAWN_REACH: Final = MOB_DESPAWN_DISTANCE - 1
"""How many rows or columns from the player a spawn may land: the C's ``limit``."""

PASSIVE_RING: Final = 0
HOSTILE_RING: Final = 1
BOSS_RING: Final = 2


# Entry ``[ring, dr, 0]`` is the largest column offset ``dc <= SPAWN_REACH`` with ``dr²
# + dc² < max_exclusive`` and ``[ring, dr, 1]`` the largest with ``dr² + dc² <=
# min_exclusive``, each -1 when there is none, so a tile at ``(dr, dc)`` is in the band
# exactly when ``[.., 1] < |dc| <= [.., 0]``.
def _ring_table(*bands: tuple[int, int]) -> np.ndarray:
    """Tabulate each ``(min_exclusive, max_exclusive)`` squared-distance band by row."""
    table = np.full((len(bands), SPAWN_REACH + 1, 2), -1, dtype=np.int64)
    for ring, (low, high) in enumerate(bands):
        for dr in range(SPAWN_REACH + 1):
            for dc in range(SPAWN_REACH + 1):
                if dr * dr + dc * dc < high:
                    table[ring, dr, 0] = dc
                if dr * dr + dc * dc <= low:
                    table[ring, dr, 1] = dc
    return table


SPAWN_RINGS: Final = cast(
    "Array3[int]",
    _ring_table(
        (9, MOB_DESPAWN_DISTANCE * MOB_DESPAWN_DISTANCE),
        (81, MOB_DESPAWN_DISTANCE * MOB_DESPAWN_DISTANCE),
        (-1, 37),
    ),
)
"""The squared-distance bands the C's spawns pass to ``collect_spawn_cells``:
passive ``(9, 196)``, hostile ``(81, 196)`` and hostile on the boss floor
``(-1, 37)``, indexed by ``PASSIVE_RING``, ``HOSTILE_RING`` and ``BOSS_RING``."""


@jit
def update_mobs_numba(state: EnvState, rng: Array1[np.uint32]) -> None:
    """Move every creature slot and projectile on the player's floor, in the C's order.

    Args:
      state: The environment's world, mutated in place.
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.

    """
    level = state.player_level
    move_melee_slot_numba(state, level, 0, rng)
    move_melee_slot_numba(state, level, 1, rng)
    move_melee_slot_numba(state, level, 2, rng)
    move_passive_slot_numba(state, level, 0, rng)
    move_passive_slot_numba(state, level, 1, rng)
    move_passive_slot_numba(state, level, 2, rng)
    move_ranged_slot_numba(state, level, 0, rng)
    move_ranged_slot_numba(state, level, 1, rng)
    update_projectiles_numba(state, False)
    update_projectiles_numba(state, True)


@jit
def spawn_mobs_numba(state: EnvState, rng: Array1[np.uint32]) -> int:
    """Roll the spawns for this step; return how many spawn draws hit the edge.

        Returns the number of undefined-behaviour events (a spawn-index draw of
        ``n``), for which the port used candidate ``n - 1``.

    Args:
      state: The environment's world, mutated in place.
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.

    Returns:
      undefined: How many spawn-index draws hit the 1.0 edge.

    """
    level = state.player_level
    boss = fighting_boss_numba(state)
    coeff = 1 + (
        2 if state.monsters_killed[level] < MONSTERS_KILLED_TO_CLEAR_LEVEL else 0
    )
    if boss:
        coeff *= 1000 if state.boss_timestep_to_spawn_this_round >= 1 else 0

    night = np.float32(1.0) - state.light_level
    melee_chance = SPAWN_CHANCES[level, 1] + SPAWN_CHANCES[level, 3] * night * night
    hostile = state.boss_progress if boss else level

    passive_count, passive_slot = count_and_empty_numba(
        state.passive_mobs[level],
        unliteral(MAX_PASSIVE_MOBS),
    )
    passive_type = floor_mob_type_numba(level, MobType.PASSIVE)
    state.passive_mobs[level].type_id[passive_slot] = passive_type

    melee_count, melee_slot = count_and_empty_numba(
        state.melee_mobs[level],
        unliteral(MAX_MELEE_MOBS),
    )
    melee_type = floor_mob_type_numba(hostile, MobType.MELEE)
    state.melee_mobs[level].type_id[melee_slot] = melee_type

    ranged_count, ranged_slot = count_and_empty_numba(
        state.ranged_mobs[level],
        unliteral(MAX_RANGED_MOBS),
    )
    ranged_type = floor_mob_type_numba(hostile, MobType.RANGED)
    state.ranged_mobs[level].type_id[ranged_slot] = ranged_type

    # Conditional draws over a per-environment candidate count: scalar on
    # purpose (the package docstring's spawn-collection measurement).
    try_passive = (
        not boss
        and passive_count < MAX_PASSIVE_MOBS
        and rng_f32_numba(rng) < SPAWN_CHANCES[level, 0]
    )
    try_melee = melee_count < MAX_MELEE_MOBS and rng_f32_numba(
        rng,
    ) < melee_chance * np.float32(coeff)
    try_ranged = ranged_count < MAX_RANGED_MOBS and rng_f32_numba(
        rng,
    ) < SPAWN_CHANCES[level, 2] * np.float32(coeff)
    undefined = 0
    hostile_ring = BOSS_RING if boss else HOSTILE_RING
    if try_passive:
        n = count_spawn_cells_numba(state, level, PASSIVE_RING, False, False)
        if n > 0:
            chosen = rng_int_numba(rng, 0, n)
            if chosen == n:
                chosen = n - 1
                undefined += 1
            row, col = select_spawn_cell_numba(
                state,
                level,
                PASSIVE_RING,
                False,
                False,
                chosen,
            )
            spawn_into_slot_numba(
                state, level, state.passive_mobs[level], passive_slot, MobType.PASSIVE,
                passive_type, row, col,
            )  # fmt: skip
    if try_melee:
        n = count_spawn_cells_numba(state, level, hostile_ring, boss, False)
        if n > 0:
            chosen = rng_int_numba(rng, 0, n)
            if chosen == n:
                chosen = n - 1
                undefined += 1
            row, col = select_spawn_cell_numba(
                state,
                level,
                hostile_ring,
                boss,
                False,
                chosen,
            )
            spawn_into_slot_numba(
                state, level, state.melee_mobs[level], melee_slot, MobType.MELEE,
                melee_type, row, col,
            )  # fmt: skip
    if try_ranged:
        water = ranged_type == 5
        n = count_spawn_cells_numba(state, level, hostile_ring, boss, water)
        if n > 0:
            chosen = rng_int_numba(rng, 0, n)
            if chosen == n:
                chosen = n - 1
                undefined += 1
            row, col = select_spawn_cell_numba(
                state,
                level,
                hostile_ring,
                boss,
                water,
                chosen,
            )
            spawn_into_slot_numba(
                state, level, state.ranged_mobs[level], ranged_slot, MobType.RANGED,
                ranged_type, row, col,
            )  # fmt: skip
    return undefined


@jit
def floor_mob_type_numba(level: int, mob_class: int) -> int:
    """Return the species a floor spawns for a creature class.

    Args:
      level: The floor (or the boss progress on the last floor).
      mob_class: The creature class.

    Returns:
      type_id: The species.

    """
    return int(
        FLOOR_MOB_TYPES[
            clampi_numba(level, 0, NUM_LEVELS - 1),
            clampi_numba(mob_class, 0, 2),
        ],
    )


@jit
def count_spawn_cells_numba(
    state: EnvState,
    level: int,
    ring: int,
    boss: bool,
    water_only: bool,
) -> int:
    """Count the free spawnable tiles in a ring around the player (``collect_spawn_cells``).

        The C lists them into arrays and indexes the list with the spawn draw;
        the count and :func:`select_spawn_cell_numba` give the same tile from the
        rows' bitsets without the list.

    Args:
      state: The environment's world.
      level: The floor.
      ring: ``PASSIVE_RING``, ``HOSTILE_RING`` or ``BOSS_RING``.
      boss: Scan the grave bitset instead of land.
      water_only: Scan the water bitset instead of land.

    Returns:
      count: How many tiles qualify.

    """
    row_player = state.player_position[0]
    count = 0
    for row in range(
        max(row_player - SPAWN_REACH, 0),
        min(row_player + SPAWN_REACH, MAP_SIZE - 1) + 1,
    ):
        count += popcount(
            spawn_row_cells_numba(state, level, ring, boss, water_only, row),
        )
    return count


@jit
def select_spawn_cell_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    state: EnvState,
    level: int,
    ring: int,
    boss: bool,
    water_only: bool,
    index: int,
) -> tuple[int, int]:
    """Return the tile at ``index`` in the C's list order: rows ascending, then columns.

    Args:
      state: The environment's world.
      level: The floor.
      ring: ``PASSIVE_RING``, ``HOSTILE_RING`` or ``BOSS_RING``.
      boss: Scan the grave bitset instead of land.
      water_only: Scan the water bitset instead of land.
      index: Below :func:`count_spawn_cells_numba`'s count.

    Returns:
      row: The tile's row, -1 when ``index`` is past the count.
      col: The tile's column, likewise.

    """
    row_player = state.player_position[0]
    for row in range(
        max(row_player - SPAWN_REACH, 0),
        min(row_player + SPAWN_REACH, MAP_SIZE - 1) + 1,
    ):
        bits = spawn_row_cells_numba(state, level, ring, boss, water_only, row)
        in_row = popcount(bits)
        if index < in_row:
            for _ in range(index):
                bits &= bits - np.uint64(1)
            return row, trailing_zeros(bits)
        index -= in_row
    return -1, -1


@jit
def spawn_row_cells_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    state: EnvState,
    level: int,
    ring: int,
    boss: bool,
    water_only: bool,
    row: int,
) -> np.uint64:
    """Return one row's free spawnable tiles in the ring, bit ``c`` for column ``c``.

    Args:
      state: The environment's world.
      level: The floor.
      ring: ``PASSIVE_RING``, ``HOSTILE_RING`` or ``BOSS_RING``.
      boss: Read the grave bitset instead of land.
      water_only: Read the water bitset instead of land.
      row: The map row.

    Returns:
      cells: The row's qualifying columns as bits.

    """
    distance = abs(row - state.player_position[0])
    outer = SPAWN_RINGS[ring, distance, 0]
    if outer < 0:
        return np.uint64(0)
    inner = SPAWN_RINGS[ring, distance, 1]
    col_player = state.player_position[1]
    if boss:
        terrain = state.spawn_grave[level, row]
    elif water_only:
        terrain = state.spawn_water[level, row]
    else:
        terrain = state.spawn_land[level, row]
    return np.uint64(
        terrain
        & ~state.mob_bits[level, row]
        & column_span_numba(col_player - outer, col_player + outer)
        & ~column_span_numba(col_player - inner, col_player + inner),
    )


@jit
def column_span_numba(first: int, last: int) -> np.uint64:
    """Return bits ``first..last`` of a map row, clipped to the map.

    Args:
      first: The lowest column, possibly off the map.
      last: The highest column, possibly off the map.

    Returns:
      bits: Bit ``c`` set for each column ``c`` in range; none if the range
        is empty.

    """
    first = max(first, 0)
    last = min(last, MAP_SIZE - 1)
    if first > last:
        return np.uint64(0)
    return (~np.uint64(0) << np.uint64(first)) & (
        (np.uint64(1) << np.uint64(last + 1)) - np.uint64(1)
    )


@jit
def spawn_into_slot_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured), and the C signature has this many.
    state: EnvState,
    level: int,
    mobs: Mobs,
    slot: int,
    mob_class: int,
    type_id: int,
    row: int,
    col: int,
) -> None:
    """Fill a slot with a creature of ``type_id`` at full health on the tile.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      mobs: The slot table.
      slot: The slot to fill.
      mob_class: The creature class.
      type_id: The species.
      row: Tile row.
      col: Tile column.

    """
    idx = clampi_numba(type_id, 0, NUM_MOB_TYPES - 1)
    health = MELEE_HEALTH[idx]
    if mob_class == MobType.PASSIVE:
        health = PASSIVE_HEALTH[idx]
    elif mob_class == MobType.RANGED:
        health = RANGED_HEALTH[idx]
    mobs.position[slot, 0] = row
    mobs.position[slot, 1] = col
    mobs.health[slot] = health
    mobs.mask[slot] = 1
    set_mob_bit_numba(state, level, row, col, True)


@jit
def count_and_empty_numba(mobs: Mobs, slots: int) -> tuple[int, int]:
    """Return how many slots are live and the first free one (0 when none is).

    Args:
      mobs: The slot table.
      slots: How many slots it uses.

    Returns:
      count: Live slots.
      empty: The first free slot, 0 when none is free.

    """
    n = 0
    first = 0
    found = False
    for i in range(slots):
        n += 1 if mobs.mask[i] else 0
        if not mobs.mask[i] and not found:
            first = i
            found = True
    return n, first


@jit
def choose_direction_numba(rng: Array1[np.uint32], count: int) -> tuple[int, int]:
    """Draw one of ``count`` choices; the first four are the compass steps, the rest stay put.

    Args:
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      count: Choices to draw from.

    Returns:
      direction: ``(row, column)`` step.

    """
    choice = rng_int_numba(rng, 0, count)
    if choice == 0:
        return 0, -1
    if choice == 1:
        return 0, 1
    if choice == 2:
        return -1, 0
    if choice == 3:
        return 1, 0
    return 0, 0


@jit
def choose_player_axis_numba(
    rng: Array1[np.uint32],
    distance_row: int,
    distance_col: int,
) -> int:
    """Pick the axis (0 rows, 1 columns) along which to approach the player.

    Args:
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      distance_row: Row distance to the player.
      distance_col: Column distance to the player.

    Returns:
      axis: 0 for rows, 1 for columns.

    """
    total = distance_row + distance_col
    if total == 0:
        return 1
    maximum = max(distance_row, distance_col)
    return choose_weighted_pair_numba(
        rng,
        np.float32(1.0) / np.float32(total)
        if distance_row == maximum
        else np.float32(0.0),
        np.float32(1.0) / np.float32(total)
        if distance_col == maximum
        else np.float32(0.0),
    )


@jit
def move_melee_slot_numba(
    state: EnvState,
    level: int,
    slot: int,
    rng: Array1[np.uint32],
) -> None:
    """Move one melee slot: chase or wander, strike when adjacent, despawn when far.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      slot: The slot to move.
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.

    """
    mobs = state.melee_mobs[level]
    alive = mobs.mask[slot] != 0
    old_row = mobs.position[slot, 0]
    old_col = mobs.position[slot, 1]
    type_id = mobs.type_id[slot]
    cooldown = mobs.attack_cooldown[slot]

    random_dr, random_dc = choose_direction_numba(rng, 4)
    distance_row = abs(state.player_position[0] - old_row)
    distance_col = abs(state.player_position[1] - old_col)
    axis = choose_player_axis_numba(rng, distance_row, distance_col)
    player_dr = 0
    player_dc = 0
    if axis == 0:
        dr = state.player_position[0] - old_row
        player_dr = int(dr > 0) - int(dr < 0)
    else:
        dc = state.player_position[1] - old_col
        player_dc = int(dc > 0) - int(dc < 0)
    dist = distance_row + distance_col
    chase_roll = rng_f32_numba(rng)
    chase = (dist < 10 or fighting_boss_numba(state)) and chase_roll < np.float32(0.75)
    proposed_row = old_row + player_dr if chase else old_row + random_dr
    proposed_col = old_col + player_dc if chase else old_col + random_dc
    attacking = dist == 1 and cooldown <= 0 and alive
    if attacking:
        proposed_row = old_row
        proposed_col = old_col
        physical, fire, ice = mob_damage_vector_numba(type_id, MobType.MELEE)
        sleep = np.float32(1.0) + np.float32(2.5) * np.float32(state.is_sleeping)
        physical *= sleep
        fire *= sleep
        ice *= sleep
        state.player_health -= damage_to_player_numba(state, physical, fire, ice)
        state.achievements[Achievement.WAKE_UP.value] = (
            1
            if state.achievements[Achievement.WAKE_UP.value] or state.is_sleeping
            else 0
        )
        state.is_sleeping = 0
        state.is_resting = 0
    new_cooldown = 5 if attacking else cooldown - 1
    valid = valid_typed_mob_position_numba(
        state, level, MobType.MELEE, type_id, proposed_row, proposed_col, old_row, old_col,
    )  # fmt: skip
    new_row = proposed_row if valid else old_row
    new_col = proposed_col if valid else old_col
    keep = alive and (dist < MOB_DESPAWN_DISTANCE or fighting_boss_numba(state))
    move_mob_occupancy_numba(state, level, old_row, old_col, new_row, new_col, keep)
    mobs.position[slot, 0] = new_row
    mobs.position[slot, 1] = new_col
    mobs.attack_cooldown[slot] = new_cooldown
    mobs.mask[slot] = 1 if keep else 0


@jit
def move_passive_slot_numba(
    state: EnvState,
    level: int,
    slot: int,
    rng: Array1[np.uint32],
) -> None:
    """Move one passive slot one random step in eight, half of which stay put.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      slot: The slot to move.
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.

    """
    mobs = state.passive_mobs[level]
    alive = mobs.mask[slot] != 0
    old_row = mobs.position[slot, 0]
    old_col = mobs.position[slot, 1]
    type_id = mobs.type_id[slot]
    dr, dc = choose_direction_numba(rng, 8)
    proposed_row = old_row + dr
    proposed_col = old_col + dc
    valid = valid_typed_mob_position_numba(
        state, level, MobType.PASSIVE, type_id, proposed_row, proposed_col, old_row, old_col,
    )  # fmt: skip
    new_row = proposed_row if valid else old_row
    new_col = proposed_col if valid else old_col
    dist = abs(state.player_position[0] - old_row) + abs(
        state.player_position[1] - old_col,
    )
    keep = alive and dist < MOB_DESPAWN_DISTANCE
    move_mob_occupancy_numba(state, level, old_row, old_col, new_row, new_col, keep)
    mobs.position[slot, 0] = new_row
    mobs.position[slot, 1] = new_col
    mobs.mask[slot] = 1 if keep else 0


@jit
def move_ranged_slot_numba(
    state: EnvState,
    level: int,
    slot: int,
    rng: Array1[np.uint32],
) -> None:
    """Move one ranged slot: keep its distance, shoot when in range, despawn when far.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      slot: The slot to move.
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.

    """
    mobs = state.ranged_mobs[level]
    alive = mobs.mask[slot] != 0
    old_row = mobs.position[slot, 0]
    old_col = mobs.position[slot, 1]
    type_id = mobs.type_id[slot]
    cooldown = mobs.attack_cooldown[slot]

    random_dr, random_dc = choose_direction_numba(rng, 4)
    distance_row = abs(state.player_position[0] - old_row)
    distance_col = abs(state.player_position[1] - old_col)
    axis = choose_player_axis_numba(rng, distance_row, distance_col)
    player_dr = 0
    player_dc = 0
    if axis == 0:
        dr = state.player_position[0] - old_row
        player_dr = int(dr > 0) - int(dr < 0)
    else:
        dc = state.player_position[1] - old_col
        player_dc = int(dc > 0) - int(dc < 0)
    dist = distance_row + distance_col
    proposed_row = old_row + player_dr if dist >= 6 else old_row + random_dr
    proposed_col = old_col + player_dc if dist >= 6 else old_col + random_dc
    if dist <= 3:
        proposed_row = old_row - player_dr
        proposed_col = old_col - player_dc
    if rng_f32_numba(rng) <= np.float32(0.85):
        proposed_row = old_row + random_dr
        proposed_col = old_col + random_dc
    valid = valid_typed_mob_position_numba(
        state, level, MobType.RANGED, type_id, proposed_row, proposed_col, old_row, old_col,
    )  # fmt: skip
    attacking = (
        ((4 <= dist <= 5) or (dist <= 3 and not valid)) and cooldown <= 0 and alive
    )
    if attacking:
        spawn_projectile_numba(
            state, False, RANGED_PROJECTILES[clampi_numba(type_id, 0, 7)], old_row, old_col, player_dr, player_dc,
        )  # fmt: skip
        proposed_row = old_row
        proposed_col = old_col
    new_cooldown = 4 if attacking else cooldown - 1
    valid = valid_typed_mob_position_numba(
        state, level, MobType.RANGED, type_id, proposed_row, proposed_col, old_row, old_col,
    )  # fmt: skip
    new_row = proposed_row if valid else old_row
    new_col = proposed_col if valid else old_col
    keep = alive and (dist < MOB_DESPAWN_DISTANCE or fighting_boss_numba(state))
    move_mob_occupancy_numba(state, level, old_row, old_col, new_row, new_col, keep)
    mobs.position[slot, 0] = new_row
    mobs.position[slot, 1] = new_col
    mobs.attack_cooldown[slot] = new_cooldown
    mobs.mask[slot] = 1 if keep else 0


@jit
def update_projectiles_numba(state: EnvState, from_player: bool) -> None:
    """Fly every projectile of one side one tile, striking what it meets.

    Args:
      state: The environment's world, mutated in place.
      from_player: The player's projectiles, else the creatures'.

    """
    level = state.player_level
    if from_player:
        for i in range(MAX_PLAYER_PROJECTILES):
            _fly_player_projectile_numba(state, level, i)
    else:
        for i in range(MAX_MOB_PROJECTILES):
            _fly_mob_projectile_numba(state, level, i)


@jit
def _fly_player_projectile_numba(state: EnvState, level: int, i: int) -> None:
    """One player projectile: damage what it is on and what it moves into."""
    projectiles = state.player_projectiles[level]
    if not projectiles.mask[i]:
        return
    old_row = projectiles.position[i, 0]
    old_col = projectiles.position[i, 1]
    proposed_row = old_row + state.player_projectile_directions[level, i, 0]
    proposed_col = old_col + state.player_projectile_directions[level, i, 1]
    ptype = projectiles.type_id[i]
    physical, fire, ice = mob_damage_vector_numba(ptype, MobType.PROJECTILE)
    arrow = ptype in (ProjectileType.ARROW, ProjectileType.ARROW2)
    if arrow and state.bow_enchantment == 1:
        fire += physical * np.float32(0.5)
    if arrow and state.bow_enchantment == 2:
        ice += physical * np.float32(0.5)
    coeff = np.float32(1.0)
    if arrow:
        coeff = np.float32(1.0) + np.float32(0.2) * np.float32(
            state.player_dexterity - 1,
        )
    elif ptype in (ProjectileType.FIREBALL, ProjectileType.ICEBALL):
        coeff = np.float32(1.0) + np.float32(0.5) * np.float32(
            state.player_intelligence - 1,
        )
    physical *= coeff
    fire *= coeff
    ice *= coeff

    hit_old = False
    mob_class, mob_slot = find_mob_at_numba(state, level, old_row, old_col)
    if mob_slot >= 0:
        target = mobs_for_class_numba(state, level, mob_class)
        hit_old = damage_mob_at_numba(
            state, level, old_row, old_col,
            damage_to_mob_numba(physical, fire, ice, target.type_id[mob_slot], mob_class),
            False, True,
        )  # fmt: skip

    second_physical = np.float32(0.0) if hit_old else physical
    second_fire = np.float32(0.0) if hit_old else fire
    second_ice = np.float32(0.0) if hit_old else ice
    hit_new = False
    mob_class, mob_slot = find_mob_at_numba(state, level, proposed_row, proposed_col)
    if mob_slot >= 0:
        target = mobs_for_class_numba(state, level, mob_class)
        hit_new = damage_mob_at_numba(
            state, level, proposed_row, proposed_col,
            damage_to_mob_numba(second_physical, second_fire, second_ice, target.type_id[mob_slot], mob_class),
            False, True,
        )  # fmt: skip

    proposed_in_bounds = 0 <= proposed_row < MAP_SIZE and 0 <= proposed_col < MAP_SIZE
    proposed_block = (
        state.map[level, proposed_row, proposed_col] if proposed_in_bounds else 0
    )
    in_wall = is_solid_block_numba(proposed_block) and proposed_block != BlockType.WATER
    keep = proposed_in_bounds and not in_wall and not hit_old and not hit_new
    projectiles.position[i, 0] = proposed_row
    projectiles.position[i, 1] = proposed_col
    projectiles.mask[i] = 1 if keep else 0


@jit
def _fly_mob_projectile_numba(state: EnvState, level: int, i: int) -> None:
    """One creature projectile: hurt the player it reaches, else fly on or stop."""
    projectiles = state.mob_projectiles[level]
    if not projectiles.mask[i]:
        return
    old_row = projectiles.position[i, 0]
    old_col = projectiles.position[i, 1]
    proposed_row = old_row + state.mob_projectile_dirs[level, i, 0]
    proposed_col = old_col + state.mob_projectile_dirs[level, i, 1]
    proposed_in_player = (
        proposed_row == state.player_position[0]
        and proposed_col == state.player_position[1]
    )
    proposed_in_bounds = 0 <= proposed_row < MAP_SIZE and 0 <= proposed_col < MAP_SIZE
    proposed_block = (
        state.map[level, proposed_row, proposed_col] if proposed_in_bounds else 0
    )
    in_wall = is_solid_block_numba(proposed_block) and proposed_block != BlockType.WATER
    in_mob = mob_at_numba(state, level, proposed_row, proposed_col) or (
        state.player_position[0] == proposed_row
        and state.player_position[1] == proposed_col
    )
    keep_moving = proposed_in_bounds and not in_wall and not in_mob
    hit_player = (
        old_row == state.player_position[0] and old_col == state.player_position[1]
    ) or proposed_in_player
    keep_moving = keep_moving and not hit_player
    hit_bench = proposed_block in (BlockType.FURNACE, BlockType.CRAFTING_TABLE)
    new_block = BlockType.PATH if hit_bench else proposed_block

    projectiles.position[i, 0] = proposed_row
    projectiles.position[i, 1] = proposed_col
    projectiles.mask[i] = 1 if keep_moving else 0
    if hit_player:
        physical, fire, ice = mob_damage_vector_numba(
            projectiles.type_id[i],
            MobType.PROJECTILE,
        )
        state.player_health -= damage_to_player_numba(state, physical, fire, ice)
        state.is_sleeping = 0
        state.is_resting = 0
    if proposed_in_bounds:
        set_block_numba(state, level, proposed_row, proposed_col, new_block)
