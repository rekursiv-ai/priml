"""What the player sees: 843 floats and the 43-entry action mask.

The observation is a 9x11 window of tiles around the player, 8 channels per
tile -- the block, the item plus one, whether the tile is lit, and one slot
per creature class holding the species plus one -- laid out tile-major, then
51 scalars for the inventory, the meters, the facing, the armour, the light
and a few flags. Dark tiles stay zero, which is how the fog of a cave works.

Every scalar is written as the C computes it in float32: square roots of
counts over ten, meters over ten, flags as 0 or 1. The views are written one
environment and one tile at a time: the same packing in compiled torch took
1,952 ns per environment against the C's 218 (the package docstring). The policy's embedding bag
reads the first 792 values as integer ids, so they must be exact integers.

The mask marks every action that would do something, so the sampler never
wastes a step on a recipe the player cannot afford.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h``, pin
        ``6ffa5b10``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from priml.baselines.craftax.game.jit import clampi_numba, jit, sqrtf, unliteral
from priml.baselines.craftax.game.rules import (
    boss_vulnerable_numba,
    max_health_numba,
    max_need_numba,
)
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    MAP_SIZE,
    MAX_ATTRIBUTE,
    MAX_MELEE_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PASSIVE_MOBS,
    MAX_PLAYER_PROJECTILES,
    MAX_RANGED_MOBS,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_LEVELS,
    NUM_MOB_TYPES,
    NUM_POTIONS,
    OBS_COLS,
    OBS_ROWS,
    OBS_TILE_CHANNELS,
    SYMBOLIC_TILE_CHANNELS,
    VISIBLE_LIGHT_THRESHOLD,
    Action,
    ItemType,
)


if TYPE_CHECKING:
    from priml.baselines.craftax.game.state import Array1, EnvState, Mobs


@jit
def compute_observations_numba(
    state: EnvState,
    obs: Array1[np.float32],
    mask: Array1[int],
) -> None:
    """Write the 843-float observation and the 43-byte action mask for one environment.

    Args:
      state: The world after its step.
      obs: float32 ``[843]``, overwritten.
      mask: uint8 ``[43]``, overwritten.

    """
    map_obs = OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS
    for i in range(map_obs):
        obs[i] = np.float32(0.0)

    level = state.player_level
    row = state.player_position[0]
    col = state.player_position[1]
    row_radius = OBS_ROWS // 2
    col_radius = OBS_COLS // 2
    r0 = clampi_numba(-row, -row_radius, row_radius)
    r1 = clampi_numba(MAP_SIZE - 1 - row, -row_radius, row_radius)
    c0 = clampi_numba(-col, -col_radius, col_radius)
    c1 = clampi_numba(MAP_SIZE - 1 - col, -col_radius, col_radius)

    # Tile by tile rather than a vectorized window copy: see the module docstring.
    for r in range(r0, r1 + 1):
        obs_row = row + r
        tile = ((r + row_radius) * OBS_COLS + (c0 + col_radius)) * OBS_TILE_CHANNELS
        for c in range(c0, c1 + 1):
            obs_col = col + c
            if state.light_map[level, obs_row, obs_col] > VISIBLE_LIGHT_THRESHOLD:
                obs[tile] = np.float32(state.map[level, obs_row, obs_col])
                obs[tile + 1] = np.float32(state.item_map[level, obs_row, obs_col] + 1)
                obs[tile + 2] = np.float32(1.0)
            tile += OBS_TILE_CHANNELS

    _write_mob_obs_numba(
        obs,
        state,
        state.melee_mobs[level],
        unliteral(MAX_MELEE_MOBS),
        0,
    )
    _write_mob_obs_numba(
        obs,
        state,
        state.passive_mobs[level],
        unliteral(MAX_PASSIVE_MOBS),
        1,
    )
    _write_mob_obs_numba(
        obs,
        state,
        state.ranged_mobs[level],
        unliteral(MAX_RANGED_MOBS),
        2,
    )
    _write_mob_obs_numba(
        obs,
        state,
        state.mob_projectiles[level],
        unliteral(MAX_MOB_PROJECTILES),
        3,
    )
    _write_mob_obs_numba(
        obs,
        state,
        state.player_projectiles[level],
        unliteral(MAX_PLAYER_PROJECTILES),
        4,
    )
    write_scalars_numba(state, obs, map_obs)
    compute_action_mask_numba(state, mask)


@jit
def compute_symbolic_observations_numba(
    state: EnvState,
    obs: Array1[np.float32],
    mask: Array1[int],
) -> None:
    """Write original Craftax's 8,268-float observation and the action mask.

        Craftax-Symbolic-v1's layout (``renderer.py:9-198``): per tile of the
        9x11 view, one-hot the block (37), the item (5) and each creature's
        (class, species) pair (5 x 8), then whether the tile is lit; a dark or
        off-map tile stays all zero. The 51 scalars follow, the same as
        :func:`compute_observations_numba` writes. A creature marks its pair when
        it is live, in view and lit; the C marks the same ones, and JAX also
        clears a pair where an earlier slot set it (D9 of
        ``docs/differences.md``), which this does not.

    Args:
      state: The world after its step.
      obs: float32 ``[8268]``, overwritten.
      mask: uint8 ``[43]``, overwritten.

    """
    channels = unliteral(SYMBOLIC_TILE_CHANNELS)
    map_obs = OBS_ROWS * OBS_COLS * channels
    for i in range(map_obs):
        obs[i] = np.float32(0.0)
    item_channel = unliteral(NUM_BLOCK_TYPES)
    visible_channel = channels - 1
    level = state.player_level
    row = state.player_position[0]
    col = state.player_position[1]
    row_radius = OBS_ROWS // 2
    col_radius = OBS_COLS // 2
    for r in range(
        clampi_numba(-row, -row_radius, row_radius),
        clampi_numba(MAP_SIZE - 1 - row, -row_radius, row_radius) + 1,
    ):
        for c in range(
            clampi_numba(-col, -col_radius, col_radius),
            clampi_numba(MAP_SIZE - 1 - col, -col_radius, col_radius) + 1,
        ):
            if state.light_map[level, row + r, col + c] > VISIBLE_LIGHT_THRESHOLD:
                tile = ((r + row_radius) * OBS_COLS + c + col_radius) * channels
                obs[tile + state.map[level, row + r, col + c]] = np.float32(1.0)
                obs[tile + item_channel + state.item_map[level, row + r, col + c]] = (
                    np.float32(1.0)
                )
                obs[tile + visible_channel] = np.float32(1.0)
    _write_symbolic_mob_obs_numba(
        obs,
        state,
        state.melee_mobs[level],
        unliteral(MAX_MELEE_MOBS),
        0,
    )
    _write_symbolic_mob_obs_numba(
        obs,
        state,
        state.passive_mobs[level],
        unliteral(MAX_PASSIVE_MOBS),
        1,
    )
    _write_symbolic_mob_obs_numba(
        obs,
        state,
        state.ranged_mobs[level],
        unliteral(MAX_RANGED_MOBS),
        2,
    )
    _write_symbolic_mob_obs_numba(
        obs,
        state,
        state.mob_projectiles[level],
        unliteral(MAX_MOB_PROJECTILES),
        3,
    )
    _write_symbolic_mob_obs_numba(
        obs,
        state,
        state.player_projectiles[level],
        unliteral(MAX_PLAYER_PROJECTILES),
        4,
    )
    write_scalars_numba(state, obs, map_obs)
    compute_action_mask_numba(state, mask)


@jit
def write_scalars_numba(state: EnvState, obs: Array1[np.float32], start: int) -> None:
    """Write the 51 scalars both layouts end with, from ``obs[start]`` on.

    Inventory, potions, meters, facing, armour, enchantments, then light,
    sleep, rest, spells, floor, the floor-cleared flag and the boss flag,
    each in float32 as the C computes it.

    Args:
      state: The world after its step.
      obs: The observation row.
      start: Where the scalars begin: 792 packed, 8,217 symbolic.

    """
    level = state.player_level
    inv = state.inventory
    ten = np.float32(10.0)
    i = start
    obs[i] = sqrtf(np.float32(inv.wood)) / ten
    obs[i + 1] = sqrtf(np.float32(inv.stone)) / ten
    obs[i + 2] = sqrtf(np.float32(inv.coal)) / ten
    obs[i + 3] = sqrtf(np.float32(inv.iron)) / ten
    obs[i + 4] = sqrtf(np.float32(inv.diamond)) / ten
    obs[i + 5] = sqrtf(np.float32(inv.sapphire)) / ten
    obs[i + 6] = sqrtf(np.float32(inv.ruby)) / ten
    obs[i + 7] = sqrtf(np.float32(inv.sapling)) / ten
    obs[i + 8] = sqrtf(np.float32(inv.torches)) / ten
    obs[i + 9] = sqrtf(np.float32(inv.arrows)) / ten
    obs[i + 10] = np.float32(inv.books) / np.float32(2.0)
    obs[i + 11] = np.float32(inv.pickaxe) / np.float32(4.0)
    obs[i + 12] = np.float32(inv.sword) / np.float32(4.0)
    obs[i + 13] = np.float32(state.sword_enchantment)
    obs[i + 14] = np.float32(state.bow_enchantment)
    obs[i + 15] = np.float32(inv.bow)
    i += 16
    for potion in range(NUM_POTIONS):
        obs[i + potion] = sqrtf(np.float32(inv.potions[potion])) / ten
    i += NUM_POTIONS

    obs[i] = state.player_health / ten
    obs[i + 1] = np.float32(state.player_food) / ten
    obs[i + 2] = np.float32(state.player_drink) / ten
    obs[i + 3] = np.float32(state.player_energy) / ten
    obs[i + 4] = np.float32(state.player_mana) / ten
    obs[i + 5] = np.float32(state.player_xp) / ten
    obs[i + 6] = np.float32(state.player_dexterity) / ten
    obs[i + 7] = np.float32(state.player_strength) / ten
    obs[i + 8] = np.float32(state.player_intelligence) / ten
    i += 9

    direction_index = state.player_direction - Action.LEFT
    for d in range(4):
        obs[i + d] = np.float32(1.0) if d == direction_index else np.float32(0.0)
    i += 4
    for a in range(4):
        obs[i + a] = np.float32(inv.armour[a]) / np.float32(2.0)
    i += 4
    for a in range(4):
        obs[i + a] = np.float32(state.armour_enchantments[a])
    i += 4

    obs[i] = state.light_level
    obs[i + 1] = np.float32(1.0) if state.is_sleeping else np.float32(0.0)
    obs[i + 2] = np.float32(1.0) if state.is_resting else np.float32(0.0)
    obs[i + 3] = np.float32(1.0) if state.learned_spells[0] else np.float32(0.0)
    obs[i + 4] = np.float32(1.0) if state.learned_spells[1] else np.float32(0.0)
    obs[i + 5] = np.float32(state.player_level) / ten
    obs[i + 6] = (
        np.float32(1.0)
        if state.monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        else np.float32(0.0)
    )
    obs[i + 7] = np.float32(1.0) if boss_vulnerable_numba(state) else np.float32(0.0)


@jit
def compute_action_mask_numba(state: EnvState, mask: Array1[int]) -> None:
    """Mark the actions that would do something now (``compute_action_mask``).

    Args:
      state: The environment's world, mutated in place.
      mask: uint8 ``[43]``, overwritten.

    """
    inv = state.inventory
    for a in range(ATN_DIM):
        mask[a] = 0
    mask[Action.NOOP.value] = 1
    if state.is_sleeping or state.is_resting:
        return
    mask[Action.LEFT.value] = 1
    mask[Action.RIGHT.value] = 1
    mask[Action.UP.value] = 1
    mask[Action.DOWN.value] = 1
    mask[Action.DO.value] = 1
    mask[Action.SLEEP.value] = state.player_energy < max_need_numba(state)
    mask[Action.REST.value] = state.player_health < np.float32(max_health_numba(state))
    mask[Action.PLACE_STONE.value] = inv.stone > 0
    mask[Action.PLACE_FURNACE.value] = inv.stone > 0
    mask[Action.PLACE_TABLE.value] = inv.wood >= 2
    mask[Action.PLACE_PLANT.value] = inv.sapling > 0
    mask[Action.PLACE_TORCH.value] = inv.torches > 0
    mask[Action.MAKE_WOOD_PICKAXE.value] = inv.wood > 0 and inv.pickaxe < 1
    mask[Action.MAKE_STONE_PICKAXE.value] = (
        inv.wood > 0 and inv.stone > 0 and inv.pickaxe < 2
    )
    mask[Action.MAKE_IRON_PICKAXE.value] = (
        inv.wood > 0
        and inv.stone > 0
        and inv.iron > 0
        and inv.coal > 0
        and inv.pickaxe < 3
    )
    mask[Action.MAKE_DIAMOND_PICKAXE.value] = (
        inv.wood > 0 and inv.diamond >= 3 and inv.pickaxe < 4
    )
    mask[Action.MAKE_WOOD_SWORD.value] = inv.wood > 0 and inv.sword < 1
    mask[Action.MAKE_STONE_SWORD.value] = (
        inv.wood > 0 and inv.stone > 0 and inv.sword < 2
    )
    mask[Action.MAKE_IRON_SWORD.value] = (
        inv.wood > 0
        and inv.stone > 0
        and inv.iron > 0
        and inv.coal > 0
        and inv.sword < 3
    )
    mask[Action.MAKE_DIAMOND_SWORD.value] = (
        inv.wood > 0 and inv.diamond >= 2 and inv.sword < 4
    )
    mask[Action.MAKE_ARROW.value] = inv.wood > 0 and inv.stone > 0 and inv.arrows < 99
    mask[Action.MAKE_TORCH.value] = inv.wood > 0 and inv.coal > 0 and inv.torches < 99
    missing_iron = 0
    missing_diamond = 0
    armour = 0
    for k in range(4):
        missing_iron += 1 if inv.armour[k] < 1 else 0
        missing_diamond += 1 if inv.armour[k] < 2 else 0
        armour += inv.armour[k]
    mask[Action.MAKE_IRON_ARMOUR.value] = (
        missing_iron != 0 and inv.iron >= 3 and inv.coal >= 3
    )
    mask[Action.MAKE_DIAMOND_ARMOUR.value] = missing_diamond != 0 and inv.diamond >= 3
    item = state.item_map[
        state.player_level,
        state.player_position[0],
        state.player_position[1],
    ]
    mask[Action.DESCEND.value] = (
        item == ItemType.LADDER_DOWN
        and state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        and state.player_level < NUM_LEVELS - 1
    )
    mask[Action.ASCEND.value] = item == ItemType.LADDER_UP and state.player_level > 0
    mask[Action.SHOOT_ARROW.value] = inv.bow > 0 and inv.arrows > 0
    mask[Action.CAST_FIREBALL.value] = (
        state.learned_spells[0] != 0 and state.player_mana >= 2
    )
    mask[Action.CAST_ICEBALL.value] = (
        state.learned_spells[1] != 0 and state.player_mana >= 2
    )
    for k in range(NUM_POTIONS):
        mask[Action.DRINK_POTION_RED.value + k] = inv.potions[k] > 0
    mask[Action.READ_BOOK.value] = inv.books > 0
    enchant = state.player_mana >= 9 and (inv.ruby > 0 or inv.sapphire > 0)
    mask[Action.ENCHANT_SWORD.value] = enchant and inv.sword > 0
    mask[Action.ENCHANT_ARMOUR.value] = enchant and armour > 0
    mask[Action.ENCHANT_BOW.value] = enchant and inv.bow > 0
    mask[Action.LEVEL_UP_DEXTERITY.value] = (
        state.player_xp > 0 and state.player_dexterity < MAX_ATTRIBUTE
    )
    mask[Action.LEVEL_UP_STRENGTH.value] = (
        state.player_xp > 0 and state.player_strength < MAX_ATTRIBUTE
    )
    mask[Action.LEVEL_UP_INTELLIGENCE.value] = (
        state.player_xp > 0 and state.player_intelligence < MAX_ATTRIBUTE
    )


@jit
def _write_mob_obs_numba(
    obs: Array1[np.float32],
    state: EnvState,
    mobs: Mobs,
    slots: int,
    channel: int,
) -> None:
    """Write ``species + 1`` for each live, lit, in-window slot into its channel."""
    for i in range(slots):
        tile = _visible_tile_numba(state, mobs, i)
        if tile >= 0:
            obs[tile * OBS_TILE_CHANNELS + 3 + channel] = np.float32(
                mobs.type_id[i] + 1,
            )


@jit
def _write_symbolic_mob_obs_numba(
    obs: Array1[np.float32],
    state: EnvState,
    mobs: Mobs,
    slots: int,
    mob_class: int,
) -> None:
    """Set the (class, species) one-hot bit of each live, lit, in-window slot."""
    for i in range(slots):
        tile = _visible_tile_numba(state, mobs, i)
        if tile >= 0:
            obs[
                tile * SYMBOLIC_TILE_CHANNELS
                + NUM_BLOCK_TYPES
                + NUM_ITEM_TYPES
                + mob_class * NUM_MOB_TYPES
                + mobs.type_id[i]
            ] = np.float32(1.0)


@jit
def _visible_tile_numba(state: EnvState, mobs: Mobs, i: int) -> int:
    """Return slot ``i``'s tile in the view (row-major), or -1 unless live, in view and lit."""
    if not mobs.mask[i]:
        return -1
    world_row = mobs.position[i, 0]
    world_col = mobs.position[i, 1]
    local_row = world_row - state.player_position[0] + OBS_ROWS // 2
    local_col = world_col - state.player_position[1] + OBS_COLS // 2
    if local_row < 0 or local_row >= OBS_ROWS or local_col < 0 or local_col >= OBS_COLS:
        return -1
    if (
        world_row < 0
        or world_row >= MAP_SIZE
        or world_col < 0
        or world_col >= MAP_SIZE
        or state.light_map[state.player_level, world_row, world_col]
        <= VISIBLE_LIGHT_THRESHOLD
    ):
        return -1
    return int(local_row * OBS_COLS + local_col)
