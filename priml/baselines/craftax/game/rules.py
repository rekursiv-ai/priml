"""The rules a tick plays: the shared mechanics, and the phases built from them.

The shared mechanics are the pieces used in more than one place: how a block
is written so the spawn bitsets stay in step with the map, what the player's
meters cap at, whether a tile can be stood on, where a creature is, how hard a
hit lands and how a creature is struck. Keeping them together means a phase
states what it does rather than restating the arithmetic, and each rule is
checkable on its own. The C ``Damage`` struct is three float32 values here --
physical, fire, ice -- passed and returned as a tuple. Every phase is scalar
Numba code on one environment's record, and its ``np.float32`` casts keep the
C's fp32 arithmetic (the package docstring says why).

The phases follow, in the order a tick runs them:

- :func:`craft_numba` turns materials into tools, armour, arrows and torches when the
  right workbench is within one tile, and :func:`place_numba` puts a block, torch or
  sapling on the tile the player faces. Both are ``if / else if`` chains in the
  C, so at most one recipe fires per step and the order of the tests is part
  of the rules.
- :func:`interact_numba` is the ``DO`` action on the faced tile: a creature there is
  struck first, and only if nothing was struck does the block itself respond
  -- trees give wood, ores give ore to a good enough pickaxe, water quenches, a
  ripe plant feeds, a chest spills loot, and the necromancer takes a hit when
  vulnerable. The chest is the one place the step draws from the stream in
  bulk: fifteen draws in a fixed order, several of them for loot that is then
  discarded, so the order and count are part of the contract. A potion id of 6
  -- possible when ``rng_f32_numba`` returns 1.0 -- indexes past ``potions[]`` in the
  C and lands on ``books``, which this does explicitly.
- :func:`abilities_numba` gathers everything the player does with what they carry
  rather than with the world: shooting an arrow or casting a spell in the faced
  direction, drinking a potion whose effect this episode's shuffle hides,
  reading a book to learn a spell, enchanting gear at an elemental table, and
  spending experience on an attribute. :func:`grow_plants_numba` lets sown plants
  ripen, after the creatures have moved.
- :func:`move_player_numba` walks one tile if it is passable and turns the player to
  face the way they tried to go. :func:`tick_numba` closes every tick: it starts or
  ends sleep and rest, advances hunger, thirst, fatigue, recovery and mana in
  float32, clamps the inventory and the meters, awards the achievements that
  follow from what the player now holds, advances the clock and the
  :func:`daylight_numba`, and says whether the episode ended.

Sleep is what makes a step take more than one tick: while the player sleeps
or rests, the step repeats the phases with a no-op action until they wake or
die, so credit assignment sees the wake, the hit or the death.

Every function takes one environment's ``EnvState`` record and mutates it in
place, as the C takes a ``State*``.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h``, pin
        ``6ffa5b10``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

from numba.extending import intrinsic

import numba.core.types as nbtypes
import numpy as np

from priml.baselines.craftax.game.jit import (
    clampf_numba,
    clampi_numba,
    cosf,
    fmodf,
    jit,
    powf,
    sqrtf,
    unliteral,
)
from priml.baselines.craftax.game.rng import (
    choose_weighted_numba,
    choose_weighted_pair_numba,
    rng_f32_numba,
    rng_int_numba,
)
from priml.baselines.craftax.game.state import (
    BOSS_SPAWN_TURNS,
    DAY_LENGTH,
    DEFAULT_MAX_TIMESTEPS,
    MAP_SIZE,
    MAX_ATTRIBUTE,
    MAX_GROWING_PLANTS,
    MAX_MELEE_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PASSIVE_MOBS,
    MAX_PLAYER_PROJECTILES,
    MAX_RANGED_MOBS,
    NUM_BLOCK_TYPES,
    NUM_LEVELS,
    NUM_POTIONS,
    PI,
    Achievement,
    Action,
    BlockType,
    ItemType,
    MobType,
    ProjectileType,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from llvmlite.ir import IRBuilder, Value
    from numba.core.typing.templates import Signature

    from priml.baselines.craftax.game.state import (
        Array1,
        Array2,
        Array3,
        EnvState,
        Mobs,
    )


def block_table(*blocks: BlockType) -> Array1[bool]:
    """Return a ``bool [NUM_BLOCK_TYPES]`` table set at ``blocks``."""
    table = np.zeros(NUM_BLOCK_TYPES, dtype=np.bool_)
    table[list(blocks)] = True
    return cast("Array1[bool]", table)


LAND_BLOCKS: Final = block_table(
    BlockType.GRASS,
    BlockType.PATH,
    BlockType.FIRE_GRASS,
    BlockType.ICE_GRASS,
)
"""Blocks a land mob may spawn on (``refresh_spawn_cell_numba``)."""

GRAVE_BLOCKS: Final = block_table(BlockType.GRAVE, BlockType.GRAVE2, BlockType.GRAVE3)
"""Blocks the boss's minions rise from."""

_SOLID: Final = (
    BlockType.STONE,
    BlockType.TREE,
    BlockType.COAL,
    BlockType.IRON,
    BlockType.DIAMOND,
    BlockType.CRAFTING_TABLE,
    BlockType.FURNACE,
    BlockType.PLANT,
    BlockType.RIPE_PLANT,
    BlockType.WALL,
    BlockType.WALL_MOSS,
    BlockType.STALAGMITE,
    BlockType.RUBY,
    BlockType.SAPPHIRE,
    BlockType.CHEST,
    BlockType.FOUNTAIN,
    BlockType.FIRE_TREE,
    BlockType.ENCHANTMENT_TABLE_FIRE,
    BlockType.ENCHANTMENT_TABLE_ICE,
    BlockType.GRAVE,
    BlockType.GRAVE2,
    BlockType.GRAVE3,
    BlockType.NECROMANCER,
)

SOLID_BLOCKS: Final = block_table(*_SOLID)
"""``is_solid_block_numba``: blocks neither the player nor a creature may enter."""

PLAYER_BLOCKED: Final = block_table(*_SOLID, BlockType.WATER, BlockType.LAVA)
"""Blocks the player cannot walk onto: the solid ones plus water and lava."""

MOB_BLOCKED: Final = cast(
    "Array3[int]",
    np.array(
        [
            [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
            [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
            [[0, 1, 1], [0, 0, 1], [0, 1, 1]],
            [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
            [[0, 1, 1], [0, 1, 1], [1, 0, 1]],
            [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
            [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
        ],
        dtype=np.uint8,
    ),
)
"""``[type_id, mob_class, terrain]``: whether ground (0), water (1) or lava (2) stops it."""

MOB_DAMAGE: Final = cast(
    "Array3[np.float32]",
    np.array(
        [
            [[0, 0, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0]],
            [[0, 0, 0], [4, 0, 0], [0, 0, 0], [4, 0, 0]],
            [[0, 0, 0], [3, 0, 0], [0, 0, 0], [0, 3, 0]],
            [[0, 0, 0], [5, 0, 0], [0, 0, 0], [0, 0, 3]],
            [[0, 0, 0], [6, 0, 0], [0, 0, 0], [5, 0, 0]],
            [[0, 0, 0], [6, 1, 1], [0, 0, 0], [4, 3, 3]],
            [[0, 0, 0], [3, 5, 0], [0, 0, 0], [3, 5, 0]],
            [[0, 0, 0], [4, 0, 5], [0, 0, 0], [4, 0, 5]],
        ],
        dtype=np.float32,
    ),
)
"""``[type_id, mob_class, (physical, fire, ice)]``: what a creature's hit carries."""

MOB_DEFENSE: Final = cast(
    "Array3[np.float32]",
    np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0.2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0.9, 1, 0], [0.9, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0.9, 0, 1], [0.9, 0, 1], [0, 0, 0]],
        ],
        dtype=np.float32,
    ),
)
"""``[type_id, mob_class, (physical, fire, ice)]``: the fraction a creature shrugs off."""

KILL_ACHIEVEMENTS: Final = cast(
    "Array2[int]",
    np.array(
        [
            [
                Achievement.EAT_COW,
                Achievement.EAT_BAT,
                Achievement.EAT_SNAIL,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                Achievement.DEFEAT_ZOMBIE,
                Achievement.DEFEAT_GNOME_WARRIOR,
                Achievement.DEFEAT_ORC_SOLIDER,
                Achievement.DEFEAT_LIZARD,
                Achievement.DEFEAT_KNIGHT,
                Achievement.DEFEAT_TROLL,
                Achievement.DEFEAT_PIGMAN,
                Achievement.DEFEAT_FROST_TROLL,
            ],
            [
                Achievement.DEFEAT_SKELETON,
                Achievement.DEFEAT_GNOME_ARCHER,
                Achievement.DEFEAT_ORC_MAGE,
                Achievement.DEFEAT_KOBOLD,
                Achievement.DEFEAT_ARCHER,
                Achievement.DEFEAT_DEEP_THING,
                Achievement.DEFEAT_FIRE_ELEMENTAL,
                Achievement.DEFEAT_ICE_ELEMENTAL,
            ],
        ],
        dtype=np.int64,
    ),
)
"""``[mob_class, type_id]``: the achievement for killing (or eating) it."""


NEARBY_TILES: Final = cast(
    "Array2[int]",
    np.array(
        [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]],
        dtype=np.int64,
    ),
)
"""The eight neighbours a workbench may stand on, in the C's order."""

TORCH_GROUND: Final = block_table(
    BlockType.GRASS,
    BlockType.SAND,
    BlockType.PATH,
    BlockType.FIRE_GRASS,
    BlockType.ICE_GRASS,
)
"""Blocks a torch may be placed on."""


SWORD_DAMAGE: Final = cast(
    "Array1[np.float32]",
    np.array([1, 2, 3, 5, 8], dtype=np.float32),
)
"""Base melee damage by sword level."""

CHEST_ORE_WEIGHTS: Final = np.array([0.3, 0.3, 0.15, 0.125, 0.125], dtype=np.float32)
"""Coal, iron, diamond, sapphire, ruby."""

CHEST_TOOL_WEIGHTS: Final = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
"""Wood, stone, iron, diamond."""


@jit
def refresh_spawn_cell_numba(state: EnvState, level: int, row: int, col: int) -> None:
    """Recompute one cell's bits in the land, grave and water spawn bitsets.

        Spawning scans these bitsets rather than the map, so every block write
        goes through ``set_block_numba``, which calls this.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      row: Tile row.
      col: Tile column.

    """
    block = state.map[level, row, col]
    bit = np.uint64(1) << np.uint64(col)
    land = LAND_BLOCKS[block]
    grave = GRAVE_BLOCKS[block]
    water = block == BlockType.WATER
    state.spawn_land[level, row] = (state.spawn_land[level, row] & ~bit) | (
        bit if land else np.uint64(0)
    )
    state.spawn_grave[level, row] = (state.spawn_grave[level, row] & ~bit) | (
        bit if grave else np.uint64(0)
    )
    state.spawn_water[level, row] = (state.spawn_water[level, row] & ~bit) | (
        bit if water else np.uint64(0)
    )


@jit
def set_block_numba(
    state: EnvState,
    level: int,
    row: int,
    col: int,
    block: int,
) -> None:
    """Write ``block`` to the map and refresh the cell's spawn bits."""
    state.map[level, row, col] = block
    refresh_spawn_cell_numba(state, level, row, col)


@jit
def max_health_numba(state: EnvState) -> int:
    """Health cap, which strength raises."""
    return 8 + state.player_strength


@jit
def equipped_armour_numba(state: EnvState) -> int:
    """Sum of the four armour levels.

    Args:
      state: The environment's world, mutated in place.

    Returns:
      armour: The sum of the four armour levels.

    """
    return int(
        state.inventory.armour[0]
        + state.inventory.armour[1]
        + state.inventory.armour[2]
        + state.inventory.armour[3],
    )


@jit
def max_need_numba(state: EnvState) -> int:
    """Food, drink and energy cap, which dexterity raises."""
    return 7 + 2 * state.player_dexterity


@jit
def max_mana_numba(state: EnvState) -> int:
    """Mana cap, which intelligence raises."""
    return 6 + 3 * state.player_intelligence


@jit
def fighting_boss_numba(state: EnvState) -> bool:
    """Whether the player is on the last floor."""
    return state.player_level == NUM_LEVELS - 1


@jit
def boss_vulnerable_numba(state: EnvState) -> bool:
    """Whether the necromancer can be struck: no wave pending and no hostiles alive.

    Args:
      state: The environment's world, mutated in place.

    Returns:
      vulnerable: Whether the necromancer can be struck.

    """
    if state.boss_timestep_to_spawn_this_round > 0:
        return False
    level = state.player_level
    for i in range(MAX_MELEE_MOBS):
        if state.melee_mobs[level].mask[i]:
            return False
    for i in range(MAX_RANGED_MOBS):  # noqa: SIM110 -- Numba cannot compile all() over a generator (NotImplementedError, measured).
        if state.ranged_mobs[level].mask[i]:
            return False
    return True


@jit
def action_to_direction_numba(action: int) -> tuple[int, int]:
    """Return the ``(row, column)`` unit step a movement action names, else zeros.

    Args:
      action: An ``Action`` value.

    Returns:
      direction: ``(row, column)`` unit step, zeros for a non-movement action.

    """
    if action == Action.LEFT:
        return 0, -1
    if action == Action.RIGHT:
        return 0, 1
    if action == Action.UP:
        return -1, 0
    if action == Action.DOWN:
        return 1, 0
    return 0, 0


@jit
def is_solid_block_numba(block: int) -> bool:
    """Whether ``block`` stops movement."""
    return bool(SOLID_BLOCKS[block])


@jit
def mob_at_numba(state: EnvState, level: int, row: int, col: int) -> bool:
    """Whether a creature occupies the tile; false outside the map."""
    if row < 0 or row >= MAP_SIZE or col < 0 or col >= MAP_SIZE:
        return False
    return bool((state.mob_bits[level, row] >> np.uint64(col)) & np.uint64(1))


@jit
def set_mob_bit_numba(
    state: EnvState,
    level: int,
    row: int,
    col: int,
    on: bool,
) -> None:
    """Set or clear a tile's occupancy bit; a no-op outside the map.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      row: Tile row.
      col: Tile column.
      on: Whether to set or clear the bit.

    """
    if row < 0 or row >= MAP_SIZE or col < 0 or col >= MAP_SIZE:
        return
    bit = np.uint64(1) << np.uint64(col)
    if on:
        state.mob_bits[level, row] |= bit
    else:
        state.mob_bits[level, row] &= ~bit


@jit
def move_mob_occupancy_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured), and the C signature has this many.
    state: EnvState,
    level: int,
    old_row: int,
    old_col: int,
    new_row: int,
    new_col: int,
    keep: bool,
) -> None:
    """Clear the old tile's bit and, if the creature survives, set the new one's."""
    set_mob_bit_numba(state, level, old_row, old_col, False)
    if keep:
        set_mob_bit_numba(state, level, new_row, new_col, True)


@jit
def mobs_at_numba(mobs: Mobs, slots: int, row: int, col: int) -> int:
    """Return the first live slot of ``mobs`` at the tile, or -1."""
    for i in range(slots):
        if mobs.mask[i] and mobs.position[i, 0] == row and mobs.position[i, 1] == col:
            return i
    return -1


@jit
def mobs_for_class_numba(state: EnvState, level: int, mob_class: int) -> Mobs:
    """Return the slot table of a creature class on ``level``."""
    if mob_class == MobType.PASSIVE:
        return state.passive_mobs[level]
    if mob_class == MobType.RANGED:
        return state.ranged_mobs[level]
    return state.melee_mobs[level]


@jit
def find_mob_at_numba(
    state: EnvState,
    level: int,
    row: int,
    col: int,
) -> tuple[int, int]:
    """Return ``(mob_class, slot)`` of the creature at the tile, or ``(0, -1)``.

        Melee slots are searched first, then passive, then ranged, as the C does.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      row: Tile row.
      col: Tile column.

    Returns:
      mob_class: The creature class, 0 when none.
      slot: Its slot, -1 when none.

    """
    slot = mobs_at_numba(state.melee_mobs[level], unliteral(MAX_MELEE_MOBS), row, col)
    if slot >= 0:
        return MobType.MELEE, slot
    slot = mobs_at_numba(
        state.passive_mobs[level],
        unliteral(MAX_PASSIVE_MOBS),
        row,
        col,
    )
    if slot >= 0:
        return MobType.PASSIVE, slot
    slot = mobs_at_numba(state.ranged_mobs[level], unliteral(MAX_RANGED_MOBS), row, col)
    if slot >= 0:
        return MobType.RANGED, slot
    return 0, -1


@jit
def valid_typed_mob_position_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured), and the C signature has this many.
    state: EnvState,
    level: int,
    mob_class: int,
    type_id: int,
    row: int,
    col: int,
    old_row: int,
    old_col: int,
) -> bool:
    """Whether a creature of this type may stand on the tile.

        Off the map, under the player, in a solid block, on terrain its type
        cannot cross, or under another creature are all refused -- except that
        its own current tile is always allowed.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      mob_class: The creature class.
      type_id: Its species.
      row: Proposed row.
      col: Proposed column.
      old_row: Its current row.
      old_col: Its current column.

    Returns:
      valid: Whether it may stand there.

    """
    if row < 0 or row >= MAP_SIZE or col < 0 or col >= MAP_SIZE:
        return False
    if row == state.player_position[0] and col == state.player_position[1]:
        return False
    block = state.map[level, row, col]
    if is_solid_block_numba(block):
        return False
    terrain = 1 if block == BlockType.WATER else (2 if block == BlockType.LAVA else 0)
    if MOB_BLOCKED[clampi_numba(type_id, 0, 7), clampi_numba(mob_class, 0, 2), terrain]:
        return False
    return (not mob_at_numba(state, level, row, col)) or (
        row == old_row and col == old_col
    )


@jit
def mob_damage_vector_numba(
    type_id: int,
    mob_class: int,
) -> tuple[np.float32, np.float32, np.float32]:
    """Return the ``(physical, fire, ice)`` damage a creature or projectile deals.

    Args:
      type_id: The species.
      mob_class: The creature class, ``MobType.PROJECTILE`` for projectiles.

    Returns:
      damage: ``(physical, fire, ice)`` float32.

    """
    t = clampi_numba(type_id, 0, 7)
    c = clampi_numba(mob_class, 0, 3)
    return (
        np.float32(MOB_DAMAGE[t, c, 0]),
        np.float32(MOB_DAMAGE[t, c, 1]),
        np.float32(MOB_DAMAGE[t, c, 2]),
    )


@jit
def damage_to_mob_numba(
    physical: np.float32,
    fire: np.float32,
    ice: np.float32,
    type_id: int,
    mob_class: int,
) -> np.float32:
    """Return the damage a creature takes from a ``(physical, fire, ice)`` hit.

    Args:
      physical: Physical damage.
      fire: Fire damage.
      ice: Ice damage.
      type_id: The target's species.
      mob_class: The target's class.

    Returns:
      damage: The float32 damage taken.

    """
    t = clampi_numba(type_id, 0, 7)
    c = clampi_numba(mob_class, 0, 3)
    return np.float32(
        physical * (np.float32(1.0) - MOB_DEFENSE[t, c, 0])
        + fire * (np.float32(1.0) - MOB_DEFENSE[t, c, 1])
        + ice * (np.float32(1.0) - MOB_DEFENSE[t, c, 2]),
    )


@jit
def damage_to_player_numba(
    state: EnvState,
    physical: np.float32,
    fire: np.float32,
    ice: np.float32,
) -> np.float32:
    """Return the damage the player takes, after armour and its enchantments.

    Args:
      state: The environment's world, mutated in place.
      physical: Physical damage.
      fire: Fire damage.
      ice: Ice damage.

    Returns:
      damage: The float32 damage taken.

    """
    physical_defense = np.float32(0.0)
    fire_defense = np.float32(0.0)
    ice_defense = np.float32(0.0)
    for i in range(4):
        physical_defense += np.float32(0.1) * np.float32(state.inventory.armour[i])
        fire_defense += np.float32(0.2) * np.float32(state.armour_enchantments[i] == 1)
        ice_defense += np.float32(0.2) * np.float32(state.armour_enchantments[i] == 2)
    coeff = np.float32(1.5) if fighting_boss_numba(state) else np.float32(1.0)
    return np.float32(
        coeff
        * (
            physical * (np.float32(1.0) - physical_defense)
            + fire * (np.float32(1.0) - fire_defense)
            + ice * (np.float32(1.0) - ice_defense)
        ),
    )


@jit
def damage_mob_at_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured), and the C signature has this many.
    state: EnvState,
    level: int,
    row: int,
    col: int,
    damage: np.float32,
    can_eat: bool,
    can_get_achievement: bool,
) -> bool:
    """Strike whatever creature stands on the tile; return whether one was there.

        A kill frees the slot and its occupancy bit, counts toward clearing the
        floor unless it was passive, awards the kill achievement, and feeds the
        player if it was passive and edible.

    Args:
      state: The environment's world, mutated in place.
      level: The floor.
      row: Tile row.
      col: Tile column.
      damage: The float32 damage dealt.
      can_eat: Whether a passive kill feeds the player.
      can_get_achievement: Whether a kill awards its achievement.

    Returns:
      struck: Whether a creature stood on the tile.

    """
    mob_class, slot = find_mob_at_numba(state, level, row, col)
    if slot < 0:
        return False
    mobs = mobs_for_class_numba(state, level, mob_class)
    if not mobs.mask[slot]:
        return False

    mobs.health[slot] -= damage
    if mobs.health[slot] > np.float32(0.0):
        return True

    type_id = mobs.type_id[slot]
    mobs.mask[slot] = 0
    set_mob_bit_numba(state, level, row, col, False)
    state.monsters_killed[level] += 0 if mob_class == MobType.PASSIVE else 1
    if can_get_achievement:
        state.achievements[
            KILL_ACHIEVEMENTS[
                clampi_numba(mob_class, 0, 2),
                clampi_numba(type_id, 0, 7),
            ]
        ] = 1

    if mob_class == MobType.PASSIVE and can_eat:
        state.player_food = clampi_numba(
            state.player_food + 6,
            0,
            max_need_numba(state),
        )
        state.player_hunger = np.float32(0.0)
    return True


@jit
def spawn_projectile_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured), and the C signature has this many.
    state: EnvState,
    from_player: bool,
    projectile_type: int,
    row: int,
    col: int,
    dir_row: int,
    dir_col: int,
) -> bool:
    """Launch a projectile from the first free slot; return whether one was free.

        Its health is the sum of its damage vector, which is what a hit deals.

    Args:
      state: The environment's world, mutated in place.
      from_player: Whether the player fired it.
      projectile_type: A ``ProjectileType`` value.
      row: Starting row.
      col: Starting column.
      dir_row: Row step per tick.
      dir_col: Column step per tick.

    Returns:
      fired: Whether a slot was free.

    """
    level = state.player_level
    projectiles = (
        state.player_projectiles[level] if from_player else state.mob_projectiles[level]
    )
    slots = MAX_PLAYER_PROJECTILES if from_player else MAX_MOB_PROJECTILES
    for i in range(slots):
        if projectiles.mask[i]:
            continue
        projectiles.position[i, 0] = row
        projectiles.position[i, 1] = col
        physical, fire, ice = mob_damage_vector_numba(
            projectile_type,
            MobType.PROJECTILE,
        )
        projectiles.health[i] = physical + fire + ice
        projectiles.attack_cooldown[i] = 0
        projectiles.type_id[i] = projectile_type
        projectiles.mask[i] = 1
        if from_player:
            state.player_projectile_directions[level, i, 0] = dir_row
            state.player_projectile_directions[level, i, 1] = dir_col
        else:
            state.mob_projectile_dirs[level, i, 0] = dir_row
            state.mob_projectile_dirs[level, i, 1] = dir_col
        return True
    return False


@jit
def craft_numba(state: EnvState, action: int) -> None:
    """Apply the one crafting recipe ``action`` names, if its workbench and materials allow.

    Args:
      state: The environment's world, mutated in place.
      action: The action taken this tick.

    """
    level = state.player_level
    at_table = False
    at_furnace = False
    for i in range(8):
        row = state.player_position[0] + NEARBY_TILES[i, 0]
        col = state.player_position[1] + NEARBY_TILES[i, 1]
        if row < 0 or row >= MAP_SIZE or col < 0 or col >= MAP_SIZE:
            continue
        at_table = at_table or state.map[level, row, col] == BlockType.CRAFTING_TABLE
        at_furnace = at_furnace or state.map[level, row, col] == BlockType.FURNACE

    inv = state.inventory
    if (
        action == Action.MAKE_WOOD_PICKAXE
        and at_table
        and inv.wood >= 1
        and inv.pickaxe < 1
    ):
        inv.wood -= 1
        inv.pickaxe = 1
    elif (
        action == Action.MAKE_STONE_PICKAXE
        and at_table
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.pickaxe < 2
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.pickaxe = 2
    elif (
        action == Action.MAKE_IRON_PICKAXE
        and at_table
        and at_furnace
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.iron >= 1
        and inv.coal >= 1
        and inv.pickaxe < 3
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.iron -= 1
        inv.coal -= 1
        inv.pickaxe = 3
    elif (
        action == Action.MAKE_DIAMOND_PICKAXE
        and at_table
        and inv.wood >= 1
        and inv.diamond >= 3
        and inv.pickaxe < 4
    ):
        inv.wood -= 1
        inv.diamond -= 3
        inv.pickaxe = 4
    elif (
        action == Action.MAKE_WOOD_SWORD
        and at_table
        and inv.wood >= 1
        and inv.sword < 1
    ):
        inv.wood -= 1
        inv.sword = 1
    elif (
        action == Action.MAKE_STONE_SWORD
        and at_table
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.sword < 2
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.sword = 2
    elif (
        action == Action.MAKE_IRON_SWORD
        and at_table
        and at_furnace
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.iron >= 1
        and inv.coal >= 1
        and inv.sword < 3
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.iron -= 1
        inv.coal -= 1
        inv.sword = 3
    elif (
        action == Action.MAKE_DIAMOND_SWORD
        and at_table
        and inv.wood >= 1
        and inv.diamond >= 2
        and inv.sword < 4
    ):
        inv.wood -= 1
        inv.diamond -= 2
        inv.sword = 4
    elif (
        action == Action.MAKE_ARROW
        and at_table
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.arrows < 99
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.arrows += 2
    elif (
        action == Action.MAKE_TORCH
        and at_table
        and inv.wood >= 1
        and inv.coal >= 1
        and inv.torches < 99
    ):
        inv.wood -= 1
        inv.coal -= 1
        inv.torches += 4
    elif (
        action == Action.MAKE_IRON_ARMOUR
        and at_table
        and at_furnace
        and inv.iron >= 3
        and inv.coal >= 3
    ):
        for i in range(4):
            if inv.armour[i] < 1:
                inv.iron -= 3
                inv.coal -= 3
                inv.armour[i] = 1
                state.achievements[Achievement.MAKE_IRON_ARMOUR.value] = 1
                break
    elif action == Action.MAKE_DIAMOND_ARMOUR and at_table and inv.diamond >= 3:
        for i in range(4):
            if inv.armour[i] < 2:
                inv.diamond -= 3
                inv.armour[i] = 2
                state.achievements[Achievement.MAKE_DIAMOND_ARMOUR.value] = 1
                break


@jit
def place_numba(state: EnvState, action: int) -> None:
    """Put a table, furnace, stone, torch or sapling on the faced tile, if allowed.

        The C reads the faced tile and its occupancy for every action; every
        read is pure, so this returns before them unless the action is one of
        the five that place.

    Args:
      state: The environment's world, mutated in place.
      action: The action taken this tick.

    """
    if action != Action.PLACE_TORCH and (
        action < Action.PLACE_STONE or action > Action.PLACE_PLANT
    ):
        return
    dr, dc = action_to_direction_numba(state.player_direction)
    row = state.player_position[0] + dr
    col = state.player_position[1] + dc
    if row < 0 or row >= MAP_SIZE or col < 0 or col >= MAP_SIZE:
        return
    level = state.player_level
    inv = state.inventory
    block = state.map[level, row, col]
    occupied = (
        is_solid_block_numba(block)
        or state.item_map[level, row, col] != ItemType.NONE
        or mob_at_numba(state, level, row, col)
    )

    if action == Action.PLACE_TABLE and not occupied and inv.wood >= 2:
        set_block_numba(state, level, row, col, BlockType.CRAFTING_TABLE)
        inv.wood -= 2
        state.achievements[Achievement.PLACE_TABLE.value] = 1
    elif action == Action.PLACE_FURNACE and not occupied and inv.stone >= 1:
        set_block_numba(state, level, row, col, BlockType.FURNACE)
        inv.stone -= 1
        state.achievements[Achievement.PLACE_FURNACE.value] = 1
    elif (
        action == Action.PLACE_STONE
        and (block == BlockType.WATER or not occupied)
        and inv.stone >= 1
    ):
        set_block_numba(state, level, row, col, BlockType.STONE)
        inv.stone -= 1
        state.achievements[Achievement.PLACE_STONE.value] = 1
    elif (
        action == Action.PLACE_TORCH
        and TORCH_GROUND[block]
        and state.item_map[level, row, col] == ItemType.NONE
        and inv.torches >= 1
    ):
        state.item_map[level, row, col] = ItemType.TORCH
        for light_dr in range(-4, 5):
            light_row = row + light_dr
            if light_row < 0 or light_row >= MAP_SIZE:
                continue
            for light_dc in range(-4, 5):
                light_col = col + light_dc
                if light_col < 0 or light_col >= MAP_SIZE:
                    continue
                torch_light = np.float32(1.0) - sqrtf(
                    np.float32(light_dr * light_dr + light_dc * light_dc),
                ) / np.float32(5.0)
                torch_light = max(torch_light, np.float32(0.0))
                light = clampf_numba(
                    np.float32(state.light_map[level, light_row, light_col])
                    / np.float32(255.0)
                    + torch_light,
                    np.float32(0.0),
                    np.float32(1.0),
                )
                state.light_map[level, light_row, light_col] = np.uint8(
                    light * np.float32(255.0),
                )
        inv.torches -= 1
        state.achievements[Achievement.PLACE_TORCH.value] = 1
    elif (
        action == Action.PLACE_PLANT
        and block == BlockType.GRASS
        and state.item_map[level, row, col] == ItemType.NONE
        and inv.sapling >= 1
    ):
        set_block_numba(state, level, row, col, BlockType.PLANT)
        inv.sapling -= 1
        for i in range(MAX_GROWING_PLANTS):
            if not state.growing_plants_mask[i]:
                state.growing_plants_pos[i, 0] = row
                state.growing_plants_pos[i, 1] = col
                state.growing_plants_age[i] = 0
                state.growing_plants_mask[i] = 1
                break
        state.achievements[Achievement.PLACE_PLANT.value] = 1


@jit
def interact_numba(state: EnvState, rng: Array1[np.uint32], action: int) -> None:
    """Perform the ``DO`` action on the faced tile; a no-op for any other action.

    Args:
      state: The environment's world, mutated in place.
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      action: The action taken this tick.

    """
    if action != Action.DO:
        return
    dr, dc = action_to_direction_numba(state.player_direction)
    row = state.player_position[0] + dr
    col = state.player_position[1] + dc
    in_bounds = 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE
    level = state.player_level
    inv = state.inventory

    did_attack = False
    attack_class, attack_slot = find_mob_at_numba(state, level, row, col)
    if attack_slot >= 0:
        attack_mobs = mobs_for_class_numba(state, level, attack_class)
        base = SWORD_DAMAGE[clampi_numba(inv.sword, 0, 4)]
        physical = base * (
            np.float32(1.0) + np.float32(0.25) * np.float32(state.player_strength - 1)
        )
        magic = (
            base
            * np.float32(0.5)
            * (
                np.float32(1.0)
                + np.float32(0.05) * np.float32(state.player_intelligence - 1)
            )
        )
        fire = magic if state.sword_enchantment == 1 else np.float32(0.0)
        ice = magic if state.sword_enchantment == 2 else np.float32(0.0)
        did_attack = damage_mob_at_numba(
            state,
            level,
            row,
            col,
            damage_to_mob_numba(
                physical,
                fire,
                ice,
                attack_mobs.type_id[attack_slot],
                attack_class,
            ),
            True,
            True,
        )
    if did_attack or not in_bounds:
        return

    block = state.map[level, row, col]
    if block in (BlockType.TREE, BlockType.FIRE_TREE, BlockType.ICE_SHRUB):
        ground = BlockType.GRASS
        if block == BlockType.FIRE_TREE:
            ground = BlockType.FIRE_GRASS
        elif block == BlockType.ICE_SHRUB:
            ground = BlockType.ICE_GRASS
        set_block_numba(state, level, row, col, ground)
        inv.wood += 1
    elif block == BlockType.STONE and inv.pickaxe >= 1:
        set_block_numba(state, level, row, col, BlockType.PATH)
        inv.stone += 1
    elif block == BlockType.COAL and inv.pickaxe >= 1:
        set_block_numba(state, level, row, col, BlockType.PATH)
        inv.coal += 1
    elif block == BlockType.IRON and inv.pickaxe >= 2:
        set_block_numba(state, level, row, col, BlockType.PATH)
        inv.iron += 1
    elif block == BlockType.DIAMOND and inv.pickaxe >= 3:
        set_block_numba(state, level, row, col, BlockType.PATH)
        inv.diamond += 1
    elif block == BlockType.SAPPHIRE and inv.pickaxe >= 4:
        set_block_numba(state, level, row, col, BlockType.PATH)
        inv.sapphire += 1
    elif block == BlockType.RUBY and inv.pickaxe >= 4:
        set_block_numba(state, level, row, col, BlockType.PATH)
        inv.ruby += 1
    elif block == BlockType.STALAGMITE and inv.pickaxe >= 1:
        set_block_numba(state, level, row, col, BlockType.PATH)
        inv.stone += 1
    elif block in (BlockType.CRAFTING_TABLE, BlockType.FURNACE):
        set_block_numba(state, level, row, col, BlockType.PATH)
    elif block in (BlockType.WATER, BlockType.FOUNTAIN):
        state.player_drink = clampi_numba(
            state.player_drink + 1,
            0,
            max_need_numba(state),
        )
        state.player_thirst = np.float32(0.0)
        state.achievements[Achievement.COLLECT_DRINK.value] = 1
    elif block == BlockType.RIPE_PLANT:
        set_block_numba(state, level, row, col, BlockType.PLANT)
        for i in range(MAX_GROWING_PLANTS):
            if (
                state.growing_plants_pos[i, 0] == row
                and state.growing_plants_pos[i, 1] == col
            ):
                state.growing_plants_age[i] = 0
                break
        state.player_food = clampi_numba(
            state.player_food + 4,
            0,
            max_need_numba(state),
        )
        state.player_hunger = np.float32(0.0)
        state.achievements[Achievement.EAT_PLANT.value] = 1
    elif block == BlockType.CHEST:
        set_block_numba(state, level, row, col, BlockType.PATH)
        _open_chest_numba(state, rng, level)
    elif (
        block == BlockType.NECROMANCER
        and boss_vulnerable_numba(state)
        and fighting_boss_numba(state)
    ):
        state.boss_progress += 1
        state.boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS
        state.achievements[Achievement.DAMAGE_NECROMANCER.value] = 1
    if block == BlockType.GRASS and rng_f32_numba(rng) < np.float32(0.1):
        inv.sapling += 1
    state.chests_opened[level] |= block == BlockType.CHEST


@jit
def abilities_numba(state: EnvState, rng: Array1[np.uint32], action: int) -> None:
    """Fire, drink, read, enchant, count the boss round down, and level up.

    Args:
      state: The environment's world, mutated in place.
      rng: The environment's stream, ``uint32 [1]``, advanced by every draw.
      action: The action taken this tick.

    """
    dr, dc = action_to_direction_numba(state.player_direction)
    row = state.player_position[0] + dr
    col = state.player_position[1] + dc
    in_bounds = 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE
    level = state.player_level
    inv = state.inventory

    fire_row = dr
    fire_col = dc
    if fire_row == 0 and fire_col == 0:
        fire_row = 1
    prow = state.player_position[0]
    pcol = state.player_position[1]
    if action == Action.SHOOT_ARROW and inv.bow > 0 and inv.arrows > 0:
        fired = spawn_projectile_numba(
            state,
            True,
            ProjectileType.ARROW2,
            prow,
            pcol,
            fire_row,
            fire_col,
        )
        if fired:
            inv.arrows -= 1
            state.achievements[Achievement.FIRE_BOW.value] = 1
    elif (
        action == Action.CAST_FIREBALL
        and state.learned_spells[0]
        and state.player_mana >= 2
    ):
        cast = spawn_projectile_numba(
            state,
            True,
            ProjectileType.FIREBALL,
            prow,
            pcol,
            fire_row,
            fire_col,
        )
        if cast:
            state.player_mana -= 2
            state.achievements[Achievement.CAST_FIREBALL.value] = 1
    elif (
        action == Action.CAST_ICEBALL
        and state.learned_spells[1]
        and state.player_mana >= 2
    ):
        cast = spawn_projectile_numba(
            state,
            True,
            ProjectileType.ICEBALL,
            prow,
            pcol,
            fire_row,
            fire_col,
        )
        if cast:
            state.player_mana -= 2
            state.achievements[Achievement.CAST_ICEBALL.value] = 1

    potion = action - Action.DRINK_POTION_RED
    if potion >= 0 and potion < NUM_POTIONS and inv.potions[potion] > 0:
        effect = state.potion_mapping[potion]
        inv.potions[potion] -= 1
        if effect == 0:
            state.player_health += np.float32(8.0)
        elif effect == 1:
            state.player_health -= np.float32(3.0)
        elif effect == 2:
            state.player_mana += 8
        elif effect == 3:
            state.player_mana -= 3
        elif effect == 4:
            state.player_energy += 8
        else:
            state.player_energy -= 3
        state.achievements[Achievement.DRINK_POTION.value] = 1

    if action == Action.READ_BOOK and inv.books > 0:
        spell = choose_weighted_pair_numba(
            rng,
            np.float32(0.0) if state.learned_spells[0] else np.float32(1.0),
            np.float32(0.0) if state.learned_spells[1] else np.float32(1.0),
        )
        inv.books -= 1
        state.learned_spells[spell] = 1
        state.achievements[
            Achievement.LEARN_FIREBALL.value
            if spell == 0
            else Achievement.LEARN_ICEBALL.value
        ] = 1

    eblock = state.map[level, row, col] if in_bounds else 0
    enchant = 0
    if eblock == BlockType.ENCHANTMENT_TABLE_FIRE:
        enchant = 1
    elif eblock == BlockType.ENCHANTMENT_TABLE_ICE:
        enchant = 2
    gems = inv.ruby if enchant == 1 else inv.sapphire
    could = state.player_mana >= 9 and enchant != 0 and gems >= 1
    enchanting_sword = could and action == Action.ENCHANT_SWORD and inv.sword > 0
    enchanting_bow = could and action == Action.ENCHANT_BOW and inv.bow > 0
    enchanting_armour = (
        could and action == Action.ENCHANT_ARMOUR and equipped_armour_numba(state) > 0
    )
    armour_target = 0
    # The candidate weights are read only by the draw, so they are built only
    # on that path: an ``np.empty`` on every tick was an NRT allocation per
    # environment (measured, 1 of the 7 per tick).
    if enchanting_armour:
        unenchanted = 0
        for i in range(4):
            unenchanted += 1 if state.armour_enchantments[i] == 0 else 0
        candidates = np.empty(4, dtype=np.float32)
        for i in range(4):
            opposite = (
                state.armour_enchantments[i] != 0
                and state.armour_enchantments[i] != enchant
            )
            candidates[i] = (
                np.float32(1.0)
                if state.armour_enchantments[i] == 0 or (unenchanted == 0 and opposite)
                else np.float32(0.0)
            )
        armour_target = choose_weighted_numba(rng, candidates, 4)
    if enchanting_sword:
        state.sword_enchantment = enchant
        state.achievements[Achievement.ENCHANT_SWORD.value] = 1
    if enchanting_bow:
        state.bow_enchantment = enchant
    if enchanting_armour:
        state.armour_enchantments[armour_target] = enchant
        state.achievements[Achievement.ENCHANT_ARMOUR.value] = 1
    if enchanting_sword or enchanting_bow or enchanting_armour:
        if enchant == 1:
            inv.ruby -= 1
        else:
            inv.sapphire -= 1
        state.player_mana -= 9

    state.achievements[Achievement.DEFEAT_NECROMANCER.value] |= (
        state.boss_progress >= NUM_LEVELS - 1
    )
    if fighting_boss_numba(state):
        state.boss_timestep_to_spawn_this_round -= 1

    if state.player_xp >= 1:
        leveled = False
        if (
            action == Action.LEVEL_UP_DEXTERITY
            and state.player_dexterity < MAX_ATTRIBUTE
        ):
            state.player_dexterity += 1
            leveled = True
        elif (
            action == Action.LEVEL_UP_STRENGTH and state.player_strength < MAX_ATTRIBUTE
        ):
            state.player_strength += 1
            leveled = True
        elif (
            action == Action.LEVEL_UP_INTELLIGENCE
            and state.player_intelligence < MAX_ATTRIBUTE
        ):
            state.player_intelligence += 1
            leveled = True
        if leveled:
            state.player_xp -= 1


@jit
def grow_plants_numba(state: EnvState) -> None:
    """Age every sown plant; one that reaches 600 ripens on the surface.

    Args:
      state: The environment's world, mutated in place.

    """
    for plant in range(MAX_GROWING_PLANTS):
        if not state.growing_plants_mask[plant]:
            continue
        state.growing_plants_age[plant] += 1
        if state.growing_plants_age[plant] >= 600:
            set_block_numba(
                state,
                0,
                state.growing_plants_pos[plant, 0],
                state.growing_plants_pos[plant, 1],
                BlockType.RIPE_PLANT,
            )


@jit
def move_player_numba(state: EnvState, action: int) -> None:
    """Step one tile in the action's direction if passable; face that way regardless.

        The C also tests the player's own tile when the action moves nowhere,
        and writes the position back unchanged; the test is pure, so this
        returns first.

    Args:
      state: The environment's world, mutated in place.
      action: The action taken this tick.

    """
    dr, dc = action_to_direction_numba(action)
    if dr == 0 and dc == 0:
        return
    level = state.player_level
    proposed_row = state.player_position[0] + dr
    proposed_col = state.player_position[1] + dc
    valid = 0 <= proposed_row < MAP_SIZE and 0 <= proposed_col < MAP_SIZE
    if valid:
        pblock = state.map[level, proposed_row, proposed_col]
        valid = not PLAYER_BLOCKED[pblock] and not mob_at_numba(
            state,
            level,
            proposed_row,
            proposed_col,
        )
    if valid:
        state.player_position[0] = proposed_row
        state.player_position[1] = proposed_col
    state.player_direction = action


@jit
def tick_numba(state: EnvState, action: int, max_timesteps: int) -> bool:
    """Close one tick; return whether the episode has ended.

        Sleep and rest, the meters, the clamps, the inventory achievements, the
        clock and the daylight, in the C's order.

    Args:
      state: The environment's world, mutated in place.
      action: The action taken this tick.
      max_timesteps: Ticks after which the episode ends; the default
        ``Rules``' is ``DEFAULT_MAX_TIMESTEPS``.

    Returns:
      done: Whether the player died or the clock ran out.

    """
    start_sleep = action == Action.SLEEP and state.player_energy < max_need_numba(state)
    state.is_sleeping = 1 if state.is_sleeping or start_sleep else 0

    wake_from_sleep = state.is_sleeping and state.player_energy >= max_need_numba(state)
    state.is_sleeping = 1 if state.is_sleeping and not wake_from_sleep else 0
    state.achievements[Achievement.WAKE_UP.value] = (
        1 if state.achievements[Achievement.WAKE_UP.value] or wake_from_sleep else 0
    )

    start_rest = action == Action.REST and state.player_health < np.float32(
        max_health_numba(state),
    )
    state.is_resting = 1 if state.is_resting or start_rest else 0

    wake_from_rest = state.is_resting and (
        state.player_health >= np.float32(max_health_numba(state))
        or state.player_food <= 0
        or state.player_drink <= 0
    )
    state.is_resting = 1 if state.is_resting and not wake_from_rest else 0

    not_boss = not fighting_boss_numba(state)
    decay = np.float32(1.0) - np.float32(0.125) * np.float32(state.player_dexterity - 1)

    state.player_hunger += (
        np.float32(0.5) if state.is_sleeping else np.float32(1.0)
    ) * decay
    if state.player_hunger > np.float32(25.0):
        state.player_hunger = np.float32(0.0)
        state.player_food = clampi_numba(
            state.player_food - (1 if not_boss else 0),
            0,
            max_need_numba(state),
        )

    state.player_thirst += (
        np.float32(0.5) if state.is_sleeping else np.float32(1.0)
    ) * decay
    if state.player_thirst > np.float32(20.0):
        state.player_thirst = np.float32(0.0)
        state.player_drink = clampi_numba(
            state.player_drink - (1 if not_boss else 0),
            0,
            max_need_numba(state),
        )

    if state.is_sleeping:
        state.player_fatigue = state.player_fatigue - np.float32(1.0)
        state.player_fatigue = min(state.player_fatigue, np.float32(0.0))
    else:
        state.player_fatigue += decay
    if state.player_fatigue > np.float32(30.0):
        state.player_fatigue = np.float32(0.0)
        state.player_energy = clampi_numba(
            state.player_energy - (1 if not_boss else 0),
            0,
            max_need_numba(state),
        )
    elif state.player_fatigue < np.float32(-10.0):
        state.player_fatigue = np.float32(0.0)
        state.player_energy = clampi_numba(
            state.player_energy + 1,
            0,
            max_need_numba(state),
        )

    all_necessities = (
        state.player_food > 0
        and state.player_drink > 0
        and (state.player_energy > 0 or state.is_sleeping)
    )
    if all_necessities:
        state.player_recover += (
            np.float32(2.0) if state.is_sleeping else np.float32(1.0)
        )
    else:
        state.player_recover += (
            np.float32(-0.5) if state.is_sleeping else np.float32(-1.0)
        ) * (np.float32(1.0) if not_boss else np.float32(0.0))

    if state.player_recover > np.float32(25.0):
        state.player_recover = np.float32(0.0)
        state.player_health = clampf_numba(
            state.player_health + np.float32(1.0),
            np.float32(0.0),
            np.float32(max_health_numba(state)),
        )
    elif state.player_recover < np.float32(-15.0):
        state.player_recover = np.float32(0.0)
        state.player_health -= np.float32(1.0)

    mana_gain = np.float32(2.0) if state.is_sleeping else np.float32(1.0)
    mana_coeff = np.float32(1.0) + np.float32(0.25) * np.float32(
        state.player_intelligence - 1,
    )
    state.player_recover_mana = (state.player_recover_mana + mana_gain) * mana_coeff
    if state.player_recover_mana > np.float32(30.0):
        state.player_recover_mana = np.float32(0.0)
        state.player_mana = clampi_numba(
            state.player_mana + 1,
            0,
            max_mana_numba(state),
        )

    clip_meters_numba(state)
    unlock_from_inventory_numba(state)

    state.timestep += 1
    light = daylight_table()
    state.light_level = (
        light[state.timestep]
        if state.timestep < light.shape[0]
        else daylight_numba(state.timestep)
    )
    return bool(
        state.player_health <= np.float32(0.0) or state.timestep >= max_timesteps,
    )


@jit
def clip_meters_numba(state: EnvState) -> None:
    """Clamp every inventory count to ``[0, 99]`` and every meter to its cap.

    Args:
      state: The environment's world, mutated in place.

    """
    inv = state.inventory
    inv.wood = clampi_numba(inv.wood, 0, 99)
    inv.stone = clampi_numba(inv.stone, 0, 99)
    inv.coal = clampi_numba(inv.coal, 0, 99)
    inv.iron = clampi_numba(inv.iron, 0, 99)
    inv.diamond = clampi_numba(inv.diamond, 0, 99)
    inv.sapling = clampi_numba(inv.sapling, 0, 99)
    inv.pickaxe = clampi_numba(inv.pickaxe, 0, 99)
    inv.sword = clampi_numba(inv.sword, 0, 99)
    inv.bow = clampi_numba(inv.bow, 0, 99)
    inv.arrows = clampi_numba(inv.arrows, 0, 99)
    inv.torches = clampi_numba(inv.torches, 0, 99)
    inv.ruby = clampi_numba(inv.ruby, 0, 99)
    inv.sapphire = clampi_numba(inv.sapphire, 0, 99)
    inv.books = clampi_numba(inv.books, 0, 99)
    for i in range(4):
        inv.armour[i] = clampi_numba(inv.armour[i], 0, 99)
    for i in range(NUM_POTIONS):
        inv.potions[i] = clampi_numba(inv.potions[i], 0, 99)

    state.player_health = clampf_numba(
        state.player_health,
        np.float32(0.0),
        np.float32(max_health_numba(state)),
    )
    state.player_food = clampi_numba(state.player_food, 0, max_need_numba(state))
    state.player_drink = clampi_numba(state.player_drink, 0, max_need_numba(state))
    state.player_energy = clampi_numba(state.player_energy, 0, max_need_numba(state))
    state.player_mana = clampi_numba(state.player_mana, 0, max_mana_numba(state))


@jit
def unlock_from_inventory_numba(state: EnvState) -> None:
    """Award the achievements that holding something implies.

    Args:
      state: The environment's world, mutated in place.

    """
    inv = state.inventory
    ach = state.achievements
    ach[Achievement.COLLECT_WOOD.value] |= inv.wood > 0
    ach[Achievement.COLLECT_STONE.value] |= inv.stone > 0
    ach[Achievement.COLLECT_COAL.value] |= inv.coal > 0
    ach[Achievement.COLLECT_IRON.value] |= inv.iron > 0
    ach[Achievement.COLLECT_DIAMOND.value] |= inv.diamond > 0
    ach[Achievement.COLLECT_SAPPHIRE.value] |= inv.sapphire > 0
    ach[Achievement.COLLECT_RUBY.value] |= inv.ruby > 0
    ach[Achievement.COLLECT_SAPLING.value] |= inv.sapling > 0
    ach[Achievement.FIND_BOW.value] |= inv.bow > 0
    ach[Achievement.MAKE_ARROW.value] |= inv.arrows > 0
    ach[Achievement.MAKE_TORCH.value] |= inv.torches > 0
    ach[Achievement.MAKE_WOOD_PICKAXE.value] |= inv.pickaxe >= 1
    ach[Achievement.MAKE_STONE_PICKAXE.value] |= inv.pickaxe >= 2
    ach[Achievement.MAKE_IRON_PICKAXE.value] |= inv.pickaxe >= 3
    ach[Achievement.MAKE_DIAMOND_PICKAXE.value] |= inv.pickaxe >= 4
    ach[Achievement.MAKE_WOOD_SWORD.value] |= inv.sword >= 1
    ach[Achievement.MAKE_STONE_SWORD.value] |= inv.sword >= 2
    ach[Achievement.MAKE_IRON_SWORD.value] |= inv.sword >= 3
    ach[Achievement.MAKE_DIAMOND_SWORD.value] |= inv.sword >= 4


@jit
def daylight_numba(timestep: int) -> np.float32:
    """Return the surface light level at ``timestep``, on ``[0, 1]``.

        ``1 - powf(fabsf(cosf(pi * (fmodf(t / 300, 1) + 0.3))), 3)``: a day is
        300 steps, the episode starts a third of the way in, and the cube
        flattens the peak into a long bright afternoon.

    Args:
      timestep: Steps elapsed this episode.

    Returns:
      light: The float32 light level.

    """
    day_progress = fmodf(
        np.float32(timestep) / np.float32(DAY_LENGTH),
        np.float32(1.0),
    ) + np.float32(0.3)
    return np.float32(
        np.float32(1.0) - powf(abs(cosf(PI * day_progress)), np.float32(3.0)),
    )


DAYLIGHT_TIMESTEPS: Final = DEFAULT_MAX_TIMESTEPS + 1
"""The timesteps :data:`daylight_table` covers: every one a default episode reaches."""


def _daylight_table_signature(
    typingctx: object,
) -> tuple[Signature, Callable[..., Value]]:
    """Type :data:`daylight_table` as a read-only float32 array, lowered to a constant."""
    del typingctx
    return _DAYLIGHT_ARRAY(), _emit_daylight_table


def _emit_daylight_table(
    context: _ConstantArrays,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit the table, computed now through the jitted :func:`daylight_numba`, as a constant array."""
    del signature, args
    table = np.fromiter(
        (daylight_numba(timestep) for timestep in range(DAYLIGHT_TIMESTEPS)),
        np.float32,
        DAYLIGHT_TIMESTEPS,
    )
    return context.make_constant_array(builder, _DAYLIGHT_ARRAY, table)


class _ConstantArrays(Protocol):
    """The lowering context's one method the table's codegen calls (Numba's ``BaseContext``)."""

    def make_constant_array(
        self,
        builder: IRBuilder,
        typ: nbtypes.Array,
        ary: np.ndarray,
    ) -> Value:
        """Emit ``ary`` as a constant array of type ``typ``."""
        ...


_DAYLIGHT_ARRAY: Final = nbtypes.Array(nbtypes.float32, 1, "C", readonly=True)

daylight_table = cast("Callable[[], np.ndarray]", intrinsic(_daylight_table_signature))
""":func:`daylight_numba` at timesteps ``0..DEFAULT_MAX_TIMESTEPS``, float32, read-only.

A constant computed by :func:`daylight_numba` itself when the calling kernel compiles,
so the step looks the light up where the C calls ``fmodf``, ``cosf`` and
``powf``, for the same bits. Callable from kernels only.
"""


@jit
def _open_chest_numba(state: EnvState, rng: Array1[np.uint32], level: int) -> None:
    """Roll the chest's loot in the C's fifteen-draw order and hand it over."""
    inv = state.inventory
    torch = rng_f32_numba(rng) < np.float32(0.6)
    torches = rng_int_numba(rng, 4, 8)
    ore = rng_f32_numba(rng) < np.float32(0.6)
    ore_id = choose_weighted_numba(rng, CHEST_ORE_WEIGHTS, 5)
    ore_amount_coal = rng_int_numba(rng, 1, 4)
    ore_amount_iron = rng_int_numba(rng, 1, 3)
    ore_amount_diamond = rng_int_numba(rng, 1, 2)
    ore_amount_sapphire = rng_int_numba(rng, 1, 2)
    ore_amount_ruby = rng_int_numba(rng, 1, 2)
    potion = rng_f32_numba(rng) < np.float32(0.5)
    potion_id = rng_int_numba(rng, 0, 6)
    potion_amount = rng_int_numba(rng, 1, 3)
    arrows = rng_f32_numba(rng) < np.float32(0.25)
    arrow_amount = rng_int_numba(rng, 1, 5)
    tool = rng_f32_numba(rng) < np.float32(0.2)
    tool_id = rng_int_numba(rng, 0, 2)
    pickaxe = choose_weighted_numba(rng, CHEST_TOOL_WEIGHTS, 4) + 1
    sword = choose_weighted_numba(rng, CHEST_TOOL_WEIGHTS, 4) + 1
    inv.torches += torches if torch else 0
    if ore:
        if ore_id == 0:
            inv.coal += ore_amount_coal
        elif ore_id == 1:
            inv.iron += ore_amount_iron
        elif ore_id == 2:
            inv.diamond += ore_amount_diamond
        elif ore_id == 3:
            inv.sapphire += ore_amount_sapphire
        else:
            inv.ruby += ore_amount_ruby
    # rng_int returns 6 on the 1.0 edge; the C then writes potions[6], the
    # struct's next field.
    if potion_id == NUM_POTIONS:
        inv.books += potion_amount if potion else 0
    else:
        inv.potions[potion_id] += potion_amount if potion else 0
    inv.arrows += arrow_amount if arrows else 0
    if tool and tool_id == 0 and pickaxe > inv.pickaxe:
        inv.pickaxe = pickaxe
    if tool and tool_id == 1 and sword > inv.sword:
        inv.sword = sword
    if not state.chests_opened[level]:
        if level == 1:
            inv.bow = 1
        if level in (3, 4):
            inv.books += 1
    state.achievements[Achievement.OPEN_CHEST.value] = 1
