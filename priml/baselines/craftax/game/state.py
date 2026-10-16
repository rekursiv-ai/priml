"""The Craftax world's fixed vocabulary, and its state laid out as the C ``State``.

The constants are facts about the game rather than tunables: the map and
observation geometry, the block, item, mob and action vocabularies, and what
each achievement is worth. They are ``Final`` for that reason -- an experiment
that wants different numbers is playing a different game, not running a
variant. Every value is the C constant of the same name, so a kernel reads as
the C it ports. Numba folds module-level ints, floats, ``IntEnum`` members and
numpy arrays into compiled code as constants, which is why the tables are
arrays rather than lists, and the state is a numpy record rather than tensors
for the same reason (the package docstring has the measurements). The
per-floor terrain recipes (``SMOOTH_LEVEL_CONFIGS`` and
``DUNGEON_LEVEL_CONFIGS``) live in ``world_gen.py``.

One environment's world is one record of ``STATE_DTYPE``, 80,248 bytes laid
out exactly as ``craftax.h`` lays out ``struct State``, so a state dumped by
the C env compares to ours with ``tobytes()`` and no translation. The pool of
fresh worlds and the live batch are arrays of these records, and a Numba
kernel takes one record (``states[i]``) and mutates it in place.

Three choices in the dtype are forced by that byte contract rather than by
taste:

- ``Mobs.mask`` is ``uint8``, not ``bool``. C's ``bool`` is one byte holding
  0 or 1, so the bytes agree, and Numba cannot type a ``bool`` sub-array
  inside a record.
- The C structs pad in two places, a byte after ``Mobs.mask`` and four bytes
  after ``State.timestep``, and those bytes are a field too, ``PADDING``.
  numpy copies a structured array field by field, so a byte that is no
  field's comes out of ``copy()``, ``np.array`` or fancy indexing as the new
  allocation found it, and the C env's state hash covers it. The C env's
  padding is zero and stays zero.
- With the padding spelled out, numpy's packed layout of the fields in C order
  is C's: every offset and size equals ``offsetof`` and ``sizeof``.
  ``align=True`` gives the same offsets, but it also marks the records
  aligned, and Numba then compiles their field accesses to other machine
  code (measured: the step's).

``state_test.py`` pins every offset to the values measured from the C structs
with ``offsetof``.

A kernel reads a record's fields as attributes (``state.inventory.wood``),
which Numba types from the dtype but a static checker cannot. The ``EnvState``,
``Inventory`` and ``Mobs`` protocols declare those fields, so a kernel
annotated with them is checked for field names and widths, and ``env_state``
is the one place a raw record is given that type.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h`` and
        ``ocean/craftax/constants.h``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Final,
    LiteralString,
    NamedTuple,
    Protocol,
    Self,
    SupportsIndex,
    cast,
    overload,
)

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


OBS_ROWS: Final = 9
OBS_COLS: Final = 11
MAP_SIZE: Final = 48
NUM_LEVELS: Final = 9

NUM_BLOCK_TYPES: Final = 37
NUM_ITEM_TYPES: Final = 5
NUM_MOB_CLASSES: Final = 5
NUM_MOB_TYPES: Final = 8
INVENTORY_OBS_SIZE: Final = 51
OBS_TILE_CHANNELS: Final = 3 + NUM_MOB_CLASSES
OBS_SIZE: Final = OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS + INVENTORY_OBS_SIZE
"""843: a 9x11 view with 8 channels per tile, then 51 inventory scalars."""

SYMBOLIC_TILE_CHANNELS: Final = (
    NUM_BLOCK_TYPES + NUM_ITEM_TYPES + NUM_MOB_CLASSES * NUM_MOB_TYPES + 1
)
"""83: original Craftax's one-hot tile: block, item, (mob class, type), visible."""

SYMBOLIC_OBS_SIZE: Final = (
    OBS_ROWS * OBS_COLS * SYMBOLIC_TILE_CHANNELS + INVENTORY_OBS_SIZE
)
"""8,268: Craftax-Symbolic-v1's observation: the view one-hot, then the 51 scalars."""

ATN_DIM: Final = 43
NUM_ACHIEVEMENTS: Final = 67

NO_ACTION: Final = ATN_DIM
"""43: the previous-action field at an episode start, one past the last action id."""

ACTION_OBS_SIZE: Final = OBS_SIZE + 1
"""844: the packed observation, then the id of the action that led to it."""

MAX_MELEE_MOBS: Final = 3
MAX_PASSIVE_MOBS: Final = 3
MAX_RANGED_MOBS: Final = 2
MAX_MOB_PROJECTILES: Final = 3
MAX_PLAYER_PROJECTILES: Final = 3
MAX_GROWING_PLANTS: Final = 10
NUM_POTIONS: Final = 6
MOB_SLOTS: Final = 3
"""Slots in a ``Mobs`` struct; ranged mobs use only the first two."""

DEFAULT_MAX_TIMESTEPS: Final = 100_000
DAY_LENGTH: Final = 300
VISIBLE_LIGHT_THRESHOLD: Final = 12
MAX_ATTRIBUTE: Final = 5
MOB_DESPAWN_DISTANCE: Final = 14
MONSTERS_KILLED_TO_CLEAR_LEVEL: Final = 8

DUNGEON_ROOM_COUNT: Final = 8
DUNGEON_MIN_ROOM_SIZE: Final = 5
DUNGEON_MAX_ROOM_SIZE: Final = 10
DUNGEON_CHUNK_SIZE: Final = 16
BOSS_SPAWN_TURNS: Final = 7

MAP_CELLS: Final = MAP_SIZE * MAP_SIZE
NOISE_PI2: Final = np.float32(6.28318530717958647692)
NOISE_SQRT2: Final = np.float32(1.41421356237309504880)
PI: Final = np.float32(3.14159265358979323846)
"""The float32 pi literal the C env spells out in the light-level formula."""


class BlockType(IntEnum):
    """A tile of the world map."""

    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36


class ItemType(IntEnum):
    """An object occupying a tile alongside its block."""

    NONE = 0
    TORCH = 1
    LADDER_DOWN = 2
    LADDER_UP = 3
    LADDER_DOWN_BLOCKED = 4


class MobType(IntEnum):
    """A class of creature, which picks its slot table and its rules."""

    PASSIVE = 0
    MELEE = 1
    RANGED = 2
    PROJECTILE = 3


class ProjectileType(IntEnum):
    """What a projectile slot carries."""

    ARROW = 0
    DAGGER = 1
    FIREBALL = 2
    ICEBALL = 3
    ARROW2 = 4
    SLIMEBALL = 5
    FIREBALL2 = 6
    ICEBALL2 = 7


class Achievement(IntEnum):
    """The 67 achievements, in the order the C tables index them."""

    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21
    MAKE_ARROW = 22
    MAKE_TORCH = 23
    PLACE_TORCH = 24
    MAKE_DIAMOND_SWORD = 25
    MAKE_IRON_ARMOUR = 26
    MAKE_DIAMOND_ARMOUR = 27
    ENTER_GNOMISH_MINES = 28
    ENTER_DUNGEON = 29
    ENTER_SEWERS = 30
    ENTER_VAULT = 31
    ENTER_TROLL_MINES = 32
    ENTER_FIRE_REALM = 33
    ENTER_ICE_REALM = 34
    ENTER_GRAVEYARD = 35
    DEFEAT_GNOME_WARRIOR = 36
    DEFEAT_GNOME_ARCHER = 37
    DEFEAT_ORC_SOLIDER = 38
    DEFEAT_ORC_MAGE = 39
    DEFEAT_LIZARD = 40
    DEFEAT_KOBOLD = 41
    DEFEAT_TROLL = 42
    DEFEAT_DEEP_THING = 43
    DEFEAT_PIGMAN = 44
    DEFEAT_FIRE_ELEMENTAL = 45
    DEFEAT_FROST_TROLL = 46
    DEFEAT_ICE_ELEMENTAL = 47
    DAMAGE_NECROMANCER = 48
    DEFEAT_NECROMANCER = 49
    EAT_BAT = 50
    EAT_SNAIL = 51
    FIND_BOW = 52
    FIRE_BOW = 53
    COLLECT_SAPPHIRE = 54
    LEARN_FIREBALL = 55
    CAST_FIREBALL = 56
    LEARN_ICEBALL = 57
    CAST_ICEBALL = 58
    COLLECT_RUBY = 59
    MAKE_DIAMOND_PICKAXE = 60
    OPEN_CHEST = 61
    DRINK_POTION = 62
    ENCHANT_SWORD = 63
    ENCHANT_ARMOUR = 64
    DEFEAT_KNIGHT = 65
    DEFEAT_ARCHER = 66


class Action(IntEnum):
    """The 43 discrete actions."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    DO = 5
    SLEEP = 6
    PLACE_STONE = 7
    PLACE_TABLE = 8
    PLACE_FURNACE = 9
    PLACE_PLANT = 10
    MAKE_WOOD_PICKAXE = 11
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16
    REST = 17
    DESCEND = 18
    ASCEND = 19
    MAKE_DIAMOND_PICKAXE = 20
    MAKE_DIAMOND_SWORD = 21
    MAKE_IRON_ARMOUR = 22
    MAKE_DIAMOND_ARMOUR = 23
    SHOOT_ARROW = 24
    MAKE_ARROW = 25
    CAST_FIREBALL = 26
    CAST_ICEBALL = 27
    PLACE_TORCH = 28
    DRINK_POTION_RED = 29
    DRINK_POTION_GREEN = 30
    DRINK_POTION_BLUE = 31
    DRINK_POTION_PINK = 32
    DRINK_POTION_CYAN = 33
    DRINK_POTION_YELLOW = 34
    READ_BOOK = 35
    ENCHANT_SWORD = 36
    ENCHANT_ARMOUR = 37
    MAKE_TORCH = 38
    LEVEL_UP_DEXTERITY = 39
    LEVEL_UP_STRENGTH = 40
    LEVEL_UP_INTELLIGENCE = 41
    ENCHANT_BOW = 42


ACHIEVEMENT_REWARD_MAP: Final = np.array(
    [
        *[1.0] * 25,
        3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0, 3.0, 3.0, 3.0, 3.0,
        5.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0,
    ],
    dtype=np.float32,
)  # fmt: skip
"""Reward for each achievement, indexed by ``Achievement``; sums to 226."""

MAX_ACHIEVEMENT_RETURN: Final = np.float32(226.0)
"""``max_achievement_return()``: the fp32 sum of ``ACHIEVEMENT_REWARD_MAP``.

The C function sums the table in fp32 at every episode end; every partial sum
is an integer below 2^24, so the sum is exact and a constant replaces it.
"""


type Scalar = int | float | np.number | np.bool
"""What a kernel may store into any numeric array: numpy and Numba convert it on the store."""


class Array1[T](Protocol):
    """A one-dimensional array as a kernel reads it: each element a ``T``.

    numpy's stubs type an element read as ``Any``. An array declared with these
    protocols reads as ``T``: ``int`` for the integer dtypes, which Numba reads
    as machine integers, and the numpy scalar where the width is the point
    (``np.float32``, ``np.uint32``, ``np.uint64``). Any ``ndarray`` of the rank
    satisfies them.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's extent."""
        ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __array__(self) -> NDArray[np.generic]: ...
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> T: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: T | Scalar | Array1[T],
        /,
    ) -> None: ...
    def sum(self) -> T:
        """Return the elements' sum."""
        ...


class Array2[T](Protocol):
    """A two-dimensional :class:`Array1`: a full index reads a ``T``, one a row."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's extent."""
        ...

    def __len__(self) -> int: ...
    def __array__(self) -> NDArray[np.generic]: ...
    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex], /) -> T: ...
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> Array1[T]: ...
    def __setitem__(
        self,
        index: tuple[SupportsIndex, SupportsIndex] | tuple[SupportsIndex, slice],
        value: T | Scalar,
        /,
    ) -> None: ...


class Array3[T](Protocol):
    """A three-dimensional :class:`Array1`: a full index reads a ``T``, one a grid."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's extent."""
        ...

    def __len__(self) -> int: ...
    def __array__(self) -> NDArray[np.generic]: ...
    @overload
    def __getitem__(
        self,
        index: tuple[SupportsIndex, SupportsIndex, SupportsIndex],
        /,
    ) -> T: ...
    @overload
    def __getitem__(
        self,
        index: tuple[SupportsIndex, SupportsIndex],
        /,
    ) -> Array1[T]: ...
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> Array2[T]: ...
    def __setitem__(
        self,
        index: tuple[SupportsIndex, SupportsIndex, SupportsIndex],
        value: T | Scalar,
        /,
    ) -> None: ...


class Records[R](Protocol):
    """An array of structured records as a kernel reads it: each record an ``R``.

    ``R`` declares the record's fields (:class:`EnvState`, :class:`EnvStats`
    ...), which Numba reads as attributes.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's extent."""
        ...

    def __len__(self) -> int: ...
    def __array__(self) -> NDArray[np.generic]: ...
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> R: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def __setitem__(self, index: SupportsIndex, value: R, /) -> None: ...
    def view(self, dtype: type[np.uint8], /) -> Array1[int]:
        """Return the records' bytes."""
        ...


class Records2[R](Protocol):
    """A two-dimensional :class:`Records`: a full index reads one record."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's extent."""
        ...

    def __len__(self) -> int: ...
    def __array__(self) -> NDArray[np.generic]: ...
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex], /) -> R: ...


class Inventory(Protocol):
    """One ``INVENTORY_DTYPE`` record as a kernel sees it; counts are int32."""

    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    pickaxe: int
    sword: int
    bow: int
    arrows: int
    armour: Array1[int]
    """int32 [4]: the level of each armour piece."""

    torches: int
    ruby: int
    sapphire: int
    potions: Array1[int]
    """int32 [6]: potions held per colour; index 6 is ``books`` in C."""

    books: int


class Mobs(Protocol):
    """One ``MOBS_DTYPE`` record: a creature class on one floor, three slots."""

    position: Array2[int]
    """int32 [3, 2]: row and column of each slot."""

    health: Array1[np.float32]
    """float32 [3]."""

    mask: Array1[int]
    """uint8 [3]: 1 where the slot holds a live creature."""

    attack_cooldown: Array1[int]
    """int32 [3]."""

    type_id: Array1[int]
    """int32 [3]: the species in each slot."""


class EnvState(Protocol):
    """One ``STATE_DTYPE`` record as a kernel sees it.

    Grids are ``[levels, rows, columns]`` uint8; the bit rows are
    ``[levels, rows]`` uint64 with one bit per column; scalars are int32 or
    float32 as the C struct declares them.
    """

    map: Array3[int]
    item_map: Array3[int]
    light_map: Array3[int]
    mob_bits: Array2[np.uint64]
    spawn_land: Array2[np.uint64]
    spawn_grave: Array2[np.uint64]
    spawn_water: Array2[np.uint64]
    down_ladders: Array2[int]
    up_ladders: Array2[int]
    chests_opened: Array1[int]
    monsters_killed: Array1[int]
    player_position: Array1[int]
    player_level: int
    player_direction: int
    player_health: np.float32
    player_food: int
    player_drink: int
    player_energy: int
    player_mana: int
    is_sleeping: int
    is_resting: int
    player_recover: np.float32
    player_hunger: np.float32
    player_thirst: np.float32
    player_fatigue: np.float32
    player_recover_mana: np.float32
    player_xp: int
    player_dexterity: int
    player_strength: int
    player_intelligence: int
    inventory: Inventory
    melee_mobs: Sequence[Mobs]
    passive_mobs: Sequence[Mobs]
    ranged_mobs: Sequence[Mobs]
    mob_projectiles: Sequence[Mobs]
    mob_projectile_dirs: Array3[int]
    player_projectiles: Sequence[Mobs]
    player_projectile_directions: Array3[int]
    growing_plants_pos: Array2[int]
    growing_plants_age: Array1[int]
    growing_plants_mask: Array1[int]
    potion_mapping: Array1[int]
    learned_spells: Array1[int]
    sword_enchantment: int
    bow_enchantment: int
    armour_enchantments: Array1[int]
    boss_progress: int
    boss_timestep_to_spawn_this_round: int
    light_level: np.float32
    achievements: Array1[int]
    state_rng: Array1[np.uint32]
    timestep: int


INVENTORY_DTYPE: Final = np.dtype(
    [
        ("wood", np.int32),
        ("stone", np.int32),
        ("coal", np.int32),
        ("iron", np.int32),
        ("diamond", np.int32),
        ("sapling", np.int32),
        ("pickaxe", np.int32),
        ("sword", np.int32),
        ("bow", np.int32),
        ("arrows", np.int32),
        ("armour", np.int32, (4,)),
        ("torches", np.int32),
        ("ruby", np.int32),
        ("sapphire", np.int32),
        ("potions", np.int32, (NUM_POTIONS,)),
        ("books", np.int32),
    ],
)
"""``Inventory``: what the player carries. ``books`` follows ``potions`` because
the chest's loot can write ``potions[6]``, which is this field."""

PADDING: Final = "padding"
"""The field that names a struct's C padding bytes, so numpy copies them too."""

MOBS_DTYPE: Final = np.dtype(
    [
        ("position", np.int32, (MOB_SLOTS, 2)),
        ("health", np.float32, (MOB_SLOTS,)),
        ("mask", np.uint8, (MOB_SLOTS,)),
        (PADDING, np.uint8),
        ("attack_cooldown", np.int32, (MOB_SLOTS,)),
        ("type_id", np.int32, (MOB_SLOTS,)),
    ],
)
"""``Mobs``: one class of creature on one floor, three fixed slots."""

STATE_DTYPE: Final = np.dtype(
    [
        ("map", np.uint8, (NUM_LEVELS, MAP_SIZE, MAP_SIZE)),
        ("item_map", np.uint8, (NUM_LEVELS, MAP_SIZE, MAP_SIZE)),
        ("light_map", np.uint8, (NUM_LEVELS, MAP_SIZE, MAP_SIZE)),
        ("mob_bits", np.uint64, (NUM_LEVELS, MAP_SIZE)),
        ("spawn_land", np.uint64, (NUM_LEVELS, MAP_SIZE)),
        ("spawn_grave", np.uint64, (NUM_LEVELS, MAP_SIZE)),
        ("spawn_water", np.uint64, (NUM_LEVELS, MAP_SIZE)),
        ("down_ladders", np.int32, (NUM_LEVELS, 2)),
        ("up_ladders", np.int32, (NUM_LEVELS, 2)),
        ("chests_opened", np.int32, (NUM_LEVELS,)),
        ("monsters_killed", np.int32, (NUM_LEVELS,)),
        ("player_position", np.int32, (2,)),
        ("player_level", np.int32),
        ("player_direction", np.int32),
        ("player_health", np.float32),
        ("player_food", np.int32),
        ("player_drink", np.int32),
        ("player_energy", np.int32),
        ("player_mana", np.int32),
        ("is_sleeping", np.int32),
        ("is_resting", np.int32),
        ("player_recover", np.float32),
        ("player_hunger", np.float32),
        ("player_thirst", np.float32),
        ("player_fatigue", np.float32),
        ("player_recover_mana", np.float32),
        ("player_xp", np.int32),
        ("player_dexterity", np.int32),
        ("player_strength", np.int32),
        ("player_intelligence", np.int32),
        ("inventory", INVENTORY_DTYPE),
        ("melee_mobs", MOBS_DTYPE, (NUM_LEVELS,)),
        ("passive_mobs", MOBS_DTYPE, (NUM_LEVELS,)),
        ("ranged_mobs", MOBS_DTYPE, (NUM_LEVELS,)),
        ("mob_projectiles", MOBS_DTYPE, (NUM_LEVELS,)),
        ("mob_projectile_dirs", np.int32, (NUM_LEVELS, MAX_MOB_PROJECTILES, 2)),
        ("player_projectiles", MOBS_DTYPE, (NUM_LEVELS,)),
        (
            "player_projectile_directions",
            np.int32,
            (NUM_LEVELS, MAX_PLAYER_PROJECTILES, 2),
        ),
        ("growing_plants_pos", np.int32, (MAX_GROWING_PLANTS, 2)),
        ("growing_plants_age", np.int32, (MAX_GROWING_PLANTS,)),
        ("growing_plants_mask", np.int32, (MAX_GROWING_PLANTS,)),
        ("potion_mapping", np.int32, (NUM_POTIONS,)),
        ("learned_spells", np.int32, (2,)),
        ("sword_enchantment", np.int32),
        ("bow_enchantment", np.int32),
        ("armour_enchantments", np.int32, (4,)),
        ("boss_progress", np.int32),
        ("boss_timestep_to_spawn_this_round", np.int32),
        ("light_level", np.float32),
        ("achievements", np.int32, (NUM_ACHIEVEMENTS,)),
        ("state_rng", np.uint32, (2,)),
        ("timestep", np.int32),
        (PADDING, np.uint8, (4,)),
    ],
)
"""``State``: one environment's whole world. ``potion_mapping`` is followed by
``learned_spells`` because world generation's shuffle can write
``potion_mapping[6]``, which is ``learned_spells[0]``."""

LOG_DTYPE: Final = np.dtype(
    [
        ("perf", np.float32),
        ("achievement_rate", np.float32),
        ("score", np.float32),
        ("episode_return", np.float32),
        ("episode_length", np.float32),
        ("floors", np.float32, (NUM_LEVELS,)),
        ("achievements", np.float32, (NUM_ACHIEVEMENTS,)),
        ("n", np.float32),
    ],
)
"""``struct Log``: one environment's fp32 episode sums, added in the C order."""

STATS_DTYPE: Final = np.dtype(
    [
        ("episode_return_accum", np.float32),
        ("episode_length_accum", np.int32),
        ("max_floor_accum", np.int32),
        ("steps", np.int32),
        ("last_ticks", np.int32),
        ("undefined_spawns", np.int32),
        ("first_undefined_step", np.int32),
        ("log", LOG_DTYPE),
    ],
)
"""The C ``Env``'s per-environment accumulators plus the undefined-spawn event record.

``steps`` counts every step the environment took since its reset and
``last_ticks`` how many ticks the last of them ran (more than one while the
player slept or rested). ``first_undefined_step`` is the 0-based index, since
the reset, of the step whose spawn draw first hit the 1.0 edge (-1 while none
has): where a bits comparator stops comparing that environment. A reset starts
all four afresh.
"""

TRAINING_STATS_DTYPE: Final = np.dtype(
    [
        ("stall_limit", np.int32),
        ("last_gain", np.int32),
        ("escape", np.uint32, (1,)),
        ("branch", np.int32),
        ("branch_steps", np.int64),
        *((name, STATS_DTYPE[name]) for name in STATS_DTYPE.names or ()),
    ],
)
"""The training-only options' per-environment fields, then ``STATS_DTYPE``'s.

The stall cap's clocks: ``stall_limit``, the steps without achievement reward
that end this episode (0 when its reset's draw left it uncapped), and
``last_gain``, the episode step of its last achievement reward. ``escape`` is
the ``rand_r`` stream those draws come from, apart from the game's so the
game draws as it would without the cap. ``branch`` is 1 while the environment
plays a practice branch, whose end is not logged, and ``branch_steps`` counts
the steps it has played in branches. An environment whose options are all off
keeps ``STATS_DTYPE``, so its bytes are PufferLib's accumulators' alone.

The training fields lead: Numba takes a record whose fields begin with another
record's for a subtype of it (width subtyping, experimental), and would then
run a kernel compiled for ``STATS_DTYPE`` on these records instead of
compiling one for them.
"""


class EnvStats(Protocol):
    """One ``STATS_DTYPE`` record as a kernel sees it."""

    episode_return_accum: np.float32
    episode_length_accum: int
    max_floor_accum: int
    steps: int
    last_ticks: int
    undefined_spawns: int
    first_undefined_step: int
    log: Log


class TrainingStats(EnvStats, Protocol):
    """One ``TRAINING_STATS_DTYPE`` record as a kernel sees it."""

    stall_limit: int
    last_gain: int
    escape: Array1[np.uint32]
    """uint32 [1]: the stall draws' ``rand_r`` stream."""

    branch: int
    branch_steps: int


class Log(Protocol):
    """One ``LOG_DTYPE`` record as a kernel sees it."""

    perf: np.float32
    achievement_rate: np.float32
    score: np.float32
    episode_return: np.float32
    episode_length: np.float32
    floors: Array1[np.float32]
    achievements: Array1[np.float32]
    n: np.float32


class Archive(NamedTuple):
    """Practice's saved worlds and the donors that fill them, as ``game.archive`` keeps them.

    Its kernels mutate every array in place. Entry ``level * per_level +
    index`` holds a return level's ``index``-th world; a level's entries are
    its first ``sizes[level]``.

    Attributes:
      states: ``STATE_DTYPE [slots]``, each entry's world.
      rngs: ``uint32 [slots]``, its game stream.
      stats: ``TRAINING_STATS_DTYPE [slots]``, its accumulators and clocks.
      actions: ``float32 [slots]``, the action that led to it.
      rewards: ``float32 [slots]``, that step's reward.
      worlds: ``uint64 [slots]``, the hash of its episode's first map.
      sizes: ``int64 [levels]``, entries held per level.
      reach: ``float64 [levels]``, decayed donor episodes reaching each level.
      weights: ``float64 [levels]``, the level draw's weights, recomputed
        before each rollout's restores.
      stream: ``uint32 [1]``, the archive's ``rand_r`` stream.
      donor_levels: ``int64 [donors]``, each donor episode's highest level.
      donor_worlds: ``uint64 [donors]``, each donor episode's world hash.
      donor_steps: ``int64 [donors]``, steps each donor was observed this
        episode; 0 until its episode's first.
      counts: ``int64 [buffers]``, each buffer's environment steps; the
        stats count each environment's branch steps.
      save_slots: ``int32 [num_envs]``, the entry each donor saved into on its
        last step, else -1.
      restore_slots: ``int32 [num_envs]``, the entry each row was restored
        from for the next step, else -1; cleared by that step.
      first_donor: The first donor row.
      envs_per_buffer: Rows per buffer, which index ``counts``.
      per_level: Entries a level holds: the slots over the levels.
      per_world: Entries a level keeps from one world.
      level_width: Achievement return per level, float32 as the return is.
      reach_decay: What every reach is multiplied by at a donor episode start.

    """

    states: NDArray[np.void]
    rngs: NDArray[np.uint32]
    stats: NDArray[np.void]
    actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    worlds: NDArray[np.uint64]
    sizes: NDArray[np.int64]
    reach: NDArray[np.float64]
    weights: NDArray[np.float64]
    stream: NDArray[np.uint32]
    donor_levels: NDArray[np.int64]
    donor_worlds: NDArray[np.uint64]
    donor_steps: NDArray[np.int64]
    counts: NDArray[np.int64]
    save_slots: NDArray[np.int32]
    restore_slots: NDArray[np.int32]
    first_donor: int
    envs_per_buffer: int
    per_level: int
    per_world: int
    level_width: np.float32
    reach_decay: float


class ArchiveView(Protocol):
    """An :class:`Archive` as the kernels read it."""

    @property
    def states(self) -> Records[EnvState]:
        """``Archive.states``."""
        ...

    @property
    def rngs(self) -> Array1[np.uint32]:
        """``Archive.rngs``."""
        ...

    @property
    def stats(self) -> Records[TrainingStats]:
        """``Archive.stats``."""
        ...

    @property
    def actions(self) -> Array1[np.float32]:
        """``Archive.actions``."""
        ...

    @property
    def rewards(self) -> Array1[np.float32]:
        """``Archive.rewards``."""
        ...

    @property
    def worlds(self) -> Array1[np.uint64]:
        """``Archive.worlds``."""
        ...

    @property
    def sizes(self) -> Array1[int]:
        """``Archive.sizes``."""
        ...

    @property
    def reach(self) -> Array1[float]:
        """``Archive.reach``."""
        ...

    @property
    def weights(self) -> Array1[float]:
        """``Archive.weights``."""
        ...

    @property
    def stream(self) -> Array1[np.uint32]:
        """``Archive.stream``."""
        ...

    @property
    def donor_levels(self) -> Array1[int]:
        """``Archive.donor_levels``."""
        ...

    @property
    def donor_worlds(self) -> Array1[np.uint64]:
        """``Archive.donor_worlds``."""
        ...

    @property
    def donor_steps(self) -> Array1[int]:
        """``Archive.donor_steps``."""
        ...

    @property
    def counts(self) -> Array1[int]:
        """``Archive.counts``."""
        ...

    @property
    def save_slots(self) -> Array1[int]:
        """``Archive.save_slots``."""
        ...

    @property
    def restore_slots(self) -> Array1[int]:
        """``Archive.restore_slots``."""
        ...

    @property
    def first_donor(self) -> int:
        """``Archive.first_donor``."""
        ...

    @property
    def envs_per_buffer(self) -> int:
        """``Archive.envs_per_buffer``."""
        ...

    @property
    def per_level(self) -> int:
        """``Archive.per_level``."""
        ...

    @property
    def per_world(self) -> int:
        """``Archive.per_world``."""
        ...

    @property
    def level_width(self) -> np.float32:
        """``Archive.level_width``."""
        ...

    @property
    def reach_decay(self) -> float:
        """``Archive.reach_decay``."""
        ...


def byte_offset(name: LiteralString, dtype: np.dtype[np.void] = STATE_DTYPE) -> int:
    """Return where field ``name`` of ``dtype`` begins, in bytes from the record's start.

    Args:
      name: A top-level field.
      dtype: A structured dtype.

    Returns:
      offset: The field's C ``offsetof``.

    """
    if dtype.fields is None:
        raise ValueError("a structured dtype is required")
    return int(dtype.fields[name][1])


def semantic_bytes(dtype: np.dtype[np.void] = STATE_DTYPE) -> NDArray[np.bool]:
    """Return a ``bool [itemsize]`` mask of the bytes that belong to a C field.

    The C structs carry padding -- one byte after ``Mobs.mask`` and four after
    ``State.timestep``, the ``PADDING`` fields here -- that the C env never
    writes, so a dump made from a struct copy holds whatever those bytes were.
    A bit-for-bit comparison of two states covers the C fields' bytes only.

    Args:
      dtype: A structured dtype.

    Returns:
      mask: True where a byte belongs to some field other than ``PADDING``, at
        any nesting.

    """
    mask = np.zeros(dtype.itemsize, dtype=np.bool_)
    if dtype.fields is None:
        raise ValueError("a structured dtype is required")
    for name, entry in dtype.fields.items():
        if name == PADDING:
            continue
        field_dtype, offset = entry[:2]
        base = field_dtype.base if field_dtype.subdtype is not None else field_dtype
        count = (
            int(np.prod(field_dtype.shape)) if field_dtype.subdtype is not None else 1
        )
        if base.fields is None:
            mask[offset : offset + field_dtype.itemsize] = True
            continue
        inner = semantic_bytes(base)
        for k in range(count):
            start = offset + k * base.itemsize
            mask[start : start + base.itemsize] = inner
    return mask


def new_states(num_envs: int) -> NDArray[np.void]:
    """Return ``num_envs`` zeroed world records, as C's ``calloc`` of ``State``.

    Args:
      num_envs: Records to allocate.

    Returns:
      states: A zeroed ``[num_envs]`` array of ``STATE_DTYPE``.

    """
    return np.zeros(num_envs, dtype=STATE_DTYPE)


def new_stats(
    num_envs: int,
    dtype: np.dtype[np.void] = STATS_DTYPE,
) -> NDArray[np.void]:
    """Return ``num_envs`` zeroed stats records with no undefined event recorded.

    Args:
      num_envs: Records to allocate.
      dtype: ``STATS_DTYPE``, or ``TRAINING_STATS_DTYPE`` for an environment
        with a training-only option on.

    Returns:
      stats: A ``[num_envs]`` array of ``dtype``.

    """
    stats = np.zeros(num_envs, dtype=dtype)
    stats["first_undefined_step"] = -1
    return stats


def env_stats(stats: NDArray[np.void], index: int) -> EnvStats:
    """Return one record of a ``STATS_DTYPE`` or ``TRAINING_STATS_DTYPE`` array, typed.

    The record is :func:`env_state`'s kind: Python reads and writes its fields
    as attributes, and a kernel takes it.

    Args:
      stats: A ``STATS_DTYPE`` or ``TRAINING_STATS_DTYPE`` array.
      index: Which record.

    Returns:
      stats: The record, a view into ``stats``.

    """
    if stats.dtype not in (STATS_DTYPE, TRAINING_STATS_DTYPE):
        raise ValueError("stats must be a STATS_DTYPE or TRAINING_STATS_DTYPE array")
    return cast(EnvStats, stats.view(np.recarray)[index])


def env_log(logs: NDArray[np.void], index: int) -> Log:
    """Return one record of a ``LOG_DTYPE`` array, typed as :func:`env_stats`."""
    if logs.dtype != LOG_DTYPE:
        raise ValueError("logs must be a LOG_DTYPE array")
    return cast(Log, logs.view(np.recarray)[index])


def training_stats(stats: NDArray[np.void], index: int) -> TrainingStats:
    """Return one record of a ``TRAINING_STATS_DTYPE`` array, typed as :func:`env_stats`."""
    if stats.dtype != TRAINING_STATS_DTYPE:
        raise ValueError("stats must be a TRAINING_STATS_DTYPE array")
    return cast(TrainingStats, stats.view(np.recarray)[index])


def env_state(states: NDArray[np.void], index: int) -> EnvState:
    """Return one record of a ``STATE_DTYPE`` array, typed.

    The record is a ``numpy.record`` view, so Python reads and writes its
    fields as attributes, as a kernel does, and a write lands in ``states``.
    Numba types it as it types ``states[index]``, so a kernel compiled for one
    runs the other.

    Args:
      states: A ``STATE_DTYPE`` array.
      index: Which record.

    Returns:
      state: The record, a view into ``states``.

    """
    if states.dtype != STATE_DTYPE:
        raise ValueError("states must be a STATE_DTYPE array")
    return cast(EnvState, states.view(np.recarray)[index])


def new_logs(num_envs: int) -> NDArray[np.void]:
    """Return ``num_envs`` zeroed episode logs.

    Args:
      num_envs: Records to allocate.

    Returns:
      logs: A zeroed ``[num_envs]`` array of ``LOG_DTYPE``.

    """
    return np.zeros(num_envs, dtype=LOG_DTYPE)
