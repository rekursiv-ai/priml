"""Fixed properties of the Craftax world.

Everything here is a fact about the game rather than a tunable: the block and
item vocabularies, which blocks stop movement, how much damage each mob deals
on each floor, and what each achievement is worth. They are ``Final`` for that
reason -- an experiment that wants different numbers is playing a different
game, not running a variant.

Tables are indexed by the enum values below, so ``SOLID_BLOCK[BlockType.STONE]``
reads as the question it answers. Every table is a CPU tensor; a caller moves
what it needs to its device once, at construction.

References:
    https://arxiv.org/abs/2402.16801
        Matthews et al. 2024. Craftax: a lightning-fast benchmark for
        open-ended reinforcement learning.

"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import torch


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


class Action(IntEnum):
    """One of the 43 things the agent may attempt each step."""

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


class MobClass(IntEnum):
    """Which of the five mob arrays a creature is stored in."""

    PASSIVE = 0
    MELEE = 1
    RANGED = 2
    MOB_PROJECTILE = 3
    PLAYER_PROJECTILE = 4


class ProjectileType(IntEnum):
    """The visual and damage identity of a flying object."""

    ARROW = 0
    DAGGER = 1
    FIREBALL = 2
    ICEBALL = 3
    ARROW2 = 4
    SLIMEBALL = 5
    FIREBALL2 = 6
    ICEBALL2 = 7


class Achievement(IntEnum):
    """One of the 67 one-time goals whose unlock is the reward signal."""

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
    # The tail is deliberately out of order: achievements added after the
    # first release took the next free value instead of renumbering the
    # published ones, and the reward table is indexed by these values, so
    # sorting them would silently re-price the game.
    COLLECT_SAPPHIRE = 54
    COLLECT_RUBY = 59
    MAKE_DIAMOND_PICKAXE = 60
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
    DEFEAT_KNIGHT = 65
    DEFEAT_ARCHER = 66
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
    LEARN_FIREBALL = 55
    CAST_FIREBALL = 56
    LEARN_ICEBALL = 57
    CAST_ICEBALL = 58
    OPEN_CHEST = 61
    DRINK_POTION = 62
    ENCHANT_SWORD = 63
    ENCHANT_ARMOUR = 64


OBS_DIM: Final = (9, 11)
"""Rows and columns of the local view the agent observes."""

MAX_OBS_DIM: Final = max(OBS_DIM)
"""Padding width that guarantees a full view at any map position."""

INVENTORY_OBS_SIZE: Final = 51
"""Scalar features appended to the flattened local view."""

NUM_LEVELS: Final = 9
"""Floors of the world, from the overworld down to the boss level."""

MAP_SIZE: Final = (48, 48)
"""Rows and columns of one floor."""

MAX_MELEE_MOBS: Final = 3
MAX_PASSIVE_MOBS: Final = 3
MAX_RANGED_MOBS: Final = 2
MAX_MOB_PROJECTILES: Final = 3
MAX_PLAYER_PROJECTILES: Final = 3
MAX_GROWING_PLANTS: Final = 10

MONSTERS_KILLED_TO_CLEAR_LEVEL: Final = 8
"""Kills required before a floor's ladder down unblocks."""

BOSS_FIGHT_EXTRA_DAMAGE: Final = 0.5
BOSS_FIGHT_SPAWN_TURNS: Final = 7

MAX_TIMESTEPS: Final = 100_000
"""Steps before an episode is cut off regardless of the player's state."""

DAY_LENGTH: Final = 300
"""Steps in one full day/night cycle."""

MOB_DESPAWN_DISTANCE: Final = 14
MAX_ATTRIBUTE: Final = 5

REWARD_CEILING: Final = 226.0
"""Total achievement reward available, the score's normalizing denominator."""


def _directions() -> torch.Tensor:
    """Map every action to the step it moves the player.

    The table spans the WHOLE action space, not just the four movement
    actions: the player's facing is stored as an action value and any action
    may be looked up here, so a table sized to the movement actions alone
    would index out of bounds rather than return the intended zero step.
    """
    steps = torch.zeros((len(Action), 2), dtype=torch.int32)
    steps[Action.LEFT] = torch.tensor([0, -1], dtype=torch.int32)
    steps[Action.RIGHT] = torch.tensor([0, 1], dtype=torch.int32)
    steps[Action.UP] = torch.tensor([-1, 0], dtype=torch.int32)
    steps[Action.DOWN] = torch.tensor([1, 0], dtype=torch.int32)
    return steps


DIRECTIONS: Final = _directions()
"""The step each action moves the player, indexed by ``Action``."""

CLOSE_BLOCKS: Final = torch.tensor(
    [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]],
    dtype=torch.int32,
)
"""The eight neighbors of a tile, orthogonal first."""

SOLID_BLOCK: Final = torch.zeros(len(BlockType), dtype=torch.bool)
SOLID_BLOCK[
    [
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
        BlockType.SAPPHIRE,
        BlockType.RUBY,
        BlockType.CHEST,
        BlockType.FOUNTAIN,
        BlockType.FIRE_TREE,
        BlockType.ENCHANTMENT_TABLE_FIRE,
        BlockType.ENCHANTMENT_TABLE_ICE,
        BlockType.NECROMANCER,
        BlockType.GRAVE,
        BlockType.GRAVE2,
        BlockType.GRAVE3,
    ]
] = True
"""Whether a block stops movement.

``NECROMANCER_VULNERABLE`` is deliberately absent: the boss stops blocking
exactly while it is exposed, which is what lets the player reach it."""

CAN_PLACE_ITEM_ON: Final = torch.zeros(len(BlockType), dtype=torch.bool)
CAN_PLACE_ITEM_ON[
    [
        BlockType.GRASS,
        BlockType.SAND,
        BlockType.PATH,
        BlockType.FIRE_GRASS,
        BlockType.ICE_GRASS,
    ]
] = True
"""Whether a torch or ladder may be placed on a block."""

FLOOR_MOB_TYPE: Final = torch.tensor(
    [
        [0, 0, 0],
        [2, 2, 2],
        [1, 1, 1],
        [2, 3, 3],
        [2, 4, 4],
        [1, 5, 5],
        [1, 6, 6],
        [1, 7, 7],
        [0, 0, 0],
    ],
    dtype=torch.int32,
)
"""Per floor, the ``(passive, melee, ranged)`` species that spawns there."""

FLOOR_MOB_SPAWN_CHANCE: Final = torch.tensor(
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
    dtype=torch.float32,
)
"""Per floor, spawn probability for ``(passive, melee, ranged, melee at night)``."""

_LAND: Final = [False, True, True]
_FLYING: Final = [False, False, False]
_AQUATIC: Final = [True, False, True]
_AMPHIBIAN: Final = [False, False, True]

MOB_COLLIDES_WITH: Final = torch.tensor(
    [
        [_LAND, _LAND, _LAND, _FLYING],
        [_FLYING, _LAND, _LAND, _FLYING],
        [_LAND, _LAND, _LAND, _FLYING],
        [_LAND, _AMPHIBIAN, _LAND, _FLYING],
        [_LAND, _LAND, _LAND, _FLYING],
        [_LAND, _LAND, _AQUATIC, _FLYING],
        [_LAND, _LAND, _FLYING, _FLYING],
        [_LAND, _LAND, _FLYING, _FLYING],
        [_LAND, _LAND, _LAND, _FLYING],
    ],
    dtype=torch.bool,
)
"""Per floor and mob class, whether ``(path, water, lava)`` blocks movement."""

MOB_DAMAGE: Final = torch.tensor(
    [
        [[0.0, 0, 0], [2.0, 0, 0], [0.0, 0, 0], [2.0, 0, 0]],
        [[0.0, 0, 0], [4.0, 0, 0], [0.0, 0, 0], [4.0, 0, 0]],
        [[0.0, 0, 0], [3.0, 0, 0], [0.0, 0, 0], [0.0, 3, 0]],
        [[0.0, 0, 0], [5.0, 0, 0], [0.0, 0, 0], [0.0, 0, 3]],
        [[0.0, 0, 0], [6.0, 0, 0], [0.0, 0, 0], [5.0, 0, 0]],
        [[0.0, 0, 0], [6.0, 1, 1], [0.0, 0, 0], [4.0, 3, 3]],
        [[0.0, 0, 0], [3.0, 5, 0], [0.0, 0, 0], [3.0, 5, 0]],
        [[0.0, 0, 0], [4.0, 0, 5], [0.0, 0, 0], [4.0, 0, 5]],
    ],
    dtype=torch.float32,
)
"""Per species and mob class, ``(physical, fire, ice)`` damage dealt."""

MOB_HEALTH: Final = torch.tensor(
    [
        [3.0, 5.0, 3.0, 0.0],
        [4.0, 7.0, 5.0, 0.0],
        [6.0, 9.0, 6.0, 0.0],
        [8.0, 11.0, 8.0, 0.0],
        [0.0, 12.0, 12.0, 0.0],
        [0.0, 20.0, 4.0, 0.0],
        [0.0, 20.0, 14.0, 0.0],
        [0.0, 24.0, 16.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
"""Per floor and mob class, the health a freshly spawned creature has."""

MOB_DEFENSE: Final = torch.tensor(
    [
        [[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.2, 0, 0], [0.0, 0, 0], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.9, 1, 0], [0.9, 1, 0], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.9, 0, 1], [0.9, 0, 1], [0.0, 0, 0]],
        [[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]],
    ],
    dtype=torch.float32,
)
"""Per floor and mob class, the fraction of ``(physical, fire, ice)`` resisted."""

RANGED_MOB_PROJECTILE: Final = torch.tensor(
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
    dtype=torch.int32,
)
"""What each ranged species shoots."""


def _achievement_reward() -> torch.Tensor:
    """Price each achievement by how deep into the game it sits.

    The basic survival loop pays 1, mid-game crafting and the first two
    dungeons pay 3, the middle floors and magic pay 5, and the late realms and
    their bosses pay 8. Tier membership is spelled out rather than sliced by
    index because the achievement values are not in difficulty order.
    """
    tiers = {
        3.0: (
            Achievement.MAKE_DIAMOND_SWORD,
            Achievement.MAKE_IRON_ARMOUR,
            Achievement.MAKE_DIAMOND_ARMOUR,
            Achievement.ENTER_GNOMISH_MINES,
            Achievement.ENTER_DUNGEON,
            Achievement.DEFEAT_GNOME_WARRIOR,
            Achievement.DEFEAT_GNOME_ARCHER,
            Achievement.DEFEAT_ORC_SOLIDER,
            Achievement.DEFEAT_ORC_MAGE,
            Achievement.EAT_BAT,
            Achievement.EAT_SNAIL,
            Achievement.FIND_BOW,
            Achievement.FIRE_BOW,
            Achievement.COLLECT_SAPPHIRE,
            Achievement.COLLECT_RUBY,
            Achievement.MAKE_DIAMOND_PICKAXE,
            Achievement.OPEN_CHEST,
            Achievement.DRINK_POTION,
        ),
        5.0: (
            Achievement.ENTER_SEWERS,
            Achievement.ENTER_VAULT,
            Achievement.ENTER_TROLL_MINES,
            Achievement.DEFEAT_LIZARD,
            Achievement.DEFEAT_KOBOLD,
            Achievement.DEFEAT_TROLL,
            Achievement.DEFEAT_DEEP_THING,
            Achievement.LEARN_FIREBALL,
            Achievement.CAST_FIREBALL,
            Achievement.LEARN_ICEBALL,
            Achievement.CAST_ICEBALL,
            Achievement.ENCHANT_SWORD,
            Achievement.ENCHANT_ARMOUR,
            Achievement.DEFEAT_KNIGHT,
            Achievement.DEFEAT_ARCHER,
        ),
        8.0: (
            Achievement.ENTER_FIRE_REALM,
            Achievement.ENTER_ICE_REALM,
            Achievement.ENTER_GRAVEYARD,
            Achievement.DEFEAT_PIGMAN,
            Achievement.DEFEAT_FIRE_ELEMENTAL,
            Achievement.DEFEAT_FROST_TROLL,
            Achievement.DEFEAT_ICE_ELEMENTAL,
            Achievement.DAMAGE_NECROMANCER,
            Achievement.DEFEAT_NECROMANCER,
        ),
    }
    rewards = torch.ones(len(Achievement), dtype=torch.float32)
    for reward, achievements in tiers.items():
        rewards[list(achievements)] = reward
    return rewards


ACHIEVEMENT_REWARD: Final = _achievement_reward()
"""Reward for unlocking each achievement; sums to ``REWARD_CEILING``."""

LEVEL_ACHIEVEMENT: Final = torch.tensor(
    [
        0,
        # Floor 1 is the dungeon and floor 2 the gnomish mines, which is the
        # reverse of the order the achievement names are declared in.
        Achievement.ENTER_DUNGEON,
        Achievement.ENTER_GNOMISH_MINES,
        Achievement.ENTER_SEWERS,
        Achievement.ENTER_VAULT,
        Achievement.ENTER_TROLL_MINES,
        Achievement.ENTER_FIRE_REALM,
        Achievement.ENTER_ICE_REALM,
        Achievement.ENTER_GRAVEYARD,
    ],
    dtype=torch.int32,
)
"""The achievement unlocked by first arriving at each floor."""

MOB_ACHIEVEMENT: Final = torch.tensor(
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
    dtype=torch.int32,
)
"""Per mob class and species, the achievement its defeat unlocks."""


def _torch_light_map() -> torch.Tensor:
    """Build the radial falloff a placed torch casts on its 9x9 neighborhood.

    The squared distance is summed as exact integers before its single
    conversion to float, and the division by the radius is rounded to float32
    before the subtraction, matching the order the reference applies them.

    The 40 off-axis entries still differ from the reference in their last
    mantissa bit, and deliberately so: the reference's square root truncates
    where IEEE-754 rounds to nearest -- ``sqrt(20)`` is ``0x1.1e3779b9...``,
    which this returns as ``0x1.1e377a`` and the reference as ``0x1.1e3778``.
    Reproducing that would mean shipping a deliberately less accurate square
    root. The gap is one part in 8 million of a light level that is compared
    against a 0.05 threshold, so it cannot change a visibility decision.
    """
    offsets = (torch.arange(9, dtype=torch.int32) - 4).abs()
    squared = offsets[:, None] ** 2 + offsets[None, :] ** 2
    scaled = squared.to(torch.float32).sqrt() / 5.0
    return (1.0 - scaled).clamp(0.0, 1.0)


TORCH_LIGHT_MAP: Final = _torch_light_map()
"""Light contributed by a torch, brightest at its own tile."""
