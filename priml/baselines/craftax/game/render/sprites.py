"""Which sprite draws which thing.

One table per enum, in enum order, so a lookup is an index rather than a
dictionary miss. The names are upstream's: matching them is what makes a frame
here recognizably the same game as a frame there.

Two blocks have no file. ``OUT_OF_BOUNDS`` is flat grey and ``DARKNESS`` is
flat black -- they are absences, not objects, and upstream draws them as solid
colour rather than art.
"""

from __future__ import annotations

from typing import Final


BLOCK_SPRITES: Final[tuple[str, ...]] = (
    "debug_tile.png",  # INVALID
    "",  # OUT_OF_BOUNDS: solid grey
    "grass.png",
    "water.png",
    "stone.png",
    "tree.png",
    "wood.png",
    "path.png",
    "coal.png",
    "iron.png",
    "diamond.png",
    "table.png",
    "furnace.png",
    "sand.png",
    "lava.png",
    "plant_on_grass.png",
    "ripe_plant_on_grass.png",
    "wall2.png",
    "",  # DARKNESS: solid black
    "wall_moss.png",
    "stalagmite.png",
    "sapphire.png",
    "ruby.png",
    "chest.png",
    "fountain.png",
    "fire_grass.png",
    "ice_grass.png",
    "gravel.png",
    "fire_tree.png",
    "ice_shrub.png",
    "enchantment_table_fire.png",
    "enchantment_table_ice.png",
    "necromancer.png",
    "grave.png",
    "grave2.png",
    "grave3.png",
    "necromancer_vulnerable.png",
)
"""One sprite per ``BlockType``, in value order."""

OUT_OF_BOUNDS_COLOR: Final = (128, 128, 128)
"""Flat grey for tiles past the edge of the map."""

DARKNESS_COLOR: Final = (0, 0, 0)
"""Flat black for a tile the player cannot see into."""

ITEM_SPRITES: Final[tuple[str, ...]] = (
    "",  # NONE: nothing lies here
    "torch_in_inventory.png",
    "ladder_down.png",
    "ladder_up.png",
    "ladder_down_blocked.png",
)
"""One sprite per ``ItemType``, in value order."""

PLAYER_SPRITES: Final[tuple[str, ...]] = (
    "player-left.png",
    "player-right.png",
    "player-up.png",
    "player-down.png",
    "player-sleep.png",
)
"""Facing sprites, indexed by ``Action`` direction minus one; sleep is last."""

MELEE_SPRITES: Final[tuple[str, ...]] = (
    "zombie.png",
    "gnome_warrior.png",
    "orc_soldier.png",
    "lizard.png",
    "knight.png",
    "troll.png",
    "pigman.png",
    "frost_troll.png",
)
"""Creatures that attack in contact, by species id."""

PASSIVE_SPRITES: Final[tuple[str, ...]] = (
    "cow.png",
    "bat.png",
    "snail.png",
)
"""Creatures that flee, by species id."""

RANGED_SPRITES: Final[tuple[str, ...]] = (
    "skeleton.png",
    "gnome_archer.png",
    "orc_mage.png",
    "kobold.png",
    "knight_archer.png",
    "deep_thing.png",
    "fire_elemental.png",
    "ice_elemental.png",
)
"""Creatures that shoot, by species id."""

PROJECTILE_SPRITES: Final[tuple[str, ...]] = (
    "arrow-up.png",
    "dagger.png",
    "fireball.png",
    "iceball.png",
    "arrow-up.png",
    "slimeball.png",
    "fireball.png",
    "iceball.png",
)
"""Flying objects, by ``ProjectileType``."""

NIGHT_COLOR: Final = (0, 16, 64)
"""The blue the world fades toward after dark."""


def every_sprite() -> tuple[str, ...]:
    """Return every sprite file the viewer can draw.

    Used to warm the cache in one pass rather than paying a round trip per
    sprite on the first frame.

    Returns:
      names: Sorted, de-duplicated file names.

    """
    names = {
        *BLOCK_SPRITES,
        *ITEM_SPRITES,
        *PLAYER_SPRITES,
        *MELEE_SPRITES,
        *PASSIVE_SPRITES,
        *RANGED_SPRITES,
        *PROJECTILE_SPRITES,
    }
    return tuple(sorted(name for name in names if name))
