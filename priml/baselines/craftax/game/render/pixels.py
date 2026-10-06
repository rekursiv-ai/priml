"""Draw one world the way a person would look at it.

This is a viewer, not an observation. The agent reads
:mod:`~priml.baselines.craftax.game.observation`, a flat float vector; a
person reads pixels, and the two have no reason to share code. Which is why
this composites sprites with pygame instead of accumulating masked tensors:
drawing one frame for one worker is a blit loop, and writing it as a batched
tensor program would be slower AND harder to read.

The frame is the player's own view, 9x11 unless told otherwise -- pass the
``view`` the policy was configured with and a replay shows the window the agent
saw. Darkness, night, and sleep shade it, graded by how lit each tile is the
way upstream's renderer shades; the observation instead hides a tile outright
below a fixed light threshold.

Nothing here initializes SDL or chooses its video driver. Surfaces load, scale,
and composite without either, and SDL binds a driver once per process, so a
viewer that bound one would decide the display of every later window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import Action, BlockType, ItemType
from priml.baselines.craftax.game.render import assets, sprites


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.craftax.game.state import EnvState, Mobs


class Renderer:
    """Draws worlds, reusing one set of scaled sprites.

    Constructing this downloads and scales every sprite, which is why it is an
    object rather than a function: a replay draws ten thousand frames and
    should pay that once.
    """

    def __init__(
        self,
        *,
        block_pixels: int = 64,
        asset_dir: Path | None = None,
        view: tuple[int, int] = constants.OBS_DIM,
    ) -> None:
        """Load and scale every sprite.

        Args:
          block_pixels: Edge of one tile in the output image; 64 is the size
            upstream calls "human".
          asset_dir: Where sprites are cached; defaults to the user cache.
          view: Tiles drawn around the player, ``(rows, columns)``; pass the
            environment's ``view`` to draw what its policy sees.

        Raises:
          ValueError: The tile size or the view is not positive.

        """
        if block_pixels <= 0:
            raise ValueError("block_pixels must be positive")
        if min(view) <= 0:
            raise ValueError("view must be positive in both dimensions")
        # No ``pygame.init`` and no ``set_mode``: loading, scaling, and
        # blitting need neither, while an init binds SDL's video driver for
        # the whole process and ``set_mode`` maps a window. Sprites are
        # therefore kept unconverted -- ``convert`` and ``convert_alpha`` need
        # a display's pixel format. ``play`` opens its own display.
        self.block_pixels = block_pixels
        self.view = view
        self._directory = asset_dir
        self._cache: dict[str, pygame.Surface] = {}
        for name in sprites.every_sprite():
            self._load(name)

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(height, width)`` of every frame ``render`` returns.

        Exists so a caller sizing a video writer does not restate
        ``view * block_pixels``: a writer told one geometry and fed another
        produces a file ffmpeg cannot read, and closes without raising.
        """
        rows, columns = self.view
        return rows * self.block_pixels, columns * self.block_pixels

    def render(self, state: EnvState, *, index: int = 0) -> np.ndarray:
        """Draw one worker's view.

        Args:
          state: The batched world.
          index: Which worker to draw.

        Returns:
          frame: ``[height, width, 3]`` uint8 RGB.

        """
        height, width = self.frame_shape
        surface = pygame.Surface((width, height))

        self._draw_terrain(surface, state, index)
        self._draw_creatures(surface, state, index)
        self._draw_player(surface, state, index)
        self._shade(surface, state, index)

        # ``pygame`` is column-major in its array view; transpose back to the
        # row-major convention every image tool expects.
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def _draw_terrain(
        self,
        surface: pygame.Surface,
        state: EnvState,
        index: int,
    ) -> None:
        """Fill every tile with its block, then the item lying on it."""
        rows, columns = self.view
        level = int(state.player_level[index])
        blocks = state.map[index, level]
        items = state.item_map[index, level]
        top, left = self._corner(state, index)

        # A ladder down is drawn blocked until the floor is cleared, which is
        # the only cue that descending is not yet allowed.
        cleared = bool(
            state.monsters_killed[index, level]
            >= constants.MONSTERS_KILLED_TO_CLEAR_LEVEL,
        )
        vulnerable = bool(mechanics.is_boss_vulnerable(state)[index])

        for row in range(rows):
            for column in range(columns):
                position = (column * self.block_pixels, row * self.block_pixels)
                map_row, map_column = top + row, left + column
                if map_row not in range(len(blocks)) or map_column not in range(
                    len(blocks[0]),
                ):
                    surface.fill(
                        sprites.OUT_OF_BOUNDS_COLOR,
                        (*position, self.block_pixels, self.block_pixels),
                    )
                    continue

                block = int(blocks[map_row, map_column])
                if block == int(BlockType.NECROMANCER) and vulnerable:
                    block = int(BlockType.NECROMANCER_VULNERABLE)
                self._blit_block(surface, block, position)

                item = int(items[map_row, map_column])
                if item == int(ItemType.LADDER_DOWN) and not cleared:
                    item = int(ItemType.LADDER_DOWN_BLOCKED)
                name = sprites.ITEM_SPRITES[item]
                if name:
                    surface.blit(self._load(name), position)

    def _draw_creatures(
        self,
        surface: pygame.Surface,
        state: EnvState,
        index: int,
    ) -> None:
        """Draw every live creature and projectile inside the view."""
        for mobs, names in (
            (state.passive_mobs, sprites.PASSIVE_SPRITES),
            (state.melee_mobs, sprites.MELEE_SPRITES),
            (state.ranged_mobs, sprites.RANGED_SPRITES),
            (state.mob_projectiles, sprites.PROJECTILE_SPRITES),
            (state.player_projectiles, sprites.PROJECTILE_SPRITES),
        ):
            self._draw_mobs(surface, state, index, mobs=mobs, names=names)

    def _draw_mobs(
        self,
        surface: pygame.Surface,
        state: EnvState,
        index: int,
        *,
        mobs: Mobs,
        names: tuple[str, ...],
    ) -> None:
        """Draw one creature array's live slots."""
        level = int(state.player_level[index])
        mask = mobs.mask[index, level]
        positions = mobs.position[index, level]
        species = mobs.type_id[index, level]
        rows, columns = self.view
        top, left = self._corner(state, index)

        valid_rows, valid_columns = range(rows), range(columns)
        for slot in range(int(mask.shape[0])):
            if not bool(mask[slot]):
                continue
            row = int(positions[slot, 0]) - top
            if row not in valid_rows:
                continue
            column = int(positions[slot, 1]) - left
            if column not in valid_columns:
                continue
            name = names[int(species[slot]) % len(names)]
            surface.blit(
                self._load(name),
                (column * self.block_pixels, row * self.block_pixels),
            )

    def _draw_player(
        self,
        surface: pygame.Surface,
        state: EnvState,
        index: int,
    ) -> None:
        """Draw the player at the centre, facing the way they last moved."""
        rows, columns = self.view
        if bool(state.is_sleeping[index]):
            sprite = sprites.PLAYER_SPRITES[-1]
        else:
            direction = int(state.player_direction[index])
            facing = {
                int(Action.LEFT): 0,
                int(Action.RIGHT): 1,
                int(Action.UP): 2,
                int(Action.DOWN): 3,
            }.get(direction, 3)
            sprite = sprites.PLAYER_SPRITES[facing]
        surface.blit(
            self._load(sprite),
            (
                (columns // 2) * self.block_pixels,
                (rows // 2) * self.block_pixels,
            ),
        )

    def _shade(
        self,
        surface: pygame.Surface,
        state: EnvState,
        index: int,
    ) -> None:
        """Darken unlit tiles, then the whole frame for night and sleep."""
        rows, columns = self.view
        level = int(state.player_level[index])
        light = state.light_map[index, level]
        top, left = self._corner(state, index)

        # Each tile darkens in proportion to its missing light, so only a fully
        # unlit tile goes black. Graded like upstream's renderer, not cut at the
        # observation's threshold: a viewer shows how lit a tile is.
        shadow = pygame.Surface((self.block_pixels, self.block_pixels))
        shadow.fill((0, 0, 0))
        for row in range(rows):
            for column in range(columns):
                map_row, map_column = top + row, left + column
                if map_row not in range(len(light)) or map_column not in range(
                    len(light[0]),
                ):
                    continue
                lit = float(light[map_row, map_column])
                alpha = max(0, int((1.0 - lit) * 255))
                if alpha:
                    shadow.set_alpha(alpha)
                    surface.blit(
                        shadow,
                        (column * self.block_pixels, row * self.block_pixels),
                    )

        # Night only falls on the surface; the caves are lit by their own
        # rules and do not brighten at dawn.
        night_alpha = max(
            0,
            0 if level > 0 else int((1.0 - float(state.light_level[index])) * 255),
        )
        if night_alpha:
            night = pygame.Surface(surface.get_size())
            night.fill(sprites.NIGHT_COLOR)
            night.set_alpha(night_alpha)
            surface.blit(night)

        if bool(state.is_sleeping[index]):
            closed = pygame.Surface(surface.get_size())
            closed.fill((0, 0, 0))
            closed.set_alpha(128)
            surface.blit(closed)

    def _corner(self, state: EnvState, index: int) -> tuple[int, int]:
        """Return the map coordinate of the view's top-left tile."""
        rows, columns = self.view
        return (
            int(state.player_position[index, 0]) - rows // 2,
            int(state.player_position[index, 1]) - columns // 2,
        )

    def _blit_block(
        self,
        surface: pygame.Surface,
        block: int,
        position: tuple[int, int],
    ) -> None:
        """Draw one block, as art or as flat colour."""
        name = sprites.BLOCK_SPRITES[block]
        if name:
            surface.blit(self._load(name), position)
            return
        colour = (
            sprites.DARKNESS_COLOR
            if block == int(BlockType.DARKNESS)
            else sprites.OUT_OF_BOUNDS_COLOR
        )
        surface.fill(colour, (*position, self.block_pixels, self.block_pixels))

    def _load(self, name: str) -> pygame.Surface:
        """Return one sprite, scaled to the tile size and cached."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        path = assets.fetch(name, directory=self._directory)
        # Loaded unconverted: ``convert_alpha`` requires a display surface,
        # and creating one to satisfy it would open a window.
        surface = pygame.image.load(str(path))
        surface = pygame.transform.scale(
            surface,
            (self.block_pixels, self.block_pixels),
        )
        self._cache[name] = surface
        return surface
