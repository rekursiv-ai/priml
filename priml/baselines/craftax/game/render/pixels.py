"""Draw one world the way a person would look at it.

This is a viewer, not an observation. The agent reads
:mod:`~priml.baselines.craftax.game.observation`, a flat float vector; a
person reads pixels, and the two have no reason to share code. Which is why
this composites sprites with pygame instead of accumulating masked tensors:
drawing one frame for one worker is a blit loop, and writing it as a batched
tensor program would be slower AND harder to read.

The frame is the player's own 9x11 view -- the same window the policy sees, so
watching a replay shows what the agent knew, not what it could not have known.
Darkness, night, and sleep dim it exactly as they dim the observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os

import numpy as np
import pygame

from priml.baselines.craftax.game import constants, mechanics
from priml.baselines.craftax.game.constants import Action, BlockType, ItemType
from priml.baselines.craftax.game.render import assets, sprites


if TYPE_CHECKING:
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
    ) -> None:
        """Load and scale every sprite.

        Args:
          block_pixels: Edge of one tile in the output image; 64 is the size
            upstream calls "human".
          asset_dir: Where sprites are cached; defaults to the user cache.

        Raises:
          ValueError: The tile size is not positive.

        """
        if block_pixels <= 0:
            raise ValueError("block_pixels must be positive")
        # Headless: this never opens a window, so it runs on a GPU worker and
        # under pytest. ``play`` opens its own display.
        #
        # The driver request comes FIRST. SDL resolves it during the first init
        # of any subsystem and caches the result for the process, so setting it
        # after ``pygame.init()`` binds x11 on a machine with a display and puts
        # a window on the operator's screen. ``setdefault`` so ``play`` -- which
        # wants a real window -- can export its own choice beforehand.
        if pygame.display.get_surface() is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        if not pygame.get_init():
            pygame.init()
        # ``convert_alpha`` needs a pixel format to convert TO, which normally
        # comes from the display. A dummy 1x1 surface supplies one without
        # opening a window, and converting matters: an unconverted surface
        # blits through a per-pixel format check on every draw.
        if pygame.display.get_surface() is None:
            pygame.display.init()
            pygame.display.set_mode((1, 1))
        self.block_pixels = block_pixels
        self._directory = asset_dir
        self._cache: dict[str, pygame.Surface] = {}
        for name in sprites.every_sprite():
            self._load(name)

    def render(self, state: EnvState, *, index: int = 0) -> np.ndarray:
        """Draw one worker's view.

        Args:
          state: The batched world.
          index: Which worker to draw.

        Returns:
          frame: ``[height, width, 3]`` uint8 RGB.

        """
        rows, columns = constants.OBS_DIM
        surface = pygame.Surface(
            (columns * self.block_pixels, rows * self.block_pixels)
        )

        self._draw_terrain(surface, state, index)
        self._draw_creatures(surface, state, index)
        self._draw_player(surface, state, index)
        self._shade(surface, state, index)

        # pygame is column-major in its array view; transpose back to the
        # row-major convention every image tool expects.
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def _draw_terrain(
        self,
        surface: pygame.Surface,
        state: EnvState,
        index: int,
    ) -> None:
        """Fill every tile with its block, then the item lying on it."""
        rows, columns = constants.OBS_DIM
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
                if not (
                    0 <= map_row < blocks.shape[0] and 0 <= map_column < blocks.shape[1]
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
        rows, columns = constants.OBS_DIM
        top, left = self._corner(state, index)

        for slot in range(int(mask.shape[0])):
            if not bool(mask[slot]):
                continue
            row = int(positions[slot, 0]) - top
            column = int(positions[slot, 1]) - left
            if not (0 <= row < rows and 0 <= column < columns):
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
        rows, columns = constants.OBS_DIM
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
        rows, columns = constants.OBS_DIM
        level = int(state.player_level[index])
        light = state.light_map[index, level]
        top, left = self._corner(state, index)

        # Unlit tiles go fully black rather than dim: darkness genuinely hides
        # the world here, exactly as it does in the observation.
        shadow = pygame.Surface((self.block_pixels, self.block_pixels))
        shadow.fill((0, 0, 0))
        for row in range(rows):
            for column in range(columns):
                map_row, map_column = top + row, left + column
                if not (
                    0 <= map_row < light.shape[0] and 0 <= map_column < light.shape[1]
                ):
                    continue
                lit = float(light[map_row, map_column])
                if lit >= 1.0:
                    continue
                shadow.set_alpha(int((1.0 - lit) * 255))
                surface.blit(
                    shadow,
                    (column * self.block_pixels, row * self.block_pixels),
                )

        # Night only falls on the surface; the caves are lit by their own
        # rules and do not brighten at dawn.
        daylight = 1.0 if level > 0 else float(state.light_level[index])
        if daylight < 1.0:
            night = pygame.Surface(surface.get_size())
            night.fill(sprites.NIGHT_COLOR)
            night.set_alpha(int((1.0 - daylight) * 255))
            surface.blit(night, (0, 0))

        if bool(state.is_sleeping[index]):
            closed = pygame.Surface(surface.get_size())
            closed.fill((0, 0, 0))
            closed.set_alpha(128)
            surface.blit(closed, (0, 0))

    def _corner(self, state: EnvState, index: int) -> tuple[int, int]:
        """Return the map coordinate of the view's top-left tile."""
        rows, columns = constants.OBS_DIM
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
        surface = pygame.image.load(str(path)).convert_alpha()
        surface = pygame.transform.scale(
            surface,
            (self.block_pixels, self.block_pixels),
        )
        self._cache[name] = surface
        return surface
