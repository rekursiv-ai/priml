"""Tests for the pixel viewer.

These draw GENERATED sprites, not the real ones: a flat, distinctly-coloured
square per file name. Every property asserted here -- which tile a thing lands
on, what is hidden, what the shading does -- is about placement and masking,
and a solid colour tests those more sharply than art does, because any leak
shows up as an exact colour that should not be there.

It also keeps the unit tier offline. Downloading 143 PNGs to assert that a
distant mob is not drawn would make every run depend on GitHub being up.
``test_a_real_sprite_downloads_and_loads`` covers a genuine asset and is marked
``integration`` for that reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import hashlib
import os

from numpy.typing import NDArray

import numpy as np
import pygame
import pytest
import torch

from priml.baselines.craftax.conftest import generated_world
from priml.baselines.craftax.game import constants
from priml.baselines.craftax.game.constants import Action, BlockType, ItemType
from priml.baselines.craftax.game.render import assets, sprites
from priml.baselines.craftax.game.render.pixels import Renderer
from priml.baselines.craftax.game.state import EnvState, empty_state


if TYPE_CHECKING:
    from pathlib import Path


TILE: Final = 8


@pytest.fixture(scope="module")
def sprite_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write one flat-coloured PNG per sprite the viewer can draw.

    The colour is derived from the file name, so two different sprites are
    always distinguishable and the same sprite is always identical -- which is
    what the placement assertions actually depend on.
    """
    directory = tmp_path_factory.mktemp("sprites")
    for name in sprites.every_sprite():
        colour = _colour(name)
        surface = pygame.Surface((TILE, TILE), flags=pygame.SRCALPHA)
        surface.fill((*colour, 255))
        pygame.image.save(surface, str(directory / name))
    return directory


@pytest.fixture(scope="module")
def renderer(sprite_dir: Path) -> Renderer:
    """One renderer for the module; loading sprites is the expensive part."""
    return Renderer(block_pixels=TILE, asset_dir=sprite_dir)


def _colour(name: str) -> tuple[int, int, int]:
    """Return a stable, distinct colour for one sprite name."""
    digest = hashlib.sha256(name.encode()).digest()
    # Never black and never the out-of-bounds grey: both are drawn as flat
    # fills by the viewer, so a sprite sharing one would make a real leak
    # indistinguishable from correct output.
    return (digest[0] | 0x40, digest[1] | 0x40, digest[2] | 0x41)


def _sprite_colour(name: object) -> tuple[int, int, int]:
    return _colour(cast(str, name))


def _pixel(frame: NDArray[np.uint8], row: int, column: int) -> tuple[int, int, int]:
    row %= constants.OBS_DIM[0] * TILE
    column %= constants.OBS_DIM[1] * TILE
    pixel = cast(tuple[int, int, int], frame[row, column])
    return (pixel[0], pixel[1], pixel[2])


def _state(num_envs: int = 1) -> EnvState:
    state = empty_state(num_envs=num_envs, device=torch.device("cpu"))
    state.player_position[:] = torch.tensor([20, 20], dtype=torch.int32)
    state.player_direction[:] = int(Action.DOWN)
    state.map[:] = int(BlockType.GRASS)
    state.light_map[:] = 1.0
    state.light_level[:] = 1.0
    state.player_health[:] = 9.0
    return state


def _center(frame: np.ndarray) -> np.ndarray:
    """Return the tile the player stands on."""
    rows, columns = constants.OBS_DIM
    row, column = (rows // 2) * TILE, (columns // 2) * TILE
    return frame[row : row + TILE, column : column + TILE]


@pytest.mark.compute_large_fixture
def test_a_frame_is_the_players_own_view(renderer: Renderer) -> None:
    # The same 9x11 window the policy reads, so a replay shows what the agent
    # knew rather than what it could not have known.
    rows, columns = constants.OBS_DIM
    frame = cast(NDArray[np.uint8], renderer.render(_state()))
    assert frame.shape == (rows * TILE, columns * TILE, 3)
    assert frame.dtype == np.uint8


def test_each_worker_draws_its_own_world(renderer: Renderer) -> None:
    state = generated_world(num_envs=3, seed=0)
    frames = [renderer.render(state, index=index) for index in range(3)]
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])


@pytest.mark.compute_large_fixture
def test_the_player_is_drawn_at_the_centre(renderer: Renderer) -> None:
    bare = _state()
    bare.player_position[:] = torch.tensor([20, 20], dtype=torch.int32)
    with_player = renderer.render(bare)
    # Grass alone is uniform across tiles; the centre must differ from a
    # neighbour precisely because the player stands there.
    neighbour = with_player[0:TILE, 0:TILE]
    assert not np.array_equal(_center(with_player), neighbour)


@pytest.mark.compute_large_fixture
def test_facing_changes_the_player_sprite(renderer: Renderer) -> None:
    left = _state()
    left.player_direction[:] = int(Action.LEFT)
    right = _state()
    right.player_direction[:] = int(Action.RIGHT)
    assert not np.array_equal(
        _center(renderer.render(left)),
        _center(renderer.render(right)),
    )


@pytest.mark.compute_large_fixture
def test_a_sleeping_player_is_drawn_asleep(renderer: Renderer) -> None:
    awake = _state()
    asleep = _state()
    asleep.is_sleeping[:] = True
    assert not np.array_equal(renderer.render(awake), renderer.render(asleep))


@pytest.mark.compute_large_fixture
def test_beyond_the_map_edge_is_flat_grey(renderer: Renderer) -> None:
    # Not black and not grass: the edge of the world has to read as an edge.
    state = _state()
    state.player_position[:] = torch.tensor([0, 0], dtype=torch.int32)
    frame = cast(NDArray[np.uint8], renderer.render(state))
    pixel = tuple(
        frame[0, 0],  # pyright: ignore[reportAny] -- numpy indexing is dtype-erased.
    )
    assert pixel == sprites.OUT_OF_BOUNDS_COLOR


@pytest.mark.compute_large_fixture
def test_an_unlit_tile_is_black(renderer: Renderer) -> None:
    # Shading is proportional to missing light, so no light at all is black.
    state = _state()
    state.light_map[:] = 0.0
    frame = cast(NDArray[np.uint8], renderer.render(state))
    maximum = int(
        _center(frame).max(),  # pyright: ignore[reportAny] -- numpy reduction is dtype-erased.
    )
    assert maximum == 0


def test_night_tints_the_surface_but_not_the_caves(renderer: Renderer) -> None:
    day = _state()
    night = _state()
    night.light_level[:] = 0.5
    pixel = (0, 0)
    day_frame = cast(NDArray[np.uint8], renderer.render(day))
    night_frame = cast(NDArray[np.uint8], renderer.render(night))
    assert _pixel(day_frame, *pixel) == _sprite_colour(
        sprites.BLOCK_SPRITES[int(BlockType.GRASS)],
    )
    assert _pixel(night_frame, *pixel) == (46, 69, 158)

    cave_day = _state()
    cave_day.player_level[:] = 1
    cave_night = _state()
    cave_night.player_level[:] = 1
    cave_night.light_level[:] = 0.5
    assert np.array_equal(renderer.render(cave_day), renderer.render(cave_night))


@pytest.mark.compute_large_fixture
def test_a_blocked_ladder_looks_different_from_an_open_one(
    renderer: Renderer,
) -> None:
    # The only cue that descending is not yet allowed. Placed beside the
    # player rather than under them: the player is drawn last, so a ladder on
    # their own tile is covered and the two frames would match either way.
    blocked = _state()
    blocked.item_map[:, 0, 20, 21] = int(ItemType.LADDER_DOWN)
    open_ladder = _state()
    open_ladder.item_map[:, 0, 20, 21] = int(ItemType.LADDER_DOWN)
    open_ladder.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    assert not np.array_equal(
        renderer.render(blocked),
        renderer.render(open_ladder),
    )


@pytest.mark.compute_large_fixture
def test_a_creature_in_view_is_drawn(renderer: Renderer) -> None:
    empty = _state()
    occupied = _state()
    occupied.melee_mobs.mask[:, 0, 0] = True
    occupied.melee_mobs.position[:, 0, 0] = torch.tensor([20, 21], dtype=torch.int32)
    assert not np.array_equal(renderer.render(empty), renderer.render(occupied))


@pytest.mark.compute_large_fixture
def test_a_creature_outside_the_view_is_not_drawn(renderer: Renderer) -> None:
    # Clamping instead of skipping would paint a distant mob on the view edge,
    # showing the agent something it cannot see.
    empty = _state()
    distant = _state()
    distant.melee_mobs.mask[:, 0, 0] = True
    distant.melee_mobs.position[:, 0, 0] = torch.tensor([40, 40], dtype=torch.int32)
    assert np.array_equal(renderer.render(empty), renderer.render(distant))


@pytest.mark.compute_large_fixture
def test_a_dead_creature_is_not_drawn(renderer: Renderer) -> None:
    empty = _state()
    ghost = _state()
    ghost.melee_mobs.position[:, 0, 0] = torch.tensor([20, 21], dtype=torch.int32)
    ghost.melee_mobs.mask[:, 0, 0] = False
    assert np.array_equal(renderer.render(empty), renderer.render(ghost))


def test_the_vulnerable_boss_looks_different(renderer: Renderer) -> None:
    frames: list[np.ndarray] = []
    for vulnerable in (False, True):
        state = _state()
        state.player_level[:] = 8
        state.map[:, 8] = int(BlockType.GRASS)
        state.map[:, 8, 20, 21] = int(BlockType.NECROMANCER)
        state.boss_progress[:] = 3 if vulnerable else 0
        state.boss_timesteps_to_spawn_this_round[:] = 0 if vulnerable else 10
        frames.append(renderer.render(state))
    assert not np.array_equal(frames[0], frames[1])


@pytest.mark.compute_large_fixture
def test_rendering_does_not_mutate_the_world(renderer: Renderer) -> None:
    state = generated_world(num_envs=1, seed=1)
    before = state.map.clone()
    renderer.render(state)
    assert torch.equal(before, state.map)


@pytest.mark.compute_large_fixture
def test_the_same_world_draws_the_same_frame(renderer: Renderer) -> None:
    state = _state()
    assert np.array_equal(renderer.render(state), renderer.render(state))


@pytest.mark.compute_large_fixture
def test_a_degenerate_tile_size_is_refused(sprite_dir: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        Renderer(block_pixels=0, asset_dir=sprite_dir)


@pytest.mark.network_github
@pytest.mark.compute_large_fixture
def test_a_real_sprite_downloads_and_loads(tmp_path: Path) -> None:
    """One genuine asset downloads and decodes as nontrivial pixel art.

    Marked ``integration`` because it reaches GitHub, and it SKIPS rather than
    fails when that fetch does not succeed. The distinction matters: this test
    asserts the asset path works, not that the network is up. Generated sprites
    above already exercise the renderer's loading, scaling, and composition;
    downloading all 143 identical-contract files added no independent signal.
    """
    try:
        path = assets.fetch(sprites.PLAYER_SPRITES[0], directory=tmp_path)
    except RuntimeError as error:  # pragma: no cover -- network-dependent
        pytest.skip(f"could not fetch the Craftax sprites: {error}")
    image = pygame.image.load(str(path))
    assert image.get_width() > 0
    assert image.get_height() > 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "3dc10f41b6d9de35fbf187198fbd12ec7063999778c69fe0610b46be7e580e42"
    )


def test_default_renderer_has_default_tile_geometry(sprite_dir: Path) -> None:
    renderer = Renderer(asset_dir=sprite_dir)
    rows, columns = constants.OBS_DIM
    assert renderer.block_pixels == 64
    assert renderer.frame_shape == (rows * 64, columns * 64)


def test_rendering_leaves_sdl_and_the_environment_untouched(
    sprite_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A viewer object must not decide which display a later window gets.

    SDL binds its video driver at the first subsystem init and keeps it for
    the process, so a renderer that initialized SDL, or exported a driver,
    would leave a later ``play`` in the same process without a window.
    Loading, scaling, compositing, and reading back surfaces need neither.
    The variable is cleared first, because the surrounding suite sets it.
    """
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    inits: list[str] = []
    monkeypatch.setattr(pygame, "init", lambda: inits.append("pygame"))
    monkeypatch.setattr(pygame.display, "init", lambda: inits.append("display"))

    Renderer(block_pixels=TILE, asset_dir=sprite_dir).render(_state())

    assert "SDL_VIDEODRIVER" not in os.environ
    assert inits == []


@pytest.mark.parametrize("view", [(5, 7), (3, 3)])
def test_the_frame_is_the_configured_view(
    sprite_dir: Path,
    view: tuple[int, int],
) -> None:
    renderer = Renderer(block_pixels=TILE, asset_dir=sprite_dir, view=view)
    frame = cast(NDArray[np.uint8], renderer.render(_state()))
    assert renderer.frame_shape == (view[0] * TILE, view[1] * TILE)
    assert frame.shape == (*renderer.frame_shape, 3)
    # The player stands at the centre of the configured view, not of 9x11.
    center = (view[0] // 2 * TILE, view[1] // 2 * TILE)
    assert _pixel(frame, *center) == _colour(
        sprites.PLAYER_SPRITES[3],
    )


@pytest.mark.parametrize("view", [(0, 3), (3, -1)])
def test_a_degenerate_view_is_refused(sprite_dir: Path, view: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=r"^view must be positive in both dimensions$"):
        Renderer(asset_dir=sprite_dir, view=view)


def test_constructing_a_renderer_creates_no_display_surface(
    sprite_dir: Path,
) -> None:
    """No surface means no window, whatever driver is bound.

    The other half of the guarantee: ``set_mode`` is what MAPS a window, and
    the renderer must never call it -- on a forwarded X connection that is
    seconds, and on an operator's desktop it is a window they did not ask
    for. ``play`` calls it deliberately.
    """
    Renderer(block_pixels=TILE, asset_dir=sprite_dir)
    assert pygame.display.get_surface() is None


def test_renderer_covers_composition_and_shading_branches(renderer: Renderer) -> None:
    state = _state()
    state.item_map[:, 0, 20, 21] = int(ItemType.LADDER_DOWN)
    state.monsters_killed[:, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    state.map[:, 0, 20, 21] = int(BlockType.DARKNESS)
    state.melee_mobs.mask[:, 0, 0] = True
    state.melee_mobs.position[:, 0, 0] = torch.tensor([20, 21], dtype=torch.int32)
    state.mob_projectiles.mask[:, 0, 0] = True
    state.mob_projectiles.position[:, 0, 0] = torch.tensor([20, 19], dtype=torch.int32)
    state.player_projectiles.mask[:, 0, 0] = True
    state.player_projectiles.position[:, 0, 0] = torch.tensor(
        [19, 20],
        dtype=torch.int32,
    )
    state.light_map[:, 0, 20, 20] = 0.5
    state.light_level[:] = 0.5
    frame = cast(NDArray[np.uint8], renderer.render(state))
    assert frame.shape == (*renderer.frame_shape, 3)
    assert frame.dtype == np.uint8

    state.player_level[:] = 1
    state.is_sleeping[:] = True
    assert renderer.render(state).shape == frame.shape


def test_renderer_rejects_an_invalid_worker_index(renderer: Renderer) -> None:
    with pytest.raises(IndexError):
        renderer.render(_state(), index=2)


def test_frame_shape_and_tile_size_are_exact(sprite_dir: Path) -> None:
    renderer = Renderer(block_pixels=3, asset_dir=sprite_dir)
    rows, columns = constants.OBS_DIM
    assert renderer.block_pixels == 3
    assert renderer.frame_shape == (rows * 3, columns * 3)


def test_terrain_uses_exact_tile_colours(renderer: Renderer) -> None:
    state = _state()
    state.map[:, 0, 20, 21] = int(BlockType.DARKNESS)
    frame = cast(NDArray[np.uint8], renderer.render(state))
    center_row, center_column = constants.OBS_DIM[0] // 2, constants.OBS_DIM[1] // 2
    grass = _pixel(frame, center_row * TILE, (center_column - 1) * TILE)
    darkness = _pixel(frame, center_row * TILE, (center_column + 1) * TILE)
    assert tuple(grass) == _sprite_colour(sprites.BLOCK_SPRITES[int(BlockType.GRASS)])
    assert tuple(darkness) == sprites.DARKNESS_COLOR


def test_out_of_bounds_block_uses_the_out_of_bounds_colour(
    renderer: Renderer,
) -> None:
    state = _state()
    state.map[0, 0, 20, 21] = int(BlockType.OUT_OF_BOUNDS)
    frame = cast(NDArray[np.uint8], renderer.render(state))
    center_row, center_column = constants.OBS_DIM[0] // 2, constants.OBS_DIM[1] // 2
    assert _pixel(frame, center_row * TILE, (center_column + 1) * TILE) == (
        128,
        128,
        128,
    )


def test_creature_visibility_and_position_are_exact(renderer: Renderer) -> None:
    state = _state()
    state.melee_mobs.mask[0, 0, 1] = True
    state.melee_mobs.position[0, 0, 1] = torch.tensor([19, 18], dtype=torch.int32)
    frame = cast(NDArray[np.uint8], renderer.render(state))
    rows, columns = constants.OBS_DIM
    row, column = rows // 2 - 1, columns // 2 - 2
    species = int(state.melee_mobs.type_id[0, 0, 1])
    name = sprites.MELEE_SPRITES[species % len(sprites.MELEE_SPRITES)]
    expected = _colour(name)
    assert _pixel(frame, row * TILE, column * TILE) == expected
    grass = _sprite_colour(sprites.BLOCK_SPRITES[int(BlockType.GRASS)])
    assert _pixel(frame, (row + 1) * TILE, column * TILE) == grass


def test_out_of_view_creature_does_not_stop_later_creatures(
    renderer: Renderer,
) -> None:
    state = _state()
    state.melee_mobs.mask[0, 0, :3] = True
    state.melee_mobs.position[0, 0, 0] = torch.tensor([15, 21], dtype=torch.int32)
    state.melee_mobs.position[0, 0, 1] = torch.tensor([20, 30], dtype=torch.int32)
    state.melee_mobs.position[0, 0, 2] = torch.tensor([20, 21], dtype=torch.int32)
    name = sprites.MELEE_SPRITES[int(state.melee_mobs.type_id[0, 0, 2])]

    frame = cast(NDArray[np.uint8], renderer.render(state))

    rows, columns = constants.OBS_DIM
    assert _pixel(frame, (rows // 2) * TILE, (columns // 2 + 1) * TILE) == (
        _colour(name)
    )


def test_mob_on_the_view_corner_is_visible(renderer: Renderer) -> None:
    state = _state()
    state.melee_mobs.mask[0, 0, 0] = True
    state.melee_mobs.position[0, 0, 0] = torch.tensor([16, 15], dtype=torch.int32)
    name = sprites.MELEE_SPRITES[int(state.melee_mobs.type_id[0, 0, 0])]
    frame = cast(NDArray[np.uint8], renderer.render(state))
    assert _pixel(frame, 0, 0) == _sprite_colour(name)


def test_player_direction_and_sleep_sprite_are_exact(
    renderer: Renderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for direction, facing in (
        (Action.LEFT, 0),
        (Action.RIGHT, 1),
        (Action.UP, 2),
        (Action.DOWN, 3),
        (99, 3),
    ):
        state = _state()
        state.player_direction[:] = int(direction)
        frame = cast(NDArray[np.uint8], renderer.render(state))
        rows, columns = constants.OBS_DIM
        center = (rows // 2 * TILE, columns // 2 * TILE)
        assert _pixel(frame, center[0], center[1]) == _colour(
            sprites.PLAYER_SPRITES[facing],
        )

    asleep = _state()
    asleep.is_sleeping[:] = True
    white = pygame.Surface((TILE, TILE))
    white.fill((255, 255, 255))
    monkeypatch.setitem(renderer._cache, sprites.PLAYER_SPRITES[-1], white)
    frame = cast(NDArray[np.uint8], renderer.render(asleep))
    rows, columns = constants.OBS_DIM
    center = (rows // 2 * TILE, columns // 2 * TILE)
    assert _pixel(frame, center[0], center[1]) == (127, 127, 127)


def test_renderer_uses_asset_directory_and_caches_sprites(
    sprite_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = assets.fetch
    requests: list[tuple[str, Path | None]] = []

    def record_fetch(name: str, *, directory: Path | None = None) -> Path:
        requests.append((name, directory))
        return fetch(name, directory=directory)

    monkeypatch.setattr(assets, "fetch", record_fetch)
    renderer = Renderer(block_pixels=TILE, asset_dir=sprite_dir)
    expected = [(name, sprite_dir) for name in sprites.every_sprite()]
    assert requests == expected
    renderer._load(sprites.PLAYER_SPRITES[0])
    assert requests == expected


def test_renderer_draws_map_edge_and_rejects_nonpositive_size(sprite_dir: Path) -> None:
    renderer = Renderer(block_pixels=TILE, asset_dir=sprite_dir)
    state = _state()
    state.player_position[:] = torch.tensor([3, 4], dtype=torch.int32)
    state.map[0, 0, 0, 0] = int(BlockType.DARKNESS)
    frame = cast(NDArray[np.uint8], renderer.render(state))
    assert _pixel(frame, 0, 0) == sprites.OUT_OF_BOUNDS_COLOR
    assert _pixel(frame, TILE, TILE) == sprites.DARKNESS_COLOR
    Renderer(block_pixels=1, asset_dir=sprite_dir)
    with pytest.raises(ValueError, match=r"^block_pixels must be positive$"):
        Renderer(block_pixels=0, asset_dir=sprite_dir)


def test_lower_and_upper_map_edges_are_drawn(renderer: Renderer) -> None:
    state = _state()
    state.player_position[:] = torch.tensor([0, 0], dtype=torch.int32)
    state.map[0, 0, 0, 0] = int(BlockType.DARKNESS)
    state.light_map[0, 0, 0, 0] = 0.0
    frame = cast(NDArray[np.uint8], renderer.render(state))
    rows, columns = constants.OBS_DIM
    assert _pixel(frame, (rows // 2) * TILE, (columns // 2) * TILE) == (0, 0, 0)
    assert _pixel(frame, 0, 0) == sprites.OUT_OF_BOUNDS_COLOR

    state = _state()
    state.player_position[:] = torch.tensor(
        [state.map.shape[-2] - 1, state.map.shape[-1] - 1],
        dtype=torch.int32,
    )
    state.map[0, 0, -1, -1] = int(BlockType.DARKNESS)
    state.light_map[0, 0, -1, -1] = 0.0
    frame = cast(NDArray[np.uint8], renderer.render(state))
    assert _pixel(frame, (rows // 2) * TILE, (columns // 2) * TILE) == (0, 0, 0)
    assert _pixel(frame, -1, -1) == sprites.OUT_OF_BOUNDS_COLOR


def test_ladder_and_shading_outputs_are_exact(renderer: Renderer) -> None:
    blocked = _state()
    blocked.item_map[0, 0, 20, 21] = int(ItemType.LADDER_DOWN)
    frame_blocked = cast(NDArray[np.uint8], renderer.render(blocked))
    rows, columns = constants.OBS_DIM
    row, column = rows // 2, columns // 2 + 1
    blocked_name = sprites.ITEM_SPRITES[int(ItemType.LADDER_DOWN_BLOCKED)]
    assert _pixel(frame_blocked, row * TILE, column * TILE) == _colour(blocked_name)

    open_ladder = _state()
    open_ladder.item_map[0, 0, 20, 21] = int(ItemType.LADDER_DOWN)
    open_ladder.monsters_killed[0, 0] = constants.MONSTERS_KILLED_TO_CLEAR_LEVEL
    frame_open = cast(NDArray[np.uint8], renderer.render(open_ladder))
    open_name = sprites.ITEM_SPRITES[int(ItemType.LADDER_DOWN)]
    assert _pixel(frame_open, row * TILE, column * TILE) == _colour(open_name)

    dim = _state()
    dim.light_map[0, 0, 20, 21] = 0.5
    dimmed = cast(NDArray[np.uint8], renderer.render(dim))
    grass = _sprite_colour(sprites.BLOCK_SPRITES[int(BlockType.GRASS)])
    assert _pixel(dimmed, row * TILE, column * TILE) == (46, 61, 126)
    assert grass == (92, 123, 251)

    dark = _state()
    dark.light_map[0, 0, 20, 21] = 0.0
    dark_frame = cast(NDArray[np.uint8], renderer.render(dark))
    assert _pixel(dark_frame, row * TILE, column * TILE) == (0, 0, 0)


def test_one_alpha_level_of_shading_changes_white_by_one(
    renderer: Renderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    white = pygame.Surface((TILE, TILE))
    white.fill((255, 255, 255))
    monkeypatch.setitem(
        renderer._cache,
        sprites.BLOCK_SPRITES[int(BlockType.GRASS)],
        white,
    )
    rows, columns = constants.OBS_DIM
    tile_pixel = ((rows // 2) * TILE, (columns // 2 + 1) * TILE)

    fully_lit = _state()
    frame = cast(NDArray[np.uint8], renderer.render(fully_lit))
    assert _pixel(frame, *tile_pixel) == (255, 255, 255)

    state = _state()
    state.light_map[0, 0, 20, 21] = 0.996
    frame = cast(NDArray[np.uint8], renderer.render(state))
    assert _pixel(frame, *tile_pixel) == (254, 254, 254)

    night = _state()
    night.light_level[:] = 0.996
    frame = cast(NDArray[np.uint8], renderer.render(night))
    assert _pixel(frame, *tile_pixel) == (254, 254, 254)

    cave = _state()
    cave.player_level[:] = 1
    cave.light_level[:] = 0.5
    frame = cast(NDArray[np.uint8], renderer.render(cave))
    assert _pixel(frame, *tile_pixel) == (255, 255, 255)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
