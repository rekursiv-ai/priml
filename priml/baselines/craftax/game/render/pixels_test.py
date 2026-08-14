"""Tests for the pixel viewer.

These draw GENERATED sprites, not the real ones: a flat, distinctly-coloured
square per file name. Every property asserted here -- which tile a thing lands
on, what is hidden, what the shading does -- is about placement and masking,
and a solid colour tests those more sharply than art does, because any leak
shows up as an exact colour that should not be there.

It also keeps the unit tier offline. Downloading 143 PNGs to assert that a
distant mob is not drawn would make every run depend on GitHub being up.
``test_the_real_sprites_draw`` covers the genuine assets and is marked
``integration`` for that reason.
"""

from __future__ import annotations

from pathlib import Path

import hashlib
import os
import subprocess
import sys

import numpy as np
import pygame
import pytest
import torch

from priml.baselines.craftax.game import constants, world_gen
from priml.baselines.craftax.game.constants import Action, BlockType, ItemType
from priml.baselines.craftax.game.render import sprites
from priml.baselines.craftax.game.render.pixels import Renderer
from priml.baselines.craftax.game.state import EnvState, empty_state


TILE = 8
"""Small tiles: these assert on colour and difference, never on detail."""


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


def test_a_frame_is_the_players_own_view(renderer: Renderer) -> None:
    # The same 9x11 window the policy reads, so a replay shows what the agent
    # knew rather than what it could not have known.
    rows, columns = constants.OBS_DIM
    frame = renderer.render(_state())
    assert frame.shape == (rows * TILE, columns * TILE, 3)
    assert frame.dtype == np.uint8


def test_each_worker_draws_its_own_world(renderer: Renderer) -> None:
    state = world_gen.generate_world(
        num_envs=3,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    frames = [renderer.render(state, index=index) for index in range(3)]
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])


def test_the_player_is_drawn_at_the_centre(renderer: Renderer) -> None:
    bare = _state()
    bare.player_position[:] = torch.tensor([20, 20], dtype=torch.int32)
    with_player = renderer.render(bare)
    # Grass alone is uniform across tiles; the centre must differ from a
    # neighbour precisely because the player stands there.
    neighbour = with_player[0:TILE, 0:TILE]
    assert not np.array_equal(_center(with_player), neighbour)


def test_facing_changes_the_player_sprite(renderer: Renderer) -> None:
    left = _state()
    left.player_direction[:] = int(Action.LEFT)
    right = _state()
    right.player_direction[:] = int(Action.RIGHT)
    assert not np.array_equal(
        _center(renderer.render(left)),
        _center(renderer.render(right)),
    )


def test_a_sleeping_player_is_drawn_asleep(renderer: Renderer) -> None:
    awake = _state()
    asleep = _state()
    asleep.is_sleeping[:] = True
    assert not np.array_equal(renderer.render(awake), renderer.render(asleep))


def test_beyond_the_map_edge_is_flat_grey(renderer: Renderer) -> None:
    # Not black and not grass: the edge of the world has to read as an edge.
    state = _state()
    state.player_position[:] = torch.tensor([0, 0], dtype=torch.int32)
    frame = renderer.render(state)
    assert tuple(frame[0, 0]) == sprites.OUT_OF_BOUNDS_COLOR


def test_an_unlit_tile_is_black(renderer: Renderer) -> None:
    # Darkness genuinely hides the world here, exactly as it does in the
    # observation the agent reads.
    state = _state()
    state.light_map[:] = 0.0
    frame = renderer.render(state)
    assert int(_center(frame).max()) == 0


def test_night_tints_the_surface_but_not_the_caves(renderer: Renderer) -> None:
    day = _state()
    night = _state()
    night.light_level[:] = 0.0
    assert not np.array_equal(renderer.render(day), renderer.render(night))

    # Underground has its own light; dawn does not reach it.
    cave_day = _state()
    cave_day.player_level[:] = 1
    cave_night = _state()
    cave_night.player_level[:] = 1
    cave_night.light_level[:] = 0.0
    assert np.array_equal(renderer.render(cave_day), renderer.render(cave_night))


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


def test_a_creature_in_view_is_drawn(renderer: Renderer) -> None:
    empty = _state()
    occupied = _state()
    occupied.melee_mobs.mask[:, 0, 0] = True
    occupied.melee_mobs.position[:, 0, 0] = torch.tensor([20, 21], dtype=torch.int32)
    assert not np.array_equal(renderer.render(empty), renderer.render(occupied))


def test_a_creature_outside_the_view_is_not_drawn(renderer: Renderer) -> None:
    # Clamping instead of skipping would paint a distant mob on the view edge,
    # showing the agent something it cannot see.
    empty = _state()
    distant = _state()
    distant.melee_mobs.mask[:, 0, 0] = True
    distant.melee_mobs.position[:, 0, 0] = torch.tensor([40, 40], dtype=torch.int32)
    assert np.array_equal(renderer.render(empty), renderer.render(distant))


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


def test_rendering_does_not_mutate_the_world(renderer: Renderer) -> None:
    state = world_gen.generate_world(
        num_envs=1,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    before = state.map.clone()
    renderer.render(state)
    assert torch.equal(before, state.map)


def test_the_same_world_draws_the_same_frame(renderer: Renderer) -> None:
    state = _state()
    assert np.array_equal(renderer.render(state), renderer.render(state))


def test_a_degenerate_tile_size_is_refused(sprite_dir: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        Renderer(block_pixels=0, asset_dir=sprite_dir)


@pytest.mark.integration
def test_the_real_sprites_draw() -> None:
    """The genuine assets download, load, and produce a plausible frame.

    Marked ``integration`` because it reaches GitHub. Everything above runs
    offline against generated sprites; this is the one test that proves the
    real ones exist at the pinned revision and decode.
    """
    rows, columns = constants.OBS_DIM
    frame = Renderer(block_pixels=16).render(
        world_gen.generate_world(
            num_envs=1,
            generator=torch.Generator().manual_seed(0),
            device=torch.device("cpu"),
        ),
    )
    assert frame.shape == (rows * 16, columns * 16, 3)
    # Art, not a flat fill: a frame of one colour would mean every sprite
    # failed to load and the viewer silently drew background.
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 16


def _has_windowing() -> bool:
    """Whether a real windowing system is reachable from this process.

    Probed by asking SDL to open one in a throwaway interpreter, not by
    reading ``DISPLAY``: a headless GPU worker often HAS the variable set with
    nothing listening on it, and SDL only discovers that when it tries to
    connect (``xcb_connection_has_error() returned true``).
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pygame; pygame.display.init(); "
                "pygame.display.set_mode((1, 1)); "
                "print(pygame.display.get_driver())"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "SDL_VIDEODRIVER"},
    )
    if probe.returncode != 0:
        return False
    # A zero exit is not enough. With nothing reachable, SDL quietly settles
    # on "dummy" or "offscreen" -- which draw nowhere, and are the very
    # outcomes this test asserts. Treating those as a windowing system would
    # let the test pass while proving nothing.
    return probe.stdout.strip().splitlines()[-1] not in {"dummy", "offscreen"}


@pytest.mark.skipif(
    not _has_windowing(),
    reason="no reachable windowing system: nothing to open an unwanted window ON",
)
def test_constructing_a_renderer_opens_no_window(sprite_dir: Path) -> None:
    """A viewer object must not put a window on an operator's screen.

    SDL picks its video driver at the FIRST init and caches the choice, so this
    has to run in a fresh interpreter -- in-process, some earlier test has
    already bound the driver and the check is vacuous.

    The probe inherits the caller's environment rather than forcing
    ``DISPLAY=:0``. A headless host has no X server there, so forcing it makes
    SDL abort and reports a broken machine as a broken viewer. Paired with the
    capability skip above, this test now asks the only question it can answer:
    given that a window COULD be opened, does the viewer decline to open one.

    The probe is pointed at the generated sprites: it is asserting which
    driver SDL bound, and downloading the real art to find that out would put
    the network on the path of a windowing test.
    """
    source = (
        (
            "import os; os.environ.pop('SDL_VIDEODRIVER', None); "
            "from pathlib import Path; "
            "from MODULE import Renderer; "
            "Renderer(block_pixels=8, asset_dir=Path(SPRITES)); "
            "import pygame; "
            "assert pygame.display.get_driver() == 'dummy', pygame.display.get_driver()"
        )
        .replace("MODULE", Renderer.__module__)
        .replace("SPRITES", repr(str(sprite_dir)))
    )
    package = __import__(Renderer.__module__.split(".", 1)[0])
    assert package.__file__ is not None
    probe = subprocess.run(  # noqa: S603 -- argv is this module's own import path.
        [sys.executable, "-c", source],
        cwd=Path(package.__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        # Only the driver request is dropped; the rest of the environment is
        # the caller's, which the capability skip proved can open a window.
        env={k: v for k, v in os.environ.items() if k != "SDL_VIDEODRIVER"},
    )
    assert probe.returncode == 0, probe.stderr


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
