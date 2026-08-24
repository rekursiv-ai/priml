"""Tests for playing and recording.

Recording draws GENERATED sprites, for the same reason
:mod:`pixels_test` does: what these assert -- that a video is written, that it
stops at the episode end, that the policy is untouched -- has nothing to do
with what a zombie looks like, and downloading 143 PNGs to prove it would put
GitHub on the path of every test run.
"""

from __future__ import annotations

from pathlib import Path

import hashlib

import imageio_ffmpeg
import pygame
import pytest
import torch

from priml.baselines.craftax.game import constants
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.render import play, sprites
from priml.baselines.craftax.game.render.pixels import Renderer
from priml.baselines.craftax.model import ActorCritic


@pytest.fixture(scope="module")
def sprite_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write one flat-coloured PNG per sprite, so recording stays offline."""
    directory = tmp_path_factory.mktemp("sprites")
    pygame.init()
    for name in sprites.every_sprite():
        digest = hashlib.sha256(name.encode()).digest()
        surface = pygame.Surface((8, 8), flags=pygame.SRCALPHA)
        surface.fill((digest[0] | 0x40, digest[1] | 0x40, digest[2] | 0x41, 255))
        pygame.image.save(surface, str(directory / name))
    return directory


def _read_video(path: Path) -> tuple[int, tuple[int, int]]:
    """Return the frame count and (width, height) ffmpeg reports for ``path``.

    Reading back through the decoder rather than an array API keeps the test
    honest about geometry: a writer that silently rescaled would still hand a
    plausible array to ``imread``.
    """
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    # Metadata is yielded first and frame bytes after; narrowing is what
    # distinguishes the two.
    assert isinstance(meta, dict)
    size = meta["size"]
    return sum(1 for _ in reader), (int(size[0]), int(size[1]))


def _policy() -> ActorCritic:
    config = ActorCritic.Config()
    config.channels_in = 8
    config.num_layers = 1
    return config.make()


def test_every_bound_key_names_a_real_action() -> None:
    # A typo here would silently bind a key to nothing, and the game would
    # look broken rather than the binding.
    for action in play.KEYS.values():
        assert action in set(Action)


def test_movement_is_on_the_usual_keys() -> None:
    assert play.KEYS[pygame.K_w] is Action.UP
    assert play.KEYS[pygame.K_a] is Action.LEFT
    assert play.KEYS[pygame.K_s] is Action.DOWN
    assert play.KEYS[pygame.K_d] is Action.RIGHT
    assert play.KEYS[pygame.K_SPACE] is Action.DO


def test_no_action_is_bound_to_two_keys() -> None:
    # ``KEYS`` is a dict, so its KEYS cannot repeat -- comparing them to their
    # own set proved nothing. Duplication is only possible on the value side,
    # where two keys silently doing the same thing is the real mistake.
    actions = list(play.KEYS.values())
    assert len(actions) == len(set(actions))


@pytest.mark.compute_large_fixture
def test_recording_writes_a_playable_video(
    tmp_path: Path,
    sprite_dir: Path,
) -> None:
    path = tmp_path / "replay.mp4"
    steps = play.record(
        _policy(),
        path,
        seed=1,
        max_steps=6,
        block_pixels=8,
        asset_dir=sprite_dir,
    )
    assert steps > 0
    count, size = _read_video(path)
    # Not an exact count: h264 pads to a macro-block-aligned frame count, so
    # the file may hold slightly more frames than steps were played.
    assert count >= 1
    # Geometry is macro-block-aligned, not exact: the encoder rounds each axis
    # up to a multiple of 16. Asserting the rounded size still pins the aspect
    # and catches a writer that rescaled to something unrelated.
    width, height = size
    assert width >= 8 * constants.OBS_DIM[1]
    assert height >= 8 * constants.OBS_DIM[0]
    assert width % 16 == 0
    assert height % 16 == 0


@pytest.mark.compute_large_fixture
def test_recording_stops_when_the_episode_ends(
    tmp_path: Path,
    sprite_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Otherwise the video would keep rolling into a freshly reset world, and
    # the replay would show two episodes as one. The step limit is shortened
    # so a real episode actually ends inside a test.
    monkeypatch.setattr(constants, "MAX_TIMESTEPS", 3)
    steps = play.record(
        _policy(),
        tmp_path / "short.mp4",
        seed=2,
        max_steps=50,
        block_pixels=8,
        asset_dir=sprite_dir,
    )
    assert steps == 3


@pytest.mark.compute_large_fixture
def test_the_same_seed_records_the_same_episode(
    tmp_path: Path,
    sprite_dir: Path,
) -> None:
    # Frame count and geometry alone would pass for two entirely different
    # episodes of equal length, so the bytes are what get compared.
    policy = _policy()
    recorded: list[bytes] = []
    for name in ("a.mp4", "b.mp4"):
        path = tmp_path / name
        play.record(
            policy,
            path,
            seed=5,
            max_steps=8,
            block_pixels=8,
            asset_dir=sprite_dir,
        )
        recorded.append(path.read_bytes())
    assert recorded[0] == recorded[1]


def test_a_degenerate_recording_is_refused(tmp_path: Path) -> None:
    # Checked before any sprite is loaded, so this needs no assets at all.
    with pytest.raises(ValueError, match="positive"):
        play.record(_policy(), tmp_path / "x.mp4", max_steps=0)
    # The other half of the same guard; only max_steps was covered.
    with pytest.raises(ValueError, match="positive"):
        play.record(_policy(), tmp_path / "x.mp4", fps=0)
    # block_pixels is validated too, by the renderer it is handed to.
    with pytest.raises(ValueError, match="positive"):
        play.record(_policy(), tmp_path / "x.mp4", block_pixels=0)


@pytest.mark.compute_large_fixture
def test_the_writer_is_sized_by_the_renderer_that_fills_it(
    tmp_path: Path,
    sprite_dir: Path,
) -> None:
    """Written geometry must come from the renderer, not a parallel formula.

    ``record`` recomputed ``OBS_DIM * block_pixels`` itself, and nothing tied
    that to what ``render`` produces. A mismatch is silent: ffmpeg writes an
    unreadable file -- measured at 1667 bytes for a one-tile drift -- and
    ``writer.close()`` returns without raising.
    """
    path = tmp_path / "sized.mp4"
    _ = play.record(
        _policy(),
        path,
        seed=7,
        max_steps=3,
        block_pixels=8,
        asset_dir=sprite_dir,
    )
    expected_height, expected_width = Renderer(
        block_pixels=8, asset_dir=sprite_dir
    ).frame_shape

    _, size = _read_video(path)
    # macro_block_size=16 rounds the written frame up, so the match is to
    # within one macroblock rather than exact.
    assert 0 <= size[0] - expected_width < 16
    assert 0 <= size[1] - expected_height < 16


@pytest.mark.compute_large_fixture
def test_recording_leaves_the_policy_untouched(
    tmp_path: Path,
    sprite_dir: Path,
) -> None:
    # Watching must never train: a replay is an observation of the policy, so
    # it may not move a weight.
    policy = _policy()
    before = policy.policy[0].weight.detach().clone()
    play.record(
        policy,
        tmp_path / "x.mp4",
        seed=3,
        max_steps=4,
        block_pixels=8,
        asset_dir=sprite_dir,
    )
    assert torch.equal(before, policy.policy[0].weight.detach())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
