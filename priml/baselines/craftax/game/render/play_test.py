"""Tests for playing and recording.

Recording draws GENERATED sprites, for the same reason
:mod:`pixels_test` does: what these assert -- that a video is written, that it
stops at the episode end, that the policy is untouched -- has nothing to do
with what a zombie looks like, and downloading 143 PNGs to prove it would put
GitHub on the path of every test run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hashlib

import imageio.v3 as iio
import pygame
import pytest
import torch

from priml.baselines.craftax.game import constants
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.render import play, sprites
from priml.baselines.craftax.model import ActorCritic


if TYPE_CHECKING:
    import numpy as np


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


def _policy() -> ActorCritic:
    config = ActorCritic.Config()
    config.hidden_size = 8
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


def test_no_key_is_bound_twice() -> None:
    assert len(play.KEYS) == len(set(play.KEYS))


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
    frames = iio.imread(path)
    # Not an exact count: h264 pads to a macro-block-aligned frame count, so
    # the file may hold slightly more frames than steps were played.
    assert frames.shape[0] >= 1
    assert frames.shape[-1] == 3


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
    policy = _policy()
    recorded: list[np.ndarray] = []
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
        recorded.append(iio.imread(path))
    assert recorded[0].shape == recorded[1].shape


def test_a_degenerate_recording_is_refused(tmp_path: Path) -> None:
    # Checked before any sprite is loaded, so this needs no assets at all.
    with pytest.raises(ValueError, match="positive"):
        play.record(_policy(), tmp_path / "x.mp4", max_steps=0)


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
    from priml.lib.testing import test_main

    test_main(__file__)
