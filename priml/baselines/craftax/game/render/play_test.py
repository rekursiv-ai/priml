"""Tests for playing and recording.

Recording draws GENERATED sprites, for the same reason
:mod:`pixels_test` does: what these assert -- that a video is written, that it
stops at the episode end, that the policy is untouched -- has nothing to do
with what a zombie looks like, and downloading 143 PNGs to prove it would put
GitHub on the path of every test run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import hashlib

import imageio_ffmpeg
import numpy as np
import pygame
import pytest
import torch

from priml.baselines.craftax.game import constants, step, world_gen
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.render import play, sprites
from priml.baselines.craftax.game.render.pixels import Renderer
from priml.baselines.craftax.model import ActorCritic


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


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


# Reading back through the decoder rather than an array API keeps the test honest about
# geometry: a writer that silently rescaled would still hand a plausible array to
# ``imread``.
def _read_video(path: Path) -> tuple[int, tuple[int, int]]:
    """Return the frame count and (width, height) ffmpeg reports for ``path``."""
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
    # One frame per step. This asserted only ``count >= 1``, which a writer
    # that dropped every frame but the first would satisfy -- and blamed
    # macro-block padding, which pads the frame GEOMETRY (see the size checks
    # below), not the frame COUNT. Measured at 3, 6, and 12 steps: the file
    # holds exactly as many frames as steps were played.
    assert count == steps
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
    with pytest.raises(ValueError, match=r"^Replay geometry must be positive$"):
        play.record(_policy(), tmp_path / "x.mp4", max_steps=0)
    # The other half of the same guard; only max_steps was covered.
    with pytest.raises(ValueError, match=r"^Replay geometry must be positive$"):
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
        block_pixels=8,
        asset_dir=sprite_dir,
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
    layer = policy.policy[0]
    assert isinstance(layer, torch.nn.Linear)
    before = layer.weight.detach().clone()
    play.record(
        policy,
        tmp_path / "x.mp4",
        seed=3,
        max_steps=4,
        block_pixels=8,
        asset_dir=sprite_dir,
    )
    assert torch.equal(before, layer.weight.detach())


def test_play_closes_cleanly_on_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object()

    class FakeRenderer:
        frame_shape = (2, 4)

        def __init__(self, *, block_pixels: int, asset_dir: Path | None) -> None:
            assert (block_pixels, asset_dir) == (64, None)

        def render(self, state: object) -> np.ndarray:
            del state
            return np.zeros((2, 4, 3), dtype=np.uint8)

    def generate_world(
        *,
        num_envs: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> object:
        assert (num_envs, device) == (1, torch.device("cpu"))
        assert generator.initial_seed() == 0
        return state

    def show(screen: pygame.Surface, frame: np.ndarray) -> None:
        del screen, frame

    def set_caption(caption: str) -> None:
        del caption

    monkeypatch.setattr(world_gen, "generate_world", generate_world)
    monkeypatch.setattr(play, "Renderer", FakeRenderer)
    monkeypatch.setattr(play, "_show", show)
    monkeypatch.setattr(pygame.event, "wait", lambda: pygame.event.Event(pygame.QUIT))
    monkeypatch.setattr(pygame.display, "set_mode", pygame.Surface)
    monkeypatch.setattr(pygame.display, "set_caption", set_caption)
    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame, "quit", lambda: None)

    assert play.play() is state


def test_show_blits_the_rgb_frame_at_the_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[1, 2] = (17, 83, 201)
    surface = pygame.Surface((4, 2))
    destinations: list[tuple[int, int]] = []

    class ScreenSpy:
        def blit(
            self,
            source: pygame.Surface,
            dest: tuple[int, int],
        ) -> None:
            destinations.append(dest)
            surface.blit(source, dest)

    monkeypatch.setattr(pygame.display, "flip", lambda: None)

    play._show(cast(pygame.Surface, ScreenSpy()), frame)

    assert destinations == [(0, 0)]
    assert surface.get_at((2, 1))[:3] == (17, 83, 201)
    assert surface.get_at((0, 0))[:3] == (0, 0, 0)


def test_play_routes_input_and_resets_finished_worlds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial, moved, reset, moved_again = object(), object(), object(), object()
    events = iter(
        (
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
        ),
    )
    world_calls: list[dict[str, object]] = []
    states = iter((initial, reset))

    def generate_world(**kwargs: object) -> object:
        world_calls.append(kwargs)
        return next(states)

    class FakeRenderer:
        frame_shape = (2, 4)

        def __init__(self, *, block_pixels: int, asset_dir: Path | None) -> None:
            assert (block_pixels, asset_dir) == (8, tmp_path)

        def render(self, state: object) -> np.ndarray:
            return np.full((2, 4, 3), states_seen.index(state), dtype=np.uint8)

    screen = pygame.Surface((4, 2))
    screen_sizes: list[tuple[int, int]] = []
    captions: list[str] = []
    shown: list[tuple[pygame.Surface, np.ndarray]] = []
    states_seen = [initial, reset, moved_again]
    actions: list[tuple[object, torch.Tensor, torch.Generator]] = []
    monkeypatch.setattr(world_gen, "generate_world", generate_world)
    monkeypatch.setattr(pygame.event, "wait", lambda: next(events))

    def set_mode(size: tuple[int, int]) -> pygame.Surface:
        screen_sizes.append(size)
        return screen

    monkeypatch.setattr(pygame.display, "set_mode", set_mode)
    monkeypatch.setattr(pygame.display, "set_caption", captions.append)
    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame, "quit", lambda: None)
    monkeypatch.setattr(play, "Renderer", FakeRenderer)

    def show(target: pygame.Surface, frame: np.ndarray) -> None:
        shown.append((target, frame))

    monkeypatch.setattr(play, "_show", show)

    def take_step(
        state: object,
        action: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[object, None]:
        actions.append((state, action, generator))
        return (moved if len(actions) == 1 else moved_again), None

    def is_done(state: object) -> torch.Tensor:
        return torch.tensor([state is moved])

    monkeypatch.setattr(step, "step", take_step)
    monkeypatch.setattr(step, "is_done", is_done)

    result = play.play(seed=37, block_pixels=8, asset_dir=tmp_path)

    assert result is moved_again
    assert screen_sizes == [(4, 2)]
    assert captions == ["Craftax (seed 37)"]
    assert [state for state, _ in shown] == [screen, screen, screen]
    assert [frame[0, 0, 0] for _, frame in shown] == [0, 1, 2]
    assert len(world_calls) == 2
    assert all(call["num_envs"] == 1 for call in world_calls)
    assert all(call["device"] == torch.device("cpu") for call in world_calls)
    generator = world_calls[0]["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.initial_seed() == 37
    assert world_calls[1]["generator"] is generator
    assert [state for state, _, _ in actions] == [initial, reset]
    assert [action for _, action, _ in actions] == [
        torch.tensor([int(Action.UP)]),
        torch.tensor([int(Action.RIGHT)]),
    ]
    assert all(generator is world_calls[0]["generator"] for _, _, generator in actions)


def test_record_uses_defaults_and_closes_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        state = object()

        def reset(self) -> torch.Tensor:
            return torch.tensor([[1, 2]])

        def step(self, action: torch.Tensor) -> object:
            del action
            return type(
                "Transition",
                (),
                {"observation": torch.tensor([[2, 3]]), "done": torch.tensor([False])},
            )()

    environment = FakeEnv()

    class FakeConfig:
        num_envs = 1
        device = "cpu"
        seed = 0

        def make(self) -> FakeEnv:
            assert (self.num_envs, self.device, self.seed) == (1, "cpu", 0)
            return environment

    class FakeRenderer:
        frame_shape = (2, 3)

        def __init__(self, *, block_pixels: int, asset_dir: Path | None) -> None:
            assert (block_pixels, asset_dir) == (64, tmp_path)

        def render(self, state: object) -> np.ndarray:
            del state
            return np.zeros((2, 4, 3), dtype=np.uint8)

    frames: list[bytes] = []
    closed: list[bool] = []
    observations: list[torch.Tensor] = []
    writer_args: list[tuple[tuple[object, ...], dict[str, object]]] = []
    seeds: set[int] = set()

    def make_writer(*args: object, **kwargs: object) -> Generator[None, bytes, None]:
        writer_args.append((args, kwargs))
        try:
            while True:
                frames.append((yield None))
        finally:
            closed.append(True)

    def sample(
        probabilities: torch.Tensor,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del probabilities, num_samples
        assert generator is not None
        seeds.add(generator.initial_seed())
        return torch.tensor([[0]])

    def policy(observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observations.append(observation.clone())
        # FakeConfig.num_envs is 1; record reads only the logits.
        return torch.zeros((1, 2)), torch.zeros((1, 1))

    class FakeEnvironmentType:
        Config = FakeConfig

    monkeypatch.setattr(play, "CraftaxEnv", FakeEnvironmentType)
    monkeypatch.setattr(play, "Renderer", FakeRenderer)
    monkeypatch.setattr(imageio_ffmpeg, "write_frames", make_writer)
    monkeypatch.setattr(torch, "multinomial", sample)

    path = tmp_path / "default.mp4"
    steps = play.record(policy, path, asset_dir=tmp_path)

    assert steps == 1_000
    assert len(frames) == 1_000
    assert writer_args == [
        ((str(path), (3, 2)), {"fps": 10, "macro_block_size": 16}),
    ]
    assert seeds == {0}
    assert torch.equal(observations[0], torch.tensor([[1, 2]]))
    assert torch.equal(observations[1], torch.tensor([[2, 3]]))
    assert closed == [True]


def test_record_passes_exact_geometry_rng_and_terminal_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        state = "live"

        def reset(self) -> torch.Tensor:
            return torch.tensor([[1, 2, 3], [4, 5, 6]])

        def step(self, action: torch.Tensor) -> object:
            assert torch.equal(action, torch.tensor([2, 0]))
            self.state = "reset"
            return type(
                "Transition",
                (),
                {
                    "observation": torch.zeros((2, 3)),
                    "done": torch.tensor([True]),
                    "terminal_state": "terminal",
                },
            )()

    environment = FakeEnv()

    class FakeConfig:
        def __init__(self) -> None:
            self.num_envs = 256
            self.device = "auto"
            self.seed = 0

        def make(self) -> FakeEnv:
            assert (self.num_envs, self.device, self.seed) == (1, "cpu", 23)
            return environment

    class FakeEnvironmentType:
        Config = FakeConfig

    rendered: list[object] = []

    class FakeRenderer:
        frame_shape = (2, 4)

        def __init__(self, *, block_pixels: int, asset_dir: Path | None) -> None:
            assert (block_pixels, asset_dir) == (4, None)

        def render(self, state: object) -> np.ndarray:
            rendered.append(state)
            return np.full((2, 4, 3), len(rendered), dtype=np.uint8)

    writer_args: list[object] = []
    frames: list[bytes] = []
    closed: list[bool] = []

    def make_writer(*args: object, **kwargs: object) -> Generator[None, bytes, None]:
        writer_args.extend((args, kwargs))
        try:
            while True:
                frames.append((yield None))
        finally:
            closed.append(True)

    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    policy_observations: list[torch.Tensor] = []

    def policy(observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_observations.append(observation.clone())
        # The policy contract has one scalar value per environment.
        return logits, torch.zeros((2, 1))

    multinomial = torch.multinomial
    sampling: list[tuple[torch.Tensor, torch.Generator | None]] = []

    def sample(
        probabilities: torch.Tensor,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        sampling.append((probabilities.clone(), generator))
        return multinomial(probabilities, num_samples, generator=generator)

    monkeypatch.setattr(play, "CraftaxEnv", FakeEnvironmentType)
    monkeypatch.setattr(play, "Renderer", FakeRenderer)
    monkeypatch.setattr(imageio_ffmpeg, "write_frames", make_writer)
    monkeypatch.setattr(torch, "multinomial", sample)

    result = play.record(
        policy,
        tmp_path / "episode.mp4",
        seed=23,
        max_steps=1,
        fps=1,
        block_pixels=4,
    )

    assert result == 1
    assert writer_args == [
        (str(tmp_path / "episode.mp4"), (4, 2)),
        {"fps": 1, "macro_block_size": 16},
    ]
    assert rendered == ["live", "terminal"]
    assert frames == [
        np.full((2, 4, 3), 1, dtype=np.uint8).tobytes(),
        np.full((2, 4, 3), 2, dtype=np.uint8).tobytes(),
    ]
    assert closed == [True]
    assert len(policy_observations) == 1
    assert torch.equal(policy_observations[0], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert len(sampling) == 1
    probabilities, generator = sampling[0]
    assert torch.equal(probabilities, logits.softmax(-1))
    assert generator is not None
    assert generator.initial_seed() == 23


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
