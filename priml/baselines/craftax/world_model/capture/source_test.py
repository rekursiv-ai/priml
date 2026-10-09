"""Tests for the episode sources: a trained policy's checkpoint, and random play.

Every source plays the game's kernels as Python (``eager``) on tiny worlds of
grass whose clock runs out a few decisions after the reset.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import hashlib

import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game.state import DEFAULT_MAX_TIMESTEPS
from priml.baselines.craftax.rollout import TorchPhiloxSampler
from priml.baselines.craftax.testing import (
    tiny_board_policy,
    tiny_policy,
)
from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.capture.env import Schedule
from priml.baselines.craftax.world_model.capture.source import (
    CaptureError,
    PolicySource,
    RandomSource,
)


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from priml.baselines.craftax.world_model.capture.env import Captured


_DECISIONS: Final = 3
"""Decisions an episode plays before its clock runs out."""


@pytest.fixture(autouse=True)
def cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture as a host without CUDA does: unpinned buffers, no CUDA context to start.

    On a GPU host the first pinned buffer of a process starts the CUDA context,
    120 ms on the x86 host, which these tests of play and recording never use.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def short() -> Generator[None]:
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - _DECISIONS)):
        yield


def test_a_policy_source_plays_its_checkpoints_policy(
    tmp_path: Path,
    short: None,
) -> None:
    del short
    config = _policy_source(tmp_path, board=False)
    source = config.make()
    try:
        source.start(
            schedule=Schedule(arm=2, worker=1, budget=_DECISIONS),
            run_dir=tmp_path,
        )
        episodes = _poll(source)
    finally:
        source.close()
    assert source.provenance()["rollout_seed"] == str(73 + 1_000_000 * 9)
    assert source.provenance()["checkpoint_sha256"] == config.checkpoint_sha256
    assert sum(len(e.actions) for e in episodes) >= _DECISIONS
    for episode in episodes:
        assert episode.receipt.arm == 2
        assert replay.verify(episode) == replay.MATCHED


def test_a_policy_source_writes_its_policys_observation_layout(
    tmp_path: Path,
    short: None,
) -> None:
    del short
    source = _policy_source(tmp_path, board=True).make()
    try:
        source.start(schedule=Schedule(budget=1), run_dir=tmp_path)
        assert source.env is not None
        assert source.env.observations.shape[-1] == 844
        episodes = _poll(source)
    finally:
        source.close()
    assert len(episodes) == source.env.num_envs


def test_a_checkpoint_of_another_sha256_is_refused(tmp_path: Path) -> None:
    config = _policy_source(tmp_path, board=False)
    config.checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="wrong SHA-256"):
        config.make()


def test_random_play_draws_every_environment_from_its_own_stream(
    tmp_path: Path,
    short: None,
) -> None:
    del short
    config = RandomSource.Config(steps_per_poll=_DECISIONS)
    config.env.num_envs = 2
    first, second = config.make(), config.make()
    for source in (first, second):
        source.start(schedule=Schedule(arm=1, budget=10**9), run_dir=tmp_path)
    assert first.streams == [1_000_000 * 4 + i for i in range(2)]
    played = [e.actions.tolist() for e in first.poll()]
    assert played == [e.actions.tolist() for e in second.poll()]
    # Each row's own stream: the two rows' episodes differ.
    assert len(played) == 2
    assert played[0] != played[1]


def test_random_play_takes_one_step_per_poll_but_not_none() -> None:
    RandomSource.Config(steps_per_poll=1).make()
    with pytest.raises(ValueError, match="positive steps_per_poll"):
        RandomSource.Config(steps_per_poll=0).make()


def test_random_play_raises_a_failure_once_its_episodes_are_taken(
    tmp_path: Path,
    short: None,
) -> None:
    del short
    config = RandomSource.Config(steps_per_poll=_DECISIONS)
    config.env.num_envs = 2
    config.env.max_decisions = _DECISIONS - 1
    source = config.make()
    source.start(schedule=Schedule(budget=10**9), run_dir=tmp_path)
    with pytest.raises(CaptureError, match=f"exceeds {_DECISIONS - 1}"):
        source.poll()


def _policy_source(directory: Path, *, board: bool) -> PolicySource.Config:
    """Return a policy source of a tiny float32 policy on the CPU, its checkpoint written."""
    step = CraftaxTrainStep.Config()
    step.model = (tiny_board_policy if board else tiny_policy)(dtype=torch.float32)
    step.env.rules.previous_action = board
    step.sampler = TorchPhiloxSampler.Config()
    step.rollout.horizon = _DECISIONS
    torch.manual_seed(0)
    path = directory / "policy.pt"
    torch.save({"step": {"model": step.model.make().state_dict()}}, path)
    config = PolicySource.Config()
    config.policy = step
    config.checkpoint = path
    config.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    config.env.num_envs = 1
    config.device = "cpu"
    return config


def _poll(source: PolicySource) -> list[Captured]:
    """Poll ``source`` until it has finished; return every episode it handed over."""
    episodes: list[Captured] = []
    for _ in range(10):
        finished = source.finished()
        episodes += source.poll()
        if finished:
            break
    return episodes


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
