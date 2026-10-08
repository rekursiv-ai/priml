"""Tests for the episode sources: a trained policy's checkpoint, and random play."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib

import pytest
import torch

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
    from pathlib import Path

    from priml.baselines.craftax.world_model.capture.env import Captured


# A source finishes every episode it starts, so each environment plays a whole
# episode with the policy, ~500 rollout steps: 0.30 s warm on x86.
@pytest.mark.compute_large_fixture
def test_a_policy_source_plays_its_checkpoints_policy(tmp_path: Path) -> None:
    config = _policy_source(tmp_path, board=False)
    source = config.make()
    try:
        source.start(schedule=Schedule(arm=2, worker=1, budget=300), run_dir=tmp_path)
        episodes = _poll(source)
    finally:
        source.close()
    assert source.provenance()["rollout_seed"] == str(73 + 1_000_000 * 9)
    assert source.provenance()["checkpoint_sha256"] == config.checkpoint_sha256
    assert sum(len(e.actions) for e in episodes) >= 300
    for episode in episodes:
        assert episode.receipt.arm == 2
        assert replay.verify(episode) == replay.MATCHED


# Even a budget of one decision plays each environment's episode to its end,
# ~350 steps of the convolutional board policy: 0.47 s warm on x86.
@pytest.mark.compute_large_fixture
def test_a_policy_source_writes_its_policys_observation_layout(tmp_path: Path) -> None:
    source = _policy_source(tmp_path, board=True).make()
    try:
        source.start(schedule=Schedule(budget=1), run_dir=tmp_path)
        assert source.env is not None
        assert source.env.observations.shape[-1] == 844
        episodes = _poll(source)
    finally:
        source.close()
    assert episodes


def test_a_checkpoint_of_another_sha256_is_refused(tmp_path: Path) -> None:
    config = _policy_source(tmp_path, board=False)
    config.checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="wrong SHA-256"):
        config.make()


def test_random_play_draws_every_environment_from_its_own_stream(
    tmp_path: Path,
) -> None:
    config = RandomSource.Config(steps_per_poll=8)
    config.env.num_envs = 3
    first, second = config.make(), config.make()
    for source in (first, second):
        source.start(schedule=Schedule(arm=1, budget=10**9), run_dir=tmp_path)
    assert first.streams == [1_000_000 * 4 + i for i in range(3)]
    for _ in range(40):
        assert [e.actions.tolist() for e in first.poll()] == [
            e.actions.tolist() for e in second.poll()
        ]


def test_random_play_takes_one_step_per_poll_but_not_none() -> None:
    RandomSource.Config(steps_per_poll=1).make()
    with pytest.raises(ValueError, match="positive steps_per_poll"):
        RandomSource.Config(steps_per_poll=0).make()


def test_random_play_raises_a_failure_once_its_episodes_are_taken(
    tmp_path: Path,
) -> None:
    config = RandomSource.Config(steps_per_poll=8)
    config.env.num_envs = 2
    config.env.max_decisions = 4
    source = config.make()
    source.start(schedule=Schedule(budget=10**9), run_dir=tmp_path)
    with pytest.raises(CaptureError, match="exceeds 4"):
        source.poll()


def _policy_source(directory: Path, *, board: bool) -> PolicySource.Config:
    """Return a policy source of a tiny policy on the CPU, its checkpoint written."""
    step = CraftaxTrainStep.Config()
    step.model = tiny_board_policy() if board else tiny_policy()
    step.env.rules.previous_action = board
    step.sampler = TorchPhiloxSampler.Config()
    step.rollout.horizon = 16
    torch.manual_seed(0)
    path = directory / "policy.pt"
    torch.save({"step": {"model": step.model.make().state_dict()}}, path)
    config = PolicySource.Config()
    config.policy = step
    config.checkpoint = path
    config.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    config.env.num_envs = 2
    config.device = "cpu"
    return config


def _poll(source: PolicySource) -> list[Captured]:
    """Poll ``source`` until it has finished; return every episode it handed over."""
    episodes: list[Captured] = []
    for _ in range(1_000):
        finished = source.finished()
        episodes += source.poll()
        if finished:
            break
    return episodes


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
