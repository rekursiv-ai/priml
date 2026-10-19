"""Tests for the viewer's scripted episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from priml.baselines.craftax.eager import eager
from priml.baselines.craftax.game.state import (
    DEFAULT_MAX_TIMESTEPS,
    BlockType,
    env_state,
)
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.viewer import testing


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def tiny() -> Generator[None]:
    with eager(world=testing.world):
        yield


def test_the_world_pens_the_cow_and_leaves_the_player_tired() -> None:
    states, _ = replay.reset_world(4)
    state = env_state(states, 0)
    row, col = testing.COW
    for dr, dc in ((-1, 0), (1, 0), (0, -1)):
        assert state.map[0, row + dr, col + dc] == BlockType.STONE
    cows = state.passive_mobs[0]
    assert (int(cows.mask[0]), int(cows.type_id[0])) == (1, 0)
    assert (int(cows.position[0, 0]), int(cows.position[0, 1])) == testing.COW
    assert int(state.player_energy) == 8
    assert float(state.player_fatigue) == testing.SLEEP_TICKS - 13
    expected_tick = DEFAULT_MAX_TIMESTEPS - (
        len(testing.SCRIPT) - 1 + testing.SLEEP_TICKS
    )
    assert int(state.timestep) == expected_tick


def test_the_record_plays_the_script_from_world_4() -> None:
    record = testing.record()
    assert record.actions.tolist() == [int(action) for action in testing.SCRIPT]
    assert (record.receipt.world_seed, record.receipt.sampling_seed) == (4, 2)


def test_the_episode_replays_one_frame_per_decision_and_ends() -> None:
    episode = testing.episode()
    decisions = len(testing.SCRIPT)
    assert episode.actions.tolist() == [int(action) for action in testing.SCRIPT]
    assert episode.cells.shape[0] == decisions
    assert episode.reward.shape == episode.done.shape == (decisions,)
    assert bool(episode.done[-1])
    assert not bool(episode.done[:-1].any())
    assert torch.equal(episode.hashes, testing.record().hashes)
    assert np.unique(episode.cells.numpy()).size > 1


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
