"""Tests for the endings report: the decision of an unlock, and each tier's endings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import numpy as np

from priml.baselines.craftax.game.state import (
    Achievement,
    env_state,
)
from priml.baselines.craftax.ghosts import endings
from priml.baselines.craftax.ghosts.endings import (
    Ending,
    episode_endings,
    summary,
    unlocked_at,
)
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    FloorTrace,
    ReplayEpisode,
    write_replay_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import TRAIN
from priml.baselines.craftax.world_model.capture.source import (
    RandomSource,
)
from priml.baselines.craftax.world_model.capture.worker import (
    CaptureWorker,
    shard_directory,
)
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_unlocked_at_finds_the_first_state_that_holds_the_achievement() -> None:
    episode = replay.record(world_seed=7, sampling_seed=11, max_decisions=100_000)
    last = len(episode.actions) - 1
    start, final = _achievements(episode, 0), _achievements(episode, last)
    achievement = np.flatnonzero(final & ~start).item(-1)
    found = unlocked_at(episode, achievement=achievement, after=0)
    assert 0 < found <= last
    assert _achievements(episode, found)[achievement]
    assert not _achievements(episode, found - 1)[achievement]


def test_an_achievement_never_unlocked_reads_as_the_episodes_length() -> None:
    episode = replay.record(world_seed=7, sampling_seed=11, max_decisions=100_000)
    found = unlocked_at(
        episode,
        achievement=Achievement.DEFEAT_NECROMANCER,
        after=0,
    )
    assert found == len(episode.actions)


def test_a_claimed_win_is_timed_by_replay_from_the_last_floor(tmp_path: Path) -> None:
    episode = replay.record(world_seed=7, sampling_seed=11, max_decisions=100_000)
    directory = shard_directory(tmp_path, split=TRAIN, arm=3, worker=0)
    directory.mkdir(parents=True)
    claimed = ReplayEpisode(
        receipt=episode.receipt,
        actions=episode.actions,
        hashes=episode.hashes,
        snapshots=b"",
        floors=FloorTrace(changes=((0, 0), (40, 8)), died=True),
        summary={
            "episode": 0,
            "death": 1,
            "achievements": [int(Achievement.DEFEAT_NECROMANCER)],
        },
    )
    write_replay_shard(
        directory,
        index=0,
        episodes=[claimed],
        stride=256,
        provenance={},
    )
    (end,) = episode_endings(tmp_path, arm=3)
    # Random play never fells the necromancer: replay finds no earlier state
    # holding the win than the episode's last.
    assert (end.outcome, end.end) == ("win", len(episode.actions))


def test_summary_counts_endings_within_the_limit_and_their_quantiles() -> None:
    ends = [
        Ending(ordinal=0, decisions=500, outcome="death", end=500),
        Ending(ordinal=1, decisions=90_000, outcome="win", end=9_000),
        Ending(ordinal=2, decisions=99_000, outcome="win", end=12_000),
        Ending(ordinal=3, decisions=8_000, outcome="timeout", end=8_000),
    ]
    counts = summary(ends, within=10_000)
    assert (counts["death"], counts["timeout"], counts["win"]) == (1, 1, 2)
    assert counts["ended_within"] == 2
    assert counts["end_decisions"] == {
        "min": 500,
        "p10": 500,
        "p25": 500,
        "p50": 8_000,
        "p75": 9_000,
        "p90": 9_000,
        "max": 12_000,
    }
    assert from_plain(counts["win_decisions"], dict[str, int])["min"] == 9_000
    assert summary([], within=1)["end_decisions"] == {}


def test_main_reports_a_captures_endings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CaptureWorker.Config()
    config.root = tmp_path / "capture"
    config.run_root = tmp_path / "runs"
    config.arm = 3
    config.decisions = 1
    config.poll_seconds = 0.0
    source = config.source = RandomSource.Config()
    source.env.num_envs = 3
    source.env.world_seeds = (15,)
    config.make().capture()
    ends = episode_endings(config.root, arm=3)
    assert [end.ordinal for end in ends] == [0, 1, 2]
    assert {(end.outcome, end.end == end.decisions) for end in ends} == {
        ("death", True),
    }
    output = tmp_path / "endings.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["endings.py", str(config.root), "--tier=boss:3", f"--output={output}"],
    )
    assert endings.main() == 0
    (row,) = from_plain(loads(output.read_text()), list[dict[str, object]])
    assert (row["tier"], row["death"], row["ended_within"]) == ("boss", 3, 3)
    assert len(from_plain(row["episodes"], list[object])) == 3


def _achievements(episode: replay.Replayable, decision: int) -> np.ndarray:
    """Return the achievements of the state before ``decision``, as booleans."""
    start = replay.save(*replay.reset_world(episode.receipt.world_seed))
    states, _ = replay.load(
        replay.xor_bytes(replay.origin(episode, decision=decision), right=start),
    )
    return np.array(env_state(states, 0).achievements, dtype=np.bool)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
