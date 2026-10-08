"""Tests for branch pools: point selection, the pool file, and the feeder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataclasses

import pytest
import torch

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    FloorTrace,
    Receipt,
    ReplayEpisode,
    write_replay_shard,
)
from priml.baselines.craftax.world_model.capture.branches import (
    BranchFeeder,
    BranchPoint,
    read_pool,
    select_points,
    write_pool,
)
from priml.baselines.craftax.world_model.snapshots import stored_episode


if TYPE_CHECKING:
    from pathlib import Path


def test_points_are_first_entries_and_spaced_times_on_the_chosen_floors(
    tmp_path: Path,
) -> None:
    visits = ((0, 0), (100, 5), (3_000, 6), (3_500, 8))
    episodes = [
        _stored(10_000, visits),
        _stored(9_000, visits, split=1),
        _stored(8_000, visits, arm=0),
        _stored(7_000, visits, frames=b"not replayable"),
        _stored(6_000, ((0, 0), (10, 5), (20, 4), (5_000, 5))),
        _stored(5_000, visits, branch=True),
    ]
    line = write_replay_shard(
        tmp_path,
        index=4,
        episodes=episodes,
        stride=256,
        provenance={},
    )
    points = select_points(
        [(tmp_path, line)],
        arm=1,
        floors={5, 6, 8},
        spacing=1_000,
        windows={8: 2_000},
    )
    got = [(p.episode, p.decision, p.floor, p.kind) for p in points]
    assert got == [
        (0, 100, 5, "entry"),
        (0, 1_100, 5, "time"),
        (0, 2_100, 5, "time"),
        (0, 3_000, 6, "entry"),
        (0, 3_500, 8, "entry"),
        (0, 4_500, 8, "time"),
        (4, 10, 5, "entry"),
        (4, 5_990, 5, "time"),
    ]
    assert all((p.directory, p.shard) == (tmp_path, "shard-000004") for p in points)


def test_a_pool_round_trips_in_order(tmp_path: Path) -> None:
    points = [
        BranchPoint(
            directory=tmp_path / "train" / "arm0" / "w1",
            shard="shard-000002",
            episode=episode,
            decision=decision,
            floor=6,
            kind="time",
        )
        for episode, decision in ((3, 900), (0, 12), (3, 5))
    ]
    pool = tmp_path / "pools" / "arm0-w1.jsonl"
    write_pool(pool, points)
    assert read_pool(pool) == points


def test_a_parent_missing_from_the_pools_node_is_named(tmp_path: Path) -> None:
    point = BranchPoint(
        directory=tmp_path / "train" / "arm2" / "w0",
        shard="shard-000000",
        episode=0,
        decision=3,
        floor=0,
        kind="time",
    )
    write_pool(tmp_path / "pools" / "arm2-w1.jsonl", [point])
    feeder = BranchFeeder(
        BranchFeeder.Config(pools=tmp_path / "pools"),
        arm=2,
        worker=1,
        first_episode=0,
    )
    try:
        with pytest.raises(FileNotFoundError, match="publishes no shard-000000"):
            feeder.start(0)
    finally:
        feeder.close()


# Records a parent and replays it on two threads: 0.63 s warm on the Mac, and a
# cold Numba compile of record and replay near CI's 60 s default-tier timeout.
@pytest.mark.compute_large_fixture
def test_the_feeder_hands_out_each_parents_state_then_none(tmp_path: Path) -> None:
    parent = replay.record(world_seed=5, sampling_seed=3, max_decisions=2_000)
    parent = dataclasses.replace(
        parent,
        receipt=dataclasses.replace(parent.receipt, arm=2),
    )
    shard = tmp_path / "train" / "arm2" / "w0"
    shard.mkdir(parents=True)
    line = write_replay_shard(
        shard,
        index=0,
        episodes=[stored_episode(parent, stride=256)],
        stride=256,
        provenance={},
    )
    decisions = [len(parent.actions) - 1, 3, len(parent.actions) - 2]
    points = [
        BranchPoint(
            directory=shard,
            shard=line.shard,
            episode=0,
            decision=decision,
            floor=0,
            kind="time",
        )
        for decision in decisions
    ]
    write_pool(tmp_path / "pools" / "arm2-w1.jsonl", points)
    config = BranchFeeder.Config(pools=tmp_path / "pools", ahead=2, threads=2)
    feeder = BranchFeeder(config, arm=2, worker=1, first_episode=1)
    try:
        for ordinal in (1, 2):
            start = feeder.start(ordinal)
            assert start is not None
            assert start.world_seed == 5
            assert start.point == points[ordinal].to_json()
            assert start.origin == replay.origin(parent, decision=decisions[ordinal])
        assert feeder.start(3) is None
    finally:
        feeder.close()


def _stored(
    decisions: int,
    changes: tuple[tuple[int, int], ...],
    *,
    arm: int = 1,
    split: int = 0,
    frames: bytes = b"",
    branch: bool = False,
) -> ReplayEpisode:
    """Return a replay-shard episode of ``decisions`` NOOPs with a floor trace."""
    return ReplayEpisode(
        receipt=Receipt(
            world_seed=100_000_000 + decisions,
            sampling_seed=0,
            initial_state_hash=0,
            arm=arm,
            split=split,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(-(-decisions // 256) + 1, dtype=torch.int64),
        snapshots=b"",
        floors=FloorTrace(changes=changes, died=False),
        summary={"branch": {"decision": 100}} if branch else {},
        frames=frames,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
