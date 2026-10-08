"""Tests for the world choice: outcomes per world, their distance and the ranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import sys

import pytest
import torch

from priml.baselines.craftax.game.state import NUM_LEVELS, Achievement
from priml.baselines.craftax.ghosts import select_world
from priml.baselines.craftax.ghosts.select_world import (
    Evaluation,
    Outcomes,
    distance,
    evaluation,
    outcomes,
    rank,
)
from priml.baselines.craftax.metric import FLOOR_NAMES
from priml.baselines.craftax.world_model.archive import (
    FloorTrace,
    Receipt,
    ReplayEpisode,
    write_replay_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def test_outcomes_join_both_splits_of_each_world(tmp_path: Path) -> None:
    _archive(
        tmp_path,
        arm=1,
        episodes=[
            _episode(world=5, split=TRAIN, returned=10, deepest=1, decisions=3),
            _episode(world=5, split=TRAIN, returned=20, deepest=3, decisions=5),
            _episode(world=5, split=VALIDATION, returned=30, deepest=8, decisions=7),
            _episode(world=9, split=TRAIN, returned=4, deepest=0, decisions=2),
        ],
    )
    played = outcomes(tmp_path, arm=1)
    assert list(played) == [5, 9]
    assert played[5] == Outcomes(
        episodes=3,
        mean_return=20.0,
        mean_decisions=5.0,
        reach=(1.0, 1.0, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3),
        deaths=2,
        timeouts=1,
        wins=1,
    )
    assert played[9].reach == (1.0,) + (0.0,) * 8
    assert outcomes(tmp_path, arm=0) == {}


def test_distance_adds_the_return_error_and_the_deepest_floors_moved() -> None:
    evaluated = Evaluation(mean_return=100.0, reach=(1.0, 0.5, 0.25) + (0.0,) * 6)
    world = _outcomes(mean_return=110.0, reach=(1.0, 0.75) + (0.0,) * 7)
    # A tenth off in return; floors 1 and 2 differ by a quarter each, over a
    # mean deepest floor of three quarters.
    assert distance(world, evaluated) == pytest.approx(0.1 + 0.5 / 0.75)
    same = _outcomes(mean_return=100.0, reach=evaluated.reach)
    assert distance(same, evaluated) == 0.0


def test_rank_sums_the_tiers_over_the_worlds_all_of_them_played() -> None:
    evaluated = Evaluation(mean_return=100.0, reach=(1.0, 1.0) + (0.0,) * 7)
    near, far = (
        _outcomes(mean_return=r, reach=evaluated.reach) for r in (100.0, 150.0)
    )
    ranking = rank(
        {"a": {1: far, 2: near, 3: near, 4: near}, "b": {1: near, 2: far, 3: near}},
        evaluations={"a": evaluated, "b": evaluated},
    )
    assert ranking == [(3, 0.0), (1, 0.5), (2, 0.5)]


def test_an_evaluation_reads_its_return_and_reach(tmp_path: Path) -> None:
    path = _metrics(tmp_path / "metrics.json", returned=34.5, reach=[1.0, 0.7])
    assert evaluation(path) == Evaluation(
        mean_return=34.5,
        reach=(1.0, 0.7) + (0.0,) * 7,
    )


def test_main_writes_the_tables_and_the_chosen_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    for arm, deepest in ((0, 1), (2, 3)):
        _archive(
            root,
            arm=arm,
            episodes=[
                _episode(world=4, split=TRAIN, returned=10, deepest=0, decisions=2),
                _episode(
                    world=7,
                    split=TRAIN,
                    returned=20,
                    deepest=deepest,
                    decisions=2,
                ),
            ],
        )
    early = _metrics(tmp_path / "early.json", returned=20.0, reach=[1.0, 1.0])
    high = _metrics(tmp_path / "high.json", returned=20.0, reach=[1.0] * 4)
    output = tmp_path / "world.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_world.py",
            str(root),
            f"--tier=early:0:{early}",
            f"--tier=high:2:{high}",
            f"--output={output}",
        ],
    )
    assert select_world.main() == 0
    written = from_plain(loads(output.read_text()), dict[str, object])
    assert written["world"] == 7
    tiers = from_plain(written["tiers"], dict[str, dict[str, object]])
    assert from_plain(tiers["high"]["worlds"], dict[str, dict[str, object]])["7"][
        "distance"
    ] == pytest.approx(0.0)


def test_a_tier_spec_without_an_arm_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_world.py", str(tmp_path), "--tier=early:x:m.json", "--output=o.json"],
    )
    with pytest.raises(SystemExit):
        select_world.main()


def _episode(
    *,
    world: int,
    split: int,
    returned: int,
    deepest: int,
    decisions: int,
) -> ReplayEpisode:
    """Return a published episode: death off floor 3, a timeout on it, a win on 8."""
    summary = {
        "episode": 0,
        "environment": 0,
        "return": returned,
        "death": int(deepest != 3),
        "timeout": int(deepest == 3),
        "achievements": [0, int(Achievement.DEFEAT_NECROMANCER)]
        if deepest == 8
        else [0],
        "floors": [{"reached": int(k <= deepest)} for k in range(NUM_LEVELS)],
    }
    return ReplayEpisode(
        receipt=Receipt(
            world_seed=world,
            sampling_seed=0,
            initial_state_hash=0,
            arm=0,
            split=split,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(2, dtype=torch.int64),
        snapshots=b"",
        floors=FloorTrace(changes=((0, 0),), died=deepest != 3),
        summary=summary,
    )


def _archive(root: Path, *, arm: int, episodes: Sequence[ReplayEpisode]) -> None:
    """Publish an arm's episodes as one shard per split, as worker 0's."""
    for split in (TRAIN, VALIDATION):
        directory = shard_directory(root, split=split, arm=arm, worker=0)
        directory.mkdir(parents=True)
        mine = [e for e in episodes if e.receipt.split == split]
        if mine:
            write_replay_shard(
                directory,
                index=0,
                episodes=mine,
                stride=256,
                provenance={},
            )


def _outcomes(*, mean_return: float, reach: tuple[float, ...]) -> Outcomes:
    """Return outcomes with this return and reach; the counts play no part."""
    return Outcomes(
        episodes=1,
        mean_return=mean_return,
        mean_decisions=1.0,
        reach=reach,
        deaths=0,
        timeouts=0,
        wins=0,
    )


def _metrics(path: Path, *, returned: float, reach: list[float]) -> Path:
    """Write an evaluation's ``metrics.json``; floors past ``reach`` are unreached."""
    floors = reach + [0.0] * (NUM_LEVELS - len(reach))
    metrics = {
        "eval/craftax_episode_return": returned,
        "eval/craftax_perf": returned / 226,
        **{f"eval/craftax_{n}": r for n, r in zip(FLOOR_NAMES, floors, strict=True)},
    }
    path.write_text(json.dumps(metrics))
    return path


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
