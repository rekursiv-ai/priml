"""Tests for the endings report: the decision of an unlock, and each tier's endings."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import sys

import pytest
import torch

from priml.baselines.craftax.eager import eager, scripted, tiny_world
from priml.baselines.craftax.game.state import (
    DEFAULT_MAX_TIMESTEPS,
    Achievement,
    Action,
)
from priml.baselines.craftax.ghosts import endings
from priml.baselines.craftax.ghosts.endings import (
    Ending,
    episode_endings,
    summary,
    unlocked_at,
)
from priml.baselines.craftax.world_model.archive import (
    FloorTrace,
    Receipt,
    Record,
    ReplayEpisode,
    write_replay_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import TRAIN, VALIDATION
from priml.baselines.craftax.world_model.capture.worker import shard_directory
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.craftax.world_model.replay import Replayable


_LENGTH: Final = 1_000


@pytest.mark.parametrize("unlock", [1, 2, 3, 4, 7, 8, 9, 500, 999, _LENGTH])
def test_unlocked_at_finds_the_first_state_that_holds_the_achievement(
    monkeypatch: pytest.MonkeyPatch,
    unlock: int,
) -> None:
    # The state before decision ``d`` holds the achievement from ``unlock`` on;
    # the final state, after the last decision, is the episode's length.
    probed: list[int] = []
    monkeypatch.setattr(
        endings,
        "_holds",
        partial(_holds_from, unlock=unlock, probed=probed),
    )
    record = _record(_LENGTH)
    assert unlocked_at(record, achievement=Achievement.COLLECT_WOOD, after=0) == unlock
    assert len(probed) <= 2 * _LENGTH.bit_length()
    assert all(0 < d < _LENGTH for d in probed)


def test_unlocked_at_searches_only_from_its_first_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[int] = []
    monkeypatch.setattr(
        endings,
        "_holds",
        partial(_holds_from, unlock=700, probed=probed),
    )
    assert unlocked_at(_record(_LENGTH), achievement=0, after=600) == 700
    assert min(probed) > 600


def test_the_state_before_a_decision_holds_what_the_decisions_before_it_unlocked() -> (
    None
):
    # Facing the tree above the start, the second decision cuts it: wood.
    world = partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 3)
    with eager(world=world):
        record = scripted([Action.NOOP, Action.DO, Action.NOOP], world_seed=7)
        holds = [
            endings._holds(record, achievement=Achievement.COLLECT_WOOD, decision=d)
            for d in (1, 2)
        ]
        never = endings._holds(
            record,
            achievement=Achievement.COLLECT_STONE,
            decision=2,
        )
    assert holds == [False, True]
    assert not never


def test_a_claimed_win_is_timed_by_replay_from_the_last_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searches: list[tuple[int, int]] = []
    monkeypatch.setattr(endings, "unlocked_at", partial(_unlocked, searches=searches))
    _publish(
        tmp_path,
        split=TRAIN,
        episodes=[
            _episode(0, decisions=90, floors=((0, 0), (40, 8), (60, 7), (70, 8))),
            _episode(1, decisions=50, floors=((0, 0), (20, 1)), death=True),
        ],
    )
    _publish(
        tmp_path,
        split=VALIDATION,
        episodes=[_episode(2, decisions=30, floors=((0, 0),), win=False)],
    )
    ends = episode_endings(tmp_path, arm=3)
    # The necromancer lives on the last floor: the search starts where the
    # episode first stood on it.
    assert searches == [(Achievement.DEFEAT_NECROMANCER, 40)]
    assert ends == [
        Ending(ordinal=0, decisions=90, outcome="win", end=_UNLOCKED),
        Ending(ordinal=1, decisions=50, outcome="death", end=50),
        Ending(ordinal=2, decisions=30, outcome="timeout", end=30),
    ]


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
    root = tmp_path / "capture"
    _publish(
        root,
        split=TRAIN,
        episodes=[
            _episode(k, decisions=d, floors=((0, 0),), death=True)
            for k, d in ((0, 300), (2, 20_000))
        ],
    )
    _publish(
        root,
        split=VALIDATION,
        episodes=[_episode(1, decisions=200, floors=((0, 0),), death=True)],
    )
    output = tmp_path / "endings.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["endings.py", str(root), "--tier=boss:3", f"--output={output}"],
    )
    assert endings.main() == 0
    (row,) = from_plain(loads(output.read_text()), list[dict[str, object]])
    assert (row["tier"], row["root"], row["death"]) == ("boss", str(root), 3)
    assert row["ended_within"] == 2
    episodes = from_plain(row["episodes"], list[dict[str, object]])
    assert [e["ordinal"] for e in episodes] == [0, 1, 2]


def test_a_tier_spec_without_an_arm_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["endings.py", str(tmp_path), "--tier=boss", f"--output={tmp_path / 'e.json'}"],
    )
    with pytest.raises(SystemExit):
        endings.main()


_UNLOCKED: Final = 77
"""The decision the stand-in search reports for every win."""


def _holds_from(
    record: Replayable,
    *,
    achievement: int,
    decision: int,
    unlock: int,
    probed: list[int],
) -> bool:
    """Stand in for ``endings._holds``: whether ``decision`` is at or past ``unlock``."""
    del record, achievement
    probed.append(decision)
    return decision >= unlock


def _unlocked(
    record: Replayable,
    *,
    achievement: int,
    after: int,
    searches: list[tuple[int, int]],
) -> int:
    """Stand in for ``endings.unlocked_at``: note the search, report ``_UNLOCKED``."""
    del record
    searches.append((achievement, after))
    return _UNLOCKED


def _record(decisions: int) -> Record:
    """Return a record of ``decisions`` NOOPs; only its length is read."""
    return Record(
        receipt=Receipt(
            world_seed=1,
            sampling_seed=0,
            initial_state_hash=0,
            arm=3,
            split=TRAIN,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(-(-decisions // 256) + 1, dtype=torch.int64),
    )


def _episode(
    ordinal: int,
    *,
    decisions: int,
    floors: tuple[tuple[int, int], ...],
    death: bool = False,
    win: bool = True,
) -> ReplayEpisode:
    """Return a replay-shard episode whose summary claims the win unless it died."""
    record = _record(decisions)
    claimed = [Achievement.COLLECT_WOOD.value]
    if win and not death:
        claimed.append(Achievement.DEFEAT_NECROMANCER.value)
    return ReplayEpisode(
        receipt=record.receipt,
        actions=record.actions,
        hashes=record.hashes,
        snapshots=b"",
        floors=FloorTrace(changes=floors, died=death),
        summary={"episode": ordinal, "death": int(death), "achievements": claimed},
    )


def _publish(root: Path, *, split: int, episodes: list[ReplayEpisode]) -> None:
    """Write ``episodes`` as arm 3's first replay shard of ``split``."""
    directory = shard_directory(root, split=split, arm=3, worker=0)
    directory.mkdir(parents=True)
    write_replay_shard(directory, index=0, episodes=episodes, stride=256, provenance={})


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
