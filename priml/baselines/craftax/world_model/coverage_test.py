"""Check coverage counters, transition signatures, merging, and the report."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import dataclasses
import json

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    Receipt,
    read_manifest,
    write_shard,
)
from priml.baselines.craftax.world_model.coverage import (
    Coverage,
    count_shard,
    load_coverage,
    merge,
    report,
    transition_signatures,
)
from priml.baselines.craftax.world_model.index import FLOOR_AUX
from priml.baselines.craftax.world_model.index_test import inline_pools
from priml.baselines.craftax.world_model.snapshots_test import (
    replay_twin,
)
from priml.lib.codec import from_plain, loads


def _episode(
    decisions: int,
    *,
    floors: list[int] | None = None,
    split: int = 0,
    achievements: list[int] | None = None,
) -> Episode:
    aux = torch.zeros(decisions, 51, dtype=torch.int16)
    aux[:, 31] = 1
    if floors is not None:
        aux[:, FLOOR_AUX] = torch.tensor(floors, dtype=torch.int16)
    summary = {"achievements": achievements or []}
    return Episode(
        receipt=Receipt(
            world_seed=0,
            sampling_seed=0,
            initial_state_hash=0,
            arm=0,
            split=split,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=torch.zeros(decisions, 99, 8, dtype=torch.uint8),
        aux=aux,
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=torch.zeros(decisions, dtype=torch.bool),
        summary=summary,
    )


def _distinct(episode: Episode) -> int:
    return len(transition_signatures(episode).unique(dim=0))


def test_values_are_counted_per_floor_on_the_training_split() -> None:
    train = _episode(3, floors=[0, 0, 1])
    train.cells[:2, :, 0] = 5
    train.cells[2, :, 0] = 7
    train.cells[0, 4, 3] = 2
    train.cells[1, 4, 6] = 63
    val = _episode(2, split=1)
    val.cells[:, :, 0] = 9
    coverage = count_shard([train, val])
    assert coverage.decisions.tolist() == [2, 1] + [0] * 7
    assert coverage.cells[0, 0, 5] == 2 * 99
    assert coverage.cells[1, 0, 7] == 99
    assert coverage.cells[0, 0, 9] == 0
    assert coverage.cells[0, 3, [0, 2]].tolist() == [2 * 99 - 1, 1]
    assert coverage.cells[0, 6, [0, 63]].tolist() == [2 * 99 - 1, 1]
    assert int(coverage.cells[0, [1, 2, 4, 5, 7], 0].sum()) == 5 * 2 * 99
    assert coverage.aux[0, FLOOR_AUX, 0] == 2
    assert coverage.aux[1, FLOOR_AUX, 1] == 1


def test_reach_counts_episodes_per_split_and_floor() -> None:
    coverage = count_shard(
        [
            _episode(3, floors=[0, 1, 1]),
            _episode(2, floors=[0, 0]),
            _episode(1, floors=[2], split=1),
        ],
    )
    assert coverage.reach[0, :3].tolist() == [2, 1, 0]
    assert coverage.reach[1, :3].tolist() == [0, 0, 1]


def test_achievements_come_from_training_summaries() -> None:
    coverage = count_shard(
        [
            _episode(1, achievements=[0, 5]),
            _episode(1, achievements=[5]),
            _episode(1, split=1, achievements=[6]),
        ],
    )
    assert coverage.achievements[[0, 5, 6]].tolist() == [1, 2, 0]
    assert int(count_shard([_episode(1, achievements=[66])]).achievements[66]) == 1


@pytest.mark.parametrize(
    "summary",
    [
        {},
        {"achievements": None},
        {"achievements": "0,5"},
        {"achievements": {"0": True}},
        {"achievements": [0, "5"]},
        {"achievements": [0, 5.0]},
        {"achievements": [True]},
        {"achievements": [5, 0]},
        {"achievements": [5, 5]},
        {"achievements": [-1]},
        {"achievements": [67]},
    ],
)
@pytest.mark.parametrize("split", [0, 1])
def test_malformed_achievements_fail_closed(
    summary: dict[str, object],
    split: int,
) -> None:
    episode = dataclasses.replace(_episode(1, split=split), summary=summary)
    with pytest.raises(ValueError, match="achievements"):
        count_shard([episode])


def _set_action(e: Episode) -> None:
    e.actions[0] = 3


def _set_facing_block(e: Episode) -> None:
    e.cells[0, 48, 0] = 5


def _set_facing_item(e: Episode) -> None:
    e.cells[0, 48, 1] = 2


def _set_neighbor_mob(e: Episode) -> None:
    e.cells[0, 38, 3] = 2


def _set_inventory_gain(e: Episode) -> None:
    e.aux[1, 0] = 1


def _set_inventory_loss(e: Episode) -> None:
    e.aux[0, 21] = 1


def _set_reward(e: Episode) -> None:
    e.reward[0] = 1


def _set_done(e: Episode) -> None:
    e.done[0] = True


def _set_floor(e: Episode) -> None:
    e.aux[0, FLOOR_AUX] = 1


def _set_other_neighbor_block(e: Episode) -> None:
    e.cells[0, 50, 0] = 5


def _set_far_mob(e: Episode) -> None:
    e.cells[0, 0, 3] = 2


def _set_health(e: Episode) -> None:
    e.aux[1, 22] = 5


@pytest.mark.parametrize(
    "mutate",
    [
        _set_action,
        _set_facing_block,
        _set_facing_item,
        _set_neighbor_mob,
        _set_inventory_gain,
        _set_inventory_loss,
        _set_reward,
        _set_done,
        _set_floor,
    ],
)
def test_each_signature_component_changes_the_signature(
    mutate: Callable[[Episode], None],
) -> None:
    episode = _episode(2)
    base = transition_signatures(episode)[0].clone()
    mutate(episode)
    assert not torch.equal(transition_signatures(episode)[0], base)


@pytest.mark.parametrize(
    "mutate",
    [_set_other_neighbor_block, _set_far_mob, _set_health],
)
def test_signature_ignores_everything_else(mutate: Callable[[Episode], None]) -> None:
    episode = _episode(2)
    base = transition_signatures(episode)[0].clone()
    mutate(episode)
    assert torch.equal(transition_signatures(episode)[0], base)


# Cells beside the player at slot 49 (row 4, column 5 of the 9 x 11 view), in
# the order of the facing flags aux 31-34: left, right, up, down.
_BESIDE = (48, 50, 38, 60)


def _facing(direction: int) -> Episode:
    episode = _episode(2)
    episode.aux[:, 31:35] = 0
    episode.aux[:, 31 + direction] = 1
    return episode


@pytest.mark.parametrize("direction", range(4))
def test_signature_reads_the_cell_the_player_faces(direction: int) -> None:
    base = transition_signatures(_facing(direction))[0]
    for slot in _BESIDE:
        episode = _facing(direction)
        episode.cells[0, slot, 0] = 5
        changed = not torch.equal(transition_signatures(episode)[0], base)
        assert changed == (slot == _BESIDE[direction]), slot


def test_last_decision_has_no_inventory_change() -> None:
    episode = _episode(2)
    episode.aux[1, 0] = 1
    assert _distinct(episode) == 2
    assert transition_signatures(episode).shape == (2, 2)


def _assert_same(left: Coverage, right: Coverage) -> None:
    for name in Coverage.__dataclass_fields__:
        assert torch.equal(
            cast("torch.Tensor", getattr(left, name)),
            cast("torch.Tensor", getattr(right, name)),
        ), name


def _varied(decisions: int, *, seed: int, split: int = 0) -> Episode:
    generator = torch.Generator().manual_seed(seed)
    episode = _episode(
        decisions,
        floors=from_plain(
            torch.randint(0, 3, (decisions,), generator=generator).tolist(),
            list[int],
        ),
        split=split,
        achievements=[seed],
    )
    episode.actions[:] = torch.randint(0, 4, (decisions,), generator=generator)
    episode.cells[:, 48, 0] = torch.randint(0, 3, (decisions,), generator=generator)
    return episode


def test_merge_equals_counting_the_concatenation() -> None:
    first = [_varied(20, seed=1), _varied(5, seed=2, split=1)]
    second = [_varied(30, seed=3)]
    merged = merge([count_shard(first), count_shard(second)])
    _assert_same(merged, count_shard(first + second))


def _signature_episode(actions: list[int], *, floor: int = 0) -> Episode:
    episode = _episode(len(actions), floors=[floor] * len(actions))
    episode.actions[:] = torch.tensor(actions, dtype=torch.uint8)
    return episode


def _loose(
    parts: list[Coverage],
    *,
    min_train_reach: int = 1,
    min_count: int = 1,
    max_new_rate: float = 2.0,
    increment_decisions: int = 1,
) -> dict[str, object]:
    result = report(
        parts,
        min_train_reach=min_train_reach,
        min_val_reach=0,
        min_count=min_count,
        max_unseen_mass=2.0,
        max_new_rate=max_new_rate,
        increment_decisions=increment_decisions,
    )
    return from_plain(loads(json.dumps(result)), dict[str, object])


def _floor(result: dict[str, object], floor: int) -> dict[str, object]:
    return from_plain(result["floors"], list[dict[str, object]])[floor]


def _values(entry: dict[str, object], group: str, field: str) -> list[int]:
    return from_plain(from_plain(entry[group], dict[str, object])[field], list[int])


def test_good_turing_unseen_mass_is_singletons_over_transitions() -> None:
    parts = [count_shard([_signature_episode([1, 1, 2, 3])])]
    floor = _floor(_loose(parts), 0)
    assert floor["transitions"] == 4
    assert floor["signatures"] == 3
    assert floor["singletons"] == 2
    assert floor["unseen_mass"] == 0.5


def test_new_signature_rate_compares_the_last_increment_with_the_rest() -> None:
    parts = [
        count_shard([_signature_episode([1, 2, 2, 2])]),
        count_shard([_signature_episode([1, 3])]),
    ]
    assert _floor(_loose(parts, increment_decisions=2), 0)["new_rate"] == 0.5
    assert _floor(_loose(parts, increment_decisions=10), 0)["new_rate"] is None


def test_report_lists_rare_and_never_observed_values() -> None:
    episode = _signature_episode([0, 0])
    episode.cells[0, 0, 0] = 5
    floor = _floor(_loose([count_shard([episode])], min_count=2), 0)
    assert _values(floor, "rare", "block") == [5]
    assert 1 in _values(floor, "never", "block")
    assert 0 not in _values(floor, "never", "block")
    assert _values(floor, "never", "floor") == list(range(1, 9))


# Read straight from the report: its JSON round trip, which ``_loose`` tests, walks
# every floor's value lists.
def _checks(
    parts: list[Coverage],
    *,
    min_train_reach: int = 1,
    min_count: int = 0,
    max_new_rate: float = 2.0,
) -> dict[str, bool]:
    """Return the loose report's checks and its verdict as ``sufficient``."""
    result = report(
        parts,
        min_train_reach=min_train_reach,
        min_val_reach=0,
        min_count=min_count,
        max_unseen_mass=2.0,
        max_new_rate=max_new_rate,
        increment_decisions=1,
    )
    checks = from_plain(result["checks"], dict[str, bool])
    return {**checks, "sufficient": from_plain(result["sufficient"], bool)}


def test_report_sufficiency_checks() -> None:
    parts = [count_shard([_signature_episode([1, 1, 1, 1], floor=f)]) for f in range(9)]
    assert _checks(parts) == {
        "reach": True,
        "values": True,
        "missing_mass": True,
        "sufficient": True,
    }
    assert _checks(parts, min_train_reach=2) == {
        "reach": False,
        "values": True,
        "missing_mass": True,
        "sufficient": False,
    }
    assert _checks(parts, min_count=5)["values"] is False
    # The increment is floor 8's shard, whose one signature is new: rate 1.0.
    assert _checks(parts, max_new_rate=1.0)["missing_mass"] is False


def test_load_coverage_builds_once_then_reads_the_cache(tmp_path: Path) -> None:
    shards = tmp_path / "w0"
    shards.mkdir()
    line = write_shard(shards, index=0, episodes=[_varied(9, seed=4)], provenance={})
    cache = tmp_path / "cache"
    built = load_coverage(shards, line, coverage_dir=cache)
    for path in shards.glob("shard-*"):
        path.unlink()
    _assert_same(load_coverage(shards, line, coverage_dir=cache), built)


def test_republished_summaries_miss_the_coverage_cache(tmp_path: Path) -> None:
    # The two shards share their token frames; only the achievements differ.
    cache = tmp_path / "cache"
    for achievement in (1, 2):
        directory = tmp_path / f"achievement{achievement}"
        directory.mkdir()
        episodes = [_episode(3, achievements=[achievement])]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        coverage = load_coverage(directory, line, coverage_dir=cache, workers=1)
        assert coverage.achievements.nonzero()[:, 0].tolist() == [achievement]


def test_load_coverage_defaults_to_the_measured_fastest_pool() -> None:
    assert (load_coverage.__kwdefaults__ or {})["workers"] == 4


def test_chunked_coverage_equals_counting_the_whole_shard(tmp_path: Path) -> None:
    episodes = [_varied(9, seed=4), _varied(6, seed=5, split=1), _varied(7, seed=6)]
    line = write_shard(tmp_path, index=0, episodes=episodes, provenance={})
    chunked = load_coverage(
        tmp_path,
        line,
        coverage_dir=tmp_path / "cache",
        workers=1,
        chunk_decisions=1,
    )
    _assert_same(chunked, count_shard(episodes))


def test_a_replay_shard_counts_as_its_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = [_varied(9, seed=4), _varied(6, seed=5, split=1), _varied(7, seed=6)]
    name = "train/arm0/w0"
    (tmp_path / "frames" / name).mkdir(parents=True)
    write_shard(tmp_path / "frames" / name, index=0, episodes=episodes, provenance={})
    replay_twin(tmp_path / "frames", tmp_path / "replay", monkeypatch)
    directory = tmp_path / "replay" / name
    (line,) = read_manifest(directory)
    coverage = load_coverage(
        directory,
        line,
        coverage_dir=tmp_path / "cache",
        workers=1,
        chunk_decisions=1,
    )
    _assert_same(coverage, count_shard(episodes))


def test_coverage_in_a_process_pool_equals_counting_the_whole_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = inline_pools(monkeypatch)
    episodes = [_varied(9, seed=4), _varied(6, seed=5, split=1), _varied(7, seed=6)]
    line = write_shard(tmp_path, index=0, episodes=episodes, provenance={})
    pooled = load_coverage(
        tmp_path,
        line,
        coverage_dir=tmp_path / "cache",
        workers=2,
        chunk_decisions=1,
    )
    assert [workers for workers, _, _ in built] == [2]
    _assert_same(pooled, count_shard(episodes))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
