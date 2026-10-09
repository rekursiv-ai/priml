"""Check stratum assignment, run-length spans, and the per-shard index cache."""

from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast

import dataclasses
import pickle
import stat

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    FloorTrace,
    ManifestLine,
    Receipt,
    ReplayEpisode,
    write_replay_shard,
    write_shard,
)
from priml.baselines.craftax.world_model.index import (
    FLOOR_AUX,
    STRATA,
    Event,
    ShardIndex,
    build_index,
    decision_strata,
    floor_trace,
    load_index,
    map_chunks,
    save_tensors,
    shard_key,
    trace_strata,
)


type _Built = tuple[int, object, tuple[object, ...]]
"""A pool's worker count, initializer and its arguments."""


def _episode(floors: list[int], *, died: bool, split: int = 0) -> Episode:
    decisions = len(floors)
    aux = torch.zeros(decisions, 51, dtype=torch.int16)
    aux[:, FLOOR_AUX] = torch.tensor(floors, dtype=torch.int16)
    reward = torch.zeros(decisions, dtype=torch.int16)
    reward[-1] = -1 if died else 0
    return Episode(
        receipt=Receipt(
            world_seed=decisions,
            sampling_seed=0,
            initial_state_hash=0,
            arm=0,
            split=split,
        ),
        actions=torch.zeros(decisions, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=torch.zeros(decisions, 99, 8, dtype=torch.uint8),
        aux=aux,
        reward=reward,
        done=torch.arange(decisions) == decisions - 1,
        summary={},
    )


def _strata(episode: Episode) -> list[int]:
    strata = decision_strata(aux=episode.aux, reward=episode.reward, done=episode.done)
    return [int(s) for s in strata]


def _code(floor: int, event: Event) -> int:
    return floor * len(Event) + event


def test_entry_then_ordinary_then_pre_death_wins_overlap() -> None:
    strata = _strata(_episode([0] * 100 + [1] * 100, died=True))
    assert strata[:64] == [_code(0, Event.ENTRY)] * 64
    assert strata[64:100] == [_code(0, Event.ORDINARY)] * 36
    assert strata[100:136] == [_code(1, Event.ENTRY)] * 36
    assert strata[136:] == [_code(1, Event.PRE_DEATH)] * 64


def test_timeout_has_no_pre_death() -> None:
    strata = _strata(_episode([2] * 130, died=False))
    assert strata == [_code(2, Event.ENTRY)] * 64 + [_code(2, Event.ORDINARY)] * 66


def test_entry_counts_the_first_64_decisions_on_a_floor_across_visits() -> None:
    strata = _strata(_episode([0] * 10 + [1] * 10 + [0] * 100, died=False))
    assert strata[:10] == [_code(0, Event.ENTRY)] * 10
    assert strata[20:74] == [_code(0, Event.ENTRY)] * 54
    assert strata[74:] == [_code(0, Event.ORDINARY)] * 46


def test_index_spans_and_counts_cover_every_decision() -> None:
    episodes = [
        _episode([0] * 70, died=True),
        _episode([0] * 5 + [3] * 5, died=False, split=0),
    ]
    index = build_index(episodes)
    assert index.decisions.tolist() == [70, 10]
    assert index.span_episode.tolist() == [0, 0, 1, 1]
    assert index.span_start.tolist() == [0, 6, 0, 5]
    assert index.span_length.tolist() == [6, 64, 5, 5]
    assert index.span_stratum.tolist() == [
        _code(0, Event.ENTRY),
        _code(0, Event.PRE_DEATH),
        _code(0, Event.ENTRY),
        _code(3, Event.ENTRY),
    ]
    counts = index.counts()
    assert counts.shape == (STRATA,)
    assert counts[_code(0, Event.ENTRY)] == 11
    assert int(counts.sum()) == 80


def test_split_is_recorded_per_episode() -> None:
    index = build_index([_episode([0] * 3, died=False, split=1)])
    assert index.split.tolist() == [1]


def test_load_index_builds_once_then_reads_the_cache(tmp_path: Path) -> None:
    shards = tmp_path / "w0"
    shards.mkdir()
    line = write_shard(
        shards,
        index=0,
        episodes=[_episode([0] * 70, died=True)],
        provenance={},
    )
    cache = tmp_path / "cache"
    built = load_index(shards, line, index_dir=cache)
    assert len(list(cache.iterdir())) == 1
    for path in shards.glob("shard-*"):
        path.unlink()
    cached = load_index(shards, line, index_dir=cache)
    assert torch.equal(cached.span_length, built.span_length)
    assert torch.equal(cached.span_stratum, built.span_stratum)


def test_a_cached_index_is_as_readable_as_any_new_file(tmp_path: Path) -> None:
    # Another user sharing the datasets filesystem reads the cache the umask allows.
    cached = tmp_path / "index.pt"
    save_tensors({"a": torch.zeros(2)}, cached)
    created = tmp_path / "created"
    created.write_bytes(b"")
    assert stat.S_IMODE(cached.stat().st_mode) == stat.S_IMODE(created.stat().st_mode)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["created", "index.pt"]


@pytest.mark.parametrize("suffix", ["bin", "frames", "meta"])
def test_shard_key_changes_with_every_file_digest(suffix: str) -> None:
    line = ManifestLine(
        shard="shard-000000",
        episodes=1,
        decisions=1,
        sha256={"bin": "a" * 64, "frames": "b" * 64, "meta": "c" * 64},
        provenance={},
    )
    digests = {**line.sha256, suffix: "0" * 64}
    assert shard_key(dataclasses.replace(line, sha256=digests)) != shard_key(line)


def test_republished_split_misses_the_index_cache(tmp_path: Path) -> None:
    # The two shards share their token frames; only the split differs.
    cache = tmp_path / "cache"
    for split in (0, 1):
        directory = tmp_path / f"split{split}"
        directory.mkdir()
        episodes = [_episode([0] * 5, died=False, split=split)]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        index = load_index(directory, line, index_dir=cache, workers=1)
        assert index.split.tolist() == [split]


def test_load_index_defaults_to_the_measured_fastest_pool() -> None:
    assert (load_index.__kwdefaults__ or {})["workers"] == 4


def _mixed() -> list[Episode]:
    return [
        _episode([0] * 70, died=True),
        _episode([0] * 5 + [3] * 5, died=False),
        _episode([2] * 3, died=True),
        _episode([1] * 4 + [0] * 66, died=False),
    ]


def _assert_same_index(left: ShardIndex, right: ShardIndex) -> None:
    for field in dataclasses.fields(ShardIndex):
        a = cast("torch.Tensor", getattr(left, field.name))
        b = cast("torch.Tensor", getattr(right, field.name))
        assert torch.equal(a, b), field.name


def test_chunked_index_equals_the_whole_shard_index(tmp_path: Path) -> None:
    episodes = _mixed()
    line = write_shard(tmp_path, index=0, episodes=episodes, provenance={})
    chunked = load_index(
        tmp_path,
        line,
        index_dir=tmp_path / "cache",
        workers=1,
        chunk_decisions=60,
    )
    _assert_same_index(chunked, build_index(episodes))


def test_map_chunks_splits_at_episode_boundaries_in_order(tmp_path: Path) -> None:
    line = write_shard(tmp_path, index=0, episodes=_mixed(), provenance={})
    sizes = map_chunks(tmp_path, line, _decisions, workers=1, chunk_decisions=60)
    assert sizes == [[70], [10, 3, 70]]
    # A chunk that reaches chunk_decisions exactly closes there.
    exact = map_chunks(tmp_path, line, _decisions, workers=1, chunk_decisions=70)
    assert exact == [[70], [10, 3, 70]]
    exact = map_chunks(tmp_path, line, _decisions, workers=1, chunk_decisions=80)
    assert exact == [[70, 10], [3, 70]]
    assert map_chunks(tmp_path, line, _decisions, workers=1, chunk_decisions=1) == [
        [70],
        [10],
        [3],
        [70],
    ]


def test_map_chunks_in_a_process_pool_equals_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = inline_pools(monkeypatch)
    line = write_shard(tmp_path, index=0, episodes=_mixed(), provenance={})
    inline = map_chunks(tmp_path, line, build_index, workers=1, chunk_decisions=1)
    assert not built
    pooled = map_chunks(tmp_path, line, build_index, workers=2, chunk_decisions=1)
    # One torch thread per worker process: the pool is the parallelism.
    assert built == [(2, torch.set_num_threads, (1,))]
    assert len(pooled) == 4
    for left, right in zip(pooled, inline, strict=True):
        _assert_same_index(left, right)


def test_floor_trace_marks_every_floor_change_and_death() -> None:
    episode = _episode([0] * 3 + [2] * 2 + [0] * 4, died=True)
    trace = floor_trace(aux=episode.aux, reward=episode.reward, done=episode.done)
    assert trace == FloorTrace(changes=((0, 0), (3, 2), (5, 0)), died=True)
    timeout = _episode([4] * 5, died=False)
    assert not floor_trace(
        aux=timeout.aux,
        reward=timeout.reward,
        done=timeout.done,
    ).died


@pytest.mark.parametrize("episode", _mixed(), ids=range(len(_mixed())))
def test_trace_strata_equal_the_frame_strata(episode: Episode) -> None:
    trace = floor_trace(aux=episode.aux, reward=episode.reward, done=episode.done)
    assert trace_strata(trace, decisions=len(episode.actions)).tolist() == _strata(
        episode,
    )


def test_replay_shard_index_equals_the_frame_shard_index(tmp_path: Path) -> None:
    episodes = _mixed()
    frames, replay = tmp_path / "frames", tmp_path / "replay"
    frames.mkdir()
    replay.mkdir()
    old = write_shard(frames, index=0, episodes=episodes, provenance={})
    stored = [
        ReplayEpisode(
            receipt=e.receipt,
            actions=e.actions,
            hashes=e.hashes,
            snapshots=b"",
            floors=floor_trace(aux=e.aux, reward=e.reward, done=e.done),
            summary=e.summary,
        )
        for e in episodes
    ]
    new = write_replay_shard(
        replay,
        index=0,
        episodes=stored,
        stride=256,
        provenance={},
    )
    expected = load_index(frames, old, index_dir=tmp_path / "cache", workers=1)
    chunked = load_index(
        replay,
        new,
        index_dir=tmp_path / "cache",
        workers=1,
        chunk_decisions=60,
    )
    _assert_same_index(chunked, expected)
    assert len(list((tmp_path / "cache").iterdir())) == 2


def _decisions(episodes: list[Episode]) -> list[int]:
    return [len(e.actions) for e in episodes]


def inline_pools(monkeypatch: pytest.MonkeyPatch) -> list[_Built]:
    """Run ``index``'s process pools in this process; return each pool's settings as built.

    A task's function and chunk make the round trip through pickle a worker
    process makes them, so an unpicklable one fails as it would in a pool.
    """
    built: list[_Built] = []

    def pool(
        *,
        max_workers: int,
        initializer: object,
        initargs: tuple[object, ...],
    ) -> _InlinePool:
        built.append((max_workers, initializer, initargs))
        return _InlinePool()

    monkeypatch.setattr(
        "priml.baselines.craftax.world_model.index.futures",
        SimpleNamespace(ProcessPoolExecutor=pool),
    )
    return built


class _InlinePool:
    """Stand in for a process pool: each task pickled, unpickled and run here, in order."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info

    def map[T, R](self, function: Callable[[T], R], items: Iterable[T]) -> list[R]:
        work = cast("Callable[[T], R]", pickle.loads(pickle.dumps(function)))
        return [work(cast("T", pickle.loads(pickle.dumps(item)))) for item in items]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
