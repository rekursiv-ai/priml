"""Check replay shards: snapshots, storing episodes, and the replay-backed frames.

Episodes are recorded and replayed on the port's game (``replay.py``), so
nothing here needs a C build.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import dataclasses
import functools
import itertools

import pytest
import torch

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
    ReplayEpisode,
    read_corpus,
    read_manifest,
    read_records,
    read_shard,
    read_snapshots,
    read_summaries,
    record_frame,
    write_corpus,
    write_replay_shard,
    write_shard,
)
from priml.baselines.craftax.world_model.data import (
    EpisodeCache,
    EvalSpans,
    ReplayStream,
)
from priml.baselines.craftax.world_model.index import (
    floor_trace,
    load_index,
)
from priml.baselines.craftax.world_model.snapshots import (
    ReplayCache,
    check_episode,
    decode_snapshots,
    replay_episodes,
    snapshot_episode,
    stored_episode,
    verify_snapshots,
)
from priml.lib import zstd_compat


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from priml.baselines.craftax.world_model.batch import Segment


_STRIDE = 256

_NAMES = ("train/arm0/w0", "train/arm0/w1", "val/arm0/w0")


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return frame shards of recorded episodes: two training, one validation."""
    root = tmp_path_factory.mktemp("archive")
    # Random play rarely passes 375 decisions; seeds 311 and 364 (375 and 346)
    # give the first shard two episodes past the 256-decision snapshot.
    episodes = [
        replay.record(world_seed=seed, sampling_seed=seed, max_decisions=1_000)
        for seed in (311, 2, 364, 154, 209, 143, 160, 349, 331)
    ]
    entries: list[tuple[Path, ManifestLine]] = []
    for name, chosen, split in zip(
        _NAMES,
        (episodes[:4], episodes[4:7], episodes[7:]),
        (0, 0, 1),
        strict=True,
    ):
        directory = root / name
        directory.mkdir(parents=True)
        shard = [
            dataclasses.replace(
                e,
                receipt=dataclasses.replace(e.receipt, split=split),
                summary={"episode": n},
            )
            for n, e in enumerate(chosen)
        ]
        line = write_shard(directory, index=2, episodes=shard, provenance={"p": "q"})
        entries.append((directory, line))
    write_corpus(root / "corpora" / "tiny.json", entries=entries)
    return root


@pytest.fixture(scope="module")
def converted(archive: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the archive's shards written as replay shards, with its corpus."""
    root = tmp_path_factory.mktemp("converted")
    entries: list[tuple[Path, ManifestLine]] = []
    for directory, line in read_corpus(archive / "corpora" / "tiny.json"):
        output = root / directory.relative_to(archive)
        output.mkdir(parents=True)
        written = write_replay_shard(
            output,
            index=int(line.shard.removeprefix("shard-")),
            episodes=[
                stored_episode(e, stride=_STRIDE) for e in read_shard(directory, line)
            ],
            stride=_STRIDE,
            provenance=line.provenance,
        )
        entries.append((output, written))
    write_corpus(root / "corpora" / "tiny.json", entries=entries)
    return root


def replay_twin(root: Path, output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write an archive root's frame shards as replay shards under ``output``.

    For readers' tests over synthetic frames, which no game replays: every
    shard a worker manifest or a corpus of ``root`` names is written at the
    same path under ``output``, each episode by its record and floor trace
    alone, so every frame a reader gets must come from replay; the corpora are
    written naming the replay shards. ``replay.replay`` is replaced, in this
    process only, by a stand-in returning the frames captured beside the same
    record, as the game's replay regenerates them.

    Args:
      root: Archive root of frame shards, with any corpora under ``corpora``.
      output: Root of the replay shards.
      monkeypatch: The test's, which restores ``replay.replay`` afterwards.

    """
    manifests = sorted(root.glob("*/*/*/MANIFEST.jsonl"))
    entries = [(m.parent, line) for m in manifests for line in read_manifest(m.parent)]
    corpora = {path: read_corpus(path) for path in root.glob("corpora/*.json")}
    entries += [e for listed in corpora.values() for e in listed if e not in entries]
    twins: dict[tuple[Path, str], ManifestLine] = {}
    recorded: dict[bytes, Episode] = {}
    for directory, line in entries:
        episodes = read_shard(directory, line)
        for episode in episodes:
            # Replay is a function of the record: two episodes with one record
            # and different frames cannot both be served.
            same = recorded.setdefault(record_frame(episode), episode)
            assert all(
                torch.equal(
                    cast("torch.Tensor", getattr(same, name)),
                    cast("torch.Tensor", getattr(episode, name)),
                )
                for name in ("cells", "aux", "reward", "done")
            ), f"Episodes of receipt {episode.receipt} share a record, not frames."
        longest = max(len(e.actions) for e in episodes)
        target = output / directory.relative_to(root)
        target.mkdir(parents=True, exist_ok=True)
        twins[directory, line.shard] = write_replay_shard(
            target,
            index=int(line.shard.removeprefix("shard-")),
            episodes=[
                ReplayEpisode(
                    receipt=e.receipt,
                    actions=e.actions,
                    hashes=e.hashes,
                    snapshots=b"",
                    floors=floor_trace(aux=e.aux, reward=e.reward, done=e.done),
                    summary=e.summary,
                )
                for e in episodes
            ],
            stride=-(-longest // 256) * 256,
            provenance=line.provenance,
        )
    for path, listed in corpora.items():
        write_corpus(
            output / "corpora" / path.name,
            entries=[
                (output / d.relative_to(root), twins[d, line.shard])
                for d, line in listed
            ],
        )
    monkeypatch.setattr(replay, "replay", functools.partial(_recorded, recorded))


def test_recorded_episodes_span_zero_and_one_snapshots(archive: Path) -> None:
    lengths = [
        s.decisions
        for name in _NAMES
        for s in read_summaries(archive / name, read_manifest(archive / name)[0])
    ]
    assert max(lengths) > 300
    assert min(lengths) < _STRIDE


def test_stored_snapshots_decode_to_the_replayed_states(archive: Path) -> None:
    for episode in _episodes(archive, "train/arm0/w0"):
        stored = snapshot_episode(episode, stride=_STRIDE)
        base, *expected = replay.snapshots(episode, stride=_STRIDE)
        assert decode_snapshots(stored, base=base, stride=_STRIDE) == expected
        assert len(expected) == (len(episode.actions) - 1) // _STRIDE
        verify_snapshots(episode, stored, stride=_STRIDE)


def test_check_and_verify_reject_damage(archive: Path) -> None:
    episodes = _episodes(archive, "train/arm0/w0")
    long = max(episodes, key=lambda e: len(e.actions))
    stored = snapshot_episode(long, stride=_STRIDE)
    assert check_episode(long, stored, stride=_STRIDE) == len(long.actions)
    cells = long.cells.clone()
    cells[-1, 0, 0] ^= 1
    with pytest.raises(ValueError, match="frames differ"):
        check_episode(dataclasses.replace(long, cells=cells), stored, stride=_STRIDE)
    with pytest.raises(ValueError, match="cover"):
        check_episode(long, b"", stride=_STRIDE)
    sparser = snapshot_episode(long, stride=2 * _STRIDE)
    with pytest.raises(ValueError, match="differ"):
        verify_snapshots(long, sparser, stride=_STRIDE)
    actions = long.actions.clone()
    actions[:_STRIDE] = (actions[:_STRIDE] + 1) % 43
    with pytest.raises(ValueError, match="hash"):
        verify_snapshots(
            dataclasses.replace(long, actions=actions),
            stored,
            stride=_STRIDE,
        )


def test_several_snapshots_decode_in_decision_order() -> None:
    generator = torch.Generator().manual_seed(0)
    base, *states = (
        torch.randint(0, 256, (1_000,), dtype=torch.uint8, generator=generator)
        for _ in range(4)
    )
    stored = b"".join(
        zstd_compat.compress(state.bitwise_xor(base).numpy().tobytes())
        for state in states
    )
    decoded = decode_snapshots(
        stored,
        base=replay.Snapshot(decision=0, state=base.numpy().tobytes()),
        stride=512,
    )
    assert [s.decision for s in decoded] == [512, 1_024, 1_536]
    assert [s.state for s in decoded] == [s.numpy().tobytes() for s in states]
    assert decode_snapshots(b"", base=decoded[0], stride=512) == []


def test_a_replay_shard_keeps_its_episodes_records_summaries_and_index(
    archive: Path,
    converted: Path,
) -> None:
    for name in _NAMES:
        (old,) = read_manifest(archive / name)
        (new,) = read_manifest(converted / name)
        assert (new.shard, new.episodes, new.decisions) == (
            old.shard,
            old.episodes,
            old.decisions,
        )
        assert new.snapshot_stride == _STRIDE
        assert new.sha256["bin"] == old.sha256["bin"]
        old_summaries = read_summaries(archive / name, old)
        new_summaries = read_summaries(converted / name, new)
        assert [s.summary for s in new_summaries] == [s.summary for s in old_summaries]
        assert [s.bin for s in new_summaries] == [s.bin for s in old_summaries]
        indexes = [
            load_index(root / name, line, index_dir=root / "index", workers=1)
            for root, line in ((archive, old), (converted, new))
        ]
        for field in dataclasses.fields(indexes[0]):
            assert torch.equal(
                cast("torch.Tensor", getattr(indexes[0], field.name)),
                cast("torch.Tensor", getattr(indexes[1], field.name)),
            )


def test_replay_episodes_read_either_format_with_its_frames(
    archive: Path,
    converted: Path,
) -> None:
    for name in _NAMES:
        episodes: list[list[Episode]] = []
        for root in (archive, converted):
            (line,) = read_manifest(root / name)
            summaries = read_summaries(root / name, line)
            episodes.append(
                replay_episodes(root / name, line, summaries=summaries[::-1]),
            )
        for frames, replayed in zip(episodes[0], episodes[1], strict=True):
            assert replayed.receipt == frames.receipt
            assert replayed.summary == frames.summary
            for field in ("actions", "hashes", "cells", "aux", "reward", "done"):
                assert torch.equal(
                    cast("torch.Tensor", getattr(replayed, field)),
                    cast("torch.Tensor", getattr(frames, field)),
                )


def test_episodes_that_do_not_replay_are_stored_with_their_frames(
    archive: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    episodes = _episodes(archive, "train/arm0/w0")
    long = max(range(len(episodes)), key=lambda e: len(episodes[e].actions))
    aux = episodes[long].aux.clone()
    aux[-1, 0] += 1
    other = (long + 1) % len(episodes)
    actions = episodes[other].actions.clone()
    actions[:5] = (actions[:5] + 1) % 43
    damaged = list(episodes)
    damaged[long] = dataclasses.replace(episodes[long], aux=aux)
    damaged[other] = dataclasses.replace(episodes[other], actions=actions)
    stored = [stored_episode(e, stride=_STRIDE) for e in damaged]
    assert "aux frames differ" in caplog.text
    line = write_replay_shard(
        tmp_path,
        index=0,
        episodes=stored,
        stride=_STRIDE,
        provenance={},
    )
    assert sorted(line.sha256) == ["bin", "frames", "meta", "snap"]
    summaries = read_summaries(tmp_path, line)
    framed = [not s.replayed for s in summaries]
    assert framed == [e in {other, long} for e in range(len(episodes))]
    cache = ReplayCache([(tmp_path, line)], capacity=10_000)
    read = replay_episodes(tmp_path, line, summaries=summaries)
    for e, expected in enumerate(damaged):
        decisions = len(expected.actions)
        segment, _ = cache.segment(0, e, start=1, stop=decisions)
        assert torch.equal(segment.aux, expected.aux[1:])
        assert torch.equal(read[e].aux, expected.aux)
        assert torch.equal(read[e].actions, expected.actions)


@pytest.mark.parametrize("block_decisions", [128, 512])
def test_replay_cache_segments_equal_the_frames(
    archive: Path,
    converted: Path,
    block_decisions: int,
) -> None:
    frames = EpisodeCache(_entries(archive, split=0), capacity=100_000)
    replayed = ReplayCache(
        _entries(converted, split=0),
        capacity=100_000,
        block_decisions=block_decisions,
    )
    for shard in range(2):
        assert replayed.episodes(shard) == frames.episodes(shard)
        for e in range(frames.episodes(shard)):
            decisions = frames.decisions(shard, e)
            assert replayed.decisions(shard, e) == decisions
            cuts = {0, 1, 127, 128, 255, 256, 257, 511, 600, decisions - 1}
            for start in sorted(c for c in cuts if 0 <= c < decisions):
                for stop in {start, min(start + 300, decisions), decisions}:
                    _assert_same_segment(
                        replayed.segment(shard, e, start=start, stop=stop),
                        frames.segment(shard, e, start=start, stop=stop),
                    )


def test_replay_cache_holds_recent_blocks_within_capacity(converted: Path) -> None:
    entries = _entries(converted, split=0)
    cache = ReplayCache(entries, capacity=300, block_decisions=128)
    long = max(range(4), key=lambda e: cache.decisions(0, e))
    # Past 300, the segment below spans blocks 1 and 2 and they outgrow the
    # capacity once block 0 joins them.
    assert cache.decisions(0, long) > 300
    cache.segment(0, long, start=200, stop=300)
    assert cache.resident == [(0, long, 1), (0, long, 2)]
    cache.segment(0, long, start=130, stop=140)
    assert cache.resident == [(0, long, 2), (0, long, 1)]
    cache.segment(0, long, start=10, stop=20)
    assert cache.resident == [(0, long, 1), (0, long, 0)]


def test_replay_cache_serves_replay_shards_only(archive: Path) -> None:
    with pytest.raises(ValueError, match="replay shards"):
        ReplayCache(_entries(archive, split=0), capacity=100)


def test_replay_cache_refuses_empty_blocks(converted: Path) -> None:
    with pytest.raises(ValueError, match="block_decisions=0"):
        ReplayCache(_entries(converted, split=0), capacity=100, block_decisions=0)


def test_replay_cache_checks_the_snapshot_it_restores(
    converted: Path,
    tmp_path: Path,
) -> None:
    directory, line = _entries(converted, split=0)[0]
    summaries = read_summaries(directory, line)
    records = read_records(directory, line, summaries=summaries)
    stored = read_snapshots(directory, line, summaries=summaries)
    long = max(range(len(records)), key=lambda e: len(records[e].actions))
    other = max(
        (e for e in range(len(records)) if e != long),
        key=lambda e: len(records[e].actions),
    )
    assert stored[other], "Another episode with a snapshot swaps in."
    swapped = [s if e != long else stored[other] for e, s in enumerate(stored)]
    output = tmp_path / "w0"
    output.mkdir()
    _write_like(directory, line, output, snapshots=swapped)
    cache = ReplayCache(
        [(output, read_manifest(output)[0])],
        capacity=10_000,
        block_decisions=128,
    )
    cache.segment(0, long, start=0, stop=10)
    with pytest.raises(ValueError, match="hash"):
        cache.segment(0, long, start=_STRIDE + 1, stop=_STRIDE + 10)


def test_replay_stream_serves_the_frame_streams_batches(
    archive: Path,
    converted: Path,
) -> None:
    streams: list[ReplayStream] = []
    for root in (archive, converted):
        config = ReplayStream.Config(
            working_dir=root,
            corpus="corpora/tiny.json",
            t_g=700,
            device="cpu",
        )
        config.validation = EvalSpans.Config(spans=8, span_decisions=64)
        streams.append(config.make())
    frame_stream, replay_stream = streams
    assert isinstance(frame_stream.train_cache, EpisodeCache)
    assert isinstance(replay_stream.train_cache, ReplayCache)
    assert isinstance(replay_stream.eval_cache, ReplayCache)
    pairs = [
        *zip(
            itertools.islice(frame_stream.train_dataloader(), 6),
            itertools.islice(replay_stream.train_dataloader(), 6),
            strict=True,
        ),
        *zip(
            frame_stream.eval_dataloader(),
            replay_stream.eval_dataloader(),
            strict=True,
        ),
    ]
    assert len(pairs) > 6
    for left, right in pairs:
        assert left.keys() == right.keys()
        for key in left:
            _assert_same_value(left[key], right[key])


def test_a_corpus_mixing_formats_is_refused(
    archive: Path,
    converted: Path,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpora" / "mixed.json"
    write_corpus(
        corpus,
        entries=[
            *_entries(archive, split=0)[:1],
            *_entries(converted, split=0)[1:],
            *_entries(converted, split=1),
        ],
    )
    config = ReplayStream.Config(
        working_dir=tmp_path,
        corpus="corpora/mixed.json",
        device="cpu",
    )
    with pytest.raises(ValueError, match="mixes"):
        config.make()


def test_a_stored_episode_keeps_its_checked_snapshots(
    archive: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    episodes = _episodes(archive, "train/arm0/w1")
    for episode in episodes:
        stored = stored_episode(episode, stride=_STRIDE)
        assert not stored.frames
        assert stored.snapshots == snapshot_episode(episode, stride=_STRIDE)
    reward = episodes[0].reward.clone()
    reward[0] += 1
    damaged = stored_episode(
        dataclasses.replace(episodes[0], reward=reward),
        stride=_STRIDE,
    )
    assert "reward frames differ" in caplog.text
    assert damaged.frames
    assert not damaged.snapshots


def test_a_stride_off_the_hash_grid_is_refused_not_stored_as_frames(
    archive: Path,
) -> None:
    (episode, *_) = _episodes(archive, "train/arm0/w1")
    with pytest.raises(ValueError, match="multiple of 256"):
        stored_episode(episode, stride=300)


def test_branches_and_truncated_episodes_store_replay_and_serve(
    archive: Path,
    tmp_path: Path,
) -> None:
    parent = max(
        [*_episodes(archive, "train/arm0/w0"), *_episodes(archive, "train/arm0/w1")],
        key=lambda e: len(e.actions),
    )
    assert len(parent.actions) > _STRIDE
    rest = slice(_STRIDE, None)
    branch = dataclasses.replace(
        parent,
        receipt=dataclasses.replace(
            parent.receipt,
            initial_state_hash=int(parent.hashes[1]) % (1 << 64),
        ),
        actions=parent.actions[rest],
        hashes=parent.hashes[1:],
        cells=parent.cells[rest],
        aux=parent.aux[rest],
        reward=parent.reward[rest],
        done=parent.done[rest],
        summary={"episode": 0},
        origin=replay.origin(parent, decision=_STRIDE),
    )
    head = slice(0, _STRIDE)
    truncated = dataclasses.replace(
        parent,
        actions=parent.actions[head],
        hashes=parent.hashes[:2],
        cells=parent.cells[head],
        aux=parent.aux[head],
        reward=parent.reward[head],
        done=parent.done[head],
        summary={"episode": 1},
        truncated=True,
    )
    stored = [stored_episode(e, stride=_STRIDE) for e in (branch, truncated)]
    assert not any(s.frames for s in stored)
    line = write_replay_shard(
        tmp_path,
        index=0,
        episodes=stored,
        stride=_STRIDE,
        provenance={},
    )
    summaries = read_summaries(tmp_path, line)
    read = replay_episodes(tmp_path, line, summaries=summaries)
    cache = ReplayCache([(tmp_path, line)], capacity=100_000, block_decisions=128)
    for e, episode in enumerate((branch, truncated)):
        assert (read[e].receipt, read[e].origin, read[e].truncated) == (
            episode.receipt,
            episode.origin,
            episode.truncated,
        )
        for name in ("actions", "hashes", "cells", "aux", "reward", "done"):
            assert torch.equal(
                cast("torch.Tensor", getattr(read[e], name)),
                cast("torch.Tensor", getattr(episode, name)),
            ), name
        segment, _ = cache.segment(0, e, start=0, stop=len(episode.actions))
        # A branch begins mid-episode: its first frame follows no start token.
        assert segment.starts_episode == episode.truncated
        for name in ("cells", "aux", "reward", "done"):
            assert torch.equal(
                cast("torch.Tensor", getattr(segment, name)),
                cast("torch.Tensor", getattr(episode, name)),
            ), name


def _recorded(recorded: Mapping[bytes, Episode], episode: Episode) -> Episode:
    """Stand in for ``replay.replay``: the frames captured with the same record."""
    captured = recorded[record_frame(episode)]
    return dataclasses.replace(
        episode,
        cells=captured.cells,
        aux=captured.aux,
        reward=captured.reward,
        done=captured.done,
    )


def _episodes(root: Path, name: str) -> list[Episode]:
    directory = root / name
    (line,) = read_manifest(directory)
    cache = EpisodeCache([(directory, line)], capacity=100_000)
    return [cache.get(0, e).episode for e in range(line.episodes)]


def _entries(root: Path, *, split: int) -> list[tuple[Path, ManifestLine]]:
    names = _NAMES[:2] if split == 0 else _NAMES[2:]
    return [(root / n, read_manifest(root / n)[0]) for n in names]


def _write_like(
    directory: Path,
    line: ManifestLine,
    output: Path,
    *,
    snapshots: list[bytes],
) -> None:
    """Write a replay shard with ``line``'s episodes and the given snapshots."""
    summaries = read_summaries(directory, line)
    records = read_records(directory, line, summaries=summaries)
    stored: list[ReplayEpisode] = []
    for record, summary, snapshot in zip(records, summaries, snapshots, strict=True):
        assert summary.floors is not None
        stored.append(
            ReplayEpisode(
                receipt=record.receipt,
                actions=record.actions,
                hashes=record.hashes,
                snapshots=snapshot,
                floors=summary.floors,
                summary=summary.summary,
            ),
        )
    write_replay_shard(
        output,
        index=0,
        episodes=stored,
        stride=line.snapshot_stride,
        provenance={},
    )


def _assert_same_segment(
    left: tuple[Segment, torch.Tensor],
    right: tuple[Segment, torch.Tensor],
) -> None:
    (segment, strata), (expected, expected_strata) = left, right
    assert segment.starts_episode == expected.starts_episode
    for name in ("cells", "aux", "actions", "reward", "done"):
        a = cast("torch.Tensor", getattr(segment, name))
        b = cast("torch.Tensor", getattr(expected, name))
        assert a.dtype == b.dtype, name
        assert torch.equal(a, b), name
    assert strata.dtype == expected_strata.dtype
    assert torch.equal(strata, expected_strata)


def _assert_same_value(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
        return
    assert dataclasses.is_dataclass(left)
    for field in dataclasses.fields(left):
        _assert_same_value(
            cast("object", getattr(left, field.name)),
            cast("object", getattr(right, field.name)),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
