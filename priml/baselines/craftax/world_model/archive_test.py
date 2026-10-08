"""Check shard round trips, atomic close, manifests, corruption, and corpora."""

from pathlib import Path
from typing import cast

import dataclasses
import json
import struct

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    FloorTrace,
    Receipt,
    ReplayEpisode,
    read_corpus,
    read_episodes,
    read_manifest,
    read_records,
    read_shard,
    read_snapshots,
    read_summaries,
    record_frame,
    token_frame,
    write_corpus,
    write_replay_shard,
    write_shard,
)
from priml.lib import zstd_compat
from priml.lib.codec import (
    from_plain,
    loads,
)


def _episode(decisions: int, *, seed: int) -> Episode:
    generator = torch.Generator().manual_seed(seed)
    return Episode(
        receipt=Receipt(
            world_seed=100_000_000 + seed,
            sampling_seed=seed,
            initial_state_hash=0xFFFF_FFFF_FFFF_FFFF - seed,
            arm=1,
            split=0,
        ),
        actions=torch.randint(
            0,
            43,
            (decisions,),
            dtype=torch.uint8,
            generator=generator,
        ),
        hashes=torch.arange(decisions // 256 + 2, dtype=torch.int64) - 7,
        cells=torch.randint(
            0,
            37,
            (decisions, 99, 8),
            dtype=torch.uint8,
            generator=generator,
        ),
        aux=torch.randint(
            0,
            261,
            (decisions, 51),
            dtype=torch.int16,
            generator=generator,
        ),
        reward=torch.randint(
            -1,
            5,
            (decisions,),
            dtype=torch.int16,
            generator=generator,
        ),
        done=torch.arange(decisions) == decisions - 1,
        summary={"death": 1, "floors_reached": [1, 1, 0]},
    )


def _assert_same(left: Episode, right: Episode) -> None:
    assert left.receipt == right.receipt
    assert left.summary == right.summary
    for name in ("actions", "hashes", "cells", "aux", "reward", "done"):
        assert torch.equal(
            cast("torch.Tensor", getattr(left, name)),
            cast("torch.Tensor", getattr(right, name)),
        ), name


def test_shard_round_trip_through_manifest(tmp_path: Path) -> None:
    episodes = [_episode(300, seed=1), _episode(5, seed=2)]
    line = write_shard(tmp_path, index=0, episodes=episodes, provenance={"sha": "x"})
    assert line.episodes == 2
    assert line.decisions == 305
    assert read_manifest(tmp_path) == [line]
    for left, right in zip(episodes, read_shard(tmp_path, line), strict=True):
        _assert_same(left, right)


def test_manifest_lists_shards_in_close_order(tmp_path: Path) -> None:
    first = write_shard(
        tmp_path,
        index=0,
        episodes=[_episode(3, seed=1)],
        provenance={},
    )
    second = write_shard(
        tmp_path,
        index=1,
        episodes=[_episode(4, seed=2)],
        provenance={},
    )
    assert read_manifest(tmp_path) == [first, second]


def test_leftover_temporary_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "shard-000000.bin.zst.tmp").write_bytes(b"partial")
    assert read_manifest(tmp_path) == []
    line = write_shard(tmp_path, index=0, episodes=[_episode(3, seed=1)], provenance={})
    assert read_manifest(tmp_path) == [line]
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupted_shard_is_rejected(tmp_path: Path) -> None:
    line = write_shard(tmp_path, index=0, episodes=[_episode(3, seed=1)], provenance={})
    path = tmp_path / f"{line.shard}.frames.zst"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        read_shard(tmp_path, line)


def test_existing_shard_index_is_refused(tmp_path: Path) -> None:
    write_shard(tmp_path, index=0, episodes=[_episode(3, seed=1)], provenance={})
    with pytest.raises(FileExistsError):
        write_shard(tmp_path, index=0, episodes=[_episode(3, seed=2)], provenance={})


def test_single_episodes_are_read_without_decoding_the_shard(tmp_path: Path) -> None:
    episodes = [_episode(300, seed=1), _episode(5, seed=2), _episode(40, seed=3)]
    line = write_shard(tmp_path, index=0, episodes=episodes, provenance={})
    summaries = read_summaries(tmp_path, line)
    assert [s.decisions for s in summaries] == [300, 5, 40]
    assert [s.receipt for s in summaries] == [e.receipt for e in episodes]
    picked = read_episodes(tmp_path, line, summaries=[summaries[2], summaries[0]])
    _assert_same(picked[0], episodes[2])
    _assert_same(picked[1], episodes[0])


def test_damaged_episode_frame_fails_its_checksum(tmp_path: Path) -> None:
    episodes = [_episode(30, seed=1), _episode(30, seed=2)]
    line = write_shard(tmp_path, index=0, episodes=episodes, provenance={})
    summaries = read_summaries(tmp_path, line)
    path = tmp_path / f"{line.shard}.frames.zst"
    data = bytearray(path.read_bytes())
    span = summaries[1].frames
    assert span is not None
    data[span.offset + span.size // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    assert len(read_episodes(tmp_path, line, summaries=[summaries[0]])) == 1
    with pytest.raises(ValueError, match="episode"):
        read_episodes(tmp_path, line, summaries=[summaries[1]])


def test_records_are_read_without_the_frames(tmp_path: Path) -> None:
    episodes = [_episode(300, seed=1), _episode(5, seed=2)]
    line = write_shard(tmp_path, index=0, episodes=episodes, provenance={})
    summaries = read_summaries(tmp_path, line)
    (tmp_path / f"{line.shard}.frames.zst").unlink()
    (record,) = read_records(tmp_path, line, summaries=[summaries[0]])
    assert record.receipt == episodes[0].receipt
    assert torch.equal(record.actions, episodes[0].actions)
    assert torch.equal(record.hashes, episodes[0].hashes)
    path = tmp_path / f"{line.shard}.bin.zst"
    data = bytearray(path.read_bytes())
    data[summaries[1].bin.offset + summaries[1].bin.size // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="CRC-32"):
        read_records(tmp_path, line, summaries=[summaries[1]])


def test_corpus_freezes_manifest_lines(tmp_path: Path) -> None:
    shard_dir = tmp_path / "train" / "b0" / "w0"
    shard_dir.mkdir(parents=True)
    line = write_shard(
        shard_dir,
        index=0,
        episodes=[_episode(3, seed=1)],
        provenance={},
    )
    corpus = tmp_path / "corpora" / "tiny.json"
    write_corpus(corpus, entries=[(shard_dir, line)])
    assert read_corpus(corpus) == [(shard_dir, line)]
    write_shard(shard_dir, index=1, episodes=[_episode(3, seed=2)], provenance={})
    assert read_corpus(corpus) == [(shard_dir, line)]


def _replay_episode(episode: Episode, *, snapshots: bytes) -> ReplayEpisode:
    return ReplayEpisode(
        receipt=episode.receipt,
        actions=episode.actions,
        hashes=episode.hashes,
        snapshots=snapshots,
        floors=FloorTrace(changes=((0, 0), (2, 1)), died=True),
        summary=episode.summary,
    )


def test_replay_shard_round_trip_through_manifest(tmp_path: Path) -> None:
    episodes = [_episode(300, seed=1), _episode(5, seed=2), _episode(40, seed=3)]
    blocks = [
        zstd_compat.compress(b"a" * 64) + zstd_compat.compress(b"b" * 64),
        b"",
        b"xyz",
    ]
    stored = [
        _replay_episode(e, snapshots=s) for e, s in zip(episodes, blocks, strict=True)
    ]
    line = write_replay_shard(
        tmp_path,
        index=3,
        episodes=stored,
        stride=256,
        provenance={"sha": "x"},
    )
    assert line.shard == "shard-000003"
    assert (line.episodes, line.decisions, line.snapshot_stride) == (3, 345, 256)
    assert sorted(line.sha256) == ["bin", "meta", "snap"]
    assert read_manifest(tmp_path) == [line]
    assert not list(tmp_path.glob("*.frames.zst"))
    summaries = read_summaries(tmp_path, line)
    assert [s.floors for s in summaries] == [e.floors for e in stored]
    assert all(s.replayed for s in summaries)
    picked = [summaries[2], summaries[0]]
    assert read_snapshots(tmp_path, line, summaries=picked) == [blocks[2], blocks[0]]
    for record, episode in zip(
        read_records(tmp_path, line, summaries=picked),
        [episodes[2], episodes[0]],
        strict=True,
    ):
        assert record.receipt == episode.receipt
        assert torch.equal(record.actions, episode.actions)
        assert torch.equal(record.hashes, episode.hashes)


def test_replay_shard_records_are_the_frame_shards_bytes(tmp_path: Path) -> None:
    episodes = [_episode(300, seed=1), _episode(5, seed=2)]
    frames, replay = tmp_path / "frames", tmp_path / "replay"
    frames.mkdir()
    replay.mkdir()
    old = write_shard(frames, index=0, episodes=episodes, provenance={})
    new = write_replay_shard(
        replay,
        index=0,
        episodes=[_replay_episode(e, snapshots=b"") for e in episodes],
        stride=256,
        provenance={},
    )
    assert new.sha256["bin"] == old.sha256["bin"]
    assert [s.bin for s in read_summaries(replay, new)] == [
        s.bin for s in read_summaries(frames, old)
    ]


def test_replay_shard_has_no_frames_to_read(tmp_path: Path) -> None:
    line = write_replay_shard(
        tmp_path,
        index=0,
        episodes=[_replay_episode(_episode(3, seed=1), snapshots=b"")],
        stride=256,
        provenance={},
    )
    summaries = read_summaries(tmp_path, line)
    with pytest.raises(ValueError, match="stores no frames"):
        read_episodes(tmp_path, line, summaries=summaries)
    with pytest.raises(ValueError, match="stores no frames"):
        read_shard(tmp_path, line)


def test_damaged_snapshots_fail_their_checksum(tmp_path: Path) -> None:
    blocks = [zstd_compat.compress(b"a" * 64), zstd_compat.compress(b"b" * 64)]
    line = write_replay_shard(
        tmp_path,
        index=0,
        episodes=[
            _replay_episode(_episode(30, seed=s), snapshots=b)
            for s, b in zip((1, 2), blocks, strict=True)
        ],
        stride=256,
        provenance={},
    )
    summaries = read_summaries(tmp_path, line)
    path = tmp_path / f"{line.shard}.snap.zst"
    data = bytearray(path.read_bytes())
    span = summaries[1].snapshots
    assert span is not None
    data[span.offset + span.size // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    assert read_snapshots(tmp_path, line, summaries=[summaries[0]]) == blocks[:1]
    with pytest.raises(ValueError, match="CRC-32"):
        read_snapshots(tmp_path, line, summaries=[summaries[1]])


def test_frame_shard_lines_read_as_stride_zero(tmp_path: Path) -> None:
    line = write_shard(tmp_path, index=0, episodes=[_episode(3, seed=1)], provenance={})
    text = from_plain(
        loads((tmp_path / "MANIFEST.jsonl").read_text()),
        dict[str, object],
    )
    del text["snapshot_stride"]
    (tmp_path / "MANIFEST.jsonl").write_text(json.dumps(text) + "\n")
    assert read_manifest(tmp_path) == [line]
    assert line.snapshot_stride == 0
    (summary,) = read_summaries(tmp_path, line)
    assert summary.snapshots is None
    assert summary.floors is None


def test_origin_and_truncation_round_trip_in_a_version_2_record(
    tmp_path: Path,
) -> None:
    plain = _replay_episode(_episode(30, seed=1), snapshots=b"")
    branch = dataclasses.replace(
        _replay_episode(_episode(40, seed=2), snapshots=b""),
        origin=bytes(range(256)) * 3,
        truncated=True,
    )
    cut = dataclasses.replace(
        _replay_episode(_episode(50, seed=3), snapshots=b""),
        truncated=True,
    )
    line = write_replay_shard(
        tmp_path,
        index=0,
        episodes=[plain, branch, cut],
        stride=256,
        provenance={},
    )
    records = read_records(tmp_path, line, summaries=read_summaries(tmp_path, line))
    assert [(r.origin, r.truncated) for r in records] == [
        (b"", False),
        (branch.origin, True),
        (b"", True),
    ]
    for record, episode in zip(records, (plain, branch, cut), strict=True):
        assert record.receipt == episode.receipt
        assert torch.equal(record.actions, episode.actions)
        assert torch.equal(record.hashes, episode.hashes)
    versions = [
        struct.unpack_from("<4sHH", zstd_compat.decompress(record_frame(e)))
        for e in (plain, branch, cut)
    ]
    assert versions == [(b"CXE1", 1, 0), (b"CXE1", 2, 3), (b"CXE1", 2, 1)]


def test_frame_shard_keeps_a_truncated_episodes_flag(tmp_path: Path) -> None:
    episode = dataclasses.replace(_episode(30, seed=1), truncated=True)
    line = write_shard(tmp_path, index=0, episodes=[episode], provenance={})
    (read,) = read_shard(tmp_path, line)
    assert read.truncated
    assert read.origin == b""


def test_a_replay_shards_frame_stored_episode_is_not_replayed(tmp_path: Path) -> None:
    replayed = _replay_episode(_episode(3, seed=1), snapshots=b"")
    framed = dataclasses.replace(
        _replay_episode(_episode(4, seed=2), snapshots=b""),
        frames=token_frame(_episode(4, seed=2)),
    )
    line = write_replay_shard(
        tmp_path,
        index=0,
        episodes=[replayed, framed],
        stride=256,
        provenance={},
    )
    summaries = read_summaries(tmp_path, line)
    assert [s.replayed for s in summaries] == [True, False]
    with pytest.raises(ValueError, match="stores its frames, not snapshots"):
        read_snapshots(tmp_path, line, summaries=summaries)
    frames = tmp_path / "frames"
    frames.mkdir()
    old = write_shard(frames, index=0, episodes=[_episode(3, seed=1)], provenance={})
    assert not read_summaries(frames, old)[0].replayed


def test_an_empty_shard_is_refused_by_both_writers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one episode"):
        write_shard(tmp_path, index=0, episodes=[], provenance={})
    with pytest.raises(ValueError, match="at least one episode"):
        write_replay_shard(tmp_path, index=0, episodes=[], stride=256, provenance={})
    assert not list(tmp_path.iterdir())


def test_an_unpublished_orphan_does_not_block_its_index(tmp_path: Path) -> None:
    # A writer that crashed after writing its files but before appending its
    # manifest line left a shard no reader sees.
    (tmp_path / "shard-000000.bin.zst").write_bytes(b"orphan")
    line = write_shard(tmp_path, index=0, episodes=[_episode(3, seed=1)], provenance={})
    assert read_manifest(tmp_path) == [line]
    (episode,) = read_shard(tmp_path, line)
    _assert_same(episode, _episode(3, seed=1))


def test_a_torn_manifest_line_is_not_published_and_is_repaired(tmp_path: Path) -> None:
    first = write_shard(
        tmp_path,
        index=0,
        episodes=[_episode(3, seed=1)],
        provenance={},
    )
    manifest = tmp_path / "MANIFEST.jsonl"
    manifest.write_text(manifest.read_text() + '{"shard": "shard-0000')
    assert read_manifest(tmp_path) == [first]
    second = write_shard(
        tmp_path,
        index=1,
        episodes=[_episode(4, seed=2)],
        provenance={},
    )
    assert read_manifest(tmp_path) == [first, second]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
