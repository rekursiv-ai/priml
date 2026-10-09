"""Check that a corpus is frozen, indexed, counted, and reported with its costs."""

from pathlib import Path

import sys

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    Receipt,
    read_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.capture.control import (
    CaptureHaltedError,
    halt,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.baselines.craftax.world_model.index import FLOOR_AUX
from priml.baselines.craftax.world_model.scripts import coverage_report
from priml.baselines.craftax.world_model.snapshots_test import (
    replay_twin,
)
from priml.lib.codec import from_plain, loads


def _episode(floors: list[int], *, split: int, world_seed: int = 1) -> Episode:
    decisions = len(floors)
    aux = torch.zeros(decisions, 51, dtype=torch.int16)
    aux[:, 31] = 1
    aux[:, FLOOR_AUX] = torch.tensor(floors, dtype=torch.int16)
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    return Episode(
        receipt=Receipt(
            world_seed=world_seed,
            sampling_seed=2,
            initial_state_hash=3,
            arm=3,
            split=split,
        ),
        actions=torch.arange(decisions, dtype=torch.uint8),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=torch.ones(decisions, 99, 8, dtype=torch.uint8),
        aux=aux,
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={"achievements": [0, 5]},
    )


def _archive(root: Path) -> None:
    """Publish one training shard for each of two workers and one validation shard."""
    for worker in (0, 1):
        directory = shard_directory(root, split=TRAIN, arm=3, worker=worker)
        directory.mkdir(parents=True)
        episodes = [
            _episode([0, 0, 1], split=TRAIN, world_seed=worker),
            _episode([0, 1, 2], split=TRAIN, world_seed=10 + worker),
        ]
        write_shard(directory, index=0, episodes=episodes, provenance={})
    directory = shard_directory(root, split=VALIDATION, arm=3, worker=0)
    directory.mkdir(parents=True)
    write_shard(
        directory,
        index=0,
        episodes=[_episode([0, 1], split=VALIDATION)],
        provenance={},
    )


def test_freeze_writes_every_published_shard_and_reports_reach(tmp_path: Path) -> None:
    _archive(tmp_path)
    result = coverage_report.build_report(
        tmp_path,
        corpus="small",
        freeze=True,
        cache_dir=tmp_path / "index",
        workers=1,
    )
    shards = read_corpus(tmp_path / "corpora" / "small.json")
    assert [str(d.relative_to(tmp_path)) for d, _ in shards] == [
        "train/arm3/w0",
        "train/arm3/w1",
        "val/arm3/w0",
    ]
    assert result["decisions"] == 14
    assert result["episodes"] == 5
    reach = from_plain(result["reach"], dict[str, object])
    assert reach["train"] == [4, 4, 2] + [0] * 6
    assert reach["val"] == [1, 1] + [0] * 7
    assert len(list((tmp_path / "index").glob("index-v1-*.pt"))) == 3
    assert len(list((tmp_path / "index").glob("coverage-v1-*.pt"))) == 3


def test_bytes_per_decision_are_the_shard_files_over_their_decisions(
    tmp_path: Path,
) -> None:
    _archive(tmp_path)
    result = coverage_report.build_report(
        tmp_path,
        corpus="small",
        freeze=True,
        cache_dir=tmp_path / "index",
        workers=1,
    )
    sizes = {
        suffix: sum(p.stat().st_size for p in tmp_path.rglob(f"shard-*.{suffix}"))
        for suffix in ("bin.zst", "frames.zst", "meta.jsonl")
    }
    rates = from_plain(result["bytes_per_decision"], dict[str, object])
    assert from_plain(rates["bin"], float) == sizes["bin.zst"] / 14
    assert from_plain(rates["frames"], float) == sizes["frames.zst"] / 14
    assert from_plain(rates["meta"], float) == sizes["meta.jsonl"] / 14


def test_every_shard_is_timed_for_index_and_coverage(tmp_path: Path) -> None:
    _archive(tmp_path)
    result = coverage_report.build_report(
        tmp_path,
        corpus="small",
        freeze=True,
        cache_dir=tmp_path / "index",
        workers=1,
    )
    seconds = from_plain(result["seconds"], dict[str, object])
    assert len(from_plain(seconds["index"], list[float])) == 3
    assert len(from_plain(seconds["coverage"], list[float])) == 3
    assert from_plain(seconds["report"], float) >= 0


def test_an_existing_corpus_is_read_not_refrozen(tmp_path: Path) -> None:
    _archive(tmp_path)
    coverage_report.build_report(
        tmp_path,
        corpus="small",
        freeze=True,
        cache_dir=tmp_path / "index",
        workers=1,
    )
    extra = shard_directory(tmp_path, split=TRAIN, arm=3, worker=2)
    extra.mkdir(parents=True)
    write_shard(extra, index=0, episodes=[_episode([0], split=TRAIN)], provenance={})
    with pytest.raises(FileExistsError):
        coverage_report.build_report(
            tmp_path,
            corpus="small",
            freeze=True,
            cache_dir=tmp_path / "index",
            workers=1,
        )
    result = coverage_report.build_report(
        tmp_path,
        corpus="small",
        freeze=False,
        cache_dir=tmp_path / "index",
        workers=1,
    )
    assert result["decisions"] == 14


def test_a_corpus_frozen_from_a_relative_root_reads_from_anywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _archive(tmp_path / "archive")
    monkeypatch.chdir(tmp_path)
    coverage_report.build_report(
        Path("archive"),
        corpus="small",
        freeze=True,
        cache_dir=Path("archive/index"),
        workers=1,
    )
    shards = read_corpus(tmp_path / "archive" / "corpora" / "small.json")
    assert [directory.is_absolute() for directory, _ in shards] == [True] * 3


def test_freezing_a_halted_archive_is_refused(tmp_path: Path) -> None:
    _archive(tmp_path)
    halt(tmp_path, shard="train/arm3/w0/shard-000000", reason="State hash mismatch.")
    with pytest.raises(CaptureHaltedError):
        coverage_report.build_report(
            tmp_path,
            corpus="small",
            freeze=True,
            cache_dir=tmp_path / "index",
            workers=1,
        )
    assert not (tmp_path / "corpora").exists()


def test_a_corpus_of_replay_shards_reports_as_its_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, replay = tmp_path / "frames", tmp_path / "replay"
    _archive(frames)
    results = [
        coverage_report.build_report(
            frames,
            corpus="small",
            freeze=True,
            cache_dir=frames / "index",
            workers=1,
        ),
    ]
    replay_twin(frames, replay, monkeypatch)
    results.append(
        coverage_report.build_report(
            replay,
            corpus="small",
            freeze=False,
            cache_dir=replay / "index",
            workers=1,
        ),
    )
    rates = from_plain(results[1]["bytes_per_decision"], dict[str, object])
    for result in results:
        for key in ("corpus", "seconds", "bytes_per_decision"):
            result.pop(key)
    assert results[0] == results[1]
    snap = sum(p.stat().st_size for p in replay.rglob("shard-*.snap.zst"))
    assert sorted(rates) == ["bin", "meta", "snap"]
    assert from_plain(rates["snap"], float) == snap / 14


def test_main_writes_the_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _archive(tmp_path)
    output = tmp_path / "reports" / "small.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage_report.py",
            str(tmp_path),
            "--corpus=small",
            "--freeze",
            f"--output={output}",
            "--workers=1",
        ],
    )
    assert coverage_report.main() == 0
    written = from_plain(loads(output.read_text()), dict[str, object])
    assert written["decisions"] == 14
    assert (tmp_path / "index").is_dir()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
