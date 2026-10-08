"""Check the branch pool CLI: corpora in, a shuffled, repeated pool out."""

from pathlib import Path

import collections
import sys

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    FloorTrace,
    ManifestLine,
    Receipt,
    ReplayEpisode,
    write_corpus,
    write_replay_shard,
)
from priml.baselines.craftax.world_model.capture.branches import (
    read_pool,
)
from priml.baselines.craftax.world_model.scripts import branch_pool
from priml.lib.codec import loads


def _corpus(root: Path) -> Path:
    """Write two shards of arm 2 training episodes that reach Troll and Fire."""
    entries: list[tuple[Path, ManifestLine]] = []
    for index in range(2):
        directory = root / "train" / "arm2" / f"w{index}"
        directory.mkdir(parents=True)
        episodes = [
            ReplayEpisode(
                receipt=Receipt(
                    world_seed=180_000_000 + 10 * index + e,
                    sampling_seed=0,
                    initial_state_hash=0,
                    arm=2,
                    split=0,
                ),
                actions=torch.zeros(4_000, dtype=torch.uint8),
                hashes=torch.zeros(17, dtype=torch.int64),
                snapshots=b"",
                floors=FloorTrace(changes=((0, 0), (1_000, 5), (3_000, 6)), died=True),
                summary={},
            )
            for e in range(3)
        ]
        line = write_replay_shard(
            directory,
            index=0,
            episodes=episodes,
            stride=256,
            provenance={},
        )
        entries.append((directory, line))
    corpus = root / "corpora" / "tiny.json"
    write_corpus(corpus, entries=entries)
    return corpus


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["branch_pool.py", *args])
    return branch_pool.main()


def test_pool_holds_every_point_in_each_copy_shuffled_by_its_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = _corpus(tmp_path)
    common = ["--corpus", str(corpus), "--arm", "2", "--floors", "5", "6"]
    common += ["--spacing", "500", "--copies", "2"]
    assert _run(monkeypatch, *common, "--output", str(tmp_path / "a.jsonl")) == 0
    report = loads(capsys.readouterr().out)
    assert _run(monkeypatch, *common, "--output", str(tmp_path / "b.jsonl")) == 0
    assert (
        _run(monkeypatch, *common, "--seed", "1", "--output", str(tmp_path / "c.jsonl"))
        == 0
    )
    a, b, c = (read_pool(tmp_path / f"{n}.jsonl") for n in "abc")
    # Per episode: Troll entry at 1,000 and times at 1,500-2,500; Fire entry at
    # 3,000 and time at 3,500. Six episodes, then the second copy.
    assert len(a) == 2 * 6 * 6
    assert a == b
    assert a != c
    assert collections.Counter(a[:36]) == collections.Counter(a[36:])
    assert a[:36] != a[36:]
    assert report == {
        "points": 36,
        "copies": 2,
        "by_floor_and_kind": {"5/entry": 6, "5/time": 18, "6/entry": 6, "6/time": 6},
    }


def test_window_limits_how_long_after_an_entry_times_are_taken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(tmp_path)
    output = tmp_path / "pool.jsonl"
    args = ["--corpus", str(corpus), "--arm", "2", "--floors", "5", "--spacing"]
    args += ["500", "--window", "5=1000", "--output", str(output)]
    assert _run(monkeypatch, *args) == 0
    assert sorted({p.decision for p in read_pool(output)}) == [1_000, 1_500]


def test_a_root_contributes_every_shard_its_manifests_list_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(tmp_path)
    output = tmp_path / "pool.jsonl"
    args = ["--arm", "2", "--floors", "6", "--output", str(output)]
    assert _run(monkeypatch, "--root", str(tmp_path), *args) == 0
    from_root = read_pool(output)
    assert (
        _run(monkeypatch, "--root", str(tmp_path), "--corpus", str(corpus), *args) == 0
    )
    assert sorted(read_pool(output), key=repr) == sorted(from_root, key=repr)
    assert len(from_root) == 6


def test_a_relative_root_writes_absolute_parent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "pool.jsonl"
    args = ["--root", ".", "--arm", "2", "--floors", "6", "--output", str(output)]
    assert _run(monkeypatch, *args) == 0
    # A capture job reads the pool from its own working directory.
    assert [p.directory.is_absolute() for p in read_pool(output)] == [True] * 6


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
