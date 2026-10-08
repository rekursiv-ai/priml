"""Check that gather publishes exactly the source's published shards, safely."""

from pathlib import Path

import os
import sys

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    FloorTrace,
    Receipt,
    ReplayEpisode,
    read_manifest,
    read_shard,
    write_replay_shard,
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
from priml.baselines.craftax.world_model.scripts import gather


pytestmark = pytest.mark.cli_rsync


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand in for ssh: drop the host and run the command through a shell.

    Returns:
      log: File holding each remote command, one per line.

    """
    rsh = tmp_path / "rsh"
    rsh.write_text(
        '#!/bin/sh\nshift\nprintf "%s\\n" "$*" >> "$RSH_LOG"\nexec sh -c "$*"\n',
    )
    rsh.chmod(0o755)
    log = tmp_path / "rsh.log"
    monkeypatch.setenv("RSYNC_RSH", str(rsh))
    monkeypatch.setenv("RSH_LOG", str(log))
    return log


@pytest.mark.parametrize(
    ("source_host", "destination_host"),
    [("", ""), ("node:", ""), ("", "node:")],
)
def test_published_shards_of_both_splits_are_published_alone(
    tmp_path: Path,
    remote: Path,
    source_host: str,
    destination_host: str,
) -> None:
    del remote
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    _publish(source, decisions=4)
    _publish(source, split=VALIDATION, decisions=2)
    _publish(source, arm=2, decisions=5)
    train = shard_directory(source, split=TRAIN, arm=1, worker=0)
    (train / "shard-000002.bin.zst.tmp").write_bytes(b"unpublished")
    destination.mkdir()
    shards = gather.gather(
        f"{source_host}{source}",
        f"{destination_host}{destination}",
        workers=[(1, 0)],
    )
    assert shards == [
        "train/arm1/w0/shard-000000",
        "train/arm1/w0/shard-000001",
        "val/arm1/w0/shard-000000",
    ]
    for split in (TRAIN, VALIDATION):
        copied = shard_directory(destination, split=split, arm=1, worker=0)
        original = shard_directory(source, split=split, arm=1, worker=0)
        manifest = (copied / "MANIFEST.jsonl").read_bytes()
        assert manifest == (original / "MANIFEST.jsonl").read_bytes()
        for line in read_manifest(copied):
            read_shard(copied, line)
    assert not list(destination.rglob("*.tmp"))
    assert not (destination / "train" / "arm2").exists()


def test_gathering_again_publishes_only_new_shards(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    assert gather.gather(str(source), str(destination), workers=[(1, 0)])
    assert gather.gather(str(source), str(destination), workers=[(1, 0)]) == []
    _publish(source, decisions=4)
    shards = gather.gather(str(source), str(destination), workers=[(1, 0)])
    assert shards == ["train/arm1/w0/shard-000001"]
    copied = shard_directory(destination, split=TRAIN, arm=1, worker=0)
    assert [line.decisions for line in read_manifest(copied)] == [3, 4]


def test_a_torn_manifest_tail_gathered_once_does_not_refuse_later_gathers(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    original = shard_directory(source, split=TRAIN, arm=1, worker=0)
    # A writer killed while appending its next line leaves an unterminated tail.
    with (original / "MANIFEST.jsonl").open("a") as manifest:
        manifest.write('{"shard": "shard-000001", "epis')
    assert gather.gather(str(source), str(destination), workers=[(1, 0)]) == [
        "train/arm1/w0/shard-000000",
    ]
    # The next append drops the tail.
    _publish(source, decisions=4)
    shards = gather.gather(str(source), str(destination), workers=[(1, 0)])
    assert shards == ["train/arm1/w0/shard-000001"]


def test_an_unpublished_torn_copy_is_replaced(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    copied = shard_directory(destination, split=TRAIN, arm=1, worker=0)
    copied.mkdir(parents=True)
    original = shard_directory(source, split=TRAIN, arm=1, worker=0)
    frames = original / "shard-000000.frames.zst"
    torn = copied / frames.name
    torn.write_bytes(frames.read_bytes()[:-1] + b"\0")
    # Same size and time, so only a content comparison tells the copies apart.
    os.utime(torn, ns=(frames.stat().st_atime_ns, frames.stat().st_mtime_ns))
    gather.gather(str(source), str(destination), workers=[(1, 0)])
    (line,) = read_manifest(copied)
    read_shard(copied, line)


def test_another_capture_of_a_worker_is_refused_before_any_copy(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, arm=2, decisions=5)
    _publish(source, decisions=3)
    _publish(destination, decisions=4)
    before = sorted(destination.rglob("*"))
    with pytest.raises(ValueError, match="not a prefix"):
        gather.gather(str(source), str(destination), workers=[(2, 0), (1, 0)])
    assert sorted(destination.rglob("*")) == before


@pytest.mark.parametrize("destination_host", ["", "node:"])
def test_a_shard_that_mismatches_its_line_is_never_published(
    tmp_path: Path,
    remote: Path,
    destination_host: str,
) -> None:
    del remote
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    original = shard_directory(source, split=TRAIN, arm=1, worker=0)
    (original / "shard-000000.meta.jsonl").write_text("{}\n")
    destination.mkdir()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        gather.gather(str(source), f"{destination_host}{destination}", workers=[(1, 0)])
    assert not list(destination.rglob("MANIFEST.jsonl"))


def test_a_remote_destination_is_synced_before_its_manifest_is_published(
    tmp_path: Path,
    remote: Path,
) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    destination.mkdir()
    gather.gather(str(source), f"node:{destination}", workers=[(1, 0)])
    commands = remote.read_text().splitlines()
    synced = [i for i, command in enumerate(commands) if command.startswith("sync")]
    assert synced == [len(commands) - 1]
    assert (destination / "train" / "arm1" / "w0" / "MANIFEST.jsonl").exists()


def test_a_halted_source_is_refused(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    halt(source, shard="train/arm1/w0/shard-000000", reason="State hash mismatch.")
    with pytest.raises(CaptureHaltedError):
        gather.gather(str(source), str(destination), workers=[(1, 0)])
    assert not list(destination.rglob("MANIFEST.jsonl"))


def test_two_remote_roots_are_refused() -> None:
    with pytest.raises(ValueError, match="At most one"):
        gather.gather("a:/archive", "b:/archive", workers=[(1, 0)])


def test_main_gathers_the_named_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    _publish(source, decisions=3)
    _publish(source, arm=3, worker=2, decisions=4)
    argv = ["gather.py", str(source), str(destination), "arm1/w0", "arm3/w2"]
    monkeypatch.setattr(sys, "argv", argv)
    assert gather.main() == 0
    assert "train/arm3/w2/shard-000000" in capsys.readouterr().out
    assert len(list(destination.rglob("MANIFEST.jsonl"))) == 2


def test_main_rejects_a_malformed_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gather.py", "a", "b", "arm1-w0"])
    with pytest.raises(SystemExit):
        gather.main()


def test_replay_shards_are_gathered_with_every_file_they_hold(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    directory = shard_directory(source, split=TRAIN, arm=1, worker=0)
    directory.mkdir(parents=True)
    trace = FloorTrace(changes=((0, 0),), died=False)
    for index, frames in enumerate((b"", b"framed")):
        episode = ReplayEpisode(
            receipt=Receipt(
                world_seed=index,
                sampling_seed=0,
                initial_state_hash=0,
                arm=1,
                split=0,
            ),
            actions=torch.zeros(3, dtype=torch.uint8),
            hashes=torch.zeros(2, dtype=torch.int64),
            snapshots=b"" if frames else b"snapshots",
            floors=trace,
            summary={"episode": index},
            frames=frames,
        )
        write_replay_shard(
            directory,
            index=index,
            episodes=[episode],
            stride=256,
            provenance={},
        )
    shards = gather.gather(str(source), str(destination), workers=[(1, 0)])
    assert shards == ["train/arm1/w0/shard-000000", "train/arm1/w0/shard-000001"]
    copied = shard_directory(destination, split=TRAIN, arm=1, worker=0)
    assert sorted(p.name for p in copied.iterdir()) == sorted(
        p.name for p in directory.iterdir()
    )
    assert (copied / "shard-000001.frames.zst").exists()
    assert not (copied / "shard-000000.frames.zst").exists()


def _publish(
    root: Path,
    *,
    split: int = TRAIN,
    arm: int = 1,
    worker: int = 0,
    decisions: int,
) -> None:
    """Publish one single-episode shard of ``decisions`` decisions."""
    directory = shard_directory(root, split=split, arm=arm, worker=worker)
    directory.mkdir(parents=True, exist_ok=True)
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = True
    episode = Episode(
        receipt=Receipt(
            world_seed=decisions,
            sampling_seed=0,
            initial_state_hash=0,
            arm=arm,
            split=split,
        ),
        actions=torch.arange(decisions, dtype=torch.uint8),
        hashes=torch.zeros(2, dtype=torch.int64),
        cells=torch.full((decisions, 99, 8), decisions, dtype=torch.uint8),
        aux=torch.zeros(decisions, 51, dtype=torch.int16),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=done,
        summary={"episode": 0},
    )
    index = len(read_manifest(directory))
    write_shard(directory, index=index, episodes=[episode], provenance={})


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
