"""Tests for the capture control files: the archive halt and the launch markers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from priml.baselines.craftax.world_model.capture.control import (
    CaptureHaltedError,
    check_halt,
    completed,
    halt,
    mark_complete,
    mark_started,
    started,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_the_first_halt_stops_every_worker_and_keeps_its_reason(tmp_path: Path) -> None:
    check_halt(tmp_path)
    halt(tmp_path, shard="train/arm0/w0/shard-000000", reason="first")
    halt(tmp_path, shard="train/arm1/w0/shard-000003", reason="second")
    with pytest.raises(CaptureHaltedError, match="first"):
        check_halt(tmp_path)


def test_markers_count_the_workers_of_their_launch_only(tmp_path: Path) -> None:
    mark_started(tmp_path, launch="a", arm=0, worker=1)
    mark_started(tmp_path, launch="a", arm=2, worker=0)
    mark_started(tmp_path, launch="b", arm=0, worker=1)
    mark_complete(tmp_path, launch="a", arm=0, worker=1, decisions=5)
    mark_complete(tmp_path, launch="a", arm=0, worker=1, decisions=7)
    assert started(tmp_path, launch="a") == 2
    assert completed(tmp_path, launch="a") == 1
    assert completed(tmp_path, launch="b") == 0
    assert '"decisions": 7' in (tmp_path / "complete/a/arm0-w1.json").read_text()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
