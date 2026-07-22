"""Tests for priml.train.progress: the progress.json writer."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from priml.train.progress import write_progress


def test_write_progress_lands_step_total_metrics(tmp_path: Path) -> None:
    path = write_progress(
        5,
        100,
        working_dir=tmp_path,
        metrics={"loss": 0.25},
    )
    assert path == tmp_path / "progress.json"
    data = json.loads(path.read_text())
    assert data["step"] == 5
    assert data["total"] == 100
    assert data["metrics"] == {"loss": 0.25}
    assert data["updated_at"].endswith("Z")


def test_write_progress_rejects_empty_working_dir() -> None:
    with pytest.raises(ValueError, match="working_dir must not be empty"):
        write_progress(1, 2, working_dir="")


def test_write_progress_overwrites_atomically(tmp_path: Path) -> None:
    """Successive writes replace the file; no tmp residue is left behind."""
    _ = write_progress(1, 10, working_dir=tmp_path)
    path = write_progress(2, 10, working_dir=tmp_path)
    assert json.loads(path.read_text())["step"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["progress.json"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
