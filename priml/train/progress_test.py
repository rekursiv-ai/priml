"""Tests for priml.train.progress: the progress.json writer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from priml.lib.custom_json import parse
from priml.train import progress
from priml.train.progress import write_progress


if TYPE_CHECKING:
    from pathlib import Path


def test_write_progress_lands_step_total_metrics(tmp_path: Path) -> None:
    path = write_progress(
        5,
        100,
        working_dir=tmp_path,
        metrics={"loss": 0.25},
    )
    assert path == tmp_path / "progress.json"
    data = parse(path.read_text(), dict[str, object])
    assert data["step"] == 5
    assert data["total"] == 100
    assert data["metrics"] == {"loss": 0.25}
    updated_at = data["updated_at"]
    assert isinstance(updated_at, str)
    assert updated_at.endswith("Z")


def test_write_progress_rejects_empty_working_dir() -> None:
    with pytest.raises(ValueError, match=r"^working_dir must not be empty$"):
        write_progress(1, 2, working_dir="")


def test_write_progress_creates_nested_working_dir(tmp_path: Path) -> None:
    working_dir = tmp_path / "nested" / "job"

    path = write_progress(3, 7, working_dir=working_dir)

    assert path == working_dir / "progress.json"
    assert path.is_file()


def test_write_progress_uses_utc_iso_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)
    monkeypatch.setattr(progress, "datetime", clock)

    path = write_progress(1, 2, working_dir=tmp_path)

    data = parse(path.read_text(), dict[str, object])
    assert data["updated_at"] == "2025-02-03T04:05:06Z"
    clock.now.assert_called_once_with(tz=UTC)


def test_write_progress_overwrites_atomically(tmp_path: Path) -> None:
    """Successive writes replace the file; no tmp residue is left behind."""
    _ = write_progress(1, 10, working_dir=tmp_path)
    path = write_progress(2, 10, working_dir=tmp_path)
    assert parse(path.read_text(), dict[str, object])["step"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["progress.json"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
