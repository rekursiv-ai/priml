"""Tests for the ROGII staging entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest

from priml.baselines.wellbore_geology.scripts import prepare_data
from priml.baselines.wellbore_geology.scripts.prepare_data import StagedData


if TYPE_CHECKING:
    from pathlib import Path


def test_prepare_command_routes_explicit_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []

    def record(source: Path, *, destination: Path) -> StagedData:
        calls.append((source, destination))
        return StagedData(directory=destination, train_wells=("a",), test_wells=("b",))

    monkeypatch.setattr(prepare_data, "stage_data", record)
    source = tmp_path / "source"
    destination = tmp_path / "staged"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data", "--source", str(source), "--directory", str(destination)],
    )
    assert prepare_data.main() == 0
    assert calls == [(source, destination)]


def test_prepare_help_contains_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    text = capsys.readouterr().out
    assert "Stage the real ROGII CSV release" in text
    assert "exec uv" not in text


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
