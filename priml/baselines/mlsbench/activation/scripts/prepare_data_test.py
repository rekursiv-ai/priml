"""Tests for the CIFAR-10 preparation CLI."""

from __future__ import annotations

from pathlib import Path

import sys

import pytest

from priml.baselines.mlsbench.activation.scripts import prepare_data
from priml.lib.testing.checks import assert_help_without_docstring


def test_default_directory_matches_the_loop_resolution() -> None:
    assert prepare_data.default_directory() == Path("/opt/scratch/datasets/cifar10")


def test_main_prepares_cifar10_in_the_requested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[Path | str, str]] = []

    def record(directory: Path | str, *, dataset: str) -> None:
        called.append((directory, dataset))

    monkeypatch.setattr(prepare_data, "prepare", record)
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--directory", str(tmp_path)])
    assert prepare_data.main() == 0
    assert called == [(tmp_path, "cifar10")]


def test_main_help_describes_the_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    assert "Download CIFAR-10 and cache it as 8-bit tensors." in capsys.readouterr().out


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, prepare_data, prepare_data.main)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
