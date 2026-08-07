"""Tests for the CIFAR-10 preparation CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from priml.baselines.cifar10.scripts import prepare_data


if TYPE_CHECKING:
    import pytest


def test_default_directory_matches_the_loop_resolution() -> None:
    """The preparer and a default run must agree on where the data lives.

    They resolve it independently -- the CLI here, the dataset config at
    finalize -- so a divergence would let a successful preparation be followed
    by a run that cannot find the file.
    """
    assert prepare_data.default_directory() == Path("/opt/scratch/datasets/cifar10")


def test_main_prepares_the_requested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[Path] = []
    monkeypatch.setattr(prepare_data, "prepare", called.append)
    monkeypatch.setattr("sys.argv", ["prepare_data", "--directory", str(tmp_path)])
    assert prepare_data.main() == 0
    assert called == [tmp_path]


def test_main_falls_back_to_the_default_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[Path] = []
    monkeypatch.setattr(prepare_data, "prepare", called.append)
    monkeypatch.setattr("sys.argv", ["prepare_data"])
    assert prepare_data.main() == 0
    assert called == [prepare_data.default_directory()]


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
