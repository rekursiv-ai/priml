"""Tests for the CIFAR-10 preparation CLI."""

from __future__ import annotations

from pathlib import Path

import logging
import sys

import pytest

from priml.baselines.cifar10.scripts import prepare_data


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
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--directory", str(tmp_path)])
    assert prepare_data.main() == 0
    assert called == [tmp_path]


def test_main_falls_back_to_the_default_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[Path] = []
    monkeypatch.setattr(prepare_data, "prepare", called.append)
    monkeypatch.setattr(sys, "argv", ["prepare_data"])
    assert prepare_data.main() == 0
    assert called == [prepare_data.default_directory()]


def test_main_help_describes_the_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    help_text = capsys.readouterr().out
    assert (
        "Download CIFAR-10 and cache it as normalized tensors.\n\nRun once" in help_text
    )
    assert "prepare_data.py --directory /datasets/my-cifar10" in help_text


def test_main_uses_line_delimited_docstring_for_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(prepare_data, "__doc__", "first\n\nexact description")
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    help_text = capsys.readouterr().out
    assert "\nexact description\n" in help_text


def test_main_requires_its_module_docstring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_data, "__doc__", None)
    with pytest.raises(
        ValueError,
        match=r"\AExpected __doc__ is not None\.\Z",
    ):
        prepare_data.main()


def test_main_configures_info_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record_basic_config(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(logging, "basicConfig", record_basic_config)
    prepared: list[Path] = []
    monkeypatch.setattr(prepare_data, "prepare", prepared.append)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data", "--directory", str(tmp_path)],
    )
    assert prepare_data.main() == 0
    assert calls == [{"level": logging.INFO, "format": "%(message)s"}]
    assert prepared == [tmp_path]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
