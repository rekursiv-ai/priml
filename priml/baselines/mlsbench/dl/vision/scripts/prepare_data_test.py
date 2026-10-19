"""Tests for the dataset preparation CLI."""

from __future__ import annotations

from pathlib import Path

import logging
import sys

import pytest

from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.scripts import prepare_data
from priml.baselines.mlsbench.dl.vision.settings import mobilenetv2_fmnist
from priml.lib.testing.checks import assert_help_without_docstring


def test_default_directory_matches_the_loop_resolution() -> None:
    """The preparer and a default run must agree on where the data lives.

    They resolve it independently -- the CLI here, the dataset config at
    finalize -- so a divergence would let a successful preparation be followed
    by a run that cannot find the file.
    """
    assert prepare_data.default_directory() == Path("/opt/scratch/datasets/cifar100")


@pytest.mark.parametrize(
    ("dataset", "directory"),
    [
        ("cifar10", "/opt/scratch/datasets/cifar10"),
        ("cifar100", "/opt/scratch/datasets/cifar100"),
        ("fmnist", "/opt/scratch/datasets/fmnist"),
    ],
)
def test_each_datasets_default_is_where_its_recipe_reads(
    dataset: prepare_data.Dataset,
    directory: str,
) -> None:
    assert prepare_data.default_directory(dataset) == Path(directory)


def test_the_fmnist_default_is_the_settings_own_directory() -> None:
    config = mobilenetv2_fmnist(exp000()).dataset
    config.base_dir = "/opt/scratch"
    assert prepare_data.default_directory("fmnist") == Path(
        config.copy_tree().finalize().working_dir,
    )


class Recorder:
    """Stands in for ``prepare``, keeping each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, directory: Path, *, dataset: str) -> None:
        """Record the call."""
        self.calls.append((directory, dataset))


def test_main_prepares_the_requested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(prepare_data, "prepare", recorder)
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--directory", str(tmp_path)])
    assert prepare_data.main() == 0
    assert recorder.calls == [(tmp_path, "cifar100")]


def test_main_falls_back_to_the_datasets_default_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(prepare_data, "prepare", recorder)
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--dataset", "fmnist"])
    assert prepare_data.main() == 0
    assert recorder.calls == [(prepare_data.default_directory("fmnist"), "fmnist")]


def test_an_unknown_dataset_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--dataset", "mnist"])
    with pytest.raises(SystemExit, match="2"):
        prepare_data.main()


def test_main_help_describes_the_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    help_text = capsys.readouterr().out
    assert (
        "Download a dataset, CIFAR-100 by default, and cache it as 8-bit tensors.\n\n"
        "Run once" in help_text
    )
    assert "--dataset {cifar10,cifar100,fmnist}" in help_text
    assert "prepare_data.py --directory /datasets/my-cifar100" in help_text


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


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, prepare_data, prepare_data.main)


def test_main_configures_info_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record_basic_config(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(logging, "basicConfig", record_basic_config)
    recorder = Recorder()
    monkeypatch.setattr(prepare_data, "prepare", recorder)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data", "--directory", str(tmp_path)],
    )
    assert prepare_data.main() == 0
    assert calls == [{"level": logging.INFO, "format": "%(message)s"}]
    assert recorder.calls == [(tmp_path, "cifar100")]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
