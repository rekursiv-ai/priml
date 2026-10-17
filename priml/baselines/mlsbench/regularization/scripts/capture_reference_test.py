"""Tests for the ``dl-regularization`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56, its launch script passes another
rate and momentum, and its baselines' regularizers are not the real ones, so a
capture of it must name those and nothing it shares with the port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.baselines.mlsbench.regularization import reference, upstream
from priml.baselines.mlsbench.regularization.scripts import capture_reference
from priml.baselines.mlsbench.regularization.upstream_test import write_clone
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


def test_capture_names_what_the_stand_in_does_differently(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report = capture_reference.capture(root, scratch).report
    keys = {line.split(":")[0] for line in report}
    assert "pristine" not in keys
    assert "setting/batch_size" not in keys
    assert {
        "schedule/lr",
        "optimizer/settings",
        "draws/exp000",
        "regularizer/confidence_penalty/value",
        "regularizer/orthogonal_reg/parameters_grad",
    } <= keys
    assert any(line.startswith("lockstep/zero/1: ") for line in report)
    assert any(line.startswith("loader/train/") for line in report)


def test_a_pristine_regularizer_that_returns_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    script = root / upstream.PRISTINE
    script.write_text(script.read_text().replace("raise NotImplementedError(", "("))
    assert capture_reference.check_pristine(root) == ["pristine: does not raise"]


def test_the_batch_size_is_the_settings_for_both_loaders() -> None:
    argv = ["script.py", "--batch-size", "128", "--lr", "0.1"]
    assert torch.equal(
        reference.setting_batch_size(argv),
        reference.port_batch_size(),
    )


def test_main_prints_the_report_and_writes_nothing_when_something_differs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = vision.Captured(record={}, sites={}, report=["a differs"])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    argv = ["capture", "--root", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert capture_reference.main() == 1
    assert capsys.readouterr().out == "a differs\n1 mismatches; nothing written.\n"
    assert not out.exists()


def test_main_writes_the_golden_when_every_comparison_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    record = {"x": torch.arange(3)}
    result = vision.Captured(record=record, sites={}, report=[])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    argv = ["capture", "--root", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert capture_reference.main() == 0
    assert f"0 mismatches; wrote 1 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"x"}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
