"""Tests for the ``cv-multitask-loss`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines'
combinations are not the real ones, so a capture of it must name those and
nothing it shares with the port: the superclass-free schedule and optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest
import torch

from priml.baselines.mlsbench.dl.multitask import upstream
from priml.baselines.mlsbench.dl.multitask.scripts import capture_reference
from priml.baselines.mlsbench.dl.multitask.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


@pytest.mark.slow
def test_capture_names_what_the_stand_in_does_differently(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report = capture_reference.capture(root, scratch).report
    keys = {line.split(":")[0] for line in report}
    assert "schedule/lr" not in keys
    assert "optimizer/settings" not in keys
    assert "combination/sum/value" not in keys
    assert {"combination/dwa/value", "data/coarse_of_fine", "draws/sum"} <= keys
    assert any(line.startswith("lockstep/sum/1: ") for line in report)
    assert any(line.startswith("loader/train/") for line in report)


def test_wiring_reports_a_variant_told_the_wrong_epochs() -> None:
    good = upstream.Trace(epochs=[(e, 200) for e in range(200)])
    bad = upstream.Trace(epochs=[(0, 200)])
    assert capture_reference.compare_wiring({"sum": good, "dwa": bad}) == [
        "main/dwa: epoch arguments differ",
    ]


def test_main_prints_the_report_and_writes_nothing_when_something_differs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = vision.Captured(record={}, sites={}, report=["a differs"])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture", "--root", str(tmp_path), "--out", str(out)],
    )
    assert capture_reference.main() == 1
    assert capsys.readouterr().out == "a differs\n1 mismatches; nothing written.\n"
    assert not out.exists()


def test_main_writes_the_golden_when_every_comparison_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = vision.Captured(record={"x": torch.arange(3)}, sites={}, report=[])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture", "--root", str(tmp_path), "--out", str(out)],
    )
    assert capture_reference.main() == 0
    assert f"0 mismatches; wrote 1 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"x"}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
