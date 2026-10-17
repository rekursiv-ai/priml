"""Tests for the ``cv-pooling-aggregation`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines' pools are not
the real ones, so a capture of it must name those and nothing it shares with
the port: the pristine pool, the schedule and the optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest
import torch

from priml.baselines.mlsbench.pooling.scripts import capture_reference
from priml.baselines.mlsbench.pooling.upstream_test import write_clone
from priml.baselines.mlsbench.pytorch_vision import reference as vision
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
    assert (
        not {
            "pool/avg/value",
            "pool/avg/grad",
            "pool/global_max/value",
            "schedule/lr",
            "optimizer/settings",
        }
        & keys
    )
    assert {"pool/gem/value", "pool/avg_max/value", "draws/avg"} <= keys
    assert any(line.startswith("state: ") for line in report)
    assert any(line.startswith("lockstep/avg/1: ") for line in report)
    assert any(line.startswith("loader/train/") for line in report)


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
