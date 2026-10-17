"""Tests for the ``cv-data-augmentation`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines' pipelines are
not the real ones, so a capture of it must name those and nothing it shares
with the port: the pristine pipeline, the test pipeline, the schedule and the
optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest
import torch

from priml.baselines.mlsbench.augmentation.scripts import capture_reference
from priml.baselines.mlsbench.augmentation.upstream_test import write_clone
from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.baselines.mlsbench.pytorch_vision.scripts import (
    capture_reference as vision_capture,
)
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
    assert {"pipeline/pad_crop_flip", "pipeline/test", "schedule/lr"} & keys == set()
    assert "optimizer/settings" not in keys
    assert {
        "pipeline/cutout",
        "pipeline/randaugment",
        "pipeline/trivialaugment",
        "draws/exp000",
        "reference randaugment",
        "randaugment",
    } <= keys
    assert not any(line.startswith("port ") for line in report)
    assert "pad_crop_flip" not in keys
    assert {"cutout", "trivialaugment"} <= keys
    assert any(line.startswith("lockstep/pad_crop_flip/1: ") for line in report)


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


@pytest.mark.parametrize(("size", "code"), [(3, 0), (vision_capture.CEILING, 1)])
def test_main_writes_the_golden_and_refuses_one_past_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    size: int,
    code: int,
) -> None:
    record = {"x": torch.zeros(size, dtype=torch.uint8)}
    result = vision.Captured(record=record, sites={}, report=[])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    out.mkdir()
    (out / "reference.pt").write_bytes(b"previous")
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture", "--root", str(tmp_path), "--out", str(out)],
    )
    assert capture_reference.main() == code
    printed = capsys.readouterr().out
    if code:
        assert "the ceiling is" in printed
        assert (out / "reference.pt").read_bytes() == b"previous"
    else:
        assert f"0 mismatches; wrote 1 values to {out}." in printed
        assert read_tensors(out / "reference.pt").keys() == {"x"}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
