"""Tests for the ``cv-data-augmentation`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines' pipelines are
not the real ones, so a capture of it must name those and nothing it shares
with the port: the pristine pipeline, the test pipeline, the schedule and the
optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import sys

import pytest
import torch

from priml.baselines.mlsbench.dl.augmentation import (
    reference as augmentation,
    upstream,
)
from priml.baselines.mlsbench.dl.augmentation.scripts import capture_reference
from priml.baselines.mlsbench.dl.augmentation.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.upstream import Batches


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD.parent / "testdata"


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


def passes(fill: float) -> tuple[Batches, Batches]:
    """Return one training and one test pass of a single ``(2, 3)`` batch."""
    batch = {"media": torch.full((2, 3), fill), "label": torch.arange(2)}
    return [[batch]], [[batch]]


def port_lockstep(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Stand in for the port's lockstep: one tensor per state, zero media."""
    if states is not None:
        states += [{"w": torch.zeros(2)} for _ in range(reference.STEPS)]
    return {
        f"lockstep/{name}/media": torch.zeros(2),
        f"lockstep/{name}/loss": torch.zeros(2),
    }


def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The port-side runs are tested in ``reference_test``; here they are stand-ins."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stored = read_tensors(_DATA / "reference.pt")
    monkeypatch.setattr(
        augmentation,
        "pipeline_images",
        returning(augmentation.pipeline_images(2)),
    )
    monkeypatch.setattr(augmentation, "port_record", returning(stored))
    monkeypatch.setattr(augmentation, "lockstep_record", port_lockstep)
    monkeypatch.setattr(augmentation, "loader_batches", returning(passes(1.0)))
    monkeypatch.setattr(upstream, "upstream_batches", returning(passes(0.0)))
    monkeypatch.setattr(
        vision_capture,
        "compare_full_size",
        returning(["full size differs"]),
    )
    captured = capture_reference.capture(root, scratch)
    assert captured.record.keys() == stored.keys()
    report = captured.report
    assert report[0] == "full size differs"
    names = ("pad_crop_flip", *upstream.EDITED)
    for name in names:
        assert f"lockstep/{name}/1: state: 1 tensors against" in "\n".join(report)
        assert any(line.startswith(f"lockstep/{name}/media: ") for line in report)
        assert f"{name}: loader/train/0: 0/media: 6/6 differ" in report
        assert f"{name}: loader/test/0: 0/media: 6/6 differ" in report
    # The stand-in's policies are not torchvision's, so it reaches no operation,
    # and two images are too few for the port to reach every one.
    census = {line.rsplit(": ", 1)[0] for line in report if "never reached" in line}
    policies = capture_reference.POLICIES
    assert census == {
        f"{side} {name}" for side in ("reference", "port") for name in policies
    }
    assert "reference randaugment: posterize+ never reached" in report
    assert [line for line in report if line.endswith("operation counts differ")] == [
        f"{name}: operation counts differ" for name in policies
    ]
    assert any(line.startswith("pipeline/randaugment: ") for line in report)
    shared = ("pipeline/test", "pipeline/pad_crop_flip")
    assert not any(line.startswith(shared) for line in report)


@pytest.mark.compute_training
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
    result = reference.Captured(record={}, sites={}, report=["a differs"])
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
    result = reference.Captured(record=record, sites={}, report=[])
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
