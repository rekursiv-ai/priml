"""Tests for the ``cv-multitask-loss`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines'
combinations are not the real ones, so a capture of it must name those and
nothing it shares with the port: the superclass-free schedule and optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import sys

import pytest
import torch

from priml.baselines.mlsbench.dl.multitask import (
    reference as multitask,
    upstream,
)
from priml.baselines.mlsbench.dl.multitask.scripts import capture_reference
from priml.baselines.mlsbench.dl.multitask.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import reference
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.baselines.mlsbench.dl.multitask.experiments import (
        MultiTaskTrainLoop,
    )
    from priml.baselines.mlsbench.dl.multitask.train_step import MultiTaskStep


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD.parent / "testdata"


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


def passes(fill: float) -> tuple[multitask.Batches, multitask.Batches]:
    """Return one training and one test pass of a single ``(2, 3)`` batch."""
    batch = {"media": torch.full((2, 3), fill), "label": torch.arange(2)}
    return [[batch]], [[batch]]


def port_lockstep(
    name: str,
    *,
    states: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Stand in for the port's lockstep: one tensor per state."""
    states += [{"w": torch.zeros(2)} for _ in range(reference.STEPS)]
    return {f"lockstep/{name}/loss": torch.zeros(2)}


def shallow(factory: Callable[[], MultiTaskTrainLoop]) -> MultiTaskStep:
    """Build ``factory``'s step one block per stage, in place of the full network."""
    return multitask.tiny_step(factory, epochs=reference.EPOCHS)


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
    monkeypatch.setattr(multitask, "port_record", returning(stored))
    monkeypatch.setattr(multitask, "lockstep_record", port_lockstep)
    monkeypatch.setattr(multitask, "loader_batches", returning(passes(1.0)))
    monkeypatch.setattr(multitask, "full_step", shallow)
    monkeypatch.setattr(upstream, "upstream_batches", returning(passes(0.0)))
    captured = capture_reference.capture(root, scratch)
    report = captured.report
    assert {"combination/dwa/value", "data/coarse_of_fine", "draws/sum"} <= {
        line.split(":")[0] for line in report
    }
    assert captured.record.keys() == stored.keys()
    assert not any(line.startswith(("main/", "schedule/lr")) for line in report)
    (state,) = (line for line in report if line.startswith("state: "))
    assert state.endswith("tensors against 6")
    for mode in ("train", "eval"):
        assert any(line.startswith(f"forward/{mode}/0: ") for line in report)
        assert any(line.startswith(f"forward/{mode}/1: ") for line in report)
    for name in ("sum", *upstream.EDITED):
        assert any(line.startswith(f"lockstep/{name}/1: state: ") for line in report)
    assert "loader/train/0: 0/media: 6/6 differ" in report
    assert "loader/test/0: 0/media: 6/6 differ" in report


@pytest.mark.compute_training
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


def test_main_writes_the_golden_when_every_comparison_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = reference.Captured(record={"x": torch.arange(3)}, sites={}, report=[])
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
