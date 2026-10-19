"""Tests for the ``dl-normalization`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines' norms are not
the real ones, so a capture of it must name those and nothing it shares with
the port: the schedule, the optimizer and the reported accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import torch

from priml.baselines.mlsbench.dl.normalization import (
    reference as port,
    upstream,
)
from priml.baselines.mlsbench.dl.normalization.scripts import capture_reference
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as shared,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor

    import pytest


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object, **_keywords: object) -> T:
        return value

    return call


def stand_in_lockstep(
    name: str,
    *,
    states: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Stand in for a port lockstep: a one-tensor state per update, a loss."""
    states += [{"w": torch.ones(2)} for _ in range(reference.STEPS)]
    return {f"{name}/loss": torch.ones(2)}


def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every run is a stand-in; the runs themselves are tested elsewhere."""
    scripts = dict.fromkeys(port.NORMS, object())
    zero = {"w": torch.zeros(2)}
    lockstep = ({"lockstep/x/loss": torch.zeros(2)}, [zero] * reference.STEPS)
    batch = {"media": torch.zeros(2), "label": torch.arange(2)}
    moved = {"media": torch.ones(2), "label": torch.arange(2)}
    stand_ins = {
        (upstream, "variants"): scripts,
        (vision_upstream, "trace_main"): vision_upstream.MainTrace(),
        (upstream, "upstream_lockstep"): lockstep,
        (upstream, "upstream_record"): {"norm/x": torch.zeros(2)},
        (vision_upstream, "upstream_batches"): ([[batch]], [[batch]]),
        (shared, "compare_full_size"): ["state: 1 tensors against 2"],
        (port, "port_record"): {"norm/x": torch.ones(2)},
        (reference, "loader_batches"): ([[moved]], [[batch]]),
        (shared, "compare_task_settings"): (
            {"settings/s/x/train": torch.zeros(2)},
            ["settings/s/x/state: differs"],
        ),
        (upstream, "upstream_setting_lockstep"): {
            "s/x": ({"settings/s/lockstep/x/loss": torch.zeros(2)}, [zero] * 2),
        },
        (port, "settings_record"): {"settings/s/lockstep/x/loss": torch.ones(2)},
    }
    for (module, name), value in stand_ins.items():
        monkeypatch.setattr(module, name, returning(value))
    monkeypatch.setattr(port, "lockstep_record", stand_in_lockstep)
    monkeypatch.setattr(port, "setting_lockstep_record", stand_in_lockstep)
    captured = capture_reference.capture(tmp_path, tmp_path)
    assert set(captured.record) == {
        "norm/x",
        "settings/s/x/train",
        "settings/s/lockstep/x/loss",
    }
    assert captured.report == [
        *(f"full/{name}: state: 1 tensors against 2" for name in port.NORMS),
        "norm/x: 2/2 differ",
        *(
            f"lockstep/{name}/{step}: w: 2/2 differ"
            for name in port.NORMS
            for step in range(1, reference.STEPS + 1)
        ),
        "loader/train/0: 0/media: 2/2 differ",
        "settings/s/x/state: differs",
        "settings/s/lockstep/x/loss: 2/2 differ",
        *(f"lockstep/s/x/{step}: w: 2/2 differ" for step in (1, 2)),
    ]


def test_main_prints_the_report_and_writes_nothing_when_something_differs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = reference.Captured(record={}, sites={}, report=["a differs"])
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
    result = reference.Captured(record=record, sites={}, report=[])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    argv = ["capture", "--root", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert capture_reference.main() == 0
    assert f"0 mismatches; wrote 1 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"x"}
    assert read_tensors(out / "reference_settings.pt").keys() == set()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
