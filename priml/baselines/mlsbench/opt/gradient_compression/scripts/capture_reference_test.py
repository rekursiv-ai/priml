"""Tests for the ``optimization-gradient-compression`` capture, against a stand-in clone.

The stand-in's network, loaders and baseline compressors differ from the real
ones, so a capture of it must name those and nothing it shares with the port's
recipe: the pristine compressor, the schedule and the evaluation cadence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.opt.gradient_compression import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.opt.gradient_compression.scripts import (
    capture_reference,
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
    states += [{"w": torch.ones(2)} for _ in range(2)]
    return {f"lockstep/{name}/loss": torch.ones(2)}


def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every run is a stand-in; the runs themselves are tested elsewhere."""
    scripts = dict.fromkeys(reference.COMPRESSORS, object())
    zero = {"w": torch.zeros(2)}
    batch = {"media": torch.zeros(2), "label": torch.arange(2)}
    moved = {"media": torch.ones(2), "label": torch.arange(2)}
    stand_ins = {
        (upstream, "variants"): scripts,
        (upstream, "upstream_compressor"): {"compressor/x": torch.zeros(2)},
        (upstream, "upstream_network"): ({"draws/x": torch.zeros(2)}, zero),
        (upstream, "upstream_lockstep"): (
            {"lockstep/x/loss": torch.zeros(2)},
            [zero] * 2,
        ),
        (upstream, "upstream_trace"): {"schedule/x": torch.zeros(2)},
        (upstream, "upstream_metrics"): {"metric/x": torch.zeros(2)},
        (upstream, "upstream_transforms"): {"transform/x": torch.zeros(2)},
        (upstream, "upstream_batches"): ([[batch]], [[batch]]),
        (upstream, "upstream_sites"): {"rand_like@x": 1},
        (vision_reference, "decode_batches"): {"loader/x": torch.zeros(2)},
        (capture_reference, "compare_full_size"): ["state: 1 tensors against 2"],
        (reference, "port_record"): {"compressor/x": torch.ones(2)},
        (reference, "loader_batches"): ([[moved]], [[batch]]),
        (reference, "draw_sites"): {"rand_like@y": 1},
    }
    for (module, name), value in stand_ins.items():
        monkeypatch.setattr(module, name, returning(value))
    monkeypatch.setattr(reference, "lockstep_record", stand_in_lockstep)
    captured = capture_reference.capture(tmp_path, tmp_path)
    assert set(captured.record) == {
        "compressor/x",
        "draws/x",
        "lockstep/x/loss",
        "schedule/x",
        "metric/x",
        "transform/x",
        "loader/x",
    }
    assert captured.sites == {
        "reference": {"rand_like@x": 1},
        "port": {"rand_like@y": 1},
    }
    unreached = capture_reference.check_sites({"rand_like@x": 1}, {"rand_like@y": 1})
    assert captured.report == [
        *(["state: 1 tensors against 2"] * len(reference.SETTINGS)),
        "missing draws/x",
        "missing loader/x",
        "missing lockstep/x/loss",
        "missing metric/x",
        "missing schedule/x",
        "missing transform/x",
        "compressor/x: 2/2 differ",
        *(
            f"lockstep/{name}/{step}: w: 2/2 differ"
            for name in reference.COMPRESSORS
            for step in (1, 2)
        ),
        *(
            f"{name}/loader/train/0: 0/media: 2/2 differ"
            for name in ("exp000", "exp004")
        ),
        *unreached,
    ]


def test_compare_full_size_names_the_setting_of_each_differing_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 3)

    class Built:
        """Stands in for the port's step: just its model."""

        def __init__(self) -> None:
            self.model = model

    monkeypatch.setattr(reference, "full_step", returning(Built()))
    theirs = {k: v.clone() for k, v in model.state_dict().items()}
    assert capture_reference.compare_full_size("exp004", theirs) == []
    theirs["bias"] = theirs["bias"] + 1
    assert capture_reference.compare_full_size("exp004", theirs) == [
        "exp004/bias: 3/3 differ",
    ]


def test_check_sites_names_each_unreached_site() -> None:
    report = capture_reference.check_sites({}, {"uniform_@nn/init.py:x": 1})
    assert len(report) == (
        len(capture_reference.REQUIRED_UPSTREAM)
        + len(capture_reference.REQUIRED_PORT)
        - 1
    )


def test_main_prints_the_report_and_writes_nothing_when_something_differs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = vision_reference.Captured(record={}, sites={}, report=["a differs"])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    argv = ["capture", "--root", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert capture_reference.main() == 1
    assert capsys.readouterr().out == "a differs\n1 mismatches; nothing written.\n"
    assert not out.exists()


def test_main_writes_the_golden_and_sites_when_every_comparison_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = vision_reference.Captured(
        record={"x": torch.arange(3)},
        sites={"reference": {"a": 2}, "port": {"b": 3}},
        report=[],
    )
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    argv = ["capture", "--root", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert capture_reference.main() == 0
    assert f"0 mismatches; wrote 1 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"x"}
    assert (out / "reference_sites.txt").read_text() == "reference a 2\nport b 3\n"


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
