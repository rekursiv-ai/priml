"""Tests for the ``dl-regularization`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56, its launch script passes another
rate and momentum, and its baselines' regularizers are not the real ones, so a
capture of it must name those and nothing it shares with the port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import torch

from priml.baselines.mlsbench.dl.regularization import reference, upstream
from priml.baselines.mlsbench.dl.regularization.scripts import capture_reference
from priml.baselines.mlsbench.dl.regularization.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
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


def stand_in_forward(script: object, arch: str, setting: str) -> dict[str, Tensor]:
    """Stand in for a setting's forward: one value under the setting's prefix."""
    del script, arch
    return {f"settings/{setting}/train": torch.zeros(2)}


def stand_in_lockstep(
    name: str,
    *,
    states: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Stand in for a port lockstep: a one-tensor state per update, a loss."""
    states += [{"w": torch.ones(2)} for _ in range(vision.STEPS)]
    return {f"{name}/loss": torch.ones(2)}


def test_capture_assembles_both_sides_and_every_setting_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every run is a stand-in; the runs themselves are tested elsewhere."""
    scripts = dict.fromkeys(reference.REGULARIZERS, object())
    zero = {"w": torch.zeros(2)}
    lockstep: vision_upstream.Lockstep = ({}, [zero] * vision.STEPS)
    batch = {"media": torch.zeros(2), "label": torch.arange(2)}
    moved = {"media": torch.ones(2), "label": torch.arange(2)}
    tiny = torch.nn.Linear(2, 3)
    stand_ins = {
        (upstream, "pristine_raises"): True,
        (upstream, "variants"): scripts,
        (upstream, "setting_argv"): ["script.py", "--epochs", "200"],
        (upstream, "trace_setting"): vision_upstream.MainTrace(),
        (upstream, "upstream_lockstep"): lockstep,
        (upstream, "upstream_regularizers"): {},
        (upstream, "upstream_record"): {"regularizer/x": torch.zeros(2)},
        (reference, "setting_batch_size"): torch.ones(2),
        (vision_upstream, "upstream_batches"): ([[batch]], [[batch]]),
        (shared, "compare_wiring"): ["wiring: differs"],
        (shared, "compare_full_size"): ["state: 1 tensors against 2"],
        (reference, "port_record"): {
            "regularizer/x": torch.ones(2),
            "setting/batch_size": torch.ones(2),
        },
        (vision, "loader_batches"): ([[moved]], [[batch]]),
        (upstream, "upstream_setting_recipe"): {"settings/s/recipe": torch.zeros(2)},
        (shared, "compare_setting"): [],
        (upstream, "tiny_network"): tiny,
        (reference, "tiny_network"): tiny,
        (upstream, "upstream_setting_regularizers"): {},
        (vision, "decode_batches"): {"loader/index": torch.zeros(2)},
        (vision, "setting_batches"): ([[batch]], [[batch]]),
        (reference, "settings_record"): {},
        (vision, "task_settings_record"): {},
    }
    for (module, name), value in stand_ins.items():
        monkeypatch.setattr(module, name, returning(value))
    monkeypatch.setattr(vision_upstream, "tiny_setting_record", stand_in_forward)
    monkeypatch.setattr(reference, "lockstep_record", stand_in_lockstep)
    monkeypatch.setattr(reference, "setting_lockstep_record", stand_in_lockstep)
    captured = capture_reference.capture(tmp_path, tmp_path)
    settings = list(reference.SETTING_REGULARIZERS)
    fmnist = [s for s in settings if vision.SETTINGS[s].dataset != "cifar100"]
    assert set(captured.record) == {
        "regularizer/x",
        "setting/batch_size",
        "settings/s/recipe",
        *(
            f"settings/{s}/{name}/train"
            for s, variants in reference.SETTING_REGULARIZERS.items()
            for name in variants
        ),
        *(f"settings/{s}/loader/index" for s in fmnist),
    }
    assert captured.report == [
        "wiring: differs",
        "state: 1 tensors against 2",
        "regularizer/x: 2/2 differ",
        *(
            f"lockstep/{name}/{step}: w: 2/2 differ"
            for name in reference.REGULARIZERS
            for step in range(1, vision.STEPS + 1)
        ),
        "loader/train/0: 0/media: 2/2 differ",
        *(f"settings/{s}/wiring: differs" for s in settings),
        *(f"missing {k}" for k in sorted(captured.record) if k.startswith("settings/")),
        *(
            f"lockstep/{s}/{name}/{step}: w: 2/2 differ"
            for s in settings
            for name in reference.REGULARIZERS
            for step in range(1, vision.STEPS + 1)
        ),
    ]


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
