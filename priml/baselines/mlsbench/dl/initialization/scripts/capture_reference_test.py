"""Tests for the ``dl-weight-initialization`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its initializations are not
the real ones, so a capture of it must name those and nothing it shares with
the port: the learning rates and the optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import torch

from priml.baselines.mlsbench.dl.initialization import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.dl.initialization.scripts import capture_reference
from priml.baselines.mlsbench.dl.initialization.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture_reference,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor

    import pytest

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep


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
    states += [{"w": torch.ones(2)} for _ in range(vision_reference.STEPS)]
    return {f"{name}/loss": torch.ones(2)}


def stand_in_full_size(
    script: object,
    trace: object,
    name: str,
) -> list[str]:
    """Stand in for the full-size comparison: one line naming the variant."""
    del script, trace
    return [f"full/{name}/state: differs"]


def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every run is a stand-in; the runs themselves are tested elsewhere."""
    scripts = dict.fromkeys(reference.INITIALIZATIONS, object())
    zero = {"w": torch.zeros(2)}
    lockstep = ({"lockstep/x/loss": torch.zeros(2)}, [zero] * vision_reference.STEPS)
    batch = {"media": torch.zeros(2), "label": torch.arange(2)}
    moved = {"media": torch.ones(2), "label": torch.arange(2)}
    network = ({"network/x/train": torch.zeros(2)}, zero)
    stand_ins = {
        (upstream, "variants"): scripts,
        (vision_upstream, "trace_main"): vision_upstream.MainTrace(),
        (upstream, "upstream_network"): network,
        (upstream, "upstream_lockstep"): lockstep,
        (upstream, "upstream_record"): {"network/x/train": torch.zeros(2)},
        (vision_upstream, "upstream_batches"): ([[batch]], [[batch]]),
        (capture_reference, "compare_networks"): ["network/x/state: differs"],
        (reference, "port_record"): {"network/x/train": torch.ones(2)},
        (vision_reference, "loader_batches"): ([[moved]], [[batch]]),
        (vision_capture_reference, "compare_task_settings"): (
            {"settings/s/x/train": torch.zeros(2)},
            ["settings/s/x/state: differs"],
        ),
        (upstream, "upstream_setting_lockstep"): {
            "s/x": ({"settings/s/lockstep/x/loss": torch.zeros(2)}, [zero] * 2),
        },
        (reference, "settings_record"): {
            "settings/s/lockstep/x/loss": torch.ones(2),
        },
    }
    for (module, name), value in stand_ins.items():
        monkeypatch.setattr(module, name, returning(value))
    monkeypatch.setattr(capture_reference, "compare_full_size", stand_in_full_size)
    monkeypatch.setattr(reference, "lockstep_record", stand_in_lockstep)
    monkeypatch.setattr(
        reference,
        "setting_lockstep_record",
        stand_in_lockstep,
    )
    captured = capture_reference.capture(tmp_path, tmp_path)
    assert set(captured.record) == {
        "network/x/train",
        "settings/s/x/train",
        "settings/s/lockstep/x/loss",
    }
    assert captured.report == [
        *(f"full/{name}/state: differs" for name in reference.INITIALIZATIONS),
        "network/x/train: 2/2 differ",
        "network/x/state: differs",
        *(
            f"lockstep/{name}/{step}: w: 2/2 differ"
            for name in reference.INITIALIZATIONS
            for step in range(1, vision_reference.STEPS + 1)
        ),
        "loader/train/0: 0/media: 2/2 differ",
        "settings/s/x/state: differs",
        "settings/s/lockstep/x/loss: 2/2 differ",
        *(f"lockstep/s/x/{step}: w: 2/2 differ" for step in (1, 2)),
    ]


def test_the_full_size_comparison_names_a_network_that_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    script = upstream.variants(root, scratch)["normal"]
    with vision_upstream.hidden_accelerators():
        trace = vision_upstream.trace_main(script)
    full_step = vision_reference.full_step

    def shallow(factory: Callable[[], VisionTrainLoop]) -> VisionTrainStep:
        def build() -> VisionTrainLoop:
            config = factory()
            assert isinstance(config.step.model, ResNet.Config)
            config.step.model.blocks_per_stage = (1, 1, 1)
            return config

        return full_step(build)

    monkeypatch.setattr(vision_reference, "full_step", shallow)
    report = capture_reference.compare_full_size(script, trace, "normal")
    assert report[0].startswith("full/normal/state: ")
    assert [line.split(":")[0] for line in report[1:]] == [
        "full/normal/forward",
        "full/normal/forward",
    ]


def test_the_network_comparison_names_each_initialization_that_differs() -> None:
    ours = len(reference.tiny_model("normal").state_dict())
    theirs = {"normal": {"w": torch.zeros(2)}, "fixup": {"w": torch.zeros(2)}}
    assert capture_reference.compare_networks(theirs) == [
        f"network/normal/state: state: {ours} tensors against 1",
        f"network/fixup/state: state: {ours} tensors against 1",
    ]


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


def test_main_writes_the_golden_when_every_comparison_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    record = {"x": torch.arange(3)}
    result = vision_reference.Captured(record=record, sites={}, report=[])
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
