"""Tests for the ``cv-classification-loss`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its focal and PolyLoss edits
are not the real ones, so a capture of it must name those and nothing it shares
with the port: cross-entropy and label smoothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import sys

import torch

from priml.baselines.mlsbench.dl.classification_loss import reference, upstream
from priml.baselines.mlsbench.dl.classification_loss.scripts import (
    capture_reference,
)
from priml.baselines.mlsbench.dl.classification_loss.upstream_test import (
    write_clone,
)
from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture_reference,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    import pytest

_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD.parent / "testdata"

STAND_IN_LOSSES: Final = {
    "loss/focal/value",
    "loss/focal/grad",
    "loss/poly/value",
    "loss/poly/grad",
}
"""The loss values the stand-in's edits, not the real ones, get wrong."""


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


def clone(tmp_path: Path) -> tuple[Path, Path]:
    """Write the stand-in clone; return its root and an empty scratch directory."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return root, scratch


def test_the_stand_in_shares_cross_entropy_and_label_smoothing(tmp_path: Path) -> None:
    record = upstream.upstream_record(upstream.variants(*clone(tmp_path)), {})
    port = reference.loss_record()
    assert record.keys() == port.keys()
    shared = ("loss/cross_entropy/", "loss/label_smoothing/")
    assert (
        mismatches(
            {k: v for k, v in record.items() if k.startswith(shared)},
            {k: v for k, v in port.items() if k.startswith(shared)},
        )
        == []
    )
    report = mismatches(record, port)
    assert {line.split(":")[0] for line in report} == STAND_IN_LOSSES


def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The port-side runs are tested elsewhere; here they are stand-ins, to test the glue."""
    stored = read_tensors(_DATA / "reference.pt")
    monkeypatch.setattr(reference, "port_record", returning(stored))

    def lockstep(
        name: str,
        *,
        states: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        states += [{"w": torch.zeros(2)}] * vision_reference.STEPS
        return {name: torch.zeros(2)}

    monkeypatch.setattr(reference, "lockstep_record", lockstep)

    def setting_lockstep(
        name: str,
        *,
        states: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if states is not None:
            states += [{"w": torch.zeros(2)}] * vision_reference.STEPS
        return {f"settings/{name}": torch.zeros(2)}

    monkeypatch.setattr(reference, "setting_lockstep_record", setting_lockstep)
    empty: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(reference, "setting_loss_record", returning(empty))
    # The stand-in has a ResNet only; each setting's run trains it in place of
    # the setting's network, which ``tiny_model``'s own tests cover.
    built: list[str] = []

    def resnet(script: vision_upstream.Upstream, arch: str, classes: int) -> nn.Module:
        built.append(arch)
        return script.ResNet(script.BasicBlock, [1, 1, 1], classes)

    monkeypatch.setattr(vision_upstream, "tiny_model", resnet)

    def settings(
        *_arguments: object,
        **_keywords: object,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        return {"settings/x/draws": torch.zeros(2)}, ["settings/x/state: differs"]

    monkeypatch.setattr(vision_capture_reference, "compare_task_settings", settings)
    captured = capture_reference.capture(*clone(tmp_path))
    setting_records = {
        f"settings/{setting}/{kind}/{name}/{field}"
        for setting in reference.SETTING_LOSSES
        for name in reference.LOSSES
        for kind, field in (
            ("loss", "value"),
            ("loss", "grad"),
            ("lockstep", "loss"),
            ("lockstep", "logits"),
            ("lockstep", "probe"),
            ("lockstep", "context"),
        )
    }
    assert captured.record.keys() == (
        stored.keys() | setting_records | {"settings/x/draws"}
    )
    assert captured.sites == {}
    assert built.count("vgg16bn") == built.count("mobilenetv2") == len(reference.LOSSES)
    keys = {line.split(":")[0] for line in captured.report}
    assert {key for key in keys if key.startswith("loss/")} == STAND_IN_LOSSES
    assert "lockstep/cross_entropy/logits" in keys
    assert "lockstep/poly/1" in keys
    assert "lockstep/vgg16bn-cifar100/poly/1" in keys
    assert "settings/x/state" in keys


def test_a_lockstep_run_that_matches_reports_nothing_and_one_that_drifts_is_named() -> (
    None
):
    ours: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("cross_entropy", states=ours)
    run = reference.lockstep_record
    record: dict[str, torch.Tensor] = {}
    lockstep = {"cross_entropy": (record, ours)}
    assert vision_capture_reference.compare_lockstep(lockstep, run=run) == []
    drifted = [dict(state) for state in ours]
    weight = drifted[1]["stem.weight"].clone()
    weight.view(-1)[0] = torch.nextafter(weight.view(-1)[0], torch.tensor(torch.inf))
    drifted[1]["stem.weight"] = weight
    assert vision_capture_reference.compare_lockstep(
        {"cross_entropy": ({}, drifted)},
        run=run,
    ) == ["lockstep/cross_entropy/2: stem.weight: 1/108 differ"]


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
    record["settings/y"] = torch.arange(2)
    assert capture_reference.main() == 0
    assert f"0 mismatches; wrote 2 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"x"}
    assert read_tensors(out / "reference_settings.pt").keys() == {"settings/y"}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
