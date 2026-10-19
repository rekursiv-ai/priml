"""Tests for the ``security-backdoor-defense`` capture, against a stand-in clone."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import sys

import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.tl.backdoor_defense import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.tl.backdoor_defense.scripts import (
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
    name: str = "pristine",
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Stand in for the port's lockstep: a one-tensor state per update."""
    assert states is not None
    states += [{"w": torch.ones(2)} for _ in range(vision_reference.STEPS)]
    return {f"lockstep/{name}/loss": torch.ones(2)}


def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every run is a stand-in; the runs themselves are tested elsewhere."""
    traced: list[tuple[str, str]] = []

    def trace_driver(script: str, setting: str) -> str:
        traced.append((script, setting))
        return script

    zero = torch.zeros(2)
    batch = {"media": torch.zeros(2), "label": torch.arange(2)}
    moved = {"media": torch.ones(2), "label": torch.arange(2)}
    shared = dict.fromkeys(
        (
            "tiny/x",
            "settings/x/state",
            "decode/x",
            "poison/x",
            "trigger/x",
            "defense/x",
            "lockstep/x/loss",
        ),
        zero,
    )
    stand_ins = {
        (upstream, "variants"): {"pristine": "p", "spectral": "s"},
        (upstream, "filter_record"): {"filter/x": zero},
        (upstream, "filter_inputs"): {"inputs/x": zero},
        (upstream, "main_record"): {"main/x": zero},
        (upstream, "tiny_record"): {"tiny/x": zero},
        (upstream, "setting_record"): (
            torch.nn.Sequential(),
            torch.arange(2),
            {"settings/x/state": zero},
        ),
        (upstream, "poisoned_batches"): ([[batch]], [[batch]]),
        (reference, "decode_record"): {"decode/x": zero},
        (reference, "loader_batches"): ([[moved]], [[batch]]),
        (upstream, "poison_record"): {"poison/x": zero},
        (upstream, "trigger_record"): {"trigger/x": zero},
        (upstream, "defense_record"): {"defense/x": zero},
        (upstream, "lockstep"): (
            {"lockstep/x/loss": zero},
            [{"w": zero}] * vision_reference.STEPS,
        ),
        (upstream, "upstream_sites"): dict.fromkeys(
            capture_reference.REQUIRED_UPSTREAM[1:],
            1,
        ),
        (reference, "draw_sites"): dict.fromkeys(
            capture_reference.REQUIRED_PORT,
            1,
        ),
        (reference, "port_record"): {
            "main/x": torch.ones(2),
            **shared,
        },
        (reference, "filter_record"): {"filter/x": zero},
        (vision_reference, "full_step"): SimpleNamespace(model=torch.nn.Linear(2, 3)),
        (reference, "poison_at_tied_cut"): ["filter/x: tied"],
    }
    for (module, name), value in stand_ins.items():
        monkeypatch.setattr(module, name, returning(value))
    monkeypatch.setattr(upstream, "trace_driver", trace_driver)
    monkeypatch.setattr(
        reference,
        "lockstep_record",
        stand_in_lockstep,
    )
    captured = capture_reference.capture(tmp_path, tmp_path)
    others = [setting for setting in reference.SETTINGS if setting != reference.FIRST]
    assert traced == [
        ("p", reference.FIRST),
        ("s", reference.FIRST),
    ] + [("p", setting) for setting in others]
    assert set(captured.record) == {"main/x", "filter/x", "inputs/x", *shared}
    assert captured.sites["port"] == dict.fromkeys(capture_reference.REQUIRED_PORT, 1)
    assert captured.report == [
        *(
            line
            for setting in reference.SETTINGS
            for line in (
                f"settings/{setting}/state: state: 2 tensors against 0",
                f"settings/{setting}/full/draws: torch.int64[8] vs torch.int64[2]",
                f"loader/{setting}/loader/train/0: 0/media: 2/2 differ",
            )
        ),
        "main/x: 2/2 differ",
        *(f"lockstep/pristine/{step}: w: 2/2 differ" for step in (1, 2)),
        "reference: random call site uniform_@nn/init.py was never reached",
        "filter/x: tied",
    ]


def test_a_call_site_that_a_side_never_reached_is_reported() -> None:
    port: dict[str, int] = dict.fromkeys(capture_reference.REQUIRED_PORT, 1)
    theirs: dict[str, int] = dict.fromkeys(capture_reference.REQUIRED_UPSTREAM, 1)
    assert capture_reference.check_sites(theirs, port) == []
    del theirs["randperm@data/sampler.py"]
    port["rand@vision/data.py"] = 0
    assert capture_reference.check_sites(theirs, port) == [
        "reference: random call site randperm@data/sampler.py was never reached",
        "port: random call site rand@vision/data.py was never reached",
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


def test_main_splits_the_networks_records_into_their_own_golden(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    record = {
        "settings/x/train": torch.zeros(2),
        "tiny/x/features": torch.ones(3),
        "poison/x/indices": torch.arange(3),
    }
    result = vision_reference.Captured(
        record=record,
        sites={"port": {"a@b.py:c": 2}},
        report=[],
    )
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    argv = ["capture", "--root", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert capture_reference.main() == 0
    assert f"0 mismatches; wrote 3 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"poison/x/indices"}
    assert read_tensors(out / "reference_networks.pt").keys() == {
        "settings/x/train",
        "tiny/x/features",
    }
    assert (out / "reference_sites.txt").read_text() == "port a@b.py:c 2\n"


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
