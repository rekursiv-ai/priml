"""Tests for the ``dl-residual-connection`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines' blocks are not
the real ones, so a capture of it must name those and nothing it shares with
the port: the schedule and the optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import functools
import sys

import torch

from priml.baselines.mlsbench.dl.residual import (
    reference as residual,
    upstream as residual_upstream,
)
from priml.baselines.mlsbench.dl.residual.scripts import capture_reference
from priml.baselines.mlsbench.dl.residual.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    import pytest

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
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
    states: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Stand in for the port's lockstep: one tensor per state."""
    states += [{"w": torch.zeros(2)} for _ in range(reference.STEPS)]
    return {f"lockstep/{name}/loss": torch.zeros(2)}


def shallow(factory: Callable[[], VisionTrainLoop]) -> Callable[[], VisionTrainLoop]:
    """Return ``factory`` with one block per stage, in place of ResNet-56."""

    def build() -> VisionTrainLoop:
        config = factory()
        assert isinstance(config.step.model, ResNet.Config)
        config.step.model.blocks_per_stage = (1, 1, 1)
        return config

    return build


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
    monkeypatch.setattr(residual, "port_record", returning(stored))
    monkeypatch.setattr(residual, "lockstep_record", port_lockstep)
    monkeypatch.setattr(
        residual,
        "VARIANTS",
        {name: shallow(factory) for name, factory in residual.VARIANTS.items()},
    )
    monkeypatch.setattr(reference, "loader_batches", returning(passes(1.0)))
    monkeypatch.setattr(upstream, "upstream_batches", returning(passes(0.0)))
    monkeypatch.setattr(
        vision_capture,
        "compare_full_size",
        returning(["full size differs"]),
    )
    # The real ``capture_settings`` trains every setting; its own test below covers it.
    settings = {"settings/x": torch.zeros(2)}
    monkeypatch.setattr(
        capture_reference,
        "capture_settings",
        returning((settings, ["settings differ"])),
    )
    monkeypatch.setattr(residual, "settings_record", returning(settings))
    captured = capture_reference.capture(root, scratch)
    assert settings.items() <= captured.record.items()
    report = captured.report
    assert report[0] == "full size differs"
    assert "settings differ" in report
    assert {"block/basic/identity/train", "draws/basic"} <= {
        line.split(":")[0] for line in report
    }
    shared = ("schedule/lr", "optimizer/settings")
    assert not any(line.startswith(shared) for line in report)
    for name in ("pre_activation", "gated_residual", "stochastic_depth"):
        assert any(line.startswith(f"full/{name}: state: ") for line in report)
        for mode in ("train", "eval"):
            assert any(
                line.startswith(f"full/{name}: forward/{mode}: ") for line in report
            )
        assert any(line.startswith(f"lockstep/{name}/1: state: ") for line in report)
    assert not any(line.startswith("full/basic: ") for line in report)
    assert "loader/train/0: 0/media: 6/6 differ" in report
    assert "block/stochastic_depth: the draws reach only one branch" not in report


class Calls:
    """Stand-ins that return fixed values and record every call's arguments."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def returning[T](self, name: str, value: T) -> Callable[..., T]:
        """Return a stand-in named ``name`` that records its arguments."""

        def call(*arguments: object, **keywords: object) -> T:
            self.seen.append((name, arguments, keywords))
            return value

        return call

    def named(self, name: str) -> list[tuple[tuple[object, ...], dict[str, object]]]:
        """Return the arguments of every call to the stand-in ``name``, in order."""
        return [(a, k) for n, a, k in self.seen if n == name]


def test_capture_settings_joins_every_settings_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runs are tested elsewhere; here they are stubbed, to test the glue."""
    calls = Calls()
    trace = upstream.MainTrace(
        rates=[0.25],
        settings=dict.fromkeys(reference.HYPERPARAMETERS, 0.5),
    )
    positions = torch.arange(3)
    lockstep: upstream.Lockstep = ({"lockstep/x/loss": torch.ones(2)}, [])
    stubs: list[tuple[object, str, object]] = [
        (vision_capture, "compare_task_settings", ({"shared": positions}, ["shared"])),
        (residual_upstream, "block_positions", positions),
        (upstream, "trace_main", trace),
        (
            reference,
            "schedule_record",
            {"schedule/lr": torch.tensor([0.25], dtype=torch.float64)},
        ),
        (
            reference,
            "optimizer_record",
            {
                "optimizer/settings": torch.zeros(
                    len(reference.HYPERPARAMETERS),
                    dtype=torch.float64,
                ),
            },
        ),
        (residual_upstream, "upstream_lockstep", lockstep),
        (vision_capture, "compare_lockstep", ["lockstep differs"]),
    ]
    for module, name, value in stubs:
        monkeypatch.setattr(module, name, calls.returning(name, value))
    scripts = {name: cast("upstream.Upstream", object()) for name in residual.VARIANTS}
    record, report = capture_reference.capture_settings(scripts, tmp_path)
    settings = residual.SETTINGS
    expected = {"shared": positions, "lockstep/x/loss": torch.ones(2)} | {
        f"settings/{s}/stochastic_depth/positions": positions for s in settings
    }
    assert mismatches(expected, record) == []
    assert report == ["shared"] + [
        line
        for s in settings
        for line in (
            (
                f"settings/{s}/optimizer/settings: "
                f"{len(reference.HYPERPARAMETERS)}/{len(reference.HYPERPARAMETERS)} differ"
            ),
            f"settings/{s}/lockstep differs",
        )
    ]
    ((arguments, keywords),) = calls.named("compare_task_settings")
    assert arguments == (scripts, settings, tmp_path)
    assert keywords == {"pristine": "basic"}
    assert calls.named("block_positions") == [
        ((scripts["stochastic_depth"], s), {}) for s in settings
    ]
    assert calls.named("trace_main") == [
        (
            (scripts["basic"],),
            {
                "arch": reference.SETTINGS[s].arch,
                "dataset": reference.SETTINGS[s].dataset,
                "classes": reference.SETTINGS[s].classes,
            },
        )
        for s in settings
    ]
    assert [a for a, _ in calls.named("schedule_record")] == [
        (settings[s]["basic"],) for s in settings
    ]
    assert calls.named("upstream_lockstep") == [
        (
            (scripts[n], n, trace),
            {
                "classes": reference.SETTINGS[s].classes,
                "key": residual.lockstep_key(n, setting=s),
            },
        )
        for s in settings
        for n in residual.VARIANTS
    ]
    runs = [k["run"] for _, k in calls.named("compare_lockstep")]
    assert [(r.func, r.keywords) for r in runs if isinstance(r, functools.partial)] == [
        (residual.lockstep_record, {"setting": s}) for s in settings
    ]


def test_branches_are_reported_unless_the_draws_together_fall_on_both_sides() -> None:
    keep = torch.tensor([0.25, 0.375])
    drop = torch.tensor([0.75])
    assert (
        capture_reference.check_branches(
            {
                "block/stochastic_depth/identity/draws": keep,
                "block/stochastic_depth/projected/draws": drop,
            },
        )
        == []
    )
    assert capture_reference.check_branches(
        {
            "block/stochastic_depth/identity/draws": keep,
            "block/basic/identity/draws": drop,
        },
    ) == ["block/stochastic_depth: the draws reach only one branch"]


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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
