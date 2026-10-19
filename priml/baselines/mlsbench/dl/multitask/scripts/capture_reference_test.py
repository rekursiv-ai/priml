"""Tests for the ``cv-multitask-loss`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines'
combinations are not the real ones, so a capture of it must name those and
nothing it shares with the port: the superclass-free schedule and optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import sys

from torch import nn

import torch

from priml.baselines.mlsbench.dl.multitask import (
    reference as port,
    upstream,
)
from priml.baselines.mlsbench.dl.multitask.scripts import capture_reference
from priml.baselines.mlsbench.dl.multitask.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import reference
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    import pytest

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


def passes(fill: float) -> tuple[port.Batches, port.Batches]:
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
    return port.tiny_step(factory, epochs=reference.EPOCHS)


def test_the_full_size_comparison_names_the_stand_ins_other_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stand-in's network is not ResNet-56; its state and forwards differ."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    scripts = upstream.variants(root, scratch)
    monkeypatch.setattr(port, "full_step", shallow)
    trace = upstream.trace_main(scripts["sum"])
    report = capture_reference.compare_full_size(scripts, trace)
    (state,) = (line for line in report if line.startswith("state: "))
    assert state.endswith("tensors against 6")
    for mode in ("train", "eval"):
        assert any(line.startswith(f"forward/{mode}/0: ") for line in report)
        assert any(line.startswith(f"forward/{mode}/1: ") for line in report)


class Builder:
    """Stands in for a script: builds an empty network and notes each ``--arch``."""

    def __init__(self) -> None:
        self.archs: list[str] = []

    def build_model(self, arch: str) -> nn.Module:
        """Note ``arch`` and return a network with no state."""
        self.archs.append(arch)
        return nn.Module()


def setting_run(
    scripts: object,
    setting: str,
) -> tuple[dict[str, upstream.Trace], dict[str, Tensor], dict[str, upstream.Lockstep]]:
    """Stand in for one setting's upstream run: one value, one lockstep."""
    del scripts
    lockstep: upstream.Lockstep = ({}, [{"w": torch.zeros(2)}])
    return (
        {"sum": upstream.Trace()},
        {f"settings/{setting}/x": torch.zeros(2)},
        {f"{setting}/sum": lockstep},
    )


class Walks:
    """Stands in for ``check_walk``: notes each coarse head, reports one line."""

    def __init__(self) -> None:
        self.coarse: list[str] = []

    def __call__(self, pcgrad: object, model: object, **kwargs: object) -> list[str]:
        """Note the coarse head's prefix and report ``walk``."""
        del pcgrad, model
        self.coarse.append(str(kwargs["coarse"]))
        return ["walk"]


def setting_differs(scripts: object, traces: object, setting: str) -> list[str]:
    """Stand in for ``compare_setting``: one line naming the setting."""
    del scripts, traces
    return [f"{setting} state differs"]


def test_capture_settings_joins_every_settings_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runs are tested elsewhere; here they are stubbed, to test the glue."""
    builder = Builder()
    script = cast("upstream.Script", cast(object, builder))
    scripts = {"sum": script, "pcgrad": script}
    walks = Walks()
    monkeypatch.setattr(upstream, "upstream_setting", setting_run)
    monkeypatch.setattr(upstream, "check_walk", walks)
    monkeypatch.setattr(capture_reference, "compare_setting", setting_differs)
    monkeypatch.setattr(
        port,
        "settings_record",
        returning({f"settings/{s}/x": torch.ones(2) for s in port.SETTINGS}),
    )
    monkeypatch.setattr(
        port,
        "setting_lockstep_record",
        LockstepRun({"w": torch.ones(2)}),
    )
    record, report = capture_reference.capture_settings(scripts)
    assert record.keys() == {f"settings/{s}/x" for s in port.SETTINGS}
    assert builder.archs == list(port.SETTINGS.values())
    assert walks.coarse == ["fc_coarse.", "classifier_coarse."]
    per_setting = [
        line
        for s in port.SETTINGS
        for line in (
            f"settings/{s}/main/sum: epoch arguments differ",
            f"{s} state differs",
            f"settings/{s}/walk",
        )
    ]
    assert report == [
        *per_setting,
        *(f"settings/{s}/x: 2/2 differ" for s in sorted(port.SETTINGS)),
        *(f"lockstep/{s}/sum/1: w: 2/2 differ" for s in port.SETTINGS),
    ]


class Calls:
    """Stand-ins that return fixed values and record every call's arguments."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, tuple[object, ...]]] = []

    def returning[T](self, name: str, value: T) -> Callable[..., T]:
        """Return a stand-in named ``name`` that records its arguments."""

        def call(*arguments: object) -> T:
            self.seen.append((name, arguments))
            return value

        return call


def test_capture_joins_both_sides_and_every_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runs are tested elsewhere; here they are stubbed, to test the glue."""
    calls = Calls()
    trace = upstream.Trace(epochs=[(e, 200) for e in range(200)])
    no_batches: tuple[port.Batches, port.Batches] = ([], [])
    root, scratch = tmp_path / "root", tmp_path / "scratch"
    sum_script, dwa_script = object(), object()
    scripts = {"sum": sum_script, "dwa": dwa_script}
    lockstep = ({"lockstep/x": torch.zeros(2)}, [{"w": torch.zeros(2)}])
    for module, name, value in (
        (upstream, "variants", scripts),
        (upstream, "trace_main", trace),
        (upstream, "upstream_lockstep", lockstep),
        (upstream, "upstream_record", {"theirs": torch.ones(2)}),
        (upstream, "upstream_batches", no_batches),
        (capture_reference, "compare_full_size", ["full size differs"]),
        (port, "port_record", {"theirs": torch.zeros(2)}),
        (port, "loader_batches", no_batches),
        (
            capture_reference,
            "capture_settings",
            ({"settings/x": torch.zeros(3)}, ["setting differs"]),
        ),
    ):
        monkeypatch.setattr(module, name, calls.returning(name, value))
    monkeypatch.setattr(port, "lockstep_record", LockstepRun({"w": torch.ones(2)}))
    captured = capture_reference.capture(root, scratch)
    assert captured.record.keys() == {"theirs", "settings/x"}
    assert captured.sites == {}
    assert captured.report == [
        "full size differs",
        "theirs: 2/2 differ",
        "lockstep/sum/1: w: 2/2 differ",
        "lockstep/dwa/1: w: 2/2 differ",
        "setting differs",
    ]
    traces = {"sum": trace, "dwa": trace}
    assert calls.seen == [
        ("variants", (root, scratch)),
        ("trace_main", (sum_script,)),
        ("trace_main", (dwa_script,)),
        ("upstream_lockstep", (sum_script, "sum", trace)),
        ("upstream_lockstep", (dwa_script, "dwa", trace)),
        (
            "upstream_record",
            (scripts, traces, {"sum": lockstep, "dwa": lockstep}),
        ),
        ("upstream_batches", (sum_script,)),
        ("compare_full_size", (scripts, trace)),
        ("port_record", ()),
        ("loader_batches", (scratch,)),
        ("capture_settings", (scripts,)),
    ]


class LockstepRun:
    """A port lockstep run that hands back ``state`` after one update."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = state

    def __call__(
        self,
        name: str,
        *,
        states: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Append the one state; return no values."""
        del name
        states.append(self.state)
        return {}


def test_compare_setting_names_each_combinations_state_that_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(port, "full_step", shallow)
    setting = next(iter(port.SETTINGS))
    combinations = port.SETTING_COMBINATIONS[setting]
    states: dict[str, dict[str, Tensor]] = {}
    for name, factory in combinations.items():
        with reference.seeded(reference.SEED), host_agnostic_numerics():
            states[name] = shallow(factory).model.state_dict()
    moved = {k: v + 1 if v.is_floating_point() else v for k, v in states["dwa"].items()}
    traces = {name: upstream.Trace(state=s) for name, s in states.items()}
    traces["dwa"] = upstream.Trace(state=moved)
    scripts = dict.fromkeys(combinations, cast("upstream.Script", object()))
    report = capture_reference.compare_setting(scripts, traces, setting)
    assert report
    assert all(line.startswith(f"settings/{setting}/dwa/state: ") for line in report)


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
