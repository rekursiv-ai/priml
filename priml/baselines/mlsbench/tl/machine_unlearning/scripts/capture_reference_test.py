"""Tests for the ``security-machine-unlearning`` capture.

The stand-in clone's harness trains another network and its edits put another
method in place, so a capture of it must report those differences and none it
shares with the port: the recipe and the split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sys

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.vision.reference import Captured
from priml.baselines.mlsbench.tl.machine_unlearning import reference, upstream
from priml.baselines.mlsbench.tl.machine_unlearning.reference import (
    MainTrace,
    Step,
)
from priml.baselines.mlsbench.tl.machine_unlearning.scripts import (
    capture_reference,
)
from priml.baselines.mlsbench.tl.machine_unlearning.upstream_test import (
    write_clone,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


class FakePort:
    """Stands in for the port's runs, keeping every argument it is given."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def method_lockstep(
        self,
        name: str,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Record the variant; return no states."""
        self.calls.append(("lockstep", name))
        return {}, []

    def port_main(self, name: str, directory: Path) -> MainTrace:
        """Record the variant and directory; return a run with one port site."""
        self.calls.append(("main", (name, directory)))
        return MainTrace(sites={"port:site": 1})

    def port_record(
        self,
        directory: Path,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Record the directory; return no values and no full networks."""
        self.calls.append(("record", directory))
        return {}, {key: {} for key in reference.SETTINGS}


def drawing_main(
    script: upstream.Driver,
    module: upstream.MethodModule,
) -> tuple[MainTrace, bool]:
    """Stand in for ``main``: one step, a teacher and a reference site per edited method.

    ``upstream_test`` runs the harness's ``main``; here only the glue is tested.
    """
    del script
    edited = hasattr(module.UnlearningMethod(), "teacher")
    batches = (torch.zeros(1),) * 4
    trace = MainTrace(
        initial=state(0),
        pretrained=state(0),
        steps=[Step(loss=0.0, state=state(0), batches=batches)],
        teachers={"teacher": state(0)} if edited else {},
        sites={"randperm@harness:f": 1},
    )
    return trace, not edited


def method_lockstep(
    script: upstream.Driver,
    module: upstream.MethodModule,
    name: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Stand in for the harness's lockstep: one probe, a state per update and teacher."""
    del script
    teachers = int(hasattr(module.UnlearningMethod(), "teacher"))
    states = [state(0)] * (len(reference.SCHEDULE) + teachers)
    return {f"methods/{name}/probe": torch.zeros(2)}, states


def test_capture_runs_every_harness_comparison_against_the_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    port = FakePort()
    monkeypatch.setattr(reference, "method_lockstep", port.method_lockstep)
    monkeypatch.setattr(reference, "port_main", port.port_main)
    monkeypatch.setattr(reference, "port_record", port.port_record)
    monkeypatch.setattr(upstream, "upstream_main", drawing_main)
    monkeypatch.setattr(upstream, "upstream_method_lockstep", method_lockstep)
    forwards: list[str] = []

    def compare_full_forward(script: object, key: str) -> list[str]:
        del script
        forwards.append(key)
        return [f"full/{key}/train: differs"]

    monkeypatch.setattr(capture_reference, "compare_full_forward", compare_full_forward)
    captured = capture_reference.capture(root, scratch)
    assert forwards == list(reference.SETTINGS)
    assert [line for line in captured.report if line.endswith("/train: differs")] == [
        f"full/{key}/train: differs" for key in reference.SETTINGS
    ]
    variants = list(reference.VARIANTS)
    assert port.calls == [
        *(("lockstep", name) for name in variants),
        *(("main", (name, scratch)) for name in variants),
        ("record", scratch),
    ]
    assert {k.split("/")[0] for k in captured.record} == {
        "recipe",
        "split",
        "evaluation",
        "full",
        "settings",
        "methods",
        "main",
    }
    for name in variants:
        assert f"methods/{name}/probe" in captured.record
        assert f"main/{name}/loss" in captured.record
        assert captured.sites[f"port-{name}"] == {"port:site": 1}
        assert captured.sites[f"reference-{name}"] == {"randperm@harness:f": 1}
    report = captured.report
    drawn = [line for line in report if "generator was drawn" in line]
    # Every edit of the stand-in holds a teacher; only the pristine method does not.
    assert drawn == [
        f"main/{name}: Python's or NumPy's generator was drawn" for name in variants[1:]
    ]
    assert (
        f"main/pristine: port random call site {capture_reference.REQUIRED_PORT[0]}"
        " was never reached"
    ) in report
    # The stand-in's edits hold one teacher, recorded after the student's states.
    states = {"pristine": len(reference.SCHEDULE)} | {
        name: len(reference.SCHEDULE) + 1 for name in variants[1:]
    }
    for name in variants:
        assert f"methods/{name}: 0 states against {states[name]}" in report
        assert f"main/{name}: 0 steps against 1" in report
    assert any(
        line.startswith("full/resnet20-cifar10-class0/state: ") for line in report
    )
    assert "unexpected recipe/values" not in report
    assert any(line.startswith("missing recipe/values") for line in report)


def state(value: float) -> dict[str, torch.Tensor]:
    """Return a one-tensor state holding ``value``."""
    return {"w": torch.tensor([value])}


def test_the_full_forward_compares_each_sides_full_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    ours, theirs = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
    full = {"ours": ours, "theirs": theirs}

    def upstream_network(script: object, key: str) -> tuple[torch.nn.Module, Tensor]:
        del script
        assert key == "resnet20-cifar10-class0"
        return full["theirs"], torch.empty(0)

    def port_network(key: str) -> tuple[torch.nn.Module, Tensor]:
        assert key == "resnet20-cifar10-class0"
        return full["ours"], torch.empty(0)

    def flattened(
        model: torch.nn.Module,
        draws: Tensor,
        *,
        key: str,
    ) -> dict[str, Tensor]:
        del draws
        weight = cast(Tensor, model.get_parameter("weight"))
        return {f"full/{key}/train": weight.detach()}

    monkeypatch.setattr(upstream, "upstream_full_network", upstream_network)
    monkeypatch.setattr(reference, "full_network", port_network)
    monkeypatch.setattr(reference, "full_size_values", flattened)
    script = cast("upstream.Driver", object())
    key = "resnet20-cifar10-class0"
    assert capture_reference.compare_full_forward(script, key) == [
        f"full/{key}/train: 12/12 differ",
    ]
    full["theirs"] = ours
    assert capture_reference.compare_full_forward(script, key) == []


def test_compare_lockstep_reports_counts_and_tensors() -> None:
    assert capture_reference.compare_lockstep("m", [state(1)], []) == [
        "methods/m: 0 states against 1",
    ]
    assert capture_reference.compare_lockstep("m", [state(1)], [state(2)]) == [
        "methods/m/1: w: 1/1 differ",
    ]


def trace(value: float, *, teachers: tuple[str, ...] = ()) -> MainTrace:
    """Return a trace of one step whose every tensor holds ``value``."""
    batches = (torch.tensor([value]),) * 4
    return MainTrace(
        initial=state(value),
        pretrained=state(value),
        steps=[Step(loss=value, state=state(value), batches=batches)],
        teachers={name: state(value) for name in teachers},
    )


def test_compare_main_reports_every_state_batch_and_teacher() -> None:
    assert capture_reference.compare_main("v", trace(1), trace(1)) == []
    report = capture_reference.compare_main(
        "v",
        trace(1, teachers=("teacher",)),
        trace(2, teachers=("teacher",)),
    )
    assert report == [
        "main/v/initial: w: 1/1 differ",
        "main/v/pretrained: w: 1/1 differ",
        "main/v/step0: fx: 1/1 differ",
        "main/v/step0: fy: 1/1 differ",
        "main/v/step0: rx: 1/1 differ",
        "main/v/step0: ry: 1/1 differ",
        "main/v/step0: w: 1/1 differ",
        "main/v/teacher: w: 1/1 differ",
    ]
    short = MainTrace(initial=state(1), pretrained=state(1))
    assert capture_reference.compare_main("v", trace(1), short)[-1] == (
        "main/v: 0 steps against 1"
    )
    assert capture_reference.compare_main(
        "v",
        trace(1, teachers=("a",)),
        trace(1),
    ) == ["main/v: teachers [] against ['a']"]


def test_check_sites_names_each_unreached_site() -> None:
    reached = {f"{site}:f": 1 for site in capture_reference.REQUIRED_PORT}
    upstream = {f"{site}:f": 1 for site in capture_reference.REQUIRED_UPSTREAM}
    assert capture_reference.check_sites("v", upstream, reached) == []
    missing = capture_reference.check_sites("v", {}, reached)
    assert len(missing) == len(capture_reference.REQUIRED_UPSTREAM)
    assert missing[0].startswith("main/v: reference random call site")
    site = capture_reference.REQUIRED_PORT[0]
    unreached = reached | {f"{site}:f": 0, "elsewhere:g": 3}
    assert capture_reference.check_sites("v", upstream, unreached) == [
        f"main/v: port random call site {site} was never reached",
    ]


def test_main_prints_the_report_and_writes_nothing_when_something_differs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = Captured(record={}, sites={}, report=["a differs"])
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
    result = Captured(record={"x": torch.arange(3)}, sites={}, report=[])
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
