"""Tests for the ``security-membership-inference-defense`` capture's glue and checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import torch

from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.tl.membership_inference_defense import (
    reference as port,
    upstream,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.scripts import (
    capture_reference,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.upstream_test import (
    DATA,
    scripts,
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


def returning_any[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments, keywords too, and gives back ``value``."""

    def call(*_arguments: object, **_keywords: object) -> T:
        return value

    return call


def test_capture_assembles_both_sides_into_one_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every run is a stand-in; the runs themselves are tested elsewhere."""
    variants = {"pristine": object(), "relaxloss": object()}
    zero = {"w": torch.zeros(1)}
    lockstep = ({"lockstep/x/loss": torch.zeros(2)}, [zero] * port.LOCKSTEP_EPOCHS)
    stand_ins: dict[tuple[object, str], object] = {
        (upstream, "variants"): returning_any(variants),
        (upstream, "trace_main"): returning_any(upstream.Trace()),
        (upstream, "initial_main"): returning_any((upstream.Trace(), None)),
        (upstream, "upstream_lockstep"): returning_any(lockstep),
        (capture_reference, "upstream_record"): returning_any(
            {"defense/x": torch.zeros(2)},
        ),
        (capture_reference, "compare_recipe"): returning_any(["recipe: differs"]),
        (capture_reference, "compare_initial"): returning_any(["initial: differs"]),
        (capture_reference, "upstream_pipelines"): returning_any(
            {"pipeline/x": torch.zeros(2)},
        ),
        (capture_reference, "upstream_sites"): returning_any({"a": 1}),
        (capture_reference, "port_sites"): returning_any({"b": 2}),
        (capture_reference, "check_sites"): returning_any(["sites: unreached"]),
        (vision_upstream, "tiny_setting_record"): returning_any(
            {"settings/x/train": torch.zeros(2)},
        ),
        (port, "pipeline_record"): returning_any({"pipeline/x": torch.ones(2)}),
        (port, "port_record"): returning_any({"defense/x": torch.ones(2)}),
        (port, "lockstep_record"): _states_of_zeros,
    }
    for (module, name), value in stand_ins.items():
        monkeypatch.setattr(module, name, value)
    captured = capture_reference.capture(tmp_path, tmp_path)
    assert set(captured.record) == {
        "defense/x",
        "settings/x/train",
        "lockstep/x/loss",
    }
    assert captured.sites == {"reference": {"a": 1}, "port": {"b": 2}}
    assert captured.report == [
        "recipe: differs",
        *["initial: differs"] * len(port.SETTINGS),
        "pipeline/x: 2/2 differ",
        "missing lockstep/x/loss",
        "missing settings/x/train",
        "defense/x: 2/2 differ",
        "sites: unreached",
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


def test_main_writes_the_golden_and_the_sites(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = reference.Captured(
        record={"x": torch.arange(3)},
        sites={"port": {"a@b.py:c": 2}},
        report=[],
    )
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
    assert (out / "reference_sites.txt").read_text() == "port a@b.py:c 2\n"


def test_the_recipe_check_names_each_way_main_could_differ() -> None:
    trace = upstream.Trace()
    trace.epochs = list(range(port.EPOCHS))
    trace.loaders = [
        {"shuffle": shuffle, "batch_size": 128}
        for shuffle in (True, False, False, False)
    ]
    assert capture_reference.compare_recipe(trace) == []
    trace.epochs = [*trace.epochs[1:], 0]
    trace.loaders[1] = {"shuffle": True, "batch_size": 64}
    epochs, shuffles, sizes = capture_reference.compare_recipe(trace)
    assert epochs == "recipe: the defense was told other epochs"
    assert shuffles == "recipe: loaders shuffle as [True, True, False, False]"
    assert sizes.startswith("recipe: loaders batch {")


def test_a_call_site_a_side_never_reached_is_reported() -> None:
    port: dict[str, int] = dict.fromkeys(capture_reference.REQUIRED_PORT, 1)
    theirs: dict[str, int] = dict.fromkeys(capture_reference.REQUIRED_UPSTREAM, 1)
    assert capture_reference.check_sites(theirs, port) == []
    port["randperm@vision/data.py"] = 0
    assert capture_reference.check_sites(theirs, port) == [
        "port: random call site randperm@vision/data.py was never reached",
    ]


def test_the_upstream_record_holds_every_fixed_input_key_the_golden_does(
    tmp_path: Path,
) -> None:
    variants = scripts(tmp_path)
    # The recipe's trace is tested in ``upstream_test``; its 300 epochs are slow.
    recipe = upstream.Trace(
        rates=[0.1] * port.EPOCHS,
        settings=dict.fromkeys(reference.HYPERPARAMETERS, 0.0),
        subsets=[[0, 1], [0, 1], [2, 3]],
    )
    record = capture_reference.upstream_record(variants, recipe)
    stored = read_tensors(DATA / "reference.pt")
    fixed = {
        k
        for k in stored
        if not k.startswith(("lockstep/", "settings/", "draws/", "network/"))
    }
    assert set(record) == fixed
    pipelines = capture_reference.upstream_pipelines(variants["pristine"])
    assert set(pipelines) == set(port.pipeline_record())


def test_compare_initial_names_a_split_or_network_that_differs(
    tmp_path: Path,
) -> None:
    script = scripts(tmp_path)["pristine"]
    key = next(iter(port.SETTINGS))
    setting, factory = port.SETTINGS[key]
    trace, _ = upstream.initial_main(
        script,
        setting,
        count=capture_reference.TRAINING_IMAGES,
        build=upstream.full_model,
    )
    report = capture_reference.compare_initial(trace, key=key, factory=factory)
    assert not [line for line in report if line.startswith("split/")]
    assert report
    assert all(line.startswith(f"settings/{key}/state: ") for line in report)


def test_each_sides_draws_are_counted_over_one_lockstep(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    theirs = capture_reference.upstream_sites(script)
    ours = capture_reference.port_sites(tmp_path)
    assert capture_reference.check_sites(theirs, ours) == []


def _states_of_zeros(
    name: str,
    directory: Path,
    *,
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stand in for a port lockstep run: one zero state per update."""
    del name, directory
    states.extend({"w": torch.zeros(1)} for _ in range(port.LOCKSTEP_EPOCHS))
    return {}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
