"""Tests for the ``security-poison-robust-learning`` capture, against a stand-in clone.

The stand-in's network, pipelines and three of its losses differ from the real
driver's, so a capture of it must name those, and not the label flip, the
evaluations, the optimizer or the pristine loss it shares with the port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sys
import types

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.tl.poison_robust_learning import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.scripts import (
    capture_reference,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.upstream_test import (
    DATA,
    scripts,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


class _Calls:
    """Records each call's arguments and returns the next of ``results``, the last again."""

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


def returning_any[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments, keywords too, and gives back ``value``."""

    def call(*_arguments: object, **_keywords: object) -> T:
        return value

    return call


def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every run is a stand-in; the runs themselves are tested elsewhere."""
    variants = {
        name: cast(upstream.Script, types.SimpleNamespace(name=name))
        for name in ("pristine", "bootstrap")
    }
    zero = {"w": torch.zeros(2)}
    model = torch.nn.Linear(3, 2)
    initial = upstream.Trace(draws=torch.arange(2), models=[model])
    recipe = upstream.Trace(epochs=[0])
    traces = _Calls(recipe, *[initial] * len(reference.SETTINGS))
    lockstep = _Calls(({"lockstep/x/loss": torch.zeros(2)}, [zero, zero]))
    stand_ins: dict[tuple[object, str], object] = {
        (upstream, "variants"): returning(variants),
        (upstream, "trace_main"): traces,
        (capture_reference, "upstream_record"): returning({"loss/x": torch.zeros(2)}),
        (capture_reference, "compare_recipe"): returning(["recipe: differs"]),
        (vision_upstream, "tiny_setting_record"): returning_any(
            {"settings/x/train": torch.ones(2)},
        ),
        (capture_reference, "compare_initial"): returning_any(["initial: differs"]),
        (upstream, "upstream_lockstep"): lockstep,
        (reference, "port_record"): returning(
            {"loss/x": torch.ones(2)},
        ),
        (reference, "lockstep_record"): stand_in_lockstep,
        (capture_reference, "compare_pristine_run"): returning(["pixels: differ"]),
        (capture_reference, "upstream_sites"): returning({"a": 1}),
        (capture_reference, "port_sites"): returning({"b": 2}),
        (capture_reference, "check_sites"): returning(["sites: unreached"]),
    }
    for (module, name), value in stand_ins.items():
        monkeypatch.setattr(module, name, value)
    captured = capture_reference.capture(tmp_path / "clone", tmp_path)
    assert set(captured.record) == {"loss/x", "lockstep/x/loss", "settings/x/train"}
    assert captured.sites == {"reference": {"a": 1}, "port": {"b": 2}}
    # ``mismatches`` names the stand-in port's missing keys first, sorted.
    assert captured.report == [
        "recipe: differs",
        *["initial: differs"] * len(reference.SETTINGS),
        *(f"missing {key}" for key in ("lockstep/x/loss", "settings/x/train")),
        "loss/x: 2/2 differ",
        *(
            f"lockstep/{name}/{step}: w: 2/2 differ"
            for name in variants
            for step in (1, 2)
        ),
        "pixels: differ",
        "sites: unreached",
    ]
    # The recipe runs the linear model serially; each setting stops at its scheduler.
    (_, first), *settings = traces.calls
    assert first == {
        "build": upstream.linear_model,
        "split": first["split"],
        "serial": True,
    }
    assert [kwargs["stop"] for _, kwargs in settings] == [True] * len(
        reference.SETTINGS,
    )
    assert [args[1] for args, _ in settings] == [
        upstream.recipe_argv(setting) for setting, _ in reference.SETTINGS.values()
    ]
    assert [args[1] for args, _ in lockstep.calls] == list(variants)


def stand_in_lockstep(
    name: str,
    directory: Path,
    *,
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Return a two-update run whose states are all ones."""
    del directory
    states += [{"w": torch.ones(2)}] * 2
    return {f"lockstep/{name}/loss": torch.ones(2)}


@pytest.fixture(name="stand_in")
def fixture_stand_in(tmp_path: Path) -> dict[str, upstream.Script]:
    """Import every variant of the stand-in driver, its loaders worker-free."""
    return scripts(tmp_path)


def test_the_upstream_record_holds_every_fixed_input_quantity(
    stand_in: dict[str, upstream.Script],
) -> None:
    recipe = upstream.Trace(
        rates=[0.1, 0.05],
        settings=dict.fromkeys(vision_reference.HYPERPARAMETERS, 0.5),
    )
    record = capture_reference.upstream_record(stand_in, recipe)
    flip = reference.flip_record()
    assert mismatches({k: record[k] for k in flip}, flip) == []
    assert torch.equal(record["flip/items"], flip["flip/labels"])
    # The stand-in's evaluations are the driver's, so they score as the port does.
    assert (
        mismatches(
            {k: record[k] for k in ("evaluate/test_acc", "evaluate/poison_fit")},
            reference.evaluate_record(),
        )
        == []
    )
    assert record["recipe/lr"].tolist() == [0.1, 0.05]
    assert record["recipe/optimizer"].tolist() == [0.5] * len(
        vision_reference.HYPERPARAMETERS,
    )
    assert {k for k in record if k.startswith("loss/")} == {
        f"loss/{name}/{case}/{kind}"
        for name in stand_in
        for case in reference.loss_cases()
        for kind in ("value", "grad")
    }


def loader_kwargs(*, shuffle: bool, workers: int, batch: int) -> dict[str, object]:
    """Return the keyword arguments a recorded loader was built with."""
    return {"shuffle": shuffle, "num_workers": workers, "batch_size": batch}


def test_a_recipe_run_like_the_ports_reports_nothing() -> None:
    trace = upstream.Trace(
        epochs=list(range(reference.EPOCHS)),
        t_max=reference.EPOCHS,
        loaders=[
            loader_kwargs(shuffle=shuffle, workers=4, batch=128)
            for shuffle in (True, False, False)
        ],
    )
    assert capture_reference.compare_recipe(trace) == []


def test_every_way_a_recipe_run_can_differ_is_named() -> None:
    trace = upstream.Trace(
        epochs=[0],
        t_max=3,
        loaders=[loader_kwargs(shuffle=True, workers=2, batch=64)],
    )
    report = capture_reference.compare_recipe(trace)
    assert report == [
        "recipe: the loss was told other epochs",
        "recipe: the schedule's horizon is 3",
        "recipe: loaders shuffle as [True]",
        "recipe: loaders run [2] workers",
        "recipe: loaders batch {64}",
    ]


def test_the_initial_run_compares_the_flip_and_the_state(
    stand_in: dict[str, upstream.Script],
) -> None:
    setting, factory = reference.SETTINGS["exp000"]
    trace = upstream.trace_main(
        stand_in["pristine"],
        upstream.recipe_argv(setting),
        build=upstream.full_model,
        split=(torch.zeros(20, 3, 2, 4, dtype=torch.uint8), torch.arange(20) * 7 % 10),
        serial=True,
        stop=True,
    )
    report = capture_reference.compare_initial(trace, key="exp000", factory=factory)
    # The stand-in flips as the driver does but builds a two-layer network.
    ours = len(vision_reference.full_step(factory).model.state_dict())
    assert report == [f"settings/exp000/state: state: {ours} tensors against 4"]


def test_the_pristine_run_compares_pixels_and_generators(
    stand_in: dict[str, upstream.Script],
    tmp_path: Path,
) -> None:
    report = capture_reference.compare_pristine_run(stand_in["pristine"], tmp_path)
    # The stand-in crops but never mirrors, so some pixels differ; it seeds
    # Python's and NumPy's generators and never draws from them.
    assert report
    assert all(line.startswith("evaluate/media/") for line in report)


def test_a_pristine_run_that_never_finished_is_refused(
    stand_in: dict[str, upstream.Script],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A run ended before ``main`` returned records no generator states.
    monkeypatch.setattr(upstream, "trace_main", returning_any(upstream.Trace()))
    with pytest.raises(ValueError, match="Expected after is not None"):
        capture_reference.compare_pristine_run(stand_in["pristine"], tmp_path)


def test_a_pristine_run_that_draws_from_python_and_numpy_is_named(
    stand_in: dict[str, upstream.Script],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = upstream.seeded_generator_states()
    moved = upstream.Generators(python=(0,), numpy=("MT19937", b"", 0, 0, 0.0))
    monkeypatch.setattr(upstream, "seeded_generator_states", returning(moved))
    report = capture_reference.compare_pristine_run(stand_in["pristine"], tmp_path)
    assert report[-2:] == [
        "generators: python random was drawn from",
        "generators: numpy random was drawn from",
    ]
    assert seeded != moved


def test_both_sides_count_their_draws_over_one_lockstep_run(
    stand_in: dict[str, upstream.Script],
    tmp_path: Path,
) -> None:
    theirs = capture_reference.upstream_sites(stand_in["pristine"])
    ours = capture_reference.port_sites(tmp_path)
    assert theirs["randperm@poison/run_poison_robust.py:__init__"] == 1
    assert ours["randperm@poison_robust_learning/data.py:flipped_labels"] == 1
    assert capture_reference.check_sites(theirs, ours) == []


def test_main_prints_the_report_and_writes_nothing_when_something_differs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = vision_reference.Captured(record={}, sites={}, report=["a differs"])
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
    result = vision_reference.Captured(
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


def test_the_port_lockstep_run_hands_back_each_update(tmp_path: Path) -> None:
    states: list[dict[str, torch.Tensor]] = []
    record = capture_reference.PortLockstep(tmp_path)("bootstrap", states=states)
    # The whole split is one batch: one update in each of two epochs.
    assert len(states) == len(record["lockstep/bootstrap/loss"]) == 2
    key = "lockstep/bootstrap/loss"
    assert torch.equal(record[key], read_tensors(DATA / "reference.pt")[key])


def test_a_call_site_a_side_never_reached_is_reported() -> None:
    port: dict[str, int] = dict.fromkeys(capture_reference.REQUIRED_PORT, 1)
    theirs: dict[str, int] = dict.fromkeys(capture_reference.REQUIRED_UPSTREAM, 1)
    assert capture_reference.check_sites(theirs, port) == []
    site = "randperm@poison_robust_learning/data.py"
    port[site] = 0
    assert capture_reference.check_sites(theirs, port) == [
        f"port: random call site {site} was never reached",
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
