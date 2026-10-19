"""Tests for the ``cv-sample-weighting`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-32 and its baselines' weightings are
not the real ones, so a capture of it must name those and nothing it shares
with the port: the long-tail cut, the uniform weights, the schedule and the
optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import sys

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, normalize
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.weighting import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.dl.weighting.scripts import capture_reference
from priml.baselines.mlsbench.dl.weighting.upstream_test import (
    write_clone,
)
from priml.testing.golden import read_tensors, rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
    from priml.baselines.mlsbench.dl.vision.upstream import Batches, Lockstep


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD.parent / "testdata"


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object, **_keywords: object) -> T:
        return value

    return call


def passes(order: list[int]) -> tuple[Batches, Batches]:
    """Return one training and one test pass of the split's images in ``order``."""
    config = Cifar100Data.Config()
    media, label = reference.long_tail_split()
    codes = normalize(
        media,
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    batch = {"media": codes[order], "label": label[order]}
    return [[batch]], [[batch]]


def port_lockstep(
    name: str,
    *,
    states: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Stand in for the port's lockstep: one tensor per state."""
    states += [{"w": torch.zeros(2)} for _ in range(vision_reference.STEPS)]
    return {f"lockstep/{name}/loss": torch.zeros(2)}


def shallow(factory: Callable[[], VisionTrainLoop]) -> Callable[[], VisionTrainLoop]:
    """Return ``factory`` with one block per stage, in place of ResNet-32."""

    def build() -> VisionTrainLoop:
        config = factory()
        assert isinstance(config.step.model, ResNet.Config)
        config.step.model.blocks_per_stage = (1, 1, 1)
        return config

    return build


@pytest.mark.parametrize("cifar100", [None, Path("cifar")])
def test_capture_assembles_both_sides_into_one_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cifar100: Path | None,
) -> None:
    """The port-side runs are tested in ``reference_test``; here they are stand-ins."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stored = read_tensors(_DATA / "reference.pt")
    real = {
        "real/indices": torch.arange(2),
        "real/counts": torch.ones(2),
        "label": torch.tensor([0, 1, 1]),
    }
    monkeypatch.setattr(reference, "port_record", returning(stored))
    monkeypatch.setattr(reference, "lockstep_record", port_lockstep)
    monkeypatch.setattr(
        reference,
        "loader_batches",
        returning(passes([1, 0])),
    )
    full_step = vision_reference.full_step

    def shallow_step(factory: Callable[[], VisionTrainLoop]) -> VisionTrainStep:
        return full_step(shallow(factory))

    monkeypatch.setattr(vision_reference, "full_step", shallow_step)
    monkeypatch.setattr(upstream, "upstream_batches", returning(passes([0, 1])))
    monkeypatch.setattr(
        upstream,
        "real_subset",
        returning((real["real/indices"], real["real/counts"], real["label"])),
    )
    # The settings are joined in ``test_capture_joins_both_sides_and_every_setting``.
    no_settings: tuple[dict[str, Tensor], list[str]] = ({}, [])
    monkeypatch.setattr(capture_reference, "capture_settings", returning(no_settings))
    report = capture_reference.capture(root, scratch, cifar100=cifar100).report
    keys = {line.split(":")[0] for line in report}
    assert {"weights/inverse_freq", "draws/exp000", "state"} <= keys
    assert not {"main/indices", "schedule/lr", "optimizer/settings"} & keys
    assert any(line.startswith("forward/train: ") for line in report)
    assert any(line.startswith("forward/eval: ") for line in report)
    for name in ("uniform", *upstream.EDITED):
        assert any(line.startswith(f"lockstep/{name}/1: state: ") for line in report)
    assert any(line.startswith("loader/train/0: 0/media: ") for line in report)
    assert any(line.startswith("loader/test/0: 0/media: ") for line in report)
    # The faked cut's two counts and positions are not the port's cut of its labels.
    assert ({"real/indices", "real/counts"} <= keys) == (cifar100 is not None)


class EmptyStep:
    """Stands in for a full step: a network with no state."""

    model: Final = torch.nn.Module()


def test_capture_settings_joins_every_settings_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runs are tested elsewhere; here they are stubbed, to test the glue."""
    trace = upstream.Trace(indices=torch.arange(2))
    trace.main.state = {"w": torch.zeros(2)}
    with vision_reference.seeded(vision_reference.SEED):
        trace.main.draws = rng_fingerprint()
    no_batches: tuple[Batches, Batches] = ([], [])

    def setting_run(
        scripts: object,
        name: str,
    ) -> tuple[dict[str, upstream.Trace], dict[str, Tensor], dict[str, Lockstep]]:
        del scripts
        lockstep: Lockstep = ({}, [{"w": torch.zeros(2)}])
        return (
            {"uniform": trace},
            {f"settings/{name}/x": torch.zeros(2)},
            {f"{name}/uniform": lockstep} if reference.SETTINGS[name].lockstep else {},
        )

    no_counts: dict[str, Tensor] = {}
    for module, name, value in (
        (upstream, "upstream_setting", setting_run),
        (
            reference,
            "setting_counts_record",
            returning((no_counts, torch.arange(3))),
        ),
        (vision_reference, "full_step", returning(EmptyStep())),
        (upstream, "upstream_batches", returning(no_batches)),
        (reference, "loader_batches", returning(no_batches)),
        (
            reference,
            "decode_indices",
            returning({"loader/index": torch.arange(2)}),
        ),
        (
            reference,
            "settings_record",
            returning(
                {f"settings/{s}/x": torch.ones(2) for s in reference.SETTINGS},
            ),
        ),
        (
            reference,
            "setting_lockstep_record",
            lockstep_run({"w": torch.ones(2)}),
        ),
    ):
        monkeypatch.setattr(module, name, value)
    scripts = {"uniform": cast("upstream.Script", object())}
    record, report = capture_reference.capture_settings(scripts, tmp_path)
    loader = "settings/resnet32-cifar10lt/loader/index"
    assert record.keys() == {loader} | {f"settings/{s}/x" for s in reference.SETTINGS}
    per_setting = [
        line
        for s in reference.SETTINGS
        for line in (
            f"settings/{s}/uniform/state: state: 0 tensors against 1",
            f"settings/{s}/uniform/indices: torch.int64[3] vs torch.int64[2]",
        )
    ]
    assert report == [
        *per_setting,
        *(f"settings/{s}/x: 2/2 differ" for s in sorted(reference.SETTINGS)),
        "lockstep/vgg16bn-cifar100lt/uniform/1: w: 2/2 differ",
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


def lockstep_run(
    state: dict[str, torch.Tensor],
) -> Callable[..., dict[str, torch.Tensor]]:
    """Return a port lockstep run that hands back ``state`` after one update."""

    def run(
        name: str,
        *,
        states: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        del name
        states.append(state)
        return {}

    return run


@pytest.mark.parametrize("real", [False, True], ids=["no-real", "real"])
def test_capture_joins_both_sides_and_every_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    real: bool,
) -> None:
    """The runs are tested elsewhere; here they are stubbed, to test the glue."""
    calls = Calls()
    trace = upstream.Trace(indices=torch.arange(3))
    train: Batches = [[{"label": torch.zeros(2)}]]
    passes: tuple[Batches, Batches] = (train, [])
    ours: tuple[Batches, Batches] = ([[{"label": torch.ones(2)}]], [])
    root, scratch = tmp_path / "root", tmp_path / "scratch"
    cifar100 = tmp_path / "cifar" if real else None
    uniform, other = object(), object()
    scripts = {"uniform": uniform, "other": other}
    lockstep = ({"lockstep/x": torch.zeros(2)}, [{"w": torch.zeros(2)}])
    weights = {"weights": torch.zeros(1)}
    subset = {"subset": torch.zeros(1)}
    no_counts: dict[str, torch.Tensor] = {}
    for module, name, value in (
        (upstream, "variants", scripts),
        (upstream, "trace_main", trace),
        (upstream, "upstream_lockstep", lockstep),
        (upstream, "upstream_batches", passes),
        (upstream, "upstream_weights", weights),
        (upstream, "upstream_subset", subset),
        (upstream, "upstream_record", {"theirs": torch.ones(2)}),
        (capture_reference, "compare_full_size", ["full size differs"]),
        (reference, "counts_record", (no_counts, torch.arange(2))),
        (reference, "port_record", {"theirs": torch.zeros(2)}),
        (reference, "loader_batches", ours),
        (capture_reference, "compare_real", ["real differs"]),
        (
            capture_reference,
            "capture_settings",
            ({"settings/x": torch.zeros(3)}, ["setting differs"]),
        ),
    ):
        monkeypatch.setattr(module, name, calls.returning(name, value))
    monkeypatch.setattr(
        reference,
        "lockstep_record",
        lockstep_run({"w": torch.ones(2)}),
    )
    captured = capture_reference.capture(root, scratch, cifar100=cifar100)
    assert captured.record.keys() == {"theirs", "settings/x"}
    assert captured.sites == {}
    assert captured.report == [
        "full size differs",
        "theirs: 2/2 differ",
        "main/indices: torch.int64[2] vs torch.int64[3]",
        "lockstep/uniform/1: w: 2/2 differ",
        "lockstep/other/1: w: 2/2 differ",
        "loader/train/0: 0/label: 2/2 differ",
        *(["real differs"] if real else []),
        "setting differs",
    ]
    assert calls.seen == [
        ("variants", (root, scratch)),
        ("trace_main", (uniform,)),
        ("upstream_lockstep", (uniform, "uniform", trace)),
        ("upstream_lockstep", (other, "other", trace)),
        ("upstream_batches", (uniform,)),
        ("upstream_weights", (scripts,)),
        ("upstream_subset", (uniform,)),
        (
            "upstream_record",
            (trace, weights, {"uniform": lockstep, "other": lockstep}, subset, train),
        ),
        ("compare_full_size", (uniform, trace)),
        ("counts_record", ()),
        ("port_record", (scratch,)),
        ("loader_batches", (scratch,)),
        *([("compare_real", (uniform, cifar100))] if real else []),
        ("capture_settings", (scripts, scratch)),
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
