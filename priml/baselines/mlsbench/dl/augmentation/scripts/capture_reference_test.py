"""Tests for the ``cv-data-augmentation`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-56 and its baselines' pipelines are
not the real ones, so a capture of it must name those and nothing it shares
with the port: the pristine pipeline, the test pipeline, the schedule and the
optimizer.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import sys

import pytest
import torch

from priml.baselines.mlsbench.dl.augmentation import (
    reference as augmentation,
    upstream,
)
from priml.baselines.mlsbench.dl.augmentation.scripts import capture_reference
from priml.baselines.mlsbench.dl.augmentation.upstream_test import write_clone
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as upstream_vision,
)
from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from PIL import Image
    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.upstream import Batches, Upstream


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
    states: list[dict[str, Tensor]] | None = None,
    setting: str = "",
) -> dict[str, Tensor]:
    """Stand in for the port's lockstep at ``exp000``'s setting: one tensor per state."""
    assert not setting
    if states is not None:
        states += [{"w": torch.zeros(2)} for _ in range(reference.STEPS)]
    return {
        f"lockstep/{name}/media": torch.zeros(2),
        f"lockstep/{name}/loss": torch.zeros(2),
    }


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
    monkeypatch.setattr(
        augmentation,
        "pipeline_images",
        returning(augmentation.pipeline_images(2)),
    )
    monkeypatch.setattr(augmentation, "port_record", returning(stored))
    monkeypatch.setattr(augmentation, "lockstep_record", port_lockstep)
    monkeypatch.setattr(augmentation, "loader_batches", returning(passes(1.0)))
    monkeypatch.setattr(upstream, "upstream_batches", returning(passes(0.0)))
    monkeypatch.setattr(
        vision_capture,
        "compare_full_size",
        returning(["full size differs"]),
    )
    # The other settings' path runs for real in the compute_training test below.
    settings = {"settings/x": torch.zeros(2)}
    monkeypatch.setattr(
        capture_reference,
        "capture_settings",
        returning((settings, ["settings differ"])),
    )
    captured = capture_reference.capture(root, scratch)
    assert captured.record.keys() == stored.keys() | settings.keys()
    report = captured.report
    assert report[:2] == ["full size differs", "settings differ"]
    names = ("pad_crop_flip", *upstream.EDITED)
    for name in names:
        assert f"lockstep/{name}/1: state: 1 tensors against" in "\n".join(report)
        assert any(line.startswith(f"lockstep/{name}/media: ") for line in report)
        assert f"{name}: loader/train/0: 0/media: 6/6 differ" in report
        assert f"{name}: loader/test/0: 0/media: 6/6 differ" in report
    # The stand-in's policies are not torchvision's, so it reaches no operation,
    # and two images are too few for the port to reach every one.
    census = {line.rsplit(": ", 1)[0] for line in report if "never reached" in line}
    policies = capture_reference.POLICIES
    assert census == {
        f"{side} {name}" for side in ("reference", "port") for name in policies
    }
    assert "reference randaugment: posterize+ never reached" in report
    assert [line for line in report if line.endswith("operation counts differ")] == [
        f"{name}: operation counts differ" for name in policies
    ]
    assert any(line.startswith("pipeline/randaugment: ") for line in report)
    shared = ("pipeline/test", "pipeline/pad_crop_flip")
    assert not any(line.startswith(shared) for line in report)


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


@contextmanager
def counting(counts: dict[str, int]) -> Generator[dict[str, int]]:
    """Stand in for an operation census that saw ``counts``."""
    yield counts


SCRIPTS: Final = {name: cast("Upstream", object()) for name in augmentation.AUGMENTS}
"""Stand-in scripts, one per variant; only their identity is checked."""

TRAIN: Final = cast("Callable[[Image.Image], Tensor]", object())
"""A stand-in training pipeline; only its identity is checked."""

LOCKSTEP: Final[tuple[dict[str, Tensor], list[dict[str, Tensor]]]] = (
    {"lockstep/x/media": torch.ones(2), "lockstep/x/loss": torch.ones(2)},
    [],
)
"""A stand-in lockstep run: a batch to drop and a loss to keep."""


def summed(media: Tensor) -> Tensor:
    """Stand in for the classifier: the sum of every pixel."""
    return media.sum()


def test_capture_settings_joins_every_settings_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runs are tested elsewhere; here they are stubbed, to test the glue."""
    calls = Calls()
    logits = torch.zeros(2)
    trace = upstream_vision.MainTrace()
    stubs: list[tuple[object, str, object]] = [
        (vision_capture, "compare_task_settings", ({"shared": logits}, ["shared"])),
        (upstream, "pipelines", (TRAIN, "test")),
        (capture_reference, "compare_pipeline", (["pixels"], ["census"])),
        (upstream, "upstream_pipeline", torch.ones(2)),
        (augmentation, "source_images", torch.ones(2)),
        (augmentation, "classified", logits),
        (upstream, "upstream_batches", passes(0.0)),
        (augmentation, "setting_loader_batches", passes(1.0)),
        (augmentation, "setting_pipeline_record", {}),
        (capture_reference, "compare_test_pipeline", ["test"]),
        (upstream_vision, "trace_main", trace),
        (upstream, "upstream_lockstep", LOCKSTEP),
        (augmentation, "lockstep_images", torch.ones(2)),
        (capture_reference, "compare_lockstep", ["lockstep"]),
    ]
    for module, name, value in stubs:
        monkeypatch.setattr(module, name, calls.returning(name, value))
    scripts = dict(SCRIPTS)
    record, report = capture_reference.capture_settings(scripts, tmp_path)
    settings = augmentation.SETTINGS
    variants = list(augmentation.AUGMENTS)
    expected = {"shared": logits, "lockstep/x/loss": torch.ones(2)} | {
        f"settings/{s}/{n}/augment": logits for s in settings for n in variants
    }
    assert mismatches(expected, record) == []
    per_variant = [
        "pixels",
        "census",
        *(
            f"settings/{{s}}/{{n}}/loader/{k}/0: 0/media: 6/6 differ"
            for k in ("train", "test")
        ),
        "missing settings/{s}/{n}/augment",
    ]
    assert report == ["shared"] + [
        line
        for s in settings
        for line in [
            *(p.format(s=s, n=n) for n in variants for p in per_variant),
            "test",
            "lockstep",
        ]
    ]
    ((arguments, keywords),) = calls.named("compare_task_settings")
    assert arguments == (scripts, settings, tmp_path)
    assert keywords == {"pristine": "pad_crop_flip"}
    datasets = [reference.SETTINGS[s].dataset for s in settings]
    assert [k["dataset"] for _, k in calls.named("pipelines")] == [
        d for d in datasets for _ in variants
    ]
    assert [k["dataset"] for _, k in calls.named("upstream_batches")] == [
        d for d in datasets for _ in variants
    ]
    assert [a for a, _ in calls.named("compare_pipeline")] == [
        (s, n, TRAIN) for s in settings for n in variants
    ]
    assert [a for a, _ in calls.named("source_images")] == [
        (s, augmentation.GOLDEN_IMAGES) for s in settings for _ in variants
    ]
    assert [a for a, _ in calls.named("setting_loader_batches")] == [
        (s, n, tmp_path) for s in settings for n in variants
    ]
    assert [a for a, _ in calls.named("compare_test_pipeline")] == [
        (SCRIPTS["pad_crop_flip"], s) for s in settings
    ]
    assert [k for _, k in calls.named("trace_main")] == [
        {
            "arch": reference.SETTINGS[s].arch,
            "dataset": reference.SETTINGS[s].dataset,
            "classes": reference.SETTINGS[s].classes,
        }
        for s in settings
    ]
    assert [
        (a[1], k["setting"], k["key"]) for a, k in calls.named("upstream_lockstep")
    ] == [
        (n, s, augmentation.lockstep_key(n, setting=s))
        for s in settings
        for n in variants
    ]
    assert [k for _, k in calls.named("compare_lockstep")] == [
        {"setting": s} for s in settings
    ]


def test_compare_pipeline_reports_pixels_and_the_census_of_a_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Calls()
    full = {f"{op}+": 1 for op in upstream.OPS.values()} | {
        f"{op}-": 1 for op in upstream.SIGNED
    }
    source, prepared = object(), object()
    for module, name, value in (
        (augmentation, "source_images", source),
        (augmentation, "prepared", prepared),
        (upstream, "upstream_pipeline", torch.zeros(2)),
        (augmentation, "augmented", torch.ones(2)),
    ):
        monkeypatch.setattr(module, name, calls.returning(name, value))
    monkeypatch.setattr(upstream, "op_census", lambda: counting({"identity+": 1}))
    monkeypatch.setattr(augmentation, "op_census", lambda: counting(full))
    setting = "mobilenetv2-fmnist"
    key = f"settings/{setting}/randaugment/pipeline"
    pixels, census = capture_reference.compare_pipeline(setting, "randaugment", TRAIN)
    assert pixels == [f"{key}: 2/2 differ"]
    gaps = upstream.census_gaps({"identity+": 1})
    assert census == [f"reference {key}: {k} never reached" for k in gaps] + [
        f"{key}: operation counts differ",
    ]
    assert calls.named("source_images") == [
        ((setting, augmentation.PIPELINE_IMAGES), {}),
    ]
    assert calls.named("upstream_pipeline") == [
        ((TRAIN, source), {"seed": reference.SEED}),
    ]
    assert calls.named("prepared") == [((setting, source), {})]
    ((arguments, keywords),) = calls.named("augmented")
    assert arguments == ("randaugment", prepared)
    assert keywords == {
        "seed": reference.SEED,
        "factory": augmentation.SETTINGS[setting]["randaugment"],
    }
    monkeypatch.setattr(upstream, "op_census", lambda: counting(full))
    _, census = capture_reference.compare_pipeline(setting, "randaugment", TRAIN)
    assert census == []
    _, census = capture_reference.compare_pipeline(setting, "cutout", TRAIN)
    assert census == []


def test_compare_test_pipeline_normalizes_the_prepared_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setting = "mobilenetv2-fmnist"
    calls = Calls()
    images = augmentation.source_images(setting, 2)
    config = augmentation.SETTINGS[setting]["pad_crop_flip"]().dataset
    want = normalize(
        augmentation.prepared(setting, images),
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    monkeypatch.setattr(
        augmentation,
        "source_images",
        calls.returning("images", images),
    )
    monkeypatch.setattr(upstream, "pipelines", calls.returning("pipelines", ("x", "t")))
    monkeypatch.setattr(
        upstream,
        "upstream_pipeline",
        calls.returning("upstream_pipeline", want),
    )
    script = SCRIPTS["pad_crop_flip"]
    assert capture_reference.compare_test_pipeline(script, setting) == []
    assert calls.named("pipelines") == [((script,), {"dataset": "fmnist"})]
    assert calls.named("images") == [((setting, augmentation.PIPELINE_IMAGES), {})]
    ((arguments, keywords),) = calls.named("upstream_pipeline")
    assert arguments[0] == "t"
    assert arguments[1] is images
    assert keywords == {"seed": reference.SEED}
    monkeypatch.setattr(
        upstream,
        "upstream_pipeline",
        calls.returning("upstream_pipeline", want + 1),
    )
    key = f"settings/{setting}/test/pipeline"
    assert capture_reference.compare_test_pipeline(script, setting) == [
        f"{key}: {want.numel()}/{want.numel()} differ",
    ]


def test_compare_pipelines_counts_the_census_and_matches_the_test_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Calls()
    images = augmentation.pipeline_images(2)
    config = augmentation.AUGMENTS["pad_crop_flip"]().dataset
    test = normalize(
        images,
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    full = {f"{op}+": 1 for op in upstream.OPS.values()} | {
        f"{op}-": 1 for op in upstream.SIGNED
    }
    shifts = {name: float(i) for i, name in enumerate(augmentation.AUGMENTS)}
    scripts = dict(SCRIPTS)

    def pipelines(script: object) -> tuple[str, str]:
        (name,) = (n for n, s in SCRIPTS.items() if s is script)
        return name, "test"

    def running(run: str, source: Tensor, *, seed: int) -> Tensor:
        assert seed == reference.SEED
        assert source is images
        return test if run == "test" else source.float() + shifts[run]

    def augmented(name: str, source: Tensor, *, seed: int) -> Tensor:
        return running(name, source, seed=seed)

    monkeypatch.setattr(
        augmentation,
        "pipeline_images",
        calls.returning("pipeline_images", images),
    )
    monkeypatch.setattr(upstream, "pipelines", pipelines)
    monkeypatch.setattr(upstream, "upstream_pipeline", running)
    monkeypatch.setattr(augmentation, "augmented", augmented)
    monkeypatch.setattr(upstream, "op_census", lambda: counting(full))
    monkeypatch.setattr(augmentation, "op_census", lambda: counting(full))
    monkeypatch.setattr(augmentation, "classified", summed)
    report, census, record = capture_reference.compare_pipelines(scripts)
    assert (report, census) == ([], [])
    assert calls.named("pipeline_images") == [((augmentation.PIPELINE_IMAGES,), {})]
    assert torch.equal(record["augment/test/logits"], test.sum())
    monkeypatch.setattr(augmentation, "op_census", lambda: counting({}))
    _, census, _ = capture_reference.compare_pipelines(scripts)
    gaps = upstream.census_gaps({})
    assert census == [
        line
        for name in capture_reference.POLICIES
        for line in [
            *(f"port {name}: {k} never reached" for k in gaps),
            f"{name}: operation counts differ",
        ]
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


@pytest.mark.parametrize(("size", "code"), [(3, 0), (vision_capture.CEILING, 1)])
def test_main_writes_the_golden_and_refuses_one_past_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    size: int,
    code: int,
) -> None:
    record = {"x": torch.zeros(size, dtype=torch.uint8)}
    result = reference.Captured(record=record, sites={}, report=[])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    out.mkdir()
    (out / "reference.pt").write_bytes(b"previous")
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture", "--root", str(tmp_path), "--out", str(out)],
    )
    assert capture_reference.main() == code
    printed = capsys.readouterr().out
    if code:
        assert "the ceiling is" in printed
        assert (out / "reference.pt").read_bytes() == b"previous"
    else:
        assert f"0 mismatches; wrote 1 values to {out}." in printed
        assert read_tensors(out / "reference.pt").keys() == {"x"}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
