"""Tests for the ``cv-sample-weighting`` capture, against a stand-in clone.

The stand-in's network differs from ResNet-32 and its baselines' weightings are
not the real ones, so a capture of it must name those and nothing it shares
with the port: the long-tail cut, the uniform weights, the schedule and the
optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import sys

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, normalize
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.weighting import (
    reference as weighting,
    upstream,
)
from priml.baselines.mlsbench.dl.weighting.scripts import capture_reference
from priml.baselines.mlsbench.dl.weighting.upstream_test import (
    write_cifar100,
    write_clone,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
    from priml.baselines.mlsbench.dl.vision.upstream import Batches


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
    media, label = weighting.long_tail_split()
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
    states += [{"w": torch.zeros(2)} for _ in range(reference.STEPS)]
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
    monkeypatch.setattr(weighting, "port_record", returning(stored))
    monkeypatch.setattr(weighting, "lockstep_record", port_lockstep)
    monkeypatch.setattr(weighting, "loader_batches", returning(passes([1, 0])))
    full_step = reference.full_step

    def shallow_step(factory: Callable[[], VisionTrainLoop]) -> VisionTrainStep:
        return full_step(shallow(factory))

    monkeypatch.setattr(reference, "full_step", shallow_step)
    monkeypatch.setattr(upstream, "upstream_batches", returning(passes([0, 1])))
    monkeypatch.setattr(
        upstream,
        "real_subset",
        returning((real["real/indices"], real["real/counts"], real["label"])),
    )
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


@pytest.mark.compute_training
def test_capture_names_what_the_stand_in_does_differently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    write_cifar100(tmp_path)
    monkeypatch.setattr(
        "torchvision.datasets.cifar.check_integrity",
        returning(value=True),
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report = capture_reference.capture(
        root,
        scratch,
        cifar100=tmp_path,
    ).report
    keys = {line.split(":")[0] for line in report}
    shared = {
        "weights/uniform",
        "loss/uniform/value",
        "counts/cifar100lt",
        "main/weight",
        "main/indices",
        "subset/indices",
        "schedule/lr",
        "optimizer/settings",
        "real/indices",
        "real/counts",
    }
    assert not shared & keys, report
    # The stand-in's loaders draw in-process rather than through four workers,
    # so its shuffle reaches different images.
    assert {"weights/inverse_freq", "draws/exp000", "state", "loader/index"} <= keys
    assert any(line.startswith("lockstep/uniform/1: ") for line in report)


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
