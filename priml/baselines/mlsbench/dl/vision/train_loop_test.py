"""Tests for the loop that evaluates only after the epochs its policy picks."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Final, cast

from configgle import Makes, PartialConfig

import pytest

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.experiments_test import (
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.train_loop import VisionLoop
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.train.epoch_loop import EpochTrainLoop, first_periodic_and_last


if TYPE_CHECKING:
    from pathlib import Path

    from priml.train.train_loop import TrainLoop

EPOCHS: Final = 4
"""Epochs of the loop run."""


class EvalSpy:
    """Forwards to a loop's ``eval`` and notes the epoch each call followed."""

    def __init__(self, loop: TrainLoop) -> None:
        self.loop = loop
        self.evaluate = loop.eval
        self.epochs: list[int] = []

    def __call__(self) -> dict[str, object]:
        """Note the completed epochs, then evaluate."""
        self.epochs.append(self.loop.current_epoch)
        return self.evaluate()


class _StepConfig(VisionTrainStep.Config, kw_only=True):
    """A step config a task narrows the loop's ``step`` slot to."""

    marker: int = 0
    """Reached only through the narrowed slot."""


class _DataConfig(Makes["Cifar100Data"], Cifar100Data.Config):
    """A dataset config a task narrows the loop's ``dataset`` slot to."""

    marker: int = 0
    """Reached only through the narrowed slot."""


class _NarrowStepLoop(
    Makes["EpochTrainLoop"],
    VisionLoop[_StepConfig, Cifar100Data.Config],
):
    """Narrows ``step``; the dataset stays CIFAR-100's."""

    step: _StepConfig = field(default_factory=_StepConfig)
    """The narrowed step."""


class _NarrowDataLoop(
    Makes["EpochTrainLoop"],
    VisionLoop[VisionTrainStep.Config, _DataConfig],
):
    """Narrows ``dataset``; the step stays the vision step."""

    dataset: _DataConfig = field(default_factory=_DataConfig)
    """The narrowed dataset."""


class _VisionLoop(
    Makes["EpochTrainLoop"],
    VisionLoop[VisionTrainStep.Config, Cifar100Data.Config],
):
    """The loop with its default step and dataset."""


def evaluating_loop(directory: Path) -> _VisionLoop:
    """Return ``exp000``'s step and data in the evaluating loop, shrunk, over ``directory``."""
    source = exp000()
    config = _VisionLoop()
    config.step = source.step
    config.dataset = source.dataset
    config.metrics_eval = source.metrics_eval
    config.seed = source.seed
    config.num_steps_eval = -1
    shrink(
        cast("TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config]", config),
        directory=directory,
    )
    config.step.train_budget_epochs = config.max_epochs = EPOCHS
    return config


def test_the_loop_evaluates_only_after_the_picked_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_dataset(tmp_path)
    config = evaluating_loop(tmp_path)
    config.eval_epochs = PartialConfig(first_periodic_and_last, every=3)
    loop = config.make()
    spy = EvalSpy(loop)
    monkeypatch.setattr(loop, "eval", spy)
    loop.train()
    assert spy.epochs == [1, 3, 4]
    assert loop.step.global_step == 2 * EPOCHS


def test_a_policy_picking_every_epoch_evaluates_after_every_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_dataset(tmp_path)
    config = evaluating_loop(tmp_path)
    config.eval_epochs = PartialConfig(first_periodic_and_last, every=1)
    loop = config.make()
    spy = EvalSpy(loop)
    monkeypatch.setattr(loop, "eval", spy)
    loop.train()
    assert spy.epochs == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("max_epochs", 201), ("eval_only", True)],
)
def test_a_loop_that_would_leave_the_references_draws_is_refused(
    field_name: str,
    value: object,
) -> None:
    config = VisionLoop()
    config.step.train_budget_epochs = config.max_epochs = 200
    config.num_steps_eval = -1
    setattr(config, field_name, value)
    with pytest.raises(ValueError, match=field_name):
        config.copy_tree().finalize()


def test_a_subclass_narrows_the_step_or_the_dataset_slot(tmp_path: Path) -> None:
    assert _NarrowStepLoop().step.marker == 0
    assert _NarrowDataLoop().dataset.marker == 0
    assert _NarrowStepLoop.parent_class is EpochTrainLoop
    assert isinstance(_NarrowDataLoop().step, VisionTrainStep.Config)
    write_dataset(tmp_path)
    config = _NarrowDataLoop()
    config.step = exp000().step
    config.num_steps_eval = -1
    shrink(config, directory=tmp_path)
    loop: EpochTrainLoop = config.make()
    assert loop.eval_epochs(1, total_epochs=1)


def test_the_loop_defaults_to_the_vision_step_and_cifar100() -> None:
    config = VisionLoop()
    assert isinstance(config.step, VisionTrainStep.Config)
    assert isinstance(config.dataset, Cifar100Data.Config)
    assert VisionLoop.parent_class is EpochTrainLoop


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
