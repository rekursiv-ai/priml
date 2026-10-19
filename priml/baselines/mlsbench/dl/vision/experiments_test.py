"""Tests for the recipe every MLS-Bench DL task forks.

The fields ``exp000`` pins are checked by value: it is the control every task's
fork is measured against, so a change to it invalidates published results and
the test makes that change deliberate. The task packages' tests check their
forks with ``shrink``, ``write_dataset`` and ``changed_fields`` from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import inspect

from configgle import InlineConfig

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import experiments
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.experiments import (
    VisionTrainLoop,
    exp000,
)
from priml.baselines.mlsbench.dl.vision.metrics import BestAccuracy
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.settings import (
    mobilenetv2_fmnist,
    resnet110_cifar100,
    vgg16bn_cifar100,
)
from priml.baselines.mlsbench.dl.vision.train_step import (
    VisionTrainStep,
    cross_entropy,
)
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.runtime import SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.parallelism import NoParallel
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import functools

    from priml.baselines.mlsbench.dl.vision.reference import VisionLoop
    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[VisionTrainLoop]]] = [exp000]


def shrink[C: TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config]](
    config: C,
    *,
    directory: Path,
) -> C:
    """Narrow ``config`` to a size a CPU test can run, preserving its recipe.

    Only SIZE changes: width, depth, batch, horizon, for a ResNet, VGG16BN or
    MobileNetV2 network. Optimizer, schedule, loss, augmentation and every
    network slot stay exactly as the experiment set them.
    """
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.device = "cpu"
    config.step.parallelism = NoParallel.Config(device="cpu")
    model = config.step.model
    if isinstance(model, VGG16BN.Config):
        model.channels_hidden = ((4,), (5,))
        model.channels_hidden_head = 6
    elif isinstance(model, MobileNetV2.Config):
        model.channels_stem = 4
        model.channels_hidden = (5, 6)
        model.blocks_per_stage = (1, 1)
        model.expand_ratios = (1, 2)
        model.strides = (1, 2)
        model.channels_last = 7
    else:
        assert isinstance(model, ResNet.Config)
        model.blocks_per_stage = (1, 1, 1)
        model.channels_hidden = (4, 5, 6)
    config.step.train_budget_epochs = config.max_epochs = 1
    # Set the ROOT's base_dir: TrainLoop.finalize pushes it into the dataset,
    # overwriting anything set on the child directly.
    config.base_dir = None
    config.working_dir = directory
    config.dataset.working_dir = directory
    config.dataset.batch_size = config.dataset.eval_batch_size = 4
    config.dataset.translate_pad = 1
    config.checkpointer = None
    config.tracker = None
    return config


def write_dataset(directory: Path, *, classes: int = 100) -> None:
    """Write a miniature dataset in the prepared format, labels below ``classes``."""
    generator = torch.Generator().manual_seed(0)
    for split in ("train", "test"):
        torch.save(
            {
                "media": torch.randint(0, 256, (8, 3, 6, 10), generator=generator).to(
                    torch.uint8,
                ),
                "label": torch.randint(0, classes, (8,), generator=generator),
            },
            directory / f"{split}.pt",
        )


def test_experiment_binds_the_vision_step_and_cifar100() -> None:
    cfg = VisionTrainLoop()
    assert isinstance(cfg.step, VisionTrainStep.Config)
    assert isinstance(cfg.dataset, Cifar100Data.Config)
    assert VisionTrainLoop.parent_class is TrainLoop


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_finalizes(
    factory: ExperimentFactory[VisionTrainLoop],
) -> None:
    assert factory().copy_tree().finalize() is not None


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_name_matches_the_factory(
    factory: ExperimentFactory[VisionTrainLoop],
) -> None:
    # The launcher derives the output directory from this field; a mismatch
    # silently writes one run over another.
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_construction_reads_no_files(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building a config must not touch the filesystem or the network."""
    monkeypatch.chdir(tmp_path)

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("experiment construction must not load tensors")

    monkeypatch.setattr(torch, "load", boom)
    _ = factory().copy_tree().finalize()


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    """Each recipe runs end to end for one epoch and is evaluated at its end."""
    write_dataset(tmp_path)
    loop = shrink(factory(), directory=tmp_path).make()
    loop.train()
    assert loop.step.global_step == 2
    assert loop.dataset.timer_epoch.global_count == 1
    accuracy = loop.metrics_eval["test_acc"]
    assert isinstance(accuracy, BestAccuracy)
    assert accuracy.total == 8


@pytest.mark.parametrize(
    "switch",
    [vgg16bn_cifar100, mobilenetv2_fmnist, resnet110_cifar100],
    ids=["vgg16bn_cifar100", "mobilenetv2_fmnist", "resnet110_cifar100"],
)
def test_every_setting_trains_at_minimum_size(
    switch: Callable[[VisionTrainLoop], VisionTrainLoop],
    tmp_path: Path,
) -> None:
    """``shrink`` narrows each setting's network, so a task's setting fork can train."""
    write_dataset(tmp_path, classes=10)
    loop = shrink(switch(exp000()), directory=tmp_path).make()
    loop.train()
    assert loop.step.global_step == 2
    assert loop.dataset.timer_epoch.global_count == 1


def test_exp000_pins_the_reference_recipe() -> None:
    """exp000 is frozen: a change here invalidates every measured comparison."""
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    assert cfg.step.model.blocks_per_stage == (9, 9, 9)
    assert cfg.step.model.channels_hidden == (16, 32, 64)
    assert cfg.step.model.channels_out == 100
    step = cfg.step.make()
    assert isinstance(step.optimizer, torch.optim.SGD)
    group = step.optimizer.param_groups[0]
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.1, 0.9, 5e-4)
    assert (group["nesterov"], group["dampening"]) == (False, 0)
    assert cfg.dataset.batch_size == 128
    assert cfg.dataset.num_streams == 4
    assert cfg.dataset.translate_pad == 4
    assert cfg.seed == 42


def test_exp000_trains_with_cross_entropy_for_200_epochs() -> None:
    cfg = exp000()
    loss = cfg.step.loss
    assert isinstance(loss, InlineConfig)
    made = cast("functools.partial[object]", loss.make())
    assert (made.func, made.args, made.keywords) == (cross_entropy, (), {})
    # A schedule horizon shorter than the run leaves the tail at zero learning
    # rate; longer, and the run stops before the schedule anneals.
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 200


def test_exp000_evaluates_after_every_epoch_and_nowhere_else() -> None:
    cfg = exp000()
    assert cfg.eval_every_epoch
    assert cfg.num_steps_eval == -1


def test_exp000_reports_the_best_and_the_final_test_accuracy() -> None:
    assert isinstance(exp000().metrics_eval["test_acc"], BestAccuracy.Config)


def changed_fields(parent: VisionLoop, child: VisionLoop) -> set[str]:
    """Return the field paths ``udiff`` reports between two unfinalized configs."""
    table = parent.udiff(child, mode="fields", finalize=False)
    return {line.split("│")[0].strip() for line in table.splitlines()[1:]}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[VisionTrainLoop],
) -> None:
    doc = inspect.getdoc(factory) or ""
    assert "Hypothesis:" in doc
    assert "References:" in doc
    assert "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")


def test_module_docstring_lists_every_experiment() -> None:
    documented = inspect.getdoc(experiments) or ""
    for factory in ALL_EXPERIMENTS:
        assert factory.__name__ in documented


def test_exp000_matches_its_golden_config() -> None:
    """Pin the WHOLE finalized ``exp000``: the control every fork is measured against."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_epochs", 201),
        ("num_steps_eval", 100),
        ("eval_warmup_batches", 1),
        ("eval_only", True),
    ],
)
def test_a_loop_that_would_leave_the_references_draws_is_refused(
    field: str,
    value: object,
) -> None:
    config = exp000()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.copy_tree().finalize()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
