"""Tests for the vision experiment chain.

Two kinds of assertion. The fields ``exp000`` pins are checked by value: it is the
control every fork is measured against, so a change to it invalidates published
results and the test makes that change deliberate. Each fork is checked as a
DELTA, exactly which fields differ from its parent, which keeps "one change per
experiment" a property of the code rather than a claim in a docstring.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Final, cast

import inspect

from configgle import InlineConfig
from treekle.absent import ABSENT

import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision import experiments
from priml.baselines.mlsbench.pytorch_vision.data import Cifar100Data
from priml.baselines.mlsbench.pytorch_vision.experiments import (
    VisionTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.pytorch_vision.metrics import BestAccuracy
from priml.baselines.mlsbench.pytorch_vision.model import ResNet
from priml.baselines.mlsbench.pytorch_vision.train_step import (
    VisionTrainStep,
    cross_entropy,
)
from priml.runtime import SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.parallelism import NoParallel
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from pathlib import Path

    import functools

    from _typeshed import DataclassInstance

    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[VisionTrainLoop]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
]


def shrink(config: VisionTrainLoop, *, directory: Path) -> VisionTrainLoop:
    """Narrow ``config`` to a size a CPU test can run, preserving its recipe.

    Only SIZE changes: depth, batch, horizon. Optimizer, schedule, loss and
    augmentation stay exactly as the experiment set them.
    """
    assert isinstance(config.runtime, SingleProcess.Config)
    config.runtime.device = "cpu"
    config.step.parallelism = NoParallel.Config(device="cpu")
    assert isinstance(config.step.model, ResNet.Config)
    config.step.model.blocks_per_stage = (1, 1, 1)
    config.step.model.channels_hidden = (4, 5, 6)
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


def write_dataset(directory: Path) -> None:
    """Write a miniature dataset in the prepared format."""
    generator = torch.Generator().manual_seed(0)
    for split in ("train", "test"):
        torch.save(
            {
                "media": torch.randint(0, 256, (8, 3, 8, 10), generator=generator).to(
                    torch.uint8,
                ),
                "label": torch.randint(0, 100, (8,), generator=generator),
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
    assert cfg.seed is None


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


def changed_fields(parent: VisionTrainLoop, child: VisionTrainLoop) -> set[str]:
    """Return the field paths ``udiff`` reports between two unfinalized configs."""
    table = parent.udiff(child, mode="fields", finalize=False)
    return {line.split("│")[0].strip() for line in table.splitlines()[1:]}


@pytest.mark.parametrize(
    ("factory", "changed"),
    [
        (exp001, {"step.loss.func", "step.loss.smoothing"}),
        (exp002, {"step.loss.func", "step.loss.gamma"}),
        (exp003, {"step.loss.func", "step.loss.epsilon"}),
    ],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_fork_changes_only_the_loss(
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(exp000(), factory()) == {"experiment_name", *changed}
    assert _deltas(exp000(), factory()) == {"experiment_name", "step.loss"}


def test_forks_do_not_mutate_their_parent() -> None:
    baseline = exp000()
    before = _flatten(baseline)
    _ = exp003()
    assert _flatten(baseline) == before


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for child in (exp001, exp002, exp003):
        assert exp000.__name__ in (inspect.getdoc(child) or "").splitlines()[0], (
            child.__name__
        )


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


# A field holding a nested Config is descended into, so a change buried in
# ``step.model`` is reported at the leaf that actually moved; an injected callable is
# compared by its repr, which carries the function and its bound arguments.
def _deltas(parent: VisionTrainLoop, child: VisionTrainLoop) -> set[str]:
    """Return the dotted names of the fields that differ between two configs."""
    flat_parent, flat_child = _flatten(parent), _flatten(child)
    return {
        name
        for name in flat_parent.keys() | flat_child.keys()
        if flat_parent.get(name, ABSENT) != flat_child.get(name, ABSENT)
    }


def _flatten(config: DataclassInstance, prefix: str = "") -> dict[str, object]:
    """Return a dotted-name to value map, descending into nested Configs."""
    flat: dict[str, object] = {}
    for field in fields(config):
        value = cast(object, getattr(config, field.name))
        dotted = f"{prefix}{field.name}"
        if isinstance(value, InlineConfig):
            flat[dotted] = repr(cast("InlineConfig[object]", value))
        elif isinstance(value, dict):
            flat[dotted] = {
                k: repr(v) for k, v in cast("dict[str, object]", value).items()
            }
        elif is_dataclass(value):
            flat[dotted] = type(value)
            flat.update(_flatten(cast("DataclassInstance", value), prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
