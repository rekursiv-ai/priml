"""Tests for the CIFAR-10 experiment configs.

Two kinds of assertion, and the distinction matters:

* The fields ``exp000`` pins are checked by value. It is the control every
  fork is measured against, so a change to it invalidates published results;
  the test exists to make that change deliberate rather than incidental.
* Each fork is checked as a DELTA -- exactly which fields differ from its
  parent. That is what keeps "one named change per experiment" true as a
  property of the code rather than a claim in a docstring.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from configgle import InlineConfig

import pytest
import torch

from priml.baselines.cifar10 import experiments
from priml.baselines.cifar10.data import Cifar10Data
from priml.baselines.cifar10.experiments import (
    Cifar10TrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp_smoke,
)
from priml.baselines.cifar10.model import ConvBlock, ResNet, SpeedNet
from priml.baselines.cifar10.train_step import Cifar10TrainStep
from priml.metrics.topk import TopK
from priml.optimizers import CompositeOptimizer
from priml.train.parallelism import NoParallel
from priml.train.train_loop import TrainLoop


class _Experiment(Protocol):
    """An experiment factory, named so tests can report which one failed."""

    __name__: str

    def __call__(self) -> Cifar10TrainLoop: ...


ALL_EXPERIMENTS: list[_Experiment] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp_smoke,
]


def shrink(config: Cifar10TrainLoop, *, directory: Path) -> Cifar10TrainLoop:
    """Narrow ``config`` to a size a CPU test can run, preserving its recipe.

    Only SIZE changes -- widths, depth, batch, horizon. Optimizer, schedule,
    loss, and augmentation stay exactly as the experiment set them, so what
    runs here is the published recipe at minimum scale.
    """
    config.step.total_train_steps = 2
    config.step.parallelism = NoParallel.Config(device="cpu")
    config.step.compile = None
    config.step.translate_pad = 1
    config.step.whiten_num_images = 4

    model = config.step.model
    if isinstance(model, ResNet.Config):
        model.channels_hidden = (8, 16)
        model.blocks_per_stage = 1
    elif isinstance(model, SpeedNet.Config):
        model.channels_hidden = (8, 16, 24)
        block = model.block = ConvBlock.Config()
        block.num_convs = 1

    # Set the ROOT's base_dir: TrainLoop.finalize pushes it into the dataset,
    # overwriting anything set on the child directly.
    config.base_dir = None
    config.working_dir = directory
    config.dataset.working_dir = directory
    config.dataset.batch_size = 4
    config.dataset.eval_batch_size = 4
    config.dataset.device = "cpu"

    config.max_steps = 2
    config.max_epochs = 1
    config.num_steps_eval = 2
    config.checkpointing = None
    config.tracker = None
    return config


def write_dataset(directory: Path, *, image: int = 32) -> None:
    """Write a miniature dataset in the prepared format."""
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    for split in ("train", "test"):
        torch.save(
            {
                "media": torch.randn(8, 3, image, image, generator=generator),
                "label": torch.randint(0, 10, (8,), generator=generator),
            },
            directory / f"{split}.pt",
        )


def test_experiment_binds_the_cifar10_step_and_dataset() -> None:
    """The two slots carry their concrete types straight from the defaults.

    That binding is what lets a factory write ``cfg.step.model`` without an
    ``isinstance`` narrow. Drop the field defaults and the slots silently
    revert to the base loop's ``TrainStep`` and ``DummyDataset``, which
    type-checks and then fails at run time.
    """
    cfg = Cifar10TrainLoop()
    assert isinstance(cfg.step, Cifar10TrainStep.Config)
    assert isinstance(cfg.dataset, Cifar10Data.Config)
    assert isinstance(cfg.step.model, ResNet.Config)


def test_experiment_makes_a_train_loop() -> None:
    assert Cifar10TrainLoop.parent_class is TrainLoop


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=lambda f: f.__name__)
def test_every_experiment_finalizes(factory: _Experiment) -> None:
    assert factory().copy_tree().finalize() is not None


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=lambda f: f.__name__)
def test_experiment_name_matches_the_factory(
    factory: _Experiment,
) -> None:
    # The launcher derives run identity, and therefore the output directory,
    # from this field; a mismatch silently writes one run over another.
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=lambda f: f.__name__)
def test_construction_reads_no_files(
    factory: _Experiment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building a config must not touch the filesystem or the network.

    Staging belongs to ``scripts/prepare_data.py``, so a config can be built
    and inspected on a laptop that has neither the dataset nor a GPU.
    """
    monkeypatch.chdir(tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("experiment construction must not load tensors")

    monkeypatch.setattr(torch, "load", boom)
    _ = factory().copy_tree().finalize()


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=lambda f: f.__name__)
def test_experiment_trains_at_minimum_size(
    factory: _Experiment,
    tmp_path: Path,
) -> None:
    """Each published recipe runs end to end, shrunk to test size."""
    write_dataset(tmp_path)
    loop = shrink(factory(), directory=tmp_path).make()
    loop.train()
    assert loop.step.global_step == 2


def test_exp000_pins_the_baseline_recipe() -> None:
    """exp000 is frozen: a change here invalidates every measured comparison."""
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    assert cfg.step.label_smoothing == 0.1
    assert cfg.step.translate_pad == 4
    assert cfg.step.use_tta is False
    assert cfg.step.cutout_size == 0
    step = cfg.step.make()
    assert isinstance(step.optimizer, CompositeOptimizer)
    assert [type(o) for o in step.optimizer.optimizers] == [torch.optim.AdamW]
    assert step.optimizer.param_groups[0]["initial_lr"] == 1e-3


def test_exp000_horizon_matches_the_step_budget() -> None:
    # A schedule horizon shorter than the run leaves the tail at zero learning
    # rate; longer, and the run stops before the schedule anneals.
    cfg = exp000()
    assert cfg.step.total_train_steps == cfg.max_steps


def test_exp000_reports_top1_accuracy() -> None:
    cfg = exp000()
    accuracy = cfg.metrics["accuracy"]
    assert isinstance(accuracy, TopK.Config)
    assert accuracy.k_values == [1]


def test_exp001_changes_only_the_architecture_and_budget() -> None:
    # The budget moves with the architecture: SpeedNet exists to reach
    # accuracy in far fewer epochs, so running it for exp000's 30 would not
    # be the same experiment. ``>=`` because swapping the model Config class
    # also reports every field the two classes do not share.
    assert _deltas(exp000(), exp001()) >= {
        "experiment_name",
        "step.model",
        "step.total_train_steps",
        "max_steps",
    }
    assert not _deltas(exp000(), exp001()) & {
        "step.optimizer",
        "step.schedule",
        "step.label_smoothing",
        "step.use_tta",
        "step.translate_pad",
        "seed",
    }


def test_exp002_changes_only_the_optimizer() -> None:
    # The schedule shape moves with the optimizer: Muon's published recipe
    # anneals polynomially, so the two are one change, not two.
    assert _deltas(exp001(), exp002()) == {
        "experiment_name",
        "step.optimizer.optimizers",
        "step.optimizer.select",
        "step.schedule",
    }


def test_exp003_changes_only_evaluation() -> None:
    assert _deltas(exp002(), exp003()) == {"experiment_name", "step.use_tta"}


def test_exp004_changes_only_initialization() -> None:
    assert _deltas(exp003(), exp004()) == {
        "experiment_name",
        "step.model.init_conv",
    }


def test_forks_do_not_mutate_their_parent() -> None:
    # Each factory rebuilds its parent, so a fork's mutations must not leak
    # into a config the caller built earlier.
    baseline = exp000()
    before = _flatten(baseline)
    _ = exp004()
    assert _flatten(baseline) == before


def test_smoke_is_marked_as_carrying_no_result() -> None:
    # It exists to verify an installation, so it must not be mistaken for a
    # measurement in the experiment chain.
    assert "Not a result" in (exp_smoke.__doc__ or "")


def test_module_docstring_lists_every_experiment() -> None:
    documented = experiments.__doc__ or ""
    for factory in ALL_EXPERIMENTS:
        if factory is exp_smoke:
            continue
        assert factory.__name__ in documented


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for child, parent in (
        (exp001, exp000),
        (exp002, exp001),
        (exp003, exp002),
        (exp004, exp003),
    ):
        summary = (child.__doc__ or "").splitlines()[0]
        assert parent.__name__ in summary, child.__name__


def _deltas(parent: Cifar10TrainLoop, child: Cifar10TrainLoop) -> set[str]:
    """Return the dotted names of the fields that differ between two configs.

    A field holding a nested Config is descended into, so a change buried in
    ``step.model`` is reported at the leaf that actually moved -- unless the
    two sides hold DIFFERENT Config classes, in which case the swap itself is
    the single change and its fields are not comparable.
    """
    flat_parent, flat_child = _flatten(parent), _flatten(child)
    return {
        name
        for name in flat_parent.keys() | flat_child.keys()
        if flat_parent.get(name, _MISSING) != flat_child.get(name, _MISSING)
    }


def _flatten(config: Any, prefix: str = "") -> dict[str, Any]:
    """Return a dotted-name to value map, descending into nested Configs."""
    flat: dict[str, Any] = {}
    fields = cast(dict[str, Any], type(config).__dataclass_fields__)
    for name in fields:
        value: Any = getattr(config, name)
        dotted = f"{prefix}{name}"
        if isinstance(value, InlineConfig):
            # An injected callable: its identity is the function plus its bound
            # arguments, which ``repr`` captures and ``==`` does not (comparing
            # two raises on the absent ``parent_class``).
            flat[dotted] = repr(cast(InlineConfig[Any], value))
        elif isinstance(value, list):
            # A list of injected members (optimizers, selectors): compare by
            # element repr for the same reason, since ``==`` on the list would
            # reach each element's absent ``parent_class``.
            flat[dotted] = [repr(item) for item in cast(list[Any], value)]
        elif is_dataclass(value):
            # Record the class itself, so swapping in a different Config
            # registers as one change rather than a diff of every field.
            flat[dotted] = type(value)
            flat.update(_flatten(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


_MISSING = object()
"""Sentinel for a field only one of the two configs has."""


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
