"""Tests for the ``security-poison-robust-learning`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from its setting's pristine experiment, so
"one slot per idea" is a property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import functools
import inspect

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import (
    cosine_annealing,
    cross_entropy,
)
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.baselines.mlsbench.tl.poison_robust_learning import experiments
from priml.baselines.mlsbench.tl.poison_robust_learning.data import PoisonedData
from priml.baselines.mlsbench.tl.poison_robust_learning.experiments import (
    PoisonTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp009,
    exp010,
    exp011,
    exp012,
    exp013,
    exp014,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.losses import (
    bootstrap,
    generalized_ce,
    symmetric_ce,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.metrics import (
    PoisonRobustness,
)
from priml.model.swiglu import relu
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from pathlib import Path

    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[PoisonTrainLoop]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp009,
    exp010,
    exp011,
    exp012,
    exp013,
    exp014,
]

LOSS: Final = {
    "cross_entropy": set[str](),
    "generalized_ce": {"step.loss.func", "step.loss.q", "step.loss.min_probability"},
    "symmetric_ce": {
        "step.loss.func",
        "step.loss.alpha",
        "step.loss.beta",
        "step.loss.min_probability",
        "step.loss.min_target",
    },
    "bootstrap": {"step.loss.func", "step.loss.beta"},
}
"""The fields each baseline's fork changes on its setting's pristine experiment."""


def bound(config: object) -> dict[str, object]:
    """Return the function and keywords a ``PartialConfig`` binds."""
    assert isinstance(config, PartialConfig)
    made = config.make()
    assert isinstance(made, functools.partial)
    return {"fn": made.func, **made.keywords}


def test_exp000_pins_the_reference_recipe() -> None:
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    assert (cfg.step.model.blocks_per_stage, cfg.step.model.channels_out) == (
        (3, 3, 3),
        10,
    )
    step = cfg.step.make()
    assert isinstance(step.optimizer, torch.optim.SGD)
    group = step.optimizer.param_groups[0]
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.1, 0.9, 5e-4)
    assert bound(cfg.step.loss) == {"fn": cross_entropy}
    assert bound(cfg.step.schedule) == {"fn": cosine_annealing}
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 100
    assert isinstance(cfg.dataset, PoisonedData.Config)
    assert cfg.dataset.batch_size == cfg.dataset.eval_batch_size == 128
    assert (cfg.dataset.poison_fraction, cfg.dataset.poison_seed) == (0.1, 42)
    assert cfg.seed == 42
    assert cfg.dataset.mean == (0.4914, 0.4822, 0.4465)
    assert cfg.dataset.std == (0.2023, 0.1994, 0.2010)


def test_the_flip_wraps_at_the_networks_class_count() -> None:
    for factory, classes in ((exp000, 10), (exp005, 100), (exp010, 10)):
        assert factory().copy_tree().finalize().dataset.num_classes == classes
        assert factory().dataset.num_classes == -1


def test_exp000_evaluates_once_after_training() -> None:
    cfg = exp000()
    assert (cfg.num_steps_eval, cfg.eval_every_epoch) == (-1, False)
    assert list(cfg.metrics_eval) == ["poison"]
    assert isinstance(cfg.metrics_eval["poison"], PoisonRobustness.Config)


@pytest.mark.parametrize(
    ("parent", "forks"),
    [
        (exp000, (exp001, exp002, exp003, exp004)),
        (exp005, (exp006, exp007, exp008, exp009)),
        (exp010, (exp011, exp012, exp013, exp014)),
    ],
    ids=["resnet20-cifar10", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
)
def test_a_fork_changes_only_the_loss(
    parent: ExperimentFactory[PoisonTrainLoop],
    forks: tuple[ExperimentFactory[PoisonTrainLoop], ...],
) -> None:
    for fork, changed in zip(forks, LOSS.values(), strict=True):
        assert changed_fields(parent(), fork()) == {"experiment_name", *changed}


@pytest.mark.parametrize(
    "forks",
    [
        (exp002, exp003, exp004),
        (exp007, exp008, exp009),
        (exp012, exp013, exp014),
    ],
    ids=["exp000", "exp005", "exp010"],
)
def test_each_loss_carries_the_baselines_settings(
    forks: tuple[ExperimentFactory[PoisonTrainLoop], ...],
) -> None:
    gce, sce, boot = (bound(fork().step.loss) for fork in forks)
    assert gce == {"fn": generalized_ce, "q": 0.7, "min_probability": 1e-8}
    assert sce == {
        "fn": symmetric_ce,
        "alpha": 1.0,
        "beta": 0.5,
        "min_probability": 1e-8,
        "min_target": 1e-4,
    }
    assert boot == {"fn": bootstrap, "beta": 0.8}


@pytest.mark.parametrize("setting", [exp005, exp010], ids=["exp005", "exp010"])
def test_each_setting_changes_only_the_network_and_the_data(
    setting: ExperimentFactory[PoisonTrainLoop],
) -> None:
    changed = changed_fields(exp000(), setting())
    roots = {
        "step.model" if path.startswith("step.model") else path.split("[")[0]
        for path in changed
    }
    assert roots <= {
        "experiment_name",
        "step.model",
        "dataset.working_dir",
        "dataset.mean",
        "dataset.std",
        "dataset.poison_fraction",
    }
    assert {"experiment_name", "step.model", "dataset.working_dir"} <= roots


def test_the_settings_build_the_tasks_networks_and_fractions() -> None:
    assert isinstance(exp005().step.model, VGG16BN.Config)
    assert exp005().dataset.poison_fraction == 0.1
    model = exp010().step.model
    assert isinstance(model, MobileNetV2.Config)
    assert model.activation is relu
    assert exp010().dataset.working_dir == "/datasets/fmnist"
    assert exp010().dataset.poison_fraction == 0.15


@pytest.mark.parametrize(
    "factory",
    [exp000, exp003, exp005, exp010],
    ids=["exp000", "exp003", "exp005", "exp010"],
)
def test_experiment_trains_and_evaluates_at_minimum_size(
    factory: ExperimentFactory[PoisonTrainLoop],
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path, classes=10)
    cfg = shrink(factory(), directory=tmp_path)
    cfg.dataset.poison_fraction = 0.5
    loop = cfg.make()
    loop.train()
    # Eight training images in two batches of four.
    assert loop.step.global_step == 2
    metric = loop.metrics_eval["poison"]
    assert isinstance(metric, PoisonRobustness)
    assert metric.total == {"test": 8, "poisoned": 4}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[PoisonTrainLoop],
) -> None:
    doc = inspect.getdoc(factory) or ""
    assert "Hypothesis:" in doc
    assert "References:" in doc
    assert "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")
    assert factory().experiment_name == factory.__name__


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    parents = {"exp000": [*ALL_EXPERIMENTS[1:5], exp005, exp010]}
    parents["exp005"] = ALL_EXPERIMENTS[6:10]
    parents["exp010"] = ALL_EXPERIMENTS[11:]
    for parent, children in parents.items():
        for child in children:
            doc = inspect.getdoc(child) or ""
            assert doc.startswith(f"{parent} "), child.__name__


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
