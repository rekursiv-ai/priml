"""Tests for the ``security-membership-inference-defense`` experiment chain.

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

from priml.baselines.mlsbench.dl.classification_loss.losses import (
    label_smoothing,
)
from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import cross_entropy
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.baselines.mlsbench.tl.membership_inference_defense import experiments
from priml.baselines.mlsbench.tl.membership_inference_defense.data import (
    MembershipData,
    normalize_only,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.defenses import (
    confidence_penalty,
    relaxloss,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.experiments import (
    MembershipTrainLoop,
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
from priml.baselines.mlsbench.tl.membership_inference_defense.metrics import (
    MembershipInference,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.schedules import (
    multistep,
)
from priml.model.swiglu import relu
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from pathlib import Path

    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[MembershipTrainLoop]]] = [
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

DEFENSE: Final = {
    "erm": set[str](),
    "label_smoothing": {"step.loss.func", "step.loss.smoothing"},
    "confidence_penalty": {
        "step.loss.func",
        "step.loss.entropy_weight",
        "step.loss.min_probability",
    },
    "relaxloss": {"step.loss.func", "step.loss.alpha", "step.loss.upper"},
}
"""The fields each baseline's fork changes on its setting's pristine experiment."""


def bound(config: object) -> dict[str, object]:
    """Return the function and keywords a ``PartialConfig`` binds."""
    assert isinstance(config, PartialConfig)
    made = config.make()
    assert isinstance(made, functools.partial)
    return {"fn": made.func, **made.keywords}


def defense(factory: ExperimentFactory[MembershipTrainLoop]) -> dict[str, object]:
    """Return the function and keywords of an experiment's defense."""
    return bound(factory().step.loss)


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
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.1, 0.9, 1e-4)
    assert defense(exp000) == {"fn": cross_entropy}
    assert bound(cfg.step.schedule) == {
        "fn": multistep,
        "milestones": [150, 225],
        "gamma": 0.1,
    }
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 300
    assert isinstance(cfg.dataset, MembershipData.Config)
    assert cfg.dataset.batch_size == cfg.dataset.eval_batch_size == 128
    assert cfg.dataset.split_seed == cfg.seed == 42
    assert cfg.dataset.mean == (0.4914, 0.4822, 0.4465)
    assert cfg.dataset.std == (0.2023, 0.1994, 0.2010)
    assert bound(cfg.dataset.augment) == {"fn": normalize_only}


def test_exp000_evaluates_once_after_training() -> None:
    cfg = exp000()
    assert (cfg.num_steps_eval, cfg.eval_every_epoch) == (-1, False)
    assert list(cfg.metrics_eval) == ["privacy"]
    assert isinstance(cfg.metrics_eval["privacy"], MembershipInference.Config)


@pytest.mark.parametrize(
    ("parent", "forks"),
    [
        (exp000, (exp001, exp002, exp003, exp004)),
        (exp005, (exp006, exp007, exp008, exp009)),
        (exp010, (exp011, exp012, exp013, exp014)),
    ],
    ids=["resnet20-cifar10", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
)
def test_a_fork_changes_only_the_defense(
    parent: ExperimentFactory[MembershipTrainLoop],
    forks: tuple[ExperimentFactory[MembershipTrainLoop], ...],
) -> None:
    for fork, changed in zip(forks, DEFENSE.values(), strict=True):
        assert changed_fields(parent(), fork()) == {"experiment_name", *changed}


@pytest.mark.parametrize(
    ("forks", "alpha"),
    [((exp002, exp003, exp004), 1.0), ((exp007, exp008, exp009), 0.5)],
    ids=["ten-classes", "hundred-classes"],
)
def test_each_defense_carries_the_baselines_settings(
    forks: tuple[ExperimentFactory[MembershipTrainLoop], ...],
    alpha: float,
) -> None:
    smoothing, penalty, relax = forks
    assert defense(smoothing) == {"fn": label_smoothing, "smoothing": 0.1}
    assert defense(penalty) == {
        "fn": confidence_penalty,
        "entropy_weight": 0.1,
        "min_probability": 1e-8,
    }
    assert defense(relax) == {"fn": relaxloss, "alpha": alpha, "upper": 1.0}


@pytest.mark.parametrize("setting", [exp005, exp010], ids=["exp005", "exp010"])
def test_each_setting_changes_only_the_network_and_the_data(
    setting: ExperimentFactory[MembershipTrainLoop],
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
    }
    assert {"experiment_name", "step.model", "dataset.working_dir"} <= roots


def test_the_settings_build_the_tasks_networks() -> None:
    assert isinstance(exp005().step.model, VGG16BN.Config)
    model = exp010().step.model
    assert isinstance(model, MobileNetV2.Config)
    assert model.activation is relu
    assert exp010().dataset.working_dir == "/datasets/fmnist"


def test_exp014_takes_the_ten_class_level_of_relaxloss() -> None:
    assert defense(exp014) == {"fn": relaxloss, "alpha": 1.0, "upper": 1.0}


@pytest.mark.parametrize(
    "factory",
    [exp000, exp004, exp005, exp010],
    ids=["exp000", "exp004", "exp005", "exp010"],
)
def test_experiment_trains_and_attacks_at_minimum_size(
    factory: ExperimentFactory[MembershipTrainLoop],
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path, classes=10)
    loop = shrink(factory(), directory=tmp_path).make()
    loop.train()
    # Eight training images: four members in one batch of four.
    assert loop.step.global_step == 1
    metric = loop.metrics_eval["privacy"]
    assert isinstance(metric, MembershipInference)
    assert metric.total == {"test": 8, "member": 4, "nonmember": 4}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[MembershipTrainLoop],
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
