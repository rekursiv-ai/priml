"""Tests for the ``security-machine-unlearning`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so one slot per idea is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.baselines.mlsbench.tl.machine_unlearning import experiments
from priml.baselines.mlsbench.tl.machine_unlearning.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
)
from priml.baselines.mlsbench.tl.machine_unlearning.methods import (
    BadTeacher,
    NegativeGradient,
    RetainFinetune,
    Scrub,
)
from priml.model.swiglu import relu
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from priml.baselines.mlsbench.tl.machine_unlearning.unlearning import (
        MachineUnlearning,
    )
    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[MachineUnlearning.Config]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
]


def changed_fields(
    parent: MachineUnlearning.Config,
    child: MachineUnlearning.Config,
) -> set[str]:
    """Return the field paths ``udiff`` reports between two unfinalized configs."""
    table = parent.udiff(child, mode="fields", finalize=False)
    return {line.split("│")[0].strip() for line in table.splitlines()[1:]}


def test_exp000_pins_the_resnet20_cifar10_class0_recipe() -> None:
    cfg = exp000()
    assert isinstance(cfg.pretrain.model, ResNet.Config)
    assert cfg.pretrain.model.blocks_per_stage == (3, 3, 3)
    assert cfg.pretrain.model.channels_out == 10
    assert cfg.pretrain.train_budget_epochs == 80
    assert cfg.unlearn_epochs == 20
    assert cfg.dataset.forget_class == 0
    assert cfg.dataset.batch_size == 128
    assert cfg.seed == 42
    assert isinstance(cfg.method, RetainFinetune.Config)
    step = cfg.pretrain.make()
    assert isinstance(step.optimizer, torch.optim.SGD)
    group = step.optimizer.param_groups[0]
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.1, 0.9, 5e-4)
    adam = cfg.unlearn_optimizer.make()(torch.nn.Linear(2, 3).parameters())
    assert isinstance(adam, torch.optim.Adam)
    assert adam.param_groups[0]["lr"] == 0.001


@pytest.mark.parametrize(
    ("factory", "changed", "method"),
    [
        (exp001, set[str](), RetainFinetune.Config),
        (exp002, {"method", "method.forget_weight"}, NegativeGradient.Config),
        (exp003, {"method", "method.reinit", "method.temperature"}, BadTeacher.Config),
        (
            exp004,
            {
                "method",
                "method.alpha",
                "method.gamma",
                "method.max_epochs",
                "method.temperature",
            },
            Scrub.Config,
        ),
    ],
    ids=["exp001", "exp002", "exp003", "exp004"],
)
def test_a_baseline_changes_only_the_method(
    factory: ExperimentFactory[MachineUnlearning.Config],
    changed: set[str],
    method: type,
) -> None:
    assert changed_fields(exp000(), factory()) == {"experiment_name", *changed}
    assert isinstance(factory().method, method)


@pytest.mark.parametrize(
    ("factory", "network", "classes", "directory"),
    [
        (exp005, VGG16BN.Config, 100, "/datasets/cifar100"),
        (exp006, MobileNetV2.Config, 10, "/datasets/fmnist"),
    ],
    ids=["exp005", "exp006"],
)
def test_a_setting_changes_only_the_network_and_the_dataset(
    factory: ExperimentFactory[MachineUnlearning.Config],
    network: type[VGG16BN.Config | MobileNetV2.Config],
    classes: int,
    directory: str,
) -> None:
    changed = changed_fields(exp000(), factory())
    assert changed - {"experiment_name"}
    assert all(
        path.startswith(("pretrain.model", "dataset."))
        for path in changed - {"experiment_name"}
    )
    cfg = factory()
    model = cfg.pretrain.model
    assert isinstance(model, (VGG16BN.Config, MobileNetV2.Config))
    assert type(model) is network
    assert model.channels_out == classes
    assert cfg.dataset.working_dir == directory
    assert isinstance(cfg.method, RetainFinetune.Config)


def test_the_mobilenet_setting_activates_with_relu_as_this_harness_does() -> None:
    model = exp006().pretrain.model
    assert isinstance(model, MobileNetV2.Config)
    assert model.activation is relu


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_every_experiment_states_its_hypothesis_sources_and_pending_results(
    factory: ExperimentFactory[MachineUnlearning.Config],
) -> None:
    doc = inspect.getdoc(factory) or ""
    assert "Hypothesis:" in doc
    assert "References:" in doc
    assert "80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")
    assert factory().experiment_name == factory.__name__
    assert factory().copy_tree().finalize() is not None


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for child in ALL_EXPERIMENTS[1:]:
        assert (inspect.getdoc(child) or "").startswith("exp000 "), child.__name__


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
