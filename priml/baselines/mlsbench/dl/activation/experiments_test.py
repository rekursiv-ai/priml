"""Tests for the ``dl-activation-function`` experiment chain.

``exp000`` is pinned by its delta from the vision recipe and by its golden;
each fork is checked as a delta, exactly which fields differ from ``exp000``,
so "one slot per idea" is a property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

import inspect

from configgle import Fig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.activation import experiments
from priml.baselines.mlsbench.dl.activation.experiments import (
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
)
from priml.baselines.mlsbench.dl.vision import (
    experiments as vision_experiments,
    settings,
)
from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.cost import cost
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[VisionTrainLoop]]] = [
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
]

SETTING_FORKS: Final = {
    exp005: exp004,
    exp006: exp004,
    exp007: exp004,
    exp009: exp008,
    exp010: exp008,
    exp011: exp008,
}
"""Each baseline at another setting and that setting's ReLU experiment."""


def test_exp004_is_the_vgg16bn_cifar100_switch_alone() -> None:
    switched = settings.vgg16bn_cifar100(exp000())
    switched.experiment_name = "exp004"
    assert changed_fields(switched, exp004()) == set()


def test_exp008_is_the_mobilenetv2_fmnist_switch_with_the_scripts_relu() -> None:
    switched = settings.mobilenetv2_fmnist(exp000())
    switched.experiment_name = "exp008"
    assert changed_fields(switched, exp008()) == {"step.model.activation"}


@pytest.mark.parametrize(
    ("factory", "parent"),
    list(SETTING_FORKS.items()),
    ids=[f.__name__ for f in SETTING_FORKS],
)
def test_a_setting_fork_changes_only_the_activation(
    factory: ExperimentFactory[VisionTrainLoop],
    parent: ExperimentFactory[VisionTrainLoop],
) -> None:
    assert changed_fields(parent(), factory()) == {
        "experiment_name",
        "step.model.activation",
    }


class Scaled(nn.Module):
    """ReLU times a learned scalar: an activation with a parameter of its own."""

    class Config(Fig["Scaled"]):
        """No settings; a config so the activation is built once per use."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.scale * torch.relu(x)


def test_exp000_is_the_vision_recipe_at_resnet20_on_cifar10() -> None:
    assert changed_fields(vision_experiments.exp000(), exp000()) == {
        "study_name",
        "step.model.channels_out",
        "dataset.working_dir",
        *(
            f"{name}[{index}]"
            for name in (
                "step.model.blocks_per_stage",
                "dataset.mean",
                "dataset.std",
            )
            for index in range(3)
        ),
    }


@pytest.mark.parametrize(
    "factory",
    [exp001, exp002, exp003],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_fork_changes_only_the_activation(
    factory: ExperimentFactory[VisionTrainLoop],
) -> None:
    assert changed_fields(exp000(), factory()) == {
        "experiment_name",
        "step.model.activation",
    }


def test_every_activation_costs_through_the_network() -> None:
    for factory in ALL_EXPERIMENTS:
        model = factory().step.model
        assert isinstance(model, ResNet.Config | VGG16BN.Config | MobileNetV2.Config)
        total = cost(model.copy_tree().finalize(), batch_size=2, dtype=None)
        assert sum(total["flops"].cells.values()) > 0


def test_one_activation_per_stem_and_per_block_as_the_script_builds_them() -> None:
    config = exp000().step.model
    assert isinstance(config, ResNet.Config)
    config.activation = Scaled.Config()
    model = config.make()
    assert sum(isinstance(m, Scaled) for m in model.modules()) == 1 + 9
    names = [name for name, _ in model.named_parameters()]
    # The script registers each block's activation after its second norm and
    # before the shortcut, and the stem's after the stem's norm.
    assert names[:4] == ["stem.weight", "norm.weight", "norm.bias", "act.scale"]
    first = [n.removeprefix("stages.1.0.") for n in names if "stages.1.0." in n]
    assert first.index("act.scale") == first.index("norm2.bias") + 1
    assert first.index("act.scale") < first.index("shortcut.0.weight")


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path, classes=10)
    loop = shrink(factory(), directory=tmp_path).make()
    loop.train()
    assert loop.step.global_step == 2
    assert factory().experiment_name == factory.__name__


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
    assert factory.__name__ in (inspect.getdoc(experiments) or "")


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    parents = dict.fromkeys((exp001, exp002, exp003), exp000) | SETTING_FORKS
    for child, parent in parents.items():
        assert (inspect.getdoc(child) or "").startswith(f"{parent.__name__} + "), (
            child.__name__
        )
    for setting in (exp004, exp008):
        assert (inspect.getdoc(setting) or "").startswith("exp000 at the "), (
            setting.__name__
        )


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
