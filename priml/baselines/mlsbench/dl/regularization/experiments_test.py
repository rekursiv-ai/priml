"""Tests for the ``dl-regularization`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.mlsbench.dl.regularization import experiments
from priml.baselines.mlsbench.dl.regularization.experiments import (
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
from priml.baselines.mlsbench.dl.regularization.train_step import (
    RegularizedTrainStep,
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
from priml.optimizers.parameter_filter import complement, with_ndim
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

DROPBLOCK: Final = {
    "step.regularization.func",
    "step.regularization.block_size",
    "step.regularization.strength",
    "step.regularization.delay",
}
CONFIDENCE: Final = {
    "step.regularization.func",
    "step.regularization.beta",
    "step.regularization.eps",
}
ORTHOGONAL: Final = {
    "step.regularization.func",
    "step.regularization.strength",
    "step.regularization.select",
}

FORKS: Final = {
    exp001: (exp000, DROPBLOCK),
    exp002: (exp000, CONFIDENCE),
    exp003: (exp000, ORTHOGONAL),
    exp005: (exp004, DROPBLOCK),
    exp006: (exp004, CONFIDENCE),
    exp007: (exp004, ORTHOGONAL),
    exp009: (exp008, DROPBLOCK),
    exp010: (exp008, CONFIDENCE),
    exp011: (exp008, ORTHOGONAL),
}
"""Each baseline fork, its setting's unregularized parent, and the fields it sets."""


def test_exp000_is_the_vision_recipe_on_the_regularized_step() -> None:
    cfg = exp000()
    assert type(cfg.step) is RegularizedTrainStep.Config
    assert changed_fields(vision_experiments.exp000(), cfg) == {
        "study_name",
        "step",
        "step.regularization",
        "step.regularization.func",
    }


@pytest.mark.parametrize(
    ("factory", "parent", "changed"),
    [(child, parent, changed) for child, (parent, changed) in FORKS.items()],
    ids=[f.__name__ for f in FORKS],
)
def test_a_fork_changes_only_the_regularizer(
    factory: ExperimentFactory[VisionTrainLoop],
    parent: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *changed}


@pytest.mark.parametrize("factory", [exp002, exp006, exp010])
def test_the_confidence_penalty_is_the_baselines_beta_and_eps(
    factory: ExperimentFactory[VisionTrainLoop],
) -> None:
    step = factory().step
    assert isinstance(step, RegularizedTrainStep.Config)
    reg = step.regularization
    assert isinstance(reg, PartialConfig)
    # The edit's ``-0.1 * entropy`` and ``torch.log(probs + 1e-8)``.
    assert reg.deferred_call()[2] == {"beta": 0.1, "eps": 1e-8}


def test_exp004_is_the_vgg16bn_setting_with_the_scripts_vgg_schedule() -> None:
    expected = settings.vgg16bn_cifar100(exp000())
    expected.experiment_name = "exp004"
    assert changed_fields(expected, exp004()) == {
        "step.schedule.func",
        "step.schedule.warmup_epochs",
    }


def test_exp008_is_the_mobilenetv2_fmnist_setting() -> None:
    expected = settings.mobilenetv2_fmnist(exp000())
    expected.experiment_name = "exp008"
    assert changed_fields(expected, exp008()) == set()


@pytest.mark.parametrize(
    ("factory", "takes_every_conv"),
    [(exp007, False), (exp011, True)],
    ids=["exp007", "exp011"],
)
def test_orthogonal_selects_as_the_scripts_conv_name_filter_does(
    factory: ExperimentFactory[VisionTrainLoop],
    *,
    takes_every_conv: bool,
    tmp_path: Path,
) -> None:
    """The script takes 4-D weights named ``conv``: none of VGG's, all of MobileNetV2's."""
    step = shrink(factory(), directory=tmp_path).step
    assert isinstance(step, RegularizedTrainStep.Config)
    reg = step.regularization
    assert isinstance(reg, PartialConfig)
    _, _, keywords = reg.deferred_call()
    select = keywords["select"]
    assert isinstance(select, complement | with_ndim)
    model = step.model.make()
    taken = {n for n, p in model.named_parameters() if select(n, p)}
    convs = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }
    assert convs
    assert taken == (convs if takes_every_conv else set())


@pytest.mark.parametrize(
    ("factory", "settings"),
    [
        (exp001, {"block_size": 3, "strength": 1e-4, "delay": 0.2}),
        (exp002, {"beta": 0.1, "eps": 1e-8}),
        (exp003, {"strength": 1e-4}),
    ],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_fork_takes_its_baselines_settings(
    factory: ExperimentFactory[VisionTrainLoop],
    settings: dict[str, float],
) -> None:
    step = factory().step
    assert isinstance(step, RegularizedTrainStep.Config)
    assert {name: getattr(step.regularization, name) for name in settings} == settings


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
    assert "b873944969c53b01e4ee6e843c77767b8c702485" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for child, (parent, _) in FORKS.items():
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
