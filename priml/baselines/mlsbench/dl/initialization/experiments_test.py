"""Tests for the ``dl-weight-initialization`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest

from priml.baselines.mlsbench.dl.initialization import experiments
from priml.baselines.mlsbench.dl.initialization.experiments import (
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
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable
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

PER_TENSOR: Final = {"step.model.init_conv", "step.model.init_head"}
"""The per-tensor initialization slots."""

SETTING_FORKS: Final = {
    exp005: exp004,
    exp006: exp004,
    exp007: exp004,
    exp009: exp008,
    exp010: exp008,
    exp011: exp008,
}
"""Each baseline at another setting and that setting's pristine experiment."""


@pytest.mark.parametrize(
    ("factory", "switch"),
    [(exp004, settings.vgg16bn_cifar100), (exp008, settings.mobilenetv2_fmnist)],
    ids=["exp004", "exp008"],
)
def test_a_setting_experiment_is_the_switch_with_the_pristine_rule(
    factory: ExperimentFactory[VisionTrainLoop],
    switch: Callable[[VisionTrainLoop], VisionTrainLoop],
) -> None:
    switched = switch(exp000())
    switched.experiment_name = factory.__name__
    assert changed_fields(switched, factory()) == PER_TENSOR


@pytest.mark.parametrize(
    ("factory", "parent"),
    list(SETTING_FORKS.items()),
    ids=[f.__name__ for f in SETTING_FORKS],
)
def test_a_setting_fork_changes_only_the_per_tensor_initialization(
    factory: ExperimentFactory[VisionTrainLoop],
    parent: ExperimentFactory[VisionTrainLoop],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *PER_TENSOR}


@pytest.mark.parametrize(
    ("fixup", "kaiming"),
    [(exp006, exp005), (exp010, exp009)],
    ids=["exp006", "exp010"],
)
def test_fixup_off_a_resnet_is_kaiming_alone_as_upstream_gates_it(
    fixup: ExperimentFactory[VisionTrainLoop],
    kaiming: ExperimentFactory[VisionTrainLoop],
) -> None:
    assert changed_fields(kaiming(), fixup()) == {"experiment_name"}


def test_exp000_is_the_vision_recipe_with_small_normal_weights() -> None:
    assert changed_fields(vision_experiments.exp000(), exp000()) == {
        "study_name",
        *PER_TENSOR,
    }


@pytest.mark.parametrize(
    ("factory", "changed"),
    [
        (exp001, PER_TENSOR),
        (exp002, {*PER_TENSOR, "step.model.init"}),
        (exp003, PER_TENSOR),
    ],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_fork_changes_only_the_initialization(
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(exp000(), factory()) == {"experiment_name", *changed}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    # Fashion-MNIST's 10 classes, the fewest any setting has.
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


def test_exp000_runs_with_the_launch_scripts_settings() -> None:
    # The capture traces ``main`` at the script's argparse defaults. These are the
    # values ``tasks/dl-weight-initialization/scripts/resnet56_cifar100.sh`` passes
    # at the pinned commit, so a recipe drifting from the launch is caught here.
    cfg = exp000()
    group = cfg.step.make().optimizer.param_groups[0]
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.1, 0.9, 5e-4)
    assert cfg.dataset.batch_size == 128
    assert cfg.max_epochs == cfg.step.train_budget_epochs == 200


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
