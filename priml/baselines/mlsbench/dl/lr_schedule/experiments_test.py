"""Tests for the ``dl-lr-schedule`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest

from priml.baselines.mlsbench.dl.lr_schedule import experiments
from priml.baselines.mlsbench.dl.lr_schedule.experiments import (
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

FORKS: Final[dict[str, tuple[ExperimentFactory[VisionTrainLoop], ...]]] = {
    "exp000": (exp001, exp002, exp003, exp004, exp008),
    "exp004": (exp005, exp006, exp007),
    "exp008": (exp009, exp010, exp011),
}
"""Each parent's direct forks, so every first line names its parent."""

SCHEDULE_DELTAS: Final = [
    {"step.schedule.func"},
    {"step.schedule.func", "step.schedule.warmup_epochs"},
    {
        "step.schedule.func",
        "step.schedule.pct_start",
        "step.schedule.div_factor",
        "step.schedule.final_div",
    },
]
"""The fields the cosine, warmup-cosine and one-cycle forks change, in order."""


def test_exp000_is_the_vision_recipe_with_a_constant_rate() -> None:
    assert changed_fields(vision_experiments.exp000(), exp000()) == {
        "study_name",
        "step.schedule.func",
    }


@pytest.mark.parametrize(
    ("parent", "factory", "changed"),
    [
        (parent, factory, changed)
        for parent, forks in (
            (exp000, (exp001, exp002, exp003)),
            (exp004, (exp005, exp006, exp007)),
            (exp008, (exp009, exp010, exp011)),
        )
        for factory, changed in zip(forks, SCHEDULE_DELTAS, strict=True)
    ],
    ids=[f"exp{n:03d}" for n in (1, 2, 3, 5, 6, 7, 9, 10, 11)],
)
def test_a_fork_changes_only_the_schedule(
    parent: ExperimentFactory[VisionTrainLoop],
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *changed}


@pytest.mark.parametrize(
    ("factory", "switch"),
    [
        (exp004, settings.resnet20_cifar10),
        (exp008, settings.mobilenetv2_fmnist),
    ],
    ids=["exp004", "exp008"],
)
def test_a_setting_fork_changes_only_the_network_and_the_dataset(
    factory: ExperimentFactory[VisionTrainLoop],
    switch: Callable[[VisionTrainLoop], VisionTrainLoop],
) -> None:
    changed = changed_fields(exp000(), factory())
    assert changed == changed_fields(exp000(), switch(exp000())) | {"experiment_name"}
    assert all(
        path == "experiment_name" or path.startswith(("step.model", "dataset."))
        for path in changed
    )


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    # Ten labels fit every setting's head: CIFAR-10's and Fashion-MNIST's.
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
    assert "b873944969c53b01e4ee6e843c77767b8c702485" in doc
    assert doc.rstrip().endswith("Results:\n  TBD.")
    assert factory.__name__ in (inspect.getdoc(experiments) or "")


def test_each_fork_names_its_parent_in_the_first_line() -> None:
    for parent, children in FORKS.items():
        for child in children:
            assert (inspect.getdoc(child) or "").startswith(f"{parent} "), (
                child.__name__
            )


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
