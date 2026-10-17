"""Tests for the ``dl-weight-initialization`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest

from priml.baselines.mlsbench.initialization import experiments
from priml.baselines.mlsbench.initialization.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.pytorch_vision import experiments as vision
from priml.baselines.mlsbench.pytorch_vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop
    from priml.testing.experiments import ExperimentFactory

ALL_EXPERIMENTS: Final[list[ExperimentFactory[VisionTrainLoop]]] = [
    exp000,
    exp001,
    exp002,
    exp003,
]

PER_TENSOR: Final = {"step.model.init_conv", "step.model.init_head"}
"""The per-tensor initialization slots."""


def test_exp000_is_the_vision_recipe_with_small_normal_weights() -> None:
    assert changed_fields(vision.exp000(), exp000()) == {"study_name", *PER_TENSOR}


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
    write_dataset(tmp_path)
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
    for child in (exp001, exp002, exp003):
        assert (inspect.getdoc(child) or "").startswith("exp000 + "), child.__name__


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
