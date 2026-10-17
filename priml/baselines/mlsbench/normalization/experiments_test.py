"""Tests for the ``dl-normalization`` experiment chain.

``exp000`` is pinned by its golden; each fork is checked as a delta, exactly
which fields differ from ``exp000``, so "one slot per idea" is a property of
the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest

from priml.baselines.mlsbench.normalization import experiments
from priml.baselines.mlsbench.normalization.experiments import (
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


def test_exp000_is_the_vision_recipe() -> None:
    assert changed_fields(vision.exp000(), exp000()) == {"study_name"}


@pytest.mark.parametrize(
    "factory",
    [exp001, exp002, exp003],
    ids=["exp001", "exp002", "exp003"],
)
def test_a_fork_changes_only_the_norm(
    factory: ExperimentFactory[VisionTrainLoop],
) -> None:
    assert changed_fields(exp000(), factory()) == {
        "experiment_name",
        "step.model.norm",
    }


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
