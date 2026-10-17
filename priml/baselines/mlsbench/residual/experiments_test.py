"""Tests for the ``dl-residual-connection`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest

from priml.baselines.mlsbench.pytorch_vision import experiments as vision
from priml.baselines.mlsbench.pytorch_vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.pytorch_vision.model import ResNet
from priml.baselines.mlsbench.residual import experiments
from priml.baselines.mlsbench.residual.blocks import (
    StochasticDepthBlock,
    stochastic_depth,
)
from priml.baselines.mlsbench.residual.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
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
    ("factory", "changed"),
    [
        (exp001, {"step.model.block", "step.model.block.scale"}),
        (exp002, {"step.model.block", "step.model.block.gate"}),
    ],
    ids=["exp001", "exp002"],
)
def test_a_fork_changes_only_the_block(
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(exp000(), factory()) == {"experiment_name", *changed}


def test_exp003_changes_only_the_block_into_one_per_position() -> None:
    changed = changed_fields(exp000(), exp003()) - {"experiment_name"}
    assert changed
    assert all(path.startswith("step.model.block") for path in changed)
    model = exp003().step.model
    assert isinstance(model, ResNet.Config)
    assert model.block == stochastic_depth(
        (16, 32, 64),
        (9, 9, 9),
        survival_last=0.5,
    )
    assert isinstance(model.block, list)
    blocks = [b for b in model.block if isinstance(b, StochasticDepthBlock.Config)]
    assert len(blocks) == 27
    assert [b.depth for b in blocks] == [1] * 9 + list(range(2, 20))
    assert {b.num_layers for b in blocks} == {19}


@pytest.mark.parametrize(
    "factory",
    [exp000, exp001, exp002],
    ids=["exp000", "exp001", "exp002"],
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


def test_exp003_trains_at_minimum_size(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    config = shrink(exp003(), directory=tmp_path)
    model = config.step.model
    assert isinstance(model, ResNet.Config)
    model.block = stochastic_depth(
        model.channels_hidden,
        model.blocks_per_stage,
        survival_last=0.5,
    )
    loop = config.make()
    loop.train()
    assert loop.step.global_step == 2
    assert exp003().experiment_name == "exp003"


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
