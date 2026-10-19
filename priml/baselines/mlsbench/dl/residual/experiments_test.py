"""Tests for the ``dl-residual-connection`` experiment chain.

``exp000`` is pinned by value and by its golden; each fork is checked as a
delta, exactly which fields differ from ``exp000``, so "one slot per idea" is a
property of the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import inspect

import pytest

from priml.baselines.mlsbench.dl.residual import experiments
from priml.baselines.mlsbench.dl.residual.blocks import StochasticDepthBlock
from priml.baselines.mlsbench.dl.residual.experiments import (
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
from priml.baselines.mlsbench.dl.vision import experiments as vision_experiments
from priml.baselines.mlsbench.dl.vision.experiments_test import (
    changed_fields,
    shrink,
    write_dataset,
)
from priml.baselines.mlsbench.dl.vision.model import ResNet, sized_blocks
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

BLOCK_DELTAS: Final = {
    "pre_activation": {"step.model.block", "step.model.block.scale"},
    "gated_residual": {"step.model.block", "step.model.block.gate"},
    "stochastic_depth": {
        "step.model.block",
        "step.model.block.depth",
        "step.model.block.num_layers",
        "step.model.block.survival_last",
        "step.model.block.reset_width",
    },
}
"""What each baseline's fork changes on top of its setting's experiment."""


def test_exp000_is_the_vision_recipe() -> None:
    assert changed_fields(vision_experiments.exp000(), exp000()) == {"study_name"}


@pytest.mark.parametrize(
    ("parent", "factory", "changed"),
    [
        (exp000, exp001, BLOCK_DELTAS["pre_activation"]),
        (exp000, exp002, BLOCK_DELTAS["gated_residual"]),
        (exp000, exp003, BLOCK_DELTAS["stochastic_depth"]),
        (exp004, exp005, BLOCK_DELTAS["pre_activation"]),
        (exp004, exp006, BLOCK_DELTAS["gated_residual"]),
        (exp004, exp007, BLOCK_DELTAS["stochastic_depth"]),
        (exp008, exp009, BLOCK_DELTAS["pre_activation"]),
        (exp008, exp010, BLOCK_DELTAS["gated_residual"]),
        (exp008, exp011, BLOCK_DELTAS["stochastic_depth"]),
    ],
    ids=[f"exp{n:03d}" for n in (1, 2, 3, 5, 6, 7, 9, 10, 11)],
)
def test_a_fork_changes_only_the_block(
    parent: ExperimentFactory[VisionTrainLoop],
    factory: ExperimentFactory[VisionTrainLoop],
    changed: set[str],
) -> None:
    assert changed_fields(parent(), factory()) == {"experiment_name", *changed}


def test_exp004_moves_to_resnet20_on_cifar10_and_nothing_else() -> None:
    assert changed_fields(exp000(), exp004()) == {
        "experiment_name",
        "step.model.channels_out",
        *(f"step.model.blocks_per_stage[{i}]" for i in range(3)),
        "dataset.working_dir",
        *(f"dataset.{name}[{i}]" for name in ("mean", "std") for i in range(3)),
    }
    cfg = exp004()
    assert isinstance(cfg.step.model, ResNet.Config)
    assert cfg.step.model.blocks_per_stage == (3, 3, 3)
    assert cfg.step.model.channels_out == 10
    assert cfg.dataset.mean == (0.4914, 0.4822, 0.4465)
    assert cfg.dataset.std == (0.2023, 0.1994, 0.2010)


def test_exp008_moves_to_resnet110_and_nothing_else() -> None:
    assert changed_fields(exp000(), exp008()) == {
        "experiment_name",
        *(f"step.model.blocks_per_stage[{i}]" for i in range(3)),
    }
    model = exp008().step.model
    assert isinstance(model, ResNet.Config)
    assert model.blocks_per_stage == (18, 18, 18)


@pytest.mark.parametrize(
    ("factory", "per_stage"),
    [(exp003, 9), (exp007, 3), (exp011, 18)],
    ids=["resnet56", "resnet20", "resnet110"],
)
def test_stochastic_depth_takes_upstreams_positions_at_each_depth(
    factory: ExperimentFactory[VisionTrainLoop],
    per_stage: int,
) -> None:
    model = factory().step.model
    assert isinstance(model, ResNet.Config)
    template = model.block
    assert isinstance(template, StochasticDepthBlock.Config)
    assert template.survival_last == 0.5
    blocks = [
        b
        for stage in sized_blocks(model)
        for b, _ in stage
        if isinstance(b, StochasticDepthBlock.Config)
    ]
    last = 2 * per_stage + 1
    assert [b.depth for b in blocks] == [1] * per_stage + list(range(2, last + 1))
    assert {b.num_layers for b in blocks} == {last}


@pytest.mark.parametrize(
    "factory",
    ALL_EXPERIMENTS,
    ids=[f.__name__ for f in ALL_EXPERIMENTS],
)
def test_experiment_trains_at_minimum_size(
    factory: ExperimentFactory[VisionTrainLoop],
    tmp_path: Path,
) -> None:
    model = factory().step.model
    assert isinstance(model, ResNet.Config)
    write_dataset(tmp_path, classes=model.channels_out)
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


@pytest.mark.parametrize(
    ("parent", "children"),
    [
        ("exp000", (exp001, exp002, exp003)),
        ("exp004", (exp005, exp006, exp007)),
        ("exp008", (exp009, exp010, exp011)),
    ],
)
def test_each_fork_names_its_parent_in_the_first_line(
    parent: str,
    children: tuple[ExperimentFactory[VisionTrainLoop], ...],
) -> None:
    for child in children:
        assert (inspect.getdoc(child) or "").startswith(f"{parent} + "), child.__name__


def test_each_setting_names_exp000_as_its_parent() -> None:
    for setting in (exp004, exp008):
        assert (inspect.getdoc(setting) or "").startswith("exp000 at the ")


def test_exp000_matches_its_golden_config() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
