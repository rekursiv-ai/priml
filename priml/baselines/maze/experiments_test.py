"""Tests for the Maze-Hard ladder.

Each fork is checked against its parent field by field: every changed path must
belong to a change its docstring declares, and every declared change must move
something. A fork that quietly moved a second knob fails here rather than
producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import inspect

import pytest

from priml.baselines.arcagi.model import (
    ConvSwiGLU,
    SingleLatentRecurrence,
    depthwise_conv,
    depthwise_shift,
)
from priml.baselines.maze import experiments
from priml.baselines.maze.data import CANONICAL, ORIGINAL
from priml.baselines.maze.scripts.prepare_data import DEFAULT_DIRECTORY
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import DeepRecurrence
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.model.attention.rope import RoPE, RoPEMixed
from priml.model.transformer.block import TransformerBlock
from priml.optimizers import AdamATan2
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.maze.experiments import MazeTrainLoop


LADDER: Final[dict[str, Callable[[], MazeTrainLoop]]] = {
    "exp000": experiments.exp000,
    "exp001": experiments.exp001,
    "exp002": experiments.exp002,
    "exp003": experiments.exp003,
    "exp_smoke": experiments.exp_smoke,
}

FORKS: Final[dict[str, tuple[str, frozenset[str]]]] = {
    "exp001": (
        "exp000",
        frozenset(
            {
                "dataset.working_dir",
                "step.model.num_layers",
                "step.model.recurrence",
                "step.model.block.ffn",
                "step.model.rope",
                "step.model.rope_grid_shape",
                "step.model.prefix.num_tokens",
                # Derived from the prefix length.
                "step.model.num_prefix_tokens",
                "step.pool.seq_len",
                "step.optimizer",
                "step.boundary",
                "step.total_train_steps",
                "max_steps",
            },
        ),
    ),
    "exp002": (
        "exp001",
        frozenset({"step.model.recurrence", "step.model.block.ffn.short_conv"}),
    ),
    "exp003": (
        "exp002",
        frozenset(
            {
                "step.model.recurrence.inner_grad_loops",
                "step.total_train_steps",
                "max_steps",
                "max_time",
            },
        ),
    ),
}
"""Each fork's parent and the config paths its docstring's bullets move."""


@pytest.mark.parametrize("name", LADDER)
def test_every_experiment_finalizes_under_its_own_name(name: str) -> None:
    factory = LADDER[name]
    assert not inspect.signature(factory).parameters
    config = factory().copy_tree().finalize()
    assert (config.study_name, config.experiment_name) == ("maze", name)


@pytest.mark.parametrize("name", ["exp000", "exp001", "exp002", "exp003", "exp_smoke"])
def test_every_experiment_matches_its_golden(name: str) -> None:
    assert_pprint_golden(test_file=__file__, name=name, config=LADDER[name]())


@pytest.mark.parametrize("name", FORKS)
def test_each_fork_names_its_parent(name: str) -> None:
    parent, _ = FORKS[name]
    assert inspect.getdoc(LADDER[name]) is not None
    assert str(inspect.getdoc(LADDER[name])).startswith(
        f"Fork {parent} with the following changes.",
    )


@pytest.mark.parametrize("name", FORKS)
def test_each_fork_moves_exactly_its_declared_fields(name: str) -> None:
    parent, declared = FORKS[name]
    rows = LADDER[parent]().udiff(LADDER[name](), mode="fields", color=False)
    moved = set[str]()
    for row in rows.splitlines()[1:]:
        path, before, after = (cell.strip() for cell in row.split("│"))
        if before.replace(parent, name) != after:
            moved.add(path)
    owners = {path: [root for root in declared if _under(path, root)] for path in moved}
    assert not [path for path, roots in owners.items() if not roots]
    assert {root for roots in owners.values() for root in roots} == declared


def test_the_baseline_is_the_reference_recipe() -> None:
    """Two-latent recursion, 2-D rotary positions, AdamATan2, stored labels."""
    config = experiments.exp000().copy_tree().finalize()
    model = config.step.model
    assert isinstance(model.recurrence, DeepRecurrence.Config)
    assert not isinstance(model.recurrence, SingleLatentRecurrence.Config)
    assert (model.recurrence.slow_cycles, model.recurrence.fast_cycles) == (3, 4)
    assert isinstance(model.rope, RoPEMixed.Config)
    assert model.rope_grid_shape == (30, 30)
    assert isinstance(model.prefix, SparsePuzzleEmbedding.Config)
    assert model.prefix.num_tokens == 16
    assert isinstance(config.step.optimizer, AdamATan2.Config)
    assert config.step.boundary is None
    assert config.dataset.working_dir == Path(DEFAULT_DIRECTORY) / ORIGINAL
    assert (config.max_steps, config.max_time) == (16_000, 3_600.0)


def test_exp001_moves_to_one_latent_on_canonical_labels() -> None:
    config = experiments.exp001().copy_tree().finalize()
    model = config.step.model
    assert model.num_layers == 4
    assert isinstance(model.recurrence, SingleLatentRecurrence.Config)
    recurrence = model.recurrence
    assert (recurrence.slow_cycles, recurrence.fast_cycles) == (2, 3)
    assert recurrence.inner_grad_loops == 0
    assert isinstance(model.block, TransformerBlock.Config)
    assert isinstance(model.block.ffn, ConvSwiGLU.Config)
    assert model.block.ffn.short_conv is depthwise_conv
    assert type(model.rope) is RoPE.Config
    assert model.rope_grid_shape == (900,)
    assert isinstance(model.prefix, SparsePuzzleEmbedding.Config)
    assert model.prefix.num_tokens == 1
    optimizer = config.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    muon = [
        member for member in optimizer.optimizers if isinstance(member, Muon.Config)
    ]
    assert [member.lr for member in muon] == [5e-3, 5e-3]
    assert config.step.boundary is not None
    assert config.step.boundary.weight == 0.1
    assert config.step.boundary.grid_shape == (30, 30)
    assert config.dataset.working_dir == Path(DEFAULT_DIRECTORY) / CANONICAL
    assert config.max_steps == config.step.total_train_steps == 24_000


def test_exp002_runs_one_deep_pass_with_five_carrying_gradient() -> None:
    model = experiments.exp002().step.model
    assert isinstance(model.recurrence, SingleLatentRecurrence.Config)
    recurrence = model.recurrence
    assert (recurrence.slow_cycles, recurrence.fast_cycles) == (1, 8)
    assert recurrence.inner_grad_loops == 5
    assert isinstance(model.block, TransformerBlock.Config)
    assert isinstance(model.block.ffn, ConvSwiGLU.Config)
    assert model.block.ffn.short_conv is depthwise_shift


def test_exp003_trains_four_hours_with_three_carrying_gradient() -> None:
    config = experiments.exp003()
    assert isinstance(config.step.model.recurrence, SingleLatentRecurrence.Config)
    assert config.step.model.recurrence.inner_grad_loops == 3
    assert config.max_time == 4 * 3_600.0
    assert config.max_steps == config.step.total_train_steps == 80_000


def test_the_dataset_lays_out_the_model() -> None:
    """Grid and batch size are set once, on the dataset, and reach every user."""
    config = experiments.exp001()
    config.dataset.spec.grid_shape = (5, 5)
    config.step.pool.batch_size = 6
    finalized = config.copy_tree().finalize()
    embedding = finalized.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert embedding.grid_shape == (5, 5)
    assert finalized.step.boundary is not None
    assert finalized.step.boundary.grid_shape == (5, 5)
    assert finalized.dataset.batch_size == 6
    prefix = finalized.step.model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    assert prefix.batch_size == 6


def test_the_smoke_run_needs_no_accelerator_and_writes_no_checkpoint() -> None:
    config = experiments.exp_smoke().copy_tree().finalize()
    assert config.step.model.compile_core is None
    assert config.dataset.device is None
    assert config.checkpointer is None
    assert config.max_steps == 4


def _under(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or one of its fields or items."""
    return path == root or path.startswith((f"{root}.", f"{root}["))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
