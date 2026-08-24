"""Tests for the ARC-AGI experiment ladder.

Each test asserts the DELTA a fork applies, which is what enforces one change
per experiment: a fork that quietly moved a second knob fails here rather than
producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training -- so the
ladder stays checkable on any machine.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from configgle.pprinting import pformat

import pytest

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.experiments import GRID_LEN, VOCAB_SIZE, ArcTrainLoop
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import SudokuNet
from priml.baselines.sudoku.prefix import (
    PrefixStack,
    RegisterTokens,
    SparsePuzzleEmbedding,
)
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.transformer import TransformerBlock


LADDER: list[tuple[str, Callable[[], ArcTrainLoop]]] = [
    ("exp000", experiments.exp000),
    ("exp001", experiments.exp001),
    ("exp002", experiments.exp002),
    ("exp003", experiments.exp003),
    ("exp_smoke", experiments.exp_smoke),
]


@pytest.mark.parametrize(("name", "factory"), LADDER, ids=[n for n, _ in LADDER])
def test_every_experiment_finalizes(
    name: str,
    factory: Callable[[], ArcTrainLoop],
) -> None:
    """A config must build without a dataset or a GPU."""
    config = factory().copy_tree().finalize()
    assert config.experiment_name == name
    assert config.study_name == "arcagi1"


def test_the_ladder_reuses_the_sudoku_solver() -> None:
    """ARC is the same classes with different values, not a second solver.

    If this ever fails, the two baselines have diverged into separate
    implementations -- the thing the shared step exists to prevent.
    """
    config = experiments.exp000()
    assert isinstance(config.step, SudokuTrainStep.Config)
    assert isinstance(config.step.model, SudokuNet.Config)


def test_the_grid_and_vocabulary_are_arcs() -> None:
    """The values that differ from sudoku, stated where a reader can see them."""
    config = experiments.exp000().copy_tree().finalize()
    embedding = config.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert embedding.grid_len == GRID_LEN
    assert config.step.model.vocab_size == VOCAB_SIZE


def test_the_prefix_carries_a_per_task_vector() -> None:
    """ARC's tasks recur under augmentation, so identity is worth learning.

    The task embedding must come FIRST: the halt head reads position 0.
    """
    config = experiments.exp000().copy_tree().finalize()
    prefix = config.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table, registers = prefix.parts
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert isinstance(registers, RegisterTokens.Config)
    # The sequence is grid plus every prefix token, counted automatically.
    expected = table.num_tokens + registers.num_tokens
    assert config.step.model.num_prefix_tokens == expected
    assert config.step.model.total_seq_len == GRID_LEN + expected


def test_exp001_changes_only_the_block() -> None:
    base, fork = experiments.exp000(), experiments.exp001()
    assert isinstance(base.step.model.block, TransformerBlock.Config)
    assert isinstance(fork.step.model.block, MLPMixerBlock.Config)
    assert fork.step.model.recurrence is base.step.model.recurrence is None
    assert fork.step.act is base.step.act is None
    assert fork.max_steps == base.max_steps


def test_exp002_adds_recurrence_and_its_feedback_channel() -> None:
    """Recurrence and prediction feedback move together, and say why.

    A recurrence that re-reads only the original task carries its belief solely
    in the latent; the feedback channel is what lets it refine its own answer.
    """
    base, fork = experiments.exp000(), experiments.exp002()
    assert base.step.model.recurrence is None
    assert fork.step.model.recurrence is not None
    assert fork.step.act is not None
    assert type(fork.step.model.block) is type(base.step.model.block)

    embedding = fork.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert [type(c) for c in embedding.channels] == [PredictionFeedback.Config]


def test_the_clue_range_matches_arcs_vocabulary() -> None:
    """Sudoku's digits stop at 10; ARC's colors run to the end of the vocab.

    A clue range copied from sudoku would let the feedback loop overwrite the
    last color, which no error would report.
    """
    config = experiments.exp002().copy_tree().finalize()
    assert config.step.act is not None
    assert config.step.act.given_high == VOCAB_SIZE - 1


def test_exp003_is_exp002_with_the_other_block() -> None:
    base, fork = experiments.exp002(), experiments.exp003()
    assert isinstance(fork.step.model.block, MLPMixerBlock.Config)
    assert fork.step.act is not None
    assert base.step.act is not None
    assert fork.step.act.max_steps == base.step.act.max_steps


def test_the_mixer_is_built_to_the_full_sequence() -> None:
    """A mixer mixes ACROSS positions, so the prefix counts too.

    Sized to the grid alone it would silently ignore the task embedding.
    """
    config = experiments.exp001().copy_tree().finalize()
    block = config.step.model.block
    assert isinstance(block, MLPMixerBlock.Config)
    assert block.seq_len == config.step.model.total_seq_len


def test_the_pool_is_built_to_the_models_shape() -> None:
    """A pool sized independently of the model would fail only at runtime."""
    config = experiments.exp002().copy_tree().finalize()
    act = config.step.act
    model = config.step.model
    assert act is not None
    assert act.grid_len == model.grid_len
    assert act.seq_len == model.total_seq_len
    assert act.channels_hidden == model.channels_in


def test_schedule_horizon_matches_the_step_budget() -> None:
    """A schedule annealing past the end of training wastes the last steps."""
    for name, factory in LADDER:
        config = factory()
        assert config.max_steps == config.step.total_train_steps, name


def test_smoke_is_small_on_every_costly_axis() -> None:
    """It answers "does this run", so anything not bearing on that is cut."""
    smoke, base = experiments.exp_smoke(), experiments.exp000()
    assert smoke.max_steps < base.max_steps
    assert smoke.step.model.channels_in < base.step.model.channels_in
    assert smoke.dataset.batch_size < base.dataset.batch_size
    assert smoke.dataset.num_tasks is not None
    # The per-task table dominates startup, so the buffer shrinks with it.
    prefix = smoke.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert table.batch_size == smoke.dataset.batch_size


def test_exp000_matches_its_golden_config(request: pytest.FixtureRequest) -> None:
    """Pin the WHOLE finalized ``exp000`` as readable text.

    ``exp000`` is the control every fork is measured against, so a change to
    it invalidates published numbers. A digest would say only that something
    moved; this golden says WHICH field, from what, to what.
    ``hide_default_values=False`` so a field that changes only because a
    library default changed still shows up here.

    Refresh with ``--golden-overwrite`` after reading the diff.
    """
    golden = Path(__file__).resolve().parent / "goldens" / "exp000_config.txt"
    rendered = pformat(
        experiments.exp000().copy_tree().finalize(), hide_default_values=False
    )
    if request.config.getoption("--golden-overwrite", default=False):
        golden.parent.mkdir(parents=True, exist_ok=True)
        _ = golden.write_text(rendered + "\n", encoding="utf-8")
    assert golden.read_text(encoding="utf-8") == rendered + "\n", (
        "exp000 changed; read the diff, then rerun with --golden-overwrite "
        "if the change is intended."
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
