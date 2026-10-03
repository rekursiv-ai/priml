"""Tests for the ARC-AGI experiment LADDER.

Each test asserts the DELTA a fork applies, which is what enforces one change
per experiment: a fork that quietly moved a second knob fails here rather than
producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training -- so the
LADDER stays checkable on any machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import inspect

from configgle import PartialConfig
from configgle.pprinting import pformat

import pytest
import torch

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.loss import MeanOverBatch
from priml.baselines.arcagi1.model import ConvSwiGLU, UrmRecurrence
from priml.baselines.sudoku.act import AtomicPool, FeedbackCarry, ZeroStart
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import SudokuNet, lattice_positions
from priml.baselines.sudoku.prefix import (
    PrefixStack,
    RegisterTokens,
    SparsePuzzleEmbedding,
)
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.lib.custom_json import FloatCodec
from priml.model.attention.rope import RoPE
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers import AdamATan2
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.arcagi1.experiments import ArcTrainLoop, TrmTrainLoop


LADDER: Final[list[tuple[str, Callable[[], ArcTrainLoop]]]] = [
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
    """Factories take no arguments; callers mutate configs before finalization."""
    assert not inspect.signature(factory).parameters
    config = factory()
    prefix = config.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    table.num_puzzles = 7
    config = config.copy_tree().finalize()
    assert config.experiment_name == name
    assert config.study_name == "arcagi1"
    prefix = config.step.model.prefix
    assert isinstance(prefix, PrefixStack.Config)
    table = prefix.parts[0]
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert table.num_puzzles == 7


def test_the_ladder_reuses_the_sudoku_solver() -> None:
    """ARC is the same classes with different values, not a second solver.

    If this ever fails, the two baselines have diverged into separate
    implementations -- the thing the shared step exists to prevent.
    """
    config = experiments.exp000()
    assert isinstance(config.step, SudokuTrainStep.Config)
    assert isinstance(config.step.model, SudokuNet.Config)


def test_unfilled_arc_model_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        experiments.exp000().step.model.make()


def test_the_grid_and_vocabulary_are_arcs() -> None:
    """The values that differ from sudoku, stated where a reader can see them."""
    config = experiments.exp000().copy_tree().finalize()
    embedding = config.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert embedding.grid_shape == (900,) == config.dataset.spec.grid_shape
    assert config.step.model.vocab_size == 12 == config.dataset.spec.vocab_size


def test_small_grid_follows_dataset_spec() -> None:
    config = experiments.exp000()
    config.dataset.spec.max_grid = 2
    finalized = config.finalize()
    embedding = finalized.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert embedding.grid_shape == (4,)
    assert finalized.step.model.vocab_size == config.dataset.spec.vocab_size


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
    assert (
        config.step.model.total_seq_len == config.dataset.spec.grid_shape[0] + expected
    )


def test_exp001_changes_only_the_block() -> None:
    base, fork = experiments.exp000(), experiments.exp001()
    assert isinstance(base.step.model.block, TransformerBlock.Config)
    assert isinstance(fork.step.model.block, MLPMixerBlock.Config)
    assert fork.step.model.recurrence is base.step.model.recurrence is None
    assert fork.step.pool is base.step.pool is None
    assert fork.max_steps == base.max_steps


def test_exp002_adds_recurrence_and_its_feedback_channel() -> None:
    """Recurrence and prediction feedback move together, and say why.

    A recurrence that re-reads only the original task carries its belief solely
    in the latent; the feedback channel is what lets it refine its own answer.
    """
    base, fork = experiments.exp000(), experiments.exp002()
    assert base.step.model.recurrence is None
    assert fork.step.model.recurrence is not None
    assert fork.step.pool is not None
    assert type(fork.step.model.block) is type(base.step.model.block)

    embedding = fork.step.model.embedding
    assert isinstance(embedding, GridEmbedding.Config)
    assert [type(c) for c in embedding.channels] == [PredictionFeedback.Config]


def test_the_clue_range_matches_arcs_vocabulary() -> None:
    """Sudoku's digits stop at 10; ARC's colors run to the end of the vocab.

    A clue range copied from sudoku would let the feedback loop overwrite the
    last color, which no error would report.
    """
    config = experiments.exp002()
    assert config.step.pool is not None
    config.step.pool.feedback = FeedbackCarry.Config(givens=(2, 10))
    finalized = config.copy_tree().finalize()
    assert finalized.step.pool is not None
    assert finalized.step.pool.feedback is not None
    vocab = finalized.dataset.spec.vocab_size
    assert finalized.step.pool.feedback.givens == (2, vocab - 1)


@pytest.mark.parametrize("factory", [experiments.exp002, experiments.exp003])
def test_the_recurrent_rungs_replay_as_trained(
    factory: Callable[[], ArcTrainLoop],
) -> None:
    """They trained with no fed-back grid and zero-seeded slots; keep it so."""
    pool = factory().step.pool
    assert pool is not None
    assert pool.feedback is None
    assert isinstance(pool.start, ZeroStart.Config)


def test_exp003_is_exp002_with_the_other_block() -> None:
    base, fork = experiments.exp002(), experiments.exp003()
    assert isinstance(fork.step.model.block, MLPMixerBlock.Config)
    assert fork.step.pool is not None
    assert base.step.pool is not None
    assert fork.step.pool.max_steps == base.step.pool.max_steps


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
    act = config.step.pool
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


REFERENCE: Final[list[tuple[str, Callable[[], TrmTrainLoop]]]] = [
    ("exp004", experiments.exp004),
    ("exp005", experiments.exp005),
    ("exp006", experiments.exp006),
    ("exp007", experiments.exp007),
    ("exp008", experiments.exp008),
]


@pytest.mark.parametrize(("name", "factory"), REFERENCE, ids=[n for n, _ in REFERENCE])
def test_reference_recipes_finalize(
    name: str,
    factory: Callable[[], TrmTrainLoop],
) -> None:
    """Each REFERENCE recipe builds a config with no data and no device."""
    config = factory().copy_tree().finalize()
    assert config.experiment_name == name
    assert config.max_steps == config.step.total_train_steps
    pool = config.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert pool.seq_len == config.step.model.total_seq_len
    table = config.step.model.prefix
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert table.batch_size == pool.batch_size == config.dataset.batch_size
    reduction = config.step.reduction
    assert isinstance(reduction, MeanOverBatch.Config)
    assert reduction.batch_size == pool.batch_size


def test_a_recipe_may_lay_its_rotary_grid_out_in_two_dimensions() -> None:
    def recipe() -> TrmTrainLoop:
        cfg = experiments.exp007()
        cfg.step.model.rope = RoPE.Config(channels_head=[32, 32])
        return cfg

    flat = recipe().copy_tree().finalize()
    assert flat.step.model.rope_grid_shape == (900,)
    grid = recipe()
    grid.step.model.rope_grid_shape = (30, 30)
    finalized = grid.copy_tree().finalize()
    assert finalized.step.model.rope_grid_shape == (30, 30)
    positions = {
        shape: lattice_positions(910, grid_shape=shape, device=torch.device("cpu"))
        for shape in ((900,), (30, 30))
    }
    assert positions[(30, 30)].shape == (910, 2)
    assert not torch.equal(positions[(30, 30)][:, -1], positions[(900,)])


def test_exp005_swaps_the_gate_norm_and_the_body_optimizer() -> None:
    base, fork = experiments.exp004(), experiments.exp005()
    assert isinstance(base.step.optimizer, AdamATan2.Config)
    assert isinstance(fork.step.optimizer, CompositeOptimizer.Config)
    for config, has_norm in ((base, False), (fork, True)):
        block = config.step.model.block
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        assert (block.ffn.norm is not None) is has_norm


def test_exp006_raises_only_the_muon_rate() -> None:
    base, fork = experiments.exp005(), experiments.exp006()
    rates: list[list[float]] = []
    for config in (base, fork):
        optimizer = config.step.optimizer
        assert isinstance(optimizer, CompositeOptimizer.Config)
        adamw, muon2, muon3 = optimizer.optimizers
        assert isinstance(adamw, PartialConfig)
        assert isinstance(muon2, Muon.Config)
        assert isinstance(muon3, Muon.Config)
        adamw_lr = FloatCodec.coerce(cast(object, adamw.lr), None)
        rates.append([adamw_lr, muon2.lr, muon3.lr])
    assert rates == [[1e-4, 5e-3, 5e-3], [1e-4, 0.01, 0.01]]


def test_exp007_moves_to_the_urm_recipe() -> None:
    fork = experiments.exp007()
    assert isinstance(fork.step.model.recurrence, UrmRecurrence.Config)
    block = fork.step.model.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, ConvSwiGLU.Config)
    assert fork.step.signals is not None
    assert set(fork.metrics_eval) == {"", "spatial_eq", "spatial_big"}


def test_exp008_adds_only_the_bundle() -> None:
    base, fork = experiments.exp007(), experiments.exp008()
    for config, bundled in ((base, False), (fork, True)):
        pool = config.step.pool
        assert isinstance(pool, AtomicPool.Config)
        assert (pool.feedback is not None) is bundled
        embedding = config.step.model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        assert bool(embedding.channels) is bundled


def test_exp000_matches_its_golden_config(request: pytest.FixtureRequest) -> None:
    """Pin the WHOLE finalized ``exp000`` as readable text.

    ``exp000`` is the control every fork is measured against, so a change to
    it invalidates published numbers. A digest would say only that something
    moved; this golden says WHICH field, from what, to what.
    ``hide_default_values=False`` so a field that changes only because a
    library default changed still shows up here.

    Refresh with ``--golden-overwrite`` after reading the diff.
    """
    assert_text_golden(
        request,
        test_file=__file__,
        name="exp000",
        rendered=pformat(
            experiments.exp000().copy_tree().finalize(),
            hide_default_values=False,
        ),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
