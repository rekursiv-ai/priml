"""Tests for the ARC-AGI experiment LADDER.

Each test asserts the DELTA a fork applies, which is what enforces one change
per experiment: a fork that quietly moved a second knob fails here rather than
producing a result nobody can attribute.

Every test builds configs only -- no data, no device, no training -- so the
LADDER stays checkable on any machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import inspect
import json

from configgle import PartialConfig

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1 import experiments
from priml.baselines.arcagi1.augmentation import (
    ArcSpec,
    ColorDihedral,
    SpatialAugmentation,
)
from priml.baselines.arcagi1.loss import MeanOverBatch, StablemaxTokens
from priml.baselines.arcagi1.metric import CanonicalPassK, SignalDumpTracker
from priml.baselines.arcagi1.model import (
    ConvSwiGLU,
    UrmRecurrence,
    depthwise_shift,
)
from priml.baselines.arcagi1.optimizer import with_ndim
from priml.baselines.arcagi1.scripts.build_dataset import (
    DEFAULT_SCALE_WEIGHTS,
    aug_policy_template,
)
from priml.baselines.arcagi1.scripts.build_spatial_eval import (
    spatial_eval_dataset_dir,
)
from priml.baselines.arcagi1.train_step import EvalSignals
from priml.baselines.arcagi2.model import PuzzleEmbedding, RotaryBlock
from priml.baselines.arcagi2.train_step import ArcDataParallel
from priml.baselines.sudoku.act import (
    AtomicPool,
    CellCorruption,
    FeedbackCarry,
    HaltTraining,
    SampledMinimum,
    ZeroStart,
)
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import (
    CoreCompile,
    DeepRecurrence,
    SudokuNet,
    lattice_positions,
)
from priml.baselines.sudoku.prefix import (
    PrefixStack,
    RegisterTokens,
    SparsePuzzleEmbedding,
)
from priml.baselines.sudoku.train_step import SudokuTrainStep
from priml.lib.codec import from_plain
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.init import corrected_fan_in_normal, kaiming_uniform
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers import AdamATan2
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import complement, excluding
from priml.runtime import MultiProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.tracker import TrackerList, WandbTracker


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
    assert table.num_tokens == 16
    assert registers.num_tokens == 1
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
    assert fork.step.model.block.seq_len == -1
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
    recurrence = fork.step.model.recurrence
    assert isinstance(recurrence, DeepRecurrence.Config)
    assert (recurrence.slow_cycles, recurrence.fast_cycles) == (3, 4)
    pool = fork.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert pool.batch_size == fork.dataset.batch_size
    assert isinstance(pool.halting, HaltTraining.Config)
    assert pool.halting.weight == 0.5
    assert isinstance(pool.halting.exploration, SampledMinimum.Config)
    assert isinstance(pool.start, ZeroStart.Config)
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
    assert isinstance(fork.step.model.block, MLPMixerBlock.Config)
    assert fork.step.model.block.seq_len == -1


def test_the_mixer_is_built_to_the_full_sequence() -> None:
    """A mixer mixes ACROSS positions, so the prefix counts too.

    Sized to the grid alone it would silently ignore the task embedding.
    """
    config = experiments.exp001().copy_tree().finalize()
    block = config.step.model.block
    assert isinstance(block, MLPMixerBlock.Config)
    assert block.seq_len == config.step.model.total_seq_len


def test_mixer_block_preserves_both_mixer_recipes() -> None:
    block = experiments._mixer_block(17)
    assert block.seq_len == 17
    assert block.prenorm is False
    for mixer in (block.token_mixer, block.channel_mixer):
        assert isinstance(mixer, SwiGLU.Config)
        assert isinstance(mixer.norm, RMSNorm.Config)
        assert mixer.init_weight is kaiming_uniform
        assert mixer.init_weight_out is kaiming_uniform


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
    assert smoke.max_steps == smoke.step.total_train_steps == 4
    assert smoke.num_steps_eval == 2
    assert smoke.step.model.channels_in == 32
    assert smoke.step.model.num_layers == 1
    assert smoke.dataset.batch_size == 8
    assert smoke.dataset.eval_batch_size == 8
    assert smoke.dataset.num_tasks == 4
    assert smoke.checkpointer is None
    assert smoke.max_steps < base.max_steps
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
    assert config.study_name == "arcagi1"
    assert config.seed == 0
    assert config.max_steps == config.step.total_train_steps == 388_670
    assert config.step.warmup_steps == 2_000
    assert config.step.lr_min_ratio == 1.0
    assert config.step.norm_log_interval == 100
    if name == "exp004":
        assert isinstance(config.step.optimizer, AdamATan2.Config)
        assert config.step.optimizer.lr == 1e-4
        assert config.step.optimizer.betas == (0.9, 0.95)
        assert config.step.optimizer.weight_decay == 0.1
    assert config.num_steps_eval == 10_000
    assert config.num_steps_log == 100
    assert config.early_train_log_steps == 100
    assert config.eval_warmup_batches == 1
    assert config.eval_every_epoch is False
    expected_batch_size = 96 if name in ("exp007", "exp008") else 256
    pool = config.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert pool.seq_len == config.step.model.total_seq_len
    assert pool.batch_size == expected_batch_size
    assert pool.max_steps == 16
    assert isinstance(pool.halting, HaltTraining.Config)
    assert pool.halting.weight == 0.5
    assert isinstance(pool.halting.exploration, SampledMinimum.Config)
    table = config.step.model.prefix
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert table.num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS
    assert table.batch_size == pool.batch_size == config.dataset.batch_size
    assert config.dataset.eval_batch_size == 256
    assert config.dataset.epochs_per_iter == 5
    assert config.dataset.num_puzzle_identifiers == (
        0 if name in ("exp007", "exp008") else experiments.NUM_PUZZLE_IDENTIFIERS
    )
    reduction = config.step.reduction
    assert isinstance(reduction, MeanOverBatch.Config)
    assert reduction.batch_size == pool.batch_size
    assert isinstance(config.step.token_loss, StablemaxTokens.Config)
    assert isinstance(config.step.ema, EMA.Config)
    assert config.step.ema.decay == 0.999
    assert config.step.ema.update_after_step == 2_000
    assert config.step.ema.warmup_seed is True
    assert config.step.ema.track_buffers is False
    assert config.step.ema.shadow_kind == "param_dict"
    expected_metrics = (
        {"", "spatial_eq", "spatial_big"} if name in ("exp007", "exp008") else {""}
    )
    assert set(config.metrics_eval) == expected_metrics
    assert isinstance(config.metrics_eval[""], CanonicalPassK.Config)
    assert isinstance(config.checkpointer, Checkpointer.Config)
    assert config.checkpointer.save_every == (
        5_000 if name in ("exp007", "exp008") else 4_000
    )
    assert config.checkpointer.keep_last_n == 8
    assert config.checkpointer.keep_every == 40_000
    assert isinstance(config.runtime, MultiProcess.Config)
    assert config.runtime.mesh_topology == {"dp": -1, "pp": 1, "tp": 1}
    if name in ("exp007", "exp008"):
        assert isinstance(config.tracker, TrackerList.Config)
        assert set(config.tracker.trackers) == {"wandb", "signals"}
    else:
        assert isinstance(config.tracker, WandbTracker.Config)
        assert config.tracker.project == "trm"


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
        adamw_lr = from_plain(cast(object, adamw.lr), float)
        rates.append([adamw_lr, muon2.lr, muon3.lr])
    assert rates == [[1e-4, 5e-3, 5e-3], [1e-4, 0.01, 0.01]]


def test_exp007_moves_to_the_urm_recipe() -> None:
    fork = experiments.exp007()
    assert isinstance(fork.step.model.recurrence, UrmRecurrence.Config)
    assert (
        fork.step.model.recurrence.slow_cycles,
        fork.step.model.recurrence.fast_cycles,
    ) == (2, 6)
    assert fork.step.model.num_layers == 4
    block = fork.step.model.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, ConvSwiGLU.Config)
    assert block.ffn.short_conv is depthwise_shift
    assert block.ffn.init_weight is corrected_fan_in_normal
    assert isinstance(block.ffn.norm, RMSNorm.Config)
    assert isinstance(fork.step.signals, EvalSignals.Config)
    assert fork.step.signals.per_step is True
    assert fork.step.emulate_precision_casts is True
    assert isinstance(fork.step.parallelism, ArcDataParallel.Config)
    assert fork.step.parallelism.gradient_as_bucket_view is True
    assert set(fork.metrics_eval) == {"", "spatial_eq", "spatial_big"}
    assert isinstance(fork.metrics_eval[""], CanonicalPassK.Config)
    assert isinstance(fork.metrics_eval["spatial_eq"], CanonicalPassK.Config)
    assert isinstance(fork.metrics_eval["spatial_big"], CanonicalPassK.Config)
    assert fork.metrics_eval[""].spatial_views == "non_spatial"
    assert fork.metrics_eval[""].max_views_per_input == 0
    assert fork.metrics_eval["spatial_eq"].spatial_views == "all"
    assert fork.metrics_eval["spatial_eq"].max_views_per_input == 1_001
    assert fork.metrics_eval["spatial_big"].spatial_views == "all"
    assert fork.metrics_eval["spatial_big"].max_views_per_input == 0
    for metric in fork.metrics_eval.values():
        assert isinstance(metric, CanonicalPassK.Config)
        assert metric.per_step_acts == 16
        assert metric.working_dir == fork.dataset.working_dir
    assert fork.dataset.working_dir == spatial_eval_dataset_dir(
        spatial_views=2,
        source_name=Path(
            aug_policy_template(translation_prob=0.2, scale_prob=0.2),
        ).name,
        working_dir="/datasets",
    )
    assert fork.dataset.augmentation.spatial.translation_prob == 0.2
    assert fork.dataset.augmentation.spatial.scale_prob == 0.2
    assert fork.dataset.augmentation.spatial.train_scale_weights == dict(
        DEFAULT_SCALE_WEIGHTS,
    )
    assert fork.dataset.augmentation.spatial_eval_views is True
    assert fork.dataset.num_puzzle_identifiers == 0
    assert isinstance(fork.tracker, TrackerList.Config)
    assert set(fork.tracker.trackers) == {"wandb", "signals"}
    assert isinstance(fork.tracker.trackers["wandb"], WandbTracker.Config)
    assert fork.tracker.trackers["wandb"].project == "trm"
    signals = fork.tracker.trackers["signals"]
    assert isinstance(signals, SignalDumpTracker.Config)
    assert isinstance(fork.checkpointer, Checkpointer.Config)
    assert signals.keep_last_n == fork.checkpointer.keep_last_n == 8
    assert signals.keep_every == fork.checkpointer.keep_every == 40_000

    optimizer = fork.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    adamw, muon2, muon3 = optimizer.optimizers
    assert isinstance(adamw, PartialConfig)
    assert from_plain(cast(object, adamw.lr), float) == 1e-4
    assert cast(tuple[float, float], adamw.betas) == (0.9, 0.95)
    assert from_plain(cast(object, adamw.weight_decay), float) == 1.0
    assert isinstance(muon2, Muon.Config)
    assert isinstance(muon3, Muon.Config)
    for muon in (muon2, muon3):
        assert muon.lr == 5e-3
        assert muon.momentum == 0.6
        assert muon.ns_steps == 3
        assert muon.weight_decay == 0.02
    assert muon2.ensemble_dims == 0
    assert muon3.ensemble_dims == 1
    on_muon = excluding(Muon.eligible_tensor, "embed", "head", "register_tokens")
    assert optimizer.select == [
        complement(on_muon),
        with_ndim(on_muon, 2),
        with_ndim(on_muon, 3),
    ]
    assert optimizer.drop_empty is True


def test_exp008_adds_only_the_bundle() -> None:
    base, fork = experiments.exp007(), experiments.exp008()
    for config, bundled in ((base, False), (fork, True)):
        pool = config.step.pool
        assert isinstance(pool, AtomicPool.Config)
        assert (pool.feedback is not None) is bundled
        embedding = config.step.model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        assert bool(embedding.channels) is bundled
    block = fork.step.model.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    assert isinstance(attn.norm_qk, RMSNorm.Config)
    assert attn.norm_qk.channels_in == attn.channels_head == 64
    pool = fork.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert isinstance(pool.feedback, FeedbackCarry.Config)
    corruption = pool.feedback.corruption
    assert isinstance(corruption, CellCorruption.Config)
    assert corruption.rate == 0.075


def test_sliced_metric_changes_only_its_requested_slice() -> None:
    base = CanonicalPassK.Config(max_views_per_input=7)
    metric = experiments._sliced(base, spatial_views="all", max_views=11)
    assert metric is not base
    assert metric.spatial_views == "all"
    assert metric.max_views_per_input == 11
    assert base.spatial_views == "all"
    assert base.max_views_per_input == 7


def test_reference_model_preserves_the_trm_architecture() -> None:
    model = experiments._reference_model(batch_size=3)
    assert isinstance(model, SudokuNet.Config)
    assert model.channels_in == 512
    assert model.num_layers == 2
    assert isinstance(model.embedding, GridEmbedding.Config)
    assert isinstance(model.block, RotaryBlock.Config)
    assert isinstance(model.block.attn, Attention.Config)
    assert model.block.attn.num_heads == 8
    assert model.block.attn.channels_head == 64
    assert model.block.attn.init_weight is corrected_fan_in_normal
    assert isinstance(model.block.rope, RoPE.Config)
    assert model.block.rope.channels_head == 64
    assert isinstance(model.recurrence, DeepRecurrence.Config)
    assert (model.recurrence.slow_cycles, model.recurrence.fast_cycles) == (3, 4)
    assert isinstance(model.prefix, PuzzleEmbedding.Config)
    assert model.prefix.num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS
    assert model.prefix.batch_size == 3
    assert isinstance(model.compile_core, CoreCompile.Config)


def test_exp000_votes_each_test_input_across_views(tmp_path: Path) -> None:
    """A task asking for two outputs is scored per output, pooled over its views.

    The build gives each VIEW one id shared by all of that view's test inputs,
    so a vote keyed by id would rank one input's answer against the other's and
    score the view, not the task.
    """
    pack = SpatialAugmentation.Config(spec=ArcSpec()).make()
    name, forward = (
        ColorDihedral.Config(separator="|||")
        .make()
        .sample("a", rng=np.random.default_rng(0))
    )
    pairs = [
        (np.array([[1]], dtype=np.uint8), np.array([[4]], dtype=np.uint8)),
        (np.array([[2]], dtype=np.uint8), np.array([[5]], dtype=np.uint8)),
    ]
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "a", name]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {
                "a": {
                    "test": [
                        {"input": i.tolist(), "output": o.tolist()} for i, o in pairs
                    ],
                },
            },
        ),
    )
    views = [*pairs, *((forward(i), forward(o)) for i, o in pairs)]
    rows = [
        pack.pack(i, o, training=False, rng=np.random.default_rng(0)) for i, o in views
    ]
    media = torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.int64)
    labels = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.int64)
    # Every view answers the second input and misses the first: an EOS in the
    # first cell crops to the empty grid whatever the view's color permutation.
    predictions = labels.clone()
    predictions[0::2, 0] = ArcSpec().vocab_eos
    config = experiments.exp000().copy_tree().finalize()
    (configured,) = config.metrics_eval.values()
    assert isinstance(configured, CanonicalPassK.Config)
    metric_config = configured.copy_tree()
    metric_config.working_dir = tmp_path
    metric = metric_config.make()
    # CanonicalPassK.update reads one q_halt header column before the grid.
    metric.update(
        torch.cat([torch.zeros(4, 1), predictions.float()], dim=-1),
        media=media,
        label=labels,
        puzzle_identifiers=torch.tensor([1, 1, 2, 2]),
        spatial_tags=torch.tensor([[1, 0, 0]] * 4),
    )
    scores = metric.compute()
    assert (scores["pass@1"], scores["pass@2"]) == (0.5, 0.5)


def test_exp000_matches_its_golden_config() -> None:
    """Pin the WHOLE finalized ``exp000``: the control every fork is measured against."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=experiments.exp000())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
