"""Tests for the ARC-AGI experiment ladder.

Each fork test asserts the DELTA a rung applies, which is what enforces one
change per experiment: a fork that quietly moved a second knob fails here
rather than producing a result nobody can attribute. The full config of every
rung is pinned by its ``testdata/<name>.txt`` golden.

Every test but the last builds configs only -- no data, no device, no
training -- so the ladder stays checkable on any machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import inspect
import json

from configgle import PartialConfig

import numpy as np
import pytest
import torch

from priml.baselines.arcagi import experiments
from priml.baselines.arcagi.augmentation import (
    ArcSpec,
    ColorDihedral,
    SpatialAugmentation,
)
from priml.baselines.arcagi.data import PuzzleData
from priml.baselines.arcagi.loss import MeanOverBatch, StablemaxTokens
from priml.baselines.arcagi.metric import (
    CanonicalPassK,
    PerOutputPass,
    StrictPass,
)
from priml.baselines.arcagi.model import (
    ConvSwiGLU,
    PuzzleEmbedding,
    RotaryBlock,
    SingleLatentRecurrence,
    depthwise_conv,
    depthwise_shift,
)
from priml.baselines.arcagi.optimizer import with_ndim
from priml.baselines.arcagi.signal_dump import (
    SignalDumpTracker,
)
from priml.baselines.arcagi.train_step import (
    ArcDataParallel,
    EvalSignals,
    PooledActTrainStep,
)
from priml.baselines.arcagi.train_step_test import port_config
from priml.baselines.sudoku.act import (
    AtomicPool,
    CellCorruption,
    FeedbackCarry,
    HaltTraining,
    SampledMinimum,
)
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import (
    CoreCompile,
    DeepRecurrence,
    SudokuNet,
    lattice_positions,
)
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.lib.codec import from_plain, loads
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.init import corrected_fan_in_normal
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.optimizers import AdamATan2
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import complement, excluding
from priml.runtime import MultiProcess, SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.parallelism import NoParallel
from priml.train.tracker import FileTracker, TrackerList, WandbTracker


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


LADDER: Final[list[tuple[str, Callable[[], experiments.Arc1TrainLoop]]]] = [
    ("exp000", experiments.exp000),
    ("exp001", experiments.exp001),
    ("exp002", experiments.exp002),
    ("exp003", experiments.exp003),
    ("exp004", experiments.exp004),
    ("exp005", experiments.exp005),
    ("exp006", experiments.exp006),
]
"""The ARC-AGI-1 rungs, each forking the one before it."""

ARC2: Final[list[tuple[str, Callable[[], experiments.Arc2TrainLoop]]]] = [
    ("exp007", experiments.exp007),
    ("exp008", experiments.exp008),
]
"""The ARC-AGI-2 rungs."""

SMOKE: Final[list[tuple[str, Callable[[], experiments.Arc1TrainLoop]]]] = [
    ("exp_smoke", experiments.exp_smoke),
]


def test_experiment_ladder_has_nine_rungs() -> None:
    """Expose exactly the nine numbered experiment factories."""
    expected = {f"exp{index:03d}" for index in range(9)}
    assert expected == {
        name
        for name in vars(experiments)
        if name.startswith("exp") and name[3:].isdigit()
    }


@pytest.mark.parametrize(
    ("name", "factory"),
    [*LADDER, *ARC2, *SMOKE],
    ids=[n for n, _ in [*LADDER, *ARC2, *SMOKE]],
)
def test_every_experiment_finalizes(
    name: str,
    factory: Callable[[], experiments.Arc1TrainLoop | experiments.Arc2TrainLoop],
) -> None:
    """Factories take no arguments and build with no data and no device."""
    assert not inspect.signature(factory).parameters
    config = factory().copy_tree().finalize()
    assert config.experiment_name == name
    assert config.study_name == "arcagi"
    assert config.seed == 0
    assert config.max_steps == config.step.total_train_steps


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("exp000", experiments.exp000),
        ("exp001", experiments.exp001),
        ("exp002", experiments.exp002),
        ("exp003", experiments.exp003),
        ("exp004", experiments.exp004),
        ("exp005", experiments.exp005),
        ("exp006", experiments.exp006),
        ("exp007", experiments.exp007),
        ("exp008", experiments.exp008),
        ("exp_smoke", experiments.exp_smoke),
    ],
)
def test_experiment_matches_its_full_config_golden(
    name: str,
    factory: Callable[[], experiments.Arc1TrainLoop | experiments.Arc2TrainLoop],
) -> None:
    assert_pprint_golden(test_file=__file__, name=name, config=factory())


@pytest.mark.parametrize("factory", [f for _, f in LADDER], ids=[n for n, _ in LADDER])
def test_arcagi1_rungs_share_the_reference_schedule(
    factory: Callable[[], experiments.Arc1TrainLoop],
) -> None:
    """Every ARC-AGI-1 rung trains the reference horizon at the reference batch."""
    config = factory().copy_tree().finalize()
    assert config.max_steps == config.step.total_train_steps == 388_670
    assert config.step.warmup_steps == 2_000
    assert config.step.lr_min_ratio == 1.0
    assert config.step.norm_log_interval == 100
    assert config.num_steps_eval == 10_000
    assert config.num_steps_log == 100
    assert config.early_train_log_steps == 100
    assert config.eval_warmup_batches == 1
    assert config.eval_every_epoch is False
    pool = config.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert pool.seq_len == config.step.model.total_seq_len
    assert pool.max_steps == 16
    assert isinstance(pool.halting, HaltTraining.Config)
    assert pool.halting.weight == 0.5
    assert isinstance(pool.halting.exploration, SampledMinimum.Config)
    table = config.step.model.prefix
    assert isinstance(table, PuzzleEmbedding.Config)
    assert table.num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS
    assert table.batch_size == pool.batch_size == config.dataset.batch_size
    reduction = config.step.reduction
    assert isinstance(reduction, MeanOverBatch.Config)
    assert reduction.batch_size == pool.batch_size
    assert isinstance(config.step.token_loss, StablemaxTokens.Config)
    assert isinstance(config.step.ema, EMA.Config)
    assert config.step.ema.decay == 0.999
    assert config.step.ema.update_after_step == 2_000
    assert config.dataset.epochs_per_iter == 5
    assert isinstance(config.runtime, MultiProcess.Config)
    assert config.runtime.mesh_topology == {"dp": -1, "pp": 1, "tp": 1}
    assert isinstance(config.step.parallelism, ArcDataParallel.Config)
    assert config.step.parallelism.gradient_as_bucket_view is True
    assert isinstance(config.tracker, TrackerList.Config)
    assert isinstance(config.tracker.trackers["metrics"], FileTracker.Config)
    wandb = config.tracker.trackers["wandb"]
    assert isinstance(wandb, WandbTracker.Config)
    assert wandb.project == "trm"


def test_exp000_states_the_reference_recipe() -> None:
    config = experiments.exp000().copy_tree().finalize()
    assert isinstance(config.step.optimizer, AdamATan2.Config)
    assert config.step.optimizer.lr == 1e-4
    assert config.step.optimizer.betas == (0.9, 0.95)
    assert config.step.optimizer.weight_decay == 0.1
    assert config.dataset.batch_size == config.dataset.eval_batch_size == 256
    assert config.dataset.num_puzzle_identifiers == experiments.NUM_PUZZLE_IDENTIFIERS
    assert set(config.metrics_eval) == {""}
    assert isinstance(config.checkpointer, Checkpointer.Config)
    assert config.checkpointer.save_every == 4_000
    assert config.checkpointer.keep_last_n == 8
    assert config.checkpointer.keep_every == 40_000


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


def test_exp001_normalizes_the_gate_and_trains_the_body_with_muon() -> None:
    base, fork = experiments.exp000(), experiments.exp001()
    assert isinstance(base.step.optimizer, AdamATan2.Config)
    assert isinstance(fork.step.optimizer, CompositeOptimizer.Config)
    for config, has_norm in ((base, False), (fork, True)):
        block = config.step.model.block
        assert isinstance(block, RotaryBlock.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        assert (block.ffn.norm is not None) is has_norm
    base.step.optimizer = fork.step.optimizer
    assert isinstance(base.step.model.block, RotaryBlock.Config)
    assert isinstance(fork.step.model.block, RotaryBlock.Config)
    base.step.model.block.ffn = fork.step.model.block.ffn
    base.experiment_name = fork.experiment_name
    assert base.pformat() == fork.pformat()


def test_exp002_raises_only_the_muon_rate() -> None:
    base, fork = experiments.exp001(), experiments.exp002()
    assert _rates(base) == [1e-4, 5e-3, 5e-3]
    assert _rates(fork) == [1e-4, 0.01, 0.01]
    base.step.optimizer = fork.step.optimizer
    base.experiment_name = fork.experiment_name
    assert base.pformat() == fork.pformat()


def test_exp003_moves_to_one_latent_and_a_conv_ffn() -> None:
    base, fork = experiments.exp002(), experiments.exp003()
    model = fork.step.model
    assert isinstance(model.recurrence, SingleLatentRecurrence.Config)
    assert (model.recurrence.slow_cycles, model.recurrence.fast_cycles) == (2, 6)
    assert model.recurrence.inner_grad_loops == 0
    assert model.num_layers == 4
    block = model.block
    assert isinstance(block, RotaryBlock.Config)
    assert isinstance(block.ffn, ConvSwiGLU.Config)
    assert block.ffn.short_conv is depthwise_conv
    assert block.ffn.init_weight is corrected_fan_in_normal
    assert isinstance(block.ffn.norm, RMSNorm.Config)
    assert _rates(fork) == [1e-4, 5e-3, 5e-3]
    pool = fork.step.pool
    assert isinstance(pool, AtomicPool.Config)
    table = model.prefix
    assert isinstance(table, PuzzleEmbedding.Config)
    assert pool.batch_size == table.batch_size == fork.dataset.batch_size == 96
    assert fork.dataset.eval_batch_size == 96
    # Everything else is exp002's.
    base.step.model = fork.step.model
    base.step.optimizer = fork.step.optimizer
    base.step.pool = fork.step.pool
    base.dataset.batch_size = base.dataset.eval_batch_size = 96
    base.experiment_name = fork.experiment_name
    assert base.pformat() == fork.pformat()


def test_exp004_speeds_up_the_same_model_and_dumps_its_views() -> None:
    base, fork = experiments.exp003(), experiments.exp004()
    block = fork.step.model.block
    assert isinstance(block, RotaryBlock.Config)
    assert isinstance(block.ffn, ConvSwiGLU.Config)
    assert block.ffn.short_conv is depthwise_shift
    assert fork.step.emulate_precision_casts is True
    assert isinstance(fork.step.signals, EvalSignals.Config)
    assert fork.step.signals.per_step is True
    assert fork.dataset.eval_batch_size == 256
    metric = fork.metrics_eval[""]
    assert isinstance(metric, CanonicalPassK.Config)
    assert metric.per_step_acts == 16
    assert isinstance(fork.checkpointer, Checkpointer.Config)
    assert fork.checkpointer.save_every == 5_000
    assert isinstance(fork.tracker, TrackerList.Config)
    assert set(fork.tracker.trackers) == {"wandb", "metrics", "signals"}
    signals = fork.tracker.trackers["signals"]
    assert isinstance(signals, SignalDumpTracker.Config)
    assert signals.keep_last_n == fork.checkpointer.keep_last_n == 8
    assert signals.keep_every == fork.checkpointer.keep_every == 40_000
    assert signals.keep_every % fork.num_steps_eval == 0
    assert base.step.model.pformat() != fork.step.model.pformat()


def test_exp005_changes_only_the_data_and_the_metric_slices() -> None:
    base, fork = experiments.exp004(), experiments.exp005()
    # The step is exp004's, which is why exp004 carries no golden of its own.
    assert base.step.pformat() == fork.step.pformat()
    assert fork.dataset.working_dir == (
        "/datasets/arc1concept-aug-1000-tr0p2-sc0p2-2w1p0-n1000-s42-spatialeval-v2"
    )
    spatial = fork.dataset.augmentation.spatial
    assert spatial.translation_prob == spatial.scale_prob == 0.2
    assert spatial.train_scale_weights == {2: 1.0}
    assert fork.dataset.augmentation.spatial_eval_views is True
    assert fork.dataset.num_puzzle_identifiers == 0
    slices = {
        name: (metric.spatial_views, metric.max_views_per_input)
        for name, metric in fork.metrics_eval.items()
        if isinstance(metric, CanonicalPassK.Config)
    }
    assert slices == {
        "": ("non_spatial", 0),
        "spatial_eq": ("all", 1_001),
        "spatial_big": ("all", 0),
    }
    for metric in fork.metrics_eval.values():
        assert isinstance(metric, CanonicalPassK.Config)
        assert metric.per_step_acts == 16
        assert metric.working_dir == fork.dataset.working_dir


def test_exp006_adds_only_the_bundle() -> None:
    base, fork = experiments.exp005(), experiments.exp006()
    for config, bundled in ((base, False), (fork, True)):
        pool = config.step.pool
        assert isinstance(pool, AtomicPool.Config)
        assert (pool.feedback is not None) is bundled
        embedding = config.step.model.embedding
        assert isinstance(embedding, GridEmbedding.Config)
        assert bool(embedding.channels) is bundled
    block = fork.step.model.block
    assert isinstance(block, RotaryBlock.Config)
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    assert isinstance(attn.norm_qk, RMSNorm.Config)
    assert attn.norm_qk.channels_in == attn.channels_head == 64
    assert isinstance(fork.step.model.embedding, GridEmbedding.Config)
    assert [type(c) for c in fork.step.model.embedding.channels] == [
        PredictionFeedback.Config,
    ]
    pool = fork.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert isinstance(pool.feedback, FeedbackCarry.Config)
    corruption = pool.feedback.corruption
    assert isinstance(corruption, CellCorruption.Config)
    assert corruption.rate == 0.075


@pytest.mark.parametrize("factory", [f for _, f in ARC2], ids=[n for n, _ in ARC2])
def test_every_arcagi2_rung_is_sized_and_scored_for_arcagi2(
    factory: Callable[[], experiments.Arc2TrainLoop],
) -> None:
    cfg = factory().copy_tree().finalize()
    prefix = cfg.step.model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    assert prefix.num_puzzles == experiments.ARC2_NUM_PUZZLE_IDENTIFIERS
    assert cfg.dataset.num_puzzle_identifiers == experiments.ARC2_NUM_PUZZLE_IDENTIFIERS
    pool = cfg.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert prefix.batch_size == pool.batch_size == cfg.dataset.batch_size
    assert cfg.max_steps == cfg.step.total_train_steps == 541_580
    assert cfg.dataset.epochs_per_iter == 4
    assert cfg.eval_extras_every_eval
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    assert cfg.checkpointer.save_every == 5_000
    for metric in cfg.metrics_eval.values():
        assert isinstance(metric, CanonicalPassK.Config)
        assert metric.working_dir == cfg.dataset.working_dir
        assert [type(rule) for rule in metric.rules] == [
            StrictPass.Config,
            PerOutputPass.Config,
        ]


@pytest.mark.parametrize(
    ("name", "source"),
    [("exp007", "exp000"), ("exp008", "exp005")],
)
def test_arcagi2_rungs_keep_the_arcagi1_step(name: str, source: str) -> None:
    """Only the task-table size and the horizon move off the ARC-AGI-1 step.

    So the ARC-AGI-2 rungs need no tensor golden: their numerics are the
    source rung's, which ``train_step_test`` pins.
    """
    ours = cast(
        "Callable[[], experiments.Arc2TrainLoop]",
        getattr(experiments, name),
    )()
    theirs = cast(
        "Callable[[], experiments.Arc1TrainLoop]",
        getattr(experiments, source),
    )()
    prefix = theirs.step.model.prefix
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    prefix.num_puzzles = experiments.ARC2_NUM_PUZZLE_IDENTIFIERS
    theirs.step.total_train_steps = experiments.ARC2_TOTAL_TRAIN_STEPS
    assert ours.step.pformat() == theirs.step.pformat()


def test_exp008_stages_the_arcagi2_spatial_trees() -> None:
    cfg = experiments.exp008()
    assert cfg.dataset.spatial_eval_views == 2
    assert str(cfg.dataset.working_dir).endswith("-spatialeval-v2")
    assert str(cfg.dataset.working_dir).startswith(str(cfg.dataset.source_dataset_dir))
    assert cfg.dataset.eval_batch_size == 256
    assert set(cfg.metrics_eval) == {"", "spatial_eq", "spatial_big"}
    assert isinstance(cfg.tracker, TrackerList.Config)
    assert set(cfg.tracker.trackers) == {"wandb", "metrics", "signals"}


def test_exp007_reads_the_plain_arcagi2_tree() -> None:
    cfg = experiments.exp007()
    assert cfg.dataset.spatial_eval_views == 0
    assert set(cfg.metrics_eval) == {""}
    assert cfg.step.signals is None


def test_a_recipe_may_lay_its_rotary_grid_out_in_two_dimensions() -> None:
    def recipe() -> experiments.Arc1TrainLoop:
        cfg = experiments.exp005()
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


def test_an_arcagi2_rung_finalizes_without_rope() -> None:
    config = experiments.exp007()
    config.step.model.rope = None
    assert config.finalize().step.model.rope is None


def test_the_muon_recipe_filters_and_rates() -> None:
    optimizer = experiments.exp003().step.optimizer
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


def test_exp000_states_the_reference_architecture() -> None:
    model = experiments.exp000().step.model
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
    assert model.prefix.batch_size == 256
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


def test_smoke_is_small_on_every_costly_axis() -> None:
    """It answers "does this run", so anything not bearing on that is cut."""
    smoke = experiments.exp_smoke().copy_tree().finalize()
    assert isinstance(smoke.runtime, SingleProcess.Config)
    assert isinstance(smoke.step.parallelism, NoParallel.Config)
    assert smoke.max_steps == smoke.step.total_train_steps == 4
    assert smoke.step.warmup_steps == 0
    assert smoke.num_steps_eval == 2
    model = smoke.step.model
    assert model.compile_core is None
    assert model.channels_in == 32
    assert isinstance(model.block, RotaryBlock.Config)
    assert isinstance(model.block.attn, Attention.Config)
    assert model.block.attn.num_heads == 2
    assert isinstance(model.recurrence, DeepRecurrence.Config)
    assert model.recurrence.slow_cycles == model.recurrence.fast_cycles == 1
    pool = smoke.step.pool
    assert isinstance(pool, AtomicPool.Config)
    assert pool.batch_size == 2
    assert pool.max_steps == 4
    assert smoke.dataset.batch_size == smoke.dataset.eval_batch_size == 2
    assert smoke.dataset.max_samples == 64
    assert smoke.dataset.num_puzzle_identifiers == 0
    assert smoke.checkpointer is None
    assert smoke.tracker is None
    table = model.prefix
    assert isinstance(table, PuzzleEmbedding.Config)
    assert table.batch_size == 2
    assert table.num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS


def test_the_loop_trains_evaluates_checkpoints_and_resumes(tmp_path: Path) -> None:
    """Run the smoke loop through a checkpoint and its next update on CPU."""
    root = tmp_path / "datasets/arc1concept-aug-1000"
    root.mkdir(parents=True)
    (root / "identifiers.json").write_text('["<blank>", "puzzle"]')
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    (root / "test_puzzles.json").write_text(
        json.dumps({"puzzle": {"test": [{"input": grid, "output": grid}]}}),
    )
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir()
        (directory / "dataset.json").write_text(
            '{"ignore_label_id": 0, "blank_identifier_id": 0}',
        )
        for name, array in {
            "inputs": np.full((3, 9), 2, dtype=np.int32),
            "labels": np.full((3, 9), 2, dtype=np.int32),
            "puzzle_identifiers": np.array([1], dtype=np.int32),
            "puzzle_indices": np.array([0, 3], dtype=np.int64),
            "group_indices": np.array([0, 1], dtype=np.int64),
        }.items():
            np.save(directory / f"all__{name}.npy", array)
    config = experiments.exp_smoke()
    config.base_dir = tmp_path
    config.step = port_config("exp000")
    config.dataset.spec.max_grid = 3
    config.runtime = SingleProcess.Config(device="cpu")
    config.dataset.device = "cpu"
    config.max_steps = 1
    config.num_steps_eval = 1
    # The warm evaluation only populates compile caches, and this loop compiles
    # nothing; the two-rank test covers a warm evaluation ahead of training.
    config.eval_warmup_batches = 0
    config.checkpointer = Checkpointer.Config(save_every=1)
    config.tracker = FileTracker.Config()
    loop = config.make()
    loop.run()
    assert isinstance(loop.step, PooledActTrainStep)
    assert loop.step.global_step == 1
    metrics = from_plain(
        loads((tmp_path / "runs/arcagi/exp_smoke/metrics.json").read_text()),
        dict[str, object],
    )
    assert any(key.endswith("pass@2") for key in metrics)
    expected = {
        name: value.clone() for name, value in loop.step.model.state_dict().items()
    }
    checkpoint = tmp_path / "runs/arcagi/exp_smoke/checkpoints/step_00000001.pt"
    assert checkpoint.is_file()
    config.max_steps = 2
    resumed = config.make()
    assert isinstance(resumed.step, PooledActTrainStep)
    assert resumed.step.global_step == 1
    for name, value in resumed.step.model.state_dict().items():
        assert torch.equal(value, expected[name])
    resumed.run()
    assert resumed.step.global_step == 2
    assert (checkpoint.parent / "step_00000002.pt").is_file()
    assert any(
        not torch.equal(value, expected[name])
        for name, value in resumed.step.model.state_dict().items()
    )


SIZING_TASKS: Final = 3
SIZING_PUZZLES_PER_TASK: Final = 2
SIZING_GRID: Final = 900


@pytest.fixture
def sizing_tree(tmp_path: Path) -> Path:
    """Write a tiny ARC tree whose identifiers start at 1, as the builder's do.

    0 is the reserved blank, so the largest id equals the puzzle count -- which
    is what makes a one-row table index out of bounds rather than merely wrong.
    """
    puzzles = SIZING_TASKS * SIZING_PUZZLES_PER_TASK
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        grids = np.full((puzzles * 2, SIZING_GRID), 2, dtype=np.int32)
        np.save(directory / "all__inputs.npy", grids)
        np.save(directory / "all__labels.npy", grids)
        np.save(
            directory / "all__puzzle_indices.npy",
            np.arange(puzzles + 1, dtype=np.int32) * 2,
        )
        np.save(
            directory / "all__group_indices.npy",
            np.arange(SIZING_TASKS + 1, dtype=np.int32) * SIZING_PUZZLES_PER_TASK,
        )
        np.save(
            directory / "all__puzzle_identifiers.npy",
            np.arange(1, puzzles + 1, dtype=np.int32),
        )
        (directory / "dataset.json").write_text(
            json.dumps(
                {"vocab_size": 12, "seq_len": SIZING_GRID, "ignore_label_id": 0},
            ),
        )
    return tmp_path


def test_exp000_table_covers_the_datasets_identifiers() -> None:
    """One row per identifier the dataset can emit, or a lookup runs off the end.

    ``exp000`` names ``arc1concept-aug-1000``, whose build assigns 876,403 ids.
    A table shorter than that is not a smaller model -- it is a crash on the
    first batch, which is why this is pinned rather than left to the default.
    """
    table = experiments.exp000().copy_tree().finalize().step.model.prefix
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert table.num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS


def test_a_real_batch_indexes_the_table_in_bounds(sizing_tree: Path) -> None:
    """The dataset's ids must index the table the experiment builds.

    Config equality alone would pass on two numbers that are wrong together, so
    this drives the real loader into the real module: the lookup must succeed.
    """
    dataset = PuzzleData.Config()
    dataset.base_dir = "/"
    dataset.working_dir = str(sizing_tree)
    dataset.device = "cpu"
    dataset.batch_size = 4
    batch = next(iter(dataset.make().train_dataloader()))
    identifiers = batch["puzzle_identifiers"]
    assert int(identifiers.max()) > 0  # The tree really does exercise the table.
    # Only the SIZE axes shrink; ``num_puzzles`` is the field under test and
    # must arrive from the experiment exactly as a real run would build it.
    table = experiments.exp000().copy_tree().finalize().step.model.prefix
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    table.batch_size = 4
    table.channels_out = 8
    table.channels_in = 8
    module = table.make()
    module.eval()
    prefix = module(4, puzzle_identifiers=identifiers)
    assert prefix.shape == (4, table.num_tokens, 8)


def test_smoke_keeps_the_full_table_despite_its_row_cap() -> None:
    """Capping the rows read does not cap the ids those rows carry.

    Ids are assigned once across the whole build, so a loaded task carries the
    id it was given there, not its position in a truncated prefix; a table
    sized to ``max_samples`` would index off the end.
    """
    smoke = experiments.exp_smoke().copy_tree().finalize()
    assert smoke.dataset.max_samples is not None
    table = smoke.step.model.prefix
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    assert table.num_puzzles == experiments.NUM_PUZZLE_IDENTIFIERS


def _rates(config: experiments.Arc1TrainLoop) -> list[float]:
    """Return the AdamW rate and both Muon rates of a Muon-body recipe."""
    optimizer = config.step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    adamw, muon2, muon3 = optimizer.optimizers
    assert isinstance(adamw, PartialConfig)
    assert isinstance(muon2, Muon.Config)
    assert isinstance(muon3, Muon.Config)
    return [from_plain(cast(object, adamw.lr), float), muon2.lr, muon3.lr]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
