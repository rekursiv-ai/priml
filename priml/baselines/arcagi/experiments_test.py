"""Tests for the ARC-AGI experiment ladder.

Each fork test asserts the DELTA a rung applies, which is what enforces one
change per experiment: a fork that quietly moved a second knob fails here
rather than producing a result nobody can attribute. The full config of every
rung is pinned by its ``testdata/<name>.txt`` golden.

Every test but the last builds configs only -- no data, no device, no
training -- so the ladder stays checkable on any machine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, cast

import inspect
import json
import math

from configgle import InlineConfig, PartialConfig
from pyarrow import parquet
from torch import Tensor
from treekle import from_plain, json_loads

import numpy as np
import pyarrow as pa
import pytest
import torch

from priml.baselines.arcagi import experiments
from priml.baselines.arcagi.augmentation import (
    ArcSpec,
    SpatialAugmentation,
)
from priml.baselines.arcagi.checkpoint import CheckpointActTrainStep
from priml.baselines.arcagi.data import ArcParquetData, PairSource, PuzzleData
from priml.baselines.arcagi.metric import (
    CanonicalPassK,
    ParquetPassK,
    PerOutputPass,
    StrictPass,
)
from priml.baselines.arcagi.model import (
    ConvSwiGLU,
    PuzzleEmbedding,
    RotaryBlock,
    depthwise_conv,
    depthwise_shift,
)
from priml.baselines.arcagi.signal_dump import (
    SignalDumpTracker,
)
from priml.baselines.arcagi.train_step import (
    ArcDataParallel,
    EvalSignals,
    PooledActTrainStep,
    apply_direct_schedule_to_body_and_sparse_optimizer_lrs,
)
from priml.baselines.arcagi.train_step_test import port_config
from priml.baselines.sudoku.act import (
    AtomicPool,
    CellCorruption,
    FeedbackCarry,
    GlobalSampledMinimum,
    HaltTraining,
    SampledMinimum,
)
from priml.data.color_dihedral import ColorDihedral
from priml.loss.act import ACTLoss
from priml.loss.token import MeanOverBatch, StablemaxTokens
from priml.math.loss import log_stablemax_via_probabilities
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.conditioning import (
    CategorySlotFiLM,
    SigmoidResidualGate,
)
from priml.model.conv import SequenceConv1d
from priml.model.embedding import Embedding
from priml.model.grid_embedding import GridEmbedding, PredictionFeedback
from priml.model.grid_solver import (
    CoreCompile,
    SingleLatentGridSolver,
    lattice_positions,
)
from priml.model.init import corrected_fan_in_normal
from priml.model.norm import RMSNorm
from priml.model.prefix import ComposedSparsePrefix, SparsePuzzleEmbedding
from priml.model.recurrence import DeepRecurrence, SingleLatentRecurrence
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.optimizers import AdamATan2, AuxAdam, SignSGD
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import complement, excluding, with_ndim
from priml.runtime import MultiProcess, SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.parallelism import NoParallel
from priml.train.tracker import FileTracker, TrackerList, WandbTracker


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

CHARM: Final[list[tuple[str, Callable[[], experiments.CharmTrainLoop.Config]]]] = [
    ("exp009", experiments.exp009),
    ("exp010", experiments.exp010),
]
"""Published CHARM source-recipe replications."""

SMOKE: Final[list[tuple[str, Callable[[], experiments.Arc1TrainLoop]]]] = [
    ("exp_smoke", experiments.exp_smoke),
]


def test_experiment_ladder_has_eleven_rungs() -> None:
    """Expose exactly the eleven numbered experiment factories."""
    expected = {f"exp{index:03d}" for index in range(11)}
    assert expected == {
        name
        for name in vars(experiments)
        if name.startswith("exp") and name[3:].isdigit()
    }


@pytest.mark.parametrize(
    ("name", "factory"),
    [*LADDER, *ARC2, *CHARM, *SMOKE],
    ids=[n for n, _ in [*LADDER, *ARC2, *CHARM, *SMOKE]],
)
def test_every_experiment_finalizes(
    name: str,
    factory: Callable[
        [],
        experiments.Arc1TrainLoop
        | experiments.Arc2TrainLoop
        | experiments.CharmTrainLoop.Config,
    ],
) -> None:
    """Factories take no arguments and build with no data and no device."""
    assert not inspect.signature(factory).parameters
    config = factory().copy_tree().finalize()
    assert config.experiment_name == name
    assert config.study_name == "arcagi"
    assert config.seed == 0
    if name in {"exp009", "exp010"}:
        assert config.max_steps == config.step.total_train_steps
    else:
        assert isinstance(config, experiments.Arc1TrainLoop | experiments.Arc2TrainLoop)
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
        ("exp009", experiments.exp009),
        ("exp010", experiments.exp010),
        ("exp_smoke", experiments.exp_smoke),
    ],
)
def test_experiment_matches_its_full_config_golden(
    name: str,
    factory: Callable[
        [],
        experiments.Arc1TrainLoop
        | experiments.Arc2TrainLoop
        | experiments.CharmTrainLoop.Config,
    ],
) -> None:
    assert_pprint_golden(test_file=__file__, name=name, config=factory())


@pytest.mark.parametrize(("name", "factory"), CHARM, ids=[name for name, _ in CHARM])
def test_published_charm_recipe_is_explicit_and_io_free(
    name: str,
    factory: Callable[[], experiments.CharmTrainLoop.Config],
) -> None:
    config = factory().copy_tree().finalize()

    assert config.study_name == "arcagi"
    assert config.experiment_name == name
    assert config.seed == config.dataset.seed == 0
    assert config.mesh_dim_data_seed is None
    assert config.model_torch_seed is torch.manual_seed
    assert config.max_steps == config.step.total_train_steps
    assert config.global_batch_size == 768
    assert config.dataset.batch_size == config.step.pool.batch_size == 96
    assert config.dataset.eval_batch_size == 96
    assert config.dataset.canvas_size == 30
    assert config.dataset.num_workers == 4
    assert config.dataset.p_translate == 0.8
    assert config.step.warmup_steps == 2_000
    assert config.step.lr_min_ratio == 1.0
    assert (
        config.step.optimizer_lr_policy
        is apply_direct_schedule_to_body_and_sparse_optimizer_lrs
    )
    assert config.step.gradient_clip_norm == float("inf")
    assert config.step.dtype_autocast is None
    assert config.step.compile is None
    assert config.step.total_train_steps == (
        1_600_000 if name == "exp009" else 2_380_000
    )
    assert config.num_steps_eval == 10_000
    assert config.eval_every_epoch is False
    assert isinstance(config.runtime, MultiProcess.Config)
    assert config.runtime.mesh_topology == {"dp": 8, "pp": 1, "tp": 1}
    assert isinstance(config.checkpointer, Checkpointer.Config)
    assert config.checkpointer.save_every == 10_000
    assert set(config.metrics_eval) == {""}
    assert isinstance(config.metrics_eval[""], ParquetPassK.Config)
    assert config.metrics_eval[""].pass_ks == (1, 2, 5, 10, 100, 1000)
    assert config.metrics_eval[""].recent_window_size is None
    assert config.metrics_eval[""].aggregated_voting is True
    assert (
        Path(config.metrics_eval[""].eval_source).name
        == (_charm_source_names(name)[-1])
    )
    step = config.step
    assert isinstance(config.dataset, ArcParquetData.Config)
    assert all(
        isinstance(source, PairSource.Config) for source in config.dataset.train_sources
    )
    assert isinstance(step.model, SingleLatentGridSolver.Config)
    assert step.model.channels_in == 512
    assert step.model.num_layers == 4
    assert step.model.forward_dtype is torch.bfloat16
    assert step.model.halt_outputs == 1
    assert step.model.halt_init_bias == -5.0
    assert step.model.vocab_size == 12
    assert isinstance(step.model.embedding, GridEmbedding.Config)
    assert step.model.embedding.grid_shape == (30, 30)
    assert isinstance(step.model.compile_core, CoreCompile.Config)
    assert step.model.compile_core.unit == "core"
    assert step.model.compile_core.mode == "default"
    assert step.model.compile_core.fullgraph is False
    assert isinstance(step.model.block, TransformerBlock.Config)
    assert isinstance(step.model.block.attn, Attention.Config)
    assert step.model.block.attn.num_heads == 8
    assert step.model.block.attn.channels_head == 64
    assert step.model.block.attn.num_heads_kv == 8
    assert isinstance(step.model.block.ffn, SwiGLU.Config)
    assert isinstance(step.model.embedding.token_table, Embedding.Config)
    assert step.model.embedding.embed_scale == 1.0
    assert step.model.embedding.token_table.lookup_dtype is torch.bfloat16
    assert isinstance(step.model.block.ffn.middle, SequenceConv1d.Config)
    assert step.model.block.ffn.middle.conv.kernel_size == 2
    assert step.model.block.ffn.middle.conv.groups == 1_536
    assert step.model.block.ffn.expansion == 4.0
    assert step.model.block.ffn.channels_hidden == 1_536
    recurrence = step.model.recurrence
    assert isinstance(recurrence, SingleLatentRecurrence.Config)
    assert (
        recurrence.slow_cycles,
        recurrence.fast_cycles,
        recurrence.inner_grad_loops,
    ) == (1, 12, 6)
    assert isinstance(step.pool, AtomicPool.Config)
    assert step.pool.max_steps == 16
    assert step.pool.steps_dtype is torch.int32
    assert isinstance(step.pool.halting, HaltTraining.Config)
    assert step.pool.halting.weight == 0.5
    assert isinstance(step.pool.halting.exploration, GlobalSampledMinimum.Config)
    assert isinstance(step.objective, ACTLoss.Config)
    assert step.objective.halt_weight == 0.5
    assert isinstance(step.objective.token_loss, StablemaxTokens.Config)
    assert step.objective.token_loss.log_normalizer is log_stablemax_via_probabilities
    assert isinstance(step.ema, EMA.Config)
    assert step.ema.decay == 0.999
    assert isinstance(step.sparse_optimizer, SignSGD.Config)
    assert step.sparse_optimizer.lr == 1e-2
    assert step.sparse_optimizer.weight_decay == 0.1
    assert isinstance(step.optimizer, CompositeOptimizer.Config)
    dense_optimizer, aux_adam_config = step.optimizer.optimizers
    assert isinstance(dense_optimizer, Muon.Config)
    assert isinstance(aux_adam_config, AuxAdam.Config)
    assert dense_optimizer.lr == 1e-4
    assert dense_optimizer.weight_decay == 0.1
    assert dense_optimizer.momentum == 0.95
    assert dense_optimizer.nesterov is True
    assert dense_optimizer.ns_steps == 5
    assert dense_optimizer.eps == 1e-7
    assert aux_adam_config.lr == 1e-4
    assert aux_adam_config.weight_decay == 0.1
    assert aux_adam_config.betas == (0.9, 0.95)
    assert aux_adam_config.eps == 1e-8

    prefix = step.model.prefix
    assert isinstance(prefix, ComposedSparsePrefix.Config)
    assert prefix.channels_out == 512
    assert prefix.num_tokens == 16
    assert isinstance(prefix.composition, CategorySlotFiLM.Config)
    assert isinstance(prefix.residual_gate, SigmoidResidualGate.Config) is (
        name == "exp009"
    )
    assert prefix.composition.num_categories == _charm_tasks(name)
    assert prefix.composition.num_augmentations == 8
    assert prefix.composition.num_slots == 9
    assert prefix.residual.num_puzzles == _charm_instance_rows(name)
    assert prefix.residual.channels_out == 32
    assert prefix.residual.num_tokens == 1
    assert prefix.residual_projection.channels_in == 32
    assert prefix.residual_projection.channels_out == 512
    assert prefix.num_tokens == 16

    assert config.dataset.num_tasks == _charm_tasks(name)
    assert config.dataset.num_instances == _charm_instance_rows(name) - 1
    sources = config.dataset.train_sources
    assert [source.pair_selection for source in sources] == [
        "both",
        "both",
        "both",
        "train",
    ]
    assert [source.multiplicity for source in sources] == [1, 1, 2, 2]
    assert [Path(source.path).name for source in sources] == _charm_source_names(name)
    assert Path(config.dataset.eval_source).name == _charm_source_names(name)[-1]
    assert isinstance(config.tracker, TrackerList.Config)
    wandb = config.tracker.trackers["wandb"]
    assert isinstance(wandb, WandbTracker.Config)
    assert wandb.project == "charm"
    assert wandb.name == (
        "arc-agi-1-charm-rearc100" if name == "exp009" else "arc-agi-2-charm-rearc2-100"
    )


@pytest.mark.parametrize(
    ("name", "factory", "budget"),
    [
        ("exp009", experiments.exp009, 1_600_000),
        ("exp010", experiments.exp010, 2_380_000),
    ],
    ids=["arc1", "arc2"],
)
def test_charm_finalize_preserves_explicit_step_caps(
    name: str,
    factory: Callable[[], experiments.CharmTrainLoop.Config],
    budget: int,
) -> None:
    """Keep an independent optimizer-step limit through recipe finalization."""
    config = factory()
    config.max_steps = 1
    finalized = config.copy_tree().finalize()

    assert finalized.experiment_name == name
    assert finalized.max_steps == 1
    assert finalized.step.total_train_steps == budget


def test_published_charm_metric_retains_votes_across_eleven_events(
    tmp_path: Path,
) -> None:
    """The source default keeps an early task vote beyond the tenth reset."""
    source = tmp_path / "eval.parquet"
    _write_two_task_eval_source(source)
    config = ParquetPassK.Config(
        eval_source=source,
        canvas_size=6,
        pass_ks=(1,),
    )
    assert config.recent_window_size is None
    full_history = config.make()
    bounded_history = config.copy_tree()
    bounded_history.recent_window_size = 10
    bounded = bounded_history.make()

    for metric in (full_history, bounded):
        for event in range(11):
            metric.reset()
            if event == 0:
                name, input_grid, output_grid = "task-a", [[1]], [[2]]
            else:
                name, input_grid, output_grid = "task-b", [[3]], [[4]]
            prediction = _packed_window_candidate(output_grid)
            canonical_input = np.asarray(input_grid, dtype=np.uint8)
            metric.update(
                torch.stack((prediction, prediction)),
                canonical_puzzle_names=[name, name],
                canonical_inputs=[canonical_input, canonical_input],
                spatial_tags=torch.tensor([[1, 1, 1], [1, 1, 1]]),
                dih_idxs=torch.tensor([0, 0]),
                colorperm_slots=torch.arange(9).repeat(2, 1),
                valid_count=2,
            )

    assert full_history.compute() == {"pass@1": 1.0}
    assert bounded.compute() == {"pass@1": 0.5}
    assert list(full_history.state_dict()["events"]) == list(range(11))
    assert list(bounded.state_dict()["events"]) == list(range(1, 11))


def _charm_tasks(name: str) -> int:
    return 960 if name == "exp009" else 1_280


def _charm_instance_rows(name: str) -> int:
    return 876_705 if name == "exp009" else 1_289_151


def _charm_source_names(name: str) -> list[str]:
    prefix = "arc1" if name == "exp009" else "arc2"
    rearc = "rearc" if prefix == "arc1" else "rearc2"
    return [
        f"{prefix}_training_aug1000.parquet",
        f"{rearc}_aug100.parquet",
        "concept_aug1000.parquet",
        f"{prefix}_eval_aug1000.parquet",
    ]


@pytest.mark.parametrize(
    ("dp_size", "local_batch"),
    [(8, 96), (2, 384), (6, 128)],
)
def test_charm_global_batch_tracks_explicit_data_parallel_size(
    dp_size: int,
    local_batch: int,
) -> None:
    config = experiments.exp009()
    assert isinstance(config.runtime, MultiProcess.Config)
    config.runtime.mesh_topology["dp"] = dp_size

    finalized = config.copy_tree().finalize()

    assert finalized.global_batch_size == 768
    assert finalized.dataset.batch_size == local_batch
    assert finalized.dataset.eval_batch_size == local_batch
    assert finalized.step.pool.batch_size == local_batch


def test_charm_global_batch_requires_integral_per_rank_batches() -> None:
    config = experiments.exp009()
    config.global_batch_size = 770

    finalized = config.finalize()
    assert finalized.global_batch_size == 770
    with pytest.raises(ValueError, match="divisible"):
        finalized.make()


def test_charm_rejects_a_nonpositive_global_batch() -> None:
    config = experiments.exp009()
    config.global_batch_size = 0

    with pytest.raises(
        ValueError,
        match=r"^CHARM global_batch_size must be positive\.$",
    ):
        config.copy_tree().finalize().make()


def test_charm_rejects_a_mesh_without_a_data_parallel_size() -> None:
    config = experiments.exp009()
    assert isinstance(config.runtime, MultiProcess.Config)
    config.runtime.mesh_topology = {"pp": 1, "tp": 1}

    with pytest.raises(
        ValueError,
        match=r"^CHARM requires an explicit positive data-parallel size\.$",
    ):
        config.copy_tree().finalize().make()


def test_charm_rejects_an_unrecognized_runtime_without_sizing_batches() -> None:
    config = experiments.exp009()
    config.runtime = InlineConfig(SingleProcess, SingleProcess.Config(device="cpu"))

    finalized = config.copy_tree().finalize()

    assert finalized.dataset.batch_size == ArcParquetData.Config().batch_size
    with pytest.raises(
        TypeError,
        match=r"^CHARM requires MultiProcess or SingleProcess runtime\.$",
    ):
        finalized.make()


def test_charm_single_process_reaches_the_train_loop_checks() -> None:
    """A valid single-process CHARM config hands off to TrainLoop's validation."""
    config = experiments.exp009()
    config.runtime = SingleProcess.Config(device="cpu")
    config.global_batch_size = 4
    config.max_steps = math.inf

    with pytest.raises(ValueError, match=r"^TrainLoop has no finite stop condition"):
        config.copy_tree().finalize().make()


@pytest.mark.parametrize(
    ("name", "parent", "factory"),
    [
        ("eval006", experiments.exp006, experiments.eval006),
        ("eval009", experiments.exp009, experiments.eval009),
        ("eval010", experiments.exp010, experiments.eval010),
    ],
    ids=["eval006", "eval009", "eval010"],
)
def test_evaluation_forks_only_load_weights_and_skip_training(
    name: str,
    parent: Callable[[], experiments.Arc1TrainLoop | experiments.CharmTrainLoop.Config],
    factory: Callable[
        [],
        experiments.Arc1TrainLoop | experiments.CharmTrainLoop.Config,
    ],
) -> None:
    base, fork = parent(), factory()
    assert isinstance(fork.step, CheckpointActTrainStep.Config)
    assert fork.eval_only is True
    assert fork.checkpointer is None
    assert isinstance(fork.tracker, FileTracker.Config)
    assert fork.experiment_name == name
    base.step = CheckpointActTrainStep.Config().update(base.step)
    if fork.step.ema_initialization is None:
        base.step.ema_initialization = None
    base.experiment_name = fork.experiment_name
    base.eval_only = True
    base.eval_warmup_batches = fork.eval_warmup_batches
    base.checkpointer = None
    base.tracker = FileTracker.Config()
    assert base.pformat() == fork.pformat()


def test_charm_ema_update_keeps_the_source_operation_order() -> None:
    ema = experiments.exp009().step.ema
    assert isinstance(ema, EMA.Config)
    assert ema.update_fn is not None
    shadow = torch.tensor([1.0, 3.0], dtype=torch.float64)
    live = torch.tensor([5.0, -1.0], dtype=torch.float64)

    ema.update_fn(shadow, live, 0.75)

    assert shadow.tolist() == [0.25 * 5.0 + 0.75 * 1.0, 0.25 * -1.0 + 0.75 * 3.0]


@pytest.mark.parametrize(("rank", "lm_loss"), [(0, 4.0), (1, 2.0)])
def test_charm_source_metrics_sum_across_ranks_on_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    lm_loss: float,
) -> None:
    """Rank 0 reports the reduced sums; other ranks keep their local values."""
    distributed = torch.distributed

    def reduce(values: Tensor, *, dst: int, op: object) -> None:
        del op
        assert dst == 0
        values.mul_(2)

    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(distributed, "reduce", reduce)
    metrics = {
        "accuracy": 2.0,
        "count": 4.0,
        "exact_accuracy": 0.0,
        "lm_loss": 8.0,
        "q_halt_accuracy": 0.0,
        "q_halt_loss": 0.0,
        "steps": 0.0,
    }

    reported = experiments._report_source_act_metrics(metrics, global_batch_size=4)

    assert reported["lm_loss"] == lm_loss


def test_charm_rejects_non_data_parallel_mesh_dimensions() -> None:
    config = experiments.exp009()
    assert isinstance(config.runtime, MultiProcess.Config)
    config.runtime.mesh_topology = {"dp": 2, "pp": 1, "tp": 2}

    finalized = config.copy_tree().finalize()

    with pytest.raises(ValueError, match="tp"):
        finalized.make()


def test_charm_release_paths_follow_the_loop_base_dir(tmp_path: Path) -> None:
    config = experiments.exp009()
    config.base_dir = tmp_path

    finalized = config.copy_tree().finalize()

    source_root = tmp_path / "datasets/charm-release-937316c8/data/augmented/v2"
    assert [Path(source.path) for source in finalized.dataset.train_sources] == [
        source_root / name for name in _charm_source_names("exp009")
    ]
    metric = finalized.metrics_eval[""]
    assert isinstance(metric, ParquetPassK.Config)
    assert Path(metric.eval_source) == source_root / (_charm_source_names("exp009")[-1])


def test_charm_metric_paths_follow_dataset_overrides(tmp_path: Path) -> None:
    config = experiments.exp009()
    config.dataset.base_dir = tmp_path / "dataset-root"
    config.dataset.eval_source = "override.parquet"

    finalized = config.copy_tree().finalize()

    metric = finalized.metrics_eval[""]
    assert isinstance(metric, ParquetPassK.Config)
    assert metric.base_dir == finalized.dataset.base_dir
    assert metric.eval_source == finalized.dataset.eval_source


def test_charm_dataset_canvas_owns_model_pool_and_metric_geometry() -> None:
    """A tiny declared canvas reaches every finalized CHARM geometry owner."""
    config = experiments.exp009()
    config.dataset.canvas_size = 2

    finalized = config.copy_tree().finalize()

    model = finalized.step.model
    assert isinstance(model, SingleLatentGridSolver.Config)
    assert isinstance(model.embedding, GridEmbedding.Config)
    assert model.embedding.grid_shape == (2, 2)
    assert model.grid_len == 4
    assert model.rope_grid_shape == (4,)
    assert model.total_seq_len == 20
    assert finalized.step.pool.seq_len == 20
    metric = finalized.metrics_eval[""]
    assert isinstance(metric, ParquetPassK.Config)
    assert metric.canvas_size == 2


def test_charm_finalization_derives_global_batch_denominators() -> None:
    config = experiments.exp009()
    config.global_batch_size = 384

    finalized = config.copy_tree().finalize()

    assert finalized.step.backward_loss_scale == 1.0 / 384
    reporter_config = finalized.step.metric_reporter
    assert isinstance(reporter_config, PartialConfig)
    reporter = cast(
        Callable[[Mapping[str, Tensor | float]], dict[str, float]],
        reporter_config.make(),
    )
    reported = reporter(
        {
            "accuracy": 1.0,
            "count": 1.0,
            "exact_accuracy": 1.0,
            "lm_loss": 384.0,
            "q_halt_accuracy": 1.0,
            "q_halt_loss": 384.0,
            "steps": 1.0,
        },
    )
    assert reported["lm_loss"] == 1.0
    assert reported["q_halt_loss"] == 1.0


def test_charm_finalization_preserves_custom_metric_reporter() -> None:
    def report_custom_metrics(
        metrics: Mapping[str, Tensor | float],
    ) -> dict[str, float]:
        del metrics
        return {"custom": 1.0}

    config = experiments.exp009()
    config.global_batch_size = 384
    config.step.metric_reporter = PartialConfig(report_custom_metrics)

    finalized = config.copy_tree().finalize()

    reporter = finalized.step.metric_reporter
    assert isinstance(reporter, PartialConfig)
    function, args, kwargs = reporter.deferred_call()
    assert function is report_custom_metrics
    assert args == []
    assert kwargs == {}


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


def test_an_arcagi2_rung_derives_the_rotary_grid_from_its_dataset() -> None:
    """Set the 2D RoPE lattice from the ARC-AGI-2 canvas."""
    config = experiments.exp007()
    config.step.model.rope = RoPE.Config(channels_head=[32, 32])

    finalized = config.finalize()

    assert finalized.step.model.rope_grid_shape == finalized.dataset.spec.grid_shape


def test_charm_single_process_finalization_sets_local_batch_geometry() -> None:
    """Use the full global batch locally when no data mesh exists."""
    config = experiments.exp009()
    config.runtime = SingleProcess.Config(device="cpu")
    config.global_batch_size = 4

    finalized = config.finalize()

    assert finalized.dataset.batch_size == 4
    assert finalized.dataset.eval_batch_size == 4
    assert finalized.step.pool.batch_size == 4


def test_charm_finalization_keeps_an_optional_prefix_disabled() -> None:
    config = experiments.exp009()
    assert isinstance(config.step.model, SingleLatentGridSolver.Config)
    config.step.model.prefix = None

    finalized = config.copy_tree().finalize()

    assert finalized.step.model.prefix is None
    assert finalized.dataset.batch_size == 96
    assert finalized.dataset.eval_batch_size == 96
    assert finalized.step.pool.batch_size == 96


def test_charm_source_metrics_use_the_published_denominators() -> None:
    metrics = {
        "accuracy": 2.0,
        "count": 4.0,
        "exact_accuracy": 1.0,
        "lm_loss": 8.0,
        "q_halt_accuracy": 3.0,
        "q_halt_loss": 12.0,
        "steps": 20.0,
    }

    assert experiments._report_source_act_metrics(metrics, global_batch_size=4) == {
        "accuracy": 0.5,
        "count": 1.0,
        "exact_accuracy": 0.25,
        "lm_loss": 2.0,
        "q_halt_accuracy": 0.75,
        "q_halt_loss": 3.0,
        "steps": 5.0,
    }


@pytest.mark.parametrize("factory", [experiments.exp009, experiments.exp010])
def test_charm_finalized_prefix_trains_at_the_local_batch_size(
    factory: Callable[[], experiments.CharmTrainLoop.Config],
) -> None:
    config = factory()
    assert isinstance(config.step.model, SingleLatentGridSolver.Config)
    assert isinstance(config.step.model.prefix, ComposedSparsePrefix.Config)
    config.step.model.channels_in = 20
    prefix = config.step.model.prefix
    prefix.channels_out = 20
    prefix.composition.channels_out = 20
    prefix.residual.num_puzzles = 128
    prefix.residual.channels_in = 2
    prefix.residual.channels_out = 2

    finalized = config.copy_tree().finalize()

    assert finalized.dataset.batch_size == 96
    assert finalized.dataset.eval_batch_size == 96
    assert finalized.step.pool.batch_size == 96
    assert isinstance(finalized.step.model.prefix, ComposedSparsePrefix.Config)
    assert finalized.step.model.prefix.residual.batch_size == 96
    prefix_module = finalized.step.model.prefix.make()
    prefix_module.train()
    batch_size = finalized.dataset.batch_size
    category_indices = torch.arange(batch_size) % _charm_tasks(
        finalized.experiment_name,
    )
    augmentation_indices = torch.arange(batch_size) % 8
    slot_permutation = torch.remainder(
        torch.arange(9) + torch.arange(batch_size).unsqueeze(1),
        9,
    )
    puzzle_identifiers = torch.arange(1, batch_size + 1)

    output = prefix_module(
        batch_size,
        category_indices=category_indices,
        augmentation_indices=augmentation_indices,
        slot_permutation=slot_permutation,
        puzzle_identifiers=puzzle_identifiers,
    )

    assert output.shape == (96, 16, 20)


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
    metrics = json_loads(
        (tmp_path / "runs/arcagi/exp_smoke/metrics.json").read_text(),
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


def _write_two_task_eval_source(path: Path) -> None:
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "format_version": 2,
                    "puzzle_id": "task-a",
                    "train": [],
                    "test": [{"input": [[1]], "output": [[2]]}],
                    "offline_augs": [""],
                },
                {
                    "format_version": 2,
                    "puzzle_id": "task-b",
                    "train": [],
                    "test": [{"input": [[3]], "output": [[4]]}],
                    "offline_augs": [""],
                },
            ],
        ),
        where=path,
    )


def _packed_window_candidate(grid: list[list[int]]) -> torch.Tensor:
    canvas_size = 6
    height = len(grid)
    width = len(grid[0])
    canvas = torch.zeros((canvas_size, canvas_size), dtype=torch.float32)
    canvas[1 : 1 + height, 1 : 1 + width] = torch.tensor(grid) + 2
    canvas[1 : 1 + height, 1 + width] = 1
    canvas[1 + height, 1 : 1 + width] = 1
    return torch.cat([torch.zeros(1), canvas.flatten()])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
