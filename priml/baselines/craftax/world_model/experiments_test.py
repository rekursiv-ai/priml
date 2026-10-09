"""Check the world-model experiments: every factory builds, and exp000 is pinned.

``exp000`` is checked by value and by a golden of its whole finalized config:
it is the control every fork is measured against, so a change to it must be
deliberate. ``exp_smoke`` trains end to end on a synthetic corpus written with
``archive.write_shard``, so the loop, loader, model, step, metric, and
checkpointer are exercised together.
"""

from dataclasses import fields, is_dataclass
from functools import partial
from pathlib import Path
from typing import (
    Final,
    Protocol,
    cast,
)

import re

from configgle import InlineConfig, PartialConfig

import pytest
import torch

from priml.baselines.craftax.world_model import experiments
from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
    Receipt,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.data import (
    EvalSpans,
    ReplayStream,
    StratifiedWindows,
)
from priml.baselines.craftax.world_model.experiments import (
    EXP001_UPDATE_FLOPS,
    FLAT_UPDATE_FLOPS,
    WARMUP_STEPS,
    WorldModelLoop,
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp010,
    exp011,
    exp012,
    exp013,
    exp014,
    exp015,
    exp020,
    exp_smoke,
)
from priml.baselines.craftax.world_model.flat import (
    FlatModel,
    flat_cost,
    flat_positions,
)
from priml.baselines.craftax.world_model.grid_encoder import (
    GridConvEncoder,
)
from priml.baselines.craftax.world_model.metric import (
    CraftaxBitsPerByte,
    NanochatSeries,
)
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    EncoderBlock,
    FrameEncoder,
    WorldModel,
    keep_attention,
    multi_hot_board,
    to_autocast_dtype,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.testing import (
    random_segment,
    small_schema,
)
from priml.baselines.craftax.world_model.train_step import (
    WorldModelTrainStep,
    compile_forward,
    muon_adamw,
)
from priml.cost import cost
from priml.lib.codec import from_plain
from priml.math.schedules import warmup
from priml.model.attention.attention import Attention
from priml.model.attention.flash4 import Flash4Varlen
from priml.model.attention.kernel import SdpaVarlen
from priml.model.custom_types import ChannelsInOutConfig
from priml.model.linear import Linear
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.optimizers import CompositeOptimizer, FusedAdamW
from priml.runtime import MultiProcess, SingleProcess
from priml.testing.golden import assert_pprint_golden
from priml.train.activation import DefaultActivationStorage
from priml.train.checkpointer import Checkpointer
from priml.train.custom_types import TrainStepOutput
from priml.train.parallelism import DataParallel, NoParallel
from priml.train.tracker import (
    AsyncTracker,
    FileTracker,
    TrackerList,
    WandbTracker,
)


class _Experiment(Protocol):
    """An experiment factory, named so tests can report which one failed."""

    __name__: str

    def __call__(self) -> WorldModelLoop.Config: ...


def _name(factory: _Experiment) -> str:
    return factory.__name__


ALL_EXPERIMENTS: Final[list[_Experiment]] = [
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp010,
    exp011,
    exp012,
    exp013,
    exp014,
    exp015,
    exp020,
    exp_smoke,
]

_ADDRESS: Final = re.compile(r" at 0x[0-9a-f]+")
"""A memory address in a repr: a function or object built anew by each factory call."""


def test_the_loop_binds_its_step_and_dataset() -> None:
    world = WorldModelLoop.Config()
    assert isinstance(world.step, WorldModelTrainStep.Config)
    assert isinstance(world.dataset, ReplayStream.Config)
    assert WorldModelLoop.Config.parent_class is WorldModelLoop


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=_name)
def test_every_experiment_finalizes(factory: _Experiment) -> None:
    assert factory().copy_tree().finalize() is not None


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=_name)
def test_experiment_name_matches_the_factory(factory: _Experiment) -> None:
    assert factory().experiment_name == factory.__name__


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=_name)
def test_every_experiment_runs_under_the_ports_own_study_and_project(
    factory: _Experiment,
) -> None:
    # The original implementation's runs share these experiment names under
    # their own study: the port must neither resume them from /runs nor log
    # beside them in W&B.
    config = factory()
    assert config.study_name == "craftax-world-model"
    if config.tracker is not None:
        assert _wandb(config).project == "craftax-world-model"


@pytest.mark.parametrize("factory", ALL_EXPERIMENTS, ids=_name)
def test_construction_reads_no_files(
    factory: _Experiment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("experiment construction must not read files")

    monkeypatch.setattr(torch, "load", boom)
    monkeypatch.setattr(Path, "read_bytes", boom)
    monkeypatch.setattr(Path, "read_text", boom)
    _ = factory().copy_tree().finalize()


def test_exp000_pins_the_plan_recipe() -> None:
    """The nodes exp001 replaces, stated whole; exp001's golden pins every other."""
    cfg = exp000()
    runtime = cfg.runtime
    assert isinstance(runtime, MultiProcess.Config)
    assert runtime == MultiProcess.Config(
        device="cuda",
        deterministic=False,
        float32_matmul_precision=None,
        backend=None,
        mesh_topology={"dp": 8, "pp": 1, "tp": 1},
    )
    assert cfg.step.parallelism == DataParallel.Config(
        mesh_dim="dp",
        bucket_cap_mb=25,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
    )
    assert _stratified(cfg).batches == 8
    assert _recomputed(cfg) == set()
    # Eight windows of 8,192 positions per optimizer step, one per GPU.
    assert (cfg.step.accumulate_grad_batches, cfg.dataset.micro_batches_per_step) == (
        1,
        1,
    )
    dp = runtime.mesh_topology["dp"]
    assert dp * cfg.dataset.windows * cfg.step.accumulate_grad_batches == 8
    assert cfg.dataset.t_g == 8_192
    assert cfg.max_steps == cfg.step.train_budget_steps
    decisions = cfg.max_steps * 8 * cfg.dataset.t_g // 2
    assert 49_000_000 < decisions <= 50_000_000
    assert cfg.step.gradient_clip_norm == 1.0
    assert cfg.step.dtype_autocast is torch.bfloat16
    model = cfg.step.model
    assert isinstance(model, WorldModel.Config)
    assert model.schema == craftax_schema()
    block = model.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.attn.attn_kernel, Flash4Varlen.Config)
    assert isinstance(cfg.metrics_eval["val"], CraftaxBitsPerByte.Config)


def test_exp000_logs_nanochat_series_to_files_and_wandb() -> None:
    tracker = exp000().tracker
    assert isinstance(tracker, NanochatSeries.Config)
    trackers = tracker.tracker
    assert isinstance(trackers, TrackerList.Config)
    files, val, wandb = trackers.trackers.values()
    assert isinstance(files, FileTracker.Config)
    assert isinstance(val, FileTracker.Config)
    assert val.capture_prefix == "val/"
    assert isinstance(wandb, AsyncTracker.Config)
    assert isinstance(wandb.tracker, WandbTracker.Config)


def test_exp001_is_exp000_accumulated_on_one_gpu() -> None:
    cfg = exp001()
    windows = cfg.dataset.windows
    expected = exp000()
    expected.experiment_name = "exp001"
    expected.runtime = SingleProcess.Config(device="cuda")
    expected.step.parallelism = NoParallel.Config()
    expected.dataset.windows = windows
    # exp000's update: 8 ranks x 1 window; its validation: 8 ranks x 8 windows.
    accumulate = 8 // windows
    expected.step.accumulate_grad_batches = accumulate
    expected.dataset.micro_batches_per_step = accumulate
    _stratified(expected).batches = 64 // windows
    _blocks(expected)["decoder"].checkpoint = True
    assert windows * cfg.step.accumulate_grad_batches == 8
    assert windows * _stratified(cfg).batches == 64
    _assert_same(cfg, expected)


def test_a_loop_refuses_a_sampler_keyed_unlike_its_accumulation() -> None:
    # The sampler keys micro-batches by (step, micro-step): counting one per
    # update while the step accumulates two, update 0 would read the keys of
    # updates 0 and 1.
    cfg = exp_smoke()
    cfg.step.accumulate_grad_batches = 2
    with pytest.raises(ValueError, match="micro_batches_per_step=1"):
        cfg.make()


def test_exp001_recomputes_each_local_decoder_block_in_backward() -> None:
    # Measured on one H200 at 8,192 positions: storing every activation peaked
    # at 141 GB of 143 and ran out of memory under the default allocator.
    assert _recomputed(exp001()) == {"decoder"}


def test_exp002_doubles_the_window_at_equal_decisions_per_update() -> None:
    cfg, parent = exp002(), exp001()
    windows = cfg.dataset.windows
    expected = exp001()
    expected.experiment_name = "exp002"
    expected.dataset.t_g = 16_384
    expected.dataset.windows = windows
    expected.step.accumulate_grad_batches = 4 // windows
    expected.dataset.micro_batches_per_step = 4 // windows
    _stratified(expected).batches = 32 // windows
    _blocks(expected)["encoder"].checkpoint = True
    _assert_same(cfg, expected)
    for per_update, per_evaluation, config in (
        (4 * 16_384, 32 * 16_384, cfg),
        (8 * 8_192, 64 * 8_192, parent),
    ):
        positions = config.dataset.windows * config.dataset.t_g
        assert positions * config.step.accumulate_grad_batches == per_update
        assert positions * _stratified(config).batches == per_evaluation
    assert cfg.max_steps == parent.max_steps == exp000().max_steps


def test_exp002_recomputes_every_encoder_and_decoder_block() -> None:
    # Measured on one H200 at 16,384 positions: recomputing only the decoder
    # blocks ran out of memory.
    assert _recomputed(exp002()) == {"encoder", "decoder"}


def test_exp010_is_exp002_with_its_speed_settings_and_natural_validation() -> None:
    cfg = exp010()
    expected = exp002()
    expected.experiment_name = "exp010"
    expected.seed = 0
    model = expected.step.model
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    for stack in (model.encoder.stack, model.decoder.stack):
        template = stack.block
        assert isinstance(template, TransformerBlock.Config | DecoderBlock.Config)
        blocks = [template.copy_tree() for _ in range(4)]
        blocks[3].checkpoint = False
        stack.block = list[ChannelsInOutConfig](blocks)
    model.encoder.cast_stream = model.decoder.cast_stream = to_autocast_dtype
    expected.dataset.count_multiple = 64
    expected.dataset.validation = EvalSpans.Config()
    _assert_same(cfg, expected)


def test_exp010_recomputes_local_blocks_0_to_2_only() -> None:
    model = exp010().step.model.copy_tree().finalize()
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    for stack in (model.encoder.stack, model.decoder.stack, model.transformer):
        assert isinstance(stack.block, list)
        flags: list[bool] = []
        for block in stack.block:
            assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
            flags.append(block.checkpoint)
        local = stack is not model.transformer
        assert flags == [local and depth < 3 for depth in range(stack.num_layers)]


def test_exp011_is_exp010_with_the_three_speed_settings() -> None:
    expected = exp010()
    expected.experiment_name = "exp011"
    model = expected.step.model
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    model.embed_board = multi_hot_board
    for stack in (model.encoder.stack, model.decoder.stack):
        assert isinstance(stack.block, list)
        for block in stack.block:
            assert isinstance(block, (EncoderBlock.Config, DecoderBlock.Config))
            assert isinstance(block.ffn, SwiGLU.Config)
            block.ffn.split_gate_projection = True
            block.recompute_policy = keep_attention
    global_block = model.transformer.block
    assert isinstance(global_block, TransformerBlock.Config)
    assert isinstance(global_block.ffn, SwiGLU.Config)
    global_block.ffn.split_gate_projection = True
    _assert_same(exp011(), expected)


def test_exp011_splits_every_gate_and_keeps_every_local_attention() -> None:
    model = exp011().step.model
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    local = [model.encoder.stack.block, model.decoder.stack.block]
    blocks = [block for stack in local if isinstance(stack, list) for block in stack]
    assert len(blocks) == 8
    for block in [*blocks, model.transformer.block]:
        assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        assert block.ffn.split_gate_projection
    for block in blocks:
        assert isinstance(block, (EncoderBlock.Config, DecoderBlock.Config))
        assert block.recompute_policy is keep_attention


def test_exp012_is_exp011_at_half_the_decisions_per_update() -> None:
    expected = exp011()
    expected.experiment_name = "exp012"
    expected.step.accumulate_grad_batches = 2
    expected.dataset.micro_batches_per_step = 2
    expected.max_steps = expected.step.train_budget_steps = 3_051
    _assert_same(exp012(), expected)


def test_exp012_trains_exp010s_decisions_and_warms_up_over_the_same_ones() -> None:
    base, fork = exp010(), exp012()
    windows = [c.dataset.windows * c.step.accumulate_grad_batches for c in (base, fork)]
    assert [
        w * c.dataset.t_g // 2 for w, c in zip(windows, (base, fork), strict=True)
    ] == [
        32_768,
        16_384,
    ]
    assert fork.max_steps * windows[1] == pytest.approx(
        base.max_steps * windows[0],
        rel=1e-3,
    )
    for config in (base, fork):
        schedule = config.step.lr_schedule
        assert isinstance(schedule, PartialConfig)
        assert from_plain(cast("object", schedule.warmup), float) == 0.1


def test_exp013_is_exp012_at_one_window_per_update() -> None:
    expected = exp012()
    expected.experiment_name = "exp013"
    expected.step.accumulate_grad_batches = 1
    expected.dataset.micro_batches_per_step = 1
    expected.max_steps = expected.step.train_budget_steps = 6_103
    expected.num_steps_eval = 500
    _assert_same(exp013(), expected)


def test_exp013_trains_and_validates_on_exp012s_decisions() -> None:
    base, fork = exp012(), exp013()
    per_update = [
        c.dataset.windows * c.step.accumulate_grad_batches * c.dataset.t_g // 2
        for c in (base, fork)
    ]
    assert per_update == [16_384, 8_192]
    assert fork.max_steps * per_update[1] == pytest.approx(
        base.max_steps * per_update[0],
        rel=1e-3,
    )
    # Validation every 4.1M decisions, as exp012's every 250 updates.
    assert fork.num_steps_eval * per_update[1] == base.num_steps_eval * per_update[0]
    schedule = fork.step.lr_schedule
    assert isinstance(schedule, PartialConfig)
    assert from_plain(cast("object", schedule.warmup), float) == 0.1


def test_exp015_is_exp014_with_the_grid_encoder() -> None:
    expected = exp014()
    expected.experiment_name = "exp015"
    model = expected.step.model
    assert isinstance(model, WorldModel.Config)
    model.encoder = GridConvEncoder.Config()
    expected.step.optimizer = muon_adamw(adamw_only=("depthwise",))
    _assert_same(exp015(), expected)


def test_exp015_trains_depthwise_kernels_with_adamw_and_matrices_with_muon() -> None:
    optimizer = exp015().step.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    adamw, muon, _ = optimizer.select
    torch.manual_seed(0)
    encoder = GridConvEncoder.Config(channels_in=16, num_slots=150).make()
    names = {f"encoder.{n}": p for n, p in encoder.named_parameters()}
    kernels = [n for n in names if "depthwise" in n]
    assert len(kernels) == 2
    assert all(adamw(n, names[n]) and not muon(n, names[n]) for n in kernels)
    assert muon("encoder.blocks.0.up", names["encoder.blocks.0.up"])
    assert muon("encoder.film", names["encoder.film"])


def test_exp014_is_exp013_on_the_converted_replay_shards() -> None:
    fork = exp014()
    assert str(fork.dataset.working_dir).endswith(
        "/datasets/craftax/world-model-reference/archive-v1-replay",
    )
    assert fork.dataset.corpus == "corpora/base.json"
    expected = exp013()
    expected.experiment_name = "exp014"
    expected.dataset.working_dir = (
        "/datasets/craftax/world-model-reference/archive-v1-replay"
    )
    _assert_same(fork, expected)
    assert _flat(fork) != _flat(exp013())


def test_exp020_scales_every_module_of_exp013_five_fold() -> None:
    sizes = [_module_sizes(_finalized_model(config)) for config in (exp013(), exp020())]
    for name in ("encoder", "transformer", "decoder", "total"):
        assert sizes[1][name] / sizes[0][name] == pytest.approx(5, rel=0.01), name
    assert sizes[1]["total"] == 1_566_925_312
    model = exp020().step.model.copy_tree().finalize()
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    assert model.encoder.channels_in == model.decoder.channels_in == 1_024
    assert model.transformer.channels_in == 2_304
    depths = [
        s.num_layers
        for s in (model.encoder.stack, model.transformer, model.decoder.stack)
    ]
    assert depths == [5, 25, 5]
    heads: set[tuple[int, int, int]] = set()
    for stack in (model.encoder.stack, model.transformer, model.decoder.stack):
        assert isinstance(stack.block, list)
        for block in stack.block:
            assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
            assert isinstance(
                block.attn,
                Attention.Config | VarlenAttention.Config,
            )
            attention = block.attn
            heads.add(
                (attention.channels_head, attention.num_heads, attention.num_heads_kv),
            )
    # Every head 128 wide: 8 per local block, GQA 18 over 6 in the global stack.
    assert heads == {(128, 8, 8), (128, 18, 6)}


def test_exp020_trains_one_8k_window_per_gpu_on_eight_gpus() -> None:
    cfg = exp020()
    runtime = cfg.runtime
    assert isinstance(runtime, MultiProcess.Config)
    assert runtime.mesh_topology == {"dp": 8, "pp": 1, "tp": 1}
    assert isinstance(cfg.step.parallelism, DataParallel.Config)
    compile_slot = cfg.step.compile
    assert isinstance(compile_slot, PartialConfig)
    made = compile_slot.make()
    assert isinstance(made, partial)
    assert made.func is compile_forward
    assert from_plain(cast("object", compile_slot.fullgraph), bool)
    # 16,384-position windows ran out of memory at this width (see exp020).
    assert (cfg.dataset.windows, cfg.dataset.t_g) == (1, 8_192)
    model = cfg.step.model
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    for stack in (model.encoder.stack, model.decoder.stack, model.transformer):
        blocks = stack.block if isinstance(stack.block, list) else [stack.block]
        for block in blocks:
            assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
            assert block.checkpoint == (stack is not model.transformer)
    assert cfg.step.accumulate_grad_batches == 1
    # Stated whole: the factory sets only ``spans``, so a changed default moves both.
    spans = EvalSpans.Config(seed=0, spans=512, span_decisions=2_048, stratum_power=0.5)
    assert cfg.dataset.validation == cfg.dataset.train_spans == spans


def test_exp020_warms_up_then_holds_its_peak_rate_forever() -> None:
    cfg = exp020()
    schedule = cfg.step.lr_schedule.make()
    budget = cfg.step.train_budget_steps
    assert budget == WARMUP_STEPS

    def multiplier(update: int) -> float:
        # ``TrainStep.progress_complete``: updates over the budget, at most 1.
        return float(schedule(min(1.0, update / budget)))

    assert multiplier(0) == 0
    assert multiplier(WARMUP_STEPS // 2) == pytest.approx(0.5)
    for update in (WARMUP_STEPS, 10 * WARMUP_STEPS, int(cfg.max_steps)):
        assert multiplier(update) == 1
    # No decay is scheduled: the run stops by hand, far short of max_steps.
    assert cfg.max_steps * 8 * cfg.dataset.t_g // 2 > 20_000_000_000


def test_exp020_keeps_resumable_and_archival_checkpoints() -> None:
    checkpoints = exp020().checkpointer
    assert isinstance(checkpoints, Checkpointer.Config)
    assert checkpoints.resume
    assert checkpoints.keep_last_n >= 2
    assert checkpoints.keep_every % checkpoints.save_every == 0
    assert checkpoints.keep_every > checkpoints.save_every


def test_exp020_is_exp013_five_fold_on_one_eight_gpu_node() -> None:
    expected = exp013()
    expected.experiment_name = "exp020"
    model = expected.step.model
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    model.encoder.channels_in = model.decoder.channels_in = 1_024
    for stack in (model.encoder.stack, model.decoder.stack):
        assert isinstance(stack.block, list)
        template = stack.block[0].copy_tree()
        assert isinstance(template, TransformerBlock.Config | DecoderBlock.Config)
        assert isinstance(template.ffn, SwiGLU.Config)
        template.ffn.channels_hidden = 2_688
        template.checkpoint = True
        stack.num_layers = 5
        stack.block = list[ChannelsInOutConfig](
            template.copy_tree() for _ in range(stack.num_layers)
        )
    model.transformer.channels_in = 2_304
    model.transformer.num_layers = 25
    block = model.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.attn.num_heads, block.attn.num_heads_kv = 18, 6
    block.ffn.channels_hidden = 6_144
    expected.runtime = MultiProcess.Config(
        device="cuda",
        mesh_topology={"dp": 8, "pp": 1, "tp": 1},
    )
    expected.step.parallelism = DataParallel.Config()
    expected.step.compile = PartialConfig(compile_forward, fullgraph=True)
    expected.dataset.t_g = 8_192
    expected.step.train_budget_steps = WARMUP_STEPS
    expected.step.lr_schedule = PartialConfig(warmup, fraction=1.0)
    # 100B decisions, rounded up to whole updates of 8 ranks' windows.
    decisions_per_update = 8 * expected.dataset.windows * expected.dataset.t_g // 2
    expected.max_steps = -(-100_000_000_000 // decisions_per_update)
    expected.dataset.corpus = "corpora/scaleup-v1.json"
    expected.dataset.validation = EvalSpans.Config(spans=512)
    expected.dataset.train_spans = EvalSpans.Config(spans=512)
    expected.num_steps_eval = 1_800
    checkpoints = expected.checkpointer
    assert isinstance(checkpoints, Checkpointer.Config)
    checkpoints.save_every = 3_600
    checkpoints.keep_last_n = 2
    checkpoints.keep_every = 4 * 3_600
    _wandb(expected).group = "scaleup"
    _assert_same(exp020(), expected)


@pytest.mark.parametrize(
    ("factory", "name", "group"),
    [
        (exp001, "exp001", "exp001"),
        (exp010, "exp010-s0", "exp010"),
        (exp003, "exp003", "exp003"),
    ],
)
def test_wandb_runs_are_named_after_their_experiment_and_seed(
    factory: _Experiment,
    name: str,
    group: str,
) -> None:
    config = factory()
    assert _wandb(config).name == ""
    finalized = config.copy_tree().finalize()
    assert (_wandb(finalized).name, _wandb(finalized).group) == (name, group)
    _wandb(config).name, _wandb(config).group = "chosen", "round1"
    finalized = config.copy_tree().finalize()
    assert (_wandb(finalized).name, _wandb(finalized).group) == ("chosen", "round1")


@pytest.mark.compute_large_fixture
def test_exp001_matches_its_golden() -> None:
    assert_pprint_golden(test_file=__file__, name="exp001", config=exp001())


def test_exp003_runs_exp001s_stack_and_recipe_on_every_slot() -> None:
    cfg, parent = exp003(), exp001()
    model, hierarchy = cfg.step.model, parent.step.model
    assert isinstance(model, FlatModel.Config)
    assert isinstance(hierarchy, WorldModel.Config)
    _assert_same(model.lm.transformer, hierarchy.transformer)
    # The step differs from exp001's only in the model, its cost, the batch
    # accumulation, and the schedule's horizon.
    expected = parent.step
    expected.model = model
    expected.cost_fn = PartialConfig(flat_cost)
    expected.accumulate_grad_batches = 1
    expected.train_budget_steps = cfg.max_steps
    schedule = expected.lr_schedule
    assert isinstance(schedule, PartialConfig)
    schedule.warmup = 2_000 / cfg.max_steps
    _assert_same(cfg.step, expected)
    # exp000's update: 8 windows of 8,192 positions, here flat ones, each the
    # longest window of global positions whose flat form fits.
    assert cfg.dataset.windows * cfg.step.accumulate_grad_batches == 8
    assert model.context == parent.dataset.t_g
    t_g = cfg.dataset.t_g
    assert flat_positions(t_g) <= model.context < flat_positions(t_g + 1)
    # 1.00 times the small corpus's decisions sampled; its training split stays
    # decoded.
    decisions = cfg.max_steps * cfg.dataset.windows * (t_g // 2)
    assert 9_990_000 < decisions <= 10_000_000
    assert cfg.dataset.corpus == "corpora/small-10m.json"
    assert cfg.dataset.cached_decisions > 10_002_004
    assert cfg.max_steps == cfg.step.train_budget_steps


def test_exp003_update_flops_are_what_flat_cost_counts() -> None:
    cfg = exp003()
    model_config = cfg.step.model
    assert isinstance(model_config, FlatModel.Config)
    # The kernel counts no matmul FLOPs, and FA4 has no macOS build.
    _attention(model_config.lm.transformer).attn_kernel = SdpaVarlen.Config()
    # A window that opens an episode holds 53 jobs; mid-episode ones hold 52.
    t_g = cfg.dataset.t_g
    episode = _episode(t_g // 2, seed=0, split=0)
    segment = Segment(
        cells=episode.cells,
        aux=episode.aux,
        actions=episode.actions,
        reward=episode.reward,
        done=torch.zeros_like(episode.done),
        starts_episode=True,
    )
    batch = pack_windows([[segment]] * cfg.dataset.windows, t_g=t_g, s_max=1)
    # The stack's blocks are copies of one template, so the count is linear in its
    # depth: one block and two, on the meta device, give it, where building all
    # of them, even there, takes 0.2 s.
    stack = model_config.lm.transformer
    assert not isinstance(stack.block, list)
    flops: list[float] = []
    for layers in (1, 2):
        shallow = model_config.copy_tree()
        shallow.lm.transformer.num_layers = layers
        with torch.device("meta"):
            flops.append(flat_cost(shallow.make(), batch).flops)
    per_block = flops[1] - flops[0]
    update = 3 * (flops[0] + (stack.num_layers - 1) * per_block)
    assert update == pytest.approx(FLAT_UPDATE_FLOPS, rel=1e-4)


def test_exp004_is_exp001_on_exp003s_corpus_for_exp003s_flops() -> None:
    cfg, flat = exp004(), exp003()
    expected = exp001()
    expected.experiment_name = "exp004"
    expected.dataset.working_dir = flat.dataset.working_dir
    expected.dataset.corpus = flat.dataset.corpus
    expected.dataset.cached_decisions = flat.dataset.cached_decisions
    expected.max_steps = expected.step.train_budget_steps = cfg.max_steps
    _assert_same(cfg, expected)
    spent = cfg.max_steps * EXP001_UPDATE_FLOPS
    assert abs(spent - flat.max_steps * FLAT_UPDATE_FLOPS) <= EXP001_UPDATE_FLOPS / 2


def test_exp005_scores_exp004s_weights_on_exp003s_micro_batches() -> None:
    scored, trained, flat = exp005(), exp004(), exp003()
    assert scored.eval_only
    _assert_same(scored.step, trained.step)
    # Every field a validation micro-batch is drawn from matches exp003's.
    for name in (
        "working_dir",
        "corpus",
        "windows",
        "t_g",
        "s_max",
        "sampler_seed",
        "stratum_power",
        "validation",
    ):
        assert getattr(scored.dataset, name) == getattr(flat.dataset, name), name
    # Finalizing the loops resolves each checkpoint directory; only the tracker
    # is dropped, as it plays no part in where checkpoints are read.
    scored.tracker = trained.tracker = None
    scored.step = trained.step = WorldModelTrainStep.Config()
    reads, wrote = (loop.copy_tree().finalize() for loop in (scored, trained))
    checkpoints, source = reads.checkpointer, wrote.checkpointer
    assert isinstance(checkpoints, Checkpointer.Config)
    assert isinstance(source, Checkpointer.Config)
    assert checkpoints.working_dir == source.working_dir
    assert reads.working_dir != wrote.working_dir


def test_exp003_trains_and_scores_a_packed_batch() -> None:
    """Over the cut schema, frames of 7 slots, not the full 150: ``flat_test``'s layout."""
    cfg = exp003()
    t_g = 4
    model = cfg.step.model
    assert isinstance(model, FlatModel.Config)
    schema = model.schema = small_schema()
    model.context = flat_positions(t_g, frame_slots=schema.frame_slots)
    _shrink_global(model.lm.transformer)
    _run_on_cpu(cfg.step)
    torch.manual_seed(0)
    step = cfg.step.make()
    segments = [random_segment(schema, 12, seed=seed) for seed in (0, 1)]
    batch = pack_windows([[segment] for segment in segments], t_g=t_g, s_max=1)
    assert step.train_step(media=batch).get("metrics")
    bits = cfg.metrics_eval["val"]
    assert isinstance(bits, CraftaxBitsPerByte.Config)
    bits.schema = schema
    metric = bits.make()
    stratum = torch.zeros(batch.kind.shape, dtype=torch.long)
    metric.update(step.eval_loss(media=batch)["model"], media=batch, stratum=stratum)
    # Near-uniform predictions: well above zero bits and below eight per byte.
    bpb = metric.compute()["bpb"]
    assert isinstance(bpb, float)
    assert 0.5 < bpb < 8


def test_module_docstring_lists_every_experiment() -> None:
    documented = experiments.__doc__ or ""
    for factory in ALL_EXPERIMENTS:
        assert factory.__name__ in documented


def test_exp_smoke_reads_its_corpus_and_scores_validations_natural_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop hands the validation split's natural decisions to its metric.

    Then its evaluation reports the natural metrics; it runs under its study.
    """
    loop = _wired_smoke(exp_smoke(), tmp_path, monkeypatch=monkeypatch)
    assert loop.working_dir == tmp_path / "runs" / "craftax-world-model" / "exp_smoke"
    metric, dataset = loop.metrics_eval["val"], loop.dataset
    assert isinstance(metric, CraftaxBitsPerByte)
    assert isinstance(dataset, ReplayStream)
    counts = dataset.eval_sampler.counts
    assert counts.sum() == 12 + 17 + 22
    assert torch.equal(metric.natural_decisions, counts.double())
    assert not loop.metrics_train_split
    evaluation = loop.eval()
    assert {"val_bpb_natural", "val_nats_per_decision_natural"} <= evaluation.keys()


def test_every_evaluation_scores_the_training_splits_spans_beside_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation, then the training split's spans in its own natural mix, prefixed.

    Over padded counts; the two splits differ by their decisions alone.
    """
    cfg = exp_smoke()
    cfg.dataset.count_multiple = 8
    cfg.dataset.validation = EvalSpans.Config(spans=6, span_decisions=8)
    cfg.dataset.train_spans = EvalSpans.Config(spans=6, span_decisions=8)
    loop = _wired_smoke(cfg, tmp_path, monkeypatch=monkeypatch)
    evaluation = loop.eval()
    natural = evaluation["val_bpb_natural"]
    fitted_natural = evaluation["train_split_val_bpb_natural"]
    assert isinstance(natural, float)
    assert isinstance(fitted_natural, float)
    assert natural != fitted_natural
    assert evaluation["train_split_val_bpb"] != evaluation["val_bpb"]
    dataset, fitted = loop.dataset, loop.metrics_train_split["val"]
    assert isinstance(dataset, ReplayStream)
    assert isinstance(fitted, CraftaxBitsPerByte)
    assert torch.equal(fitted.natural_decisions, dataset.train_sampler.counts.double())
    assert loop.metrics_eval["val"] is not fitted
    assert loop.eval().keys() == evaluation.keys()


# The model's forward, the step, the stream and the metric are each checked in their
# own tests; what these check is how the loop wires them. So a linear map, under SGD,
# stands in for the model, which the loop builds and never runs, and ``_action_nats``
# for its scorer: building exp_smoke's model and running it took 0.4 s.
def _wired_smoke(
    cfg: WorldModelLoop.Config,
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> WorldModelLoop:
    """Build ``cfg``'s loop over a synthetic corpus, with stand-ins for the model."""
    cfg.base_dir = tmp_path
    cfg.step.model = Linear.Config(channels_in=2, channels_out=3)
    cfg.step.optimizer = PartialConfig(torch.optim.SGD, lr=0.1)
    root = tmp_path / "datasets" / "craftax" / "world-model" / "smoke"
    _write_corpus(root, corpus=Path(str(cfg.dataset.corpus)))
    loop = cfg.make()
    assert isinstance(loop, WorldModelLoop)
    monkeypatch.setattr(loop.step, "eval_loss", _action_nats)
    return loop


# Each job's record is its action, then its local slots (``metric.canonical_records``).
# Stands in for the step's ``eval_loss``, whose model output the metric reads: one NLL
# per record column.
def _action_nats(**batch: object) -> TrainStepOutput:
    """Charge every record column of each job its decision's action id plus one, in nats."""
    media = batch["media"]
    assert isinstance(media, PackedBatch)
    actions = media.action.flatten()[media.job_at.long()].float()
    columns = 1 + craftax_schema().local_slots
    nll = (1 + actions)[:, None].expand(-1, columns).flatten()
    return {"loss": nll.sum().reshape(1), "model": nll}


def _blocks(
    config: WorldModelLoop.Config,
) -> dict[str, TransformerBlock.Config | DecoderBlock.Config]:
    """Return the block template of each of the world model's three stacks."""
    model = config.step.model
    assert isinstance(model, WorldModel.Config)
    assert isinstance(model.encoder, FrameEncoder.Config)
    blocks: dict[str, TransformerBlock.Config | DecoderBlock.Config] = {}
    for name, block in (
        ("encoder", model.encoder.stack.block),
        ("transformer", model.transformer.block),
        ("decoder", model.decoder.stack.block),
    ):
        assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
        blocks[name] = block
    return blocks


def _wandb(config: WorldModelLoop.Config) -> WandbTracker.Config:
    """Return the W&B tracker of nanochat's series."""
    series = config.tracker
    assert isinstance(series, NanochatSeries.Config)
    trackers = series.tracker
    assert isinstance(trackers, TrackerList.Config)
    wandb = trackers.trackers["wandb"]
    assert isinstance(wandb, AsyncTracker.Config)
    assert isinstance(wandb.tracker, WandbTracker.Config)
    return wandb.tracker


# Each config's ``cost`` owns its parameters, which ``assert_cost_matches_torch``
# checks against the built module at test size (``model_test``); counted here at
# full size from the configs, as building them, even on the meta device, takes
# 0.3-0.5 s a model.
def _module_sizes(model: WorldModel.Config) -> dict[str, int]:
    """Count a finalized model's parameters per module from its config."""
    return {
        "encoder": cost(model.encoder, batch_size=1, dtype=None).params,
        "transformer": model.transformer.cost(
            seq_len=1,
            batch_size=1,
            dtype=None,
        ).params,
        "decoder": model.decoder.cost(batch_size=1, dtype=None, memory_len=1).params,
        "total": model.cost(
            seq_len=1,
            batch_size=1,
            frames=1,
            jobs=1,
            dtype=None,
        ).params,
    }


def _finalized_model(config: WorldModelLoop.Config) -> WorldModel.Config:
    """Return a loop's world model config, finalized, its global kernel SDPA's."""
    model = config.step.model.copy_tree()
    assert isinstance(model, WorldModel.Config)
    # FA4 imports only on Linux; the kernel holds no parameters.
    _attention(model.transformer).attn_kernel = SdpaVarlen.Config()
    return model.finalize()


def _recomputed(config: WorldModelLoop.Config) -> set[str]:
    """Return the stacks whose blocks recompute their activations in backward."""
    # Only the blocks' own flag recomputes: a second mechanism on top would
    # recompute the same activations twice.
    assert isinstance(
        config.step.activation_memoization,
        DefaultActivationStorage.Config,
    )
    return {name for name, block in _blocks(config).items() if block.checkpoint}


def _stratified(config: WorldModelLoop.Config) -> StratifiedWindows.Config:
    """Return a loop's validation set of stratified windows."""
    validation = config.dataset.validation
    assert isinstance(validation, StratifiedWindows.Config)
    return validation


def _assert_same(config: object, expected: object) -> None:
    """Assert two configs, as their factories left them, hold equal values at every field."""
    got, want = _flat(config), _flat(expected)
    differ = {
        path: (got.get(path), want.get(path))
        for path in sorted(got.keys() | want.keys())
        if got.get(path) != want.get(path)
    }
    assert not differ, differ


# Not ``pformat``: configgle's printer tokenizes what it renders, 30-110 ms a world
# model's config, where these fields compare in 1 ms. Unfinalized, as finalizing
# expands every schema table; propagation is a function of these fields, so it
# cannot hide a delta.
def _flat(config: object, path: str = "") -> dict[str, str]:
    """Return every field of a config tree by its path, each leaf as its repr."""
    children: list[tuple[str, object]] = []
    if isinstance(config, InlineConfig):
        label = repr(cast("InlineConfig[object]", config))
    elif is_dataclass(config) and not isinstance(config, type):
        label = type(config).__qualname__
        children = [
            (f".{entry.name}", cast("object", getattr(config, entry.name)))
            for entry in fields(config)
        ]
    elif isinstance(config, list | tuple):
        items = cast("list[object] | tuple[object, ...]", config)
        label = f"{type(items).__name__} of {len(items)}"
        children = [(f"[{index}]", item) for index, item in enumerate(items)]
    elif isinstance(config, dict):
        entries = cast("dict[object, object]", config)
        label = f"dict of {sorted(map(repr, entries))}"
        children = [(f"[{key!r}]", item) for key, item in entries.items()]
    else:
        label = repr(config)
    flat: dict[str, str] = {path: _ADDRESS.sub("", label)}
    for suffix, child in children:
        flat |= _flat(child, path + suffix)
    return flat


def _shrink_global(stack: Transformer.Config) -> None:
    """Cut a global stack to one layer of width 36, GQA 9:3, on SDPA."""
    stack.channels_in = 36
    stack.num_layers = 1
    attention = _attention(stack)
    attention.channels_head = 4
    attention.attn_kernel = SdpaVarlen.Config()
    block = stack.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.channels_hidden = 48


def _attention(stack: Transformer.Config) -> VarlenAttention.Config:
    """Return the attention template of a global stack's blocks."""
    block = stack.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    return block.attn


def _run_on_cpu(config: WorldModelTrainStep.Config) -> None:
    """Run a step eagerly in float32 on the CPU; the recipe is kept."""
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    config.dtype_autocast = None
    optimizer = config.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    for member in optimizer.optimizers:
        if isinstance(member, FusedAdamW.Config):
            member.compile = False


def _write_corpus(root: Path, *, corpus: Path) -> None:
    """Write one training and one validation shard of random valid episodes."""
    entries: list[tuple[Path, ManifestLine]] = []
    for split, name in enumerate(("train", "val")):
        directory = root / name / "0" / "w0"
        directory.mkdir(parents=True)
        episodes = [
            _episode(12 + 5 * i, seed=10 * split + i, split=split) for i in range(3)
        ]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        entries.append((directory, line))
    write_corpus(root / corpus, entries=entries)


def _episode(decisions: int, *, seed: int, split: int) -> Episode:
    schema = craftax_schema()
    generator = torch.Generator().manual_seed(seed)
    cells = torch.stack(
        [
            torch.randint(0, field.valid, (decisions, 99), generator=generator)
            for field in schema.cell_fields
        ],
        dim=-1,
    )
    aux = torch.stack(
        [
            torch.randint(low - 155, high - 154, (decisions,), generator=generator)
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    )
    return Episode(
        receipt=Receipt(
            world_seed=seed,
            sampling_seed=0,
            initial_state_hash=0,
            arm=0,
            split=split,
        ),
        actions=torch.randint(0, 43, (decisions,), generator=generator).byte(),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=cells.byte(),
        aux=aux.short(),
        reward=torch.randint(0, 3, (decisions,), generator=generator).short(),
        done=torch.arange(decisions) == decisions - 1,
        summary={"achievements": []},
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
