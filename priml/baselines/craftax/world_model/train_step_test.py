"""Check the world-model train step: optimizer split, schedule, and metrics."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import multiprocessing

from configgle import PartialConfig
from torch import nn
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import CompileCounter
from torch.distributed._composable.replicate import replicate
from torch.utils.flop_counter import FlopCounterMode

import pytest
import torch
import torch.distributed as dist

from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.metric import craftax_target_nll
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    FrameEncoder,
    WorldModel,
    gathered_board,
    multi_hot_board,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.train_step import (
    ForwardCost,
    WorldModelTrainStep,
    compile_forward,
    world_model_cost,
    wsd,
)
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.optimizers import CompositeOptimizer, FusedAdamW, Muon
from priml.optimizers.muon import adjust_lr_match_rms_adamw
from priml.train.custom_types import TrainStepOutput
from priml.train.parallelism import NoParallel


def _tiny_model(config: WorldModel.Config) -> None:
    config.encoder.channels_in = 16
    config.decoder.channels_in = 16
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack):
        _shrink_stack(stack)
    config.transformer.channels_in = 36
    config.transformer.num_layers = 1
    block = config.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.attn.channels_head = 4
    block.ffn.channels_hidden = 48


def _shrink_stack(stack: Transformer.Config) -> None:
    stack.num_layers = 1
    block = stack.block
    assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.channels_hidden = 32


def _step_config() -> WorldModelTrainStep.Config:
    """Return the default recipe at test size, eager, on the CPU."""
    config = WorldModelTrainStep.Config()
    model = config.model
    assert isinstance(model, WorldModel.Config)
    _tiny_model(model)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    # CPU autocast runs priml's RMSNorm on a BF16 input with an FP32 scale, which
    # torch warns about once per process, and warnings fail tests here.
    config.dtype_autocast = None
    config.train_budget_steps = 10
    optimizer = config.optimizer
    assert isinstance(optimizer, CompositeOptimizer.Config)
    for member in optimizer.optimizers:
        if isinstance(member, FusedAdamW.Config):
            member.compile = False
    return config


def _segment(decisions: int, *, frames: int, starts: bool, seed: int) -> Segment:
    schema = craftax_schema()
    generator = torch.Generator().manual_seed(seed)
    cells = torch.stack(
        [
            torch.randint(0, field.valid, (frames, 99), generator=generator)
            for field in schema.cell_fields
        ],
        dim=-1,
    )
    aux = torch.stack(
        [
            torch.randint(low - 155, high - 154, (frames,), generator=generator)
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    )
    return Segment(
        cells=cells.to(torch.uint8),
        aux=aux.to(torch.int16),
        actions=torch.randint(0, 43, (decisions,), generator=generator).byte(),
        reward=torch.randint(-1, 4, (decisions,), generator=generator).short(),
        done=torch.zeros(decisions, dtype=torch.bool),
        starts_episode=starts,
    )


def _batch(seed: int = 0) -> PackedBatch:
    return pack_windows(
        [
            [_segment(3, frames=4, starts=True, seed=seed)],
            [_segment(4, frames=4, starts=False, seed=seed + 1)],
        ],
        t_g=8,
        s_max=2,
    )


def _batches() -> Iterator[dict[str, object]]:
    for seed in range(100):
        yield {"media": _batch(seed), "stratum": torch.zeros(2, 8).long()}


def _counted_batch(decisions: int) -> PackedBatch:
    """Return a batch whose frame and job counts grow with ``decisions``."""
    return pack_windows(
        [
            [_segment(decisions, frames=decisions + 1, starts=True, seed=decisions)],
            [_segment(4, frames=4, starts=False, seed=0)],
        ],
        t_g=16,
        s_max=2,
    )


def test_wsd_warms_up_holds_and_decays_linearly_to_zero() -> None:
    points = (0.0, 0.025, 0.05, 0.5, 0.85, 0.9, 1.0)
    values = [wsd(p, warmup=0.1, decay=0.2) for p in points]
    # A quarter into the decay a cosine would still be at 0.854, and a quarter
    # into the warmup at 0.146.
    assert values == pytest.approx([0.0, 0.25, 0.5, 1.0, 0.75, 0.5, 0.0])


def test_optimizer_splits_matrices_from_tables_and_scales() -> None:
    step = _step_config().make()
    optimizer = step.optimizer
    assert isinstance(optimizer, CompositeOptimizer)
    adamw, matrices, heads = optimizer.optimizers
    assert isinstance(adamw, FusedAdamW)
    assert isinstance(matrices, Muon)
    assert isinstance(heads, Muon)
    names = {id(p): n for n, p in step.model.named_parameters()}

    def owned(member: torch.optim.Optimizer) -> set[str]:
        return {
            names[id(p)]
            for g in member.param_groups
            for p in cast("list[torch.Tensor]", g["params"])
        }

    tables = owned(adamw)
    assert {"table.weight", "action_embedding.weight", "action_head.weight"} <= tables
    assert {"start", "encoder.pool", "decoder.memory_null"} <= tables
    assert {"encoder.slot_embedding", "decoder.slot_embedding"} <= tables
    assert all(
        p.ndim < 2
        for n, p in step.model.named_parameters()
        if n in tables
        and "embedding" not in n
        and n not in {"table.weight", "action_head.weight"}
    )
    assert {"obs_proj.weight", "cond_proj.weight"} <= owned(matrices)
    assert all("proj_qkv" in n for n in owned(heads))
    assert all(
        p.ndim == 2
        for g in matrices.param_groups
        for p in cast("list[torch.Tensor]", g["params"])
    )
    group = adamw.param_groups[0]
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.0
    assert group["lr"] == 3e-4
    for member, ensemble in ((matrices, 0), (heads, 1)):
        assert member.adjust_lr_fn is adjust_lr_match_rms_adamw
        muon = member.param_groups[0]
        assert muon["lr"] == 3e-4
        assert muon["weight_decay"] == 0.1
        assert muon["momentum"] == 0.95
        assert muon["nesterov"] is True
        assert muon["ns_steps"] == 5
        assert muon["ensemble_dims"] == ensemble


@pytest.mark.compute_training
def test_training_lowers_the_loss_and_reports_nanochat_series() -> None:
    torch.manual_seed(0)
    step = _step_config().make()
    batch = next(_batches())
    results = [step.train_step(**batch) for _ in range(6)]
    assert float(results[-1]["loss"]) < float(results[0]["loss"])
    metrics = _metrics(results[-1])
    assert set(metrics) == {
        "loss",
        "lrm",
        "dt",
        "tok_per_sec",
        "mfu",
        "grad_norm",
        "clip_fraction",
        "max_attention_logit",
        "total_training_flops",
        "total_training_time",
    }
    assert step.gradient_clip_norm == 1.0
    assert metrics["lrm"] == pytest.approx(wsd(5 / 10, warmup=0.1, decay=0.1))
    flops = 3 * world_model_cost(step.model, batch["media"]).flops
    assert metrics["total_training_flops"] == pytest.approx(6 * flops)


@pytest.mark.compute_training
def test_loss_is_nanochats_debiased_ema_of_the_objective() -> None:
    step = _step_config().make()
    batches = _batches()
    results = [step.train_step(**next(batches)) for _ in range(3)]
    # nanochat's base_train.py: the EMA starts at 0 and is divided by
    # 1 - beta**(step + 1), step counting from 0.
    smooth = 0.0
    for update, result in enumerate(results, start=1):
        smooth = 0.9 * smooth + 0.1 * float(result["model"])
        debiased = smooth / (1 - 0.9**update)
        assert float(_metrics(result)["loss"]) == pytest.approx(debiased)


@pytest.mark.compute_training
def test_loss_debiases_by_this_process_updates_after_a_resume() -> None:
    trained = _step_config().make()
    batches = _batches()
    for _ in range(2):
        trained.train_step(**next(batches))
    resumed = _step_config().make()
    resumed.load_state_dict(trained.state_dict())
    assert resumed.global_step == 2
    result = resumed.train_step(**next(batches))
    # The EMA is not checkpointed and restarts at zero, so the first update after
    # a resume divides by 1 - beta and reports its own objective; debiasing by
    # the global step would report 0.1 / (1 - 0.9**3) of it.
    assert float(_metrics(result)["loss"]) == pytest.approx(float(result["model"]))


@pytest.mark.compute_training
def test_window_loss_is_the_mean_of_its_micro_batches() -> None:
    config = _step_config()
    config.accumulate_grad_batches = 2
    step = config.make()
    batches = _batches()
    first = step.train_step(**next(batches))
    second = step.train_step(**next(batches))
    mean = (float(first["model"]) + float(second["model"])) / 2
    assert float(_metrics(second)["loss"]) == pytest.approx(mean)


@pytest.mark.compute_training
def test_accumulation_divides_each_micro_batch_gradient() -> None:
    batch = next(_batches())
    torch.manual_seed(0)
    single = _step_config().make()
    once = _metrics(single.train_step(**batch))
    config = _step_config()
    config.accumulate_grad_batches = 2
    torch.manual_seed(0)
    double = config.make()
    double.train_step(**batch)
    twice = _metrics(double.train_step(**batch))
    assert float(twice["grad_norm"]) == pytest.approx(float(once["grad_norm"]))


def test_throughput_divides_the_window_by_its_time() -> None:
    config = _step_config()
    config.device_peak_flops = 1e9
    step = config.make()
    batch = next(_batches())
    metrics = _metrics(step.train_step(**batch))
    cost = world_model_cost(step.model, batch["media"])
    dt = float(metrics["dt"])
    assert float(metrics["tok_per_sec"]) == pytest.approx(cost.positions / dt)
    assert float(metrics["mfu"]) == pytest.approx(3 * cost.flops / dt / 1e9)
    assert metrics["total_training_flops"] == pytest.approx(3 * cost.flops)


@pytest.mark.compute_training
def test_throughput_counts_every_micro_batch_of_the_window() -> None:
    config = _step_config()
    config.accumulate_grad_batches = 2
    config.device_peak_flops = 1e9
    step = config.make()
    window = [_batch(0), _batch(1)]
    step.train_step(media=window[0])
    metrics = _metrics(step.train_step(media=window[1]))
    costs = [world_model_cost(step.model, media) for media in window]
    dt = float(metrics["dt"])
    positions = sum(cost.positions for cost in costs)
    flops = 3 * sum(cost.flops for cost in costs)
    assert float(metrics["tok_per_sec"]) == pytest.approx(positions / dt)
    assert float(metrics["mfu"]) == pytest.approx(flops / dt / 1e9)
    assert metrics["total_training_flops"] == pytest.approx(flops)


@pytest.mark.compute_training
def test_total_training_time_excludes_the_first_update() -> None:
    step = _step_config().make()
    batches = _batches()
    updates = [_metrics(step.train_step(**next(batches))) for _ in range(3)]
    assert updates[0]["total_training_time"] == 0
    later = float(updates[1]["dt"]) + float(updates[2]["dt"])
    assert updates[2]["total_training_time"] == pytest.approx(later)


@pytest.mark.compute_training
def test_clip_fraction_is_the_share_of_clipped_updates() -> None:
    step = _step_config().make()
    batches = _batches()
    fractions: list[float] = []
    for ceiling in (1e9, 1e-9, 1e-9):
        step.gradient_clip_norm = ceiling
        metrics = _metrics(step.train_step(**next(batches)))
        fractions.append(float(metrics["clip_fraction"]))
    assert fractions == pytest.approx([0.0, 1 / 2, 2 / 3])


@pytest.mark.compute_training
def test_max_attention_logit_is_the_largest_of_the_window() -> None:
    config = _step_config()
    config.accumulate_grad_batches = 2
    step = config.make()
    model = step.model
    assert isinstance(model, WorldModel)
    largest: list[float] = []
    results: list[TrainStepOutput] = []
    for seed in (3, 0):
        results.append(step.train_step(media=_batch(seed)))
        largest.append(float(model.transformer.max_attention_logit()))
    # The first micro-batch holds the maximum, so reporting the last forward's
    # value alone would fail.
    assert largest[0] > largest[1]
    assert float(_metrics(results[1])["max_attention_logit"]) == max(largest)


@pytest.mark.compute_training
def test_max_attention_logit_restarts_each_update() -> None:
    step = _step_config().make()
    model = step.model
    assert isinstance(model, WorldModel)
    largest: list[float] = []
    reported: list[float] = []
    for seed in (3, 0):
        metrics = _metrics(step.train_step(media=_batch(seed)))
        reported.append(float(metrics["max_attention_logit"]))
        largest.append(float(model.transformer.max_attention_logit()))
    # The first update holds the larger logit, so a maximum carried over from
    # past updates would report it again for the second.
    assert largest[0] > largest[1]
    assert reported == largest


@pytest.mark.gpu_torch_cuda
def test_series_stack_on_the_training_device() -> None:
    config = _step_config()
    config.parallelism = NoParallel.Config(device="cuda")
    step = config.make()
    result = step.train_step(media=_batch().to(torch.device("cuda")))
    tensors = [v for v in _metrics(result).values() if isinstance(v, torch.Tensor)]
    # What ``TrainLoop._do_train_step`` does before its all-reduce: one stack of
    # the loss and every tensor metric, which needs them all on one device.
    stacked = torch.stack([result["loss"].mean(), *(v.mean() for v in tensors)])
    assert stacked.device.type == "cuda"


def test_eval_loss_returns_the_flat_target_nll() -> None:
    step = _step_config().make()
    batch = next(_batches())
    result = step.eval_loss(**batch)
    with torch.no_grad():
        step.model.eval()
        expected = craftax_target_nll(step.model, batch["media"])
    assert torch.allclose(result["model"], expected, atol=1e-2)


def test_eval_loss_counts_the_targets_the_metric_counts_at_their_weights() -> None:
    step = _step_config().make()
    batch = next(_batches())
    media = batch["media"]
    assert isinstance(media, PackedBatch)
    # History, repeated micro-batches and padding weigh 0; a span's targets
    # weigh the inverse of its draw probability.
    weight = torch.zeros(2, 8)
    weight[0, 4:] = 2.5
    result = step.eval_loss(**batch, weight=weight)
    per_job = result["model"].view(len(media.job_at), -1).sum(-1)
    counted = weight.flatten()[media.job_at.long()]
    assert counted.unique().tolist() == [0.0, 2.5]
    torch.testing.assert_close(result["loss"], (per_job * counted).sum().reshape(1))
    repeated = step.eval_loss(**batch, weight=torch.zeros(2, 8))
    assert repeated["loss"].item() == 0


def test_eval_loss_restores_the_models_mode() -> None:
    step = _step_config().make()
    step.model.train()
    step.eval_loss(**next(_batches()))
    assert step.model.training


@pytest.mark.compute_training
def test_state_dict_keeps_the_cumulative_series() -> None:
    step = _step_config().make()
    batches = _batches()
    for _ in range(2):
        step.train_step(**next(batches))
    assert step.total_training_time > 0
    restored = _step_config().make()
    restored.load_state_dict(step.state_dict())
    assert restored.total_training_flops == step.total_training_flops
    assert restored.total_training_time == step.total_training_time


def test_cost_counts_every_matmul_but_attention() -> None:
    config = WorldModel.Config()
    _tiny_model(config)
    model = config.make()
    batch = _batch()
    with FlopCounterMode(display=False) as counter:
        model(batch)
    assert world_model_cost(model, batch).flops == _matmul_flops(counter)
    assert world_model_cost(model, batch).positions == 16 + 152 * len(batch.job_at)


def test_cost_does_not_change_with_how_the_frame_embedding_runs() -> None:
    costs: list[ForwardCost] = []
    for embed_board in (gathered_board, multi_hot_board):
        config = WorldModel.Config()
        _tiny_model(config)
        config.embed_board = embed_board
        costs.append(world_model_cost(config.make(), _batch()))
    assert costs[0] == costs[1]


def test_eval_uses_bfloat16_autocast_by_default() -> None:
    assert WorldModelTrainStep.Config().dtype_autocast is torch.bfloat16
    assert isinstance(WorldModelTrainStep.Config().target_nll_fn, PartialConfig)


@pytest.mark.compute_torch_compile
def test_training_compiles_once_whatever_the_frame_and_job_counts() -> None:
    counter = CompileCounter()
    config = _step_config()
    config.compile = PartialConfig(torch.compile, backend=counter, fullgraph=True)
    step = config.make()
    batches = [_counted_batch(decisions) for decisions in (3, 5, 6)]
    assert len({(len(b.aux), len(b.job_at)) for b in batches}) == 3
    torch.compiler.reset()
    try:
        for batch in batches:
            step.train_step(media=batch)
    finally:
        torch.compiler.reset()
    assert counter.frame_count == 1


@pytest.fixture
def single_rank_group(tmp_path: Path) -> Iterator[None]:
    """Hold a 1-rank gloo group open, which composable ``replicate`` needs."""
    dist.init_process_group(
        backend="gloo",
        init_method=(tmp_path / "rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


# Why ``compile_forward`` exists; when torch traces replicate's
# hooks, this fails and the default compile slot serves data parallel too.
@pytest.mark.usefixtures("single_rank_group")
@pytest.mark.compute_torch_compile
def test_fullgraph_compile_of_a_replicated_module_fails_in_its_hooks() -> None:
    model = replicate(nn.Linear(4, 2))
    torch.compiler.reset()
    try:
        compiled = torch.compile(model, fullgraph=True, backend="eager")
        with pytest.raises(Unsupported):
            compiled(torch.ones(3, 4))
    finally:
        torch.compiler.reset()


@pytest.mark.usefixtures("single_rank_group")
@pytest.mark.compute_torch_compile
def test_compile_forward_trains_a_replicated_module_under_fullgraph() -> None:
    model = replicate(nn.Linear(4, 2))
    assert isinstance(model, nn.Linear)
    counter = CompileCounter()
    compile_slot = PartialConfig(compile_forward, fullgraph=True, backend=counter)
    torch.compiler.reset()
    try:
        compiled = compile_slot.make()(model)
        assert isinstance(compiled, nn.Linear)
        compiled(torch.ones(3, 4)).sum().backward()
    finally:
        torch.compiler.reset()
    assert compiled is model
    assert counter.frame_count == 1
    assert model.weight.grad is not None
    torch.testing.assert_close(model.weight.grad, torch.full((2, 4), 3.0))


@pytest.mark.usefixtures("single_rank_group")
@pytest.mark.compute_torch_compile
def test_compile_forward_keeps_one_graph_across_gradient_buckets() -> None:
    # Two 1 MB layers fill two gradient buckets; Dynamo's DDP optimizer would
    # compile one graph per bucket.
    model = replicate(nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512)))
    counter = CompileCounter()
    torch.compiler.reset()
    try:
        compiled = compile_forward(model, backend=counter)
        assert isinstance(compiled, nn.Sequential)
        compiled(torch.ones(3, 512)).sum().backward()
    finally:
        torch.compiler.reset()
    assert counter.frame_count == 1


@pytest.mark.compute_distributed
def test_series_aggregate_over_ranks(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ranks = [
        context.Process(
            target=_check_world_series,
            args=(rank, 2, str(tmp_path / "store")),
        )
        for rank in range(2)
    ]
    try:
        for process in ranks:
            process.start()
        for process in ranks:
            process.join(timeout=120)
            assert process.exitcode == 0
    finally:
        for process in ranks:
            if process.is_alive():
                process.kill()
                process.join()


def _metrics(result: TrainStepOutput) -> dict[str, float | torch.Tensor]:
    """Return the series of a call that closed an accumulation window."""
    metrics = result.get("metrics")
    assert metrics is not None
    return metrics


# ``SdpaVarlen`` forms its scores a second time, under no grad, to report the max logit;
# those run inside the kernel module, as the scores do.
def _matmul_flops(counter: FlopCounterMode) -> int:
    """Return the counted FLOPs of every matmul except attention scores."""
    counts = counter.get_flop_counts()
    total = sum(
        flops
        for op, flops in cast("dict[object, int]", counts["Global"]).items()
        if "attention" not in str(op)
    )
    probe = sum(
        flops
        for name, ops in counts.items()
        if name.endswith("attn_kernel")
        for op, flops in cast("dict[object, int]", ops).items()
        if "attention" not in str(op)
    )
    return total - probe


def _check_world_series(rank: int, world: int, store: str) -> None:
    """Train one update as ``rank`` of ``world`` and check the global series."""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world,
    )
    try:
        config = _step_config()
        config.device_peak_flops = 1e9
        step = config.make()
        model = step.model
        assert isinstance(model, WorldModel)
        if rank == 1:
            # A larger query-norm scale multiplies every logit: a spike on one rank.
            with torch.no_grad():
                for name, parameter in model.transformer.named_parameters():
                    if "norm_q" in name:
                        parameter.mul_(4)
        batch = next(_batches())
        metrics = _metrics(step.train_step(**batch))
        cost = world_model_cost(step.model, batch["media"])
        dt = float(metrics["dt"])
        assert float(metrics["tok_per_sec"]) == pytest.approx(
            world * cost.positions / dt,
        )
        assert float(metrics["mfu"]) == pytest.approx(3 * cost.flops / dt / 1e9)
        assert metrics["total_training_flops"] == pytest.approx(world * 3 * cost.flops)
        local = torch.zeros(world)
        local[rank] = model.transformer.max_attention_logit()
        dist.all_reduce(local)
        assert float(local[1]) > 2 * float(local[0])
        reported = metrics["max_attention_logit"]
        assert isinstance(reported, torch.Tensor)
        # What ``TrainLoop._do_train_step`` logs: every tensor metric averaged
        # over ranks.
        logged = reported.clone()
        dist.all_reduce(logged)
        assert float(logged / world) == float(local.max())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
