"""Check the world model: Qwen3 parity, causality, and loss counts (rung R0)."""

from collections import Counter
from collections.abc import Callable
from functools import cache, partial
from typing import override
from unittest.mock import Mock, patch

import copy
import math

from torch import Tensor, nn
from torch.utils import checkpoint
from torch.utils._python_dispatch import TorchDispatchMode

import pytest
import torch

from priml.baselines.craftax.world_model.attention import (
    VarlenAttention,
    row_cu_seqlens,
)
from priml.baselines.craftax.world_model.batch import (
    Kind,
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.loss import ModalityLoss
from priml.baselines.craftax.world_model.model import (
    BoardEmbedding,
    DecoderBlock,
    EncoderBlock,
    FrameEncoder,
    GlobalLanguageModel,
    GlobalTransformer,
    LocalDecoder,
    WorldModel,
    WorldModelLogits,
    gathered_board,
    keep_attention,
    multi_hot_board,
    to_autocast_dtype,
)
from priml.baselines.craftax.world_model.schema import (
    craftax_schema,
    done_id,
    number_id,
)
from priml.baselines.craftax.world_model.testing import naive_attention
from priml.cost import cost
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.embedding import Embedding
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.testing.cost import assert_cost_matches_torch
from priml.train.parallelism import materialize_meta, named_meta_state


def _tiny_global(config: GlobalTransformer.Config, *, layers: int) -> None:
    """Shrink the global stack to width 36: 9 query heads x 4, 3 KV heads."""
    config.channels_in = 36
    config.num_layers = layers
    block = config.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.attn.channels_head = 4
    block.ffn.channels_hidden = 48


def _tiny_local(stack: Transformer.Config) -> None:
    """Shrink an encoder or decoder stack to one layer with a narrow MLP."""
    stack.num_layers = 1
    block = stack.block
    assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.channels_hidden = 32


def _tiny_world_model_config() -> WorldModel.Config:
    config = WorldModel.Config()
    config.encoder.channels_in = 16
    config.decoder.channels_in = 16
    assert isinstance(config.encoder, FrameEncoder.Config)
    _tiny_local(config.encoder.stack)
    _tiny_local(config.decoder.stack)
    _tiny_global(config.transformer, layers=1)
    return config


def _tiny_world_model() -> WorldModel:
    """Return the tiny model seeded 0, leaving the RNG where building it leaves it."""
    model, rng = _seeded_tiny_world_model()
    torch.set_rng_state(rng)
    return copy.deepcopy(model)


# Building the model takes 15 ms and copying it 3 ms, and most tests here start
# from it; each gets its own copy, so none sees another's gradients or hooks.
@cache
def _seeded_tiny_world_model() -> tuple[WorldModel, Tensor]:
    torch.manual_seed(0)
    model = _tiny_world_model_config().make()
    return model, torch.get_rng_state()


def _tiny_language_model_config() -> GlobalLanguageModel.Config:
    config = GlobalLanguageModel.Config(vocab_size=32)
    _tiny_global(config.transformer, layers=2)
    return config


def _segment(
    decisions: int,
    *,
    frames: int,
    starts: bool,
    terminal: bool,
    seed: int,
) -> Segment:
    """Return a segment of valid random tokens."""
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
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = terminal
    return Segment(
        cells=cells.to(torch.uint8),
        aux=aux.to(torch.int16),
        actions=torch.randint(0, 43, (decisions,), generator=generator).to(torch.uint8),
        reward=torch.randint(-1, 4, (decisions,), generator=generator).to(torch.int16),
        done=done,
        starts_episode=starts,
    )


# ``windows=1`` keeps the first alone, for tests that run a model several times.
def _batch(*, windows: int = 2) -> PackedBatch:
    """Two windows: a terminal episode then a new one; mid-episode then a cut start."""
    first = [
        _segment(3, frames=3, starts=True, terminal=True, seed=1),
        _segment(4, frames=5, starts=True, terminal=False, seed=2),
    ]
    second = [
        _segment(5, frames=6, starts=False, terminal=False, seed=3),
        _segment(2, frames=2, starts=True, terminal=True, seed=4),
    ]
    return pack_windows([first, second][:windows], t_g=14, s_max=4)


def test_language_model_is_causal_and_segment_isolated() -> None:
    torch.manual_seed(0)
    model = _tiny_language_model_config().make().eval()
    # One packed row holding both segments; ``cu_seqlens`` cuts it at 4.
    tokens = torch.randint(0, 32, (1, 9))
    cu_seqlens = torch.tensor([0, 4, 9], dtype=torch.int32)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4]])
    with torch.no_grad():
        base = model(tokens, positions=positions, cu_seqlens=cu_seqlens)
        later = tokens.clone()
        later[0, 6] = (later[0, 6] + 1) % 32
        moved = model(later, positions=positions, cu_seqlens=cu_seqlens)
        earlier = tokens.clone()
        earlier[0, 1] = (earlier[0, 1] + 1) % 32
        crossed = model(earlier, positions=positions, cu_seqlens=cu_seqlens)
        alone = model(
            tokens[:, 4:],
            positions=positions[:, 4:],
            cu_seqlens=row_cu_seqlens(1, 5, device=tokens.device),
        )
    torch.testing.assert_close(moved[0, :6], base[0, :6])
    assert not torch.allclose(moved[0, 6], base[0, 6])
    torch.testing.assert_close(crossed[0, 4:], base[0, 4:])
    torch.testing.assert_close(alone[0], base[0, 4:])


# A forward and backward pass over both windows, every gradient checked:
# 0.10 s warm on x86.
@pytest.mark.compute_training
def test_world_model_forward_backward_counts_and_finite_losses() -> None:
    model = _tiny_world_model()
    batch = _batch()
    loss = model(batch)
    act_jobs = int((~batch.job_is_start).sum())
    framed_jobs = int((batch.job_next >= 0).sum())
    assert float(loss.count["action"]) == 12
    assert float(loss.count["reward"]) == act_jobs == float(loss.count["done"])
    assert float(loss.count["board"]) == 99 * framed_jobs
    assert float(loss.count["hud"]) == 51 * framed_jobs
    assert (act_jobs, framed_jobs) == (12, 14)
    assert bool(loss.loss.isfinite())
    assert float(loss.z_loss.detach()) > 0
    loss.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.isfinite().all()), name


def test_target_terms_are_the_terms_the_loss_sums() -> None:
    model = _tiny_world_model()
    batch = _batch()
    with torch.no_grad():
        logits = model.logits(batch)
        loss = model.loss(batch, logits)
        terms = model.target_terms(batch, logits)
    jobs = len(batch.job_at)
    shapes = {
        "action": batch.kind.shape,
        "reward": (jobs,),
        "done": (jobs,),
        "board": (jobs, 99),
        "hud": (jobs, 51),
    }
    assert terms.keys() == shapes.keys() == loss.nll.keys()
    for name, (value, scored) in terms.items():
        assert value.nll.shape == value.logz_sq.shape == scored.shape == shapes[name]
        torch.testing.assert_close(value.nll[scored].sum(), loss.nll[name])
        torch.testing.assert_close(value.logz_sq[scored].sum(), loss.logz_sq[name])
        assert float(scored.sum()) == float(loss.count[name]), name


def test_action_term_scores_each_obs_position_by_the_next_action() -> None:
    model = _tiny_world_model()
    batch = _batch()
    uniform = torch.zeros(*batch.kind.shape, 43)
    spiked = uniform.scatter(-1, batch.action.roll(-1, dims=-1)[..., None].long(), 50.0)
    local = _oracle_local(batch, frames=batch.job_next)
    value, scored = model.target_terms(
        batch,
        WorldModelLogits(action=spiked, local=local),
    )["action"]
    kind = batch.kind
    expected = torch.zeros_like(scored)
    expected[:, :-1] = (kind[:, :-1] == Kind.OBS) & (kind[:, 1:] == Kind.ACT)
    assert torch.equal(scored, expected)
    assert float(value.nll[scored].abs().max()) == 0


def test_frame_slots_sum_cell_fields_then_append_scalars() -> None:
    model = _tiny_world_model()
    batch = _batch()
    slots = model.frame_slots(batch.cells, batch.aux)
    schema = craftax_schema()
    assert slots.shape == (len(batch.cells), schema.frame_slots, 16)
    weight = model.table.weight
    offsets = torch.tensor([field.offset for field in schema.cell_fields])
    frame, cell, scalar = 2, 5, 3
    torch.testing.assert_close(
        slots[frame, cell],
        weight[batch.cells[frame, cell].long() + offsets].sum(0),
    )
    torch.testing.assert_close(
        slots[frame, schema.cell_slots + scalar],
        weight[int(batch.aux[frame, scalar]) + number_id(0)],
    )


def test_multi_hot_board_embeds_frames_like_the_summed_gathers() -> None:
    batch = _batch()
    gathered = _tiny_world_model()
    config = _tiny_world_model_config()
    config.embed_board = multi_hot_board
    multi_hot = config.make()
    # Integer-valued rows make both sums exact, whatever order either adds in.
    rows = torch.arange(gathered.table.weight.numel()).view_as(gathered.table.weight)
    slots: list[Tensor] = []
    gathers: list[int] = []
    for model in (gathered, multi_hot):
        with torch.no_grad():
            model.table.weight.copy_(rows % 7)
        with _OpCounter() as ops:
            slots.append(model.frame_slots(batch.cells, batch.aux))
        gathers.append(ops.counts["aten.embedding.default"])
    torch.testing.assert_close(slots[1], slots[0], rtol=0, atol=0)
    # The scalars stay one gather; the board's 8 fields become one matmul.
    assert gathers == [2, 1]


# Two models' forward and backward passes, every gradient compared: 0.20 s
# warm on x86.
@pytest.mark.compute_training
def test_multi_hot_board_keeps_the_loss_and_gradients() -> None:
    batch = _batch()
    config = _tiny_world_model_config()
    config.embed_board = multi_hot_board
    torch.manual_seed(0)
    loss_multi_hot, grads_multi_hot = _loss_and_grads(config.make(), batch)
    loss, grads = _loss_and_grads(_tiny_world_model(), batch)
    torch.testing.assert_close(loss_multi_hot, loss)
    assert grads_multi_hot.keys() == grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grads_multi_hot[name], grad, msg=name)


def test_untied_output_scores_local_slots_against_its_own_table() -> None:
    batch = _batch()
    config = _tiny_world_model_config()
    config.output_table = Embedding.Config()
    torch.manual_seed(0)
    model = config.make()
    assert model.output_table is not None
    with torch.no_grad():
        before = model.logits(batch)
        model.output_table.weight.zero_()
        after = model.logits(batch)
    # The input table still embeds frames: only the local logits read the new one.
    torch.testing.assert_close(after.action, before.action, rtol=0, atol=0)
    assert bool((before.local != 0).any())
    assert bool((after.local == 0).all())


def test_a_tied_model_keeps_its_state_dict_keys() -> None:
    model = _tiny_world_model()
    assert model.output_table is None
    assert not any(key.startswith("output_table") for key in model.state_dict())


def test_the_global_rotary_spans_the_head_unless_given_its_own_width() -> None:
    widths: list[object] = []
    for rotary in (-1, 2):
        config = _tiny_language_model_config().transformer
        block = config.block
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.attn, VarlenAttention.Config)
        assert isinstance(block.attn.rope, RoPE.Config)
        block.attn.rope.channels_head = rotary
        config.finalize()
        widths.append(block.attn.rope.channels_head)
    # Heads are 4 wide; a partial rotary of 2 is the caller's to keep.
    assert widths == [4, 2]


def test_global_max_attention_logit_is_the_max_over_blocks() -> None:
    torch.manual_seed(0)
    config = _tiny_language_model_config()
    config.transformer.num_layers = 3
    model = config.make()
    blocks = model.transformer.blocks
    # The middle block dominates, so neither the first nor the last block suffices.
    with torch.no_grad():
        middle = blocks[1]
        assert isinstance(middle, TransformerBlock)
        assert isinstance(middle.attn, VarlenAttention)
        norm_q = middle.attn.norm_q
        assert isinstance(norm_q, RMSNorm)
        assert norm_q.weight is not None
        norm_q.weight.mul_(5)
        model(
            torch.randint(0, 32, (2, 9)),
            positions=torch.arange(9).expand(2, 9),
            cu_seqlens=row_cu_seqlens(2, 9, device=torch.device("cpu")),
        )
    per_block = [
        attn.max_logit
        for attn in model.modules()
        if isinstance(attn, VarlenAttention) and attn.max_logit is not None
    ]
    assert len(per_block) == 3
    largest = model.transformer.max_attention_logit()
    assert largest.shape == ()
    torch.testing.assert_close(largest, per_block[1])
    assert float(per_block[1]) > max(float(per_block[0]), float(per_block[2]))


def test_world_model_later_frame_never_changes_earlier_logits() -> None:
    model = _tiny_world_model().eval()
    batch = _batch()
    job = 1
    frame = int(batch.job_next[job])
    obs_of_frame = int((batch.frame_of[0] == frame).nonzero()[0, 0])
    cells = batch.cells.clone()
    cells[frame, 5, 0] = (cells[frame, 5, 0] + 1) % 37
    changed = _replace(batch, cells=cells)
    with torch.no_grad():
        base = model.logits(batch)
        moved = model.logits(changed)
    # Local slot 7 is cell 5; its tokens are first an INPUT at position 8.
    torch.testing.assert_close(moved.local[: job + 1, :8], base.local[: job + 1, :8])
    assert not torch.allclose(moved.local[job, 8:], base.local[job, 8:])
    torch.testing.assert_close(
        moved.action[0, :obs_of_frame],
        base.action[0, :obs_of_frame],
    )
    assert not torch.allclose(
        moved.action[0, obs_of_frame],
        base.action[0, obs_of_frame],
    )


def test_world_model_later_action_never_changes_earlier_logits() -> None:
    model = _tiny_world_model().eval()
    batch = _batch()
    act = int((batch.kind[0] == Kind.ACT).nonzero()[1, 0])
    action = batch.action.clone()
    action[0, act] = (action[0, act] + 1) % 43
    with torch.no_grad():
        base = model.logits(batch)
        moved = model.logits(_replace(batch, action=action))
    earlier_jobs = int((batch.job_at < act).sum())
    torch.testing.assert_close(moved.local[:earlier_jobs], base.local[:earlier_jobs])
    torch.testing.assert_close(moved.action[0, :act], base.action[0, :act])
    assert not torch.allclose(moved.local[earlier_jobs], base.local[earlier_jobs])


# The design's own sizes are the subject: 312M parameters built on the meta
# device, 0.43 s warm on x86.
@pytest.mark.compute_large_fixture
def test_plan_sizes_match_the_plan_parameter_counts() -> None:
    with torch.device("meta"):
        model = WorldModel.Config().make()
    assert isinstance(model.encoder, FrameEncoder)
    counts = {
        "global": _blocks(model.transformer),
        "encoder": _blocks(model.encoder.stack),
        "decoder": _blocks(model.decoder.stack),
    }
    assert abs(counts["global"] / 283e6 - 1) < 0.005, counts
    assert abs(counts["encoder"] / 12.4e6 - 1) < 0.01, counts
    assert abs(counts["decoder"] / 16.6e6 - 1) < 0.01, counts


def test_decoder_block_cross_attends_to_memory() -> None:
    config = DecoderBlock.Config(channels_in=8)
    assert isinstance(config.cross_attn, Attention.Config)
    block = config.make()
    x = torch.randn(2, 3, 8)
    memory = torch.randn(2, 5, 8)
    out = block(x, memory=memory)
    assert not torch.allclose(block(x, memory=memory + 1), out)


# Eager CPU RMSNorm of a bfloat16 input under a float32 scale takes torch's
# unfused path and says so; compiled training decomposes the norm either way.
@pytest.mark.filterwarnings("ignore:Mismatch dtype between input and weight")
@pytest.mark.parametrize("cast", [False, True])
def test_local_blocks_run_on_a_residual_stream_of_the_autocast_dtype(
    cast: bool,
) -> None:
    model = _local_stream_model(cast=cast)
    seen = _record_block_dtypes(model)
    # Autocast leaves an embedding's float32 output alone, so without the cast
    # every local residual stream runs in float32.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(_batch())
    assert torch.isfinite(loss.loss)
    # The encoder block's input, then the decoder block's input and memory.
    assert seen == [torch.bfloat16 if cast else torch.float32] * 3


def test_an_autocast_stream_stays_float32_without_autocast() -> None:
    # Inference loads a trained model and runs without autocast.
    model = _local_stream_model(cast=True)
    seen = _record_block_dtypes(model)
    with torch.no_grad():
        loss = model(_batch())
    assert torch.isfinite(loss.loss)
    assert seen == [torch.float32] * 3


@pytest.mark.parametrize("grad", [True, False])
def test_decoder_block_checkpoints_only_when_a_backward_can_run(grad: bool) -> None:
    block = DecoderBlock.Config(channels_in=8, checkpoint=True).make()
    spy = Mock(wraps=checkpoint.checkpoint)
    with (
        patch(
            "priml.baselines.craftax.world_model.model.checkpoint",
            spy,
        ),
        torch.set_grad_enabled(grad),
    ):
        block(torch.randn(2, 3, 8), memory=torch.randn(2, 5, 8))
    assert spy.call_count == int(grad)


# Two models' forward and backward passes, one recomputing every local block:
# 0.38 s warm on x86.
@pytest.mark.compute_training
def test_checkpointed_local_blocks_recompute_with_equal_loss_and_gradients() -> None:
    batch = _batch()
    stored = _two_layer_local_model(checkpoint=False)
    recomputed = _two_layer_local_model(checkpoint=True)
    stored_calls = _count_local_calls(stored)
    recomputed_calls = _count_local_calls(recomputed)
    loss, grads = _loss_and_grads(stored, batch)
    loss_recomputed, grads_recomputed = _loss_and_grads(recomputed, batch)
    # One call per block in forward; a checkpointed block runs again in backward.
    assert stored_calls == [1] * 4
    assert recomputed_calls == [2] * 4
    torch.testing.assert_close(loss_recomputed, loss, rtol=0, atol=0)
    assert grads_recomputed.keys() == grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grads_recomputed[name], grad, rtol=0, atol=0)


# Two recomputing models' forward and backward passes, every aten op counted:
# 0.43 s warm on x86.
@pytest.mark.compute_training
def test_recompute_keeping_attention_reruns_all_but_the_attention_kernels() -> None:
    batch = _batch()
    plain = _two_layer_local_model(checkpoint=True)
    kept = _two_layer_local_model(checkpoint=True, keeps_attention=True)
    calls, calls_kept = _count_local_calls(plain), _count_local_calls(kept)
    with _OpCounter() as ops:
        loss, grads = _loss_and_grads(plain, batch)
    with _OpCounter() as ops_kept:
        loss_kept, grads_kept = _loss_and_grads(kept, batch)
    # Both re-run every block's feed-forward in backward...
    assert calls == calls_kept == [2] * 4
    # ...but a kept block's attention kernels run once: the encoder's
    # self-attention and the decoder's self- and cross-attention, in 2 blocks each.
    kernel = "aten._scaled_dot_product_flash_attention_for_cpu.default"
    assert ops.counts[kernel] - ops_kept.counts[kernel] == 2 * 1 + 2 * 2
    torch.testing.assert_close(loss_kept, loss, rtol=0, atol=0)
    assert grads_kept.keys() == grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grads_kept[name], grad, rtol=0, atol=0, msg=name)


@pytest.mark.compute_torch_compile
@pytest.mark.parametrize("keeps_attention", [False, True])
def test_checkpointed_local_blocks_compile_fullgraph_to_the_eager_gradients(
    keeps_attention: bool,
) -> None:
    batch = _batch()
    stored = _two_layer_local_model(checkpoint=False)
    recomputed = _two_layer_local_model(
        checkpoint=True,
        keeps_attention=keeps_attention,
    )
    torch.compiler.reset()
    try:
        compiled = torch.compile(recomputed, fullgraph=True, backend="aot_eager")
        loss_compiled, grads_compiled = _loss_and_grads(compiled, batch)
    finally:
        torch.compiler.reset()
    loss, grads = _loss_and_grads(stored, batch)
    torch.testing.assert_close(loss_compiled, loss, rtol=1e-6, atol=1e-6)
    for name, grad in grads.items():
        torch.testing.assert_close(
            grads_compiled[name],
            grad,
            rtol=1e-5,
            atol=1e-7,
            msg=name,
        )


@pytest.mark.parametrize(
    "build",
    [
        # The world model's init on the meta device dispatches each of its
        # draws through the device context: built twice, 0.10 s warm on x86.
        pytest.param(_tiny_world_model_config, marks=pytest.mark.compute_large_fixture),
        _tiny_language_model_config,
    ],
)
def test_meta_materialization_matches_eager_reset_bit_for_bit(
    build: Callable[[], WorldModel.Config | GlobalLanguageModel.Config],
) -> None:
    with torch.device("meta"):
        meta = build().make()
    torch.manual_seed(1)
    materialize_meta(meta, torch.device("cpu"))
    eager = build().make()
    torch.manual_seed(1)
    eager.reset_parameters()
    expected = dict(named_meta_state(eager))
    for name, tensor in named_meta_state(meta):
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize(
    "build",
    [_tiny_world_model_config, _tiny_language_model_config],
)
def test_meta_materialization_draws_the_plan_init(
    build: Callable[[], WorldModel.Config | GlobalLanguageModel.Config],
) -> None:
    with torch.device("meta"):
        model = build().make()
    torch.manual_seed(0)
    materialize_meta(model, torch.device("cpu"))
    norms = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, RMSNorm)
    }
    assert len(norms) > 4
    for name, parameter in model.named_parameters():
        if name in norms:
            assert bool((parameter == 1).all()), name
        else:
            tolerance = 0.2 if parameter.numel() >= 256 else 0.5
            assert abs(float(parameter.detach().std()) / 0.02 - 1) < tolerance, name


def test_action_is_scored_from_the_obs_position_never_the_act_position() -> None:
    model = _tiny_world_model()
    batch = _batch()
    local = _oracle_local(batch, frames=batch.job_next)
    uniform = torch.zeros(*batch.kind.shape, 43)
    # At each obs position, the spike sits on the action taken right after it.
    obs = uniform.clone()
    obs[:, :-1] = uniform[:, :-1].scatter(-1, batch.action[:, 1:, None].long(), 50.0)
    obs *= (batch.kind == Kind.OBS)[..., None]
    # At each act position, the spike sits on that position's own (input) action.
    act = uniform.scatter(-1, batch.action[..., None].long(), 50.0)
    act *= (batch.kind == Kind.ACT)[..., None]
    read_obs = model.loss(batch, WorldModelLogits(action=obs, local=local))
    read_act = model.loss(batch, WorldModelLogits(action=act, local=local))
    count = read_obs.count["action"]
    assert float(count) == 12
    assert float(read_obs.nll["action"]) == 0
    torch.testing.assert_close(read_act.nll["action"], count * math.log(43))


def test_each_job_is_scored_against_its_next_frame() -> None:
    model = _tiny_world_model()
    batch = _batch()
    action = torch.zeros(*batch.kind.shape, 43)
    # A terminal job has no next frame; its oracle names its current frame, which
    # scores nonzero only if the model wrongly scores a terminal job's frame.
    frames = torch.where(batch.job_next >= 0, batch.job_next, batch.job_memory)
    assert bool(((batch.job_next < 0) & (batch.job_memory > 0)).any())
    following = _oracle_local(batch, frames=frames)
    current = _oracle_local(batch, frames=batch.job_memory)
    right = model.loss(batch, WorldModelLogits(action=action, local=following))
    wrong = model.loss(batch, WorldModelLogits(action=action, local=current))
    for name in ("reward", "done", "board", "hud"):
        assert float(right.nll[name]) == 0, name
    assert float(wrong.nll["board"]) > 100
    assert float(wrong.nll["hud"]) > 100


def test_reward_and_done_are_scored_on_act_jobs_only() -> None:
    model = _tiny_world_model()
    batch = _batch()
    action = torch.zeros(*batch.kind.shape, 43)
    local = _oracle_local(batch, frames=batch.job_next)
    start = batch.job_is_start
    assert bool(start.any())
    assert bool((~start).any())
    # Rolling a prefix slot moves its spike off the target, onto another allowed ID.
    starts_wrong, acts_wrong = local.clone(), local.clone()
    starts_wrong[start, :2] = local[start, :2].roll(1, dims=-1)
    acts_wrong[~start, :2] = local[~start, :2].roll(1, dims=-1)
    ignored = model.loss(batch, WorldModelLogits(action=action, local=starts_wrong))
    scored = model.loss(batch, WorldModelLogits(action=action, local=acts_wrong))
    for name in ("reward", "done"):
        assert float(ignored.nll[name]) == 0, name
        assert float(scored.nll[name]) > 100, name
        assert float(scored.count[name]) == int((~start).sum()), name


def test_start_jobs_read_the_null_memory_not_a_frame() -> None:
    model = _tiny_world_model().eval()
    batch = _batch()
    start = batch.job_is_start
    # Reversing the frame order changes every frame, including frame 0, which a
    # null job's clamped index would otherwise gather.
    changed = _replace(batch, cells=batch.cells.flip(0), aux=batch.aux.flip(0))
    with torch.no_grad():
        base = model.logits(batch)
        moved = model.logits(changed)
    # Positions 0-2 read only c, reward, and done; later ones read next-frame slots.
    torch.testing.assert_close(moved.local[start, :3], base.local[start, :3])
    assert not torch.allclose(moved.local[~start, :3], base.local[~start, :3])


def test_start_jobs_are_fed_reward_zero_and_done_false() -> None:
    model = _tiny_world_model().eval()
    batch = _batch(windows=1)
    start = batch.job_is_start
    assert bool((batch.job_reward[start] == 0).all())
    assert not bool(batch.job_done[start].any())
    reward, done = batch.job_reward.clone(), batch.job_done.clone()
    reward[start], done[start] = 3, True
    with torch.no_grad():
        base = model.logits(batch)
        ignored = model.logits(_replace(batch, job_reward=reward, job_done=done))
        # The same jobs flagged as act jobs read their stored reward 0, done false.
        unflagged = model.logits(_replace(batch, job_is_start=torch.zeros_like(start)))
        reward[~start] += 1
        live = model.logits(_replace(batch, job_reward=reward))
    torch.testing.assert_close(ignored.local, base.local)
    torch.testing.assert_close(unflagged.local, base.local)
    assert not torch.allclose(live.local[~start, 2:], base.local[~start, 2:])


def test_other_segments_never_change_a_segments_logits() -> None:
    model = _tiny_world_model().eval()
    batch = _batch()
    mine = batch.segment == 1
    mine[1] = False
    jobs = mine.flatten()[batch.job_at.long()]
    frames = torch.cat(
        [batch.frame_of[mine], batch.job_next[jobs], batch.job_memory[jobs]],
    )
    others = torch.ones(len(batch.cells), dtype=torch.bool)
    others[frames[frames >= 0].long()] = False
    cells, aux = batch.cells.clone(), batch.aux.clone()
    cells[others], aux[others] = cells[others].flip(0), aux[others].flip(0)
    is_act = batch.kind == Kind.ACT
    action = torch.where(mine | ~is_act, batch.action, (batch.action + 1) % 43)
    reward = torch.where(jobs, batch.job_reward, batch.job_reward + 1)
    changed = _replace(batch, cells=cells, aux=aux, action=action, job_reward=reward)
    with torch.no_grad():
        base = model.logits(batch)
        moved = model.logits(changed)
    torch.testing.assert_close(moved.action[mine], base.action[mine])
    torch.testing.assert_close(moved.local[jobs], base.local[jobs])
    assert not torch.allclose(moved.action[~mine], base.action[~mine])
    assert not torch.allclose(moved.local[~jobs], base.local[~jobs])


def test_packed_segment_scores_like_the_segment_alone() -> None:
    model = _tiny_world_model().eval()
    packed = _batch()
    segment = _segment(4, frames=5, starts=True, terminal=False, seed=2)
    alone = pack_windows([[segment]], t_g=7, s_max=1)
    mine = packed.segment == 1
    mine[1] = False
    jobs = mine.flatten()[packed.job_at.long()]
    with torch.no_grad():
        full = model.logits(packed)
        expected = model.logits(alone)
    sliced = WorldModelLogits(action=full.action[mine][None], local=full.local[jobs])
    torch.testing.assert_close(sliced.action, expected.action)
    torch.testing.assert_close(sliced.local, expected.local)
    torch.testing.assert_close(
        model.loss(alone, sliced).loss,
        model.loss(alone, expected).loss,
    )


def test_global_positions_come_from_batch_pos() -> None:
    model = _tiny_world_model().eval()
    batch = _batch()
    with torch.no_grad():
        base = model.logits(batch)
        stretched = model.logits(_replace(batch, pos=batch.pos * 2))
    assert not torch.allclose(stretched.action, base.action)
    assert not torch.allclose(stretched.local, base.local)


# Every cost test below runs its attention through explicit products: the CPU's
# fused SDPA dispatches none that torch's FLOP counter sees, so a measurement of
# the default kernels would miss exactly the attention the cost prices.
def test_a_decoder_block_costs_its_self_and_cross_attention() -> None:
    config = DecoderBlock.Config(channels_in=16)
    assert isinstance(config.ffn, SwiGLU.Config)
    config.ffn.channels_hidden = 32
    naive_attention(config)
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 3, 16, requires_grad=True),
            torch.randn(2, 5, 16, requires_grad=True),
        ),
        run=_decoded,
        seq_len=3,
        batch_size=2,
        memory_len=5,
        dtype=None,
    )


def test_the_frame_encoder_costs_the_stack_over_its_slots_and_pool() -> None:
    config = FrameEncoder.Config(channels_in=16, num_slots=3)
    _tiny_local(config.stack)
    naive_attention(config.stack.block)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 16, requires_grad=True),
        run=_encoded,
        batch_size=2,
        dtype=None,
    )


def test_the_local_decoder_costs_its_stack_against_the_memory() -> None:
    config = LocalDecoder.Config(channels_in=16, num_slots=3)
    _tiny_local(config.stack)
    naive_attention(config.stack.block)
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 3, 16, requires_grad=True),
            torch.randn(2, 5, 16, requires_grad=True),
        ),
        run=_decoded_jobs,
        batch_size=2,
        memory_len=5,
        dtype=None,
    )


def test_the_language_model_costs_its_stack_and_a_tied_head() -> None:
    config = _tiny_language_model_config()
    naive_attention(config.transformer.block)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 32, (2, 5)),
        run=_language_logits,
        seq_len=5,
        batch_size=2,
        dtype=None,
    )


@pytest.mark.parametrize(
    ("board", "untied"),
    [(gathered_board, False), (multi_hot_board, True)],
    ids=["gathered_tied", "multi_hot_untied"],
)
def test_the_world_model_costs_the_products_torch_runs(
    board: BoardEmbedding,
    untied: bool,
) -> None:
    config = _tiny_world_model_config()
    config.embed_board = board
    if untied:
        config.output_table = Embedding.Config()
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack, config.transformer):
        naive_attention(stack.block)
    # One short episode: the cost counts positions, so a bigger batch adds
    # only time.
    segment = _segment(2, frames=2, starts=True, terminal=True, seed=1)
    batch = pack_windows([[segment]], t_g=14, s_max=4)
    assert_cost_matches_torch(
        config,
        build_input=lambda: batch.kind,
        run=partial(_packed_loss, batch=batch),
        seq_len=batch.kind.shape[-1],
        batch_size=len(batch.kind),
        frames=len(batch.aux),
        jobs=len(batch.job_at),
        dtype=None,
    )


def test_the_board_embeddings_cost_a_lookup_or_a_one_sided_product() -> None:
    gathered = cost(gathered_board, rows=6, fields=4, vocab=7, width=5, dtype=None)
    product = cost(multi_hot_board, rows=6, fields=4, vocab=7, width=5, dtype=None)
    assert gathered["flops", "matmul"].sum() == 0
    # Forward, then the table's gradient alone: the multi-hot rows take none.
    assert product["flops", "matmul"].sum() == 2 * (2 * 6 * 7 * 5)
    assert gathered.params == product.params == 0


def _oracle_local(batch: PackedBatch, *, frames: Tensor) -> Tensor:
    """Return local logits spiking on each job's prefix and on ``frames``' tokens."""
    schema = craftax_schema()
    prefix = len(schema.prefix_names)
    board_end = prefix + schema.cell_slots
    jobs = torch.arange(len(frames))
    logits = torch.zeros(len(frames), schema.local_slots, schema.vocab_size)
    logits[jobs, 0, batch.job_reward.long() + number_id(0)] = 50.0
    logits[jobs, 1, batch.job_done.long() + done_id(done=False)] = 50.0
    frame = frames.clamp(min=0).long()
    offsets = torch.tensor([field.offset for field in schema.cell_fields])
    board = batch.cells[frame].long() + offsets
    logits[:, prefix:board_end] = logits[:, prefix:board_end].scatter(-1, board, 50.0)
    hud = batch.aux[frame, :, None].long() + number_id(0)
    logits[:, board_end:] = logits[:, board_end:].scatter(-1, hud, 50.0)
    return logits


def _two_layer_local_model(
    *,
    checkpoint: bool,
    keeps_attention: bool = False,
) -> WorldModel:
    """Return a tiny model with 2-block local stacks, all checkpointed or none."""
    config = _tiny_world_model_config()
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack):
        stack.num_layers = 2
        assert isinstance(stack.block, (EncoderBlock.Config, DecoderBlock.Config))
        stack.block.checkpoint = checkpoint
        stack.block.recompute_policy = keep_attention if keeps_attention else None
    torch.manual_seed(0)
    return config.make()


class _OpCounter(TorchDispatchMode):
    """Count the aten ops that run, by name; a recompute's replayed outputs never do."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()

    @override
    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type[object], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


# A pre-hook, because a recompute stops once it has every tensor backward needs, inside
# the feed-forward, before a forward hook would fire.
def _count_local_calls(model: WorldModel) -> list[int]:
    """Return per-local-block counts of feed-forward calls, updated as they happen."""
    calls: list[int] = []
    assert isinstance(model.encoder, FrameEncoder)
    for stack in (model.encoder.stack, model.decoder.stack):
        for block in stack.blocks:
            assert isinstance(block, TransformerBlock | DecoderBlock)
            assert isinstance(block.ffn, nn.Module)
            calls.append(0)
            block.ffn.register_forward_pre_hook(
                partial(_count_call, calls, len(calls) - 1),
            )
    return calls


def _count_call(calls: list[int], index: int, *hook_args: object) -> None:
    """Forward pre-hook: add one to ``calls[index]``."""
    del hook_args
    calls[index] += 1


def _local_stream_model(*, cast: bool) -> WorldModel:
    """Return the tiny world model, its local streams cast under autocast or not."""
    config = _tiny_world_model_config()
    assert isinstance(config.encoder, FrameEncoder.Config)
    stream = to_autocast_dtype if cast else None
    config.encoder.cast_stream = config.decoder.cast_stream = stream
    torch.manual_seed(0)
    return config.make()


def _record_block_dtypes(model: WorldModel) -> list[torch.dtype]:
    """Return a list the local blocks append their inputs' dtypes to."""
    seen: list[torch.dtype] = []
    assert isinstance(model.encoder, FrameEncoder)
    for stack in (model.encoder.stack, model.decoder.stack):
        for block in stack.blocks:
            block.register_forward_pre_hook(
                partial(_record_dtypes, seen),
                with_kwargs=True,
            )
    return seen


def _record_dtypes(
    seen: list[torch.dtype],
    module: nn.Module,
    args: tuple[Tensor, ...],
    kwargs: dict[str, object],
) -> None:
    """Forward pre-hook: record the block input's dtype and its memory's, if any."""
    del module
    seen.extend(
        value.dtype
        for value in (*args, kwargs.get("memory"))
        if isinstance(value, Tensor)
    )


def _loss_and_grads(
    model: Callable[[PackedBatch], object],
    batch: PackedBatch,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the loss of ``batch`` and every parameter's gradient after its backward."""
    loss = model(batch)
    assert isinstance(loss, ModalityLoss)
    loss.loss.backward()
    assert isinstance(model, nn.Module)
    grads: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        grads[name.removeprefix("_orig_mod.")] = parameter.grad
    return loss.loss.detach(), grads


def _blocks(stack: torch.nn.Module) -> int:
    """Return the parameter count of a stack's blocks."""
    assert isinstance(stack, Transformer)
    return sum(p.numel() for p in stack.blocks.parameters())


def _replace(batch: PackedBatch, **tensors: Tensor) -> PackedBatch:
    """Return ``batch`` with some tensors swapped."""
    fields = {name: getattr(batch, name) for name in PackedBatch.__dataclass_fields__}
    return PackedBatch(**(fields | tensors))


def _decoded(model: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Run a decoder block on ``(x, memory)``."""
    assert isinstance(model, DecoderBlock)
    x, memory = inputs
    return model(x, memory=memory)


def _encoded(model: nn.Module, slots: Tensor) -> Tensor:
    """Sum both of a frame encoder's outputs, so backward reaches every product."""
    assert isinstance(model, FrameEncoder)
    pooled, memory = model(slots)
    return pooled.sum() + memory.sum()


def _decoded_jobs(model: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Run a local decoder on ``(inputs, memory)``, the second job reading the null."""
    assert isinstance(model, LocalDecoder)
    x, memory = inputs
    return model(x, memory=memory, has_memory=torch.tensor([True, False]))


def _language_logits(model: nn.Module, tokens: Tensor) -> Tensor:
    """Run the language model on one segment per row."""
    assert isinstance(model, GlobalLanguageModel)
    rows, length = tokens.shape
    positions = torch.arange(length).expand(rows, length)
    return model(
        tokens,
        positions=positions,
        cu_seqlens=row_cu_seqlens(rows, length, device=tokens.device),
    )


def _packed_loss(model: nn.Module, inputs: Tensor, *, batch: PackedBatch) -> Tensor:
    """Return ``model``'s loss on ``batch``; ``inputs`` only stands in for it."""
    del inputs
    assert isinstance(model, WorldModel)
    return model(batch).loss


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
