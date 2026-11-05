"""Tests for exp003's fused fp4 attention kernel and its packed cache."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Final, Protocol, cast

import inspect
import math

import pytest
import torch

from priml.baselines.kvcache.decode import _cache_bytes
from priml.baselines.kvcache.decode_test import _tiny
from priml.baselines.kvcache.experiments import exp003
from priml.baselines.kvcache.fp4 import (
    _CODE_VALUES,
    SCALE_COLUMNS,
    _dequantized,
    _packed,
    _quantize,
    fp4_layer_cache,
)
from priml.baselines.kvcache.fp4_fused import (
    FP4PackedKVCache,
    Fp4TritonAttention,
    _compiled_fp4_attention,
    _fp4_attention_forward_triton,
    fp4_fused_layer_cache,
)
from priml.model.attention.attention import Attention
from priml.model.attention.gated_attention import GatedAttention
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.qwen3_5_delta import Qwen35GatedDeltaNet
from priml.model.attention.rope import RoPE
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.testing.triton_interpreter import triton_interpreter
else:
    from wrapt import lazy_import

    triton_interpreter = lazy_import(
        "priml.testing.triton_interpreter",
        "triton_interpreter",
    )


_NUM_HEADS: Final = 3
_CHANNELS_HEAD: Final = 4


def test_packed_cache_serves_uint8_rows_in_the_incoming_dtype() -> None:
    cache = FP4PackedKVCache.alloc(
        batch=(2,),
        num_heads=_NUM_HEADS,
        max_seq=5,
        channels_head=_CHANNELS_HEAD,
        channels_v_head=_CHANNELS_HEAD,
        device=torch.device("cpu"),
    )
    k = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD)
    v = torch.randn(2, _NUM_HEADS, 5, _CHANNELS_HEAD)

    k_all, v_all = cache.update(k, v)

    width = _packed(_CHANNELS_HEAD) + SCALE_COLUMNS
    assert k_all.dtype == torch.uint8
    assert v_all.dtype == torch.uint8
    assert k_all.shape == (2, _NUM_HEADS, 5, width)
    # The served rows are the storage itself, not a fresh dequantized copy.
    assert k_all.data_ptr() == cache.k[..., :5, :].data_ptr()


def test_fused_allocator_counts_the_same_bytes_as_the_fp4_allocator() -> None:
    model = _tiny().make().model
    max_seq = 64
    reference = fp4_layer_cache(model, batch=2, max_seq=max_seq, dtype=torch.float32)
    fused = fp4_fused_layer_cache(model, batch=2, max_seq=max_seq, dtype=torch.float32)
    assert set(fused) == set(reference)
    for state in fused.values():
        assert isinstance(state, FP4PackedKVCache)
    assert _cache_bytes(fused) == _cache_bytes(reference)


def test_forward_refuses_features_the_decode_ladder_never_asks_for() -> None:
    attention = Fp4TritonAttention.Config().make()
    q = torch.zeros(2, 5, 3, 16, dtype=torch.bfloat16)
    packed = torch.zeros(2, 7, 3, 12, dtype=torch.uint8)
    with pytest.raises(ValueError, match="no dropout, mask, or window"):
        attention(q, packed, packed, dropout_p=0.1)
    with pytest.raises(ValueError, match="bf16"):
        attention(q.float(), packed, packed)


def test_cost_matches_the_logical_attention_it_computes() -> None:
    actual = Fp4TritonAttention.Config.cost(
        seq_len=7,
        batch_size=2,
        num_heads=3,
        channels_head=16,
        dtype=torch.bfloat16,
    )
    expected = attention_kernel_cost(
        seq_len=7,
        batch_size=2,
        num_heads=3,
        channels_head=16,
        dtype=torch.bfloat16,
    )
    assert actual == expected


@pytest.mark.parametrize("shape", [(2, 7, 3, 11), (2, 7, 4, 12), (2, 0, 3, 12)])
def test_forward_rejects_packed_geometry_before_launch(shape: tuple[int, ...]) -> None:
    attention = Fp4TritonAttention.Config().make()
    q = torch.zeros(2, 5, 3, 16, dtype=torch.bfloat16)
    packed = torch.zeros(shape, dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"packed width|line up|empty launch"):
        attention(q, packed, packed)


def test_interpreted_kernel_reads_query_strides_and_real_packed_rows() -> None:
    pytest.importorskip("triton")
    q, k, v = _cuda_case(device="cpu", seq=5, length=7)
    storage = torch.empty((*q.shape[:-1], 2 * q.shape[-1]), dtype=q.dtype)
    storage[..., ::2] = q
    storage[..., 1::2] = -7
    attention = Fp4TritonAttention.Config().make()

    with triton_interpreter(_compiled_fp4_attention):
        strided = attention(storage[..., ::2], k, v, scale=0.25)
        contiguous = attention(q, k, v, scale=0.25)

    assert torch.equal(strided, contiguous)
    assert torch.isfinite(strided).all()


@pytest.mark.parametrize("channels", [16, 64])
@pytest.mark.parametrize("block_l", [16, 128])
def test_supported_tiles_compile_for_sm80_without_a_gpu(
    channels: int,
    block_l: int,
) -> None:
    pytest.importorskip("triton")
    compiler = cast("_TritonCompiler", import_module("triton.compiler.compiler"))
    target_factory = cast("_CudaBackend", import_module("triton.backends.compiler"))
    constants = {
        "heads": 3,
        "seq": 5,
        "channels": channels,
        "block_l": block_l,
        "block_s": 16,
        "causal": True,
        "logit_scale": channels**-0.5,
        "value_limit": torch.finfo(torch.bfloat16).max,
    }
    pointer_types = {
        "q_ptr": "*bf16",
        "k_ptr": "*u8",
        "v_ptr": "*u8",
        "ks_ptr": "*fp32",
        "vs_ptr": "*fp32",
        "out_ptr": "*fp32",
        "table_ptr": "*bf16",
    }
    signature = {
        name: pointer_types.get(name, "i32")
        for name in inspect.signature(_fp4_attention_forward_triton).parameters
        if name not in constants
    }
    source = compiler.ASTSource(_compiled_fp4_attention(), signature, constants)
    compiled = compiler.compile(
        source,
        target=target_factory.GPUTarget("cuda", 80, 32),
        options={"enable_fp_fusion": True},
    )
    assert compiled is not None


@pytest.mark.parametrize("channels", [3, 6, 8, 24])
def test_forward_rejects_widths_the_dot_kernel_cannot_tile(channels: int) -> None:
    attention = Fp4TritonAttention.Config().make()
    q = torch.zeros(2, 5, 3, channels, dtype=torch.bfloat16)
    packed = torch.zeros(2, 7, 3, _packed(channels) + SCALE_COLUMNS, dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"channels.*power of two.*16"):
        attention(q, packed, packed)


@pytest.mark.parametrize("block_l", [0, 8, 24])
def test_config_rejects_key_tiles_below_the_dot_floor(block_l: int) -> None:
    cfg = Fp4TritonAttention.Config()
    cfg.block_l = block_l

    with pytest.raises(ValueError, match="block_l"):
        cfg.make()


@pytest.mark.parametrize("family", ["gated", "multistream", "delta", "latent"])
def test_fused_allocator_rejects_unsupported_attention_before_allocation(
    family: str,
) -> None:
    cfg: (
        GatedAttention.Config
        | MultiStreamAttention.Config
        | Qwen35GatedDeltaNet.Config
        | MultiHeadLatentAttention.Config
    )
    if family == "gated":
        gated = GatedAttention.Config()
        gated.num_heads = 6
        gated.num_heads_kv = 3
        gated.channels_head = 16
        cfg = gated
    elif family == "multistream":
        multistream = MultiStreamAttention.Config()
        multistream.num_heads = 6
        multistream.num_heads_kv = 3
        multistream.channels_head = 16
        cfg = multistream
    elif family == "delta":
        delta = Qwen35GatedDeltaNet.Config()
        delta.num_heads_k = 2
        delta.num_heads_v = 4
        delta.channels_k_head = 3
        delta.channels_v_head = 5
        cfg = delta
    else:
        latent = MultiHeadLatentAttention.Config()
        latent.num_heads = 3
        latent.kv_lora_rank = 4
        latent.channels_qk_nope_head = 2
        latent.channels_qk_rope_head = 6
        latent.channels_v_head = 5
        cfg = latent
    cfg.channels_in = 8
    cfg.depth_index = ((0, 1),)

    with pytest.raises(TypeError, match=r"supported.*Attention"):
        fp4_fused_layer_cache(cfg.make(), batch=2, max_seq=7)


@pytest.mark.gpu_triton
@pytest.mark.parametrize("length", [7, 127, 128, 131])
def test_uniform_attention_matches_known_values_across_tile_boundaries(
    length: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    q = torch.zeros(2, 5, 3, 16, dtype=torch.bfloat16, device="cuda")
    codes = torch.tensor(_CODE_VALUES, dtype=torch.bfloat16, device="cuda")
    exponents = (
        torch.arange(2, device="cuda")[:, None, None]
        + torch.arange(length, device="cuda")[None, :, None] % 3
        + torch.arange(3, device="cuda")[None, None, :]
    )
    values = codes * (2.0**exponents)[..., None]
    k = _quantize(torch.zeros_like(values))
    v = _quantize(values)
    expected = values.float().mean(dim=1, keepdim=True).expand_as(q).to(q.dtype)

    actual = Fp4TritonAttention.Config().make()(q, k, v)

    assert torch.equal(actual, expected), (actual, expected)


@pytest.mark.gpu_triton
@pytest.mark.parametrize("length", [7, 131])
@pytest.mark.parametrize("high_tail", [False, True])
@pytest.mark.parametrize("channels", [16, 32, 64])
def test_known_nonuniform_logits_weight_the_final_rows(
    length: int,
    high_tail: bool,
    channels: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    q = torch.zeros(2, 5, 3, channels, dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1
    keys = torch.zeros(2, length, 3, channels, dtype=torch.bfloat16, device="cuda")
    keys[..., 1] = 6
    keys[:, -3:, :, 0] = 6 if high_tail else 0
    keys[:, :-3, :, 0] = 0 if high_tail else 6
    values = torch.zeros_like(keys)
    values[:, -3:] = 1
    tail = 3 * (math.exp(6 * cast(float, channels**-0.5)) if high_tail else 1)
    prefix = (length - 3) * (
        1 if high_tail else math.exp(6 * cast(float, channels**-0.5))
    )
    expected = torch.full_like(q, tail / (prefix + tail))

    actual = Fp4TritonAttention.Config().make()(q, _quantize(keys), _quantize(values))

    assert (actual - expected).abs().max().item() < 0.02, (actual, expected)


@pytest.mark.gpu_triton
@pytest.mark.parametrize("block_l", [16, 128])
@pytest.mark.parametrize("channels", [16, 64])
@pytest.mark.parametrize(
    ("seq", "causal"),
    [(1, False), (1, True), (4, True)],
    ids=["decode", "decode-causal", "prefill"],
)
def test_kernel_matches_sdpa_over_dequantized_rows(
    seq: int,
    causal: bool,
    block_l: int,
    channels: int,
) -> None:
    # greedy_decode feeds logits[:, -1:] after prefill, so decode has one query.
    q, k, v = _cuda_case(seq=seq, channels=channels)
    config = Fp4TritonAttention.Config()
    config.block_l = block_l
    attention = config.make()

    out = attention(q, k, v, is_causal=causal)
    expected = _reference(q, k, v, causal=causal)
    key = _dequantized(k, end=k.shape[1], channels=q.shape[-1], dtype=torch.bfloat16)
    value = _dequantized(v, end=v.shape[1], channels=q.shape[-1], dtype=torch.bfloat16)
    logits = q.float().movedim(-3, -2) @ key.float().movedim(-3, -2).transpose(-1, -2)
    logits *= cast(float, q.shape[-1] ** -0.5)
    if causal:
        query_rows = torch.arange(k.shape[1] - seq, k.shape[1], device=q.device)
        keys = torch.arange(k.shape[1], device=q.device)
        logits.masked_fill_(keys[None, :] > query_rows[:, None], float("-inf"))
    manual = (
        (logits.softmax(dim=-1) @ value.float().movedim(-3, -2))
        .movedim(-3, -2)
        .to(q.dtype)
    )

    assert (expected - manual).abs().max().item() < 0.02
    assert (out - expected).abs().max().item() < 0.02


@pytest.mark.gpu_triton
def test_kernel_reads_a_strided_query_channel_axis() -> None:
    q, k, v = _cuda_case(seq=5)
    storage = torch.empty(
        (*q.shape[:-1], 2 * q.shape[-1]),
        device=q.device,
        dtype=q.dtype,
    )
    storage[..., ::2] = q
    storage[..., 1::2] = -7
    sliced = storage[..., ::2]
    attention = Fp4TritonAttention.Config().make()

    assert torch.equal(sliced, q)
    assert sliced.stride(-1) == 2
    assert torch.equal(attention(sliced, k, v), attention(q, k, v))


@pytest.mark.gpu_triton
@pytest.mark.parametrize("target", ["key", "value"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_kernel_clamps_finite_reconstructions_before_bf16_cast(
    target: str,
    sign: int,
) -> None:
    # A one-query decode with four keys makes each softmax weight exactly 1/4.
    q, k, v = _cuda_case(length=4)
    q.zero_()
    raw = torch.zeros(2, 4, 3, 16, dtype=torch.float64, device=q.device)
    raw[:, 0] = sign * 3.4e38
    extreme = _quantize(raw)
    product = _dequantized(extreme, end=4, channels=16, dtype=torch.float32)
    assert product[:, 0].abs().min() > torch.finfo(torch.bfloat16).max
    if target == "key":
        k = extreme
    else:
        v = extreme

    actual = Fp4TritonAttention.Config().make()(q, k, v)
    expected = _reference(q, k, v, causal=False)

    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@pytest.mark.gpu_triton
def test_exp003_decodes_with_the_fused_cache() -> None:
    cfg = exp003()
    cfg.device = "cuda"
    cfg.prompts = cfg.prompts[:2]
    cfg.prompt_len = 5
    cfg.num_warmup = 0
    cfg.num_repeats = 1
    cfg.num_decode_steps = 3
    cfg.model.channels_in = 96
    cfg.model.num_layers = 2
    block = cfg.model.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    attn.num_heads = 6
    attn.num_heads_kv = 3
    attn.channels_head = 16
    attn.rope = RoPE.Config(channels_head=attn.channels_head)

    report = cfg.make().measure()

    assert report.tokens.shape == (
        len(cfg.prompts),
        cfg.prompt_len + cfg.num_decode_steps + 1,
    )
    assert report.kv_bytes_per_token == (
        cfg.model.num_layers
        * 2
        * attn.num_heads_kv
        * (_packed(attn.channels_head) + SCALE_COLUMNS)
    )


def _cuda_case(
    *,
    device: str = "cuda",
    batch: int = 2,
    seq: int = 1,
    heads: int = 3,
    channels: int = 16,
    length: int = 131,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build bf16 queries over randomly packed keys and values on CUDA."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    torch.manual_seed(seed)
    q = torch.randn(batch, seq, heads, channels, device=device) * 0.5
    q = q.to(torch.bfloat16)
    k = _quantize(
        (torch.randn(batch, length, heads, channels, device=device) * 2).to(
            torch.bfloat16,
        ),
    )
    v = _quantize(
        (torch.randn(batch, length, heads, channels, device=device) * 2).to(
            torch.bfloat16,
        ),
    )
    return q, k, v


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Attend with SDPA over the same rows dequantized out of band."""
    _, length, _, width = k.shape
    channels = 2 * (width - SCALE_COLUMNS)
    key = _dequantized(
        k,
        end=length,
        channels=channels,
        dtype=torch.bfloat16,
    )
    value = _dequantized(
        v,
        end=length,
        channels=channels,
        dtype=torch.bfloat16,
    )
    key = key.movedim(-3, -2)
    value = value.movedim(-3, -2)
    attn_mask = None
    if causal:
        # Bottom-right causal: query row i (at position length - seq + i)
        # attends keys up to its own position; SDPA's is_causal is top-left
        # and disagrees whenever seq < length.
        seq = q.shape[1]
        rows = torch.arange(length - seq, length, device=q.device)
        attn_mask = torch.arange(length, device=q.device)[None, :] <= rows[:, None]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.movedim(-3, -2),
        key,
        value,
        attn_mask=attn_mask,
    )
    return out.movedim(-3, -2)


class _TritonCompiler(Protocol):
    ASTSource: Callable[[object, dict[str, str], dict[str, int | float | bool]], object]

    def compile(
        self,
        source: object,
        *,
        target: object,
        options: dict[str, bool],
    ) -> object: ...


class _CudaBackend(Protocol):
    GPUTarget: Callable[[str, int, int], object]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
