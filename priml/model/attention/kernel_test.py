"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

from configgle import Fig
from configgle.testing import assert_pprint_golden
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import pytest
import torch

from priml.cost import Cost, cost
from priml.model.attention.kernel import (
    SdpaFused,
    SdpaNaive,
    attention_kernel_cost,
)
from priml.model.attention.rope import RoPE, RoPEMixed, rotation_cost
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from priml.model.custom_types import RotaryConfig


_CWD: Final = Path(__file__).resolve().parent


class _Kernel(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.anchor = nn.Parameter(torch.zeros(()))

    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return cast(Tensor, self.inner(q, k, v, is_causal=True, window=2))


@pytest.mark.parametrize("config", [SdpaFused.Config(), SdpaNaive.Config()])
def test_kernel_config_pprint(
    config: SdpaFused.Config | SdpaNaive.Config,
) -> None:
    name = "sdpa_fused" if isinstance(config, SdpaFused.Config) else "sdpa_naive"
    assert_pprint_golden(
        test_file=__file__,
        name=name,
        config=config,
    )


def test_sdpa_fused_forward():
    kernel = SdpaFused.Config().make()
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out = kernel(q, k, v)
    assert out.shape == (2, 4, 8, 16)


def test_sdpa_naive_forward():
    kernel = SdpaNaive.Config().make()
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out = kernel(q, k, v)
    assert out.shape == (2, 4, 8, 16)


def test_naive_matches_fused_noncausal():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    sdp = SdpaFused()(q, k, v)
    eager = SdpaNaive()(q, k, v)
    assert torch.allclose(sdp, eager, atol=1e-6), (
        f"max diff: {(sdp - eager).abs().max().item():.3e}"
    )


def test_naive_matches_fused_causal():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    sdp = SdpaFused()(q, k, v, is_causal=True)
    eager = SdpaNaive()(q, k, v, is_causal=True)
    assert torch.allclose(sdp, eager, atol=1e-6), (
        f"max diff: {(sdp - eager).abs().max().item():.3e}"
    )


def test_naive_causal_masking() -> None:
    """Verify future tokens don't influence past positions."""
    kernel = SdpaNaive()
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 4, 1, 8)
    v = torch.randn(1, 4, 1, 8)
    out_full = kernel(q, k, v, is_causal=True)
    # Changing k/v at position 3 shouldn't affect output at position 0.
    k2, v2 = k.clone(), v.clone()
    k2[:, 3, :, :] = 999.0
    v2[:, 3, :, :] = 999.0
    out_mod = kernel(q, k2, v2, is_causal=True)
    assert torch.equal(out_full[:, 0, :, :], out_mod[:, 0, :, :])


def test_naive_matches_fused_causal_non_square() -> None:
    """Match SdpaFused's non-square causal masking against SdpaNaive's reference.

    Asserts a 1-token query against a 5-token key cache produces numerically
    close output for both kernels under ``is_causal=True``.
    """
    torch.manual_seed(0)
    q = torch.randn(2, 1, 4, 16)  # One new query token...
    k = torch.randn(2, 5, 4, 16)  # ...against a 5-token cache.
    v = torch.randn(2, 5, 4, 16)
    sdp = SdpaFused()(q, k, v, is_causal=True)
    eager = SdpaNaive()(q, k, v, is_causal=True)
    assert torch.allclose(sdp, eager, atol=1e-6), (
        f"max diff: {(sdp - eager).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("config", [SdpaFused.Config(), SdpaNaive.Config()])
@pytest.mark.parametrize(
    ("query_length", "window", "expected"),
    [
        (1, -1, (8.0,)),
        (1, 0, (20.0,)),
        (1, 1, (16.0,)),
        (2, -1, (4.0, 8.0)),
        (2, 0, (12.0, 20.0)),
        (2, 1, (6.0, 16.0)),
    ],
)
def test_cached_kernel_matches_independent_prefix_means(
    config: SdpaFused.Config | SdpaNaive.Config,
    query_length: int,
    window: int,
    expected: tuple[float, ...],
) -> None:
    # Zero scores give uniform weights over the permitted cached suffix/prefix.
    values = torch.tensor([0.0, 0.0, 12.0, 20.0]).reshape(1, 4, 1, 1)
    queries = torch.zeros(1, query_length, 1, 1)
    keys = torch.zeros_like(values)
    with sdpa_kernel(SDPBackend.MATH):
        actual = config.make()(queries, keys, values, is_causal=True, window=window)
    assert torch.equal(actual, torch.tensor(expected).reshape(1, query_length, 1, 1))


def test_the_kernels_agree_when_causality_and_a_mask_are_both_supplied() -> None:
    """A caller's mask must not silently switch causality off in one kernel.

    SDPA refuses ``is_causal`` beside ``attn_mask``, so the fused kernel used to
    drop causality whenever a mask arrived while the naive kernel kept it. The
    two are one algorithm: both fold causality into the mask.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 6, 2, 8) for _ in range(3))
    # A padding mask that fills only INSIDE the causal cone and trusts
    # ``is_causal`` for the rest -- the Qwen 3.5 shape. Key 1, not key 0, so
    # every query keeps at least one admissible key.
    mask = torch.zeros(6, 6)
    mask[:, 1] = float("-inf")
    fused = SdpaFused()(q, k, v, is_causal=True, attn_mask=mask)
    naive = SdpaNaive()(q, k, v, is_causal=True, attn_mask=mask)
    torch.testing.assert_close(fused, naive, rtol=1e-5, atol=1e-5)
    # And causality actually held: the future must not reach a past query.
    k2, v2 = k.clone(), v.clone()
    k2[:, -1] = 999.0
    v2[:, -1] = 999.0
    assert torch.equal(
        SdpaFused()(q, k, v, is_causal=True, attn_mask=mask)[:, :-1],
        SdpaFused()(q, k2, v2, is_causal=True, attn_mask=mask)[:, :-1],
    )


def test_the_kernels_agree_on_a_windowed_forward() -> None:
    """The fused and manual kernels are one algorithm.

    A window cannot change only one of them.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 16, 4, 8) for _ in range(3))
    fused = SdpaFused()(q, k, v, is_causal=True, window=3)
    naive = SdpaNaive()(q, k, v, is_causal=True, window=3)
    torch.testing.assert_close(fused, naive, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kernel", [SdpaFused.Config(), SdpaNaive.Config()])
def test_a_kernel_config_prices_itself_from_the_shapes_its_owner_hands_it(
    kernel: SdpaFused.Config | SdpaNaive.Config,
) -> None:
    """The owner passes head geometry the way it passes q/k/v; the kernel answers."""
    costed = cost(kernel, seq_len=32, dtype=None, num_heads=2, channels_head=8)
    assert costed == attention_kernel_cost(
        seq_len=32,
        dtype=None,
        num_heads=2,
        channels_head=8,
    )
    windowed = cost(
        kernel,
        seq_len=32,
        dtype=None,
        num_heads=2,
        channels_head=8,
        window=4,
    )
    assert windowed == costed
    # ``rows`` is the concrete query-row count for this invocation.
    fewer_rows = cost(
        kernel,
        seq_len=32,
        dtype=None,
        num_heads=2,
        channels_head=8,
        rows=7,
    )
    assert fewer_rows["flops", "matmul"].sum() == (
        7 * costed["flops", "matmul"].sum() // 32
    )
    assert fewer_rows["bytes", "matmul"].sum() < costed["bytes", "matmul"].sum()


def test_kernel_cost_is_two_products_over_the_reachable_keys() -> None:
    """QK^T and PV, two FLOPs per MAC, per head; softmax is elementwise plus sums.

    Per head over ``keys``: the max and the normalizer are two ``keys - 1``
    sums, and the adjoint's ``sum(g * p)`` one more.
    """
    f32 = torch.float32
    full = attention_kernel_cost(seq_len=32, dtype=None, num_heads=2, channels_head=8)
    assert full == Cost(
        cells={
            ("flops", "primal", "matmul", f32): 4 * 2 * 32 * 8 * 32,
            ("flops", "primal", "elementwise", f32): 2 * 4 * 32 * 32,
            ("flops", "primal", "reduction", f32): 2 * 2 * 32 * 31,
            ("flops", "adjoint", "matmul", f32): 2 * 4 * 2 * 32 * 8 * 32,
            ("flops", "adjoint", "elementwise", f32): 2 * 4 * 32 * 32,
            ("flops", "adjoint", "reduction", f32): 2 * 32 * 31,
            ("bytes", "primal", "matmul", f32): 4 * 2 * 2 * (32 * (8 + 32) + 8 * 32),
            ("bytes", "primal", "elementwise", f32): 4 * 2 * 32 * (8 * 32 + 2),
            ("bytes", "primal", "reduction", f32): 4 * 2 * 32 * 2 * (32 + 1),
            ("bytes", "adjoint", "matmul", f32): 4
            * 2
            * 2
            * 2
            * (32 * (8 + 32) + 8 * 32),
            ("bytes", "adjoint", "elementwise", f32): 4 * 2 * 32 * (10 * 32 + 1),
            ("bytes", "adjoint", "reduction", f32): 4 * 2 * 32 * (32 + 1),
        },
    )
    windowed = attention_kernel_cost(
        seq_len=32,
        dtype=None,
        num_heads=2,
        channels_head=8,
        window=4,
    )
    assert windowed["flops", "primal", "matmul"].sum() == 4 * 2 * 32 * 8 * 5
    assert windowed["flops", "primal", "elementwise"].sum() == 2 * 4 * 32 * 5
    # A window past the sequence reaches every key and nothing more.
    assert (
        attention_kernel_cost(
            seq_len=32,
            dtype=None,
            num_heads=2,
            channels_head=8,
            window=64,
        )
        == full
    )
    dropped = attention_kernel_cost(
        seq_len=32,
        dtype=None,
        num_heads=2,
        channels_head=8,
        dropout_p=0.1,
    )
    assert (
        dropped["flops", "primal", "elementwise"].sum()
        == full["flops", "primal", "elementwise"].sum() + 2 * 2 * 32 * 32
    )


def test_kernel_cost_scales_with_batch_size() -> None:
    single = attention_kernel_cost(
        seq_len=8,
        batch_size=1,
        dtype=torch.bfloat16,
        num_heads=2,
        channels_head=4,
    )
    batched = attention_kernel_cost(
        seq_len=8,
        batch_size=2,
        dtype=torch.bfloat16,
        num_heads=2,
        channels_head=4,
    )
    assert batched["flops"].sum() == 2 * single["flops"].sum()
    assert batched["bytes"].sum() == 2 * single["bytes"].sum()


@pytest.mark.parametrize("window", [-1, 4])
def test_kernel_traffic_uses_sequence_reuse_not_batch_reuse(window: int) -> None:
    small = attention_kernel_cost(
        seq_len=8,
        batch_size=1,
        dtype=torch.bfloat16,
        num_heads=2,
        channels_head=4,
        window=window,
    )
    batched = attention_kernel_cost(
        seq_len=8,
        batch_size=4,
        dtype=torch.bfloat16,
        num_heads=2,
        channels_head=4,
        window=window,
    )
    assert batched["flops"].sum() == 4 * small["flops"].sum()
    assert batched["bytes"].sum() == 4 * small["bytes"].sum()
    large = attention_kernel_cost(
        seq_len=8,
        dtype=None,
        num_heads=2,
        channels_head=4,
        window=window,
    )
    assert (
        large["bytes", "primal", torch.float32].sum()
        == small["bytes", "primal", torch.bfloat16].sum() * 2
    )
    assert (
        large["bytes", "adjoint", torch.float32].sum()
        == small["bytes", "adjoint", torch.bfloat16].sum() * 2
    )
    keys = 5 if window == 4 else 8
    assert small["bytes", "primal", "matmul"].sum() == 2 * 2 * 2 * (
        8 * (4 + keys) + 4 * keys
    )
    assert small["flops", "primal"].sum() == large["flops", "primal"].sum()


def test_rotation_cost_reads_the_rotated_width_from_the_rotary_config() -> None:
    """A rotary that rotates a prefix declares it; a stranger cannot be costed."""
    partial = RoPE.Config([4, 0])
    whole = RoPE.Config(8)
    assert partial.rotated_channels(8) == 4
    assert whole.rotated_channels(8) == 8
    assert RoPE.Config([4, 4], reduction_mode="sum").rotated_channels(8) == 4
    narrow = rotation_cost(partial, rows=4, dtype=None, channels_head=8, heads=3)
    wide = rotation_cost(whole, rows=4, dtype=None, channels_head=8, heads=3)
    assert narrow["flops", "primal"].sum() * 2 == wide["flops", "primal"].sum()

    class _Table(nn.Module):
        class Config(Fig["_Table"]):
            pass

        def __init__(self, config: Config) -> None:
            del config
            super().__init__()

    stranger = cast("RotaryConfig", _Table.Config())
    with pytest.raises(AttributeError, match="rotated_channels"):
        rotation_cost(stranger, rows=4, dtype=None, channels_head=8, heads=3)


def test_rotary_traffic_counts_factors_and_rotated_operands() -> None:
    config = RoPE.Config(8)
    factors = config.cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    # Positions are read as int64; the angle table and factors are bf16.
    assert factors["bytes", "primal", "elementwise", torch.int64] == 8 * 4
    assert factors["bytes", "primal", "elementwise", torch.bfloat16] == 2 * (
        4 + 4 * (4 + 8 * 4)
    )
    assert factors["bytes", "adjoint"].sum() == 0
    rotation = rotation_cost(
        config,
        rows=4,
        dtype=torch.bfloat16,
        channels_head=8,
        heads=3,
    )
    assert rotation["bytes", "primal", "elementwise"].sum() == 4 * 2 * 9 * 8 * 3
    assert rotation["bytes", "adjoint", "elementwise"].sum() == 4 * 2 * 9 * 8 * 3
    mixed = RoPEMixed.Config(8)
    mixed.num_heads = 2
    mixed.learnable = True
    small = mixed.cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    large = mixed.cost(seq_len=4, batch_size=1, dtype=None)
    # Positions stay int64 at either width; every payload cell doubles.
    assert (
        large["bytes", torch.float32].sum() == small["bytes", torch.bfloat16].sum() * 2
    )
    assert large["bytes", torch.int64] == small["bytes", torch.int64]


@pytest.mark.parametrize("config", [SdpaNaive.Config(), SdpaFused.Config()])
@pytest.mark.parametrize("window", [-1, 0, 1, 4, 8, 12])
def test_kernel_cost_matches_torch(
    config: SdpaNaive.Config | SdpaFused.Config,
    window: int,
) -> None:
    """Measure both kernels' logical products, traffic, and parameter count."""
    # CPU flash SDPA lacks a FLOP counter; math dispatch exposes the products.
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=lambda: tuple(
                torch.randn(1, 8, 2, 4, requires_grad=True) for _ in range(3)
            ),
            batch_size=1,
            seq_len=8,
            dtype=None,
            num_heads=2,
            channels_head=4,
            window=window,
            run=lambda module, inputs: cast(
                Tensor,
                module(*inputs, is_causal=True, window=window),
            ),
        )
    assert analytical["flops", "primal", "matmul"].sum() == 4 * 2 * 4 * 8 * 8


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
@pytest.mark.parametrize(
    ("name", "kernel"),
    [("sdpa_fused", SdpaFused.Config()), ("sdpa_naive", SdpaNaive.Config())],
)
def test_kernel_bfb(device: str, name: str, kernel: object) -> None:
    assert isinstance(kernel, (SdpaFused.Config, SdpaNaive.Config))
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name=name,
        build_module=lambda: _Kernel(kernel.make()).to(device),
        build_input=lambda: tuple(torch.randn(1, 4, 2, 8) for _ in range(3)),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
