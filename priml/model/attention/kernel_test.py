"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast, override

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.kernel import SdpaFused, SdpaNaive
from priml.model.attention.rope import RoPE, RoPEMixed, rotation_cost
from priml.model.cost import Bytes, Compute, Cost, Flops, cost
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.cost import assert_cost_matches_torch


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


def test_naive_causal_masking():
    """Verify future tokens don't influence past positions."""
    kernel = SdpaNaive()
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    out_full = kernel(q, k, v, is_causal=True)
    # Changing k/v at position 3 shouldn't affect output at position 0.
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 3, :] = 999.0
    v2[:, :, 3, :] = 999.0
    out_mod = kernel(q, k2, v2, is_causal=True)
    assert torch.equal(out_full[:, :, 0, :], out_mod[:, :, 0, :])


def test_the_kernels_agree_on_a_windowed_forward() -> None:
    """The fused and manual kernels are one algorithm.

    A window cannot change only one of them.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 16, 4, 8) for _ in range(3))
    fused = SdpaFused()(q, k, v, is_causal=True, window=3)
    naive = SdpaNaive()(q, k, v, is_causal=True, window=3)
    torch.testing.assert_close(fused, naive, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("config", [SdpaFused.Config(), SdpaNaive.Config()])
def test_kernel_cost_is_two_products_over_the_reachable_keys(
    config: SdpaFused.Config | SdpaNaive.Config,
) -> None:
    """QK^T and PV, two FLOPs per MAC, per head; softmax is elementwise plus sums.

    Per head over ``keys``: the max and the normalizer are two ``keys - 1``
    sums, and the adjoint's ``sum(g * p)`` one more.
    """
    full = cost(config, seq_len=32, num_heads=2, channels_head=8)
    assert full == Cost(
        primal=Compute(
            flops=Flops(
                matmul=4 * 2 * 8 * 32,
                elementwise=2 * 4 * 32,
                reduction=2 * 62,
            ),
            bytes=Bytes(
                matmul=4 * 2 * (4 * 8 + 2 * 32),
                elementwise=4 * 2 * (8 * 32 + 2),
                reduction=4 * 2 * 2 * (32 + 1),
            ),
        ),
        adjoint=Compute(
            flops=Flops(
                matmul=2 * 4 * 2 * 8 * 32,
                elementwise=2 * 4 * 32,
                reduction=2 * 31,
            ),
            bytes=Bytes(
                matmul=4 * 2 * 2 * (4 * 8 + 2 * 32),
                elementwise=4 * 2 * (10 * 32 + 1),
                reduction=4 * 2 * (32 + 1),
            ),
        ),
    )
    windowed = cost(config, seq_len=32, num_heads=2, channels_head=8, window=4)
    assert windowed.primal.flops.matmul == 4 * 2 * 8 * 4
    assert windowed.primal.flops.elementwise == 2 * 4 * 4
    # A window past the sequence reaches every key and nothing more.
    assert cost(config, seq_len=32, num_heads=2, channels_head=8, window=64) == full
    dropped = cost(config, seq_len=32, num_heads=2, channels_head=8, dropout_p=0.1)
    assert (
        dropped.primal.flops.elementwise == full.primal.flops.elementwise + 2 * 2 * 32
    )


@pytest.mark.parametrize("config", [SdpaFused.Config(), SdpaNaive.Config()])
@pytest.mark.parametrize("window", [-1, 4])
def test_kernel_traffic_uses_sequence_reuse_not_batch_reuse(
    config: SdpaFused.Config | SdpaNaive.Config,
    window: int,
) -> None:
    small = cost(
        config,
        seq_len=8,
        num_heads=2,
        channels_head=4,
        window=window,
        rows=8,
        itemsize=2,
    )
    large = cost(
        config,
        seq_len=8,
        num_heads=2,
        channels_head=4,
        window=window,
        rows=32,
        itemsize=4,
    )
    assert large.primal.bytes == small.primal.bytes * 2
    assert large.adjoint.bytes == small.adjoint.bytes * 2
    assert small.primal.bytes.matmul == 2 * 2 * (4 * 4 + 2 * (4 if window == 4 else 8))
    assert small.primal.flops == large.primal.flops


def test_rotary_traffic_counts_factors_and_rotated_operands() -> None:
    config = RoPE.Config(8)
    factors = config.cost(rows=4, itemsize=2)
    assert factors.primal.bytes.elementwise == 2 * (1 + 4 / 4 + 4 + 8 * 4)
    assert factors.adjoint.bytes.total == 0
    rotation = rotation_cost(config, channels_head=8, heads=3, itemsize=2)
    assert rotation.primal.bytes.elementwise == 2 * 9 * 8 * 3
    assert rotation.adjoint.bytes.elementwise == 2 * 9 * 8 * 3
    mixed = RoPEMixed.Config(8)
    mixed.num_heads = 2
    mixed.learnable = True
    small = mixed.cost(rows=4, itemsize=2)
    large = mixed.cost(rows=4, itemsize=4)
    assert large.training.bytes == small.training.bytes * 2


def test_naive_kernel_cost_matches_torch() -> None:
    """The manual kernel is two bmms each way; torch counts exactly that.

    Only the naive kernel is measurable here: the CPU SDPA op that
    ``SdpaFused`` dispatches to has no ``FlopCounterMode`` registration and
    measures zero, which is the silent-zero bug ``cost`` exists to prevent.
    """
    analytical = assert_cost_matches_torch(
        SdpaNaive.Config(),
        build_input=lambda: tuple(
            torch.randn(1, 8, 2, 4, requires_grad=True) for _ in range(3)
        ),
        num_tokens=8,
        bus={"seq_len": 8, "num_heads": 2, "channels_head": 4},
    )
    assert analytical.primal.flops.matmul == 4 * 2 * 4 * 8


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
