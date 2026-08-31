"""Tests for attention module."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.kernel import SdpaFused, SdpaNaive
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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
    # Changing k/v at position 3 shouldn't affect output at position 0
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 3, :] = 999.0
    v2[:, :, 3, :] = 999.0
    out_mod = kernel(q, k2, v2, is_causal=True)
    assert torch.equal(out_full[:, :, 0, :], out_mod[:, :, 0, :])


def test_the_kernels_agree_on_a_windowed_forward() -> None:
    """The fused and manual kernels are one algorithm, so a window cannot
    change only one of them.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 16, 4, 8) for _ in range(3))
    fused = SdpaFused()(q, k, v, is_causal=True, window=3)
    naive = SdpaNaive()(q, k, v, is_causal=True, window=3)
    torch.testing.assert_close(fused, naive, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
@pytest.mark.parametrize(
    ("name", "kernel"),
    [("sdpa_fused", SdpaFused.Config()), ("sdpa_naive", SdpaNaive.Config())],
)
def test_kernel_bfb(device: str, name: str, kernel: object) -> None:
    assert isinstance(kernel, (SdpaFused.Config, SdpaNaive.Config))
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name=name,
        build_module=lambda: _Kernel(kernel.make()).to(device),
        build_input=lambda: tuple(torch.randn(1, 4, 2, 8) for _ in range(3)),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
