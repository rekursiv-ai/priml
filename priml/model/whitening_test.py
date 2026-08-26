"""Tests for PCA whitening conv layer."""

from __future__ import annotations

from pathlib import Path

import functools

from torch import Tensor, nn

import pytest
import torch

from priml.math.stats import pca_eigh, pca_power
from priml.model.whitening import PCAWhiteningConv2d
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def _whitening() -> PCAWhiteningConv2d:
    return PCAWhiteningConv2d(1, 8, kernel_size=2, bias=False)


def _run_whitening(module: nn.Module, inputs: tuple[Tensor, Tensor]) -> Tensor:
    assert isinstance(module, PCAWhiteningConv2d)
    train_images, images = inputs
    module.init_whiten(train_images)
    return module(images)


def test_init_whiten_rejects_wrong_out_channels():
    """``out_channels`` must equal ``2 * in_channels * kH * kW``.

    Regression for MODEL-005: rank-doubling produces exactly
    ``2 * in*kH*kW`` filters, so a mismatched ``out_channels`` crashed
    on the weight assignment instead of raising a clear error.
    """
    layer = PCAWhiteningConv2d(3, 48, kernel_size=3, padding=1, bias=False)
    images = torch.randn(16, 3, 8, 8)
    with pytest.raises(ValueError, match="out_channels"):
        layer.init_whiten(images, decompose=pca_eigh)


def test_init_whiten_shape():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 8)
    layer.init_whiten(images, decompose=pca_eigh)
    assert layer.weight.shape == (54, 3, 3, 3)
    assert not layer.weight.requires_grad


def test_init_whiten_forward():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 8)
    layer.init_whiten(images, decompose=pca_eigh)
    out = layer(images[:4])
    assert out.shape == (4, 54, 8, 8)


def test_rank_doubling():
    """Verify [V, -V] structure: second half = negated first half."""
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 8)
    layer.init_whiten(images, decompose=pca_eigh)
    first_half = layer.weight.data[:27]
    second_half = layer.weight.data[27:]
    assert torch.allclose(first_half, -second_half)


def test_init_whiten_accepts_injected_decompose():
    """The layer forwards an arbitrary decomposer, e.g. the MPS-native one."""
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 8)
    layer.init_whiten(images, decompose=functools.partial(pca_power, num_iters=20))
    assert layer.weight.shape == (54, 3, 3, 3)


def test_weights_frozen():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    assert not layer.weight.requires_grad


def test_whitening_text(request: pytest.FixtureRequest) -> None:
    assert_text_golden(
        request,
        test_file=__file__,
        name="whitening",
        rendered=repr(_whitening()),
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_whitening_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="whitening",
        build_module=lambda: _whitening().to(device),
        build_input=lambda: (torch.randn(2, 1, 3, 3), torch.randn(1, 1, 3, 3)),
        seed=0,
        run=_run_whitening,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
