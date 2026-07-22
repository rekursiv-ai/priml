"""Tests for PCA whitening conv layer."""

from __future__ import annotations

import pytest
import torch

from priml.model.whitening import PCAWhiteningConv2d
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_init_whiten_rejects_wrong_out_channels():
    """``out_channels`` must equal ``2 * in_channels * kH * kW``.

    Regression for MODEL-005: rank-doubling produces exactly
    ``2 * in*kH*kW`` filters, so a mismatched ``out_channels`` crashed
    on the weight assignment instead of raising a clear error.
    """
    layer = PCAWhiteningConv2d(3, 48, kernel_size=3, padding=1, bias=False)
    images = torch.randn(16, 3, 8, 8)
    with pytest.raises(ValueError, match="out_channels"):
        layer.init_whiten(images, algorithm="eigh")


def test_init_whiten_shape():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 8)
    layer.init_whiten(images, algorithm="eigh")
    assert layer.weight.shape == (54, 3, 3, 3)
    assert not layer.weight.requires_grad


def test_init_whiten_forward():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 8)
    layer.init_whiten(images, algorithm="eigh")
    out = layer(images[:4])
    assert out.shape == (4, 54, 8, 8)


def test_rank_doubling():
    """Verify [V, -V] structure: second half = negated first half."""
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 8)
    layer.init_whiten(images, algorithm="eigh")
    first_half = layer.weight.data[:27]
    second_half = layer.weight.data[27:]
    assert torch.allclose(first_half, -second_half)


def test_weights_frozen():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    assert not layer.weight.requires_grad


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
