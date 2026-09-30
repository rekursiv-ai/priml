"""Tests for PCA whitening conv layer."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import functools

from torch import Tensor, nn

import pytest
import torch

from priml.math.stats import pca_eigh, pca_power
from priml.model.whitening import PCAWhiteningConv2d
from priml.testing.bfb import assert_bfb_against_golden, bfb_devices
from priml.testing.golden import assert_text_golden


_CWD: Final = Path(__file__).resolve().parent


def _whitening() -> PCAWhiteningConv2d:
    return PCAWhiteningConv2d(2, 16, kernel_size=2, bias=False)


def test_orientation_undoes_an_eigenvector_sign_flip() -> None:
    """A per-eigenvector sign flip, as another LAPACK returns, orients identically."""
    torch.manual_seed(0)
    reference, flipped = _whitening(), _whitening()
    train_images = torch.randn(3, 2, 4, 5)
    reference.init_whiten(train_images)

    def flipped_pca_eigh(x: Tensor) -> tuple[Tensor, Tensor]:
        eigenvalues, eigenvectors = pca_eigh(x)
        eigenvectors = eigenvectors.clone()
        eigenvectors[:, [0, 2]] *= -1
        return eigenvalues, eigenvectors

    flipped.init_whiten(train_images, decompose=flipped_pca_eigh)
    assert torch.equal(flipped.weight, reference.weight)


def test_init_whiten_rejects_wrong_out_channels():
    """``out_channels`` must equal ``2 * in_channels * kH * kW``.

    Regression for MODEL-005: rank-doubling produces exactly
    ``2 * in*kH*kW`` filters, so a mismatched ``out_channels`` crashed
    on the weight assignment instead of raising a clear error.
    """
    layer = PCAWhiteningConv2d(3, 48, kernel_size=3, padding=1, bias=False)
    images = torch.randn(16, 3, 8, 9)
    with pytest.raises(ValueError, match="out_channels"):
        layer.init_whiten(images, decompose=pca_eigh)


def test_init_whiten_shape():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 9)
    layer.init_whiten(images, decompose=pca_eigh)
    assert layer.weight.shape == (54, 3, 3, 3)
    assert not layer.weight.requires_grad


def test_init_whiten_forward():
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 9)
    layer.init_whiten(images, decompose=pca_eigh)
    out = layer(images[:4])
    assert out.shape == (4, 54, 8, 9)


def test_rank_doubling():
    """Verify [V, -V] structure: second half = negated first half."""
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 9)
    layer.init_whiten(images, decompose=pca_eigh)
    first_half = layer.weight.data[:27]
    second_half = layer.weight.data[27:]
    assert torch.allclose(first_half, -second_half)


def test_init_whiten_accepts_injected_decompose():
    """The layer forwards an arbitrary decomposer, e.g. the MPS-native one."""
    layer = PCAWhiteningConv2d(3, 54, kernel_size=3, padding=1, bias=False)
    images = torch.randn(100, 3, 8, 9)
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
    def run(module: nn.Module, inputs: tuple[Tensor, Tensor]) -> Tensor:
        assert isinstance(module, PCAWhiteningConv2d)
        module.init_whiten(inputs[0])
        return module(inputs[1])

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="whitening",
        build_module=lambda: _whitening().to(device),
        # Kernel size 2 requires spatial inputs at least 2x2.
        build_input=lambda: (torch.randn(3, 2, 4, 5), torch.randn(3, 2, 4, 5)),
        seed=0,
        run=run,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
