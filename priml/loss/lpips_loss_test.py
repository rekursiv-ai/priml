"""Tests for LPIPSLoss."""

from __future__ import annotations

from typing import TYPE_CHECKING, override
from unittest.mock import MagicMock, patch

from torch import Tensor, nn

import pytest
import torch

from priml.loss.lpips_loss import (
    LPIPSLoss,
    _normalize_cost,
    _spatial_average_cost,
)
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Callable


def _make_loss_with_mocked_lpips(
    side_effect: Callable[[Tensor, Tensor], Tensor],
    *,
    max_num_random_frames: int,
) -> LPIPSLoss:
    """Create ``LPIPSLoss`` with a patched LPIPS module boundary."""
    mock_lpips = MagicMock()
    mock_criterion = MagicMock()
    mock_criterion.side_effect = side_effect
    mock_lpips.LPIPS.return_value = mock_criterion
    with patch("priml.loss.lpips_loss.lpips", mock_lpips):
        return LPIPSLoss.Config(max_num_random_frames=max_num_random_frames).make()


@pytest.fixture
def lpips_loss() -> LPIPSLoss:
    """Create LPIPSLoss with mocked lpips criterion."""
    return _make_loss_with_mocked_lpips(
        lambda x, _: torch.rand(x.shape[0], 1, 1, 1),
        max_num_random_frames=10,
    )


def test_lpips_loss_basic(lpips_loss: LPIPSLoss) -> None:
    """Test LPIPSLoss basic functionality."""
    x = torch.randn(2, 3, 4, 64, 64)
    xhat = torch.randn(2, 3, 4, 64, 64)
    dummy_model_output = torch.randn(2, 3, 4, 64, 64)

    result = lpips_loss(dummy_model_output, x=x, xhat=xhat)

    assert "loss" in result
    assert result["loss"].shape == (2,)


def test_lpips_loss_scalar(lpips_loss: LPIPSLoss) -> None:
    """Test LPIPSLoss returns per-sample losses."""
    x = torch.randn(2, 3, 4, 64, 64)
    xhat = torch.randn(2, 3, 4, 64, 64)
    dummy_model_output = torch.randn(2, 3, 4, 64, 64)

    result = lpips_loss(dummy_model_output, x=x, xhat=xhat)

    assert result["loss"].shape == (2,)
    assert (result["loss"] >= 0).all()


def test_lpips_loss_perfect_reconstruction() -> None:
    """Test LPIPSLoss with perfect reconstruction (mocked to return near-zero)."""
    loss = _make_loss_with_mocked_lpips(
        lambda x, _: torch.zeros(x.shape[0], 1, 1, 1),
        max_num_random_frames=2,
    )

    x = torch.randn(1, 3, 2, 64, 64)
    xhat = x.clone()
    dummy_model_output = x.clone()

    result = loss(dummy_model_output, x=x, xhat=xhat)

    assert result["loss"].shape == (1,)
    assert result["loss"].item() < 0.01


def test_lpips_loss_frame_sampling(lpips_loss: LPIPSLoss) -> None:
    """Test LPIPSLoss frame sampling."""
    x = torch.randn(1, 3, 10, 64, 64)
    xhat = torch.randn(1, 3, 10, 64, 64)
    dummy_model_output = torch.randn(1, 3, 10, 64, 64)

    result = lpips_loss(dummy_model_output, x=x, xhat=xhat)

    assert result["loss"].shape == (1,)


def test_lpips_loss_returns_one_loss_per_input_sample() -> None:
    """LOSSOPT-009: output is pointwise [B] over all input samples, not sliced."""
    loss = _make_loss_with_mocked_lpips(
        lambda x, _: torch.rand(x.shape[0], 1, 1, 1),
        max_num_random_frames=2,
    )

    x = torch.randn(4, 3, 2, 8, 8)
    xhat = torch.randn(4, 3, 2, 8, 8)
    dummy_model_output = torch.randn(4, 3, 2, 8, 8)

    result = loss(dummy_model_output, x=x, xhat=xhat)

    assert result["loss"].shape == (4,)


def test_lpips_loss_fewer_frames(lpips_loss: LPIPSLoss) -> None:
    """Test LPIPSLoss when video has fewer frames than max."""
    x = torch.randn(1, 3, 2, 64, 64)
    xhat = torch.randn(1, 3, 2, 64, 64)
    dummy_model_output = torch.randn(1, 3, 2, 64, 64)

    result = lpips_loss(dummy_model_output, x=x, xhat=xhat)

    assert result["loss"].shape == (1,)


@pytest.mark.parametrize("shape", [(1, 2, 4, 4), (2, 1, 5, 7), (2, 3, 8, 6)])
def test_lpips_cost_matches_torch_for_tiny_trunk(
    shape: tuple[int, int, int, int],
) -> None:
    """Count both frozen-trunk input gradients and the trainable head's gradients."""
    b, t, h, w = shape
    scored = min(t, LPIPSLoss.Config().max_num_random_frames)
    with (
        patch("torch.hub.get_dir", side_effect=AssertionError("Weight cache accessed")),
        patch("priml.loss.lpips_loss._lpips", side_effect=_tiny_lpips),
    ):
        analytical = assert_cost_matches_torch(
            LPIPSLoss.Config(image_size=(h, w)),
            build_input=lambda: (
                torch.randn(b, 3, t, h, w, requires_grad=True),
                torch.randn(b, 3, t, h, w, requires_grad=True),
            ),
            seq_len=scored,
            batch_size=b,
            dtype=None,
            run=lambda module, inputs: _loss(
                module,
                inputs[0],
                x=inputs[0],
                xhat=inputs[1],
            ),
        )
    assert analytical.params == 58


def test_lpips_cost_prices_the_frozen_trunk_twice_and_the_head_once() -> None:
    """Cost a frozen 3x3 convolution, 2x2 max pool, and trainable 1x1 head."""
    with patch("priml.loss.lpips_loss._lpips", side_effect=_tiny_lpips):
        analytical = LPIPSLoss.Config(image_size=(4, 4)).cost(
            seq_len=1,
            batch_size=1,
            dtype=None,
        )
    trunk_products = 3 * 9 * 2 * 16
    head_products = 2 * 1 * 1 * 4
    assert analytical["flops", "primal", "matmul"].sum() == (
        2 * 2 * trunk_products + 2 * head_products
    )
    assert analytical["flops", "adjoint", "matmul"].sum() == (
        2 * 2 * trunk_products + 4 * head_products
    )
    assert analytical["flops", "adjoint", "selection"].sum() == 2 * 2 * 4
    assert analytical.bytes_state == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_lpips_operand_traffic(dtype: torch.dtype) -> None:
    itemsize = dtype.itemsize
    normalized = _normalize_cost(3, dtype=dtype)
    assert normalized["bytes", "primal", "elementwise"].sum() == itemsize * (4 * 3 + 5)
    assert normalized["bytes", "primal", "reduction"].sum() == itemsize * (3 + 1)
    assert normalized["bytes", "adjoint", "elementwise"].sum() == itemsize * (
        10 * 3 + 9
    )
    assert normalized["bytes", "adjoint", "reduction"].sum() == itemsize * (3 + 1)
    averaged = _spatial_average_cost(6, dtype=dtype)
    assert averaged["bytes", "primal", "reduction"].sum() == itemsize * 7
    assert averaged["bytes", "primal", "elementwise"].sum() == itemsize * 2
    assert averaged["bytes", "adjoint", "elementwise"].sum() == itemsize * 7


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_lpips_frame_selection_traffic(dtype: torch.dtype) -> None:
    with patch("priml.loss.lpips_loss._lpips", side_effect=_tiny_lpips):
        costed = LPIPSLoss.Config(image_size=(4, 4)).cost(
            seq_len=1,
            batch_size=1,
            dtype=dtype,
        )
    itemsize = dtype.itemsize
    positions = 16
    # Two frame gathers of the RGB pixels at ``dtype``; the shared frame index
    # is one int64 read per branch.
    assert costed["bytes", "primal", "selection", dtype] == (
        itemsize * 2 * 3 * 2 * positions
    )
    assert costed["bytes", "primal", "selection", torch.int64] == 8 * 2
    # Back: the pool's dense routing (values at dtype, argmax int64) and the
    # two branches' pixel scatters.
    pool_values = 2 * 2 * (4 + 1) * 4
    pool_index = 2 * 2 * 4
    assert costed["bytes", "adjoint", "selection", dtype] == itemsize * (
        pool_values + 2 * 3 * 3 * positions
    )
    assert costed["bytes", "adjoint", "selection", torch.int64] == 8 * (pool_index + 2)


class _TinyTrunk(nn.Module):
    """One frozen convolutional stage with two channels."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.requires_grad_(False)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor]:
        return (self.layers(x),)


class _TinyLPIPS(nn.Module):
    """Minimal two-branch perceptual network for cost accounting."""

    def __init__(self) -> None:
        super().__init__()
        self.net = _TinyTrunk()
        self.L = 1
        self.lins = nn.ModuleList(  # codespell:ignore lins
            [nn.Conv2d(2, 1, 1, bias=False)],
        )

    @override
    def forward(self, x: Tensor, xhat: Tensor) -> Tensor:
        a, b = self.net(x)[0], self.net(xhat)[0]
        a = a / (a.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)
        b = b / (b.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)
        head = self.lins[0]  # codespell:ignore lins
        return head((a - b).square()).mean(dim=(2, 3), keepdim=True)


def _tiny_lpips(net: str, *, pretrained: bool) -> _TinyLPIPS:
    """Replace heavyweight LPIPS construction on both CPU and meta devices."""
    del net, pretrained
    return _TinyLPIPS()


def _loss(module: nn.Module, model_output: Tensor, **batch: Tensor) -> Tensor:
    """Run the perceptual loss and return its ``loss`` tensor."""
    assert isinstance(module, LPIPSLoss)
    return module(model_output, **batch)["loss"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
