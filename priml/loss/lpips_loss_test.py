"""Tests for LPIPSLoss."""

from __future__ import annotations

from collections.abc import Callable
from typing import override
from unittest.mock import MagicMock, patch

from torch import Tensor, nn

import pytest
import torch

from priml.loss.lpips_loss import LPIPSLoss
from priml.testing.cost import assert_cost_matches_torch


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


def test_lpips_cost_matches_torch_for_tiny_trunk() -> None:
    """Count both frozen-trunk input gradients and the trainable head's gradients."""
    b, t, h, w = 1, 2, 4, 4
    with (
        patch("torch.hub.get_dir", side_effect=AssertionError("Weight cache accessed")),
        patch("priml.loss.lpips_loss._lpips", side_effect=_tiny_lpips),
    ):
        analytical = assert_cost_matches_torch(
            LPIPSLoss.Config(),
            build_input=lambda: (
                torch.randn(b, 3, t, h, w, requires_grad=True),
                torch.randn(b, 3, t, h, w, requires_grad=True),
            ),
            num_tokens=b * t * h * w,
            bus={"image_size": (h, w)},
            run=lambda module, inputs: _loss(
                module,
                inputs[0],
                x=inputs[0],
                xhat=inputs[1],
            ),
        )
    assert analytical.params == 58


def test_lpips_cost_prices_the_frozen_trunk_twice_and_the_head_once() -> None:
    """Price a frozen 3x3 convolution, 2x2 max pool, and trainable 1x1 head."""
    with patch("priml.loss.lpips_loss._lpips", side_effect=_tiny_lpips):
        analytical = LPIPSLoss.Config().cost(image_size=(4, 4))
    trunk = 3 * 9 * 2 * 16
    head = 2 * 4
    assert analytical.primal.flops.matmul == 2 * (2 * trunk + head) / 16
    assert analytical.adjoint.flops.matmul == 2 * (2 * trunk + 2 * head) / 16
    assert analytical.adjoint.flops.selection == 2 * (2 * 4) / 16
    assert analytical.bytes_state == 0


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
