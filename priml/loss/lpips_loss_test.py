"""Tests for LPIPSLoss."""

from __future__ import annotations

from collections.abc import Callable
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


# The real network is what gets counted, so nothing below mocks ``lpips``.
# ``make()`` reads the trunk's ImageNet weights from ``TORCH_HOME`` (vgg16 is
# 528 MB), which is the large local fixture the marker names; ``cost`` itself
# builds a random trunk on the meta device and needs no weights.
@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(
    ("net", "params"),
    [("alex", 2_470_848), ("vgg", 14_716_160), ("squeeze", 724_736)],
)
def test_lpips_cost_matches_torch_for_every_trunk(net: str, params: int) -> None:
    """Every matmul the forward issues is in the cost, and every parameter, frozen included.

    A token is one position of one scored frame, so ``T <= max_num_random_frames``
    keeps every frame scored and the count deterministic. Both inputs carry a
    gradient so the frozen first convolution's input gradient is measured too.
    """
    b, t, h, w = 1, 2, 32, 32
    analytical = assert_cost_matches_torch(
        LPIPSLoss.Config(net=net),
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
    assert analytical.params == params


def test_lpips_cost_prices_the_frozen_trunk_twice_and_the_head_once() -> None:
    """AlexNet at 32x32: five convolutions write 7x7, 3x3, 1x1, 1x1, 1x1 grids.

    The trunk runs once per branch with an input-gradient-only adjoint; each
    stage's 1x1 ``NetLinLayer`` convolution runs once on the squared difference
    with a full adjoint. The two max pools route one gradient per channel back
    to the argmax at their 3x3 and 1x1 output grids.
    """
    analytical = LPIPSLoss.Config(net="alex").cost(image_size=(32, 32))
    trunk = (
        3 * 121 * 64 * 49
        + 64 * 25 * 192 * 9
        + 192 * 9 * 384
        + 384 * 9 * 256
        + 256 * 9 * 256
    )
    head = 64 * 49 + 192 * 9 + 384 + 256 + 256
    assert analytical.primal.flops.matmul == 2 * (2 * trunk + head) / 1024
    assert analytical.adjoint.flops.matmul == 2 * (2 * trunk + 2 * head) / 1024
    assert analytical.adjoint.flops.selection == 2 * (64 * 9 + 192) / 1024
    assert analytical.bytes_state == 0


def _loss(module: nn.Module, model_output: Tensor, **batch: Tensor) -> Tensor:
    """Run the perceptual loss and return its ``loss`` tensor."""
    assert isinstance(module, LPIPSLoss)
    return module(model_output, **batch)["loss"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
