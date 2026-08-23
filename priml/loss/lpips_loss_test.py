"""Tests for LPIPSLoss."""
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

from torch import Tensor

import pytest
import torch

from priml.loss.lpips_loss import LPIPSLoss


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
