"""Tests for diffusion loss module."""

from __future__ import annotations

from typing import override

from torch import Tensor

import torch

from priml.loss.diffusion import DiffusionLoss
from priml.math.diffusion.schedule import (
    log_sigma_from_log_snr_per_variance_preserving,
    log_snr_from_log_time_per_logtan,
)
from priml.math.diffusion.target import target_eps, target_v, target_x


def test_diffusion_loss_default_config() -> None:
    """Test DiffusionLoss with default configuration."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    # Check defaults were set
    assert loss_fn.logsnr_fn is not None
    assert loss_fn.target_fn is not None
    assert loss_fn.corruption_fn is not None


def test_diffusion_loss_forward_basic() -> None:
    """Test basic forward pass."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    # Simple denoiser that returns zeros
    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return torch.zeros_like(x)

    # Input: [B, C, F, H, W]
    x0 = torch.randn(2, 3, 4, 8, 8)

    result = loss_fn(denoiser=denoiser, x0=x0)

    # Check output structure
    assert "loss" in result
    assert "x_denoised" in result
    assert "eps_denoised" in result

    # Check shapes
    assert result["loss"].shape == (2,)
    assert result["x_denoised"].shape == x0.shape
    assert result["eps_denoised"].shape == x0.shape


def test_diffusion_loss_forward_2d() -> None:
    """Test forward pass with 2D images."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x * 0.5

    # Input: [B, C, H, W]
    x0 = torch.randn(4, 3, 16, 16)

    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (4,)
    assert result["x_denoised"].shape == x0.shape


def test_diffusion_loss_forward_1d() -> None:
    """Test forward pass with 1D signals."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x

    # Input: [B, C, L]
    x0 = torch.randn(8, 16, 128)

    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (8,)
    assert result["x_denoised"].shape == x0.shape


def test_diffusion_loss_custom_logsnr() -> None:
    """Test with custom log SNR function."""
    cfg = DiffusionLoss.Config()
    cfg.logsnr_fn = lambda log_t: log_snr_from_log_time_per_logtan(log_t, shift=0)

    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return torch.zeros_like(x)

    x0 = torch.randn(2, 3, 4, 8, 8)
    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (2,)


def test_diffusion_loss_custom_target() -> None:
    """Test with custom target function."""
    cfg = DiffusionLoss.Config()
    cfg.target_fn = target_x

    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x

    x0 = torch.randn(2, 3, 4, 8, 8)
    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (2,)


def test_diffusion_loss_custom_corruption() -> None:
    """Test with custom corruption function."""
    cfg = DiffusionLoss.Config()
    cfg.corruption_fn = log_sigma_from_log_snr_per_variance_preserving

    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return torch.zeros_like(x)

    x0 = torch.randn(2, 3, 4, 8, 8)
    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (2,)


def test_diffusion_loss_variance_preserving() -> None:
    """Test variance preserving diffusion."""
    cfg = DiffusionLoss.Config()
    cfg.corruption_fn = log_sigma_from_log_snr_per_variance_preserving
    cfg.target_fn = target_v

    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return torch.randn_like(x) * 0.1

    x0 = torch.randn(4, 3, 8, 8)
    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (4,)
    assert torch.all(result["loss"] >= 0)


def test_diffusion_loss_eps_prediction() -> None:
    """Test epsilon prediction target."""
    cfg = DiffusionLoss.Config()
    cfg.target_fn = target_eps

    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return torch.randn_like(x)

    x0 = torch.randn(2, 3, 4, 8, 8)
    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (2,)


def test_diffusion_loss_deterministic() -> None:
    """Test that same seed produces same results."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x * 0.5

    x0 = torch.randn(2, 3, 4, 8, 8)

    # Run twice with same seed
    torch.manual_seed(42)
    result1 = loss_fn(denoiser=denoiser, x0=x0)

    torch.manual_seed(42)
    result2 = loss_fn(denoiser=denoiser, x0=x0)

    torch.testing.assert_close(result1["loss"], result2["loss"])


def test_diffusion_loss_batch_independence() -> None:
    """Test that batch samples are independent."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x * 0.9

    x0 = torch.randn(4, 3, 4, 8, 8)

    result = loss_fn(denoiser=denoiser, x0=x0)

    # Each sample should have different loss
    assert not torch.allclose(result["loss"][0], result["loss"][1])


def test_diffusion_loss_perfect_denoiser() -> None:
    """Test that perfect denoiser has finite and non-negative loss."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def perfect_denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        # This would require knowing the noise, so we can't make it perfect
        # But we can test the structure
        del sigma
        return x

    x0 = torch.randn(2, 3, 4, 8, 8)
    result = loss_fn(denoiser=perfect_denoiser, x0=x0)

    # Loss should be finite and non-negative
    assert torch.all(torch.isfinite(result["loss"]))
    assert torch.all(result["loss"] >= 0)


def test_diffusion_loss_gradient_flow() -> None:
    """Test that gradients flow through the loss."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    # Simple learnable denoiser
    class LearnableDenoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        @override
        def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
            del sigma
            return x * self.weight

    denoiser = LearnableDenoiser()
    x0 = torch.randn(2, 3, 4, 8, 8)

    result = loss_fn(denoiser=denoiser.forward, x0=x0)
    loss = result["loss"].mean()

    # Backward should work
    loss.backward()
    assert denoiser.weight.grad is not None
    assert torch.any(denoiser.weight.grad != 0)


def test_diffusion_loss_large_batch() -> None:
    """Test with larger batch size."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x * 0.8

    x0 = torch.randn(32, 3, 4, 8, 8)

    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (32,)
    assert torch.all(torch.isfinite(result["loss"]))


def test_diffusion_loss_different_dtypes() -> None:
    """Test with different data types."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x

    # Test with float32
    x0_f32 = torch.randn(2, 3, 4, 8, 8, dtype=torch.float32)
    result_f32 = loss_fn(denoiser=denoiser, x0=x0_f32)
    assert result_f32["loss"].dtype == torch.float32

    # Test with float64
    x0_f64 = torch.randn(2, 3, 4, 8, 8, dtype=torch.float64)
    result_f64 = loss_fn(denoiser=denoiser, x0=x0_f64)
    assert result_f64["loss"].dtype == torch.float64


def test_diffusion_loss_min_snr_gamma() -> None:
    """Test Min-SNR-γ weighting downweights high-SNR timesteps."""
    torch.manual_seed(42)

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x * 0.5

    x0 = torch.randn(64, 3, 8, 8)

    # Without Min-SNR.
    loss_fn_plain = DiffusionLoss.Config(snr_gamma=0.0).make()
    result_plain = loss_fn_plain(denoiser=denoiser, x0=x0)

    # With Min-SNR gamma=5.
    torch.manual_seed(42)
    loss_fn_snr = DiffusionLoss.Config(snr_gamma=5.0).make()
    result_snr = loss_fn_snr(denoiser=denoiser, x0=x0)

    # Min-SNR should reduce loss at high-SNR timesteps.
    # Overall mean loss with Min-SNR should be <= plain (weights ≤ 1).
    assert result_snr["loss"].mean() <= result_plain["loss"].mean()
    # Shapes should match.
    assert result_snr["loss"].shape == result_plain["loss"].shape
    # Weights are in [0, 1] so SNR loss should be non-negative.
    assert torch.all(result_snr["loss"] >= 0)


def test_diffusion_loss_min_snr_gamma_gradient_flow() -> None:
    """Gradients flow through Min-SNR weighted loss."""

    class Denoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        @override
        def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
            del sigma
            return x * self.w

    model = Denoiser()
    loss_fn = DiffusionLoss.Config(snr_gamma=5.0).make()
    result = loss_fn(denoiser=model.forward, x0=torch.randn(4, 3, 8, 8))
    result["loss"].mean().backward()
    assert model.w.grad is not None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
