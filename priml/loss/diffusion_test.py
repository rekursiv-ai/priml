"""Tests for diffusion loss module."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import functools

from torch import Tensor, nn

import pytest
import torch

from priml.cost import MEASURES, Cost, Kernel, Phase, cost
from priml.loss.diffusion import DiffusionLoss
from priml.math.diffusion.schedule import (
    log_sigma_from_log_snr_per_variance_preserving,
    log_snr_from_log_time_per_logtan,
    log_time_from_log_snr_per_logit,
)
from priml.math.diffusion.target import (
    TargetFn,
    TargetResult,
    target_eps,
    target_rectified_flow,
    target_v,
    target_v_eps,
    target_v_x,
    target_x,
)
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _fp32(
    *,
    primal: Mapping[str, Mapping[Kernel, int]] | None = None,
    adjoint: Mapping[str, Mapping[Kernel, int]] | None = None,
    **fields: int,
) -> Cost:
    """Build a ``Cost`` from whole-invocation fp32 FLOPs and bytes."""
    cells: dict[tuple[object, ...], int] = {}
    phases: tuple[tuple[Phase, Mapping[str, Mapping[Kernel, int]] | None], ...] = (
        ("primal", primal),
        ("adjoint", adjoint),
    )
    for phase, silos in phases:
        for measure in MEASURES:
            for kernel, value in (silos or {}).get(measure, {}).items():
                cells[(measure, phase, kernel, torch.float32)] = value
    return Cost(cells=cells, **fields)


def _per_sample(config: DiffusionLoss.Config) -> Cost:
    """Sum one sample's scalar work and each schedule child's own cost."""
    schedules = [config.logsnr_fn, config.corruption_fn]
    if config.time_transform is not None:
        schedules.append(config.time_transform)
    own = Cost(
        cells={
            ("flops", "primal", "elementwise", torch.float32): 5,
            ("bytes", "primal", "elementwise", torch.float32): 4 * 6,
        },
    )
    return sum((cost(fn, channels=1, dtype=None) for fn in schedules), own)


def _per_element(target_fn: TargetFn, *, elements: int, samples: int) -> Cost:
    """Sum one loss invocation and the target's own cost."""
    own = _fp32(
        primal={
            "flops": {"elementwise": 5 * elements, "reduction": elements - samples},
            "bytes": {
                "elementwise": 4 * 13 * elements,
                "reduction": 4 * (elements + samples),
            },
        },
        adjoint={
            "flops": {"elementwise": 3 * elements},
            "bytes": {
                "elementwise": 28 * elements,
                "reduction": 4 * (elements + samples),
            },
        },
    )
    return own + cost(target_fn, dtype=None, elements=elements, samples=samples)


def test_diffusion_loss_default_config() -> None:
    """Test DiffusionLoss with default configuration."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    # Check defaults were set.
    assert loss_fn.logsnr_fn is not None
    assert loss_fn.target_fn is not None
    assert loss_fn.corruption_fn is not None


def test_diffusion_loss_forward_basic() -> None:
    """Test basic forward pass."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    # Simple denoiser that returns zeros.
    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return torch.zeros_like(x)

    # Input: [B, C, F, H, W].
    x0 = torch.randn(2, 3, 4, 8, 8)

    result = loss_fn(denoiser=denoiser, x0=x0)

    # Check output structure.
    assert "loss" in result
    assert "x_denoised" in result
    assert "eps_denoised" in result

    # Check shapes.
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

    # Input: [B, C, H, W].
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

    # Input: [B, C, L].
    x0 = torch.randn(8, 16, 128)

    result = loss_fn(denoiser=denoiser, x0=x0)

    assert result["loss"].shape == (8,)
    assert result["x_denoised"].shape == x0.shape


def test_diffusion_loss_custom_logsnr() -> None:
    """Test with custom log SNR function."""
    cfg = DiffusionLoss.Config()
    cfg.logsnr_fn = functools.partial(log_snr_from_log_time_per_logtan, shift=0)

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

    # Run twice with same seed.
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

    # Each sample should have different loss.
    assert not torch.allclose(result["loss"][0], result["loss"][1])


def test_diffusion_loss_perfect_denoiser() -> None:
    """Test that perfect denoiser has finite and non-negative loss."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    def perfect_denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        # This would require knowing the noise, so we can't make it perfect
        # But we can test the structure.
        del sigma
        return x

    x0 = torch.randn(2, 3, 4, 8, 8)
    result = loss_fn(denoiser=perfect_denoiser, x0=x0)

    # Loss should be finite and non-negative.
    assert torch.all(torch.isfinite(result["loss"]))
    assert torch.all(result["loss"] >= 0)


def test_diffusion_loss_gradient_flow() -> None:
    """Test that gradients flow through the loss."""
    cfg = DiffusionLoss.Config()
    loss_fn = cfg.make()

    # Simple learnable denoiser.
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

    # Backward should work.
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

    # Test with float32.
    x0_f32 = torch.randn(2, 3, 4, 8, 8, dtype=torch.float32)
    result_f32 = loss_fn(denoiser=denoiser, x0=x0_f32)
    assert result_f32["loss"].dtype == torch.float32

    # Test with float64.
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


def test_diffusion_loss_cost_counts_the_complete_batch() -> None:
    """Count per-element work and each sample's scalar work exactly."""
    scale = torch.ones((), requires_grad=True)

    def denoiser(x: Tensor, sigma: Tensor) -> Tensor:
        del sigma
        return x * scale

    config = DiffusionLoss.Config()
    measured = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 4, 4),
        seq_len=48,
        batch_size=2,
        dtype=None,
        run=lambda module, x0: _loss(module, denoiser=denoiser, x0=x0),
    )
    expected = _per_element(
        target_rectified_flow,
        elements=96,
        samples=2,
    ) + _per_sample(
        config,
    ).tile(2)
    assert measured == expected
    assert measured.params == 0
    assert measured["flops", "matmul"].sum() == 0


@pytest.mark.parametrize(
    ("target_fn", "primal", "adjoint"),
    [
        (target_x, 3, 0),
        (target_eps, 3, 0),
        (target_rectified_flow, 7, 0),
        (target_v, 9, 0),
        (target_v_x, 6, 1),
        (target_v_eps, 6, 1),
    ],
)
def test_diffusion_loss_cost_asks_the_target_fn_for_its_price(
    target_fn: TargetFn,
    primal: int,
    adjoint: int,
) -> None:
    """Passthrough targets add no adjoint; ``v_x``/``v_eps`` scale the gradient once."""
    config = DiffusionLoss.Config(target_fn=target_fn)
    costed = cost(config, seq_len=48, batch_size=2, dtype=None)
    assert costed == _per_element(target_fn, elements=96, samples=2) + _per_sample(
        config,
    ).tile(2)
    own = cost(target_fn, dtype=None, elements=96, samples=2)
    assert own["flops", "primal", "elementwise"].sum() == primal * 96 + 8 * 2
    assert own["flops", "adjoint", "elementwise"].sum() == adjoint * 96


def test_diffusion_loss_cost_counts_snr_weight_and_time_transform() -> None:
    """Min-SNR weighting and a time transform count once per sample."""
    plain = cost(DiffusionLoss.Config(), seq_len=48, batch_size=2, dtype=None)
    weighted = cost(
        DiffusionLoss.Config(snr_gamma=5.0),
        seq_len=48,
        batch_size=2,
        dtype=None,
    )
    assert weighted["flops", "primal", "elementwise"].sum() == (
        plain["flops", "primal", "elementwise"].sum() + 5 * 2
    )
    assert weighted["flops", "adjoint", "elementwise"].sum() == (
        plain["flops", "adjoint", "elementwise"].sum() + 1 * 2
    )


def test_diffusion_loss_cost_rejects_unpriced_target_fn() -> None:
    def target_custom(
        model: Tensor,
        x_noisy: Tensor,
        log_snr: Tensor,
        log_sigma: Tensor,
        *,
        x_original: Tensor | None = None,
        eps_original: Tensor | None = None,
    ) -> TargetResult:
        del x_noisy, log_snr, log_sigma, eps_original
        return TargetResult(x_original, model, model, model)

    with pytest.raises(TypeError, match="target_custom has no cost"):
        cost(
            DiffusionLoss.Config(target_fn=target_custom),
            seq_len=48,
            batch_size=1,
            dtype=None,
        )


def test_diffusion_loss_cost_rejects_an_unpriced_schedule_fn() -> None:
    config = DiffusionLoss.Config()
    config.logsnr_fn = lambda log_t: log_snr_from_log_time_per_logtan(log_t, shift=0)
    with pytest.raises(TypeError, match="<lambda> has no cost"):
        cost(config, seq_len=48, batch_size=2, dtype=None)


def test_diffusion_loss_cost_counts_each_schedule_fn() -> None:
    """A schedule transform runs once for every sample."""
    plain = cost(DiffusionLoss.Config(), seq_len=48, batch_size=2, dtype=None)
    transformed = cost(
        DiffusionLoss.Config(time_transform=log_time_from_log_snr_per_logit),
        seq_len=48,
        batch_size=2,
        dtype=None,
    )
    own = cost(log_time_from_log_snr_per_logit, channels=1, dtype=None)
    assert transformed["flops", "primal", "elementwise"].sum() == (
        plain["flops", "primal", "elementwise"].sum()
        + own["flops", "primal", "elementwise"].sum() * 2
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_diffusion_loss_operand_traffic(dtype: torch.dtype) -> None:
    """Every payload cell scales with the itemsize; nothing is an index."""
    costed = cost(DiffusionLoss.Config(), seq_len=48, batch_size=2, dtype=dtype)
    fp32 = cost(DiffusionLoss.Config(), seq_len=48, batch_size=2, dtype=None)
    assert costed["flops"].sum() == fp32["flops"].sum()
    assert costed["bytes"].sum() == fp32["bytes"].sum() * dtype.itemsize // 4
    assert {key[-1] for key in costed.cells} == {dtype}


def _loss(
    module: nn.Module,
    *,
    denoiser: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
) -> Tensor:
    """Run the diffusion loss and return its ``loss`` tensor."""
    assert isinstance(module, DiffusionLoss)
    return module(denoiser=denoiser, x0=x0)["loss"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
