"""Diffusion loss for training generative models."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    reduction_cost,
    traffic,
)
from priml.math.custom_types import TensorableFn
from priml.math.diffusion.schedule import (
    compute_log_alpha,
    log_sigma_from_log_snr_per_rectified_flow,
    log_snr_from_log_time_per_truncnormicdf,
)
from priml.math.diffusion.target import (
    TargetFn,
    target_rectified_flow,
)
from priml.math.numeric import safe_log


if TYPE_CHECKING:
    from collections.abc import Callable


class DiffusionLoss(nn.Module):
    """Diffusion loss for training generative models.

    Implements rectified flow training by default.

    Example:
        >>> loss_fn = DiffusionLoss.Config().make()
        >>> denoiser = lambda x, sigma: model(x, sigma)
        >>> x0 = torch.randn(2, 3, 8, 64, 64)
        >>> result = loss_fn(denoiser=denoiser, x0=x0)
        >>> loss = result["loss"].mean()

    """

    class Config(Fig["DiffusionLoss"]):
        logsnr_fn: TensorableFn = log_snr_from_log_time_per_truncnormicdf
        """Maps log_t to log signal-to-noise ratio."""

        target_fn: TargetFn = target_rectified_flow
        """Computes training target and prediction from model output."""

        corruption_fn: TensorableFn = log_sigma_from_log_snr_per_rectified_flow
        """Maps log_snr to log noise coefficient."""

        time_transform: TensorableFn | None = None
        """Optional transform applied to log_t before logsnr_fn."""

        snr_gamma: float = 0.0
        """Min-SNR-γ loss weighting (Hang et al. 2023). 0 disables.

        Multiplies per-sample loss by min(SNR, γ) / SNR, which clips the
        effective weight at high-SNR timesteps. γ=5 is recommended.
        Provides ~3.4× faster convergence with zero compute overhead.

        Reference: arXiv:2303.09556.
        """

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the complete diffusion loss invocation.

            Vector work runs for every element of ``x0``. Schedules and target
            coefficient preparation run once for every sample. The denoiser is
            supplied at forward time and priced by its owning training step.

            Args:
              seq_len: Elements per sample.
              batch_size: Samples per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete batch.

            Raises:
              TypeError: ``target_fn`` or a schedule function carries no
                ``cost`` (see :func:`priml.cost.costed`).

            """
            del kwargs
            elements = seq_len * batch_size
            dt = dtype
            schedules = [self.logsnr_fn, self.corruption_fn]
            if self.time_transform is not None:
                schedules.append(self.time_transform)
            per_sample = sum(
                (cost(fn, channels=1, dtype=dt) for fn in schedules),
                traffic("primal", "elementwise", elements=6, flops=5, dtype=dt),
            )
            if self.snr_gamma > 0:
                per_sample += traffic(
                    "primal",
                    "elementwise",
                    elements=14,
                    flops=5,
                    dtype=dt,
                ) + traffic("adjoint", "elementwise", elements=3, flops=1, dtype=dt)
            return (
                traffic(
                    "primal",
                    "elementwise",
                    elements=13 * elements,
                    flops=5 * elements,
                    dtype=dt,
                )
                + cost(
                    self.target_fn,
                    dtype=dt,
                    elements=elements,
                    samples=batch_size,
                )
                + reduction_cost(
                    input_elements=elements,
                    output_groups=batch_size,
                    dtype=dt,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=7 * elements,
                    flops=3 * elements,
                    dtype=dt,
                )
                + traffic(
                    "adjoint",
                    "reduction",
                    elements=elements + batch_size,
                    dtype=dt,
                )
                + per_sample.tile(batch_size)
            )

    class Output(TypedDict):
        """Output from diffusion loss."""

        loss: Tensor
        """Per-sample MSE loss, shape [B]."""

        x_denoised: Tensor
        """Model's estimate of x_original."""

        eps_denoised: Tensor
        """Model's estimate of eps_original."""

        log_snr: Tensor
        """Log signal-to-noise ratio, shape [B, 1, ...]."""

        log_sigma: Tensor
        """Log noise coefficient, shape [B, 1, ...]."""

    def __init__(self, config: Config):
        super().__init__()
        self.logsnr_fn = config.logsnr_fn
        self.target_fn = config.target_fn
        self.corruption_fn = config.corruption_fn
        self.time_transform = config.time_transform
        self.snr_gamma = config.snr_gamma

    @override
    def forward(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        x0: Tensor,
    ) -> Output:
        """Compute diffusion training loss.

        Args:
          denoiser: Model callable, `denoiser(x_noisy, sigma) -> prediction`.
          x0: Clean input tensor, shape [B, ...].

        Returns:
          loss: Per-sample MSE loss, shape [B].
          x_denoised: Model's estimate of x_original.
          eps_denoised: Model's estimate of eps_original.
          log_snr: Log signal-to-noise ratio, shape [B, 1, ...].
          log_sigma: Log noise coefficient, shape [B, 1, ...].

        """
        bdims = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        log_t = safe_log(torch.rand(bdims, device=x0.device, dtype=x0.dtype))
        if self.time_transform is not None:
            log_t = self.time_transform(log_t)
        eps0 = torch.randn_like(x0)

        log_snr = self.logsnr_fn(log_t)
        log_sigma = self.corruption_fn(log_snr)
        log_alpha = compute_log_alpha(log_snr, log_sigma)
        x_noisy = log_alpha.exp() * x0 + log_sigma.exp() * eps0

        model_output = denoiser(x_noisy, log_sigma.exp())

        target, predict, x_denoised, eps_denoised = self.target_fn(
            model=model_output,
            x_noisy=x_noisy,
            log_snr=log_snr,
            log_sigma=log_sigma,
            x_original=x0,
            eps_original=eps0,
        )
        if target is None:
            raise ValueError("Expected target is not None.")
        if predict is None:
            raise ValueError("Expected predict is not None.")

        loss = ((target - predict) ** 2).mean(
            dim=list(range(-x0.ndim + 1, 0)),
        )

        # Min-SNR-γ weighting: multiply loss by min(SNR, γ) / SNR.
        # In log-space: weight = exp(min(log_snr, log(γ)) - log_snr).
        # Clamps effective weight at high-SNR timesteps, resolving
        # conflicting gradients across timesteps.
        if self.snr_gamma > 0:
            log_snr_flat = log_snr.reshape(loss.shape[0])
            log_gamma = torch.tensor(
                self.snr_gamma,
                device=loss.device,
                dtype=loss.dtype,
            ).log()
            snr_weight = (torch.clamp(log_snr_flat, max=log_gamma) - log_snr_flat).exp()
            loss = loss * snr_weight

        return DiffusionLoss.Output(
            loss=loss,
            x_denoised=x_denoised,
            eps_denoised=eps_denoised,
            log_snr=log_snr,
            log_sigma=log_sigma,
        )
