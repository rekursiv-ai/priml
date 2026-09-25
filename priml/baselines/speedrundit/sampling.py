"""Latent and REG CLS sampling with the reference Euler-Maruyama SDE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Self

from torch import Tensor

import torch

from priml.baselines.speedrundit.objective import interpolant
from priml.math.diffusion import euler_maruyama
from priml.math.diffusion.euler_maruyama import (
    euler_maruyama_grid,
    guide_drift,
    integrate_two_streams,
    repa_diffusion,
    velocity_to_drift,
)
from priml.math.diffusion.time_shift import time_shift


if TYPE_CHECKING:
    from priml.baselines.speedrundit.model import ModelOutput


class SamplerConfig(Protocol):
    """Configuration values needed by classifier-free guidance."""

    num_classes: int


class SamplerModel(Protocol):
    """Model operations needed by latent sampling."""

    @property
    def config(self) -> SamplerConfig:
        """Return classifier-free guidance configuration."""
        ...

    @property
    def training(self) -> bool:
        """Return whether training mode is enabled."""
        ...

    def eval(self) -> Self:
        """Enable evaluation mode."""
        ...

    def train(self, mode: bool = True) -> Self:
        """Set training mode."""
        ...

    def __call__(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        cls_token: Tensor,
        *,
        drop_sparse_path: bool,
        route_tokens: bool,
    ) -> ModelOutput:
        """Predict latent and CLS velocities."""
        ...


def score_from_velocity(
    velocity: Tensor,
    noisy: Tensor,
    t: Tensor,
    path: Literal["linear", "cosine"],
) -> Tensor:
    """Convert interpolant velocity to the corresponding score.

    Args:
      velocity: Predicted probability-path velocity.
      noisy: Current noisy state.
      t: Flow time for each sample.
      path: Linear or cosine probability path.

    Returns:
      score: Score of the noisy-state distribution.

    """
    alpha, sigma, d_alpha, d_sigma = interpolant(t, path)
    shape = (slice(None),) + (None,) * (noisy.ndim - 1)
    return euler_maruyama.velocity_to_score(
        velocity,
        noisy,
        alpha=alpha[shape],
        sigma=sigma[shape],
        d_alpha=d_alpha[shape],
        d_sigma=d_sigma[shape],
    )


@torch.no_grad()
def sample_latents(
    model: SamplerModel,
    latents: Tensor,
    cls_latents: Tensor,
    labels: Tensor,
    *,
    num_steps: int = 250,
    cfg_scale: float = 1.0,
    cls_cfg_scale: float = 1.0,
    path: Literal["linear", "cosine"] = "linear",
    shift_time: bool = True,
    shift_base: int = 4096,
    path_drop_guidance: bool = False,
    guidance_low: float = 0.0,
    guidance_high: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Sample raw INVAE and REG CLS latents with the reverse SDE.

    Path-drop guidance is intended for qualitative images only. The paper's
    reported quantitative metrics use ordinary sampling without it.

    Args:
      model: Velocity model supporting classifier-free guidance.
      latents: Initial noisy INVAE latents.
      cls_latents: Initial noisy REG CLS latents.
      labels: Conditional class labels.
      num_steps: Number of stochastic integration intervals.
      cfg_scale: Classifier-free guidance scale for INVAE latents.
      cls_cfg_scale: Classifier-free guidance scale for CLS latents.
      path: Linear or cosine probability path.
      shift_time: Whether to shift times for latent dimensionality.
      shift_base: Reference latent dimension for time shifting.
      path_drop_guidance: Whether the unconditional branch drops SPRINT.
      guidance_low: Earliest flow time receiving guidance.
      guidance_high: Latest flow time receiving guidance.

    Returns:
      latents: Sampled INVAE latents; divide by 0.3099 before decoding.
      cls_latents: Sampled REG CLS latents.

    Raises:
      ValueError: Fewer than two stochastic steps were requested.

    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least two")
    was_training = model.training
    model.eval()
    try:
        t_steps = euler_maruyama_grid(num_steps, device=latents.device)
        if shift_time:
            t_steps = time_shift(t_steps, latents[0].numel(), shift_base)

        def drift(x: Tensor, cls: Tensor, t_cur: Tensor) -> tuple[Tensor, Tensor]:
            t = t_cur.expand(x.shape[0])
            use_cfg = cfg_scale > 1 and guidance_low <= t_cur <= guidance_high
            cond = _predict(
                model,
                x,
                cls,
                t,
                labels,
                latent_dtype=latents.dtype,
                cls_dtype=cls_latents.dtype,
                drop_path=False,
            )
            score_x = score_from_velocity(cond.velocity.double(), x, t, path)
            score_cls = score_from_velocity(cond.cls_velocity.double(), cls, t, path)
            diffusion = repa_diffusion(t_cur)
            drift_x = velocity_to_drift(cond.velocity.double(), score_x, diffusion)
            drift_cls = velocity_to_drift(
                cond.cls_velocity.double(),
                score_cls,
                diffusion,
            )
            if use_cfg:
                null = torch.full_like(labels, model.config.num_classes)
                uncond = _predict(
                    model,
                    x,
                    cls,
                    t,
                    null,
                    latent_dtype=latents.dtype,
                    cls_dtype=cls_latents.dtype,
                    drop_path=path_drop_guidance,
                )
                score_u = score_from_velocity(uncond.velocity.double(), x, t, path)
                score_cls_u = score_from_velocity(
                    uncond.cls_velocity.double(),
                    cls,
                    t,
                    path,
                )
                drift_u = velocity_to_drift(
                    uncond.velocity.double(),
                    score_u,
                    diffusion,
                )
                drift_cls_u = velocity_to_drift(
                    uncond.cls_velocity.double(),
                    score_cls_u,
                    diffusion,
                )
                drift_x = guide_drift(drift_x, drift_u, cfg_scale)
                if cls_cfg_scale > 0:
                    drift_cls = guide_drift(drift_cls, drift_cls_u, cls_cfg_scale)
            return drift_x, drift_cls

        x, cls = integrate_two_streams(latents, cls_latents, t_steps, drift)
        return x.to(latents.dtype), cls.to(cls_latents.dtype)
    finally:
        model.train(was_training)


def _predict(
    model: SamplerModel,
    x: Tensor,
    cls: Tensor,
    t: Tensor,
    labels: Tensor,
    *,
    latent_dtype: torch.dtype,
    cls_dtype: torch.dtype,
    drop_path: bool,
) -> ModelOutput:
    return model(
        x.to(latent_dtype),
        t.to(latent_dtype),
        labels,
        cls.to(cls_dtype),
        drop_sparse_path=drop_path,
        route_tokens=False,
    )
