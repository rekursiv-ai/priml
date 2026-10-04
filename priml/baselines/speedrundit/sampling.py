"""Latent and REG CLS sampling with the reference Euler-Maruyama SDE."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, Protocol, Self

from torch import Tensor

import torch

from priml.baselines.speedrundit.objective import interpolant
from priml.math.diffusion.euler_maruyama import (
    euler_maruyama_grid,
    guide_drift,
    integrate_two_streams,
    repa_diffusion,
    velocity_to_drift,
    velocity_to_score,
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
    alpha, sigma, d_alpha, d_sigma = interpolant(t, path=path)
    shape = (slice(None),) + (None,) * (noisy.ndim - 1)
    return velocity_to_score(
        velocity,
        state=noisy,
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
    """Sample diffusion-space and REG CLS latents with the reverse SDE.

    Path-drop guidance is intended for qualitative images only. The paper's
    reported quantitative metrics use ordinary sampling without it.

    Args:
      model: Velocity model supporting classifier-free guidance.
      latents: Initial noisy diffusion-space latents.
      cls_latents: Initial noisy REG CLS latents.
      labels: Conditional class labels.
      num_steps: Number of stochastic integration intervals.
      cfg_scale: Classifier-free guidance scale for the image latents.
      cls_cfg_scale: Classifier-free guidance scale for CLS latents. It acts
        only when ``cfg_scale > 1``, which is what runs the unconditional branch.
      path: Linear or cosine probability path.
      shift_time: Whether to shift times for latent dimensionality.
      shift_base: Reference latent dimension for time shifting.
      path_drop_guidance: Whether the unconditional branch drops SPRINT.
      guidance_low: Earliest flow time receiving guidance.
      guidance_high: Latest flow time receiving guidance.

    Returns:
      latents: Sampled diffusion-space latents; the step's ``latent_norm``
        denormalizes them before the autoencoder decodes.
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
            t_steps = time_shift(
                t_steps,
                latent_dimensions=latents[0].numel(),
                reference_dimensions=shift_base,
            )

        drift = partial(
            _drift,
            model=model,
            labels=labels,
            latent_dtype=latents.dtype,
            cls_dtype=cls_latents.dtype,
            cfg_scale=cfg_scale,
            cls_cfg_scale=cls_cfg_scale,
            path=path,
            path_drop_guidance=path_drop_guidance,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
        )

        x, cls = integrate_two_streams(
            latents,
            cls_token=cls_latents,
            grid=t_steps,
            drift=drift,
        )
        return x.to(latents.dtype), cls.to(cls_latents.dtype)
    finally:
        model.train(was_training)


def _drift(
    x: Tensor,
    cls: Tensor,
    t_cur: Tensor,
    *,
    model: SamplerModel,
    labels: Tensor,
    latent_dtype: torch.dtype,
    cls_dtype: torch.dtype,
    cfg_scale: float,
    cls_cfg_scale: float,
    path: Literal["linear", "cosine"],
    path_drop_guidance: bool,
    guidance_low: float,
    guidance_high: float,
) -> tuple[Tensor, Tensor]:
    """Return both SDE drifts under the requested guidance policy."""
    t = t_cur.expand(x.shape[0])
    use_cfg = cfg_scale > 1 and guidance_low <= t_cur <= guidance_high
    diffusion = repa_diffusion(t_cur)
    cond = _predict(
        model,
        x=x,
        cls=cls,
        t=t,
        labels=labels,
        latent_dtype=latent_dtype,
        cls_dtype=cls_dtype,
        drop_path=False,
    )
    drift_x, drift_cls = _drifts(
        cond,
        x=x,
        cls=cls,
        t=t,
        path=path,
        diffusion=diffusion,
    )
    if use_cfg:
        uncond = _predict(
            model,
            x=x,
            cls=cls,
            t=t,
            labels=torch.full_like(labels, model.config.num_classes),
            latent_dtype=latent_dtype,
            cls_dtype=cls_dtype,
            drop_path=path_drop_guidance,
        )
        drift_u, drift_cls_u = _drifts(
            uncond,
            x=x,
            cls=cls,
            t=t,
            path=path,
            diffusion=diffusion,
        )
        drift_x = guide_drift(drift_x, weak=drift_u, scale=cfg_scale)
        if cls_cfg_scale > 0:
            drift_cls = guide_drift(drift_cls, weak=drift_cls_u, scale=cls_cfg_scale)
    return drift_x, drift_cls


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
        t=t.to(latent_dtype),
        y=labels,
        cls_token=cls.to(cls_dtype),
        drop_sparse_path=drop_path,
        route_tokens=False,
    )


def _drifts(
    output: ModelOutput,
    *,
    x: Tensor,
    cls: Tensor,
    t: Tensor,
    path: Literal["linear", "cosine"],
    diffusion: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the latent and CLS drifts one model output implies."""
    velocity = output.velocity.double()
    cls_velocity = output.cls_velocity.double()
    score_x = score_from_velocity(velocity, noisy=x, t=t, path=path)
    score_cls = score_from_velocity(cls_velocity, noisy=cls, t=t, path=path)
    return (
        velocity_to_drift(velocity, score=score_x, diffusion=diffusion),
        velocity_to_drift(cls_velocity, score=score_cls, diffusion=diffusion),
    )
