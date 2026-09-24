"""Generation: integrate the learned velocity from noise back to data.

The reference's own sampler, ``euler_maruyama_sampler_path_drop`` at the pinned
commit -- the one its FID is measured with. Euler--Maruyama over the reverse
SDE ``dx = [v - g(t)^2 / 2 * score] dt + g(t) dW``, with the score recovered
from the predicted velocity through the SAME path the objective trained on,
and a final deterministic step that lands on ``t = 0``.

Not :func:`priml.math.diffusion.ddpm_ddim`. That is the DDPM posterior step,
which discretizes the reverse SDE whose diffusion is ``2t / (1 - t)`` under
the straight path, where this one's is ``2t``: both sample the same marginals
in the continuous limit, and at any finite step count they are different
samplers producing different images. Reproducing the reported numbers needs
this one, and ``math.diffusion`` carries no Euler--Maruyama step, nor a score
recovered from a velocity under an arbitrary path.

What is written here is also the loop, because this model carries two coupled
streams: the latent and the class token share a time grid and each feeds the
other's velocity, so they advance together.

References:
  https://github.com/SwayStar123/SpeedrunDiT
    ``samplers.py``, pinned at c24c2ff25699cce63174ca56c2afcfeeb225e367.
  https://arxiv.org/abs/2401.08740
    Ma et al. 2024, "SiT", Section 4 on SDE sampling.
  https://arxiv.org/abs/2410.06940
    Yu et al. 2024, "REPA", whose sampler this is.
  https://arxiv.org/abs/2510.21986
    Park et al. 2025, "SPRINT", whose path-drop guidance the weak branch is.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, NamedTuple, Self, override

import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor

import torch

from priml.baselines.speedrundit.loss import (
    InterpolantFn,
    TimeTransformFn,
    as_callable,
    linear_path,
    resolution_time_shift,
)
from priml.math.custom_types import TensorFn


if TYPE_CHECKING:
    from priml.baselines.speedrundit.model import SpeedrunDiT


__all__ = ["EulerMaruyamaSampler", "repa_diffusion", "velocity_to_score"]


def repa_diffusion(t: Tensor) -> Tensor:
    """Squared diffusion coefficient ``g(t)^2 = 2t`` of the reference's SDE.

    Args:
      t: Flow times.

    Returns:
      diffusion: ``2 * t``.

    """
    return 2 * t


def velocity_to_score(
    velocity: Tensor,
    state: Tensor,
    path: InterpolantFn,
    t: Tensor,
) -> Tensor:
    """Recover the score of ``p_t`` from a predicted velocity.

    Args:
      velocity: Predicted ``d x_t / d t``.
      state: The noisy sample ``x_t``.
      path: The probability path the velocity was trained against.
      t: Flow times, broadcastable to ``state``.

    Returns:
      score: ``grad log p_t(x_t)``, shaped like ``state``.

    Derivation:
      With ``x_t = alpha x + sigma eps``, the velocity is
      ``v = d_alpha x + d_sigma eps`` and the score is ``-E[eps | x_t] / sigma``.
      Eliminating ``x`` between the two with ``r = alpha / d_alpha``,

          r v - x_t = (r d_sigma - sigma) eps

      so ``E[eps | x_t] = (r v - x_t) / (r d_sigma - sigma)`` and

          score = (r v - x_t) / (sigma^2 - r d_sigma sigma).

      Spelled exactly as the reference spells it; a rearrangement lands on
      different last bits.

    """
    coefficients = path(t)
    ratio = coefficients.alpha / coefficients.d_alpha
    variance = coefficients.sigma**2 - ratio * coefficients.d_sigma * coefficients.sigma
    return (ratio * velocity - state) / variance


class EulerMaruyamaSampler:
    """Integrates the learned velocity from noise to data."""

    class Output(NamedTuple):
        """One sampling run."""

        media: Tensor
        """Generated latents, ``[batch, channels, size, size]``."""

        cls_token: Tensor
        """Generated class features, ``[batch, channels_cls]``."""

    class Config(Fig["EulerMaruyamaSampler"]):
        """Configuration for EulerMaruyamaSampler."""

        _: KW_ONLY

        num_steps: int = 250
        """Model evaluations per stream; the reported FID is measured at 250."""

        last_time: float = 0.04
        """Where the stochastic steps stop. One deterministic step then lands
        on zero, where the diffusion ``2t`` would otherwise inject noise."""

        interpolant: InterpolantFn = linear_path
        """The path the score is recovered through; the objective's own."""

        time_transform: Makeable[TimeTransformFn] | TimeTransformFn | None = field(
            default_factory=lambda: PartialConfig(resolution_time_shift, base=4096),
        )
        """Reparameterizes the time grid; ``None`` leaves it uniform. Must
        match the objective's, or the sampler walks a curve the model was
        never fit to."""

        diffusion: TensorFn = repa_diffusion
        """Squared diffusion coefficient ``g(t)^2`` of the reverse SDE."""

        guidance: float = 1.0
        """Guidance strength on both streams; guidance runs only above one.

        The reference exposes a separate class-token strength, and every
        published invocation sets it equal to this one."""

        guidance_interval: tuple[float, float] = (0.0, 1.0)
        """Times, inclusive, at which guidance applies."""

        @override
        def finalize(self) -> Self:
            if self.num_steps < 1:
                raise ValueError(f"num_steps must be positive; got {self.num_steps}.")
            if (
                math.isnan(self.last_time)
                or self.last_time <= 0.0
                or self.last_time >= 1.0
            ):
                raise ValueError(f"last_time must be in (0, 1); got {self.last_time}.")
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.time_transform: TimeTransformFn | None = (
            None
            if config.time_transform is None
            else as_callable(config.time_transform)
        )

    def times(self, shape: tuple[int, ...]) -> Tensor:
        """Build the descending time grid.

        Float64 and on the CPU, as the reference builds it: every step's
        ``dt`` and noise scale are read off this grid.

        Args:
          shape: Shape of one latent, without the batch axis.

        Returns:
          times: ``[num_steps + 1]``, from one down to exactly zero.

        """
        cfg = self.config
        grid = torch.linspace(1.0, cfg.last_time, cfg.num_steps, dtype=torch.float64)
        grid = torch.cat([grid, grid.new_zeros(1)])
        if self.time_transform is not None:
            grid = self.time_transform(grid, shape=shape)
        return grid

    @torch.no_grad()
    def __call__(
        self,
        model: SpeedrunDiT,
        media: Tensor,
        label: Tensor,
        cls_token: Tensor,
    ) -> Output:
        """Integrate from noise to data.

        Args:
          model: The trained velocity field, in eval mode.
          media: Initial latent noise, ``[batch, channels, size, size]``.
          label: Class indices to condition on, ``[batch]``.
          cls_token: Initial class-token noise, ``[batch, channels_cls]``.

        Returns:
          output: The integrated latent and class token, in their input dtypes.

        Raises:
          ValueError: If the model is training, whose label dropout and token
            routing would be sampled from; or if guidance is asked of a model
            built without the null class.

        """
        if model.training:
            raise ValueError("Sample from a model in eval mode.")
        null = None
        if self.config.guidance > 1.0:
            labels = model.y_embedder
            if labels.dropout <= 0:
                raise ValueError(
                    "Guidance needs the null class, which dropout 0 omits.",
                )
            null = torch.full_like(label, labels.num_classes)
        grid = self.times(tuple(media.shape[1:]))
        # Carried in float64 and handed to the model in its own dtype, as the
        # reference does, so the accumulation over hundreds of steps is not
        # the model's precision.
        latent, cls = media.to(torch.float64), cls_token.to(torch.float64)
        last = grid.shape[0] - 2
        for index in range(last + 1):
            t_curr, t_next = grid[index], grid[index + 1]
            drift, drift_cls = self._drift(
                model,
                latent,
                cls,
                t_curr=t_curr,
                label=label,
                null=null,
                dtype=media.dtype,
            )
            dt = t_next - t_curr
            if index == last:
                latent = latent + dt * drift
                cls = cls + dt * drift_cls
                break
            # Drawn after the forwards, latent before class, as the reference
            # draws them; the model draws nothing in eval.
            noise = torch.randn_like(latent) * torch.sqrt(torch.abs(dt))
            noise_cls = torch.randn_like(cls) * torch.sqrt(torch.abs(dt))
            scale = torch.sqrt(self.config.diffusion(t_curr))
            latent = latent + drift * dt + scale * noise
            cls = cls + drift_cls * dt + scale * noise_cls
        return EulerMaruyamaSampler.Output(
            media=latent.to(media.dtype),
            cls_token=cls.to(cls_token.dtype),
        )

    # Guidance mixes DRIFTS, not velocities, and its weak branch is the null class with
    # the sparse path dropped -- two forwards, not one doubled batch, because the
    # branches take different routes.
    def _drift(
        self,
        model: SpeedrunDiT,
        latent: Tensor,
        cls: Tensor,
        *,
        t_curr: Tensor,
        label: Tensor,
        null: Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate both streams' drifts at one time, guided when asked."""
        cfg = self.config
        low, high = cfg.guidance_interval
        in_interval = low <= float(t_curr) <= high
        batch = latent.shape[0]
        time = torch.ones(batch, dtype=torch.float64, device=latent.device) * t_curr
        diffusion = cfg.diffusion(t_curr)
        path = cfg.interpolant
        inputs = (latent.to(dtype), time.to(dtype))
        strong = model(*inputs, label, cls.to(dtype))
        drift_strong = _sde_drift(
            strong.velocity,
            state=latent,
            time=time,
            path=path,
            diffusion=diffusion,
        )
        drift_strong_cls = _sde_drift(
            strong.cls_velocity,
            state=cls,
            time=time,
            path=path,
            diffusion=diffusion,
        )
        if null is None or not in_interval:
            return drift_strong, drift_strong_cls
        weak = model(*inputs, null, cls.to(dtype), uncond=True)
        drift_weak = _sde_drift(
            weak.velocity,
            state=latent,
            time=time,
            path=path,
            diffusion=diffusion,
        )
        drift_weak_cls = _sde_drift(
            weak.cls_velocity,
            state=cls,
            time=time,
            path=path,
            diffusion=diffusion,
        )
        weight = cfg.guidance
        return (
            drift_weak + weight * (drift_strong - drift_weak),
            drift_weak_cls + weight * (drift_strong_cls - drift_weak_cls),
        )


def _sde_drift(
    velocity: Tensor,
    state: Tensor,
    *,
    time: Tensor,
    path: InterpolantFn,
    diffusion: Tensor,
) -> Tensor:
    """Reverse-SDE drift ``v - g(t)^2 / 2 * score`` of one stream, in float64."""
    velocity = velocity.to(torch.float64)
    t = time.view(-1, *[1] * (state.ndim - 1))
    score = velocity_to_score(velocity, state, path, t)
    return velocity - 0.5 * diffusion * score
