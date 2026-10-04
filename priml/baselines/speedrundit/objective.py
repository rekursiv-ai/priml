"""Velocity, REG representation alignment, CLS diffusion, and CFM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import math

from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.speedrundit.model import ModelOutput, Projection
from priml.loss.contrastive_flow import contrastive_flow_loss
from priml.math.diffusion.time_shift import time_shift


if TYPE_CHECKING:
    from collections.abc import Callable


def interpolant(
    t: Tensor,
    path: Literal["linear", "cosine"],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute probability-path coefficients and derivatives.

    Args:
      t: Flow times, one per sample.
      path: Linear or cosine probability path.

    Returns:
      alpha: Clean-data coefficients.
      sigma: Noise coefficients.
      d_alpha: Time derivatives of clean-data coefficients.
      d_sigma: Time derivatives of noise coefficients.

    """
    if path == "linear":
        return 1 - t, t, -torch.ones_like(t), torch.ones_like(t)
    angle = math.pi * t / 2
    return (
        angle.cos(),
        angle.sin(),
        -(math.pi / 2) * angle.sin(),
        (math.pi / 2) * angle.cos(),
    )


def projection_loss(
    predictions: tuple[Projection, ...],
    teacher_features: tuple[Tensor, ...],
) -> Tensor:
    """Compute cosine alignment at configured projection depths.

    Args:
      predictions: Student projections and retained token indices.
      teacher_features: Teacher token features at matching depths.

    Returns:
      loss: Per-sample sum of mean token alignment losses.

    Raises:
      ValueError: Projection counts or token shapes do not match.

    """
    if len(predictions) != len(teacher_features) or not predictions:
        raise ValueError("teacher and student projection depths must match")
    total: Tensor | float = 0.0
    for index, prediction in enumerate(predictions):
        teacher_target = teacher_features[index]
        selected = teacher_target
        if prediction.ids_keep is not None:
            selected = teacher_target.gather(
                1,
                prediction.ids_keep[..., None].expand(-1, -1, teacher_target.shape[-1]),
            )
        if prediction.tokens.shape != selected.shape:
            raise ValueError("teacher tokens do not match the student projection")
        total = total + (
            -(
                functional.normalize(prediction.tokens, dim=-1)
                * functional.normalize(selected, dim=-1)
            ).sum(dim=-1)
        ).mean(dim=1)
    assert isinstance(total, Tensor)
    return total / len(predictions)


@dataclass(frozen=True, slots=True, kw_only=True)
class LossTerms:
    """Unreduced objective and its component losses.

    Attributes:
      loss: Per-sample weighted sum of the four terms.
      mean_loss: Batch mean of ``loss``, the scalar the step backpropagates.
      velocity: Per-sample latent velocity error.
      cls: Per-sample CLS velocity error.
      projection: Per-sample REG alignment loss.
      cfm: Contrastive flow loss over the batch.
      output: The model output the terms were scored on.

    """

    loss: Tensor
    mean_loss: Tensor
    velocity: Tensor
    cls: Tensor
    projection: Tensor
    cfm: Tensor
    output: ModelOutput


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeedrunObjective:
    """Combine SiT velocity, REG alignment, CLS diffusion, and CFM.

    Attributes:
      path: Probability path that corrupts latents and CLS features.
      weighting: Flow-time distribution: uniform, or lognormal in sigma.
      cfm_weighting: Contrastive flow term weighted uniformly or by flow time.
      projection_coeff: REG projection loss weight.
      cls_coeff: CLS flow loss weight.
      cfm_coeff: Contrastive flow loss weight.
      shift_time: Shift sampled times toward noise for larger latents.
      shift_base: Reference latent dimension for time shifting.

    """

    path: Literal["linear", "cosine"] = "linear"
    weighting: Literal["uniform", "lognormal"] = "uniform"
    cfm_weighting: Literal["uniform", "linear"] = "uniform"
    projection_coeff: float = 0.5
    cls_coeff: float = 0.03
    cfm_coeff: float = 0.05
    shift_time: bool = True
    shift_base: int = 4096

    def sample_time(self, latents: Tensor) -> Tensor:
        """Draw and optionally shift one flow time per latent sample.

        Args:
          latents: Clean latent batch whose shape determines time shifting.

        Returns:
          time: One sampled flow time per batch element.

        """
        if self.weighting == "uniform":
            t = torch.rand(latents.shape[0], device=latents.device)
        else:
            sigma = torch.randn(latents.shape[0], device=latents.device).exp()
            t = (
                sigma / (1 + sigma)
                if self.path == "linear"
                else 2 * sigma.atan() / math.pi
            )
        if self.shift_time:
            t = time_shift(
                t,
                latent_dimensions=math.prod(latents.shape[1:]),
                reference_dimensions=self.shift_base,
            )
        return t

    def __call__(
        self,
        model: Callable[..., ModelOutput],
        latents: Tensor,
        labels: Tensor,
        teacher_features: tuple[Tensor, ...],
        *,
        time: Tensor | None = None,
        noise: Tensor | None = None,
        cls_noise: Tensor | None = None,
    ) -> LossTerms:
        """Corrupt latents and CLS features, run the model, and score losses.

        Args:
          model: Called positionally as ``model(noisy, time, labels, cls_noisy)``.
          latents: ``[B, ...]`` clean diffusion-space latents.
          labels: ``[B]`` class labels.
          teacher_features: Teacher token features per projection depth; the
            last one's first token is the clean CLS feature.
          time: ``[B]`` flow times; ``None`` draws them with :meth:`sample_time`.
          noise: Latent noise; ``None`` draws standard normal noise.
          cls_noise: CLS noise; ``None`` draws standard normal noise.

        Returns:
          terms: The unreduced objective, its components, and the model output.

        Raises:
          ValueError: ``labels`` and ``latents`` differ in batch size, or no
            teacher feature map is given.

        """
        if latents.shape[0] != labels.shape[0]:
            raise ValueError("labels and latents have different batch sizes")
        if not teacher_features:
            raise ValueError("REG requires at least one teacher feature map")
        cls_clean = teacher_features[-1][:, 0]
        time = self.sample_time(latents) if time is None else time
        noise = torch.randn_like(latents) if noise is None else noise
        cls_noise = torch.randn_like(cls_clean) if cls_noise is None else cls_noise
        alpha, sigma, d_alpha, d_sigma = interpolant(time, path=self.path)
        broadcast = (slice(None),) + (None,) * (latents.ndim - 1)
        noisy = alpha[broadcast] * latents + sigma[broadcast] * noise
        target = d_alpha[broadcast] * latents + d_sigma[broadcast] * noise
        cls_noisy = alpha[:, None] * cls_clean + sigma[:, None] * cls_noise
        cls_target = d_alpha[:, None] * cls_clean + d_sigma[:, None] * cls_noise
        output = model(noisy, time, labels, cls_noisy)
        velocity = (output.velocity - target).square().mean(dim=(1, 2, 3))
        cls = (output.cls_velocity - cls_target).square().mean(dim=1)
        projection = projection_loss(
            output.projections,
            teacher_features=teacher_features,
        )
        cfm = contrastive_flow_loss(
            output.velocity,
            target=target,
            weight=time[broadcast] if self.cfm_weighting == "linear" else None,
        )
        mean_loss = (
            velocity.mean()
            + self.projection_coeff * projection.mean()
            + self.cls_coeff * cls.mean()
            + self.cfm_coeff * cfm
        )
        loss = (
            velocity
            + self.projection_coeff * projection
            + self.cls_coeff * cls
            + self.cfm_coeff * cfm
        )
        return LossTerms(
            loss=loss,
            mean_loss=mean_loss,
            velocity=velocity,
            cls=cls,
            projection=projection,
            cfm=cfm,
            output=output,
        )
