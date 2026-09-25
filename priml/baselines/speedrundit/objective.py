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
    t: Tensor, path: Literal["linear", "cosine"]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return clean/noise coefficients and their time derivatives."""
    if path == "linear":
        return 1 - t, t, -torch.ones_like(t), torch.ones_like(t)
    if path == "cosine":
        angle = math.pi * t / 2
        return (
            angle.cos(),
            angle.sin(),
            -(math.pi / 2) * angle.sin(),
            (math.pi / 2) * angle.cos(),
        )
    raise ValueError(f"unsupported flow path: {path}")  # pyright: ignore[reportUnreachable]


def projection_loss(
    predictions: tuple[Projection, ...], teacher_features: tuple[Tensor, ...]
) -> Tensor:
    """Cosine alignment at configured dense or routed block depths."""
    if len(predictions) != len(teacher_features) or not predictions:
        raise ValueError("teacher and student projection depths must match")
    total: Tensor | float = 0.0
    for prediction, teacher_target in zip(predictions, teacher_features, strict=True):
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


@dataclass(slots=True)
class LossTerms:
    """Unreduced objective and its component losses."""

    loss: Tensor
    mean_loss: Tensor
    velocity: Tensor
    cls: Tensor
    projection: Tensor
    cfm: Tensor
    output: ModelOutput


@dataclass(slots=True)
class SpeedrunObjective:
    """Combine SiT velocity, REG alignment, CLS diffusion, and CFM."""

    path: Literal["linear", "cosine"] = "linear"
    weighting: Literal["uniform", "lognormal"] = "uniform"
    cfm_weighting: Literal["uniform", "linear"] = "uniform"
    projection_coeff: float = 0.5
    cls_coeff: float = 0.03
    cfm_coeff: float = 0.05
    shift_time: bool = True
    shift_base: int = 4096

    def sample_time(self, latents: Tensor) -> Tensor:
        """Draw and optionally shift one flow time per latent sample."""
        if self.weighting == "uniform":
            t = torch.rand(latents.shape[0], device=latents.device)
        elif self.weighting == "lognormal":
            sigma = torch.randn(latents.shape[0], device=latents.device).exp()
            t = (
                sigma / (1 + sigma)
                if self.path == "linear"
                else 2 * sigma.atan() / math.pi
            )
        else:
            raise ValueError(f"unsupported timestep weighting: {self.weighting}")
        if self.shift_time:
            t = time_shift(t, latents[0].numel(), self.shift_base)
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
        """Corrupt latents and CLS features, run the model, and score losses."""
        if latents.shape[0] != labels.shape[0]:
            raise ValueError("labels and latents have different batch sizes")
        if not teacher_features:
            raise ValueError("REG requires at least one teacher feature map")
        cls_clean = teacher_features[-1][:, 0]
        time = self.sample_time(latents) if time is None else time
        noise = torch.randn_like(latents) if noise is None else noise
        cls_noise = torch.randn_like(cls_clean) if cls_noise is None else cls_noise
        alpha, sigma, d_alpha, d_sigma = interpolant(time, self.path)
        broadcast = (slice(None),) + (None,) * (latents.ndim - 1)
        noisy = alpha[broadcast] * latents + sigma[broadcast] * noise
        target = d_alpha[broadcast] * latents + d_sigma[broadcast] * noise
        cls_noisy = alpha[:, None] * cls_clean + sigma[:, None] * cls_noise
        cls_target = d_alpha[:, None] * cls_clean + d_sigma[:, None] * cls_noise
        output = model(noisy, time, labels, cls_noisy)
        velocity = (output.velocity - target).square().mean(dim=(1, 2, 3))
        cls = (output.cls_velocity - cls_target).square().mean(dim=1)
        projection = projection_loss(output.projections, teacher_features)
        cfm = contrastive_flow_loss(
            output.velocity,
            target,
            weight=(
                time.reshape(-1, *([1] * (target.ndim - 1)))
                if self.cfm_weighting == "linear"
                else None
            ),
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
        return LossTerms(loss, mean_loss, velocity, cls, projection, cfm, output)
