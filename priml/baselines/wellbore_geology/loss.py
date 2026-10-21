"""The public recipe's ordered regression, alignment, and gamma-ray objective."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, cast

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor

import torch

from priml.loss.custom_types import LossOutput


if TYPE_CHECKING:
    from priml.baselines.wellbore_geology.custom_types import Batch


class MaskedLossFn(Protocol):
    """Reduce two tensor arguments under an observation mask."""

    def __call__(
        self,
        first: Tensor,
        second: Tensor,
        /,
        *,
        mask: Tensor,
        scale: Tensor | None = None,
    ) -> Tensor:
        """Return the scalar observation loss."""
        ...


class PosteriorLoss:
    """Compose the three active objective terms in their declared order."""

    class Config(Fig["PosteriorLoss"]):
        regression: Makeable[MaskedLossFn] = field(
            default_factory=lambda: PartialConfig(masked_huber, delta=1.0),
        )
        """Masked depth regression criterion."""

        alignment: Makeable[MaskedLossFn] = field(
            default_factory=lambda: PartialConfig(masked_soft_cross_entropy),
        )
        """Dense typewell posterior criterion."""

        gamma_ray: Makeable[MaskedLossFn] = field(
            default_factory=lambda: PartialConfig(masked_expectation),
        )
        """Expected gamma-ray mismatch under the predicted posterior."""

        regression_weight: float = 1 / 60
        """Regression coefficient."""

        alignment_weight: float = 1 / 15
        """Alignment coefficient."""

        gamma_ray_weight: float = 1 / 60
        """Gamma-ray coefficient."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.regression = config.regression.make()
        self.alignment = config.alignment.make()
        self.gamma_ray = config.gamma_ray.make()

    def __call__(
        self,
        prediction: Tensor,
        *,
        logits: Tensor,
        target: Tensor,
        probabilities: Tensor,
        gamma_error: Tensor,
        mask: Tensor,
        scale: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the weighted objective and all three unweighted terms.

        Args:
          prediction: Normalized depth predictions, shaped like target.
          logits: Complete last-axis typewell logits.
          target: Normalized depth labels.
          probabilities: Dense target typewell masses.
          gamma_error: Per-bin gamma-ray mismatch at every typewell candidate.
          mask: Boolean selection of target observations.
          scale: Absolute observation multipliers, excluded from normalization.

        Returns:
          loss: Weighted scalar in regression, alignment, gamma-ray order.
          details: Every unweighted component, in the same order.

        """
        regression = self.regression(prediction, target, mask=mask, scale=scale)
        alignment = self.alignment(logits, probabilities, mask=mask, scale=scale)
        gamma_ray = self.gamma_ray(logits, gamma_error, mask=mask, scale=scale)
        total = self.config.regression_weight * regression
        total = total + self.config.alignment_weight * alignment
        total = total + self.config.gamma_ray_weight * gamma_ray
        return total, {
            "regression": regression,
            "alignment": alignment,
            "GR_penalty": gamma_ray,
        }


class WellboreLoss:
    """Route the complete posterior model output into the shared loss slot."""

    class Config(Fig["WellboreLoss"]):
        objective: PosteriorLoss.Config = field(default_factory=PosteriorLoss.Config)
        """Ordered regression, alignment and gamma-ray objective."""

    def __init__(self, config: Config) -> None:
        self.objective = config.objective.make()

    def __call__(self, output: object, **preprocessed_batch: object) -> LossOutput:
        """Return the objective, once per sample, and all unweighted terms.

        Args:
          output: Complete posterior expectation and diagnostic logits.
          **preprocessed_batch: Complete tensors and original-row metadata.

        Returns:
          result: Ordered per-sample loss and every unweighted objective term.

        """
        prediction, extra = cast("tuple[Tensor, dict[str, Tensor]]", output)
        batch = cast("Batch", preprocessed_batch)
        loss, details = self.objective(
            prediction,
            logits=extra["alignment_logits"],
            target=batch["label"],
            probabilities=batch["typewell_target_probs"],
            gamma_error=batch["aux_targets"]["GR_penalty_error"],
            mask=batch["target_mask"],
        )
        return LossOutput(loss=loss.expand(batch["label"].shape[0]), **details)


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    delta: float,
    scale: Tensor | None = None,
) -> Tensor:
    """Subtract before converting to float32, then reduce Huber errors.

    Args:
      prediction: Predicted observations.
      target: Target observations in prediction's shape.
      mask: Boolean observation selection.
      delta: Positive transition from quadratic to linear error.
      scale: Absolute multipliers, excluded from the denominator.

    Returns:
      loss: Float32 scalar Huber error, with a connected zero for empty masks.

    """
    if scale is None:
        difference = (prediction[mask] - target[mask]).float()
        if difference.numel() == 0:
            return prediction.float().sum() * 0.0
    else:
        difference = (prediction - target).float()
    magnitude = difference.abs()
    quadratic = torch.minimum(magnitude, torch.full_like(magnitude, delta))
    values = 0.5 * quadratic.square() + delta * (magnitude - quadratic)
    if scale is None:
        return values.mean()
    return _masked_mean(values, mask=mask, scale=scale)


def masked_soft_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    scale: Tensor | None = None,
) -> Tensor:
    """Reduce dense target cross-entropy on the last axis in float32."""
    values = -(target.float() * logits.float().log_softmax(-1)).sum(-1)
    return _masked_mean(values, mask=mask, scale=scale)


def masked_expectation(
    logits: Tensor,
    values: Tensor,
    *,
    mask: Tensor,
    scale: Tensor | None = None,
) -> Tensor:
    """Reduce an expectation under last-axis float32 softmax probabilities."""
    expected = (logits.float().softmax(-1) * values.float()).sum(-1)
    return _masked_mean(expected, mask=mask, scale=scale)


def _masked_mean(
    values: Tensor,
    *,
    mask: Tensor,
    scale: Tensor | None,
) -> Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return selected.sum() * 0.0
    if scale is None:
        return selected.mean()
    return (selected * scale[mask].to(selected.dtype)).sum() / selected.new_tensor(
        selected.numel(),
    )
