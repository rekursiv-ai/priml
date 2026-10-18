"""Masked regression and ordered-posterior loss arithmetic."""

from torch import Tensor

import torch


def masked_mean(
    values: Tensor,
    *,
    mask: Tensor,
    weight: Tensor | None = None,
    scale: Tensor | None = None,
) -> Tensor:
    """Reduce selected values with separate normalized and unnormalized weights.

    Args:
      values: Per-observation values.
      mask: Boolean selection with the same shape as values.
      weight: Relative observation weights, included in the denominator.
      scale: Absolute observation multipliers, excluded from the denominator.

    Returns:
      mean: Weighted scalar; an empty mask or nonpositive weight gives zero.

    """
    selected = values[mask]
    if selected.numel() == 0:
        return selected.sum() * 0.0
    if weight is None:
        if scale is None:
            return selected.mean()
        return (selected * scale[mask].to(selected.dtype)).sum() / selected.new_tensor(
            selected.numel(),
        )
    selected_weight = weight[mask].to(selected.dtype)
    denominator = selected_weight.sum()
    positive = denominator > 0
    safe_denominator = torch.where(positive, denominator, 1.0)
    numerator = selected * selected_weight
    if scale is not None:
        numerator = numerator * scale[mask].to(selected.dtype)
    return torch.where(
        positive,
        numerator.sum() / safe_denominator,
        selected.sum() * 0.0,
    )


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    delta: float,
    weight: Tensor | None = None,
    scale: Tensor | None = None,
) -> Tensor:
    """Subtract before converting to float32, then reduce Huber errors.

    Args:
      prediction: Predicted observations.
      target: Target observations in prediction's shape.
      mask: Boolean observation selection.
      delta: Positive transition from quadratic to linear error.
      weight: Relative weights normalized over selected observations.
      scale: Absolute multipliers, excluded from the denominator.

    Returns:
      loss: Float32 scalar Huber error, with a connected zero for empty masks.

    """
    if weight is None and scale is None:
        difference = (prediction[mask] - target[mask]).float()
        if difference.numel() == 0:
            return prediction.float().sum() * 0.0
    else:
        difference = (prediction - target).float()
    magnitude = difference.abs()
    quadratic = torch.minimum(magnitude, torch.full_like(magnitude, delta))
    values = 0.5 * quadratic.square() + delta * (magnitude - quadratic)
    if weight is None and scale is None:
        return values.mean()
    return masked_mean(values, mask=mask, weight=weight, scale=scale)


def masked_soft_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    weight: Tensor | None = None,
    scale: Tensor | None = None,
) -> Tensor:
    """Reduce dense target cross-entropy on the last axis in float32."""
    values = -(target.float() * logits.float().log_softmax(-1)).sum(-1)
    return masked_mean(values, mask=mask, weight=weight, scale=scale)


def masked_cdf_error(
    logits: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    weight: Tensor | None = None,
    scale: Tensor | None = None,
) -> Tensor:
    """Compare cumulative posterior masses by their mean squared difference."""
    probability = logits.float().softmax(-1)
    target = target.to(device=probability.device, dtype=probability.dtype)
    values = (probability.cumsum(-1) - target.cumsum(-1)).square().mean(-1)
    return masked_mean(values, mask=mask, weight=weight, scale=scale)


def masked_expectation(
    logits: Tensor,
    values: Tensor,
    *,
    mask: Tensor,
    weight: Tensor | None = None,
    scale: Tensor | None = None,
) -> Tensor:
    """Reduce an expectation under last-axis float32 softmax probabilities."""
    expected = (logits.float().softmax(-1) * values.float()).sum(-1)
    return masked_mean(expected, mask=mask, weight=weight, scale=scale)
