"""Contrastive flow matching loss on unrelated targets within a batch."""

from __future__ import annotations

from torch import Tensor

import torch


def contrastive_flow_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
) -> Tensor:
    """Negative MSE against targets rolled by one sample.

    ``target`` and ``weight`` may broadcast against ``prediction``.
    A one-sample batch remains defined, though it has no unrelated target.

    Args:
      prediction: Predicted flow values.
      target: Target values whose batch is rolled by one sample.
      weight: Optional broadcastable element weights.

    Returns:
      loss: Negative mean squared error against rolled targets.

    """
    difference = (prediction - torch.roll(target, 1, 0)) ** 2
    if weight is not None:
        difference = difference * weight
    return -difference.mean()
