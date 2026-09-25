"""Contrastive flow matching loss on unrelated targets within a batch."""

from __future__ import annotations

from torch import Tensor

import torch


def contrastive_flow_loss(
    prediction: Tensor, target: Tensor, *, weight: Tensor | None = None
) -> Tensor:
    """Negative MSE against targets rolled by one sample.

    ``target`` and ``weight`` may broadcast against ``prediction``.
    A one-sample batch remains defined, though it has no unrelated target.
    """
    difference = (prediction - torch.roll(target, 1, 0)) ** 2
    if weight is not None:
        difference = difference * weight
    return -difference.mean()
