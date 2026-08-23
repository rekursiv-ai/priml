"""Simple wrapper for PyTorch functional losses."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import functools

from configgle import Fig
from torch.nn import functional

from priml.loss.custom_types import LossOutput, SimpleLossFn


if TYPE_CHECKING:
    from typing import Any

    from torch import Tensor


class SimpleLoss:
    """Simple loss wrapper for PyTorch functional losses.

    Extracts target from batch kwargs and calls PyTorch loss function.
    Returns dict format for composability with other losses.

    Example:
      # Binary classification (default)
      cfg = SimpleLoss.Config()
      loss = cfg.make()

      # Multi-class classification
      cfg = SimpleLoss.Config(
          loss_fn=functional.cross_entropy,
          kwargs={"label_smoothing": 0.1},
      )
      loss = cfg.make()

      # In training
      prediction = model(**batch)
      result = loss(prediction, **batch)  # Returns {"loss": tensor}

    """

    class Config(Fig["SimpleLoss"]):
        """SimpleLoss configuration."""

        loss_fn: SimpleLossFn = functional.binary_cross_entropy_with_logits
        """PyTorch loss function to use."""
        target_key: str = "label"
        """Batch key containing the target tensor."""
        kwargs: dict[str, Any] = field(
            default_factory=lambda: {"reduction": "none"},
        )
        """Extra keyword arguments passed to loss_fn."""

    def __init__(self, config: Config) -> None:
        self.target_key = config.target_key
        self._loss_fn = functools.partial(config.loss_fn, **config.kwargs)

    def __call__(
        self,
        prediction: Tensor,
        **batch: object,
    ) -> LossOutput:
        """Compute loss from model prediction and batch.

        Args:
          prediction: Model output (logits, predictions, etc).
          **batch: Batch data containing target.

        Returns:
          result: Dict with 'loss' key.

        """
        if self.target_key not in batch:
            raise KeyError(
                f"Target key '{self.target_key}' not found in batch. "
                f"Available keys: {list(batch.keys())}",
            )

        target = batch[self.target_key]
        loss = self._loss_fn(prediction, target)
        return {"loss": loss}
