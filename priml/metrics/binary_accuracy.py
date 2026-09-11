"""Binary accuracy metric for binary classification."""

from __future__ import annotations

from typing import Any

from configgle import Fig
from torch import Tensor

import torch


class BinaryAccuracy:
    """Binary accuracy metric.

    Computes accuracy for binary classification from logits.
    """

    class Config(Fig["BinaryAccuracy"]):
        """Binary accuracy metric configuration."""

        threshold: float = 0.5
        """Score at or above which a prediction counts as the positive class."""

    def __init__(self, config: Config) -> None:
        """Initialize metric.

        Args:
          config: Metric configuration.

        """
        self.threshold = config.threshold
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.correct = 0
        self.total = 0

    def update(self, logits: Tensor, **batch: object) -> None:
        """Update metric with batch.

        Args:
          logits: Model predictions [B] (pre-sigmoid).
          **batch: Batch dict. Must contain 'label' key with ground truth labels [B] (0 or 1).

        """
        targets = batch["label"]
        assert isinstance(targets, Tensor)
        batch_size = targets.size(0)
        # Drop a trailing singleton channel so [B, 1] logits compare elementwise
        # against [B] targets instead of broadcasting to [B, B].
        predictions = (torch.sigmoid(logits.squeeze(-1)) > self.threshold).to(
            targets.dtype,
        )
        self.correct += int((predictions == targets).sum().item())
        self.total += int(batch_size)

    def compute(self) -> dict[str, float]:
        """Compute binary accuracy.

        Returns:
          metrics: Dict with 'accuracy' key.

        """
        if self.total == 0:
            return {"accuracy": 0.0}

        return {"accuracy": self.correct / self.total}

    def state_dict(self) -> dict[str, Any]:
        """Get metric state for checkpointing."""
        return {
            "correct": self.correct,
            "total": self.total,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load metric state from checkpoint."""
        self.correct = state_dict.get("correct", 0)
        self.total = state_dict.get("total", 0)
