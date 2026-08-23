"""Top-K accuracy metric for classification."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from configgle import Fig
from torch import Tensor


class TopK:
    """Top-K accuracy metric.

    Computes top-1, top-5, or custom top-k accuracy.
    """

    class Config(Fig["TopK"]):
        """Top-K metric configuration."""

        k_values: list[int] = field(default_factory=lambda: [1, 5])
        """Each k reported as its own ``top{k}`` accuracy."""

    def __init__(self, config: Config) -> None:
        """Initialize metric.

        Args:
          config: Metric configuration.

        """
        if not config.k_values:
            raise ValueError("k_values must contain at least one positive integer.")
        self.k_values = config.k_values
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.correct: dict[int, int] = dict.fromkeys(self.k_values, 0)
        self.total = 0

    def update(
        self,
        logits: Tensor,
        *,
        label: object = None,
        **_kwargs: object,
    ) -> None:
        """Update metric with batch.

        Args:
          logits: Model predictions [B, C].
          label: Ground truth labels [B].

        """
        if label is None:
            raise ValueError("label must be provided")
        if not isinstance(label, Tensor):
            raise TypeError(f"label must be a Tensor, got {type(label)}.")
        targets = label
        batch_size = targets.size(0)
        num_classes = logits.size(1)
        # ``topk`` cannot request more than ``num_classes`` predictions; any
        # k beyond that is trivially perfect, so clamp the requested width.
        max_k = min(max(self.k_values), num_classes)

        # Get top-k predictions
        _, pred_topk = logits.topk(max_k, dim=1)  # [B, max_k]

        # Check correctness for each k
        for k in self.k_values:
            pred_k = pred_topk[:, : min(k, num_classes)]  # [B, min(k, C)]
            correct_k = (pred_k == targets.unsqueeze(1)).any(dim=1)
            self.correct[k] += int(correct_k.sum().item())

        self.total += int(batch_size)

    def compute(self) -> dict[str, float]:
        """Compute top-k accuracies.

        Returns:
          metrics: Dict with 'top1', 'top5', etc.

        """
        if self.total == 0:
            return {f"top{k}": 0.0 for k in self.k_values}

        return {f"top{k}": self.correct[k] / self.total for k in self.k_values}

    def state_dict(self) -> dict[str, Any]:
        """Get metric state for checkpointing."""
        return {
            "correct": self.correct,
            "total": self.total,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load metric state from checkpoint."""
        self.correct = state_dict.get("correct", dict.fromkeys(self.k_values, 0))
        self.total = state_dict.get("total", 0)
