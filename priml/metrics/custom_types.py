"""Custom types for metrics module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from priml.custom_types import CheckpointableProtocol


if TYPE_CHECKING:
    from torch import Tensor


__all__ = [
    "MetricProtocol",
]


@runtime_checkable
class MetricProtocol(CheckpointableProtocol, Protocol):
    """Protocol for metrics.

    Extends CheckpointableProtocol to support saving accumulated state
    (e.g., quantiles, confusion matrices, running averages).
    """

    def update(self, logits: Tensor, **batch: object) -> None:
        """Update metric state with batch.

        Args:
          logits: Model predictions (e.g., [B, C] for classification).
          **batch: Full batch dict. Metric extracts needed keys (e.g., 'label', 'target').

        """
        ...

    def compute(self) -> dict[str, Any]:
        """Compute final metric values.

        Returns:
          metrics: Dictionary of metric name -> value. Scalar values are logged
            and written to final metrics; non-scalar values are available to
            artifact writers.

        """
        ...

    def reset(self) -> None:
        """Reset metric state."""
        ...
