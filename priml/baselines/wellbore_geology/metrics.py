"""Pooled row RMSE for the wellbore geology competition."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.baselines.wellbore_geology.data import ExpandPrediction


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


def pooled_rmse(
    target: NDArray[np.generic],
    *,
    prediction: NDArray[np.generic],
) -> float:
    """Reduce every original row in its declared order using binary64 arithmetic.

    Args:
      target: Complete original-row target depths.
      prediction: Complete aligned predictions, retaining their source precision.

    Returns:
      score: Pooled original-row root mean squared error.

    """
    errors = target.astype(np.float64) - prediction.astype(np.float64)
    return float(cast("np.float64", np.sqrt(np.mean(np.square(errors)))))


class PooledRMSE:
    """Compute one float64 reduction over every scored row, preserving order."""

    class Config(Fig["PooledRMSE"]):
        expansion: ExpandPrediction.Config = field(
            default_factory=ExpandPrediction.Config,
        )
        """Geometry expanding each batch's model bins into original suffix rows."""

    def __init__(self, config: Config) -> None:
        self.expansion = config.expansion.make()
        self.predictions: list[Tensor] = []
        self.labels: list[Tensor] = []

    def update(self, logits: Tensor, **batch: object) -> None:
        """Record every original suffix row of one batch of wells.

        Args:
          logits: Predicted normalized depth bins, one row per well.
          **batch: Row metadata ``meta`` holding each well's original-row truth.

        """
        rows = self.expansion(
            logits.detach().float().cpu().numpy(),
            metadata=cast("Sequence[dict[str, object]]", batch["meta"]),
            include_target=True,
        )
        if rows.target is None:
            raise ValueError("Expected rows.target is not None.")
        self.predictions.append(torch.from_numpy(rows.tvt))
        self.labels.append(torch.from_numpy(rows.target))

    def compute(self) -> dict[str, float]:
        """Return the pooled row RMSE over the entire evaluation split.

        Returns:
          metrics: Pooled RMSE in feet, minimized by the campaign.

        Raises:
          ValueError: The split contains no scored rows.

        """
        if sum(chunk.numel() for chunk in self.predictions) == 0:
            raise ValueError("Cannot compute RMSE without scored rows.")
        predictions = np.concatenate(
            [chunk.numpy().astype(np.float64) for chunk in self.predictions],
        )
        labels = np.concatenate(
            [chunk.numpy().astype(np.float64) for chunk in self.labels],
        )
        return {"rmse": pooled_rmse(labels, prediction=predictions)}

    def reset(self) -> None:
        """Clear accumulated prediction rows."""
        self.predictions.clear()
        self.labels.clear()

    class StateDict(TypedDict):
        """Detached evaluation rows retained for an exact checkpoint resume."""

        predictions: list[Tensor]
        labels: list[Tensor]

    def state_dict(self) -> StateDict:
        """Return detached prediction and target chunks."""
        return {"predictions": list(self.predictions), "labels": list(self.labels)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore pending rows without reducing or reordering them."""
        state = cast("PooledRMSE.StateDict", state_dict)
        self.predictions = list(state["predictions"])
        self.labels = list(state["labels"])
