"""Reference DLinear validation reduction for ETTh1."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor

import numpy as np
import torch


if TYPE_CHECKING:
    from collections.abc import Mapping


class ForecastMSE:
    """Average float32 batch losses using the source's NumPy reduction order."""

    class Config(Fig["ForecastMSE"]):
        """Reference validation MSE has no additional settings."""

    def __init__(self, config: Config) -> None:
        del config
        self.losses: list[float] = []

    def update(self, logits: Tensor, **batch: object) -> None:
        """Record a batch's mean squared error without building a graph."""
        target = batch["label"]
        assert isinstance(target, Tensor)
        with torch.no_grad():
            self.losses.append(
                torch.nn.functional.mse_loss(logits, target=target).item(),
            )

    def compute(self) -> dict[str, float]:
        """Return the exact reference reduction, overriding generic total_loss."""
        if not self.losses:
            raise ValueError("Cannot compute validation MSE without complete batches.")
        return {"total_loss": float(np.asarray(self.losses, dtype=np.float32).mean())}

    def reset(self) -> None:
        """Start a new validation pass."""
        self.losses.clear()

    class StateDict(TypedDict):
        """Accumulated batch losses."""

        losses: list[float]

    def state_dict(self) -> StateDict:
        """Snapshot the accumulation."""
        return {"losses": list(self.losses)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the accumulation."""
        self.losses = list(cast(ForecastMSE.StateDict, state_dict)["losses"])
