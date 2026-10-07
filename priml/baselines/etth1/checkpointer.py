"""DLinear's best-checkpoint tie rule on PRIML's checkpoint storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from configgle import Makes

from priml.train.checkpointer import Checkpointer


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.custom_types import CheckpointableProtocol


class Etth1Checkpointer(Checkpointer):
    """Keep the latest checkpoint on an equal validation score, as DLinear does."""

    class Config(Makes["Etth1Checkpointer"], Checkpointer.Config):
        """Standard checkpoint storage with the source's inclusive tie rule."""

    @override
    def on_eval(
        self,
        target: CheckpointableProtocol,
        step: int,
        metrics: Mapping[str, float],
    ) -> bool:
        """Allow an equal score through the standard strict-improvement gate."""
        # Pending saves publish their best value when the storer flushes.
        self.storage.flush()
        previous = self.best_value
        if self.best_metric and metrics[self.best_metric] == previous:
            self.best_value = float("inf") if self.best_mode == "min" else -float("inf")
        try:
            return super().on_eval(target, step=step, metrics=metrics)
        finally:
            if self.best_value in (float("inf"), -float("inf")):
                self.best_value = previous
