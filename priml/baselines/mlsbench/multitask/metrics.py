"""Test accuracy of one head of a multi-head network.

The step hands every metric the heads' logits joined on the last axis, so a
head is a span of columns and the labels it is scored against. The reference
reports the fine head's best test accuracy; the coarse head's is published
beside it so the second task's progress is visible.
"""

from __future__ import annotations

from typing import override

from configgle import Makes
from torch import Tensor

from priml.baselines.mlsbench.pytorch_vision.metrics import BestAccuracy


class HeadAccuracy(BestAccuracy):
    """Top-1 accuracy of the logits in ``[logits_start, logits_stop)`` against ``label_key``."""

    class Config(Makes["HeadAccuracy"], BestAccuracy.Config):
        """Which columns are the head, and which batch entry holds its labels."""

        logits_start: int = 0
        """First column of the head's logits."""

        logits_stop: int = -1
        """One past the last column of the head's logits; the experiment sets it."""

        label_key: str = "label"
        """Batch entry with the head's target classes."""

    def __init__(self, config: Config) -> None:
        if config.logits_stop <= config.logits_start:
            raise ValueError(
                f"logits_stop must exceed logits_start; got [{config.logits_start}, "
                f"{config.logits_stop}).",
            )
        super().__init__(config)
        self.columns = slice(config.logits_start, config.logits_stop)
        self.label_key = config.label_key

    @override
    def update(
        self,
        logits: Tensor,
        *,
        label: object = None,
        **kwargs: object,
    ) -> None:
        """Count the batch's correct predictions of this head.

        Args:
          logits: ``(B, C)`` every head's logits, joined.
          label: The batch's ``label`` entry, read when ``label_key`` names it.
          **kwargs: The rest of the batch.

        """
        labels = kwargs | {"label": label}
        super().update(logits[..., self.columns], label=labels.get(self.label_key))
