"""The vision training step with a regularization term added to the loss.

The editable region of the ``dl-regularization`` script is a function of the
network, the batch, the logits and the run's configuration, whose scalar is
added to cross-entropy before every backward pass. It reads the network, which
the vision step's ``loss`` slot never sees, so here it is a slot of its own on a
subclass; the parent step is unchanged.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, override

from configgle import Makeable, PartialConfig
from torch import Tensor

from priml.baselines.mlsbench.dl.regularization.regularizers import zero
from priml.baselines.mlsbench.dl.vision.train_step import (
    LossContext,
    VisionTrainStep,
)


if TYPE_CHECKING:
    from torch import nn


class Regularizer(Protocol):
    """Maps the network and one batch's forward to a scalar added to the loss."""

    def __call__(
        self,
        model: nn.Module,
        /,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        """Return the regularization term for this update."""
        raise NotImplementedError


class RegularizedTrainStep(VisionTrainStep):
    """The vision step, minimizing its loss plus a regularization term."""

    class Config(VisionTrainStep.Config, kw_only=True):
        """The vision step's slots and the regularizer."""

        if TYPE_CHECKING:

            @override
            def make(self) -> RegularizedTrainStep:
                """Build this step."""
                ...

        regularization: Makeable[Regularizer] = field(
            default_factory=lambda: PartialConfig(zero),
        )
        """Scalar added to the loss of every update; zero adds nothing."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.regularization: Regularizer = config.regularization.make()

    @override
    def objective(
        self,
        logits: Tensor,
        *,
        media: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        """Return the loss slot's scalar plus the regularization term.

        Args:
          logits: ``(B, C)`` class scores of the batch.
          media: The batch's images, which the regularizer may read.
          label: ``(B,)`` target classes.
          context: What the loss and the regularizer may read about the run.

        Returns:
          loss: The scalar the optimizer descends.

        """
        ce = super().objective(logits, media=media, label=label, context=context)
        return ce + self.regularization(
            self.model,
            media=media,
            logits=logits,
            label=label,
            context=context,
        )
