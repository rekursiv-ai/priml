"""Cross-entropy weighted per class by the training split's class counts.

The weights are a function of the counts, and the counts are data: the
long-tailed dataset learns them when it subsets the split and hands them to the
loss through :meth:`ClassWeightedCrossEntropy.bind_class_counts`, which the
training loop's ``bind_step`` hook calls before the first update. The function
is the config slot; the counts never enter the config.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from configgle import Fig, Makeable, PartialConfig
from torch.nn import functional

from priml.baselines.mlsbench.weighting.weights import ClassWeightFn, uniform


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.pytorch_vision.train_step import LossContext
    from priml.loss.custom_types import LossOutput


@runtime_checkable
class BindsClassCounts(Protocol):
    """A loss that must learn the training split's class counts before it is called."""

    def bind_class_counts(self, class_counts: Tensor, /) -> None:
        """Receive ``(C,)`` float32 images per class."""


class ClassWeightedCrossEntropy:
    """Mean cross-entropy with one weight per class, as ``nn.CrossEntropyLoss(weight=)``.

    Raises:
      RuntimeError: When called before :meth:`bind_class_counts`.

    """

    class Config(Fig["ClassWeightedCrossEntropy"]):
        """Which weights the counts become."""

        class_weight: Makeable[ClassWeightFn] = field(
            default_factory=lambda: PartialConfig(uniform),
        )
        """Maps the per-class counts to the per-class weights."""

    def __init__(self, config: Config) -> None:
        self.class_weight: ClassWeightFn = config.class_weight.make()
        self.weight: Tensor | None = None
        """``(C,)`` weights; ``None`` until the counts are bound."""

    def bind_class_counts(self, class_counts: Tensor, /) -> None:
        """Turn ``(C,)`` float32 images per class into the weights every call uses."""
        self.weight = self.class_weight(class_counts, num_classes=len(class_counts))

    def __call__(
        self,
        logits: Tensor,
        *,
        label: Tensor,
        context: LossContext,
    ) -> LossOutput:
        """Return the weighted mean cross-entropy over the batch.

        Args:
          logits: ``(B, C)`` class scores.
          label: ``(B,)`` target classes.
          context: Unused; the loss has no schedule.

        Returns:
          loss: Scalar under ``"loss"``: ``sum_i w_{y_i} l_i / sum_i w_{y_i}``.

        Raises:
          RuntimeError: No counts were bound.

        """
        del context
        if self.weight is None:
            raise RuntimeError(
                "ClassWeightedCrossEntropy was called before bind_class_counts; "
                "its dataset must bind the training split's class counts.",
            )
        weight = self.weight.to(logits.device)
        return {"loss": functional.cross_entropy(logits, label, weight=weight)}
