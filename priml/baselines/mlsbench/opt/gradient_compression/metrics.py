"""Mean test cross-entropy, as the reference reports and scores it.

The reference sums each test batch's cross-entropy, takes it out as a Python
float, adds the floats, and divides by the number of images. That is not the
mean of the per-batch means the loop reports as ``total_loss`` when the last
batch is short, so the scored figure is its own metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor
from torch.nn import functional


if TYPE_CHECKING:
    from collections.abc import Mapping


class MeanCrossEntropy:
    """Cross-entropy summed over every test image, divided by their count."""

    class Config(Fig["MeanCrossEntropy"]):
        """Takes no parameters."""

    def __init__(self, config: Config) -> None:
        del config
        self.reset()

    def reset(self) -> None:
        """Start a new evaluation."""
        self.total_loss = 0.0
        self.total = 0

    def update(self, logits: Tensor, **batch: object) -> None:
        """Add the batch's summed cross-entropy and its size.

        Args:
          logits: Model predictions ``(B, C)``.
          **batch: The batch; ``label`` holds the ``(B,)`` ground-truth classes.

        """
        label = batch["label"]
        assert isinstance(label, Tensor)
        self.total_loss += functional.cross_entropy(
            logits,
            label,
            reduction="sum",
        ).item()
        self.total += len(label)

    def compute(self) -> dict[str, float]:
        """Return the mean cross-entropy of this evaluation.

        Returns:
          metrics: ``mean``, the summed loss over the image count.

        Raises:
          ValueError: No image was seen.

        """
        if not self.total:
            raise ValueError("The evaluation saw no examples.")
        return {"mean": self.total_loss / self.total}

    class StateDict(TypedDict):
        """The running sum and count."""

        total_loss: float
        total: int

    def state_dict(self) -> StateDict:
        """Get metric state for checkpointing.

        Returns:
          state: The summed loss and the count of the open evaluation.

        """
        return {"total_loss": self.total_loss, "total": self.total}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load metric state from a checkpoint.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(MeanCrossEntropy.StateDict, state_dict)
        self.total_loss = state["total_loss"]
        self.total = state["total"]
