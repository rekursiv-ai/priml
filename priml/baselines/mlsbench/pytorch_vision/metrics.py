"""Test accuracy as the reference reports it, beside the honest number.

The reference trains for a fixed number of epochs, evaluates the test split
after each, and reports the best accuracy seen. That picks the epoch with the
test labels, so the figure is optimistic. The accuracy after the last epoch
involves no selection; both are published so a run can be compared with the
reference and judged without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor


if TYPE_CHECKING:
    from collections.abc import Mapping


class BestAccuracy:
    """Top-1 accuracy in percent, with the best value of every evaluation so far."""

    class Config(Fig["BestAccuracy"]):
        """Takes no parameters."""

    def __init__(self, config: Config) -> None:
        del config
        self.best = 0.0
        """Highest accuracy of any completed evaluation; survives ``reset``."""
        self.reset()

    def reset(self) -> None:
        """Start a new evaluation, keeping ``best``."""
        self.correct = 0
        self.total = 0

    def update(
        self,
        logits: Tensor,
        *,
        label: object = None,
        **_kwargs: object,
    ) -> None:
        """Count the batch's correct predictions.

        Args:
          logits: Model predictions ``(B, C)``.
          label: Ground-truth classes ``(B,)``.

        """
        if not isinstance(label, Tensor):
            raise TypeError(f"label must be a Tensor, got {type(label)}.")
        self.correct += int(logits.max(1).indices.eq(label).sum())
        self.total += len(label)

    def compute(self) -> dict[str, float]:
        """Return this evaluation's accuracy and the best so far.

        Returns:
          metrics: ``final`` is this evaluation's accuracy in percent and
            ``best`` the highest of this and every earlier one.

        """
        final = 100.0 * self.correct / self.total
        self.best = max(self.best, final)
        return {"final": final, "best": self.best}

    class StateDict(TypedDict):
        """The running counts and the best accuracy."""

        correct: int
        total: int
        best: float

    def state_dict(self) -> StateDict:
        """Get metric state for checkpointing.

        Returns:
          state: The counts of the open evaluation and the best accuracy.

        """
        return {"correct": self.correct, "total": self.total, "best": self.best}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load metric state from a checkpoint.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(BestAccuracy.StateDict, state_dict)
        self.correct = state["correct"]
        self.total = state["total"]
        self.best = state["best"]
