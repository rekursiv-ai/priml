"""Clean test accuracy and the fit to the flipped labels, as the reference reports them.

The reference publishes three fractions: the accuracy on the clean test split,
the fraction of flipped training images whose flipped label the model
predicts (``poison_fit``, lower is better), and their mean robustness
``(test_acc + 1 - poison_fit) / 2``. Each divides by its count floored at one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from configgle import Fig
from torch import Tensor
from treekle import from_plain

from priml.baselines.mlsbench.tl.poison_robust_learning.data import SPLITS


if TYPE_CHECKING:
    from collections.abc import Mapping


class PoisonRobustness:
    """Correct predictions per split, the poisoned split's counted on flipped images only."""

    class Config(Fig["PoisonRobustness"]):
        """Takes no parameters."""

    def __init__(self, config: Config) -> None:
        del config
        self.reset()

    def reset(self) -> None:
        """Start a new evaluation."""
        self.correct: dict[str, int] = dict.fromkeys(SPLITS, 0)
        self.total: dict[str, int] = dict.fromkeys(SPLITS, 0)

    def update(self, logits: Tensor, **batch: object) -> None:
        """Count the batch's correct predictions under its split.

        Args:
          logits: Model predictions ``(B, C)``.
          **batch: The batch; ``label`` holds the ``(B,)`` classes, ``split``
            the 0-d index of its split in :data:`SPLITS`, and a poisoned
            batch's ``poison`` the ``(B,)`` mask of the images whose label was
            flipped.

        """
        label = batch["label"]
        assert isinstance(label, Tensor)
        index = batch["split"]
        assert isinstance(index, Tensor)
        split = SPLITS[int(index)]
        hit = logits.argmax(dim=1).eq(label)
        if split == SPLITS[1]:
            poison = batch["poison"]
            assert isinstance(poison, Tensor)
            hit = hit[poison]
        self.correct[split] += int(hit.sum())
        self.total[split] += len(hit)

    def compute(self) -> dict[str, float]:
        """Return the reference's three published metrics.

        Returns:
          metrics: ``test_acc`` and ``poison_fit`` as fractions, and
            ``robust_score``, their robustness mean.

        """
        test_acc = self.correct[SPLITS[0]] / max(self.total[SPLITS[0]], 1)
        poison_fit = self.correct[SPLITS[1]] / max(self.total[SPLITS[1]], 1)
        return {
            "test_acc": test_acc,
            "poison_fit": poison_fit,
            "robust_score": (test_acc + (1.0 - poison_fit)) / 2.0,
        }

    class StateDict(TypedDict):
        """The correct predictions and the images counted of each split."""

        correct: dict[str, int]
        total: dict[str, int]

    def state_dict(self) -> StateDict:
        """Get metric state for checkpointing.

        Returns:
          state: Every split's counts of the open evaluation.

        """
        return {"correct": dict(self.correct), "total": dict(self.total)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load metric state from a checkpoint.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        self.correct = from_plain(state_dict["correct"], dict[str, int])
        self.total = from_plain(state_dict["total"], dict[str, int])
