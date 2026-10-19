"""Test accuracy and a confidence-threshold membership attack, as the reference reports them.

The attack scores each image by the model's largest softmax probability and
asks how well that score separates the members from the non-members: its AUC,
by the Mann-Whitney statistic. The reference publishes the test accuracy, the
AUC, the gap between the two halves' mean scores, and the accuracy less the
AUC's advantage over chance.

The arithmetic is NumPy's, in the reference's order: the AUC ranks the scores
with NumPy's default (unstable) sort, so tied scores take the distinct ranks
that sort gives them, and any other sort would rank ties differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.baselines.mlsbench.tl.membership_inference_defense.data import SPLITS


if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy import typing as npt

type Scores = npt.NDArray[np.float32]
"""One membership score per image."""


class MembershipInference:
    """Accuracy and largest-probability scores per split, reduced to the privacy metrics."""

    class Config(Fig["MembershipInference"]):
        """Takes no parameters."""

    def __init__(self, config: Config) -> None:
        del config
        self.reset()

    def reset(self) -> None:
        """Start a new evaluation."""
        self.scores: dict[str, list[Tensor]] = {split: [] for split in SPLITS}
        self.correct: dict[str, int] = dict.fromkeys(SPLITS, 0)
        self.total: dict[str, int] = dict.fromkeys(SPLITS, 0)

    def update(self, logits: Tensor, **batch: object) -> None:
        """Score the batch and count its correct predictions under its split.

        Args:
          logits: Model predictions ``(B, C)``.
          **batch: The batch; ``label`` holds the ``(B,)`` classes and
            ``split`` the 0-d index of its split in :data:`SPLITS`.

        """
        label = batch["label"]
        assert isinstance(label, Tensor)
        index = batch["split"]
        assert isinstance(index, Tensor)
        split = SPLITS[int(index)]
        probs = torch.softmax(logits, dim=1)
        self.scores[split].append(probs.max(dim=1).values.cpu())
        self.correct[split] += int(probs.argmax(dim=1).eq(label).sum())
        self.total[split] += len(label)

    def compute(self) -> dict[str, float]:
        """Return the reference's four published metrics.

        Returns:
          metrics: ``test_acc`` as a fraction, ``mia_auc``, ``privacy_gap``
            (the members' mean score less the non-members') and
            ``privacy_score`` (``test_acc - max(mia_auc - 0.5, 0)``).

        Raises:
          ValueError: Either half saw no image.

        """
        if not self.scores["member"] or not self.scores["nonmember"]:
            raise ValueError("The evaluation saw no members or no non-members.")
        member, nonmember = self.split_scores("member"), self.split_scores("nonmember")
        test_acc = self.correct["test"] / max(self.total["test"], 1)
        auc = auc_from_scores(member, nonmember)
        return {
            "test_acc": test_acc,
            "mia_auc": auc,
            "privacy_gap": float(member.mean() - nonmember.mean()),
            "privacy_score": test_acc - max(auc - 0.5, 0.0),
        }

    def split_scores(self, split: str) -> Scores:
        """Return one split's scores so far, in the order they were seen."""
        return cast(
            "Scores",
            np.concatenate([score.numpy() for score in self.scores[split]]),
        )

    class StateDict(TypedDict):
        """The scores, correct predictions and images of each split."""

        scores: dict[str, list[Tensor]]
        correct: dict[str, int]
        total: dict[str, int]

    def state_dict(self) -> StateDict:
        """Get metric state for checkpointing.

        Returns:
          state: Every split's scores and counts of the open evaluation.

        """
        return {
            "scores": {k: list(v) for k, v in self.scores.items()},
            "correct": dict(self.correct),
            "total": dict(self.total),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load metric state from a checkpoint.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(MembershipInference.StateDict, state_dict)
        self.scores = {k: list(v) for k, v in state["scores"].items()}
        self.correct = dict(state["correct"])
        self.total = dict(state["total"])


def auc_from_scores(member: Scores, nonmember: Scores) -> float:
    """Return the probability a member outscores a non-member, by Mann-Whitney's U.

    Args:
      member: Scores of the members.
      nonmember: Scores of the non-members.

    Returns:
      auc: ``U / (n_member * n_nonmember)``; zero when either half is empty.

    Derivation:
      Rank all scores 1 to N in ascending order. The members' rank sum less
      its least possible value ``n (n + 1) / 2`` counts the (member,
      non-member) pairs the member wins: ``U``.

    """
    scores = np.concatenate([member, nonmember])
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[np.argsort(scores)] = np.arange(1, len(scores) + 1)
    n_member, n_nonmember = len(member), len(nonmember)
    wins = ranks[:n_member].sum() - n_member * (n_member + 1) / 2.0
    return float(wins / max(n_member * n_nonmember, 1))
