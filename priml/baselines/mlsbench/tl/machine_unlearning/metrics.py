"""The reference's evaluation: accuracy, confidence and a membership-inference AUC.

The attack scores every image by the model's top softmax probability. Members
are the forget class's training images, non-members its test images, and the
AUC is the Mann-Whitney statistic of the two score sets. An AUC of one half
means the model no longer tells the images it was trained on from unseen ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy.typing import NDArray

import numpy as np
import torch

from priml.baselines.mlsbench.tl.machine_unlearning.methods import model_logits


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor, nn


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    """The four numbers the reference prints after unlearning."""

    retain_acc: float
    """Accuracy on the retained classes' test images."""

    forget_acc: float
    """Accuracy on the forget class's test images; lower is better."""

    forget_mia_auc: float
    """Membership AUC of forget training against forget test images; lower is better."""

    forget_train_confidence: NDArray[np.floating]
    """The attack's member scores: top probability of each forget training image."""

    forget_test_confidence: NDArray[np.floating]
    """The attack's non-member scores: top probability of each forget test image."""

    @property
    def unlearn_score(self) -> float:
        """Mean of retained accuracy, forget error and one minus the AUC."""
        return (
            self.retain_acc + (1.0 - self.forget_acc) + (1.0 - self.forget_mia_auc)
        ) / 3.0

    def metrics_line(self) -> str:
        """Return the line the reference prints, to its four decimals.

        Returns:
          line: ``TEST_METRICS`` and the four metrics.

        """
        return (
            f"TEST_METRICS retain_acc={self.retain_acc:.4f} "
            f"forget_acc={self.forget_acc:.4f} "
            f"forget_mia_auc={self.forget_mia_auc:.4f} "
            f"unlearn_score={self.unlearn_score:.4f}"
        )


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
) -> tuple[float, NDArray[np.floating]]:
    """Return the accuracy and every image's top softmax probability.

    Args:
      model: A classifier, put in evaluation mode here.
      batches: ``(media, label)`` pairs.
      device: Where the model runs.

    Returns:
      accuracy: Correct predictions over images; 0 when there are none.
      confidence: Each image's largest class probability, in batch order, as
        float32; an empty float64 array when there are no images, as the
        reference returns.

    """
    model.eval()
    correct, total = 0, 0
    confidences: list[NDArray[np.float32]] = []
    for media, label in batches:
        probs = torch.softmax(model_logits(model, media.to(device)), dim=1)
        correct += int(probs.argmax(dim=1).eq(label.to(device)).sum())
        total += label.size(0)
        confidences.append(probs.max(dim=1).values.cpu().numpy())
    confidence = np.concatenate(confidences) if confidences else np.array([])
    return correct / max(total, 1), confidence


def auc_from_scores(
    member: NDArray[np.floating],
    nonmember: NDArray[np.floating],
) -> float:
    """Return the probability a member outscores a non-member, by rank sums.

    Ranks come from NumPy's default ``argsort``, which does not average tied
    scores; a tie is broken by that sort's order, as the reference breaks it.

    Args:
      member: Scores of images seen in training.
      nonmember: Scores of images never seen.

    Returns:
      auc: The Mann-Whitney U statistic over ``len(member) * len(nonmember)``.

    """
    scores = np.concatenate([member, nonmember])
    labels = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = float(ranks[labels == 1].sum())
    n_pos, n_neg = len(member), len(nonmember)
    return (positive - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
