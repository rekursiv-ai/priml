"""The fixed harness around a defense: extract features, filter, and measure the backdoor.

Each function is one step of the driver's ``main`` after the victim is
trained: features and logits of every training image, the defense's scores
turned into the images kept, and the retrained model's clean accuracy and
attack success rate. Every quantity follows the driver's arithmetic and batch
order, so the scores, the kept images and the reported numbers match it bit for
bit given the same network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from numpy.typing import NDArray
from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision.data import normalize


if TYPE_CHECKING:
    from collections.abc import Iterable

    from priml.baselines.mlsbench.tl.backdoor_defense.defenses import (
        BackdoorDefense,
    )
    from priml.baselines.mlsbench.tl.backdoor_defense.triggers import TriggerFn


class Classifier(Protocol):
    """Maps a batch to class scores; a module's ``__call__``, typed."""

    def __call__(self, media: Tensor, /) -> Tensor:
        """Return ``(B, C)`` scores."""
        raise NotImplementedError

    def eval(self) -> object:
        """Switch to evaluation mode."""
        raise NotImplementedError


@runtime_checkable
class FeatureClassifier(Protocol):
    """A classifier whose penultimate features are reachable; its ``head`` maps them."""

    def features(self, media: Tensor) -> Tensor:
        """Return ``(B, D)`` features for ``(B, C, H, W)`` images."""
        raise NotImplementedError

    def eval(self) -> object:
        """Switch to evaluation mode."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, kw_only=True)
class Filtered:
    """What a defense removed from the training split."""

    scores: NDArray[np.float64]
    """``(N,)`` the defense's suspicion of each image."""

    removed: NDArray[np.intp]
    """Positions removed, in ascending score."""

    kept: NDArray[np.intp]
    """Positions kept, ascending."""

    poison_recall: float
    """Poisoned images removed over poisoned images."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    """The numbers the driver prints after retraining on the kept images."""

    clean_acc: float
    """Accuracy of the retrained model on the clean test split."""

    asr: float
    """Fraction of triggered non-target test images classified as the target."""

    poison_recall: float
    """Poisoned images the defense removed over poisoned images."""

    @property
    def defense_score(self) -> float:
        """The task's objective: the mean of clean accuracy and one minus the ASR."""
        return 0.5 * self.clean_acc + 0.5 * (1.0 - self.asr)


@torch.no_grad()
def collect_features(
    model: object,
    batches: Iterable[dict[str, Tensor]],
    *,
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """Run the network in evaluation mode and keep features, logits and labels.

    Args:
      model: A :class:`FeatureClassifier`.
      batches: ``media`` and ``label`` batches, in the order to keep.
      device: Where the model runs.

    Returns:
      features: ``(N, D)`` penultimate features.
      logits: ``(N, C)`` the head's output on them.
      labels: ``(N,)`` the batches' labels.

    """
    assert isinstance(model, FeatureClassifier)
    assert isinstance(model, nn.Module)
    # A submodule is reached through ``nn.Module.__getattr__``, which a
    # runtime-checkable protocol's static attribute lookup does not see.
    head = _classifier(model.get_submodule("head"))
    model.eval()
    features: list[NDArray[np.float32]] = []
    logits: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int64]] = []
    for batch in batches:
        feature = model.features(batch["media"].to(device))
        logit = head(feature)
        features.append(feature.cpu().numpy())
        logits.append(logit.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(features), np.concatenate(logits), np.concatenate(labels)


def filter_poison(
    defense: BackdoorDefense,
    *,
    features: NDArray[np.float32],
    logits: NDArray[np.float32],
    labels: NDArray[np.int64],
    is_poison: NDArray[np.bool_],
    poison_fraction: float,
    multiplier: float,
) -> Filtered:
    """Fit the defense, score every image and remove the most suspicious.

    It removes ``multiplier`` times the poisoned fraction, at least one image
    and never all of them, as Tran et al. recommend over-removing.

    Args:
      defense: A fresh defense.
      features: ``(N, D)`` penultimate features.
      logits: ``(N, C)`` class scores.
      labels: ``(N,)`` labels after poisoning.
      is_poison: ``(N,)`` which images carry the backdoor; read only to report.
      poison_fraction: Poisoned images over all, as the defense is told.
      multiplier: Images removed over poisoned images.

    Returns:
      filtered: The scores, the removed and kept positions, and the recall.

    """
    defense.fit(features, labels, poison_fraction)
    scores = np.asarray(defense.score_samples(features, logits), dtype=np.float64)
    count = max(1, int(len(scores) * poison_fraction * multiplier))
    count = min(count, len(scores) - 1)
    removed = np.argsort(scores)[-count:]
    keep = np.ones(len(scores), dtype=bool)
    keep[removed] = False
    recall = float(is_poison[removed].sum()) / max(float(is_poison.sum()), 1.0)
    return Filtered(
        scores=scores,
        removed=removed,
        kept=np.nonzero(keep)[0],
        poison_recall=recall,
    )


@torch.no_grad()
def clean_accuracy(
    model: Classifier,
    batches: Iterable[dict[str, Tensor]],
    *,
    device: torch.device,
) -> float:
    """Return the fraction of test images classified correctly.

    Args:
      model: A classifier, put in evaluation mode here.
      batches: Normalized ``media`` and ``label`` batches.
      device: Where the model runs.

    Returns:
      accuracy: In ``[0, 1]``, as the driver divides its percentage by 100.

    """
    model.eval()
    correct, total = 0, 0
    for batch in batches:
        logits = model(batch["media"].to(device))
        correct += int(logits.max(1).indices.eq(batch["label"].to(device)).sum())
        total += len(batch["label"])
    return 100.0 * correct / total / 100.0


@torch.no_grad()
def attack_success_rate(
    model: Classifier,
    media: Tensor,
    label: Tensor,
    *,
    trigger: TriggerFn,
    target_class: int,
    mean: Tensor,
    std: Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return how often a triggered test image of another class is classified as the target.

    Args:
      model: A classifier, put in evaluation mode here.
      media: ``(N, C, H, W)`` uint8 test images.
      label: ``(N,)`` their labels.
      trigger: Stamps the backdoor into normalized images.
      target_class: The class the backdoor maps to; its own images are skipped.
      mean: ``(3,)`` channel means.
      std: ``(3,)`` channel standard deviations.
      batch_size: Images per forward.
      device: Where the model runs.

    Returns:
      asr: In ``[0, 1]``; 0 when every image is of the target class.

    """
    model.eval()
    triggered = trigger(
        normalize(media[label != target_class], mean=mean, std=std),
        mean=mean,
        std=std,
    )
    success = 0
    for start in range(0, len(triggered), batch_size):
        logits = model(triggered[start : start + batch_size].to(device))
        success += int(logits.argmax(dim=1).cpu().eq(target_class).sum())
    return success / max(len(triggered), 1)


def _classifier(module: Classifier) -> Classifier:
    """Return a module typed as the batch-to-scores map the harness calls."""
    return module
