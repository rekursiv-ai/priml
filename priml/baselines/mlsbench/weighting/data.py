"""Long-tailed CIFAR-100: the prepared split, subsampled per class at load.

Class ``c`` keeps ``max(int(n_max * r^(-c / (C - 1))), 1)`` images, where
``n_max`` is the count of class 0 and ``r`` the imbalance ratio. Which images it
keeps is drawn as the reference draws them: one NumPy ``RandomState`` seeded
once, shuffling each class's indices in class order, keeping each shuffle's
head. The kept images stay in that order, which the loader's shuffle then
permutes, so the subset reproduces the reference's batches exactly.

The counts are data, not config. :meth:`LongTailCifar100Data.bind_step` hands
them to a loss that asks for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import Makes
from torch import Tensor

import numpy as np
import torch

from priml.baselines.mlsbench.pytorch_vision.data import Cifar100Data
from priml.baselines.mlsbench.weighting.loss import BindsClassCounts
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepProtocol


class LongTailCifar100Data(Cifar100Data):
    """CIFAR-100 whose training split decays exponentially from class 0 to the last."""

    class Config(Makes["LongTailCifar100Data"], Cifar100Data.Config):
        """The prepared tensors, the loader, and the long tail cut from them."""

        imbalance_ratio: float = 100.0
        """Images of class 0 over images of the last class."""

        num_classes: int = 100
        """Classes the decay is spread over, whether or not the split holds each."""

        subset_seed: int = 42
        """Seed of the NumPy generator that picks each class's images."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        indices, counts = long_tail_indices(
            self.train_label,
            imbalance_ratio=config.imbalance_ratio,
            num_classes=config.num_classes,
            seed=config.subset_seed,
        )
        self.class_counts: Tensor = counts
        """``(C,)`` float32 images allotted per class."""
        self.train_media = self.train_media[indices]
        self.train_label = self.train_label[indices]

    def bind_step(self, step: TrainStepProtocol) -> None:
        """Give the step's loss the class counts, if it asks for them.

        Args:
          step: The step the loop trains; the loop calls this before the first
            update.

        """
        if isinstance(step, TrainStep) and isinstance(step.loss, BindsClassCounts):
            step.loss.bind_class_counts(self.class_counts)


def long_tail_counts(
    n_max: int,
    *,
    imbalance_ratio: float,
    num_classes: int,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """Return how many images each class keeps.

    Args:
      n_max: Images class 0 keeps.
      imbalance_ratio: Images of class 0 over images of the last class.
      num_classes: Classes the decay is spread over.

    Returns:
      counts: ``(num_classes,)`` int64, ``max(int(n_max * mu_c), 1)`` with
        ``mu_c = (1 / imbalance_ratio) ** (c / (num_classes - 1))``.

    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mu = (1.0 / imbalance_ratio) ** (c / (num_classes - 1))
        counts[c] = max(int(n_max * mu), 1)
    return counts


def long_tail_indices(
    label: Tensor,
    *,
    imbalance_ratio: float,
    num_classes: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Pick each class's images of a balanced split, decaying from class 0.

    Args:
      label: ``(N,)`` class of each image.
      imbalance_ratio: Images of class 0 over images of the last class.
      num_classes: Classes the decay is spread over.
      seed: Seed of the NumPy generator that shuffles each class.

    Returns:
      indices: ``(K,)`` int64 kept positions, class by class, each class in
        shuffled order.
      class_counts: ``(num_classes,)`` float32 images each class was allotted;
        a class the split lacks is still allotted one.

    """
    counts = long_tail_counts(
        int((label == 0).sum()),
        imbalance_ratio=imbalance_ratio,
        num_classes=num_classes,
    )
    rng = np.random.RandomState(seed)
    kept: list[np.ndarray[tuple[int], np.dtype[np.intp]]] = []
    for c in range(num_classes):
        # ``flatnonzero`` is the reference's ``np.where(targets == c)[0]`` for a
        # vector: same positions, same order, so the shuffle sees the same array.
        class_indices = np.flatnonzero((label == c).numpy())
        rng.shuffle(class_indices)
        kept.append(class_indices[: counts[c]])
    return (
        torch.from_numpy(np.concatenate(kept)),
        torch.tensor(counts, dtype=torch.float32),
    )
