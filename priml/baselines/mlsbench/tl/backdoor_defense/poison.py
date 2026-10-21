"""A training split with a backdoor planted in a fraction of it, as the driver plants it.

The driver picks the poisoned positions once, from Python's generator: every
position whose label is not the target class is eligible, the list is shuffled,
and its head is kept. A poisoned image is relabeled to the target class and
gets the trigger stamped after its augmentation, so the trigger survives
cropping and mirroring exactly as in the driver's ``PoisonedDataset``.

The positions stay with the data: :meth:`PoisonedData.keep` cuts the split to
the images a defense keeps, and the cut images keep their trigger.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import random

from configgle import Makes
from torch import Tensor
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, TrainLoader
from priml.baselines.mlsbench.tl.backdoor_defense.triggers import (
    TriggerFn,
    badnets,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


class PoisonedData(Cifar100Data):
    """A prepared dataset whose training split carries a backdoor in some images."""

    class Config(Makes["PoisonedData"], Cifar100Data.Config):
        """The prepared tensors, the loader, and the backdoor planted in them."""

        trigger: TriggerFn = badnets
        """Stamps the backdoor into a normalized image."""

        poison_fraction: float = 0.05
        """Poisoned images over all training images, rounded down, at least one."""

        target_class: int = 0
        """Label every poisoned image is given."""

        poison_seed: int = 42
        """Seed of the Python generator that picks the poisoned images."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: PoisonedData.Config = config  # pyrefly: ignore[bad-override-mutable-attribute] -- Narrowed to this class's own config; nothing assigns the base's.
        poisoned = select_poison_indices(
            from_plain(self.train_label.tolist(), list[int]),
            poison_fraction=config.poison_fraction,
            target_class=config.target_class,
            rng=python_generator(config.poison_seed),
        )
        self.is_poison = torch.zeros(len(self.train_label), dtype=torch.bool)
        """``(N,)`` whether each training image carries the backdoor."""
        self.is_poison[poisoned] = True
        self.train_label = self.train_label.masked_fill(
            self.is_poison,
            config.target_class,
        )

    @override
    def train_loader(self) -> PoisonedTrainLoader:
        return PoisonedTrainLoader(
            self.train_media,
            self.train_label,
            self.config,
            is_poison=self.is_poison,
        )

    def analysis_dataloader(self) -> PoisonedTrainLoader:
        """Return one pass over the training split in order, augmented and triggered.

        The driver extracts the features its defense scores from this pass: the
        training transform, unshuffled, through the same four loader workers.

        Returns:
          loader: Batches of ``batch_size`` in split order.

        """
        return SequentialPoisonedTrainLoader(
            self.train_media,
            self.train_label,
            self.config,
            is_poison=self.is_poison,
        )

    def keep(self, indices: Tensor) -> None:
        """Cut the training split to ``indices``, in that order.

        Args:
          indices: ``(K,)`` int64 positions in the current split.

        """
        self.train_media = self.train_media[indices]
        self.train_label = self.train_label[indices]
        self.is_poison = self.is_poison[indices]


class PoisonedTrainLoader(TrainLoader):
    """The vision training loader, stamping the trigger into each poisoned image."""

    def __init__(
        self,
        media: Tensor,
        label: Tensor,
        config: PoisonedData.Config,
        *,
        is_poison: Tensor,
    ) -> None:
        super().__init__(media, label, config)
        self.is_poison = is_poison
        self.trigger = config.trigger

    @override
    def batch(self, index: Tensor, stream: torch.Generator) -> dict[str, Tensor]:
        batch = super().batch(index, stream)
        poisoned = self.is_poison[index]
        media = batch["media"]
        media[poisoned] = self.trigger(media[poisoned], mean=self.mean, std=self.std)
        return batch


class SequentialPoisonedTrainLoader(PoisonedTrainLoader):
    """One unshuffled pass: the driver's analysis loader, which samples in order."""

    @override
    def permutation(self, count: int) -> Tensor:
        # The driver's analysis loader has ``shuffle=False``: a sequential sampler
        # draws nothing, while its workers still take their seed first.
        return torch.arange(count)


def python_generator(seed: int) -> random.Random:
    """Return Python's Mersenne Twister seeded as ``random.seed(seed)`` seeds the driver's."""
    return random.Random(seed)  # noqa: S311 -- The driver's own generator; it picks images and guards no secret.


def select_poison_indices(
    labels: Sequence[int],
    *,
    poison_fraction: float,
    target_class: int,
    rng: random.Random,
) -> list[int]:
    """Pick the positions to poison, as the driver's ``select_poison_indices`` does.

    Args:
      labels: The training split's labels, in order.
      poison_fraction: Poisoned images over all images; the count is rounded
        down and at least one.
      target_class: Images of this class are never picked.
      rng: Shuffles the eligible positions, advanced in place; the driver's
        is Python's global generator after ``random.seed``.

    Returns:
      indices: The picked positions, in the shuffled order.

    """
    eligible = [i for i, label in enumerate(labels) if label != target_class]
    count = max(1, int(len(labels) * poison_fraction))
    rng.shuffle(eligible)
    return eligible[:count]
