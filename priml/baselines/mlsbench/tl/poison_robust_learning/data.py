"""The training split with a seeded fraction of its labels flipped, as the reference flips them.

The reference draws a permutation of the training split from a generator of
its own, seeded with its ``--seed``, and moves the label of each image in the
permutation's first ``int(n * fraction)`` to the next class, wrapping at the
last. Training then sees only the flipped labels, through the vision task's
shuffle and augmentation streams unchanged.

After training the reference scores two passes: the clean test split, and the
whole poisoned training split in order, through the training pipeline -- its
random crops and mirrors included, one stream per loader worker -- counting
how often the model predicts the flipped label of a flipped image. Each pass
takes its own seed from the global generator first, as each of the
reference's two loaders does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

from configgle import Makes

import torch

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, EvalLoader


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor


SPLITS: Final = ("test", "poisoned")
"""The evaluation pass's splits, in the order the reference scores them."""


class PoisonedData(Cifar100Data):
    """A dataset that trains on flipped labels and evaluates the test and poisoned splits."""

    class Config(Makes["PoisonedData"], Cifar100Data.Config):
        """The prepared tensors, the loader, and the label flip."""

        poison_fraction: float = 0.1
        """Fraction of the training split whose label is flipped, floored to a count."""

        poison_seed: int = 42
        """Seed of the generator that picks the flipped images."""

        num_classes: int = -1
        """Classes a flipped label wraps around; the experiment sets it."""

    def __init__(self, config: Config) -> None:
        if config.num_classes < 1:
            raise ValueError(
                f"num_classes must be at least 1; got {config.num_classes}.",
            )
        super().__init__(config)
        self.config: PoisonedData.Config = config  # pyrefly: ignore[bad-override-mutable-attribute] -- Narrowed to this class's own config; nothing assigns the base's.
        self.clean_label = self.train_label
        """``(N,)`` labels of the training split as prepared."""
        self.train_label, self.poison_index = flipped_labels(
            self.clean_label,
            num_classes=config.num_classes,
            fraction=config.poison_fraction,
            seed=config.poison_seed,
        )
        self.is_poison = torch.zeros(len(self.clean_label), dtype=torch.bool)
        """``(N,)`` whether each training image's label was flipped."""
        self.is_poison[self.poison_index] = True

    @override
    def eval_dataloader(self) -> PoisonEvalLoader:
        """Return the re-iterable evaluation loader over both splits.

        Returns:
          loader: Each iteration is one sequential pass over each of
            :data:`SPLITS`, every batch tagged with its ``split`` index.

        """
        return PoisonEvalLoader(
            (self.eval_media, self.eval_label),
            poisoned=(self.train_media, self.train_label, self.is_poison),
            config=self.config,
        )


class PoisonEvalLoader(EvalLoader):
    """The test split, then the poisoned training split through the training pipeline.

    Args:
      test: ``(N, C, H, W)`` uint8 test images and their ``(N,)`` labels.
      poisoned: The training images, their flipped labels and the ``(M,)``
        mask of the flipped ones.
      config: The dataset; its batch size, streams, padding, statistics and
        ``augment``.

    """

    def __init__(
        self,
        test: tuple[Tensor, Tensor],
        *,
        poisoned: tuple[Tensor, Tensor, Tensor],
        config: Cifar100Data.Config,
    ) -> None:
        super().__init__(*test, config)
        self.poisoned_media, self.poisoned_label, self.is_poison = poisoned
        self.num_streams = config.num_streams
        self.pad = config.translate_pad
        self.augment = config.augment.make()

    @override
    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield the test batches, then the poisoned ones with their ``poison`` mask.

        ``split`` is a 0-d int64 index into :data:`SPLITS`.
        """
        for batch in super().__iter__():
            yield batch | {"split": torch.tensor(0)}
        # The reference's second loader takes its workers' base seed here; worker
        # ``w`` seeds its generator ``seed + w`` and serves every batch ``b`` with
        # ``b % workers == w``.
        seed = int(torch.empty((), dtype=torch.int64).random_())
        streams = [
            torch.Generator().manual_seed(seed + worker)
            for worker in range(self.num_streams)
        ]
        for index, start in enumerate(
            range(0, len(self.poisoned_media), self.batch_size),
        ):
            stop = start + self.batch_size
            yield {
                "media": self.augment(
                    self.poisoned_media[start:stop],
                    streams[index % self.num_streams],
                    pad=self.pad,
                    mean=self.mean,
                    std=self.std,
                ),
                "label": self.poisoned_label[start:stop],
                "poison": self.is_poison[start:stop],
                "split": torch.tensor(1),
            }

    @override
    def __len__(self) -> int:
        """Return the batches of both passes."""
        poisoned = (len(self.poisoned_media) + self.batch_size - 1) // self.batch_size
        return super().__len__() + poisoned


def flipped_labels(
    clean: Tensor,
    *,
    num_classes: int,
    fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Move a seeded fraction of the labels to the next class.

    Args:
      clean: ``(N,)`` integer labels.
      num_classes: Classes a label wraps around.
      fraction: Fraction to flip; ``int(N * fraction)`` labels are.
      seed: Seed of the generator whose permutation picks them; the global
        generator is not touched.

    Returns:
      label: ``(N,)`` int64 labels, flipped ones ``(clean + 1) % num_classes``.
      index: The flipped positions, in the permutation's order.

    """
    count = len(clean)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(count, generator=generator)[: int(count * fraction)]
    label = clean.clone().long()
    label[index] = (label[index] + 1) % num_classes
    return label, index
