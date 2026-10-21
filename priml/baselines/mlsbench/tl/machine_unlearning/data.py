"""A prepared dataset split into the class to forget and the rest.

The reference builds every loader from one training set and one test set, each
cut by class into a retain and a forget subset. Its loaders run four worker
processes, so the main process's generator gives each pass its draws: one base
seed when a pass starts, then a shuffle seed if the loader shuffles. Worker
``w`` seeds its own generator with ``base + w`` and augments every batch ``b``
with ``b % 4 == w``. :class:`Batches` takes the same draws in the same order,
so a pass yields the reference's batches without any worker process.

The forget set's training images are scored for membership through the
TRAINING pipeline, augmentation included, as the reference scores them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from configgle import Makes
from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.vision.data import (
    Augment,
    Cifar100Data,
    normalize,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


class Transform(Protocol):
    """Turns a batch of uint8 images into model input, drawing from ``stream``."""

    def __call__(self, images: Tensor, stream: torch.Generator, /) -> Tensor:
        """Return ``(B, C, H, W)`` float32 images."""
        raise NotImplementedError


class UnlearningData(Cifar100Data):
    """Training and test splits, each cut into retained classes and the forget class.

    Raises:
      FileNotFoundError: If the prepared tensors are absent. Prepare them with
        ``uv --quiet run --frozen python -m
        priml.baselines.mlsbench.dl.vision.scripts.prepare_data``.

    """

    class Config(Makes["UnlearningData"], Cifar100Data.Config):
        """The vision dataset's location, batching and pipeline, and the class to forget."""

        forget_class: int = 0
        """Label of the class whose training images are to be forgotten."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: UnlearningData.Config = config  # pyrefly: ignore[bad-override-mutable-attribute] -- Narrowed to this class's own config; nothing assigns the base's.
        self.mean = torch.tensor(config.mean)
        self.std = torch.tensor(config.std)
        self.augment: Augment = config.augment.make()
        self.train_retain, self.train_forget = split_by_class(
            self.train_label,
            forget_class=config.forget_class,
        )
        self.test_retain, self.test_forget = split_by_class(
            self.eval_label,
            forget_class=config.forget_class,
        )

    def full_loader(self) -> Batches:
        """Return shuffled, augmented passes over the whole training split."""
        return self._train(torch.arange(len(self.train_label)), shuffle=True)

    def retain_loader(self) -> Batches:
        """Return shuffled, augmented passes over the retained training images."""
        return self._train(self.train_retain, shuffle=True)

    def forget_loader(self) -> Batches:
        """Return shuffled, augmented passes over the forget class's training images."""
        return self._train(self.train_forget, shuffle=True)

    def forget_train_eval_loader(self) -> Batches:
        """Return in-order, augmented passes over the forget class's training images."""
        return self._train(self.train_forget, shuffle=False)

    def retain_test_loader(self) -> Batches:
        """Return in-order passes over the retained test images."""
        return self._test(self.test_retain)

    def forget_test_loader(self) -> Batches:
        """Return in-order passes over the forget class's test images."""
        return self._test(self.test_forget)

    def _train(self, index: Tensor, *, shuffle: bool) -> Batches:
        """Batch training images ``index`` through the training pipeline."""
        return Batches(
            self.train_media[index],
            self.train_label[index],
            batch_size=self.config.batch_size,
            num_streams=self.config.num_streams,
            shuffle=shuffle,
            transform=_Augmented(self.config, self.augment, self.mean, self.std),
        )

    def _test(self, index: Tensor) -> Batches:
        """Batch test images ``index`` through the test pipeline."""
        return Batches(
            self.eval_media[index],
            self.eval_label[index],
            batch_size=self.config.eval_batch_size,
            num_streams=self.config.num_streams,
            shuffle=False,
            transform=_Normalized(self.mean, self.std),
        )


def split_by_class(label: Tensor, *, forget_class: int) -> tuple[Tensor, Tensor]:
    """Return the indices of every other class and of ``forget_class``, ascending.

    Args:
      label: ``(N,)`` integer classes.
      forget_class: The class to separate.

    Returns:
      retain: Indices whose label differs from ``forget_class``.
      forget: Indices whose label is ``forget_class``.

    """
    forget = label == forget_class
    return torch.nonzero(~forget).flatten(), torch.nonzero(forget).flatten()


class Batches:
    """Re-iterable passes of ``(media, label)`` batches, drawn as the reference loader."""

    def __init__(
        self,
        media: Tensor,
        label: Tensor,
        *,
        batch_size: int,
        num_streams: int,
        shuffle: bool,
        transform: Transform,
    ) -> None:
        self.media = media
        self.label = label
        self.batch_size = batch_size
        self.num_streams = num_streams
        self.shuffle = shuffle
        self.transform = transform

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        # The base seed comes first and is drawn whether or not any batch
        # uses a stream; the shuffle seed follows, as the sampler draws it.
        stream_seed = int(torch.empty((), dtype=torch.int64).random_())
        count = len(self.media)
        order = torch.arange(count)
        if self.shuffle:
            shuffle_seed = int(torch.empty((), dtype=torch.int64).random_())
            order = torch.randperm(
                count,
                generator=torch.Generator().manual_seed(shuffle_seed),
            )
        streams = [
            torch.Generator().manual_seed(stream_seed + worker)
            for worker in range(self.num_streams)
        ]
        for batch, start in enumerate(range(0, count, self.batch_size)):
            index = order[start : start + self.batch_size]
            yield (
                self.transform(self.media[index], streams[batch % self.num_streams]),
                self.label[index],
            )

    def __len__(self) -> int:
        return (len(self.media) + self.batch_size - 1) // self.batch_size


def cycle(loader: Batches) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield the batches of pass after pass, starting each pass when the last ends.

    Args:
      loader: Batches to repeat.

    Yields:
      batch: The next ``(media, label)``.

    """
    while True:
        yield from loader


class _Augmented:
    """The training pipeline with the dataset's padding and statistics bound."""

    def __init__(
        self,
        config: UnlearningData.Config,
        augment: Augment,
        mean: Tensor,
        std: Tensor,
    ) -> None:
        self.pad = config.translate_pad
        self.augment = augment
        self.mean = mean
        self.std = std

    def __call__(self, images: Tensor, stream: torch.Generator, /) -> Tensor:
        return self.augment(images, stream, pad=self.pad, mean=self.mean, std=self.std)


class _Normalized:
    """The test pipeline: normalization alone, the stream unused."""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, images: Tensor, stream: torch.Generator, /) -> Tensor:
        del stream
        return normalize(images, mean=self.mean, std=self.std)
