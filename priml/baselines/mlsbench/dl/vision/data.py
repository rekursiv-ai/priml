"""CIFAR-100 batches, shuffled and augmented as the reference loader does.

The images stay as the 8-bit tensors the dataset ships. A pass over the
training split reproduces every random draw of a multi-process ``DataLoader``:
it takes two seeds from the global generator, one for the shuffle and one for
the augmentation streams. Batch ``b`` is augmented from stream ``b %
num_streams``, which draws one crop offset pair and one flip per image. A
pass over the test split draws a seed it never uses, because the reference
loader does.

:func:`prepare` downloads and caches the tensors; ``scripts/prepare_data.py``
is its command-line front end. :class:`Cifar100Data` only reads that cache, so
constructing a config never touches the network.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import logging

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor
from torch.nn import functional

import numpy as np
import torch

from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from PIL import Image
    from torchvision import datasets
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")
    datasets = lazy_import("torchvision.datasets")


logger = logging.getLogger(__name__)


class Cifar100Data:
    """CIFAR-100 yielding normalized ``media`` and ``label`` batches.

    Raises:
      FileNotFoundError: If the prepared tensors are absent. Run
        ``uv --quiet run --frozen python -m
        priml.baselines.mlsbench.dl.vision.scripts.prepare_data``
        first.

    """

    class Config(Fig["Cifar100Data"]):
        """Where the prepared tensors live and how batches are made."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/cifar100"
        """Logical directory holding ``train.pt`` and ``test.pt``.

        Resolved beneath ``base_dir`` at finalize, so it names a location
        within the resource root rather than an absolute filesystem path."""

        batch_size: int = 128
        """Images per training batch."""

        eval_batch_size: int = 128
        """Images per evaluation batch."""

        num_streams: int = 4
        """Augmentation streams, one per worker of the reference loader."""

        translate_pad: int = 4
        """Zero padding added on every side before the random crop, in pixels."""

        mean: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
        """Per-channel mean subtracted from images scaled to ``[0, 1]``."""

        std: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
        """Per-channel standard deviation divided out after centering."""

        augment: Makeable[Augment] = field(
            default_factory=lambda: PartialConfig(pad_crop_flip_normalize),
        )
        """Turns a batch of uint8 training images into normalized float images.

        Called with the batch, its stream (one per reference loader worker),
        and ``pad``, ``mean`` and ``std`` from this config, so a fork changes
        only the augmentation and keeps the dataset's statistics."""

        @override
        def finalize(self) -> Self:
            for name in ("batch_size", "eval_batch_size", "num_streams"):
                if getattr(self, name) < 1:
                    raise ValueError(
                        f"{name} must be at least 1; got {getattr(self, name)}.",
                    )
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the data; ticked by the loop when the loader runs out."""
        self._live: TrainLoader | None = None
        self._pending_loader_state: TrainLoader.StateDict | None = None
        directory = Path(config.working_dir)
        self.train_media, self.train_label = _load_split(directory / "train.pt")
        self.eval_media, self.eval_label = _load_split(directory / "test.pt")

    def train_dataloader(self) -> TrainLoader:
        """Return the re-iterable training loader.

        Returns:
          loader: Each iteration is one shuffled, augmented pass.

        """
        loader = self.train_loader()
        if self._pending_loader_state is not None:
            loader.load_state_dict(self._pending_loader_state)
            self._pending_loader_state = None
        self._live = loader
        return loader

    def train_loader(self) -> TrainLoader:
        """Build a fresh training loader; a task whose passes differ overrides this.

        :meth:`train_dataloader` restores any pending checkpoint state into it.

        Returns:
          loader: Over this dataset's training split.

        """
        return TrainLoader(self.train_media, self.train_label, self.config)

    def eval_dataloader(self) -> EvalLoader:
        """Return the re-iterable test loader.

        Returns:
          loader: Each iteration is one sequential pass over the test split.

        """
        return EvalLoader(self.eval_media, self.eval_label, self.config)

    class StateDict(TypedDict):
        """The live loader's cursor and the pass timer.

        ``loader`` is None when no pass has started and none was pending.
        """

        loader: TrainLoader.StateDict | None
        timer_epoch: NotRequired[CheckpointableStepTimer.StateDict]

    def state_dict(self) -> StateDict:
        """Return the pass count and the position inside the active pass.

        Returns:
          state: ``"loader"`` (nested loader state) and ``"timer_epoch"``.

        """
        loader_state = (
            self._live.state_dict()
            if self._live is not None
            else self._pending_loader_state
        )
        return {
            "loader": loader_state,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the pass count and the position inside the active pass.

        Args:
          state_dict: Checkpoint from state_dict() to restore training.

        """
        state = cast(Cifar100Data.StateDict, state_dict)
        if "timer_epoch" in state:
            self.timer_epoch.load_state_dict(state["timer_epoch"])
        # A checkpoint taken before any pass carries no cursor; restoring it
        # starts the next pass afresh rather than keeping this dataset's own.
        loader_state = state.get("loader") or TrainLoader.StateDict(
            order=None,
            streams=[],
            next_batch=0,
        )
        self._pending_loader_state = loader_state
        if self._live is not None:
            self._live.load_state_dict(loader_state)
            self._pending_loader_state = None


def normalize(images: Tensor, *, mean: Tensor, std: Tensor) -> Tensor:
    """Scale 8-bit images to ``[0, 1]`` and standardize each channel.

    Args:
      images: ``(B, C, H, W)`` uint8 pixels.
      mean: ``(C,)`` float32 channel means.
      std: ``(C,)`` float32 channel standard deviations.

    Returns:
      media: ``(B, C, H, W)`` float32 images.

    """
    return (images.float().div(255) - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)


class Augment(Protocol):
    """Turns uint8 training images into normalized float images, drawing from ``stream``."""

    def __call__(
        self,
        images: Tensor,
        stream: torch.Generator,
        /,
        *,
        pad: int,
        mean: Tensor,
        std: Tensor,
    ) -> Tensor:
        """Return ``(B, C, H, W)`` float32 images for ``(B, C, H, W)`` uint8 ``images``."""
        raise NotImplementedError


def pad_crop_flip_normalize(
    images: Tensor,
    stream: torch.Generator,
    /,
    *,
    pad: int,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Pad, crop and mirror at random, then normalize: the reference's training pipeline.

    Args:
      images: ``(B, C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      pad: Pixels of zeros added on every side before the crop.
      mean: ``(C,)`` float32 channel means.
      std: ``(C,)`` float32 channel standard deviations.

    Returns:
      media: ``(B, C, H, W)`` float32 images.

    """
    return normalize(pad_crop_flip(images, stream, pad=pad), mean=mean, std=std)


def pad_crop_flip(images: Tensor, stream: torch.Generator, *, pad: int) -> Tensor:
    """Zero-pad, crop at a random offset and mirror each image at random.

    Each image takes three draws from ``stream`` in this order: the crop's top
    offset, its left offset, and a uniform that mirrors the crop below one half.

    Args:
      images: ``(B, C, H, W)`` uint8 pixels.
      stream: Generator the draws come from, advanced in place.
      pad: Pixels of zeros added on every side.

    Returns:
      images: ``(B, C, H, W)`` uint8 pixels, cropped back to the input size.

    """
    height, width = images.shape[-2:]
    padded = functional.pad(images, (pad,) * 4)
    views: list[Tensor] = []
    for index in range(len(images)):
        top = int(torch.randint(0, 2 * pad + 1, (1,), generator=stream))
        left = int(torch.randint(0, 2 * pad + 1, (1,), generator=stream))
        crop = padded[index, :, top : top + height, left : left + width]
        views.append(crop.flip(-1) if torch.rand(1, generator=stream) < 0.5 else crop)
    return torch.stack(views)


def prepare(
    directory: Path | str,
    *,
    dataset: Literal["cifar10", "cifar100", "fmnist"] = "cifar100",
) -> None:
    """Download a dataset and cache its 8-bit tensors under ``directory``.

    Idempotent: a split whose ``.pt`` file already exists is left untouched, so
    this is safe to call at the start of every run. CIFAR pixels are stored as
    shipped. Fashion-MNIST is stored as ``(N, 1, 32, 32)``, already through the
    reference's ``Resize(32)``: that step comes first in both of its pipelines
    and draws nothing, so caching its output changes no pixel. Scaling,
    standardization and the grayscale channel's repeat belong to the loader,
    where they are part of the recipe.

    Args:
      directory: Destination for ``train.pt`` and ``test.pt``.
      dataset: Which dataset; the loader reads any of them.

    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        path = destination / f"{split}.pt"
        if path.exists():
            logger.info("%s: %s already prepared at %s", dataset, split, path)
            continue
        media, label = _fetch(dataset, destination, train=train)
        # Write to a sibling then rename: an interrupted download must not leave
        # a truncated .pt that the existence check above would later accept.
        staging = path.with_suffix(".pt.partial")
        torch.save({"media": media, "label": label}, staging)
        staging.replace(path)
        logger.info("%s: wrote %d %s images to %s", dataset, len(media), split, path)


def resized(images: Tensor, *, size: int) -> Tensor:
    """Resize square 8-bit images as torchvision's ``Resize(size)`` does a PIL image.

    That transform, given a square PIL image, calls ``Image.resize`` with
    bilinear resampling; this makes the same call, so its pixels are Pillow's.

    Args:
      images: ``(B, C, H, H)`` uint8 pixels, one or three channels.
      size: Side of the output.

    Returns:
      images: ``(B, C, size, size)`` uint8 pixels.

    """
    out: list[Tensor] = []
    for image in images:
        pil = Image.fromarray(image.permute(1, 2, 0).squeeze(-1).numpy())
        pixels = np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR))
        out.append(torch.from_numpy(pixels.copy()).reshape(size, size, -1))
    return torch.stack(out).permute(0, 3, 1, 2).contiguous()


class EvalLoader:
    """Sequential batches of normalized test images, as the reference's test loader.

    Each pass takes one unused draw from the global generator first, as the
    reference's loader does. A task whose test pass differs subclasses this.

    Args:
      media: ``(N, C, H, W)`` uint8 images.
      label: ``(N,)`` integer labels.
      config: The dataset; its ``eval_batch_size``, ``mean`` and ``std``.

    """

    def __init__(
        self,
        media: Tensor,
        label: Tensor,
        config: Cifar100Data.Config,
    ) -> None:
        self.media = media
        self.label = label
        self.batch_size = config.eval_batch_size
        self.mean = torch.tensor(config.mean)
        self.std = torch.tensor(config.std)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield ``media`` and ``label`` batches in order, normalized."""
        # The reference loader takes a seed from the global generator on every
        # pass, used or not. Dropping this draw shifts the shuffle of every
        # epoch that follows an evaluation.
        _ = torch.empty((), dtype=torch.int64).random_()
        for start in range(0, len(self.media), self.batch_size):
            stop = start + self.batch_size
            yield {
                "media": normalize(
                    self.media[start:stop],
                    mean=self.mean,
                    std=self.std,
                ),
                "label": self.label[start:stop],
            }

    def __len__(self) -> int:
        """Return the batches in one pass; the last may be short."""
        return (len(self.media) + self.batch_size - 1) // self.batch_size


class TrainLoader:
    """Shuffled, augmented batches with the reference loader's draws, resumable mid-pass.

    A pass takes a stream seed and then a shuffle seed from the global
    generator, and augments batch ``b`` from stream ``b % num_streams``. A
    task whose training pass differs subclasses this.

    Args:
      media: ``(N, C, H, W)`` uint8 images.
      label: ``(N,)`` integer labels.
      config: The dataset; its batch size, streams, padding, statistics and
        ``augment``.

    """

    def __init__(
        self,
        media: Tensor,
        label: Tensor,
        config: Cifar100Data.Config,
    ) -> None:
        self.media = media
        self.label = label
        self.batch_size = config.batch_size
        self.num_streams = config.num_streams
        self.pad = config.translate_pad
        self.mean = torch.tensor(config.mean)
        self.std = torch.tensor(config.std)
        self.augment = config.augment.make()
        self._order: Tensor | None = None
        self._streams: list[torch.Generator] = []
        self._next_batch = 0

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield the rest of the active pass, or a fresh pass, as augmented batches."""
        count = len(self.media)
        if self._order is None:
            stream_seed = int(torch.empty((), dtype=torch.int64).random_())
            self._order = self.permutation(count)
            self._streams = [
                torch.Generator().manual_seed(stream_seed + worker)
                for worker in range(self.num_streams)
            ]
            self._next_batch = 0
        order = self._order
        for batch_index, start in enumerate(range(0, count, self.batch_size)):
            if batch_index < self._next_batch:
                continue
            index = order[start : start + self.batch_size]
            self._next_batch = batch_index + 1
            yield self.batch(index, self._streams[batch_index % self.num_streams])
        self._order = None
        self._streams = []
        self._next_batch = 0

    def permutation(self, count: int) -> Tensor:
        """Return a pass's order: a permutation from a seed the global generator draws.

        Called once per pass, after the stream seed is drawn, as the reference
        loader's sampler draws after its workers' base seed. A task whose
        loader does not shuffle overrides it to draw nothing.

        Args:
          count: Images in the split.

        Returns:
          order: ``(count,)`` image indices.

        """
        shuffle_seed = int(torch.empty((), dtype=torch.int64).random_())
        return torch.randperm(
            count,
            generator=torch.Generator().manual_seed(shuffle_seed),
        )

    def batch(self, index: Tensor, stream: torch.Generator) -> dict[str, Tensor]:
        """Return the augmented images at ``index`` and their labels.

        A task that changes images after augmentation overrides this.

        Args:
          index: Image indices of the batch.
          stream: The batch's augmentation stream, advanced in place.

        Returns:
          batch: ``media`` and ``label``.

        """
        return {
            "media": self.augment(
                self.media[index],
                stream,
                pad=self.pad,
                mean=self.mean,
                std=self.std,
            ),
            "label": self.label[index],
        }

    def __len__(self) -> int:
        """Return the batches in one pass; the last may be short."""
        return (len(self.media) + self.batch_size - 1) // self.batch_size

    class StateDict(TypedDict):
        """The active permutation, its stream states and its next batch."""

        order: Tensor | None
        streams: list[Tensor]
        next_batch: int

    def state_dict(self) -> StateDict:
        """Return the active permutation, stream states and next batch index.

        Returns:
          state: ``None`` for ``order`` between passes.

        """
        return {
            "order": self._order,
            "streams": [stream.get_state() for stream in self._streams],
            "next_batch": self._next_batch,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the permutation, stream states and next batch index.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(TrainLoader.StateDict, state_dict)
        self._order = state["order"]
        self._streams = [
            torch.Generator().set_state(stream) for stream in state["streams"]
        ]
        self._next_batch = state["next_batch"]


def _fetch(
    dataset: Literal["cifar10", "cifar100", "fmnist"],
    destination: Path,
    *,
    train: bool,
) -> tuple[Tensor, Tensor]:
    """Download one split; return channel-first uint8 images and their labels."""
    if dataset == "fmnist":
        source = datasets.FashionMNIST(str(destination), train=train, download=True)
        return resized(source.data[:, None], size=32), source.targets
    fetch = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    cifar = fetch(str(destination), train=train, download=True)
    media = torch.as_tensor(cifar.data).moveaxis(-1, -3).contiguous()
    return media, torch.as_tensor(cifar.targets)


def _load_split(path: Path) -> tuple[Tensor, Tensor]:
    """Load one prepared split as uint8 images and integer labels."""
    payload = cast(
        dict[str, Tensor],
        torch.load(path, map_location="cpu", weights_only=True),
    )
    return payload["media"], payload["label"]
