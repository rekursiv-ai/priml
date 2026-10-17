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

import torch

from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torchvision import datasets
else:
    from wrapt import lazy_import

    datasets = lazy_import("torchvision.datasets")


logger = logging.getLogger(__name__)


class Cifar100Data:
    """CIFAR-100 yielding normalized ``media`` and ``label`` batches.

    Raises:
      FileNotFoundError: If the prepared tensors are absent. Run
        ``uv --quiet run --frozen python -m
        priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data``
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
        self._live: _TrainLoader | None = None
        self._pending_loader_state: _TrainLoader.StateDict | None = None
        directory = Path(config.working_dir)
        self.train_media, self.train_label = _load_split(directory / "train.pt")
        self.eval_media, self.eval_label = _load_split(directory / "test.pt")

    def train_dataloader(self) -> _TrainLoader:
        """Return the re-iterable training loader.

        Returns:
          loader: Each iteration is one shuffled, augmented pass.

        """
        loader = _TrainLoader(
            self.train_media,
            self.train_label,
            self.config,
        )
        if self._pending_loader_state is not None:
            loader.load_state_dict(self._pending_loader_state)
            self._pending_loader_state = None
        self._live = loader
        return loader

    def eval_dataloader(self) -> _EvalLoader:
        """Return the re-iterable test loader.

        Returns:
          loader: Each iteration is one sequential pass over the test split.

        """
        return _EvalLoader(self.eval_media, self.eval_label, self.config)

    class StateDict(TypedDict):
        """The live loader's cursor and the pass timer.

        ``loader`` is None when no pass has started and none was pending.
        """

        loader: _TrainLoader.StateDict | None
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
        loader_state = state.get("loader") or _TrainLoader.StateDict(
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
    dataset: Literal["cifar10", "cifar100"] = "cifar100",
) -> None:
    """Download CIFAR and cache its 8-bit tensors under ``directory``.

    Idempotent: a split whose ``.pt`` file already exists is left untouched, so
    this is safe to call at the start of every run. Pixels are stored as
    shipped; scaling and standardization belong to the loader, where they are
    part of the recipe.

    Args:
      directory: Destination for ``train.pt`` and ``test.pt``.
      dataset: Which CIFAR; the loader reads either.

    """
    fetch = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        path = destination / f"{split}.pt"
        if path.exists():
            logger.info("%s: %s already prepared at %s", dataset, split, path)
            continue
        source = fetch(str(destination), train=train, download=True)
        media = torch.as_tensor(source.data).moveaxis(-1, -3).contiguous()
        # Write to a sibling then rename: an interrupted download must not leave
        # a truncated .pt that the existence check above would later accept.
        staging = path.with_suffix(".pt.partial")
        torch.save({"media": media, "label": torch.as_tensor(source.targets)}, staging)
        staging.replace(path)
        logger.info("%s: wrote %d %s images to %s", dataset, len(media), split, path)


class _EvalLoader:
    """Sequential batches of normalized test images."""

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
        return (len(self.media) + self.batch_size - 1) // self.batch_size


class _TrainLoader:
    """Shuffled, augmented batches that resume inside a pass."""

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
        count = len(self.media)
        if self._order is None:
            stream_seed = int(torch.empty((), dtype=torch.int64).random_())
            shuffle_seed = int(torch.empty((), dtype=torch.int64).random_())
            self._order = torch.randperm(
                count,
                generator=torch.Generator().manual_seed(shuffle_seed),
            )
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
            yield {
                "media": self.augment(
                    self.media[index],
                    self._streams[batch_index % self.num_streams],
                    pad=self.pad,
                    mean=self.mean,
                    std=self.std,
                ),
                "label": self.label[index],
            }
        self._order = None
        self._streams = []
        self._next_batch = 0

    def __len__(self) -> int:
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
        state = cast(_TrainLoader.StateDict, state_dict)
        self._order = state["order"]
        self._streams = [
            torch.Generator().set_state(stream) for stream in state["streams"]
        ]
        self._next_batch = state["next_batch"]


def _load_split(path: Path) -> tuple[Tensor, Tensor]:
    """Load one prepared split as uint8 images and integer labels."""
    payload = cast(
        dict[str, Tensor],
        torch.load(path, map_location="cpu", weights_only=True),
    )
    return payload["media"], payload["label"]
