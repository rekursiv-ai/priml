"""CIFAR-10 loading, served entirely from device memory.

The whole dataset is 50000 32x32 RGB images -- about 180 MB in float32 -- so it
fits in the memory of any device that can train on it. Holding it resident and
slicing batches with an index permutation removes the host-to-device copy and
the worker processes a general-purpose loader needs, which matters here because
a step on this model takes single-digit milliseconds and would otherwise be
dominated by input latency.

:func:`prepare` downloads and caches the tensors; ``scripts/prepare_data.py``
is its command-line front end. :class:`Cifar10Data` only reads that cache, so
constructing a config never touches the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast, override

import logging

from configgle import Fig
from torch import Tensor

import torch

from priml.math.pixel import rgb2float
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class Cifar10Data:
    """CIFAR-10 held in device memory, yielding ``media`` / ``label`` batches.

    Emits no augmentation: the train step owns it, so an experiment can change
    the augmentation policy without touching the input pipeline.

    Raises:
      FileNotFoundError: If the prepared tensors are absent. Run
        ``uv --quiet run --frozen python -m
        priml.baselines.cifar10.scripts.prepare_data`` first.

    """

    class Config(Fig["Cifar10Data"]):
        """Where the prepared tensors live and how batches are shaped."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/cifar10"
        """Logical directory holding ``train.pt`` and ``test.pt``.

        Resolved beneath ``base_dir`` at finalize, so it names a location
        within the resource root rather than an absolute filesystem path."""

        batch_size: int = 512
        """Images per training batch."""

        eval_batch_size: int = 1000
        """Images per evaluation batch."""

        drop_last: bool = False
        """Drop a final short training batch instead of yielding it."""

        device: str = "auto"
        """Device holding the resident tensors ("auto" picks the best)."""

        dtype: torch.dtype = torch.float32
        """Storage dtype for the resident images."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0 or config.eval_batch_size <= 0:
            raise ValueError(
                "batch_size and eval_batch_size must be positive; got "
                f"{config.batch_size} and {config.eval_batch_size}.",
            )
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the data; ticked by the loop when the loader runs out."""

        device = get_device(config.device)
        directory = Path(config.working_dir)
        self.train_media, self.train_label = _load_split(
            directory / "train.pt",
            device=device,
            dtype=config.dtype,
        )
        self.eval_media, self.eval_label = _load_split(
            directory / "test.pt",
            device=device,
            dtype=config.dtype,
        )

    def train_dataloader(self) -> _BatchIterator:
        """Return a shuffling iterator over the training split."""
        return _BatchIterator(
            self.train_media,
            self.train_label,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=self.config.drop_last,
        )

    def eval_dataloader(self) -> _BatchIterator:
        """Return a sequential iterator over the test split."""
        return _BatchIterator(
            self.eval_media,
            self.eval_label,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the pass count; batch ORDER derives from the loop's RNG."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the pass count (see :meth:`state_dict`)."""
        if "timer_epoch" in state_dict:
            self.timer_epoch.load_state_dict(state_dict["timer_epoch"])


def prepare(
    directory: Path | str,
    *,
    mean: Sequence[float] = (0.4914, 0.4822, 0.4465),
    std: Sequence[float] = (0.2470, 0.2435, 0.2616),
) -> None:
    """Download CIFAR-10 and cache normalized tensors under ``directory``.

    Idempotent: a split whose ``.pt`` file already exists is left untouched, so
    this is safe to call at the start of every run. Normalization is baked into
    the cache because it is a fixed property of the dataset, not an experimental
    choice -- an experiment that wants different statistics writes its own cache
    directory.

    Args:
      directory: Destination for ``train.pt`` and ``test.pt``.
      mean: Per-channel mean subtracted from images scaled to ``[0, 1]``.
      std: Per-channel standard deviation divided out after centering.

    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        path = destination / f"{split}.pt"
        if path.exists():
            logger.info("cifar10: %s already prepared at %s", split, path)
            continue
        # Imported per missing split, not at module scope: torchvision.datasets
        # pulls in PIL and the full dataset registry (~400ms). Every training
        # run calls this function, and the common case -- data already prepared
        # -- returns above without ever needing it.
        import torchvision.datasets  # noqa: PLC0415 -- 400ms import; only a missing split needs it

        source = torchvision.datasets.CIFAR10(
            str(destination),
            train=train,
            download=True,
        )
        media = rgb2float(
            torch.as_tensor(source.data).moveaxis(-1, -3).float(),
            inplace=True,
            unit_interval=True,
        )
        media = (media - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(
            1,
            3,
            1,
            1,
        )
        # Write to a sibling then rename: an interrupted download must not leave
        # a truncated .pt that the existence check above would later accept.
        staging = path.with_suffix(".pt.partial")
        torch.save(
            {"media": media, "label": torch.as_tensor(source.targets)},
            staging,
        )
        staging.replace(path)
        logger.info("cifar10: wrote %d %s images to %s", len(media), split, path)


class _BatchIterator:
    """Slices resident tensors into batches, optionally in shuffled order."""

    def __init__(
        self,
        media: Tensor,
        label: Tensor,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.media = media
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        count = len(self.media)
        order = (
            torch.randperm(count, device=self.media.device)
            if self.shuffle
            else torch.arange(count, device=self.media.device)
        )
        for start in range(0, count, self.batch_size):
            index = order[start : start + self.batch_size]
            if self.drop_last and len(index) < self.batch_size:
                return
            yield {"media": self.media[index], "label": self.label[index]}

    def __len__(self) -> int:
        count = len(self.media)
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size


def _load_split(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Load one prepared split onto ``device`` in channels-last layout."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared CIFAR-10 split not found at {path}. Run `uv --quiet run "
            "--frozen python -m priml.baselines.cifar10.scripts.prepare_data` "
            "first.",
        )
    payload = cast(
        dict[str, Tensor],
        torch.load(path, map_location="cpu", weights_only=True),
    )
    # A cache written by some other tool can occupy this path with the same
    # filename and different keys; without this check that surfaces as a bare
    # KeyError from an unrelated line.
    missing = {"media", "label"} - set(payload)
    if missing:
        raise ValueError(
            f"{path} is not a prepared CIFAR-10 split: missing {sorted(missing)}. "
            "Delete it and re-run `uv --quiet run --frozen python -m "
            "priml.baselines.cifar10.scripts.prepare_data`, or pass "
            "`--override dataset.working_dir=/datasets/...` to read a different "
            "directory (the path is logical, resolved beneath `base_dir`).",
        )
    media = payload["media"].to(device=device, dtype=dtype)
    if device.type != "mps":
        # channels_last matches the layout cuDNN's convolution kernels want;
        # MPS has no such preference and rejects the memory format on some
        # torch releases.
        media = media.contiguous(memory_format=torch.channels_last)
    return media, payload["label"].to(device)
