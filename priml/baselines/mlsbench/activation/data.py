"""CIFAR-10, cached in the layout the vision loader reads.

The loader is :class:`~priml.baselines.mlsbench.pytorch_vision.data.Cifar100Data`:
nothing in it is specific to CIFAR-100 but its defaults, so the task's
``resnet20-cifar10`` setting fills it with CIFAR-10's directory and channel
statistics as values. Only fetching the images names the dataset, so that is
all this module holds. ``scripts/prepare_data.py`` is its command-line front
end.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import logging

import torch


if TYPE_CHECKING:
    from torchvision import datasets
else:
    from wrapt import lazy_import

    datasets = lazy_import("torchvision.datasets")


logger = logging.getLogger(__name__)


def prepare(directory: Path | str) -> None:
    """Download CIFAR-10 and cache its 8-bit tensors under ``directory``.

    Idempotent: a split whose ``.pt`` file already exists is left untouched.
    Pixels are stored as shipped; scaling and standardization belong to the
    loader, where they are part of the recipe.

    Args:
      directory: Destination for ``train.pt`` and ``test.pt``.

    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        path = destination / f"{split}.pt"
        if path.exists():
            logger.info("cifar10: %s already prepared at %s", split, path)
            continue
        source = datasets.CIFAR10(str(destination), train=train, download=True)
        media = torch.as_tensor(source.data).moveaxis(-1, -3).contiguous()
        # Write to a sibling then rename: an interrupted download must not leave
        # a truncated .pt that the existence check above would later accept.
        staging = path.with_suffix(".pt.partial")
        torch.save({"media": media, "label": torch.as_tensor(source.targets)}, staging)
        staging.replace(path)
        logger.info("cifar10: wrote %d %s images to %s", len(media), split, path)
