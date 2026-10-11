"""Read images and class labels from a prepared ImageNet latent corpus."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import math

from treekle.wire import json_loads

import numpy as np

from priml.data.arrays import load_array


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")


def read_labels(manifest: Path) -> dict[str, int]:
    """Read class labels keyed by slash-separated latent names.

    Args:
      manifest: JSON manifest containing image names and labels.

    Returns:
      labels: Class labels keyed by normalized latent names.

    """
    payload = json_loads(manifest.read_bytes(), _LabelManifest)
    return {name.replace("\\", "/"): label for name, label in payload["labels"]}


def read_image(path: Path) -> NDArray[np.uint8]:
    """Decode a stored image to ``[channels, height, width]`` uint8.

    Args:
      path: NumPy or image file to decode.

    Returns:
      image: Channel-first uint8 image.

    """
    if path.suffix.lower() == ".npy":
        try:
            array = load_array(path, np.uint8)
        except TypeError as error:
            raise ValueError(
                f"Stored image {path} must be uint8 with at least three dimensions.",
            ) from error
        if array.ndim < 3:
            raise ValueError(
                f"Stored image {path} must be uint8 with at least three dimensions.",
            )
        shape = cast("tuple[int, ...]", array.shape)
        return array.reshape(math.prod(shape[:-2]), *shape[-2:])
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"))
    return array.transpose(2, 0, 1)


class _LabelManifest(TypedDict):
    labels: list[tuple[str, int]]
