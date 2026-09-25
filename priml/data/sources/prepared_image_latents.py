"""Read images and class labels from a prepared ImageNet latent corpus."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from priml.lib.custom_json import DictCodec, IntCodec, ListCodec, StrCodec, loads


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")


def read_labels(manifest: Path) -> dict[str, int]:
    """Read class labels keyed by slash-separated latent names."""
    payload = DictCodec.coerce(loads(manifest.read_text(encoding="utf-8")))
    entries = [ListCodec.coerce(entry) for entry in ListCodec.coerce(payload["labels"])]
    return {
        StrCodec.coerce(entry[0]).replace("\\", "/"): IntCodec.coerce(
            entry[1], default=None
        )
        for entry in entries
    }


def read_image(path: Path) -> NDArray[np.uint8]:
    """Decode a stored image to ``[channels, height, width]`` uint8."""
    if path.suffix.lower() == ".npy":
        array = cast("NDArray[np.uint8]", np.load(path))
        shape = cast("tuple[int, ...]", array.shape)
        return array.reshape(-1, *shape[-2:])
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"))
    return array.transpose(2, 0, 1)
