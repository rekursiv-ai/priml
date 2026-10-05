#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Crop extracted ImageNet and encode paired 32-channel INVAE latents.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import json

from PIL import Image

from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.model.invae import encode_image, load_invae


if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")
    torch = lazy_import("torch")


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Crop an image with the ADM preprocessing sequence.

    Args:
      image: Source image.
      size: Square output side length.

    Returns:
      cropped: Box-downsampled, bicubic-resized, center-cropped image.

    """
    while min(image.size) >= 2 * size:
        image = image.resize(
            (image.width // 2, image.height // 2),
            Image.Resampling.BOX,
        )
    scale = size / min(image.size)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.BICUBIC,
    )
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def prepare(
    source: Path,
    output: Path,
    *,
    resolution: int = 256,
    checkpoint: Path | None = None,
    device: str = "cuda",
    limit: int | None = None,
) -> int:
    """Write source-compatible images, latents, and class metadata.

    Args:
      source: Extracted ImageNet directory.
      output: Destination for paired images and INVAE latents.
      resolution: Cropped image side length.
      checkpoint: Optional local INVAE checkpoint.
      device: Torch device used for INVAE encoding.
      limit: Optional maximum image count.

    Returns:
      count: Number of image-latent pairs written.

    Raises:
      ValueError: The requested resolution is unsupported.
      TypeError: An input record lacks a typed class label or path.

    """
    if resolution not in (256, 512):
        raise ValueError("resolution must be 256 or 512")
    image_source = ExtractedImageNetSource.Config(
        working_dir=source,
        split="train",
    ).make()
    labels = ImagenetSynsetToIndex.Config().make()
    vae = load_invae(checkpoint, device=device)
    metadata: list[list[str | int]] = []
    for index, record in enumerate(labels(iter(image_source))):
        if limit is not None and index >= limit:
            break
        class_id = record.get("label")
        file_path = record.get("file_path")
        if not isinstance(class_id, int):
            raise TypeError(f"expected an integer ImageNet label: {record}")
        if not isinstance(file_path, str):
            raise TypeError(f"expected an ImageNet file path: {record}")
        stem = f"{index:08d}"
        shard = stem[:5]
        image_path = output / "images" / shard / f"img{stem}.png"
        latent_path = output / "vae-in" / shard / f"img-latents-{stem}.npy"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(file_path) as opened:
            cropped = center_crop(opened.convert("RGB"), resolution)
        cropped.save(image_path)
        image = (
            torch.from_numpy(np.asarray(cropped).copy())
            .permute(2, 0, 1)[None]
            .to(device)
        )
        np.save(latent_path, encode_image(vae, image).cpu().numpy())
        metadata.append(
            [latent_path.relative_to(output / "vae-in").as_posix(), class_id],
        )
    manifest = output / "vae-in" / "dataset.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"labels": metadata}), encoding="utf-8")
    return len(metadata)


def main() -> int:
    """Prepare one paired ImageNet/INVAE corpus.

    Returns:
      exit_code: Zero after successful preparation.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    print(
        prepare(
            flags.source,
            flags.output,
            resolution=flags.resolution,
            checkpoint=flags.checkpoint,
            device=flags.device,
            limit=flags.limit,
        ),
    )
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    source: Path
    output: Path
    resolution: int
    checkpoint: Path | None
    device: str
    limit: int | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register corpus preparation flags."""
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256, choices=(256, 512))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
