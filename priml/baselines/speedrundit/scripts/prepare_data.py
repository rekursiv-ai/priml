#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Crop extracted ImageNet and encode paired 32-channel INVAE latents.

--source is the extracted ILSVRC2012 training set: train/<synset>/*.JPEG,
one directory for each of the 1,000 synsets; nothing else is read. The
INVAE checkpoint is downloaded at a pinned revision and checked against
its SHA-256; --checkpoint names a local copy, checked the same way.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import hashlib
import json

from PIL import Image

from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.model.invae import encode_image, load_invae
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from huggingface_hub import hf_hub_download

    import numpy as np
    import torch
else:
    from wrapt import lazy_import

    hf_hub_download = lazy_import("huggingface_hub", "hf_hub_download")
    np = lazy_import("numpy")
    torch = lazy_import("torch")


INVAE_REPOSITORY: Final = "REPA-E/e2e-invae"
"""Hugging Face repository publishing the INVAE checkpoint."""

INVAE_FILE: Final = "e2e-invae-400k.pt"
"""Checkpoint file within ``INVAE_REPOSITORY``."""

INVAE_REVISION: Final = "79e3ea77928e1e34bc70e4e71de68fd5b4d4b3e0"
"""Commit of ``INVAE_REPOSITORY`` the checkpoint is fetched at."""

INVAE_SHA256: Final = "4d45ef5452bd6325eaa4b5c74975328cfed5ecf15a877656bed0f3f6de23d718"
"""SHA-256 of ``INVAE_FILE`` at ``INVAE_REVISION``, as its LFS pointer records it."""


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
      checkpoint: Optional local copy of the pinned INVAE checkpoint.
      device: Torch device used for INVAE encoding.
      limit: Optional maximum image count.

    Returns:
      count: Number of image-latent pairs written.

    Raises:
      ValueError: The requested resolution is unsupported, or ``output`` is
        unusable or aliases ``source``.
      TypeError: An input record lacks a typed class label or path.

    """
    if resolution not in (256, 512):
        raise ValueError("resolution must be 256 or 512")
    output = validated_output_path(output, protected=[source])
    image_source = ExtractedImageNetSource.Config(
        working_dir=source,
        split="train",
    ).make()
    labels = ImagenetSynsetToIndex.Config().make()
    vae = load_invae(invae_checkpoint(checkpoint), device=device)
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
            cropped = center_crop(opened.convert("RGB"), size=resolution)
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


def require_imagenet_train(source: Path) -> None:
    """Refuse a source whose ``train/`` is not exactly the 1,000 synset directories.

    Args:
      source: Extracted ImageNet directory.

    Raises:
      ValueError: A synset directory is missing, or an unexpected one is present.

    """
    synsets = set(ImagenetSynsetToIndex.Config().make().synset_to_idx)
    present = {path.name for path in (source / "train").iterdir() if path.is_dir()}
    if present != synsets:
        raise ValueError(
            f"{source / 'train'} must hold one directory per ImageNet synset; "
            f"{len(synsets - present)} are missing and "
            f"{len(present - synsets)} are unexpected.",
        )


def invae_checkpoint(checkpoint: Path | None) -> Path:
    """Return the INVAE checkpoint to encode with, verified against its digest.

    Args:
      checkpoint: A local copy, or ``None`` to download ``INVAE_FILE`` at
        ``INVAE_REVISION`` into the Hugging Face cache.

    Returns:
      path: The checkpoint, its bytes hashing to ``INVAE_SHA256``.

    Raises:
      ValueError: The bytes hash to anything else.

    """
    path = (
        checkpoint
        if checkpoint is not None
        else Path(
            hf_hub_download(
                repo_id=INVAE_REPOSITORY,
                filename=INVAE_FILE,
                revision=INVAE_REVISION,
            ),
        )
    )
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != INVAE_SHA256:
        raise ValueError(f"{path} hashes to {digest}, not the pinned {INVAE_SHA256}.")
    return path


def main() -> int:
    """Prepare one paired ImageNet/INVAE corpus.

    Returns:
      exit_code: Zero after successful preparation.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    # A partial extraction would otherwise train on fewer classes without a word.
    require_imagenet_train(flags.source)
    print(
        prepare(
            flags.source,
            output=flags.output,
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
