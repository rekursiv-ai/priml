#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Stage the fixed image set the throughput experiments time.

The set is laid out as an extracted ImageNet train split,
``train/<synset>/<synset>_<n>.JPEG``, so the timed pipeline reads it through
the same source as training. Every pass drains the whole set, so it is kept
small. Two kinds:

* ``--source DIR`` links images of a real extracted ImageNet, taken round
  robin over its synsets in sorted order: the number on reference hardware.
* Without it, writes seeded synthetic JPEGs (a smooth colour field plus noise,
  300-500 pixels a side, quality 90, 4:2:0): runs anywhere, redistributable,
  and a stand-in only. Its file sizes and content are not ImageNet's.

Idempotent: an existing set of the requested kind and size is left alone, and
one of another kind or size is refused. The set is built in a fresh
directory beside ``train/`` and renamed into place only when complete, so an
interrupted run leaves no ``train/`` for a later run to mistake for a
finished one.

Examples:
  prepare_throughput_data.py
  prepare_throughput_data.py --source /datasets/imagenet

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import itertools
import logging
import shutil
import tempfile

from priml.baselines.imagenet.throughput.experiments import exp000
from priml.data.pipeline.dataset import DataPipeline
from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image

    import numpy as np
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")
    np = lazy_import("numpy")


logger = logging.getLogger(__name__)


def main() -> int:
    """Stage the image set; return the process exit code.

    Returns:
      code: 0 on success.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _ = stage(
        flags.directory,
        source=flags.source,
        num_images=flags.num_images,
        num_classes=flags.num_classes,
    )
    return 0


def stage(
    directory: Path,
    *,
    source: Path | None,
    num_images: int,
    num_classes: int,
) -> Path:
    """Build the image set under ``directory`` unless a finished one is there.

    Args:
      directory: Where the set goes; it holds ``train/``.
      source: An extracted ImageNet root to link from, or None for synthetic.
      num_images: Images in the set.
      num_classes: Synsets the synthetic images are spread across.

    Returns:
      train: The finished ``train/`` directory.

    Raises:
      ValueError: A count is out of range, or ``train/`` holds a set of
        another kind or size.

    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive; got {num_images}.")
    num_synsets = len(_synsets())
    if source is None and (num_classes < 1 or num_classes > num_synsets):
        raise ValueError(
            f"num_classes must lie in [1, {num_synsets}]; got {num_classes}.",
        )
    train = validated_output_path(directory) / "train"
    if train.exists():
        _require_kind(train, linked=source is not None, num_images=num_images)
        logger.info("%s exists; leaving it alone.", train)
        return train
    train.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix="train.partial-", dir=train.parent))
    try:
        # `mkdtemp` makes it private; the finished set is for every reader.
        partial.chmod(0o755)
        if source is None:
            write_synthetic(partial, num_images=num_images, num_classes=num_classes)
        else:
            link_subset(source / "train", partial, num_images=num_images)
        partial.rename(train)
    except BaseException:
        shutil.rmtree(partial)
        raise
    logger.info("%s is ready.", train)
    return train


def default_directory() -> Path:
    """Return the image-set directory ``exp000`` resolves under its defaults.

    Returns:
      directory: Location shared by this script and the throughput experiments.

    """
    pipeline = exp000().copy_tree().finalize().pipeline
    assert isinstance(pipeline, DataPipeline.Config)
    assert isinstance(pipeline.source, ExtractedImageNetSource.Config)
    return Path(pipeline.source.working_dir)


def write_synthetic(
    train: Path,
    *,
    num_images: int,
    num_classes: int,
    sides: tuple[int, int] = (300, 500),
) -> None:
    """Write ``num_images`` seeded JPEGs across the first ``num_classes`` synsets.

    Args:
      train: Destination ``train/`` directory.
      num_images: Images to write.
      num_classes: Synsets to spread them across, in canonical label order.
      sides: Least and greatest height and width, inclusive.

    """
    synsets = _synsets()[:num_classes]
    rng = np.random.default_rng(0)
    for index in range(num_images):
        synset = synsets[index % num_classes]
        (train / synset).mkdir(exist_ok=True)
        Image.fromarray(synthetic_pixels(rng, sides=sides)).save(
            train / synset / f"{synset}_{index}.JPEG",
            quality=90,
        )


def synthetic_pixels(
    rng: np.random.Generator,
    *,
    sides: tuple[int, int],
) -> NDArray[np.uint8]:
    """Return one image: a smooth colour field plus noise.

    Args:
      rng: The generator every image of the set draws from, in order.
      sides: Least and greatest height and width, inclusive.

    Returns:
      pixels: A ``(height, width, 3)`` RGB image.

    """
    least, most = sides
    height, width = (
        int(rng.integers(least, most + 1)),
        int(rng.integers(least, most + 1)),
    )
    coarse = rng.integers(0, 256, (height // 16 + 2, width // 16 + 2, 3), np.uint8)
    smooth = np.asarray(
        Image.fromarray(coarse).resize((width, height), Image.Resampling.BICUBIC),
    )
    noise = rng.integers(-12, 13, (height, width, 3))
    return np.clip(smooth + noise, 0, 255).astype(np.uint8)


def link_subset(source: Path, train: Path, *, num_images: int) -> None:
    """Symlink ``num_images`` images under ``source`` into ``train``, round robin.

    Each synset's images are taken in sorted order, one synset at a time in
    sorted order, so every synset contributes before any contributes twice
    and a synset that runs out leaves its share to the rest.

    Args:
      source: An extracted ImageNet ``train/`` directory.
      train: Destination ``train/`` directory.
      num_images: Images to link.

    Raises:
      ValueError: ``source`` holds fewer than ``num_images`` images.

    """
    synsets = sorted(d for d in source.resolve().iterdir() if d.is_dir())
    rounds = itertools.zip_longest(*(sorted(s.glob("*.JPEG")) for s in synsets))
    images = [image for row in rounds for image in row if image is not None]
    if len(images) < num_images:
        raise ValueError(
            f"{source} holds {len(images)} images; {num_images} requested.",
        )
    for image in images[:num_images]:
        (train / image.parent.name).mkdir(parents=True, exist_ok=True)
        (train / image.parent.name / image.name).symlink_to(image)


def _synsets() -> list[str]:
    """Return every synset in canonical label order."""
    return list(ImagenetSynsetToIndex(ImagenetSynsetToIndex.Config()).synset_to_idx)


def _require_kind(train: Path, *, linked: bool, num_images: int) -> None:
    """Raise unless ``train`` holds ``num_images`` images, linked or written."""
    images = list(train.glob("*/*.JPEG"))
    kinds = {image.is_symlink() for image in images}
    if len(images) != num_images or kinds != {linked}:
        want = "linked" if linked else "synthetic"
        raise ValueError(
            f"{train} holds {len(images)} images, not {num_images} {want} ones; "
            "move it aside to stage a new set.",
        )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        type=Path,
        default=default_directory(),
        help="Where the image set is staged; holds train/.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Extracted ImageNet root to link from; omit for synthetic JPEGs.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8_192,
        help="Images in the set: sixteen of exp000's 512-image batches.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=16,
        help="Synsets the synthetic images are spread across; unused with --source.",
    )


class _Flags(Protocol):
    """Parsed command-line flags."""

    directory: Path
    source: Path | None
    num_images: int
    num_classes: int


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
