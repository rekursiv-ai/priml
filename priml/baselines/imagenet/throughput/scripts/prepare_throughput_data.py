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

* ``--source DIR`` links the first images of each synset of a real extracted
  ImageNet, in sorted order: the number on reference hardware.
* Without it, writes seeded synthetic JPEGs (a smooth colour field plus noise,
  300-500 pixels a side, quality 90, 4:2:0): runs anywhere, redistributable,
  and a stand-in only. Its file sizes and content are not ImageNet's.

Idempotent: an existing set is left alone.

Examples:
  prepare_throughput_data.py
  prepare_throughput_data.py --source /datasets/imagenet

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import logging

from PIL import Image

import numpy as np

from priml.baselines.imagenet.throughput.experiments import exp000
from priml.data.pipeline.dataset import DataPipeline
from priml.data.processors import labels
from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource


logger = logging.getLogger(__name__)


def main() -> int:
    """Stage the image set; return the process exit code.

    Returns:
      code: 0 on success.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train = flags.directory / "train"
    if train.exists():
        logger.info("%s exists; leaving it alone.", train)
        return 0
    if flags.source is None:
        write_synthetic(
            train,
            num_images=flags.num_images,
            num_classes=flags.num_classes,
        )
    else:
        link_subset(flags.source / "train", train, num_images=flags.num_images)
    logger.info("%s is ready.", flags.directory)
    return 0


def default_directory() -> Path:
    """Return the image-set directory ``exp000`` resolves under its defaults.

    Returns:
      directory: Location shared by this script and the throughput experiments.

    """
    pipeline = exp000().copy_tree().finalize().pipeline
    assert isinstance(pipeline, DataPipeline.Config)
    assert isinstance(pipeline.source, ExtractedImageNetSource.Config)
    return Path(pipeline.source.working_dir)


def write_synthetic(train: Path, *, num_images: int, num_classes: int) -> None:
    """Write ``num_images`` seeded JPEGs across the first ``num_classes`` synsets.

    Args:
      train: Destination ``train/`` directory.
      num_images: Images to write.
      num_classes: Synsets to spread them across, in canonical label order.

    """
    labels_file = ImagenetSynsetToIndex.Config().labels_file
    synsets = (Path(labels.__file__).with_name(str(labels_file)).read_text().split())[
        :num_classes
    ]
    rng = np.random.default_rng(0)
    for index in range(num_images):
        height, width = int(rng.integers(300, 501)), int(rng.integers(300, 501))
        coarse = rng.integers(0, 256, (height // 16 + 2, width // 16 + 2, 3), np.uint8)
        smooth = np.asarray(
            Image.fromarray(coarse).resize((width, height), Image.Resampling.BICUBIC),
        )
        noise = rng.integers(-12, 13, (height, width, 3))
        pixels = np.clip(smooth.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        synset = synsets[index % num_classes]
        (train / synset).mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(
            train / synset / f"{synset}_{index}.JPEG",
            quality=90,
        )


def link_subset(source: Path, train: Path, *, num_images: int) -> None:
    """Symlink the first images of each synset under ``source`` into ``train``.

    Args:
      source: An extracted ImageNet ``train/`` directory.
      train: Destination ``train/`` directory.
      num_images: Images to link, spread evenly over every synset.

    """
    synsets = sorted(d for d in source.iterdir() if d.is_dir())
    per_class = -(-num_images // len(synsets))
    linked = 0
    for synset in synsets:
        (train / synset.name).mkdir(parents=True, exist_ok=True)
        for image in sorted(synset.glob("*.JPEG"))[
            : min(per_class, num_images - linked)
        ]:
            (train / synset.name / image.name).symlink_to(image)
            linked += 1
        if linked == num_images:
            break


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
        help="Synsets the synthetic images are spread across.",
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
