#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Verify an extracted ImageNet and write its validation labels beside it.

The train split is ``train/<synset>/*.JPEG`` (1,000 synsets) and the
validation split a flat ``val/*.JPEG`` (50,000 images). The archive carries no
validation labels, so this writes ``validation_labels.txt`` -- one synset per
line, in sorted-filename order -- from the ILSVRC2012 devkit ground truth.
Idempotent: an existing labels file is left alone.

The default location matches the one ``FfcvImageNetData`` resolves under a
default ``TrainLoop``, so preparing and training agree without either naming a
path.

Examples:
  prepare_data.py
  prepare_data.py --directory /datasets/my-imagenet

'''
# fmt: on

from __future__ import annotations

from http.client import HTTPResponse
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import logging
import urllib.request

from priml.baselines.imagenet.data import NUM_CLASSES, ImageNetData
from priml.data.pipeline.dataset import DataPipeline
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.train.train_loop import TrainLoop


logger = logging.getLogger(__name__)

NUM_VAL_IMAGES: Final = 50_000

VALIDATION_LABELS_URL: Final = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "8b12ae202a3ccf8f965c730a4e7617204e32000b/research/slim/datasets/"
    "imagenet_2012_validation_synset_labels.txt"
)
"""The devkit's validation ground truth as synsets, in ``ILSVRC2012_val_%08d``
order -- which is sorted-filename order."""


def main() -> int:
    """Verify the dataset; return the process exit code.

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
    prepare(flags.directory)
    return 0


def default_directory() -> Path:
    """Return the dataset directory a default ``TrainLoop`` would resolve.

    Returns:
      directory: Extracted-archive location shared by the preparer and the loop.

    """
    config = ImageNetData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    train = config.copy_tree().finalize().train_data_pipeline
    assert isinstance(train, DataPipeline.Config)
    source = train.source
    assert isinstance(source, ExtractedImageNetSource.Config)
    return Path(source.working_dir)


def prepare(directory: Path) -> None:
    """Verify the extracted layout and write ``validation_labels.txt``.

    Args:
      directory: Extracted ImageNet root holding ``train/`` and ``val/``.

    Raises:
      FileNotFoundError: ``train/`` or ``val/`` is missing.
      ValueError: A split, or the downloaded labels, has the wrong count.

    """
    for split in ("train", "val"):
        if not (directory / split).is_dir():
            raise FileNotFoundError(
                f"{directory / split} is missing; extract ImageNet there first.",
            )
    num_classes = sum(1 for p in (directory / "train").iterdir() if p.is_dir())
    if num_classes != NUM_CLASSES:
        raise ValueError(f"train/ holds {num_classes} synsets; expected {NUM_CLASSES}.")
    num_val = sum(1 for _ in (directory / "val").glob("*.JPEG"))
    if num_val != NUM_VAL_IMAGES:
        raise ValueError(f"val/ holds {num_val} images; expected {NUM_VAL_IMAGES}.")
    labels = directory / "validation_labels.txt"
    if not labels.exists():
        text = _download_text(VALIDATION_LABELS_URL)
        if len(text.split()) != NUM_VAL_IMAGES:
            raise ValueError(f"{VALIDATION_LABELS_URL} is not {NUM_VAL_IMAGES} lines.")
        staging = labels.with_suffix(".partial")
        _ = staging.write_text(text)
        _ = staging.replace(labels)
    logger.info("%s is ready.", directory)


def _download_text(url: str) -> str:
    """Return the body at ``url`` decoded as UTF-8."""
    response = cast(HTTPResponse, urllib.request.urlopen(url))  # noqa: S310 -- Callers pass a fixed https URL.
    with response:
        return response.read().decode()


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        type=Path,
        default=default_directory(),
        help="Extracted ImageNet root holding train/ and val/.",
    )


class _Flags(Protocol):
    """Parsed command-line flags."""

    directory: Path


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
