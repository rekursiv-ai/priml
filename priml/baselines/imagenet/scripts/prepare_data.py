#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Verify an extracted ImageNet and write its validation labels beside it.

ILSVRC2012 is licensed, so it is never downloaded here: extract it yourself
(the README's "Reproduce" section gives the steps). The train split is
``train/<synset>/*.JPEG`` (the 1,000 ILSVRC2012 synsets, 1,281,167 images) and
the validation split a flat ``val/ILSVRC2012_val_%08d.JPEG`` (50,000 images).
Every count and name is checked. The archive carries no validation labels, so
this writes ``validation_labels.txt`` -- one synset per line, in sorted-filename
order -- from the checksum-pinned ILSVRC2012 devkit ground truth. Idempotent:
an existing labels file is verified and left alone.

The default location matches the one ``ImageNetData`` resolves under a default
``TrainLoop``, so preparing and training agree without either naming a path.

Examples:
  prepare_data.py
  prepare_data.py --directory /data/imagenet

'''
# fmt: on

from __future__ import annotations

from http.client import HTTPResponse
from pathlib import Path
from typing import Final, Protocol, cast
from urllib import request

import argparse
import hashlib
import logging

from priml.baselines.imagenet.data import NUM_TRAIN_SAMPLES, ImageNetData
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

VALIDATION_LABELS_SHA256: Final = (
    "313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604"
)


def main() -> int:
    """Verify the dataset; return the process exit code.

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

    The labels are verified first: their synsets are the ones ``train/`` must
    hold, one directory each.

    Args:
      directory: Extracted ImageNet root holding ``train/`` and ``val/``.

    Raises:
      FileNotFoundError: ``train/`` or ``val/`` is missing.
      ValueError: The labels fail their checksum, or a split holds other
        synsets, image counts, or file names than ILSVRC2012.

    """
    for split in ("train", "val"):
        if not (directory / split).is_dir():
            raise FileNotFoundError(
                f"{directory / split} is missing; extract ImageNet there first.",
            )
    labels = directory / "validation_labels.txt"
    text = (
        labels.read_text() if labels.exists() else _download_text(VALIDATION_LABELS_URL)
    )
    if hashlib.sha256(text.encode()).hexdigest() != VALIDATION_LABELS_SHA256:
        raise ValueError(
            f"Validation labels SHA-256 mismatch; delete {labels} if it exists "
            "and rerun.",
        )
    _verify_train(directory / "train", synsets=set(text.split()))
    _verify_val(directory / "val")
    if not labels.exists():
        staging = labels.with_suffix(".partial")
        _ = staging.write_text(text)
        _ = staging.replace(labels)
    logger.info("%s is ready.", directory)


def _verify_train(train: Path, *, synsets: set[str]) -> None:
    """Require one directory per synset and the published image count."""
    found = {path.name for path in train.iterdir() if path.is_dir()}
    if found != synsets:
        raise ValueError(
            f"{train} holds {len(found)} synset directories; expected the "
            f"{len(synsets)} ILSVRC2012 synsets. Missing: "
            f"{sorted(synsets - found)[:3]}; "
            f"unexpected: {sorted(found - synsets)[:3]}.",
        )
    num_images = sum(1 for name in found for _ in (train / name).glob("*.JPEG"))
    if num_images != NUM_TRAIN_SAMPLES:
        raise ValueError(
            f"{train} holds {num_images} images; expected {NUM_TRAIN_SAMPLES}.",
        )


def _verify_val(val: Path) -> None:
    """Require exactly the ILSVRC2012 validation file names, flat in ``val/``."""
    found = {path.name for path in val.glob("*.JPEG")}
    expected = {
        f"ILSVRC2012_val_{index:08d}.JPEG" for index in range(1, NUM_VAL_IMAGES + 1)
    }
    if found != expected:
        raise ValueError(
            f"{val} holds {len(found)} images; expected the {NUM_VAL_IMAGES} "
            "ILSVRC2012_val_%08d.JPEG files. Missing: "
            f"{sorted(expected - found)[:3]}; "
            f"unexpected: {sorted(found - expected)[:3]}.",
        )


def _download_text(url: str) -> str:
    """Return the body at ``url`` decoded as UTF-8."""
    response = cast(HTTPResponse, request.urlopen(url))  # noqa: S310 -- Callers pass a fixed https URL.
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
