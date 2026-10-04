#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Crop extracted ImageNet and encode the latents an experiment trains on.

The experiment names the autoencoder, the storage codec, and the corpus
directory; this script builds exactly those and writes the corpus with its
receipt. Adding an autoencoder never touches this file: a new experiment
declares it. A fitted codec (uint8 tables) is fitted first on a subset of the
images and its table is written before any latent. Re-running resumes: images
and latents already on disk are kept.

Examples:
  prepare_data.py --experiment exp000 --source /datasets/imagenet
  prepare_data.py --experiment exp003 --source /datasets/imagenet --device cuda
  prepare_data.py --experiment exp000 --receipt-only

'''
# fmt: on

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import hashlib
import json
import logging
import random

from PIL import Image

from priml.baselines.speedrundit import experiments
from priml.baselines.speedrundit.corpus import (
    LABELS,
    load_table,
    save_stored,
    save_table,
    table_path,
    write_receipt,
)
from priml.baselines.speedrundit.latent_codec import FittedCodec, ScalarTableCodec
from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch import Tensor

    import numpy as np
    import torch

    from priml.baselines.speedrundit.data import PairedImageLatentDataset
    from priml.baselines.speedrundit.latent_codec import LatentCodec
    from priml.model.vision_ae.custom_types import Autoencoder
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")
    torch = lazy_import("torch")


logger = logging.getLogger(__name__)

FIT_SAMPLE_SEED: Final = 0
"""Seeds the choice of a fitted codec's images; part of the table's identity.

Changing it changes every table fitted after, which the receipt's
``indices_sha256`` records; a table already on disk is reused, never refitted.
"""


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


@dataclass(frozen=True, slots=True)
class Record:
    """One training image and the identifier its corpus files take."""

    source: str
    """Extracted ImageNet file."""

    label: int
    """Class index."""

    stem: str
    """Eight-digit corpus identifier, in source order."""


def dataset_config(experiment: str) -> PairedImageLatentDataset.Config:
    """Return the finalized latent source the named experiment trains on.

    Args:
      experiment: Name of a factory in
        :mod:`~priml.baselines.speedrundit.experiments`.

    Returns:
      config: Its ``dataset.source``, paths resolved beneath the run's base.

    """
    factory = cast("_Experiment", getattr(experiments, experiment))
    return factory().copy_tree().finalize().dataset.source


def records(imagenet: Path, *, limit: int | None = None) -> list[Record]:
    """List training images in the extracted source's deterministic order.

    Args:
      imagenet: Extracted ImageNet directory.
      limit: Keep only the first ``limit`` images.

    Returns:
      records: One per image, numbered in source order.

    Raises:
      TypeError: A record lacks a typed class label or path.

    """
    source = ExtractedImageNetSource.Config(working_dir=imagenet, split="train").make()
    labels = ImagenetSynsetToIndex.Config().make()
    found: list[Record] = []
    for index, record in enumerate(labels(iter(source))):
        if limit is not None and index >= limit:
            break
        label = record.get("label")
        path = record.get("file_path")
        if not isinstance(label, int):
            raise TypeError(f"expected an integer ImageNet label: {record}")
        if not isinstance(path, str):
            raise TypeError(f"expected an ImageNet file path: {record}")
        found.append(Record(path, label, f"{index:08d}"))
    return found


def fit_sample_indices(num_records: int, num_images: int) -> list[int]:
    """Choose which images a fitted codec's tables are estimated from.

    Args:
      num_records: Images in the corpus, in source (synset-sorted) order.
      num_images: How many the codec asks for; fewer if the corpus is smaller.

    Returns:
      indices: Distinct, sorted positions in ``[0, num_records)``, a uniform
        sample drawn with :data:`FIT_SAMPLE_SEED`.

    """
    # A private generator: ``prepare`` seeds torch's global one for the encoder's
    # posterior draws, so a selection drawn from it would shift every latent after it.
    count = min(num_images, num_records)
    return sorted(random.Random(FIT_SAMPLE_SEED).sample(range(num_records), count))  # noqa: S311 -- This RNG only picks which images a codec is fitted on and never protects secrets.


def prepare(
    config: PairedImageLatentDataset.Config,
    imagenet: Path,
    *,
    device: str = "cuda",
    batch_size: int = 64,
    limit: int | None = None,
) -> int:
    """Write images, stored latents, labels, and the receipt for one corpus.

    Args:
      config: Finalized latent source, as :func:`dataset_config` returns it.
      imagenet: Extracted ImageNet directory.
      device: Device the autoencoder encodes on.
      batch_size: Images encoded per forward.
      limit: Encode only the first ``limit`` images.

    Returns:
      count: Image/latent pairs in the corpus.

    """
    root = Path(config.working_dir)
    latent_dir = root / config.latent_subdir
    latent_dir.mkdir(parents=True, exist_ok=True)
    listed = records(imagenet, limit=limit)
    if config.seed is not None:
        torch.manual_seed(config.seed)
    autoencoder = config.autoencoder.make()
    if isinstance(autoencoder, torch.nn.Module):
        _ = autoencoder.to(device)
    codec = config.codec.make()
    fit: dict[str, object] | None = None
    table_sha256: str | None = None
    if isinstance(codec, FittedCodec):
        table_sha256, fit = _fit_or_load(
            codec,
            autoencoder,
            listed,
            root=root,
            latent_dir=latent_dir,
            size=config.autoencoder.image_size,
            device=device,
            batch_size=batch_size,
        )
    error = _ErrorTally()
    pending = [r for r in listed if not _latent_path(latent_dir, r).is_file()]
    for batch, images in batches(
        pending,
        root,
        config.autoencoder.image_size,
        batch_size,
    ):
        latents = encode_latents(autoencoder, images, device)
        stored = codec.encode(latents)
        error.add(latents, codec, stored)
        for index, record in enumerate(batch):
            path = _latent_path(latent_dir, record)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_stored(path, stored[index : index + 1])
    _write_labels(latent_dir, listed)
    write_receipt(
        latent_dir,
        autoencoder=config.autoencoder,
        codec_config=config.codec,
        codec=codec,
        table_sha256=table_sha256,
        details={
            "provenance": "encoded",
            "images": {
                "source": str(imagenet),
                "split": "train",
                "count": len(listed),
                "resolution": config.autoencoder.image_size,
                "crop": "adm_center_crop",
            },
            "encoding": {
                "batch_size": batch_size,
                "seed": config.seed,
                "device": device,
                "torch": torch.__version__,
                "newly_encoded": len(pending),
            },
            "error": error.summary(),
            "fit": fit,
        },
    )
    return len(listed)


def record_receipt(config: PairedImageLatentDataset.Config) -> int:
    """Write a receipt for a corpus encoded elsewhere, e.g. an existing REG one.

    Nothing is encoded or verified against pixels: the receipt only states that
    this corpus is taken to be the configured autoencoder's, which the loader
    then holds it to.

    Args:
      config: Finalized latent source.

    Returns:
      count: Latent files found.

    Raises:
      FileNotFoundError: The latent directory, or a fitted codec's table, is missing.

    """
    latent_dir = Path(config.working_dir) / config.latent_subdir
    if not latent_dir.is_dir():
        raise FileNotFoundError(f"{latent_dir} does not exist.")
    codec = config.codec.make()
    table_sha256 = (
        load_table(latent_dir, codec) if isinstance(codec, FittedCodec) else None
    )
    count = sum(1 for _ in latent_dir.rglob("*.npy"))
    write_receipt(
        latent_dir,
        autoencoder=config.autoencoder,
        codec_config=config.codec,
        codec=codec,
        table_sha256=table_sha256,
        details={"provenance": "imported, unverified", "latents": count},
    )
    return count


def main() -> int:
    """Prepare one experiment's latent corpus.

    Returns:
      exit_code: Zero after successful preparation.

    Raises:
      SystemExit: ``--source`` is missing when encoding.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = dataset_config(flags.experiment)
    if flags.directory is not None:
        config.working_dir = flags.directory
    if flags.receipt_only:
        print(record_receipt(config))
        return 0
    if flags.source is None:
        parser.error("--source is required unless --receipt-only is given.")
    print(
        prepare(
            config,
            flags.source,
            device=flags.device,
            batch_size=flags.batch_size,
            limit=flags.limit,
        ),
    )
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    experiment: str
    source: Path | None
    directory: Path | None
    device: str
    batch_size: int
    limit: int | None
    receipt_only: bool


class _Experiment(Protocol):
    def __call__(self) -> experiments.SpeedrunTrainLoop: ...


# Fitted before any latent is written, so every stored index refers to the one table
# saved beside it; an existing table is reused, never refitted, or latents already on
# disk would decode against the wrong levels.
def _fit_or_load(
    codec: FittedCodec,
    autoencoder: Autoencoder,
    listed: Sequence[Record],
    *,
    root: Path,
    latent_dir: Path,
    size: int,
    device: str,
    batch_size: int,
) -> tuple[str, dict[str, object]]:
    """Load the corpus's codec table, or fit and save one; return its digest."""
    if table_path(latent_dir).is_file():
        return load_table(latent_dir, codec), {"reused": True}
    chosen = fit_sample_indices(len(listed), codec.num_fit_images)
    subset = [listed[index] for index in chosen]
    logger.info("Fitting the codec on %d images.", len(subset))
    sample = torch.cat(
        [
            encode_latents(autoencoder, images, device)
            for _, images in batches(subset, root, size, batch_size)
        ],
    )
    codec.fit(sample)
    digest = hashlib.sha256(",".join(map(str, chosen)).encode()).hexdigest()
    return save_table(latent_dir, codec), {
        "num_images": len(chosen),
        "indices_sha256": digest,
    }


def batches(
    listed: Sequence[Record],
    root: Path,
    size: int,
    batch_size: int,
) -> Iterator[tuple[Sequence[Record], Tensor]]:
    """Yield records with their ``[B, 3, size, size]`` uint8 cropped images."""
    for start in range(0, len(listed), batch_size):
        batch = listed[start : start + batch_size]
        images = torch.stack([_image(root, record, size) for record in batch])
        yield batch, images


# The images directory is shared by every corpus beside it, so an image cropped for one
# autoencoder is reused by the next rather than cropped again; PNG is lossless, so the
# reused pixels are the cropped ones.
def _image(root: Path, record: Record, size: int) -> Tensor:
    """Return one cropped image as ``[3, size, size]`` uint8, writing it if new."""
    path = root / "images" / record.stem[:5] / f"img{record.stem}.png"
    if path.is_file():
        with Image.open(path) as stored:
            cropped = stored.convert("RGB")
    else:
        with Image.open(record.source) as opened:
            cropped = center_crop(opened.convert("RGB"), size)
        path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(path)
    if cropped.size != (size, size):
        raise ValueError(
            f"{path} is {cropped.size}; this autoencoder needs {size}x{size} images.",
        )
    return torch.from_numpy(np.asarray(cropped).copy()).permute(2, 0, 1)


def encode_latents(autoencoder: Autoencoder, images: Tensor, device: str) -> Tensor:
    """Encode a uint8 batch and return float32 latents on the CPU."""
    with torch.inference_mode():
        return autoencoder.encode(images.to(device)).float().cpu()


def _latent_path(latent_dir: Path, record: Record) -> Path:
    """Return where one record's stored latent goes."""
    return latent_dir / record.stem[:5] / f"img-latents-{record.stem}.npy"


def _write_labels(latent_dir: Path, listed: Sequence[Record]) -> None:
    """Write the REG label manifest for every record."""
    labels = [
        [_latent_path(latent_dir, r).relative_to(latent_dir).as_posix(), r.label]
        for r in listed
    ]
    path = latent_dir / LABELS
    staging = path.with_suffix(".json.partial")
    _ = staging.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    staging.replace(path)


class _ErrorTally:
    """Streams how far stored latents land from the encoder's own."""

    def __init__(self) -> None:
        self.squared = 0.0
        self.count = 0
        self.max_abs = 0.0
        self.saturated = 0

    def add(self, latents: Tensor, codec: LatentCodec, stored: Tensor) -> None:
        """Accumulate one batch's decoding error."""
        error = codec.decode(stored) - latents
        self.squared += float(error.double().pow(2).sum())
        self.count += error.numel()
        self.max_abs = max(self.max_abs, float(error.abs().max()))
        if isinstance(codec, ScalarTableCodec):
            self.saturated += int(codec.saturated(latents).sum())

    def summary(self) -> dict[str, float | None]:
        """Return the mean squared, worst, and saturated share of this run's latents."""
        if self.count == 0:
            return {"mse": None, "max_abs": None, "saturated_fraction": None}
        return {
            "mse": self.squared / self.count,
            "max_abs": self.max_abs,
            "saturated_fraction": self.saturated / self.count,
        }


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register corpus preparation flags."""
    parser.add_argument(
        "--experiment",
        default="exp000",
        help="Experiment in speedrundit.experiments whose corpus to prepare.",
    )
    parser.add_argument("--source", type=Path, help="Extracted ImageNet directory.")
    parser.add_argument(
        "--directory",
        type=Path,
        help="Corpus root on this machine, replacing the experiment's.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--receipt-only",
        action="store_true",
        help="Record an existing corpus's receipt without encoding.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
