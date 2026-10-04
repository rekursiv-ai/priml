#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Crop extracted ImageNet and encode the latents an experiment trains on.

The experiment names the autoencoder, the storage codec, and the corpus
directory; this script builds exactly those and writes the corpus with its
receipt. Adding an autoencoder never touches this file: a new experiment
declares it. A fitted codec (uint8 tables) is fitted first on a sample of the
images and its table is written before any latent. Re-running resumes, keeping
the images and latents already on disk, when the source, producers, seed, and
(for a seeded corpus) batch size are unchanged; anything else is refused.

Examples:
  prepare_data.py --experiment exp000 --source /datasets/imagenet
  prepare_data.py --experiment exp003 --source /datasets/imagenet --device cuda
  prepare_data.py --experiment exp000 --receipt-only --source /datasets/imagenet

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
    PREPARING,
    RECEIPT,
    CorpusMismatchError,
    load_table,
    mismatches,
    save_stored,
    save_table,
    table_path,
    verify_receipt,
    write_atomically,
    write_receipt,
)
from priml.baselines.speedrundit.latent_codec import FittedCodec, ScalarTableCodec
from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.lib.custom_json import DictCodec, loads
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

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
``indices_sha256`` records; a table already on disk is reused, never refitted,
and its receipt keeps the fit's record across reruns.
"""

IMAGE_SOURCE: Final = "source.json"
"""The shared crops' binding to their image source, inside ``images/``."""


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


@dataclass(frozen=True, slots=True, kw_only=True)
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


def records(imagenet: Path) -> list[Record]:
    """List training images in the extracted source's deterministic order.

    Args:
      imagenet: Extracted ImageNet directory.

    Returns:
      records: One per image, numbered in source order.

    Raises:
      TypeError: A record lacks a typed class label or path.

    """
    source = ExtractedImageNetSource.Config(working_dir=imagenet, split="train").make()
    labels = ImagenetSynsetToIndex.Config().make()
    found: list[Record] = []
    for index, record in enumerate(labels(iter(source))):
        label = record.get("label")
        path = record.get("file_path")
        if not isinstance(label, int):
            raise TypeError(f"expected an integer ImageNet label: {record}")
        if not isinstance(path, str):
            raise TypeError(f"expected an ImageNet file path: {record}")
        found.append(Record(source=path, label=label, stem=f"{index:08d}"))
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

    A fitted codec is fitted on a sample of EVERY source image, not only the
    first ``limit``: later runs reuse its table for the whole corpus.

    Args:
      config: Finalized latent source, as :func:`dataset_config` returns it.
      imagenet: Extracted ImageNet directory.
      device: Device the autoencoder encodes on.
      batch_size: Images encoded per forward.
      limit: Encode only the first ``limit`` images.

    Returns:
      count: Source records selected for this invocation. Previously encoded
        records outside ``limit`` remain in the corpus.

    Raises:
      ValueError: ``batch_size`` or ``limit`` is not positive, the source holds
        no images, or the autoencoder returns latents the corpus cannot hold.
      CorpusMismatchError: The shared crops, the receipt, or the preparation
        settings belong to another producer, or latents exist without a receipt.

    """
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError(
            f"batch_size and limit must be positive; got {batch_size} and {limit}.",
        )
    all_records = records(imagenet)
    if not all_records:
        raise ValueError(f"{imagenet} holds no training images.")
    listed = all_records[:limit]
    root = Path(config.working_dir)
    latent_dir = root / config.latent_subdir
    latent_dir.mkdir(parents=True, exist_ok=True)
    size = config.autoencoder.image_size
    latent_shape = config.autoencoder.latent_shape()
    ensure_image_source(
        root,
        identity=image_source_identity(imagenet, listed=all_records, size=size),
    )
    # Batch size fixes which posterior draw a seeded record gets; unseeded draws
    # cannot be reproduced whatever the batching, so it binds only a seeded corpus.
    preparation = {
        "seed": config.seed,
        "batch_size": None if config.seed is None else batch_size,
    }
    codec = config.codec.make()
    table_sha256, previous = _resume(latent_dir, config=config, codec=codec)
    recorded = previous.get("preparation")
    if recorded is not None:
        problems = mismatches(
            DictCodec.coerce(recorded, default=None),
            expected=preparation,
        )
        if problems:
            raise CorpusMismatchError(
                f"{latent_dir} was prepared with other settings:\n  "
                + "\n  ".join(problems),
            )
    # Seeded even when encoding is not: a checkpoint-free autoencoder's weights are
    # random, and only a fixed construction lets a resumed run encode with the same.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0 if config.seed is None else config.seed)
        autoencoder = config.autoencoder.make()
    if isinstance(autoencoder, torch.nn.Module):
        _ = autoencoder.to(device)
    if config.seed is not None:
        torch.manual_seed(config.seed)
    fit = previous.get("fit")
    # Fitted before any latent is written, so every stored index refers to the one
    # table saved beside it; an existing table is reused, never refitted, or latents
    # already on disk would decode against the wrong levels.
    if isinstance(codec, FittedCodec) and table_sha256 is None:
        table_sha256, fit = _fit(
            codec,
            autoencoder=autoencoder,
            listed=all_records,
            root=root,
            latent_dir=latent_dir,
            size=size,
            device=device,
            batch_size=batch_size,
            latent_shape=latent_shape,
        )
    details = {"preparation": preparation, "fit": fit}
    # The loader refuses a PREPARING corpus, so an interrupted run's latents are
    # tied to their producers yet never load as a finished corpus.
    _ = write_receipt(
        latent_dir,
        autoencoder=config.autoencoder,
        codec_config=config.codec,
        codec=codec,
        table_sha256=table_sha256,
        details={**details, "provenance": PREPARING},
    )
    pending = {
        r.stem for r in listed if not _latent_path(latent_dir, record=r).is_file()
    }
    # Windows are fixed over the WHOLE source and encoded whole, so a seeded record
    # gets the same draw however ``limit`` or an interruption split the runs. Every
    # chosen window is full except possibly the source's last, so ``batches``
    # re-chunks them unchanged.
    windows = [
        all_records[start : start + batch_size]
        for start in range(0, len(all_records), batch_size)
    ]
    encoding = [
        record
        for window in windows
        if any(r.stem in pending for r in window)
        for record in window
    ]
    error = _ErrorTally()
    for batch, images in batches(encoding, root=root, size=size, batch_size=batch_size):
        if config.seed is not None:
            torch.manual_seed(config.seed + int(batch[0].stem))
        latents = encode_latents(
            autoencoder,
            images=images,
            device=device,
            latent_shape=latent_shape,
        )
        keep = [index for index, r in enumerate(batch) if r.stem in pending]
        stored = codec.encode(latents[keep])
        error.add(latents[keep], codec=codec, stored=stored)
        for row, index in enumerate(keep):
            path = _latent_path(latent_dir, record=batch[index])
            path.parent.mkdir(parents=True, exist_ok=True)
            save_stored(path, stored=stored[row : row + 1])
    _write_labels(latent_dir, listed=all_records)
    _ = write_receipt(
        latent_dir,
        autoencoder=config.autoencoder,
        codec_config=config.codec,
        codec=codec,
        table_sha256=table_sha256,
        details={
            **details,
            "provenance": "encoded",
            "images": {
                "source": str(imagenet.resolve()),
                "split": "train",
                "selected": len(listed),
                "resolution": size,
                "crop": "adm_center_crop",
            },
            "encoding": {
                "device": device,
                "torch": torch.__version__,
                "newly_encoded": len(pending),
            },
            "error": error.summary() if pending else previous.get("error"),
        },
    )
    return len(listed)


def image_source_identity(
    imagenet: Path,
    listed: Sequence[Record],
    *,
    size: int,
) -> dict[str, object]:
    """Return what makes two crop directories interchangeable.

    Paths are taken relative to ``imagenet``, so the same images mounted
    elsewhere keep one identity.

    Args:
      imagenet: Extracted source directory.
      listed: Every source record, independent of invocation limits.
      size: ADM crop side length.

    Returns:
      identity: Digest of the source catalog, and the crop settings.

    """
    catalog = "\n".join(
        f"{Path(r.source).relative_to(imagenet).as_posix()}:{r.label}" for r in listed
    )
    return {
        "records_sha256": hashlib.sha256(catalog.encode()).hexdigest(),
        "resolution": size,
        "crop": "adm_center_crop",
    }


def ensure_image_source(
    root: Path,
    *,
    identity: Mapping[str, object],
    adopt: bool = False,
) -> None:
    """Bind the shared crops to one image source, or refuse crops of another.

    Args:
      root: Corpus root containing the shared ``images`` directory.
      identity: :func:`image_source_identity` of the source being read.
      adopt: Bind existing crops that record no source to this one. Only the
        operator can vouch for such crops; ``--receipt-only --source`` does.

    Raises:
      CorpusMismatchError: The crops are bound to another source, or exist
        unbound and ``adopt`` is false.

    """
    path = root / "images" / IMAGE_SOURCE
    if path.is_file():
        bound = DictCodec.coerce(loads(path.read_text()), default=None)
        problems = mismatches(bound, expected=identity)
        if problems:
            raise CorpusMismatchError(
                f"{path.parent} holds crops of another image source:\n  "
                + "\n  ".join(problems),
            )
        return
    if not adopt and any(path.parent.rglob("*.png")):
        raise CorpusMismatchError(
            f"{path.parent} holds crops that record no source. If they were cropped "
            "from this source, record it with prepare_data.py --receipt-only "
            "--source; otherwise delete them.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(identity), sort_keys=True)
    write_atomically(path, write=lambda stream: stream.write(text.encode()))


def record_receipt(
    config: PairedImageLatentDataset.Config,
    imagenet: Path | None = None,
) -> int:
    """Record what a corpus encoded elsewhere, e.g. an existing REG one, cannot state.

    Nothing is encoded or verified against pixels: the receipt only states that
    this corpus is taken to be the configured autoencoder's, which the loader
    then holds it to. An existing receipt is verified, never replaced.

    Args:
      config: Finalized latent source.
      imagenet: The extracted source the shared crops were cut from; binds
        crops that record no source. ``None`` leaves the crops as they are.

    Returns:
      count: Latent files found.

    Raises:
      FileNotFoundError: The latent directory, or a fitted codec's table, is missing.
      CorpusMismatchError: An existing receipt or crop binding names another
        producer.

    """
    root = Path(config.working_dir)
    latent_dir = root / config.latent_subdir
    if not latent_dir.is_dir():
        raise FileNotFoundError(f"{latent_dir} does not exist.")
    if imagenet is not None:
        identity = image_source_identity(
            imagenet,
            listed=records(imagenet),
            size=config.autoencoder.image_size,
        )
        ensure_image_source(root, identity=identity, adopt=True)
    codec = config.codec.make()
    table_sha256 = (
        load_table(latent_dir, codec=codec) if isinstance(codec, FittedCodec) else None
    )
    count = sum(1 for _ in latent_dir.rglob("*.npy"))
    if (latent_dir / RECEIPT).is_file():
        _ = verify_receipt(
            latent_dir,
            autoencoder=config.autoencoder,
            codec_config=config.codec,
            codec=codec,
            table_sha256=table_sha256,
        )
        return count
    _ = write_receipt(
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
        config.working_dir = validated_output_path(flags.directory)
    if flags.receipt_only:
        print(record_receipt(config, imagenet=flags.source))
        return 0
    if flags.source is None:
        parser.error("--source is required unless --receipt-only is given.")
    print(
        prepare(
            config,
            imagenet=flags.source,
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


def batches(
    listed: Sequence[Record],
    root: Path,
    size: int,
    batch_size: int,
) -> Iterator[tuple[Sequence[Record], Tensor]]:
    """Yield records with their cropped images, ``batch_size`` at a time.

    Args:
      listed: Records to crop, in order.
      root: Corpus root holding the shared ``images/`` directory.
      size: Square crop side.
      batch_size: Records per yielded batch.

    Yields:
      batch: Up to ``batch_size`` consecutive records.
      images: ``[B, 3, size, size]`` uint8 crops of ``batch``.

    """
    for start in range(0, len(listed), batch_size):
        batch = listed[start : start + batch_size]
        images = torch.stack(
            [_image(root, record=record, size=size) for record in batch],
        )
        yield batch, images


def encode_latents(
    autoencoder: Autoencoder,
    images: Tensor,
    *,
    device: str,
    latent_shape: tuple[int, int, int],
) -> Tensor:
    """Encode a uint8 batch to float32 CPU latents, refusing any a corpus cannot hold.

    Args:
      autoencoder: The experiment's autoencoder.
      images: ``[B, 3, H, W]`` uint8.
      device: Device the autoencoder encodes on.
      latent_shape: The ``(C, H, W)`` its config declares.

    Returns:
      latents: ``[B, *latent_shape]`` float32 on the CPU.

    Raises:
      ValueError: The latents are not one ``latent_shape`` per image, or not finite.

    """
    with torch.inference_mode():
        latents = autoencoder.encode(images.to(device)).float().cpu()
    if latents.shape != (images.shape[0], *latent_shape):
        raise ValueError(
            f"The autoencoder returned {tuple(latents.shape)} latents for "
            f"{images.shape[0]} images; its config declares {latent_shape}.",
        )
    if not bool(
        torch.isfinite(latents).all(),
    ):  # house-ignore[tensor-value-guard] -- Host tensor checked once before publication; no device sync.
        raise ValueError("Autoencoder produced non-finite latents.")
    return latents


def _latent_path(latent_dir: Path, record: Record) -> Path:
    """Return where one record's stored latent goes."""
    return latent_dir / record.stem[:5] / f"img-latents-{record.stem}.npy"


def _write_labels(latent_dir: Path, listed: Sequence[Record]) -> None:
    """Write the REG label manifest for every record."""
    labels = [
        [_latent_path(latent_dir, record=r).relative_to(latent_dir).as_posix(), r.label]
        for r in listed
    ]
    text = json.dumps({"labels": labels})
    write_atomically(
        latent_dir / LABELS,
        write=lambda stream: stream.write(text.encode()),
    )


class _ErrorTally:
    """Streams how far stored latents land from the encoder's own."""

    def __init__(self) -> None:
        self.squared = 0.0
        self.count = 0
        self.max_abs = 0.0
        self.saturated = 0

    def add(self, latents: Tensor, codec: LatentCodec, stored: Tensor) -> None:
        """Accumulate one batch's decoding error.

        Args:
          latents: The encoder's float32 latents for the batch.
          codec: The codec that stored them.
          stored: ``codec.encode(latents)``.

        """
        error = codec.decode(stored) - latents
        self.squared += float(error.double().pow(2).sum())
        self.count += error.numel()
        self.max_abs = max(self.max_abs, float(error.abs().max()))
        if isinstance(codec, ScalarTableCodec):
            self.saturated += int(codec.saturated(latents).sum())

    def summary(self) -> dict[str, float | None]:
        """Return the mean squared, worst, and saturated share of this run's latents.

        Returns:
          summary: ``mse``, ``max_abs``, and ``saturated_fraction``; each is
            ``None`` when this run stored no latents.

        """
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
    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "Extracted ImageNet directory. With --receipt-only, binds existing "
            "crops that record no source to it."
        ),
    )
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


class _Experiment(Protocol):
    def __call__(self) -> experiments.SpeedrunTrainLoop: ...


def _resume(
    latent_dir: Path,
    config: PairedImageLatentDataset.Config,
    codec: LatentCodec,
) -> tuple[str | None, dict[str, object]]:
    """Verify what an earlier run left; return its table digest and provenance."""
    if not (latent_dir / RECEIPT).is_file():
        if any(latent_dir.rglob("*.npy")):
            raise CorpusMismatchError(
                f"{latent_dir} holds latents with no receipt, so their producer is "
                "unknown. Record them with --receipt-only, or delete them.",
            )
        # Only a run interrupted between saving its table and pinning it leaves a table
        # without a receipt. No latent was coded against it, so it is refitted.
        table_path(latent_dir).unlink(missing_ok=True)
        return None, {}
    table_sha256 = (
        load_table(latent_dir, codec=codec) if isinstance(codec, FittedCodec) else None
    )
    details = verify_receipt(
        latent_dir,
        autoencoder=config.autoencoder,
        codec_config=config.codec,
        codec=codec,
        table_sha256=table_sha256,
    )
    return table_sha256, details


def _fit(
    codec: FittedCodec,
    autoencoder: Autoencoder,
    listed: Sequence[Record],
    *,
    root: Path,
    latent_dir: Path,
    size: int,
    device: str,
    batch_size: int,
    latent_shape: tuple[int, int, int],
) -> tuple[str, dict[str, object]]:
    """Fit and save the codec's table; return its digest and the fit's record."""
    chosen = fit_sample_indices(len(listed), num_images=codec.num_fit_images)
    subset = [listed[index] for index in chosen]
    logger.info("Fitting the codec on %d images.", len(subset))
    sample = torch.cat(
        [
            encode_latents(
                autoencoder,
                images=images,
                device=device,
                latent_shape=latent_shape,
            )
            for _, images in batches(
                subset,
                root=root,
                size=size,
                batch_size=batch_size,
            )
        ],
    )
    codec.fit(sample)
    digest = hashlib.sha256(",".join(map(str, chosen)).encode()).hexdigest()
    return save_table(latent_dir, codec=codec), {
        "num_images": len(chosen),
        "indices_sha256": digest,
    }


def _image(root: Path, record: Record, size: int) -> Tensor:
    """Return one cropped image as ``[3, size, size]`` uint8, writing it if new."""
    path = root / "images" / record.stem[:5] / f"img{record.stem}.png"
    # The images directory is shared by every corpus beside it, so an image cropped
    # for one autoencoder is reused by the next rather than cropped again; PNG is
    # lossless, so the reused pixels are the cropped ones.
    if path.is_file():
        with Image.open(path) as stored:
            cropped = stored.convert("RGB")
    else:
        with Image.open(record.source) as opened:
            cropped = center_crop(opened.convert("RGB"), size=size)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(path, write=lambda stream: cropped.save(stream, format="PNG"))
    if cropped.size != (size, size):
        raise ValueError(
            f"{path} is {cropped.size}; this autoencoder needs {size}x{size} images.",
        )
    return torch.from_numpy(np.asarray(cropped).copy()).permute(2, 0, 1)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
