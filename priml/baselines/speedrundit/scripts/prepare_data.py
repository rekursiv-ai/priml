#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Stage the SR-DiT latent corpus, or synthesize a small one for smoke runs.

The reference builds its corpus in two passes, and this script owns the
first: ``--convert`` reads extracted ImageNet through priml's ImageNet source
and writes the reference's ``images/`` tree -- ADM-cropped 256px PNGs named by
their sorted-order index, with ``images/dataset.json`` -- bit for bit what the
reference's ``dataset_tools.py convert`` writes. The second pass, INVAE
encoding into ``vae-in/``, needs the tokenizer's checkpoint and a GPU and is
the reference's own ``dataset_tools.py encode``, run on that tree; the
DINOv2 targets likewise come from outside. ``--verify`` then checks the
finished tree, and the synthetic path writes the whole layout with random
tensors, which is what makes the smoke experiment and the loader tests run
with no ImageNet, no tokenizer, and no network.

Publishing is atomic: the tree is built in a hidden sibling directory on the
same filesystem and renamed into place, so an interrupted run leaves either
the previous corpus or nothing, never half of one.

Examples:
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --help  # noqa: E501
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --synthetic --samples 64  # noqa: E501
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --convert --imagenet /datasets/imagenet  # noqa: E501
  uv --quiet run --frozen python -m priml.baselines.speedrundit.scripts.prepare_data --verify --source /data/imagenet256  # noqa: E501

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import json
import shutil
import tempfile

from torch import Tensor

import numpy as np

from priml.baselines.imagenet.data import NUM_CLASSES
from priml.baselines.speedrundit.data import (
    SpeedrunDiTData,
    imagenet_image_pipeline,
    read_labels,
    relative_names,
)
from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.lib.custom_json import IntCodec
from priml.train.train_loop import TrainLoop

import priml.baselines.imagenet.scripts.prepare_data


if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")  # ~60 ms; only --convert writes PNGs.


SHARD_WIDTH: Final = 5
"""Digits in the shard subdirectory the reference's preprocessing writes."""

INDEX_WIDTH: Final = 8
"""Digits in a sample's zero-padded index."""


def default_directory() -> Path:
    """Resolve the corpus path a default run would read.

    Derived by finalizing the dataset config beneath the loop's own
    ``base_dir`` rather than by naming a path, so the preparer and the trainer
    cannot drift apart.

    Returns:
      directory: Where the corpus belongs.

    """
    config = SpeedrunDiTData.Config()
    config.base_dir = TrainLoop.Config().base_dir
    return Path(config.copy_tree().finalize().working_dir)


def verify(directory: Path) -> int:
    """Check a prepared corpus against the layout the loader expects.

    Args:
      directory: Corpus root.

    Returns:
      count: Samples the corpus holds.

    Raises:
      FileNotFoundError: If a required tree or the manifest is missing.
      ValueError: If the two trees disagree or a label is unaccounted for.

    """
    latents_dir = directory / "vae-in"
    manifest = latents_dir / "dataset.json"
    if not latents_dir.is_dir():
        raise FileNotFoundError(f"No vae-in/ tree under {directory}.")
    if not manifest.is_file():
        raise FileNotFoundError(f"No label manifest at {manifest}.")
    latent_names = relative_names(latents_dir)
    images_dir = directory / "images"
    if images_dir.is_dir():
        image_names = relative_names(images_dir)
        if len(image_names) != len(latent_names):
            raise ValueError(
                f"{len(image_names)} images against {len(latent_names)} "
                "latents; the trees are paired by position.",
            )
    labels = read_labels(manifest)
    missing = [name for name in latent_names if name not in labels]
    if missing:
        raise ValueError(
            f"{len(missing)} latents have no label; first is {missing[0]}.",
        )
    return len(latent_names)


def convert(imagenet: Path, directory: Path, *, workers: int = 0) -> int:
    """Write the reference's ``images/`` tree from extracted ImageNet.

    Every image of ``train/`` becomes ``images/{index[:5]}/img{index}.png``,
    ``index`` being its position in the source's sorted walk -- the numbering
    the reference's own tool gives it, gaps included where an image fails to
    decode -- with its canonical class index in ``images/dataset.json``.

    Args:
      imagenet: Extracted ImageNet root holding ``train/<synset>/*.JPEG``.
      directory: Corpus root; ``images/`` is created beneath it.
      workers: Decoding processes; zero decodes in this one.

    Returns:
      count: Images written.

    Raises:
      FileExistsError: If ``images/`` already holds files.
      ValueError: If a class directory is not an ImageNet synset.

    """
    images = directory / "images"
    if images.exists() and any(images.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty {images}.")
    pipeline = imagenet_image_pipeline()
    source = pipeline.source
    assert isinstance(source, ExtractedImageNetSource.Config)
    source.working_dir = imagenet
    # Numbered by the source's own walk, taken once up front: workers finish
    # out of order, and a sample the pipeline drops must leave its gap.
    walk = source.copy_tree().make()
    known = ImagenetSynsetToIndex.Config().make().synset_to_idx
    unknown = sorted({str(sample.get("label")) for sample in walk} - set(known))
    if unknown:
        raise ValueError(
            f"{len(unknown)} class directories are not synsets: {unknown[:3]}",
        )
    order = {str(sample.get("file_path")): index for index, sample in enumerate(walk)}
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=directory, prefix=".images-"))
    labels: dict[int, list[object]] = {}
    try:
        for sample in pipeline.make().create_loader(num_workers=workers):
            index = order[str(sample["file_path"])]
            label = IntCodec.coerce(sample["label"], default=None)
            media = sample["media_tensor"]
            assert isinstance(media, Tensor)
            stem = f"{index:0{INDEX_WIDTH}d}"
            name = f"{stem[:SHARD_WIDTH]}/img{stem}.png"
            (staging / name).parent.mkdir(exist_ok=True)
            # Uncompressed, as the reference stores it; PNG is lossless, so
            # this sets the file size and not a single pixel.
            Image.fromarray(media[:, 0].permute(1, 2, 0).numpy()).save(
                staging / name,
                format="png",
                compress_level=0,
                optimize=False,
            )
            labels[index] = [name, label]
        (staging / "dataset.json").write_text(
            json.dumps({"labels": [labels[index] for index in sorted(labels)]}),
            encoding="utf-8",
        )
        if images.exists():
            images.rmdir()
        staging.replace(images)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return len(labels)


def synthesize(
    directory: Path,
    *,
    samples: int,
    image_size: int,
    latent_size: int,
    latent_channels: int,
    num_classes: int,
    encoder_width: int,
    seed: int,
) -> None:
    """Write a synthetic corpus in the reference's layout.

    Args:
      directory: Destination, created fresh.
      samples: Samples to write.
      image_size: Side length of each stored image.
      latent_size: Side length of each stored latent.
      latent_channels: Channels per latent.
      num_classes: Classes cycled through.
      encoder_width: Width of the synthetic alignment features.
      seed: Seed for every draw.

    Raises:
      FileExistsError: If the destination already holds a corpus.

    """
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty {directory}.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}-"),
    )
    try:
        _write_synthetic(
            staging,
            samples=samples,
            image_size=image_size,
            latent_size=latent_size,
            latent_channels=latent_channels,
            num_classes=num_classes,
            encoder_width=encoder_width,
            seed=seed,
        )
        if directory.exists():
            directory.rmdir()
        staging.replace(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    """Prepare or verify the corpus.

    Returns:
      status: Zero on success.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    directory = flags.directory or default_directory()

    if flags.synthetic:
        synthesize(
            directory,
            samples=flags.samples,
            image_size=flags.image_size,
            latent_size=flags.latent_size,
            latent_channels=flags.latent_channels,
            num_classes=flags.num_classes,
            encoder_width=flags.encoder_width,
            seed=flags.seed,
        )
        print(f"wrote {flags.samples} synthetic samples to {directory}")
        return 0

    if flags.convert:
        imagenet = (
            flags.imagenet
            or priml.baselines.imagenet.scripts.prepare_data.default_directory()
        )
        count = convert(imagenet, directory, workers=flags.workers)
        print(f"wrote {count} images from {imagenet} to {directory / 'images'}")
        return 0

    if flags.source is not None:
        count = verify(flags.source)
        print(f"{flags.source} holds {count} verified samples")
        if flags.source.resolve() != directory.resolve():
            shutil.copytree(flags.source, directory, dirs_exist_ok=False)
            print(f"published to {directory}")
        return 0

    count = verify(directory)
    print(f"{directory} holds {count} verified samples")
    return 0


class _Flags(Protocol):
    """Parsed command line."""

    directory: Path | None
    source: Path | None
    synthetic: bool
    verify: bool
    convert: bool
    imagenet: Path | None
    workers: int
    samples: int
    image_size: int
    latent_size: int
    latent_channels: int
    num_classes: int
    encoder_width: int
    seed: int


def _write_synthetic(
    root: Path,
    *,
    samples: int,
    image_size: int,
    latent_size: int,
    latent_channels: int,
    num_classes: int,
    encoder_width: int,
    seed: int,
) -> None:
    """Fill a staging directory with a synthetic corpus."""
    rng = np.random.default_rng(seed)
    images_dir = root / "images"
    latents_dir = root / "vae-in"
    labels: list[Sequence[object]] = []
    tokens = 1 + latent_size * latent_size
    for index in range(samples):
        shard = f"{index:0{INDEX_WIDTH}d}"[:SHARD_WIDTH]
        (images_dir / shard).mkdir(parents=True, exist_ok=True)
        (latents_dir / shard).mkdir(parents=True, exist_ok=True)
        stem = f"{index:0{INDEX_WIDTH}d}"
        np.save(
            images_dir / shard / f"img{stem}.npy",
            rng.integers(0, 256, (3, image_size, image_size), dtype=np.uint8),
        )
        latent = rng.standard_normal(
            (1, latent_channels, latent_size, latent_size),
        ).astype(np.float32)
        np.save(latents_dir / shard / f"img-latents-{stem}.npy", latent)
        labels.append([f"{shard}/img-latents-{stem}.npy", index % num_classes])
    (latents_dir / "dataset.json").write_text(
        json.dumps({"labels": labels}),
        encoding="utf-8",
    )
    # The alignment targets a real run precomputes with DINOv2. Stored beside
    # the corpus rather than inside vae-in/, which the loader pairs by
    # position and would otherwise mistake for latents.
    np.save(
        root / "cls_token.npy",
        rng.standard_normal((samples, encoder_width)).astype(np.float32),
    )
    np.save(
        root / "features.npy",
        rng.standard_normal((samples, tokens, encoder_width)).astype(np.float32),
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the command line."""
    parser.add_argument("--directory", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--imagenet", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--encoder-width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
