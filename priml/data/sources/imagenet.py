"""ImageNet data source."""

from __future__ import annotations

from contextlib import closing
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, override

import logging
import tarfile

from configgle import Fig
from PIL import Image

from priml.data.sources.sharding import interleave_shards, shard_and_shuffle
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from priml.data.processors.custom_types import Sample


logger = logging.getLogger(__name__)


class ImageNetSource:
    """Load samples from ImageNet ILSVRC2012 format.

    Expects tar files:
    - train: ILSVRC2012_img_train.tar (nested tars, one per class)
    - val: ILSVRC2012_img_val.tar (flat tar with all images)
    - test: ILSVRC2012_img_test*.tar (flat tar, no labels)

    Training split yields samples with synset ID labels.
    Validation requires validation_labels.txt mapping.
    Test split has label = -1 (unlabeled).
    """

    class Config(Fig["ImageNetSource"]):
        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/datasets/imagenet"
        """Logical ImageNet root containing the source archives."""

        split: Literal["train", "val", "test"] = "train"
        """Which archive is read; ``test`` carries no labels."""

        validation_labels_file: Path | str | None = None
        """Validation-label override; ``None`` reads ``validation_labels.txt``
        inside ``working_dir``. The ``val`` split has no labels in its archive,
        so one of the two must resolve."""

        worker_slice: tuple[int, int] | None = None
        """``(worker_id, num_workers)`` partition, injected by the loader."""

        shuffle: bool = False
        """Shuffle shard order before slicing, so every worker agrees on it."""

        epoch_seed: int = 0
        """Seed folded into the shuffle; the loader sets it per epoch."""

        num_concurrently_read_shards: int = 4
        """Shards interleaved at once, which hides per-shard read latency."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if self.validation_labels_file is not None:
                self.validation_labels_file = resolve_working_dir(
                    self.base_dir,
                    self.validation_labels_file,
                )
            return super().finalize()

    def __init__(self, config: Config):
        if config.num_concurrently_read_shards < 1:
            raise ValueError(
                "num_concurrently_read_shards must be >= 1, got "
                f"{config.num_concurrently_read_shards}.",
            )
        self.split = config.split
        self.shuffle = config.shuffle
        self.worker_slice = config.worker_slice
        self.epoch_seed = config.epoch_seed
        self.num_concurrently_read_shards = config.num_concurrently_read_shards
        self.dataset_dir = Path(config.working_dir)
        self._validation_members: list[tarfile.TarInfo] = []

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")

        # Determine tar file path.
        if self.split == "train":
            self.tar_path = self.dataset_dir / "ILSVRC2012_img_train.tar"
        elif self.split == "val":
            self.tar_path = self.dataset_dir / "ILSVRC2012_img_val.tar"
        elif self.split == "test":
            # Test split can have different names.
            test_files = list(self.dataset_dir.glob("ILSVRC2012_img_test*.tar"))
            if not test_files:
                raise ValueError(f"No test tar found in {self.dataset_dir}")
            if len(test_files) > 1:
                raise ValueError(f"Multiple test archives found in {self.dataset_dir}")
            self.tar_path = test_files[0]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if not self.tar_path.exists():
            raise ValueError(f"Tar file not found: {self.tar_path}")

        # After the archive check: _load_validation_labels opens the tar, so a
        # missing archive must report itself rather than as a labels error.
        if self.split == "val":
            labels_file = (
                Path(config.validation_labels_file)
                if config.validation_labels_file is not None
                else self.dataset_dir / "validation_labels.txt"
            )
            if not labels_file.exists():
                raise ValueError(
                    f"Validation labels not found at {labels_file}. Set "
                    "validation_labels_file, or place validation_labels.txt "
                    "beside the archive.",
                )
            self.validation_labels = self._load_validation_labels(labels_file)

        logger.info(
            "ImageNetSource initialized: split=%s, tar=%s",
            self.split,
            self.tar_path.name,
        )

    def _load_validation_labels(self, labels_file: Path) -> dict[str, str]:
        """Load validation labels file."""
        # Get sorted list of filenames from tar.
        with tarfile.open(self.tar_path) as tar:
            self._validation_members = [m for m in tar.getmembers() if m.isfile()]
        filenames = sorted(member.name for member in self._validation_members)

        # Load labels (one per line)
        with labels_file.open() as f:
            labels = [line.strip() for line in f]

        return dict(zip(filenames, labels, strict=True))

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples in the split."""
        if self.split == "train":
            yield from self._iter_train()
        elif self.split == "val":
            yield from self._iter_val()
        else:  # Test.
            yield from self._iter_test()

    # ``tarfile.getmembers`` scans the whole archive to build the member list, so the
    # first access pays a full sequential pass over the tar.
    def _iter_train(self) -> Iterator[Sample]:
        """Iterate training split (nested tars) with parallel interleaved reading."""
        with tarfile.open(self.tar_path) as tar:
            # Get class tar members. Fold the loader-injected epoch seed into the
            # shuffle so each epoch reshuffles while all workers share the
            # permutation.
            class_tars = shard_and_shuffle(
                [m for m in tar.getmembers() if m.name.endswith(".tar")],
                worker_slice=self.worker_slice,
                shuffle=self.shuffle,
                epoch_seed=self.epoch_seed,
            )

            yield from interleave_shards(
                class_tars,
                partial(_read_class_tar, tar),
                num_concurrently_read_shards=self.num_concurrently_read_shards,
            )

    def _iter_val(self) -> Iterator[Sample]:
        """Iterate validation split (flat tar with labels file)."""
        with tarfile.open(self.tar_path) as tar:
            # Val is never shuffled; slice deterministically over members so the
            # same sample always maps to the same worker for stable evaluation.
            members = shard_and_shuffle(
                self._validation_members,
                worker_slice=self.worker_slice,
            )

            for member in members:
                image_file = tar.extractfile(member)
                if image_file is None:
                    continue

                try:
                    image = Image.open(image_file)
                    image.load()  # pyright: ignore[reportUnknownMemberType] -- The dataset backend is third-party and untyped at this boundary.

                    label = self.validation_labels[member.name]

                    sample: Sample = {
                        "file_name": member.name,
                        "image": image,
                        "label": label,
                    }
                    yield sample
                except (OSError, Image.DecompressionBombError) as e:
                    logger.warning("Failed to load %s: %s", member.name, e)

    def _iter_test(self) -> Iterator[Sample]:
        """Iterate test split (flat tar, no labels)."""
        with tarfile.open(self.tar_path) as tar:
            # Test is never shuffled, mirroring val: deterministic slice only.
            members = shard_and_shuffle(
                [m for m in tar.getmembers() if m.isfile()],
                worker_slice=self.worker_slice,
            )

            for member in members:
                image_file = tar.extractfile(member)
                if image_file is None:
                    continue

                try:
                    image = Image.open(image_file)
                    image.load()  # pyright: ignore[reportUnknownMemberType] -- The dataset backend is third-party and untyped at this boundary.

                    sample: Sample = {
                        "file_name": member.name,
                        "image": image,
                        "label": -1,  # Test split has no labels.
                    }
                    yield sample
                except (OSError, Image.DecompressionBombError) as e:
                    logger.warning("Failed to load %s: %s", member.name, e)


def _read_class_tar(
    tar: tarfile.TarFile,
    class_tar_member: tarfile.TarInfo,
) -> Generator[Sample, None, None]:
    """Read a single class tar from ImageNet training split."""
    # Extract synset label from filename (e.g., "n01632458.tar" -> "n01632458")
    synset_label = class_tar_member.name[:-4]

    # Extract nested tar.
    class_tar_file = tar.extractfile(class_tar_member)
    if class_tar_file is None:
        logger.warning("Could not extract %s", class_tar_member.name)
        return

    # Iterate images in this class. Close the extracted fileobj when done;
    # tarfile.open(fileobj=...) does not take ownership of it.
    with (
        closing(class_tar_file),
        tarfile.open(fileobj=class_tar_file) as class_tar,
    ):
        for image_member in class_tar.getmembers():
            if not image_member.isfile():
                continue

            image_file = class_tar.extractfile(image_member)
            if image_file is None:
                continue

            try:
                image = Image.open(image_file)
                image.load()  # pyright: ignore[reportUnknownMemberType] -- The dataset backend is third-party and untyped at this boundary.

                sample: Sample = {
                    "file_name": image_member.name,
                    "image": image,
                    "label": synset_label,
                }
                yield sample
            except (OSError, Image.DecompressionBombError) as e:
                logger.warning("Failed to load %s: %s", image_member.name, e)
