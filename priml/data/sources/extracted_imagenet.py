"""ImageNet source for extracted (non-tar) format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, override

import logging

from configgle import Fig

from priml.data.sources.sharding import shard_and_shuffle
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Iterator

    from priml.data.processors.custom_types import Sample


logger = logging.getLogger(__name__)


class ExtractedImageNetSource:
    """Load samples from extracted ImageNet directory structure.

    Expects:
    - train: {working_dir}/train/{synset}/{synset}_{id}.JPEG
    - val: {working_dir}/val/ILSVRC2012_val_{id}.JPEG + labels file

    Training split yields samples with synset ID labels.
    Validation requires validation_labels.txt mapping.
    """

    class Config(Fig["ExtractedImageNetSource"]):
        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/datasets/imagenet"
        """Logical root containing extracted train and validation directories."""

        split: Literal["train", "val"] = "train"
        """Which directory tree is walked."""

        validation_labels_file: Path | None = None
        """Optional validation-label override; defaults below ``working_dir``."""

        worker_slice: tuple[int, int] | None = None
        """``(worker_id, num_workers)`` partition, injected by the loader."""

        shuffle: bool = False
        """Shuffle class order before slicing, so every worker agrees on it."""

        epoch_seed: int = 0
        """Seed folded into the shuffle; the loader sets it per epoch."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config):
        self.split = config.split
        self.shuffle = config.shuffle
        self.worker_slice = config.worker_slice
        self.epoch_seed = config.epoch_seed
        self.dataset_dir = Path(config.working_dir)

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")

        if self.split == "train":
            self.split_dir = self.dataset_dir / "train"
        elif self.split == "val":
            self.split_dir = self.dataset_dir / "val"
            if config.validation_labels_file:
                labels_file = config.validation_labels_file
                if not labels_file.is_absolute():
                    labels_file = self.dataset_dir / labels_file
                self.validation_labels = self._load_validation_labels(labels_file)
            else:
                self.validation_labels = None
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if not self.split_dir.exists():
            raise ValueError(f"Split directory does not exist: {self.split_dir}")

        logger.info(
            "ExtractedImageNetSource initialized: split=%s, dir=%s",
            self.split,
            self.split_dir,
        )

    # Expects one label per line, corresponding to sorted val filenames.
    def _load_validation_labels(self, labels_file: Path) -> dict[str, str]:
        """Load validation labels file."""
        # Get sorted list of filenames.
        filenames = sorted(
            [f.name for f in self.split_dir.glob("*.JPEG") if f.is_file()],
        )

        # Load labels (one per line)
        with labels_file.open() as f:
            labels = [line.strip() for line in f]

        if len(filenames) != len(labels):
            raise ValueError(
                f"Mismatch: {len(filenames)} files but {len(labels)} labels",
            )

        return dict(zip(filenames, labels, strict=True))

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples in the split."""
        if self.split == "train":
            yield from self._iter_train()
        else:  # Val.
            yield from self._iter_val()

    def _iter_train(self) -> Iterator[Sample]:
        """Iterate training split (class directories)."""
        # Fold the loader-injected epoch seed into the shuffle so each epoch
        # reshuffles while all workers share the permutation.
        class_dirs = shard_and_shuffle(
            sorted([d for d in self.split_dir.iterdir() if d.is_dir()]),
            worker_slice=self.worker_slice,
            shuffle=self.shuffle,
            epoch_seed=self.epoch_seed,
        )

        for class_dir in class_dirs:
            synset = class_dir.name
            # Sort for deterministic, reproducible ordering across runs/filesystems.
            for image_path in sorted(class_dir.glob("*.JPEG")):
                sample: Sample = {
                    "key": image_path.stem,
                    "file_path": str(image_path),
                    "format": "jpg",
                    "label": synset,
                    "frames": 1,
                }
                yield sample

    def _iter_val(self) -> Iterator[Sample]:
        """Iterate validation split (flat directory)."""
        # Val is never shuffled; slice deterministically over sorted files.
        image_files = shard_and_shuffle(
            sorted(self.split_dir.glob("*.JPEG")),
            worker_slice=self.worker_slice,
        )

        for image_path in image_files:
            # Get label from validation labels mapping.
            if self.validation_labels:
                label = self.validation_labels.get(image_path.name, "unknown")
            else:
                label = "unknown"

            sample: Sample = {
                "key": image_path.stem,
                "file_path": str(image_path),
                "format": "jpg",
                "label": label,
                "frames": 1,
            }
            yield sample

    def __len__(self) -> int:
        """Return approximate number of samples."""
        if self.split == "train":
            return 1_281_167
        # Val.
        return 50_000
