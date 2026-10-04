"""Map-style paired ImageNet and autoencoder latents for the reference sampler order.

The experiment owns the corpus's producers: :class:`PairedImageLatentDataset`
declares the autoencoder that encoded the latents and the codec that stored
them, the preparer builds both from that declaration, and the loader checks the
corpus receipt against it -- so a corpus made by one autoencoder or codec is
never read as another's. Images live in one shared ``images/`` directory; each
corpus's latents live in their own subdirectory beside it.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

import re

from configgle import Fig, Makeable
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import numpy as np
import torch

from priml.baselines.speedrundit.corpus import (
    LABELS,
    PREPARING,
    CorpusMismatchError,
    load_stored,
    load_table,
    verify_receipt,
)
from priml.baselines.speedrundit.latent_codec import (
    FittedCodec,
    FloatCodec,
    LatentCodec,
)
from priml.data.sources.prepared_image_latents import read_image, read_labels
from priml.model.vision_ae.custom_types import VisionAutoencoderConfig
from priml.model.vision_ae.invae import INVAE
from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class PairedImageLatentDataset(Dataset[dict[str, Tensor]]):
    """Index processed RGB images and stored latents by shared ID."""

    class Config(Fig["PairedImageLatentDataset"]):
        """Where the corpus lives, and what produced its latents."""

        base_dir: Path | str | None = None
        """Optional resource root supplied by the train loop."""

        working_dir: Path | str = "/datasets/speedrundit"
        """Directory containing ``images/`` and the latent subdirectory."""

        latent_subdir: str = "vae-in"
        """This corpus's latents beside the shared images; REG's name for INVAE."""

        autoencoder: VisionAutoencoderConfig = field(default_factory=INVAE.Config)
        """Encoded the latents. Built by the preparer; the loader only checks it."""

        codec: Makeable[LatentCodec] = field(default_factory=FloatCodec.Config)
        """Stored the latents; the loader decodes each one with it."""

        seed: int | None = None
        """Seeds the preparer's encoding; ``None`` keeps the process's own state."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(
                self.base_dir,
                working_dir=self.working_dir,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        root = Path(config.working_dir)
        image_root = root / "images"
        latent_root = root / config.latent_subdir
        self.codec = config.codec.make()
        table_sha256 = (
            load_table(latent_root, codec=self.codec)
            if isinstance(self.codec, FittedCodec)
            else None
        )
        details = verify_receipt(
            latent_root,
            autoencoder=config.autoencoder,
            codec_config=config.codec,
            codec=self.codec,
            table_sha256=table_sha256,
        )
        if details.get("provenance") == PREPARING:
            raise CorpusMismatchError(
                f"{latent_root} is unfinished: scripts/prepare_data.py was "
                "interrupted or is still running. Rerun it to complete the corpus.",
            )
        self.latent_shape = config.autoencoder.latent_shape()
        labels = read_labels(latent_root / LABELS)
        images = _index(
            image_root,
            paths=(
                path
                for path in image_root.rglob("*")
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"}
            ),
        )
        latents = _index(latent_root, paths=latent_root.rglob("*.npy"))
        if not latents or latents.keys() - images.keys():
            raise ValueError(
                f"{latent_root} needs latents, each with an image in {image_root}",
            )
        self.records: list[tuple[Path, Path, int]] = []
        for key in sorted(latents):
            latent_path = latents[key]
            relative = latent_path.relative_to(latent_root).as_posix()
            if relative not in labels:
                raise ValueError(f"missing class label for {relative}")
            self.records.append((images[key], latent_path, int(labels[relative])))
        self.sampler: DistributedSampler[dict[str, Tensor]] | None = None

    def __len__(self) -> int:
        return len(self.records)

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_path, latent_path, label = self.records[index]
        image = read_image(image_path)
        # A ``.npy`` image is read as stored; the teacher rescales from [0, 255]
        # itself, so a float image in [0, 1] would be scaled a second time.
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"{image_path} holds a {image.dtype} {list(image.shape)} image; the "
                "teacher reads [3, H, W] uint8.",
            )
        stored = load_stored(latent_path, dtype=self.codec.stored_dtype)
        if stored.ndim == 4 and stored.shape[0] == 1:
            stored = stored[0]
        if tuple(stored.shape) != self.latent_shape:
            raise ValueError(
                f"{latent_path} holds a {tuple(stored.shape)} latent; the "
                f"autoencoder produces {self.latent_shape}.",
            )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "latent": self.codec.decode(stored),
            "label": torch.tensor(label, dtype=torch.int64),
        }

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle disjoint distributed partitions between epochs."""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)


class SpeedrunImageNetData:
    """Reference-style map DataLoader with priml's epoch and resume interface."""

    class Config(Fig["SpeedrunImageNetData"]):
        source: PairedImageLatentDataset.Config = field(
            default_factory=PairedImageLatentDataset.Config,
        )
        """Indexed image/latent pairs; narrowed so the loop reads its autoencoder."""

        batch_size: int = 32
        """Samples per device and optimizer step."""

        num_workers: int = 4
        """Image decoding worker count per process."""

        prefetch_factor: int = 2
        """Batches prefetched by each worker."""

        pin_memory: bool = True
        """Pin loaded tensors for GPU transfer."""

        base_dir: Path | str | None = None
        """Resource root supplied by the train loop."""

        working_dir: Path | str = "/"
        """Logical root used to resolve the source directory."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(
                self.base_dir,
                working_dir=self.working_dir,
            )
            if self.source.base_dir is None:
                self.source.base_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset = config.source.make()
        self.timer_epoch = CheckpointableStepTimer()

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Use torch's RandomSampler, matching the reference on one device."""
        return self._loader(shuffle=True)

    def eval_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Read a deterministic finite evaluation pass."""
        return self._loader(shuffle=False)

    def state_dict(self) -> dict[str, object]:
        """Save the epoch timer for a resumable run."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the epoch timer from a checkpoint."""
        self.timer_epoch.load_state_dict(
            cast(dict[str, object], state_dict["timer_epoch"]),
        )

    def _loader(self, *, shuffle: bool) -> DataLoader[dict[str, Tensor]]:
        sampler: DistributedSampler[dict[str, Tensor]] | None = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler[dict[str, Tensor]](
                self.dataset,
                shuffle=shuffle,
                drop_last=True,
            )
        if shuffle:
            self.dataset.sampler = sampler
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            # DataLoader rejects a prefetch factor when it runs no workers.
            prefetch_factor=self.config.prefetch_factor
            if self.config.num_workers
            else None,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )


def _index(root: Path, paths: Iterable[Path]) -> dict[str, Path]:
    """Key files by pair id, refusing two files that name one id."""
    indexed: dict[str, Path] = {}
    for path in paths:
        key = _pair_key(path.relative_to(root))
        if key in indexed:
            raise ValueError(f"{indexed[key]} and {path} name the same pair {key}.")
        indexed[key] = path
    return indexed


def _pair_key(relative: Path) -> str:
    """Normalize either the reference or priml preparer's image naming."""
    match = re.fullmatch(r"img(?:-latents-)?(\d{8})", relative.stem)
    stem = f"img{match.group(1)}" if match else relative.stem
    return (relative.parent / stem).as_posix()
