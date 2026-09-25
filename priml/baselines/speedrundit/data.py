"""Map-style paired ImageNet and INVAE data for the reference sampler order."""

# NumPy's load return type is imprecise in its stubs.
# pyright: reportAny=false

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

from priml.data.sources.prepared_image_latents import read_image, read_labels
from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Mapping


def _pair_key(relative: Path) -> str:
    """Normalize either the reference or priml preparer's image naming."""
    match = re.fullmatch(r"img(?:-latents-)?(\d{8})", relative.stem)
    stem = f"img{match.group(1)}" if match else relative.stem
    return (relative.parent / stem).as_posix()


class PairedImageLatentDataset(Dataset[dict[str, Tensor]]):
    """Index processed RGB images and sampled INVAE arrays by shared ID."""

    class Config(Fig["PairedImageLatentDataset"]):
        base_dir: Path | str | None = None
        """Optional resource root supplied by the train loop."""

        working_dir: Path | str = "/datasets/speedrundit"
        """Directory containing ``images/`` and ``vae-in/``."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        root = Path(config.working_dir)
        image_root = root / "images"
        latent_root = root / "vae-in"
        labels = read_labels(latent_root / "dataset.json")
        images = {
            _pair_key(path.relative_to(image_root)): path
            for path in image_root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy"}
        }
        latents = {
            _pair_key(path.relative_to(latent_root)): path
            for path in latent_root.rglob("*.npy")
        }
        if not images or images.keys() != latents.keys():
            raise ValueError("images/ and vae-in/ need matching nonempty IDs")
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
        latent = np.load(latent_path)
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent[0]
        if latent.ndim != 3 or latent.shape[0] != 32:
            raise ValueError(f"expected 32-channel INVAE latent, got {latent.shape}")
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "latent": torch.from_numpy(np.ascontiguousarray(latent)),
            "label": torch.tensor(label, dtype=torch.int64),
        }

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle disjoint distributed partitions between epochs."""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)


class SpeedrunImageNetData:
    """Reference-style map DataLoader with priml's epoch and resume interface."""

    class Config(Fig["SpeedrunImageNetData"]):
        source: Makeable[PairedImageLatentDataset] = field(
            default_factory=PairedImageLatentDataset.Config
        )
        """Indexed processed ImageNet/INVAE pairs."""

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
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if (
                isinstance(self.source, PairedImageLatentDataset.Config)
                and self.source.base_dir is None
            ):
                self.source.base_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset = config.source.make()
        self.timer_epoch = CheckpointableStepTimer()

    def _loader(self, *, shuffle: bool) -> DataLoader[dict[str, Tensor]]:
        sampler: DistributedSampler[dict[str, Tensor]] | None = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self.dataset, shuffle=shuffle, drop_last=True)
        if shuffle:
            self.dataset.sampler = sampler
        if self.config.num_workers:
            return DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle and sampler is None,
                sampler=sampler,
                num_workers=self.config.num_workers,
                prefetch_factor=self.config.prefetch_factor,
                pin_memory=self.config.pin_memory,
                drop_last=True,
            )
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=0,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

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
            cast(dict[str, object], state_dict["timer_epoch"])
        )
