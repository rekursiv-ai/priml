"""Dummy dataset for testing and default configuration."""

from __future__ import annotations

from typing import Any

from configgle import Fig
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import torch

from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


class DummyDataset:
    """Dummy dataset returning random tensors.

    Useful for testing and default configuration.
    """

    class Config(Fig["DummyDataset"]):
        """Dummy dataset configuration."""

        num_samples: int = 100
        """Synthetic examples generated once at construction."""

        batch_size: int = 32
        """Examples per batch from either loader."""

        input_shape: tuple[int, ...] = (3, 224, 224)
        """Shape of one example, excluding the batch axis."""

        num_classes: int = 1000
        """Label range; labels are drawn uniformly below this."""

        device: torch.device | str | None = "auto"
        """Device batches are delivered on. ``"auto"`` picks the best available,
        so the default dataset is usable on a CPU-only box."""

        seed: int = 0
        """Seed for the random data/labels so runs are reproducible."""
        num_workers: int = 0
        """DataLoader worker processes."""

    def __init__(self, config: Config) -> None:
        """Initialize dummy dataset.

        Args:
          config: Dataset configuration.

        """
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the data; ticked by the loop when the loader runs out."""

        # Seed a dedicated generator so the synthetic data is reproducible and
        # independent of the global torch RNG state.
        generator = torch.Generator().manual_seed(config.seed)
        data = torch.randn(config.num_samples, *config.input_shape, generator=generator)
        labels = torch.randint(
            0,
            config.num_classes,
            (config.num_samples,),
            generator=generator,
        )

        self.dataset = TensorDataset(data, labels)

    def train_dataloader(self) -> DataLoader[Any]:
        """Get training dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )

    def eval_dataloader(self) -> DataLoader[Any]:
        """Get evaluation dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the pass count, the only state this dataset carries."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the pass count."""
        if "timer_epoch" in state_dict:
            self.timer_epoch.load_state_dict(state_dict["timer_epoch"])

    def _collate_fn(self, batch: list[tuple[Tensor, ...]]) -> dict[str, Tensor]:
        """Collate batch into dict format.

        Args:
          batch: List of (data, label) tuples.

        Returns:
          dict: Batch dict with 'media' and 'label' keys.

        """
        data_list, label_list = zip(*batch, strict=True)
        device = get_device(self.config.device)
        return {
            "media": torch.stack(data_list).to(device),
            "label": torch.stack(label_list).to(device),
        }
