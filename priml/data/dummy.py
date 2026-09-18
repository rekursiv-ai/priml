"""Dummy dataset for testing and default configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from configgle import Fig
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import torch

from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Mapping


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

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Get training dataloader.

        Returns:
          result: Shuffled DataLoader wrapping the dummy dataset.

        """
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )
        return cast(DataLoader[tuple[Tensor, Tensor]], loader)

    def eval_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Get evaluation dataloader.

        Returns:
          result: Non-shuffled DataLoader wrapping the dummy dataset.

        """
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )
        return cast(DataLoader[tuple[Tensor, Tensor]], loader)

    class StateDict(TypedDict):
        """The pass count; absent in checkpoints written before the timer existed."""

        timer_epoch: NotRequired[CheckpointableStepTimer.StateDict]

    def state_dict(self) -> StateDict:
        """Return the pass count, the only state this dataset carries.

        Returns:
          state: The epoch timer's state.

        """
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the pass count.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(DummyDataset.StateDict, state_dict)
        if "timer_epoch" in state:
            self.timer_epoch.load_state_dict(state["timer_epoch"])

    def _collate_fn(self, batch: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
        """Collate batch into dict format."""
        data_list, label_list = zip(*batch, strict=True)
        device = get_device(self.config.device)
        return {
            "media": torch.stack(data_list).to(device),
            "label": torch.stack(label_list).to(device),
        }
