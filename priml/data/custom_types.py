"""Custom types for data module."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from priml.custom_types import CheckpointableProtocol


__all__ = [
    "DatasetProtocol",
]


@runtime_checkable
class DatasetProtocol(CheckpointableProtocol, Protocol):
    """Protocol for datasets with train/eval dataloaders.

    Extends CheckpointableProtocol to support resuming from checkpoints
    (iterator position, shuffling state, etc.).
    """

    def train_dataloader(self) -> Any:
        """Get training dataloader.

        Returns:
          dataloader: Iterator yielding batches (typically dict[str, Tensor]).

        """
        ...

    def eval_dataloader(self) -> Any:
        """Get evaluation dataloader.

        Returns:
          dataloader: Iterator yielding batches (typically dict[str, Tensor]).

        """
        ...
