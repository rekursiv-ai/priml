"""Custom types for data module."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from priml.custom_types import CheckpointableProtocol
from priml.timer import CheckpointableStepTimer


__all__ = [
    "DatasetProtocol",
]


@runtime_checkable
class DatasetProtocol(CheckpointableProtocol, Protocol):
    """Protocol for datasets with train/eval dataloaders.

    Extends CheckpointableProtocol to support resuming from checkpoints
    (iterator position, shuffling state, etc.).
    """

    timer_epoch: CheckpointableStepTimer
    """Passes over the training data: how many, and how long each took.

    Owned HERE because a pass is the loader's own boundary -- nothing else can
    say when the data ran out -- and checkpointed here for the same reason. A
    learnable annealing against epochs is handed this object rather than a
    copy, so the count it reads and the count that was saved are one number.

    Its meaning after a resume is the loader's to make true: a dataset that
    restores its position resumes the partial pass, and one that cannot must
    say so (see ``NanoChatData.load_state_dict``) rather than report a count
    that silently re-walks data."""

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
