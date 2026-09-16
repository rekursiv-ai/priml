"""Custom types for data module."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol, TypeVar, runtime_checkable

from priml.custom_types import CheckpointableProtocol
from priml.timer import CheckpointableStepTimer


__all__ = [
    "DatasetProtocol",
    "Processor",
    "Source",
]

_InputSampleT_contra = TypeVar(
    "_InputSampleT_contra",
    bound=Mapping[str, object],
    contravariant=True,
)
_OutputSampleT_co = TypeVar(
    "_OutputSampleT_co",
    bound=Mapping[str, object],
    covariant=True,
)


class Source(Protocol[_OutputSampleT_co]):
    """Dataset iterator that yields samples."""

    def __iter__(self) -> Iterator[_OutputSampleT_co]:
        """Iterate over samples."""
        ...


class Processor(Protocol[_InputSampleT_contra, _OutputSampleT_co]):
    """Process a stream of samples, yielding transformed samples.

    Processors can:
    1. Filter: Yield samples that pass, skip those that don't (0 or 1 output per input)
    2. Transform: Yield transformed samples (1 output per input)
    3. Batch: Consume multiple samples, yield batches (N inputs -> 1 output)
    4. Expand: Yield multiple samples per input (1 input -> N outputs)

    Filter behavior convention:
    Filters typically only add filter_reasons to mark samples for filtering.
    Sample mutation (adding/modifying fields) is typically an optimization only.
    This allows filters to be composable without tight coupling to downstream processors.
    """

    def __call__(
        self,
        samples: Iterator[_InputSampleT_contra],
    ) -> Iterator[_OutputSampleT_co]:
        """Apply to the input."""
        ...


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

    def train_dataloader(self) -> Iterable[object]:
        """Get training dataloader.

        Returns:
          dataloader: Iterable yielding batches (typically dict[str, Tensor]).

        """
        ...

    def eval_dataloader(self) -> Iterable[object]:
        """Get evaluation dataloader.

        Returns:
          dataloader: Iterable yielding batches (typically dict[str, Tensor]).

        """
        ...
