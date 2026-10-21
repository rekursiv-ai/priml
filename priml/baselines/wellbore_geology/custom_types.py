"""Typed complete batches and training capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor, nn


class Batch(TypedDict):
    """Complete native training tensor and original-row metadata schema."""

    media: Tensor
    typewell_aux: Tensor
    z_rel: Tensor
    bin_count: Tensor
    label: Tensor
    typewell_target_probs: Tensor
    aux_targets: dict[str, Tensor]
    target_mask: Tensor
    row_mask: Tensor
    meta: list[dict[str, object]]


class BatchLoader(Protocol):
    """Iterable and length surface of complete collated training loaders."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[dict[str, object]]: ...


@runtime_checkable
class ScheduleHorizon(Protocol):
    """A learning-rate schedule whose horizon in updates the fold loop derives."""

    T_max: int


@runtime_checkable
class InitializableEMA(Protocol):
    """An averaging policy whose initial snapshot can precede the first update."""

    def initialize(self, model: nn.Module, *, reset: bool = False) -> None:
        """Snapshot complete model state without advancing update counters."""
        ...


@runtime_checkable
class RowWindow(Protocol):
    """A consumer of the family's original-row window, unset while negative."""

    prefix_rows: int
    target_rows: int
    downsample: int
