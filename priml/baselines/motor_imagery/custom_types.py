"""Capabilities the EEG networks require of the configs filling their slots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from configgle import Makeable

from priml.model.custom_types import HasOutputGrid, TensorModule


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from priml.baselines.motor_imagery.data import Trial


class SpatialConfig(Makeable[TensorModule], HasOutputGrid, Protocol):
    """Construct a tensor transform and declare its executed spatial geometry."""


@runtime_checkable
class ChannelsHidden(Protocol):
    """Has an explicit bottleneck width."""

    channels_hidden: int


@runtime_checkable
class Groups(Protocol):
    """Has a convolution group count."""

    groups: int


@runtime_checkable
class KernelSize(Protocol):
    """Has a two-dimensional kernel extent."""

    kernel_size: int | tuple[int, int]


class FoldPolicy(Protocol):
    """Partition recorded people into the validation cohorts of a campaign."""

    def __call__(self, trials: Sequence[Trial]) -> Iterator[list[str]]:
        """Yield each validation fold's people, leaving a training cohort."""
        ...
