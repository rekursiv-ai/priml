"""Declared capabilities the EEG networks require of their configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import get_protocol_members

from priml.baselines.motor_imagery.custom_types import (
    ChannelsHidden,
    FoldPolicy,
    Groups,
    KernelSize,
    SpatialConfig,
)
from priml.model.custom_types import HasOutputGrid


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from priml.baselines.motor_imagery.data import Trial


@dataclass(slots=True, kw_only=True)
class _Hidden:
    channels_hidden: int = 4


@dataclass(slots=True, kw_only=True)
class _Grouped:
    groups: int = 2


@dataclass(slots=True, kw_only=True)
class _Kernel:
    kernel_size: int | tuple[int, int] = (3, 3)


class _Folds:
    def __call__(self, trials: Sequence[Trial]) -> Iterator[list[str]]:
        del trials
        return iter(())


def test_runtime_checkable_capabilities_match_by_field() -> None:
    assert isinstance(_Hidden(), ChannelsHidden)
    assert not isinstance(_Hidden(), Groups)
    assert isinstance(_Grouped(), Groups)
    assert not isinstance(_Grouped(), KernelSize)
    assert isinstance(_Kernel(), KernelSize)
    assert not isinstance(_Kernel(), ChannelsHidden)
    assert get_protocol_members(KernelSize) == {"kernel_size"}


def test_spatial_config_is_a_protocol_with_grid() -> None:
    assert {"make", "output_grid"} <= get_protocol_members(SpatialConfig)
    assert HasOutputGrid in SpatialConfig.__mro__


def test_fold_policy_call_is_declared() -> None:
    assert get_protocol_members(FoldPolicy) == {"__call__"}
    assert FoldPolicy.__call__(_Folds(), []) is None


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
