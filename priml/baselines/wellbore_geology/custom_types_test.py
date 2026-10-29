"""Declared batch schema and training capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import get_protocol_members

from torch import nn

from priml.baselines.wellbore_geology.custom_types import (
    Batch,
    BatchLoader,
    InitializableEMA,
    RowWindow,
    ScheduleHorizon,
)


@dataclass(slots=True, kw_only=True)
class _Horizon:
    T_max: int = 3


@dataclass(slots=True, kw_only=True)
class _Window:
    prefix_rows: int = 1
    target_rows: int = 2
    downsample: int = 3


class _Ema:
    def initialize(self, model: nn.Module, *, reset: bool = False) -> None:
        del model, reset


def test_batch_schema_keys() -> None:
    assert Batch.__required_keys__ == {
        "media",
        "typewell_aux",
        "z_rel",
        "bin_count",
        "label",
        "typewell_target_probs",
        "aux_targets",
        "target_mask",
        "row_mask",
        "meta",
    }
    assert Batch.__optional_keys__ == frozenset()


def test_batch_loader_surface() -> None:
    assert get_protocol_members(BatchLoader) == {"__len__", "__iter__"}


def test_runtime_checkable_capabilities() -> None:
    assert isinstance(_Horizon(), ScheduleHorizon)
    assert not isinstance(_Window(), ScheduleHorizon)
    assert isinstance(_Window(), RowWindow)
    assert not isinstance(_Horizon(), RowWindow)
    assert isinstance(_Ema(), InitializableEMA)
    assert not isinstance(_Horizon(), InitializableEMA)
    assert InitializableEMA.initialize(_Ema(), nn.Identity()) is None


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
