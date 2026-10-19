"""Tests for the metric protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import torch

from priml.metrics.custom_types import MetricProtocol, RequiresDeviceTiming


if TYPE_CHECKING:
    from collections.abc import Mapping


class _Mean:
    def __init__(self) -> None:
        self.total = 0.0

    def update(self, logits: Tensor, **batch: object) -> None:
        del batch
        self.total += float(logits.sum())

    def compute(self) -> Mapping[str, object]:
        return {"total": self.total}

    def reset(self) -> None:
        self.total = 0.0

    def state_dict(self) -> dict[str, object]:
        return {"total": self.total}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        del state_dict


class _Synced(_Mean):
    @property
    def requires_device_timing(self) -> bool:
        return True


def test_metric_protocol_matches_a_checkpointable_metric() -> None:
    metric = _Mean()
    metric.update(torch.ones(2, 3))
    assert metric.compute() == {"total": 6.0}
    assert not isinstance(object(), MetricProtocol)


def test_requires_device_timing_is_an_opt_in_capability() -> None:
    assert isinstance(_Synced(), RequiresDeviceTiming)
    assert not isinstance(_Mean(), RequiresDeviceTiming)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
