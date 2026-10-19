"""Tests for the data-module protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.data.custom_types import DatasetProtocol, Processor, Source
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class _Dataset:
    def __init__(self) -> None:
        self.timer_epoch = CheckpointableStepTimer()

    def train_dataloader(self) -> list[object]:
        return [1]

    def eval_dataloader(self) -> list[object]:
        return [2]

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        del state_dict


class _NoEvalDataset:
    def __init__(self) -> None:
        self.timer_epoch = CheckpointableStepTimer()

    def train_dataloader(self) -> list[object]:
        return [1]

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        del state_dict


def _double(samples: Iterator[Mapping[str, object]]) -> Iterator[dict[str, object]]:
    for sample in samples:
        value = sample["x"]
        assert isinstance(value, int)
        yield {"x": 2 * value}


def test_dataset_protocol_requires_both_loaders_and_checkpointing() -> None:
    assert isinstance(_Dataset(), DatasetProtocol)
    assert not isinstance(_NoEvalDataset(), DatasetProtocol)


def test_source_and_processor_compose_structurally() -> None:
    source: Source[Mapping[str, object]] = [{"x": 1}, {"x": 3}]
    processor: Processor[Mapping[str, object], dict[str, object]] = _double
    assert list(processor(iter(source))) == [{"x": 2}, {"x": 6}]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
