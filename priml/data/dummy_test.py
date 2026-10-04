"""Tests for DummyDataset."""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

from torch.utils.data import DataLoader

import pytest
import torch

from priml.data import dummy
from priml.data.dummy import DummyDataset


def _make(seed: int = 0) -> DummyDataset:
    config = DummyDataset.Config(
        num_samples=8,
        batch_size=5,
        input_shape=(2, 3, 4),
        num_classes=10,
        device="cpu",
        seed=seed,
    )
    return DummyDataset(config)


def _tensor_field(batch: object, key: str) -> torch.Tensor:
    assert isinstance(batch, dict)
    value = cast(object, batch[key])
    assert isinstance(value, torch.Tensor)
    return value


def test_seed_is_reproducible():
    """Same seed yields identical synthetic data and ignores global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        a = _make(seed=42).dataset.tensors
        torch.manual_seed(99)
        b = _make(seed=42).dataset.tensors
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert a[1].min() == 0
    assert a[1].max() < 10


def test_different_seeds_differ():
    """Different seeds yield different synthetic data (M15)."""
    a = _make(seed=1).dataset.tensors[0]
    b = _make(seed=2).dataset.tensors[0]
    assert not torch.equal(a, b)


def test_num_workers_propagates_to_loader():
    """num_workers config reaches the DataLoader (M15)."""
    config = DummyDataset.Config(
        num_samples=8,
        batch_size=4,
        input_shape=(2, 4, 4),
        num_classes=10,
        device="cpu",
        num_workers=2,
    )
    dataset = DummyDataset(config)
    assert dataset.train_dataloader().num_workers == 2
    assert dataset.eval_dataloader().num_workers == 2


def test_loaders_batch_configured_samples_with_expected_sampling() -> None:
    dataset = _make()
    train = cast(DataLoader[object], dataset.train_dataloader())
    evaluation = cast(DataLoader[object], dataset.eval_dataloader())
    assert train.batch_size == evaluation.batch_size == 5
    train_sampler = cast(object, train.sampler)
    evaluation_sampler = cast(object, evaluation.sampler)
    assert isinstance(train_sampler, torch.utils.data.RandomSampler)
    assert isinstance(evaluation_sampler, torch.utils.data.SequentialSampler)
    assert isinstance(next(iter(train)), dict)
    eval_batches = list(evaluation)
    assert [_tensor_field(batch, "media").shape[0] for batch in eval_batches] == [5, 3]
    assert torch.equal(
        torch.cat([_tensor_field(batch, "media") for batch in eval_batches]),
        dataset.dataset.tensors[0],
    )
    assert torch.equal(
        torch.cat([_tensor_field(batch, "label") for batch in eval_batches]),
        dataset.dataset.tensors[1],
    )


def test_state_roundtrip_and_legacy_state() -> None:
    dataset = _make()
    dataset.load_state_dict({"timer_epoch": {"global_count": 7, "global_sec": 2.5}})
    state = dataset.state_dict()
    assert "timer_epoch" in state
    assert state == {"timer_epoch": {"global_count": 7, "global_sec": 2.5}}
    dataset.load_state_dict({})
    assert dataset.state_dict() == state
    dataset.load_state_dict({"TIMER_EPOCH": state["timer_epoch"]})
    assert dataset.state_dict() == state


def test_collate_produces_media_and_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collate batches both tensors onto the configured device."""
    dataset = _make()
    resolve_device = Mock(return_value=torch.device("meta"))
    monkeypatch.setattr(dummy, "get_device", resolve_device)
    loader = cast(DataLoader[object], dataset.eval_dataloader())
    batch_obj = next(iter(loader))
    assert isinstance(batch_obj, dict)
    media = cast(torch.Tensor, batch_obj["media"])
    label = cast(torch.Tensor, batch_obj["label"])
    assert media.shape == (5, 2, 3, 4)
    assert label.shape == (5,)
    assert media.device.type == "meta"
    assert label.device.type == "meta"
    assert resolve_device.call_args.args == ("cpu",)
    with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
        dataset._collate_fn([])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
