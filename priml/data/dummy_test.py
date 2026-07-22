"""Tests for DummyDataset."""

from __future__ import annotations

import torch

from priml.data.dummy import DummyDataset


def _make(seed: int = 0) -> DummyDataset:
    config = DummyDataset.Config(
        num_samples=8,
        batch_size=4,
        input_shape=(2, 4, 4),
        num_classes=10,
        device="cpu",
        seed=seed,
    )
    return DummyDataset(config)


def test_seed_is_reproducible():
    """Same seed yields identical synthetic data (M15)."""
    a = _make(seed=42).dataset.tensors[0]
    b = _make(seed=42).dataset.tensors[0]
    assert torch.equal(a, b)


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


def test_collate_produces_media_and_label():
    """Collate yields a dict with media and label batched tensors."""
    dataset = _make()
    batch = next(iter(dataset.eval_dataloader()))
    assert batch["media"].shape == (4, 2, 4, 4)
    assert batch["label"].shape == (4,)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
