"""Tests for the label-flipped training split and its two-split evaluation pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.tl.poison_robust_learning.data import (
    SPLITS,
    PoisonedData,
    flipped_labels,
)


if TYPE_CHECKING:
    from pathlib import Path


def write_split(directory: Path, *, count: int = 7, classes: int = 10) -> None:
    """Write ``count`` 5x6 images labelled ``index % classes`` as both splits."""
    media = (torch.arange(count * 3 * 5 * 6) % 251).reshape(count, 3, 5, 6)
    for split in ("train", "test"):
        torch.save(
            {"media": media.to(torch.uint8), "label": torch.arange(count) % classes},
            directory / f"{split}.pt",
        )


def config(directory: Path) -> PoisonedData.Config:
    """Return the dataset over ``directory`` with batches of two."""
    cfg = PoisonedData.Config()
    cfg.working_dir = directory
    cfg.batch_size = cfg.eval_batch_size = 2
    cfg.translate_pad = 1
    cfg.poison_fraction = 0.5
    cfg.num_classes = 10
    return cfg


def test_flipped_labels_shift_a_seeded_prefix_of_a_permutation_by_one() -> None:
    clean = torch.tensor([0, 1, 2, 9, 4, 5, 6])
    label, index = flipped_labels(clean, num_classes=10, fraction=0.5, seed=42)
    expected = torch.randperm(7, generator=torch.Generator().manual_seed(42))[:3]
    assert torch.equal(index, expected)
    assert torch.equal(label[index], (clean[index] + 1) % 10)
    untouched = torch.ones(7, dtype=torch.bool)
    untouched[index] = False
    assert torch.equal(label[untouched], clean[untouched])


def test_flipped_labels_poison_the_floor_of_the_fraction() -> None:
    clean = torch.zeros(9, dtype=torch.int64)
    _, index = flipped_labels(clean, num_classes=3, fraction=0.3, seed=0)
    assert len(index) == int(9 * 0.3)
    _, none = flipped_labels(clean, num_classes=3, fraction=0.1, seed=0)
    assert len(none) == 0


def test_flipped_labels_leave_the_global_generator_alone() -> None:
    state = torch.get_rng_state()
    _ = flipped_labels(torch.arange(5), num_classes=5, fraction=0.4, seed=3)
    assert torch.equal(torch.get_rng_state(), state)


def test_the_dataset_trains_on_the_flipped_labels(tmp_path: Path) -> None:
    write_split(tmp_path)
    data = config(tmp_path).make()
    clean = torch.arange(7)
    label, index = flipped_labels(clean, num_classes=10, fraction=0.5, seed=42)
    assert torch.equal(data.train_label, label)
    assert torch.equal(data.clean_label, clean)
    assert data.is_poison.tolist() == [i in index.tolist() for i in range(7)]
    labels = torch.cat([b["label"] for b in data.train_dataloader()])
    assert torch.equal(labels.sort().values, label.sort().values)


def test_a_dataset_without_a_class_count_is_refused(tmp_path: Path) -> None:
    write_split(tmp_path)
    cfg = config(tmp_path)
    cfg.num_classes = -1
    with pytest.raises(ValueError, match="num_classes"):
        cfg.make()


def test_evaluation_scores_the_test_split_then_the_augmented_poisoned_split(
    tmp_path: Path,
) -> None:
    write_split(tmp_path)
    cfg = config(tmp_path)
    data = cfg.make()
    loader = data.eval_dataloader()
    batches = list(loader)
    assert len(loader) == len(batches) == 8
    assert [SPLITS[int(b["split"])] for b in batches] == [SPLITS[0]] * 4 + [
        SPLITS[1],
    ] * 4
    test = batches[:4]
    mean, std = torch.tensor(cfg.mean), torch.tensor(cfg.std)
    assert torch.equal(
        torch.cat([b["media"] for b in test]),
        normalize(data.eval_media, mean=mean, std=std),
    )
    poison = batches[4:]
    assert torch.equal(torch.cat([b["label"] for b in poison]), data.train_label)
    assert torch.equal(torch.cat([b["poison"] for b in poison]), data.is_poison)
    assert all("poison" not in b for b in test)


def test_the_poisoned_pass_takes_one_seed_and_draws_each_batch_from_its_stream(
    tmp_path: Path,
) -> None:
    write_split(tmp_path)
    cfg = config(tmp_path)
    data = cfg.make()
    state = torch.get_rng_state()
    poison = [b for b in data.eval_dataloader() if int(b["split"]) == 1]
    torch.set_rng_state(state)
    # The test pass takes a seed of its own first; the poisoned pass the next.
    _ = torch.empty((), dtype=torch.int64).random_()
    seed = int(torch.empty((), dtype=torch.int64).random_())
    streams = [torch.Generator().manual_seed(seed + w) for w in range(cfg.num_streams)]
    augment = cfg.augment.make()
    mean, std = torch.tensor(cfg.mean), torch.tensor(cfg.std)
    for index, batch in enumerate(poison):
        images = data.train_media[2 * index : 2 * index + 2]
        expected = augment(
            images,
            streams[index % cfg.num_streams],
            pad=cfg.translate_pad,
            mean=mean,
            std=std,
        )
        assert torch.equal(batch["media"], expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
