"""Tests for the member split and the three-split evaluation loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import random

import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.tl.membership_inference_defense.data import (
    SPLITS,
    MembershipData,
    member_split,
    normalize_only,
)


if TYPE_CHECKING:
    from pathlib import Path


def data(directory: Path) -> MembershipData:
    """Write the synthetic split as both splits and load it at batch 2."""
    media, label = reference.synthetic_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = MembershipData.Config()
    config.working_dir = directory
    config.batch_size = config.eval_batch_size = 2
    return config.make()


def test_the_split_is_pythons_seeded_shuffle_cut_in_half() -> None:
    indices = list(range(7))
    random.Random(42).shuffle(indices)  # noqa: S311 -- The reference's own shuffle.
    member, nonmember = member_split(7, seed=42)
    assert member.tolist() == indices[:3]
    assert nonmember.tolist() == indices[3:]


def test_the_split_leaves_pythons_generator_where_it_was() -> None:
    random.seed(0)
    expected = random.random()  # noqa: S311 -- Probes the global stream.
    random.seed(0)
    _ = member_split(9, seed=42)
    assert random.random() == expected  # noqa: S311 -- Probes the global stream.


def test_training_reads_only_the_members(tmp_path: Path) -> None:
    loaded = data(tmp_path)
    member, nonmember = member_split(reference.LOADER_IMAGES, seed=42)
    assert torch.equal(loaded.train_label, member)
    assert torch.equal(loaded.nonmember_label, nonmember)
    with reference.seeded(reference.SEED):
        labels = torch.cat([b["label"] for b in loaded.train_dataloader()])
    assert torch.equal(labels.sort().values, member.sort().values)


def test_evaluation_passes_the_test_split_then_each_half(tmp_path: Path) -> None:
    loaded = data(tmp_path)
    member, nonmember = member_split(reference.LOADER_IMAGES, seed=42)
    loader = loaded.eval_dataloader()
    batches = list(loader)
    assert len(loader) == len(batches) == 4 + 2 + 2
    splits = [SPLITS[int(b["split"])] for b in batches]
    assert splits == ["test"] * 4 + ["member"] * 2 + ["nonmember"] * 2
    labels = [
        torch.cat([b["label"] for b, s in zip(batches, splits, strict=True) if s == k])
        for k in SPLITS
    ]
    assert torch.equal(labels[0], torch.arange(reference.LOADER_IMAGES))
    assert torch.equal(labels[1], member)
    assert torch.equal(labels[2], nonmember)


def test_each_evaluation_pass_takes_one_loader_seed(tmp_path: Path) -> None:
    loaded = data(tmp_path)
    with reference.seeded(0):
        _ = list(loaded.eval_dataloader())
        after = torch.get_rng_state()
    with reference.seeded(0):
        for _ in SPLITS:
            _ = torch.empty((), dtype=torch.int64).random_()
        assert torch.equal(torch.get_rng_state(), after)


def test_training_without_augmentation_only_normalizes() -> None:
    images = reference.fixed_images(2)
    mean, std = torch.tensor([0.5, 0.25, 0.75]), torch.tensor([0.5, 0.25, 0.125])
    stream = torch.Generator().manual_seed(1)
    state = stream.get_state()
    out = normalize_only(images, stream, pad=4, mean=mean, std=std)
    assert torch.equal(out, normalize(images, mean=mean, std=std))
    assert torch.equal(stream.get_state(), state)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
