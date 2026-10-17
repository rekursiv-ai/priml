"""Tests for the multi-task CIFAR-100 loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.multitask.data import (
    COARSE_OF_FINE,
    NUM_COARSE,
    MultiTaskCifar100Data,
)


if TYPE_CHECKING:
    from pathlib import Path


def config(directory: Path) -> MultiTaskCifar100Data.Config:
    """Write five 4x6 images and return a loader config over them, batches of two."""
    media = torch.arange(5 * 3 * 4 * 6).reshape(5, 3, 4, 6).to(torch.uint8)
    for split in ("train", "test"):
        torch.save(
            {"media": media, "label": torch.tensor([0, 9, 42, 99, 7])},
            directory / f"{split}.pt",
        )
    cfg = MultiTaskCifar100Data.Config()
    cfg.data.working_dir = directory
    cfg.data.batch_size = cfg.data.eval_batch_size = 2
    cfg.data.translate_pad = 1
    return cfg


def test_the_map_names_a_superclass_for_every_class() -> None:
    assert len(COARSE_OF_FINE) == 100
    assert set(COARSE_OF_FINE) == set(range(NUM_COARSE))


def test_every_batch_carries_the_superclass_of_each_label(tmp_path: Path) -> None:
    loader = config(tmp_path).make()
    for split in (loader.train_dataloader(), loader.eval_dataloader()):
        batches = list(split)
        assert len(split) == len(batches) == 3
        for batch in batches:
            labels = from_plain(batch["label"].tolist(), list[int])
            expected = [COARSE_OF_FINE[c] for c in labels]
            assert batch["label_coarse"].tolist() == expected


def test_the_media_and_draws_are_the_vision_loaders(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    plain = cfg.data.make()
    torch.manual_seed(3)
    ours = list(cfg.make().train_dataloader())
    torch.manual_seed(3)
    theirs = list(plain.train_dataloader())
    for got, want in zip(ours, theirs, strict=True):
        assert torch.equal(got["media"], want["media"])
        assert torch.equal(got["label"], want["label"])


def test_a_checkpoint_resumes_inside_a_pass(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    torch.manual_seed(5)
    loader = cfg.make()
    passes = iter(loader.train_dataloader())
    _ = next(passes)
    state = loader.state_dict()
    rest = [batch["label"] for batch in passes]
    resumed = cfg.make()
    resumed.load_state_dict(state)
    again = [batch["label"] for batch in resumed.train_dataloader()]
    assert [t.tolist() for t in again] == [t.tolist() for t in rest]
    assert resumed.timer_epoch is resumed.data.timer_epoch


def test_the_root_pushes_its_base_dir_into_the_vision_data(tmp_path: Path) -> None:
    cfg = MultiTaskCifar100Data.Config()
    cfg.base_dir = tmp_path
    assert cfg.copy_tree().finalize().data.working_dir == tmp_path / "datasets/cifar100"


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
