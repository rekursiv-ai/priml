"""Tests for CIFAR-10 loading and preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.datasets

from priml.baselines.cifar10.data import Cifar10Data, prepare


def tiny_dataset(directory: Path, *, count: int = 8) -> Cifar10Data.Config:
    """Write a miniature prepared dataset and return a config reading it."""
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    for split in ("train", "test"):
        torch.save(
            {
                "media": torch.randn(count, 3, 8, 8, generator=generator),
                "label": torch.randint(0, 10, (count,), generator=generator),
            },
            directory / f"{split}.pt",
        )
    config = Cifar10Data.Config()
    config.base_dir = None
    config.working_dir = directory
    config.batch_size = 3
    config.eval_batch_size = 4
    config.device = "cpu"
    return config


def test_batches_carry_media_and_label(tmp_path: Path) -> None:
    data = tiny_dataset(tmp_path).make()
    batch = next(iter(data.train_dataloader()))
    assert set(batch) == {"media", "label"}
    assert batch["media"].shape == (3, 3, 8, 8)
    assert batch["label"].shape == (3,)


def test_train_loader_covers_every_image_exactly_once(tmp_path: Path) -> None:
    data = tiny_dataset(tmp_path, count=9).make()
    labels = torch.cat([batch["label"] for batch in data.train_dataloader()])
    assert labels.sort().values.tolist() == data.train_label.sort().values.tolist()


def test_train_loader_shuffles(tmp_path: Path) -> None:
    torch.manual_seed(0)
    data = tiny_dataset(tmp_path, count=64).make()
    orders = {
        tuple(torch.cat([b["label"] for b in data.train_dataloader()]).tolist())
        for _ in range(4)
    }
    assert len(orders) > 1


def test_eval_loader_preserves_dataset_order(tmp_path: Path) -> None:
    data = tiny_dataset(tmp_path).make()
    labels = torch.cat([batch["label"] for batch in data.eval_dataloader()])
    assert torch.equal(labels, data.eval_label)


def test_drop_last_discards_the_short_batch(tmp_path: Path) -> None:
    config = tiny_dataset(tmp_path, count=8)
    config.drop_last = True
    data = config.make()
    loader = data.train_dataloader()
    assert len(loader) == 2
    assert all(len(batch["label"]) == 3 for batch in loader)


def test_short_batch_is_yielded_by_default(tmp_path: Path) -> None:
    data = tiny_dataset(tmp_path, count=8).make()
    sizes = [len(batch["label"]) for batch in data.train_dataloader()]
    assert sizes == [3, 3, 2]


def test_missing_split_names_the_preparer(tmp_path: Path) -> None:
    config = Cifar10Data.Config()
    config.base_dir = None
    config.working_dir = tmp_path
    config.device = "cpu"
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        _ = config.make()


def test_foreign_cache_is_rejected_by_name(tmp_path: Path) -> None:
    # A cache written by another tool can occupy this filename with different
    # keys; the loader must say so rather than raise KeyError from a later line.
    torch.save(
        {"images": torch.zeros(1), "labels": torch.zeros(1)},
        tmp_path / "train.pt",
    )
    config = Cifar10Data.Config()
    config.base_dir = None
    config.working_dir = tmp_path
    config.device = "cpu"
    with pytest.raises(ValueError, match="not a prepared CIFAR-10 split"):
        _ = config.make()


def test_rejects_nonpositive_batch_size(tmp_path: Path) -> None:
    config = tiny_dataset(tmp_path)
    config.batch_size = 0
    with pytest.raises(ValueError, match="must be positive"):
        _ = config.make()


def test_working_dir_resolves_beneath_base_dir() -> None:
    config = Cifar10Data.Config()
    config.base_dir = "/opt/scratch"
    resolved = config.copy_tree().finalize()
    assert Path(resolved.working_dir) == Path("/opt/scratch/datasets/cifar10")


def test_state_dict_carries_the_pass_count_and_nothing_else(tmp_path: Path) -> None:
    # Batch ORDER derives from the loop's RNG, which the loop checkpoints
    # itself. The pass count is the loader's own: only it knows when the data
    # ran out, and a schedule annealing against epochs reads it.
    data = tiny_dataset(tmp_path).make()
    data.timer_epoch.global_count = 3
    restored = tiny_dataset(tmp_path).make()
    restored.load_state_dict(data.state_dict())
    assert set(data.state_dict()) == {"timer_epoch"}
    assert restored.timer_epoch.global_count == 3


def test_prepare_normalizes_and_writes_both_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The download path writes the schema the loader expects.

    ``torchvision.datasets.CIFAR10`` is stubbed: the test covers OUR
    conversion -- channel order, scaling, normalization, and the atomic
    rename -- not torchvision's downloader.
    """
    monkeypatch.setattr(torchvision.datasets, "CIFAR10", _StubCifar10)
    prepare(tmp_path, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    for split, expected in (("train", 1.0), ("test", -1.0)):
        payload = torch.load(tmp_path / f"{split}.pt", weights_only=True)
        # (N, H, W, C) uint8 becomes (N, C, H, W) float: the stub's 255 scales
        # to 1.0 then normalizes to (1 - 0.5) / 0.5 = 1.0, and its 0 to -1.0.
        assert payload["media"].shape == (2, 3, 4, 4)
        assert payload["media"].unique().tolist() == pytest.approx([expected])
        assert payload["label"].tolist() == [0, 1]
    # The staging file is renamed, never left behind for the existence check
    # above to later mistake for a complete split.
    assert not list(tmp_path.glob("*.partial"))


class _StubCifar10:
    """Stands in for torchvision's CIFAR-10, without the download."""

    def __init__(self, root: str, *, train: bool, download: bool) -> None:
        del root, download
        value = 255 if train else 0
        self.data = np.full((2, 4, 4, 3), value, dtype=np.uint8)
        self.targets = [0, 1]


def test_prepare_leaves_an_existing_split_untouched(tmp_path: Path) -> None:
    tiny_dataset(tmp_path)
    before = (tmp_path / "train.pt").read_bytes()
    prepare(tmp_path)
    assert (tmp_path / "train.pt").read_bytes() == before


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
