"""Processed ImageNet pairing and reference sampler order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from PIL import Image

import numpy as np
import pytest
import torch
import torch.distributed as dist

from priml.baselines.speedrundit.data import (
    PairedImageLatentDataset,
    SpeedrunImageNetData,
)


if TYPE_CHECKING:
    from pathlib import Path


def _prepared_pair(root: Path, name: str, label: int) -> None:
    image_dir = root / "images" / "00000"
    latent_dir = root / "vae-in" / "00000"
    image_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color=(label, 0, 0)).save(image_dir / f"img{name}.png")
    # The loader strips the leading axis and requires 32 INVAE channels.
    np.save(
        latent_dir / f"img-latents-{name}.npy",
        np.full((1, 32, 2, 3), label, dtype=np.float32),
    )


def test_reference_names_and_tensor_values(tmp_path: Path) -> None:
    labels: list[list[str | int]] = []
    for index in range(10):
        name = f"{index:08d}"
        _prepared_pair(tmp_path, name, index + 5)
        labels.append([f"00000/img-latents-{name}.npy", index + 5])
    (tmp_path / "vae-in" / "dataset.json").write_text(json.dumps({"labels": labels}))
    dataset = PairedImageLatentDataset.Config(working_dir=tmp_path).make()
    assert len(dataset) == 10
    assert torch.equal(dataset[0]["latent"], torch.full((32, 2, 3), 5.0))
    assert dataset[0]["image"][0, 0, 0] == 5
    assert dataset[0]["label"] == 5

    config = SpeedrunImageNetData.Config()
    assert isinstance(config.source, PairedImageLatentDataset.Config)
    config.source.working_dir = tmp_path
    config.batch_size = 2
    config.num_workers = 0
    config.pin_memory = False
    data = config.make()
    torch.manual_seed(0)
    order = [batch["label"].tolist() for batch in data.train_dataloader()]
    assert order == [[11, 12], [6, 9], [7, 5], [14, 13], [8, 10]]


def _labelled_pairs(root: Path, count: int) -> None:
    labels: list[list[str | int]] = []
    for index in range(count):
        name = f"{index:08d}"
        _prepared_pair(root, name, index)
        labels.append([f"00000/img-latents-{name}.npy", index])
    (root / "vae-in" / "dataset.json").write_text(json.dumps({"labels": labels}))


def _data(root: Path, *, num_workers: int = 0) -> SpeedrunImageNetData:
    config = SpeedrunImageNetData.Config()
    assert isinstance(config.source, PairedImageLatentDataset.Config)
    config.source.working_dir = root
    config.batch_size = 2
    config.num_workers = num_workers
    config.pin_memory = False
    return config.make()


def test_unpaired_ids_are_rejected(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 2)
    (tmp_path / "images" / "00000" / "img00000000.png").unlink()
    with pytest.raises(ValueError, match="matching nonempty IDs"):
        PairedImageLatentDataset.Config(working_dir=tmp_path).make()


def test_unlabelled_latent_is_rejected(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 2)
    (tmp_path / "vae-in" / "dataset.json").write_text(
        json.dumps({"labels": [["00000/img-latents-00000000.npy", 0]]}),
    )
    with pytest.raises(ValueError, match="missing class label for 00000/img-latents"):
        PairedImageLatentDataset.Config(working_dir=tmp_path).make()


def test_latent_without_32_channels_is_rejected(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 2)
    np.save(
        tmp_path / "vae-in" / "00000" / "img-latents-00000000.npy",
        np.zeros((3, 2, 4), dtype=np.float32),
    )
    dataset = PairedImageLatentDataset.Config(working_dir=tmp_path).make()
    with pytest.raises(ValueError, match=r"32-channel INVAE latent, got \(3, 2, 4\)"):
        dataset[0]


def test_eval_loader_reads_every_full_batch_in_order(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 5)
    order = [batch["label"].tolist() for batch in _data(tmp_path).eval_dataloader()]
    # drop_last discards the fifth sample.
    assert order == [[0, 1], [2, 3]]


def test_worker_loader_carries_its_prefetch_settings(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 4)
    loader = _data(tmp_path, num_workers=2).train_dataloader()
    assert (loader.num_workers, loader.prefetch_factor) == (2, 2)
    assert loader.drop_last


def test_distributed_loader_uses_a_reshuffled_sampler(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 4)
    data = _data(tmp_path)
    dist.init_process_group(
        backend="gloo",
        init_method=(tmp_path / "gloo-rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        loader = data.train_dataloader()
    finally:
        dist.destroy_process_group()
    assert data.dataset.sampler is not None
    data.dataset.set_epoch(3)
    # One rank's sampler draws the whole permutation seeded by seed 0 + epoch,
    # and the loader reads its batches in that order.
    order = torch.randperm(4, generator=torch.Generator().manual_seed(3))
    labels = [batch["label"] for batch in loader]
    assert torch.equal(torch.cat(labels), order)


def test_set_epoch_without_a_sampler_is_a_no_op(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 2)
    dataset = PairedImageLatentDataset.Config(working_dir=tmp_path).make()
    dataset.set_epoch(3)
    assert dataset.sampler is None


def test_state_dict_round_trips_the_epoch_timer(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 2)
    source = _data(tmp_path)
    with source.timer_epoch:
        pass
    restored = _data(tmp_path)
    restored.load_state_dict(source.state_dict())
    assert restored.timer_epoch.global_count == 1
    assert restored.state_dict() == source.state_dict()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
