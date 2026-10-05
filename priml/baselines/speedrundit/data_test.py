"""Processed ImageNet pairing and reference sampler order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import json
import multiprocessing as mp

from PIL import Image
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

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


_WORKER_CONTEXT: Final = "fork" if "fork" in mp.get_all_start_methods() else None
"""Forkserver, the default, starts its server per process: 1.1s against 0.05s."""


@runtime_checkable
class _SamplerOptions(Protocol):
    drop_last: bool
    shuffle: bool


def _prepared_pair(
    root: Path,
    name: str,
    label: int,
    image_suffix: str = ".png",
) -> None:
    image_dir = root / "images" / "00000"
    latent_dir = root / "vae-in" / "00000"
    image_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"img{name}{image_suffix}"
    if image_suffix == ".npy":
        np.save(image_path, np.full((3, 4, 5), label, dtype=np.uint8))
    else:
        Image.new("RGB", (4, 3), color=(label, 0, 0)).save(image_path)
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


def _assert_random_sampler(sampler: object) -> None:
    assert isinstance(sampler, RandomSampler)


def _assert_sequential_sampler(sampler: object) -> None:
    assert isinstance(sampler, SequentialSampler)


def _assert_distributed_sampler(sampler: object) -> None:
    assert isinstance(sampler, DistributedSampler)


def test_image_extensions_are_case_insensitive_and_supported(tmp_path: Path) -> None:
    for index, suffix in enumerate(
        (".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG", ".npy"),
    ):
        root = tmp_path / str(index)
        name = f"{index:08d}"
        _prepared_pair(root, name, index, suffix)
        (root / "vae-in" / "dataset.json").write_text(
            json.dumps({"labels": [[f"00000/img-latents-{name}.npy", index]]}),
        )
        assert len(PairedImageLatentDataset.Config(working_dir=root).make()) == 1


def test_unrecognized_image_names_keep_their_stem(tmp_path: Path) -> None:
    _prepared_pair(tmp_path, "00000001", 1)
    image = tmp_path / "images" / "00000" / "img00000001.png"
    latent = tmp_path / "vae-in" / "00000" / "img-latents-00000001.npy"
    image.rename(image.with_name("scene.png"))
    latent.rename(latent.with_name("scene.npy"))
    (tmp_path / "vae-in" / "dataset.json").write_text(
        json.dumps({"labels": [["00000/scene.npy", 1]]}),
    )

    dataset = PairedImageLatentDataset.Config(working_dir=tmp_path).make()

    assert dataset.records[0][0].name == "scene.png"


def test_unpaired_ids_are_rejected(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 2)
    (tmp_path / "images" / "00000" / "img00000000.png").unlink()
    with pytest.raises(ValueError, match="images/") as exc_info:
        PairedImageLatentDataset.Config(working_dir=tmp_path).make()
    assert str(exc_info.value) == "images/ and vae-in/ need matching nonempty IDs"


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
    data = _data(tmp_path)
    data.config.pin_memory = True
    # Built, not iterated: iterating a pinning loader warns on a host with no
    # accelerator (CPU-only CI), and the suite turns warnings into errors.
    assert data.eval_dataloader().pin_memory is True
    data.config.pin_memory = False
    loader = data.eval_dataloader()
    order = [batch["label"].tolist() for batch in loader]
    # drop_last discards the fifth sample.
    assert order == [[0, 1], [2, 3]]
    assert loader.num_workers == 0
    assert loader.pin_memory is False


def test_worker_loader_carries_its_prefetch_settings(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 5)
    data = _data(tmp_path, num_workers=2)
    data.config.pin_memory = True
    loader = data.train_dataloader()
    assert (loader.num_workers, loader.prefetch_factor) == (2, 2)
    assert loader.batch_size == 2
    assert loader.pin_memory is True
    assert loader.drop_last
    train_sampler: object = loader.sampler
    _assert_random_sampler(train_sampler)
    data.config.pin_memory = False
    eval_loader = data.eval_dataloader()
    eval_loader.multiprocessing_context = _WORKER_CONTEXT
    assert [batch["label"].shape[0] for batch in eval_loader] == [2, 2]
    eval_sampler: object = eval_loader.sampler
    _assert_sequential_sampler(eval_sampler)


def test_distributed_loader_uses_a_reshuffled_sampler(tmp_path: Path) -> None:
    _labelled_pairs(tmp_path, 4)
    data = _data(tmp_path, num_workers=2)
    dist.init_process_group(
        backend="gloo",
        init_method=(tmp_path / "gloo-rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        loader = data.train_dataloader()
        sampler: object = loader.sampler
        _assert_distributed_sampler(sampler)
        assert isinstance(sampler, _SamplerOptions)
        assert sampler.drop_last is True
        assert sampler.shuffle is True
        eval_sampler: object = data.eval_dataloader().sampler
        _assert_distributed_sampler(eval_sampler)
        assert isinstance(eval_sampler, _SamplerOptions)
        assert eval_sampler.drop_last is True
        assert eval_sampler.shuffle is False
        assert loader.num_workers == 2
        assert loader.prefetch_factor == 2
        assert loader.pin_memory is False
    finally:
        dist.destroy_process_group()
    assert data.dataset.sampler is sampler
    data.dataset.set_epoch(3)
    # One rank's sampler draws the whole permutation seeded by seed 0 + epoch,
    # and the loader reads its batches in that order.
    order = torch.randperm(4, generator=torch.Generator().manual_seed(3))
    loader.multiprocessing_context = _WORKER_CONTEXT
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
