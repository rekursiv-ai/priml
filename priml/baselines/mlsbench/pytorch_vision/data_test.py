"""Tests for the CIFAR-100 loader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, cast

import logging

from configgle import PartialConfig
from torch import Tensor
from torch.nn import functional
from torchvision import datasets

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision.data import (
    Cifar100Data,
    normalize,
    pad_crop_flip,
    prepare,
)
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Generator


COUNT: Final = 5
"""Images in the test split."""

CHANNELS: Final = 3
"""The statistics in the config are per channel of an RGB image."""

HEIGHT: Final = 4
WIDTH: Final = 5


@pytest.fixture(autouse=True)
def restore_generator() -> Generator[None]:
    """Give the test the global generator back as it found it."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def seed(value: int) -> None:
    """Seed the global CPU generator."""
    torch.set_rng_state(torch.Generator().manual_seed(value).get_state())


def write_split(directory: Path) -> Tensor:
    """Write both splits of ``COUNT`` images labelled by their index."""
    media = (torch.arange(COUNT * CHANNELS * HEIGHT * WIDTH) % 251).to(torch.uint8)
    media = media.reshape(COUNT, CHANNELS, HEIGHT, WIDTH)
    for split in ("train", "test"):
        torch.save(
            {"media": media, "label": torch.arange(COUNT)},
            directory / f"{split}.pt",
        )
    return media


def make_data(
    directory: Path,
    *,
    batch_size: int = 2,
    streams: int = 4,
) -> Cifar100Data:
    """Build the dataset over ``directory`` with test-sized settings."""
    config = Cifar100Data.Config()
    config.base_dir = None
    config.working_dir = directory
    config.batch_size = config.eval_batch_size = batch_size
    config.num_streams = streams
    config.translate_pad = 1
    return config.make()


def test_normalize_scales_to_the_unit_interval_then_standardizes() -> None:
    images = torch.zeros(2, CHANNELS, 2, 2, dtype=torch.uint8)
    images[0] = 255
    mean, std = torch.tensor([0.5, 0.25, 0.0]), torch.tensor([0.25, 0.5, 2.0])
    out = normalize(images, mean=mean, std=std)
    assert torch.equal(
        out[0],
        torch.tensor([2.0, 1.5, 0.5]).view(3, 1, 1).expand(3, 2, 2),
    )
    assert torch.equal(
        out[1],
        torch.tensor([-2.0, -0.5, 0.0]).view(3, 1, 1).expand(3, 2, 2),
    )


def test_pad_crop_flip_draws_top_then_left_then_mirror_per_image() -> None:
    images = (torch.arange(6 * 2 * HEIGHT * WIDTH) % 251).to(torch.uint8)
    images = images.reshape(6, 2, HEIGHT, WIDTH)
    out = pad_crop_flip(images, torch.Generator().manual_seed(7), pad=1)
    replay = torch.Generator().manual_seed(7)
    padded = functional.pad(images, (1,) * 4)
    mirrored: list[bool] = []
    for index in range(6):
        top = int(torch.randint(0, 3, (1,), generator=replay))
        left = int(torch.randint(0, 3, (1,), generator=replay))
        mirrored.append(bool(torch.rand(1, generator=replay) < 0.5))
        crop = padded[index, :, top : top + HEIGHT, left : left + WIDTH]
        assert torch.equal(out[index], crop.flip(-1) if mirrored[-1] else crop)
    assert set(mirrored) == {True, False}


def test_pad_crop_flip_pads_with_zeros() -> None:
    images = torch.full((6, 2, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    out = pad_crop_flip(images, torch.Generator().manual_seed(7), pad=1)
    assert out.shape == images.shape
    assert out.unique().tolist() == [0, 255]


def test_train_pass_visits_every_image_once_in_shuffled_batches(tmp_path: Path) -> None:
    media = write_split(tmp_path)
    data = make_data(tmp_path)
    seed(0)
    batches = list(data.train_dataloader())
    labels = torch.cat([batch["label"] for batch in batches])
    assert [len(batch["label"]) for batch in batches] == [2, 2, 1]
    assert labels.sort().values.tolist() == list(range(COUNT))
    assert labels.tolist() != list(range(COUNT))
    assert batches[0]["media"].shape == (2, CHANNELS, HEIGHT, WIDTH)
    assert batches[0]["media"].dtype == torch.float32
    assert media.dtype == torch.uint8


def test_a_pass_takes_a_stream_seed_then_a_shuffle_seed_from_the_global_generator(
    tmp_path: Path,
) -> None:
    _ = write_split(tmp_path)
    loader = make_data(tmp_path).train_dataloader()
    seed(0)
    _ = list(loader)
    after_pass = rng_fingerprint()
    seed(0)
    _ = torch.empty((), dtype=torch.int64).random_()
    _ = torch.empty((), dtype=torch.int64).random_()
    assert torch.equal(after_pass, rng_fingerprint())


def test_batches_follow_the_two_seeds_and_round_robin_streams(tmp_path: Path) -> None:
    media = write_split(tmp_path)
    loader = make_data(tmp_path, streams=2).train_dataloader()
    seed(0)
    stream_seed = int(torch.empty((), dtype=torch.int64).random_())
    shuffle_seed = int(torch.empty((), dtype=torch.int64).random_())
    seed(0)
    order = torch.randperm(COUNT, generator=torch.Generator().manual_seed(shuffle_seed))
    streams = [
        torch.Generator().manual_seed(stream_seed + worker) for worker in range(2)
    ]
    mean, std = (
        torch.tensor(Cifar100Data.Config().mean),
        torch.tensor(Cifar100Data.Config().std),
    )
    for index, batch in enumerate(loader):
        chosen = order[2 * index : 2 * index + 2]
        expected = normalize(
            pad_crop_flip(media[chosen], streams[index % 2], pad=1),
            mean=mean,
            std=std,
        )
        assert torch.equal(batch["media"], expected)
        assert torch.equal(batch["label"], chosen)


def flipped(
    images: Tensor,
    stream: torch.Generator,
    /,
    *,
    pad: int,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Mirror every image and normalize; draws nothing from ``stream``."""
    del stream, pad
    return normalize(images.flip(-1), mean=mean, std=std)


def test_the_augment_slot_replaces_the_training_pipeline_only(tmp_path: Path) -> None:
    media = write_split(tmp_path)
    config = make_data(tmp_path).config
    config.augment = PartialConfig(flipped)
    data = config.make()
    seed(0)
    (first, *_) = list(data.train_dataloader())
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    expected = normalize(media[first["label"]].flip(-1), mean=mean, std=std)
    assert torch.equal(first["media"], expected)
    (evaluated, *_) = list(data.eval_dataloader())
    assert torch.equal(evaluated["media"], normalize(media[:2], mean=mean, std=std))


def test_eval_pass_is_sequential_unaugmented_and_draws_one_unused_seed(
    tmp_path: Path,
) -> None:
    media = write_split(tmp_path)
    loader = make_data(tmp_path).eval_dataloader()
    seed(0)
    batches = list(loader)
    after_pass = rng_fingerprint()
    seed(0)
    _ = torch.empty((), dtype=torch.int64).random_()
    assert torch.equal(after_pass, rng_fingerprint())
    config = Cifar100Data.Config()
    expected = normalize(
        media,
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    assert torch.equal(torch.cat([batch["media"] for batch in batches]), expected)
    assert torch.cat([batch["label"] for batch in batches]).tolist() == list(
        range(COUNT),
    )


def test_loaders_know_their_length(tmp_path: Path) -> None:
    _ = write_split(tmp_path)
    data = make_data(tmp_path)
    assert len(data.train_dataloader()) == 3
    assert len(data.eval_dataloader()) == 3


def test_a_loader_between_passes_has_no_cursor(tmp_path: Path) -> None:
    _ = write_split(tmp_path)
    loader = make_data(tmp_path).train_dataloader()
    empty: dict[str, object] = {"order": None, "streams": [], "next_batch": 0}
    assert loader.state_dict() == empty
    seed(0)
    _ = list(loader)
    assert loader.state_dict() == empty


def test_a_pass_resumes_from_its_state_exactly(tmp_path: Path) -> None:
    _ = write_split(tmp_path)
    first = make_data(tmp_path).train_dataloader()
    seed(0)
    iterator = iter(first)
    _ = next(iterator)
    state = first.state_dict()
    assert state["next_batch"] == 1
    rest = list(iterator)
    resumed = make_data(tmp_path).train_dataloader()
    resumed.load_state_dict(state)
    again = list(resumed)
    assert len(again) == len(rest) == 2
    for want, got in zip(rest, again, strict=True):
        assert torch.equal(want["media"], got["media"])
        assert torch.equal(want["label"], got["label"])


def test_dataset_state_holds_the_loader_and_the_pass_count(tmp_path: Path) -> None:
    _ = write_split(tmp_path)
    data = make_data(tmp_path)
    assert data.state_dict()["loader"] is None
    data.timer_epoch.global_count = 3
    seed(0)
    iterator = iter(data.train_dataloader())
    _ = next(iterator)
    state = data.state_dict()
    timer = state.get("timer_epoch")
    assert timer is not None
    assert timer["global_count"] == 3
    assert state["loader"] is not None
    assert state["loader"]["next_batch"] == 1


def test_loading_state_before_the_loader_exists_applies_it_to_the_next_one(
    tmp_path: Path,
) -> None:
    _ = write_split(tmp_path)
    source = make_data(tmp_path)
    seed(0)
    iterator = iter(source.train_dataloader())
    _ = next(iterator)
    state = source.state_dict()
    target = make_data(tmp_path)
    target.load_state_dict(state)
    assert target.state_dict()["loader"] is state["loader"]
    assert target.train_dataloader().state_dict()["next_batch"] == 1
    loader_state = target.state_dict()["loader"]
    assert loader_state is not None
    assert loader_state["next_batch"] == 1


def test_loading_a_fresh_state_resets_a_live_loader_to_a_new_pass(
    tmp_path: Path,
) -> None:
    _ = write_split(tmp_path)
    fresh = make_data(tmp_path).state_dict()
    target = make_data(tmp_path)
    live = target.train_dataloader()
    seed(0)
    _ = next(iter(live))
    target.load_state_dict(fresh)
    seed(0)
    labels = torch.cat([b["label"] for b in live])
    assert torch.equal(labels.sort().values, torch.arange(COUNT))


def test_loading_state_into_a_live_loader_resumes_it_at_once(tmp_path: Path) -> None:
    _ = write_split(tmp_path)
    source = make_data(tmp_path)
    seed(0)
    iterator = iter(source.train_dataloader())
    _ = next(iterator)
    target = make_data(tmp_path)
    live = target.train_dataloader()
    target.load_state_dict(source.state_dict())
    assert live.state_dict()["next_batch"] == 1
    timer = target.state_dict().get("timer_epoch")
    assert timer is not None
    assert timer["global_count"] == 0


def test_working_dir_resolves_beneath_the_base_dir() -> None:
    config = Cifar100Data.Config()
    config.base_dir = "/opt/scratch"
    assert Path(config.copy_tree().finalize().working_dir) == Path(
        "/opt/scratch/datasets/cifar100",
    )


class _FakeCifar100:
    """Two images in the layout torchvision's dataset exposes."""

    data: ClassVar = (
        np.arange(2 * 32 * 32 * 3, dtype=np.uint32)
        .astype(np.uint8)
        .reshape(2, 32, 32, 3)
    )
    targets: ClassVar = [7, 3]
    calls: ClassVar[list[tuple[str, bool, bool]]] = []

    def __init__(self, root: str, *, train: bool, download: bool) -> None:
        self.calls.append((root, train, download))


def test_prepare_writes_eight_bit_channel_first_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(datasets, "CIFAR100", _FakeCifar100)
    _FakeCifar100.calls.clear()
    prepare(tmp_path / "nested")
    assert _FakeCifar100.calls == [
        (str(tmp_path / "nested"), True, True),
        (str(tmp_path / "nested"), False, True),
    ]
    for split in ("train", "test"):
        stored = cast(
            dict[str, Tensor],
            torch.load(tmp_path / "nested" / f"{split}.pt", weights_only=True),
        )
        assert stored["media"].dtype == torch.uint8
        assert stored["media"].shape == (2, 3, 32, 32)
        assert torch.equal(
            stored["media"],
            torch.as_tensor(_FakeCifar100.data).permute(0, 3, 1, 2),
        )
        assert stored["label"].tolist() == [7, 3]
    assert not list((tmp_path / "nested").glob("*.partial"))


def test_prepare_leaves_a_prepared_split_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(datasets, "CIFAR100", _FakeCifar100)
    _FakeCifar100.calls.clear()
    (tmp_path / "train.pt").write_bytes(b"kept")
    with caplog.at_level(logging.INFO):
        prepare(tmp_path)
    assert (tmp_path / "train.pt").read_bytes() == b"kept"
    assert _FakeCifar100.calls == [(str(tmp_path), False, True)]
    assert "train already prepared" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_streams", 0), ("batch_size", 0), ("eval_batch_size", 0)],
)
def test_a_batch_or_stream_count_below_one_is_refused(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    _ = write_split(tmp_path)
    config = Cifar100Data.Config()
    config.base_dir = None
    config.working_dir = tmp_path
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.make()


def test_prepare_fetches_cifar10_when_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(datasets, "CIFAR10", _FakeCifar100)
    prepare(tmp_path, dataset="cifar10")
    assert (tmp_path / "train.pt").exists()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
