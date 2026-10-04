"""Tests for CIFAR-10 loading and preparation."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import Mock, call

import logging

from torch import Tensor
from torchvision import datasets

import numpy as np
import pytest
import torch

from priml.baselines.cifar10.data import Cifar10Data, _load_split, prepare
from priml.lib.custom_json import ListCodec
from priml.math.pixel import rgb2float


def tiny_dataset(directory: Path, *, count: int = 8) -> Cifar10Data.Config:
    """Write a miniature prepared dataset and return a config reading it."""
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    for split in ("train", "test"):
        torch.save(
            {
                "media": torch.randn(count, 3, 4, 5, generator=generator),
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
    assert batch["media"].shape == (3, 3, 4, 5)
    assert batch["label"].shape == (3,)


def test_media_uses_configured_dtype_and_channels_last(tmp_path: Path) -> None:
    config = tiny_dataset(tmp_path)
    config.dtype = torch.float64
    data = config.make()

    assert data.train_media.dtype == torch.float64
    assert data.eval_media.dtype == torch.float64
    assert data.train_media.is_contiguous(memory_format=torch.channels_last)
    assert data.train_label.dtype == torch.int64
    assert data.train_media.device.type == "cpu"


def test_split_loading_is_cpu_mapped_and_weights_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", load)
    tiny_dataset(tmp_path).make()

    assert load.call_args_list == [
        call(tmp_path / "train.pt", map_location="cpu", weights_only=True),
        call(tmp_path / "test.pt", map_location="cpu", weights_only=True),
    ]


def test_loaded_splits_follow_configured_device(tmp_path: Path) -> None:
    config = tiny_dataset(tmp_path)
    config.device = "meta"
    data = config.make()

    assert data.train_media.device == torch.device("meta")
    assert data.train_label.device == torch.device("meta")
    assert data.eval_media.device == torch.device("meta")
    assert data.eval_label.device == torch.device("meta")


def test_order_and_restored_indices_use_the_media_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tiny_dataset(tmp_path).make()
    arange = Mock(wraps=torch.arange)
    randperm = Mock(wraps=torch.randperm)
    monkeypatch.setattr(torch, "arange", arange)
    monkeypatch.setattr(torch, "randperm", randperm)

    list(data.eval_dataloader())
    list(data.train_dataloader())

    assert call(8, device=data.eval_media.device) in arange.call_args_list
    assert call(8, device=data.train_media.device) in randperm.call_args_list

    order = torch.arange(8)
    moved = Mock(wraps=order.to)
    monkeypatch.setattr(order, "to", moved)
    data.train_dataloader().load_state_dict({"order": order, "next_batch": 0})
    assert moved.call_args_list == [call(data.train_media.device)]


def test_mps_skips_channels_last_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "split.pt"
    path.touch()
    media = Mock()
    converted_media = Mock()
    media.to.return_value = converted_media
    label = Mock()
    monkeypatch.setattr(
        torch,
        "load",
        Mock(return_value={"media": media, "label": label}),
    )

    _load_split(
        path,
        device=cast(torch.device, Mock(type="mps")),
        dtype=torch.float32,
    )

    converted_media.contiguous.assert_not_called()


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


def test_eval_loader_preserves_dataset_order_and_final_short_batch(
    tmp_path: Path,
) -> None:
    config = tiny_dataset(tmp_path)
    config.eval_batch_size = 3
    data = config.make()
    loader = data.eval_dataloader()
    batches = list(loader)
    labels = torch.cat([batch["label"] for batch in batches])

    assert loader.shuffle is False
    assert loader.drop_last is False
    assert [len(batch["label"]) for batch in batches] == [3, 3, 2]
    assert torch.equal(labels, data.eval_label)


def test_drop_last_discards_the_short_batch(tmp_path: Path) -> None:
    config = tiny_dataset(tmp_path, count=8)
    config.drop_last = True
    data = config.make()
    loader = data.train_dataloader()
    batches = list(loader)
    assert len(loader) == 2
    assert len(batches) == 2
    assert [len(batch["label"]) for batch in batches] == [3, 3]
    assert loader.state_dict() == {"order": None, "next_batch": 0}


def test_short_batch_is_yielded_by_default(tmp_path: Path) -> None:
    data = tiny_dataset(tmp_path, count=8).make()
    loader = data.train_dataloader()
    sizes = [len(batch["label"]) for batch in loader]
    assert sizes == [3, 3, 2]
    assert loader.state_dict() == {"order": None, "next_batch": 0}


def test_missing_split_names_the_preparer(tmp_path: Path) -> None:
    config = Cifar10Data.Config()
    config.base_dir = None
    config.working_dir = tmp_path
    config.device = "cpu"
    with pytest.raises(FileNotFoundError) as error:
        _ = config.make()
    assert str(error.value) == (
        f"Prepared CIFAR-10 split not found at {tmp_path / 'train.pt'}. "
        "Run `uv --quiet run --frozen python -m "
        "priml.baselines.cifar10.scripts.prepare_data` first."
    )


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
    with pytest.raises(ValueError, match="not a prepared CIFAR-10 split") as error:
        _ = config.make()
    assert str(error.value) == (
        f"{tmp_path / 'train.pt'} is not a prepared CIFAR-10 split: "
        "missing ['label', 'media']. Delete it and re-run `uv --quiet run "
        "--frozen python -m priml.baselines.cifar10.scripts.prepare_data`, "
        "or pass `--override dataset.working_dir=/datasets/...` to read a "
        "different directory (the path is logical, resolved beneath `base_dir`)."
    )


def test_rejects_nonpositive_batch_sizes(tmp_path: Path) -> None:
    config = tiny_dataset(tmp_path)
    config.batch_size = 0
    with pytest.raises(
        ValueError,
        match="batch_size and eval_batch_size must be positive; got 0 and 4\\.",
    ):
        _ = config.make()

    config.batch_size = 3
    config.eval_batch_size = 0
    with pytest.raises(
        ValueError,
        match="batch_size and eval_batch_size must be positive; got 3 and 0\\.",
    ):
        _ = config.make()


def test_single_image_batches_are_valid(tmp_path: Path) -> None:
    config = tiny_dataset(tmp_path)
    config.batch_size = 1
    config.eval_batch_size = 1
    data = config.make()

    assert len(data.train_dataloader()) == 8
    assert len(data.eval_dataloader()) == 8


def test_working_dir_resolves_beneath_base_dir() -> None:
    config = Cifar10Data.Config()
    config.base_dir = "/opt/scratch"
    resolved = config.copy_tree().finalize()
    assert Path(resolved.working_dir) == Path("/opt/scratch/datasets/cifar10")


def test_state_dict_carries_the_pass_count_and_idle_loader_state(
    tmp_path: Path,
) -> None:
    data = tiny_dataset(tmp_path).make()
    data.timer_epoch.global_count = 3
    assert data.state_dict()["loader"] is None
    loader = data.train_dataloader()
    assert loader.state_dict() == {"order": None, "next_batch": 0}

    restored = tiny_dataset(tmp_path).make()
    restored.load_state_dict(data.state_dict())
    assert restored.timer_epoch.global_count == 3
    assert len(list(restored.train_dataloader())) == 3
    assert len(list(restored.train_dataloader())) == 3


def test_live_loader_restores_its_cursor_and_clears_pending_state(
    tmp_path: Path,
) -> None:
    data = tiny_dataset(tmp_path, count=9).make()
    loader = data.train_dataloader()
    iterator = iter(loader)
    first = next(iterator)
    saved = loader.state_dict()
    expected = list(iterator)

    data.load_state_dict({"loader": saved})
    observed = list(loader)

    assert saved["next_batch"] == 1
    assert len(expected) == len(observed) == 2
    assert all(
        torch.equal(actual[key], wanted[key])
        for actual, wanted in zip(observed, expected, strict=True)
        for key in ("media", "label")
    )
    assert not torch.equal(first["label"], expected[0]["label"])
    assert len(list(data.train_dataloader())) == 3


def test_iterator_restore_defaults_to_the_first_batch(tmp_path: Path) -> None:
    data = tiny_dataset(tmp_path).make()
    loader = data.eval_dataloader()
    loader.load_state_dict({"order": torch.arange(len(data.eval_media))})

    batch = next(iter(loader))

    assert torch.equal(batch["media"], data.eval_media[: loader.batch_size])
    assert torch.equal(batch["label"], data.eval_label[: loader.batch_size])


def test_checkpoint_resumes_the_unfinished_permutation(tmp_path: Path) -> None:
    torch.manual_seed(3)
    data = tiny_dataset(tmp_path, count=9).make()
    loader = data.train_dataloader()
    iterator = iter(loader)
    consumed = next(iterator)
    state = data.state_dict()
    expected = list(iterator)

    restored = tiny_dataset(tmp_path, count=9).make()
    restored.load_state_dict(state)
    resumed_loader = restored.train_dataloader()
    observed = list(resumed_loader)

    assert state["loader"] is not None
    assert state["loader"]["next_batch"] == 1
    assert state["loader"]["order"] is not None
    assert len(expected) == 2
    assert len(observed) == len(expected)
    assert all(
        torch.equal(actual[key], wanted[key])
        for actual, wanted in zip(observed, expected, strict=True)
        for key in ("media", "label")
    )
    assert not torch.equal(consumed["label"], expected[0]["label"])


def test_prepare_normalizes_and_writes_both_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The download path writes the schema the loader expects.

    ``torchvision.datasets.CIFAR10`` is stubbed: the test covers OUR
    conversion -- channel order, scaling, normalization, and the atomic
    rename -- not torchvision's downloader.
    """
    caplog.set_level(logging.INFO, logger="priml.baselines.cifar10.data")
    destination = tmp_path / "nested" / "cifar10"
    dataset = Mock(side_effect=_StubCifar10)
    conversions: list[tuple[Tensor, bool, bool]] = []

    def record_conversion(
        x: Tensor,
        *,
        float_dtype: torch.dtype | None = None,
        inplace: bool = False,
        unit_interval: bool = False,
    ) -> Tensor:
        conversions.append((x, inplace, unit_interval))
        return rgb2float(
            x,
            float_dtype=float_dtype,
            inplace=inplace,
            unit_interval=unit_interval,
        )

    save = Mock(wraps=torch.save)
    monkeypatch.setattr(datasets, "CIFAR10", dataset)
    monkeypatch.setattr(
        "priml.baselines.cifar10.data.rgb2float",
        record_conversion,
    )
    monkeypatch.setattr(torch, "save", save)
    prepare(destination, mean=(0.1, 0.2, 0.3), std=(0.5, 0.25, 0.1))

    assert dataset.call_args_list == [
        call(str(destination), train=True, download=True),
        call(str(destination), train=False, download=True),
    ]
    assert len(conversions) == 2
    assert [(inplace, unit_interval) for _, inplace, unit_interval in conversions] == [
        (True, True),
        (True, True),
    ]
    for media, _, _ in conversions:
        assert media.dtype == torch.float32
        assert media.shape == (2, 3, 4, 5)
    assert [entry.args[1] for entry in save.call_args_list] == [
        destination / "train.pt.partial",
        destination / "test.pt.partial",
    ]
    for split, expected in (
        ("train", [1.8, (128 / 255 - 0.2) / 0.25, -3.0]),
        ("test", [-0.2, (128 / 255 - 0.2) / 0.25, 7.0]),
    ):
        payload = cast(
            dict[str, Tensor],
            torch.load(
                destination / f"{split}.pt",
                weights_only=True,
            ),
        )
        # Distinct channel values and statistics expose wrong broadcast axes.
        assert payload["media"].shape == (2, 3, 4, 5)
        assert payload["media"].dtype == torch.float32
        assert payload["label"].dtype == torch.int64
        values = ListCodec.coerce(payload["media"][0, :, 0, 0].tolist(), float)
        labels = ListCodec.coerce(payload["label"].tolist(), int)
        assert values == pytest.approx(expected)
        assert labels == [0, 1]
        assert (destination / f"{split}.pt").read_bytes()
    # The staging file is renamed, never left behind for the existence check
    # above to later mistake for a complete split.
    assert not list(destination.glob("*.partial"))
    assert [record.getMessage() for record in caplog.records] == [
        f"cifar10: wrote 2 {split} images to {destination / f'{split}.pt'}"
        for split in ("train", "test")
    ]


class _StubCifar10:
    """Stands in for torchvision's CIFAR-10, without the download."""

    def __init__(self, root: str, *, train: bool, download: bool) -> None:
        del root, download
        channels = (255, 128, 0) if train else (0, 128, 255)
        self.data = np.empty((2, 4, 5, 3), dtype=np.uint8)
        self.data[...] = channels
        self.targets = [0, 1]


def test_prepare_leaves_existing_splits_untouched(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tiny_dataset(tmp_path)
    before = {
        split: (tmp_path / f"{split}.pt").read_bytes() for split in ("train", "test")
    }
    caplog.set_level(logging.INFO, logger="priml.baselines.cifar10.data")

    prepare(tmp_path)

    assert {
        split: (tmp_path / f"{split}.pt").read_bytes() for split in ("train", "test")
    } == before
    assert [record.getMessage() for record in caplog.records] == [
        f"cifar10: {split} already prepared at {tmp_path / f'{split}.pt'}"
        for split in ("train", "test")
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
