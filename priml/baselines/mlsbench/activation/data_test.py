"""Tests for the CIFAR-10 preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import logging

from torchvision import datasets

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.activation.data import prepare


if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


class _FakeCifar10:
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
    monkeypatch.setattr(datasets, "CIFAR10", _FakeCifar10)
    _FakeCifar10.calls.clear()
    prepare(tmp_path / "nested")
    assert _FakeCifar10.calls == [
        (str(tmp_path / "nested"), True, True),
        (str(tmp_path / "nested"), False, True),
    ]
    for split in ("train", "test"):
        stored = cast(
            "dict[str, Tensor]",
            torch.load(tmp_path / "nested" / f"{split}.pt", weights_only=True),
        )
        assert stored["media"].dtype == torch.uint8
        assert torch.equal(
            stored["media"],
            torch.as_tensor(_FakeCifar10.data).permute(0, 3, 1, 2),
        )
        assert stored["label"].tolist() == [7, 3]
    assert not list((tmp_path / "nested").glob("*.partial"))


def test_prepare_leaves_a_prepared_split_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(datasets, "CIFAR10", _FakeCifar10)
    _FakeCifar10.calls.clear()
    (tmp_path / "train.pt").write_bytes(b"kept")
    with caplog.at_level(logging.INFO):
        prepare(tmp_path)
    assert (tmp_path / "train.pt").read_bytes() == b"kept"
    assert _FakeCifar10.calls == [(str(tmp_path), False, True)]
    assert "cifar10: train already prepared" in caplog.text


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
