"""ARC2 sampling and padding, checked against source-minted digests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import json

import numpy as np
import torch

from priml.baselines.arcagi2.data import Arc2Data
from priml.baselines.arcagi2.record_test import assert_matches, reduce


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path


class Loader(Protocol):
    """The dataset surface the recorder drives."""

    def train_dataloader(self) -> Iterable[Mapping[str, object]]:
        """Return a training pass."""
        ...


def write_tree(root: Path) -> None:
    """Four tasks, eight rows, sharing one tree between both splits."""
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir()
        (directory / "dataset.json").write_text(
            json.dumps({"ignore_label_id": 0, "blank_identifier_id": 0}),
        )
        arrays = {
            "inputs": np.arange(72, dtype=np.int32).reshape(8, 9) % 12,
            "labels": np.arange(72, dtype=np.int32).reshape(8, 9) % 12,
            "puzzle_indices": np.array([0, 2, 4, 6, 8], dtype=np.int64),
            "group_indices": np.array([0, 1, 3, 4], dtype=np.int64),
            "puzzle_identifiers": np.array([1, 2, 3, 4], dtype=np.int32),
        }
        for name, array in arrays.items():
            np.save(directory / f"all__{name}.npy", array)


def record_passes(build: Callable[[], Loader]) -> dict[str, torch.Tensor]:
    """Record three consecutive training passes of one loader."""
    loader = build().train_dataloader()
    out: dict[str, object] = {}
    for epoch in range(3):
        for index, batch in enumerate(loader):
            for key in ("media", "label", "puzzle_identifiers"):
                value = batch[key]
                assert isinstance(value, torch.Tensor)
                out[f"{epoch}/{index}/{key}"] = value
    return reduce(out)


def port_data(root: Path) -> Arc2Data:
    """Build the exported loader over ``root``."""
    return Arc2Data.Config(
        working_dir=root,
        batch_size=2,
        device="cpu",
        epochs_per_iter=1,
    ).make()


def test_reference_batches(tmp_path: Path) -> None:
    """Sampled task order and batches match; a mid-pass resume continues."""
    write_tree(tmp_path)
    assert_matches("data", "passes", record_passes(lambda: port_data(tmp_path)))
    port = port_data(tmp_path)
    actual_loader = port.train_dataloader()
    assert len(actual_loader) == 3
    iterator = iter(actual_loader)
    next(iterator)
    resumed = port_data(tmp_path)
    resumed.load_state_dict(port.state_dict())
    rest = list(iterator)
    replay = list(resumed.train_dataloader())
    assert len(rest) == len(replay)
    for expected, actual in zip(rest, replay, strict=True):
        left, right = expected["media"], actual["media"]
        assert isinstance(left, torch.Tensor)
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right), "mid-pass resume"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
