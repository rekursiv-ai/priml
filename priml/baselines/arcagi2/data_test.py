"""ARC2 sampling and padding, checked against source-minted records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import json

from torch import Tensor

import numpy as np
import pytest
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
    """Two tasks, four rows, sharing one tree between both splits."""
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir()
        (directory / "dataset.json").write_text(
            json.dumps({"ignore_label_id": 0, "blank_identifier_id": 0}),
        )
        arrays = {
            "inputs": np.arange(36, dtype=np.int32).reshape(4, 9) % 12,
            "labels": np.arange(36, dtype=np.int32).reshape(4, 9) % 12,
            "puzzle_indices": np.array([0, 2, 4], dtype=np.int64),
            "group_indices": np.array([0, 1, 2], dtype=np.int64),
            "puzzle_identifiers": np.array([1, 2], dtype=np.int32),
        }
        for name, array in arrays.items():
            np.save(directory / f"all__{name}.npy", array)


def record_passes(build: Callable[[], Loader]) -> dict[str, Tensor]:
    """Record two consecutive training passes of one loader."""
    loader = build().train_dataloader()
    out: dict[str, object] = {}
    for epoch in range(2):
        for index, batch in enumerate(loader):
            for key in ("media", "label", "puzzle_identifiers"):
                value = batch[key]
                assert isinstance(value, Tensor)
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
    assert len(actual_loader) == 2
    iterator = iter(actual_loader)
    next(iterator)
    resumed = port_data(tmp_path)
    resumed.load_state_dict(port.state_dict())
    rest = list(iterator)
    replay = list(resumed.train_dataloader())
    assert len(rest) == len(replay)
    for expected, actual in zip(rest, replay, strict=True):
        left, right = expected["media"], actual["media"]
        assert isinstance(left, Tensor)
        assert isinstance(right, Tensor)
        assert torch.equal(left, right), "mid-pass resume"


def test_arc2_requires_resident_data_and_checkpoint_fields(tmp_path: Path) -> None:
    write_tree(tmp_path)
    config = Arc2Data.Config(working_dir=tmp_path, batch_size=2, device="cpu")
    data = config.make()
    state = {key: value for key, value in data.state_dict().items() if key != "passes"}
    with pytest.raises(KeyError):
        data.load_state_dict(state)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
