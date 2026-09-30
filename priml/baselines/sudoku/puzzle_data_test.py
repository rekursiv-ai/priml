"""Tests for puzzle dataset file and iterator validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

import numpy as np
import pytest

from priml.baselines.sudoku.puzzle_data import (
    PuzzleDataset,
    _get_device,
    _PuzzleBatchIterator,
    load_puzzle_dataset,
)
from priml.baselines.sudoku.puzzle_spec import SudokuSpec


if TYPE_CHECKING:
    from pathlib import Path


def _write(
    root: Path,
    *,
    metadata: bool = True,
    groups: bool = True,
    split_name: str = "train",
) -> None:
    split = root / split_name
    split.mkdir(parents=True, exist_ok=True)
    if metadata:
        (split / "dataset.json").write_text(
            json.dumps({"vocab_size": 11, "seq_len": 81}),
        )
    if groups:
        np.save(split / "all__group_indices.npy", np.array([0, 2], dtype=np.int32))
    np.save(split / "all__inputs.npy", np.full((2, 81), 1, dtype=np.int32))
    np.save(split / "all__labels.npy", np.full((2, 81), 1, dtype=np.int32))


def _config(root: Path, **overrides: object) -> PuzzleDataset.Config:
    config = PuzzleDataset.Config(working_dir=root, device="cpu", batch_size=2)
    config.spec = SudokuSpec()
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_loader_rejects_missing_files_and_negative_caps(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset directory"):
        load_puzzle_dataset(tmp_path, "train")
    _write(tmp_path)
    (tmp_path / "train" / "dataset.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_puzzle_dataset(tmp_path, "train")
    _write(tmp_path, metadata=True, groups=False)
    (tmp_path / "train" / "all__group_indices.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Required file"):
        load_puzzle_dataset(tmp_path, "train")
    _write(tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        load_puzzle_dataset(tmp_path, "train", max_samples=-1)


def test_dataset_validates_indices_and_spec_and_subset(tmp_path: Path) -> None:
    _write(tmp_path)
    with pytest.raises(ValueError, match="strictly ascending"):
        _config(tmp_path, eval_instance_indices=(0, 0)).make()
    with pytest.raises(ValueError, match="non-negative"):
        _config(tmp_path, eval_instance_indices=(-1,)).make()
    with pytest.raises(ValueError, match="epoch_offset"):
        _config(tmp_path).make().train_dataloader().__class__(
            tmp_path,
            "cpu",
            2,
            epoch_offset=-1,
            spec=SudokuSpec(),
        )
    wrong = _config(tmp_path)
    wrong.spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=6)
    with pytest.raises(ValueError, match="vocabulary"):
        wrong.make().train_dataloader()
    wrong.spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=11)
    with pytest.raises(ValueError, match="grid"):
        wrong.make().train_dataloader()
    assert _get_device("auto").type in {"cpu", "cuda", "mps"}
    iterator = _PuzzleBatchIterator(tmp_path, "cpu", 3, spec=SudokuSpec())
    assert next(iter(iterator))["valid_count"] == 2


def test_dataset_iterator_edges(tmp_path: Path) -> None:
    _write(tmp_path)
    loaded = load_puzzle_dataset(tmp_path, "train", max_samples=1)
    assert loaded["inputs"].shape[0] == 1
    dataset = _config(tmp_path, num_instances=1).make()
    loader = dataset.train_dataloader()
    assert len(loader) == 1
    assert next(iter(loader))["media"].shape[0] == 2
    dataset.train_dataloader()
    dataset.load_state_dict({"train_epochs": 2})
    assert dataset.state_dict().get("train_epochs") == 2
    assert dataset.eval_batch_size == 2
    assert dataset.config.device == "cpu"


def test_dataset_subset_bounds_and_state_load(tmp_path: Path) -> None:
    _write(tmp_path)
    _write(tmp_path, split_name="test")
    dataset = _config(tmp_path, eval_instance_indices=(0,)).make()
    assert next(iter(dataset.eval_dataloader()))["valid_count"] == 2
    with pytest.raises(ValueError, match="outside"):
        _config(tmp_path, eval_instance_indices=(1, 2)).make().eval_dataloader()
    dataset.load_state_dict({"train_epochs": 3})
    assert dataset.state_dict().get("train_epochs") == 3


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
