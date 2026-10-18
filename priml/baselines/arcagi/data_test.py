"""Tests for ARC data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast
from unittest.mock import Mock

import json
import logging
import math

from pyarrow import parquet
from torch import Tensor
from treekle import ReadError, from_plain

import numpy as np
import pyarrow as pa
import pytest
import torch

from priml.baselines.arcagi import data as arc_data
from priml.baselines.arcagi.data import (
    ArcInstanceMetadata,
    ArcParquetData,
    PairSource,
    ParquetSourceData,
    PuzzleBatches,
    PuzzleData,
    _load_int32,
    iter_parquet_rows,
    load_puzzle_dataset,
    parse_offline_aug,
    read_parquet_source,
    resolve_rank,
)
from priml.data.arrays import load_array
from priml.train.custom_types import TrainStepProtocol


TASKS: Final = 4
PUZZLES_PER_TASK: Final = 2
VIEWS_PER_PUZZLE: Final = 3
GRID: Final = 12


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a tiny dataset in the three-level on-disk layout."""
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        puzzles = TASKS * PUZZLES_PER_TASK
        rows = puzzles * VIEWS_PER_PUZZLE
        # Each row is a distinct constant grid, so a row is identifiable.
        inputs = np.tile(
            np.arange(rows, dtype=np.int32).reshape(rows, 1) + 2,
            (1, GRID),
        )
        np.save(directory / "all__inputs.npy", inputs)
        np.save(directory / "all__labels.npy", inputs)
        np.save(
            directory / "all__puzzle_indices.npy",
            np.arange(puzzles + 1, dtype=np.int32) * VIEWS_PER_PUZZLE,
        )
        np.save(
            directory / "all__group_indices.npy",
            np.arange(4 + 1, dtype=np.int32) * 2,
        )
        np.save(
            directory / "all__puzzle_identifiers.npy",
            np.arange(puzzles, dtype=np.int32),
        )
        (directory / "dataset.json").write_text(
            json.dumps(
                {
                    "vocab_size": 32,
                    "seq_len": GRID,
                    "ignore_label_id": 0,
                    "blank_identifier_id": 0,
                },
            ),
        )
    return tmp_path


def _write_small_puzzle_split(root: Path, name: str = "train") -> None:
    split = root / name
    split.mkdir(parents=True)
    (split / "dataset.json").write_text(
        json.dumps({"ignore_label_id": 0, "blank_identifier_id": 9}),
    )
    np.save(
        split / "all__inputs.npy",
        np.arange(48, dtype=np.int16).reshape(12, 4) + 2,
    )
    np.save(
        split / "all__labels.npy",
        np.arange(48, dtype=np.int16).reshape(12, 4) + 2,
    )
    np.save(split / "all__puzzle_indices.npy", np.array([0, 2, 5, 9, 12]))
    np.save(split / "all__group_indices.npy", np.array([0, 2, 3, 4]))
    np.save(split / "all__puzzle_identifiers.npy", np.array([1, 2, 3, 4]))


def _small_puzzle_data(root: Path, **options: object) -> PuzzleData:
    config = PuzzleData.Config(working_dir=root, device="cpu", batch_size=2)
    for name, value in options.items():
        setattr(config, name, value)
    return config.make()


def test_puzzle_data_eval_caps_and_batches(dataset_dir: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    config.max_samples = 5
    data = config.make()

    batches = list(data.eval_dataloader())

    assert len(batches) == 2
    assert [batch["valid_count"] for batch in batches] == [4, 1]
    assert torch.equal(
        _tensor(batches[1]["puzzle_identifiers"]),
        torch.tensor([1, 0, 0, 0]),
    )
    assert torch.equal(
        _tensor(batches[1]["spatial_tags"]),
        torch.tensor([[1, 0, 0]] * 4),
    )
    assert _tensor(batches[1]["label"])[1:].eq(-100).all()


def test_puzzle_batch_collate_masks_and_pads_exactly(dataset_dir: Path) -> None:
    train = dataset_dir / "train"
    labels = load_array(train / "all__labels.npy", np.generic)
    labels[0, :2] = [0, 3]
    np.save(train / "all__labels.npy", labels)
    batches = PuzzleBatches(
        dataset_dir=dataset_dir,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=True,
        seed=0,
    )
    assert batches.n_examples == TASKS * PUZZLES_PER_TASK * VIEWS_PER_PUZZLE
    assert batches.n_puzzles == TASKS * PUZZLES_PER_TASK
    assert batches.n_groups == TASKS
    assert batches.iters == 0

    batch = batches._collate(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        valid=1,
    )

    assert batch["valid_count"] == 1
    assert batch["media"].dtype == torch.int32
    assert batch["media"].tolist() == [[2] * GRID, [0] * GRID]
    assert batch["label"][0].tolist() == [-100, 3, *([2] * (GRID - 2))]
    assert batch["label"][1].tolist() == [-100] * GRID
    assert batch["puzzle_identifiers"].tolist() == [0, 0]
    assert batch["spatial_tags"].tolist() == [[1, 0, 0], [1, 0, 0]]


def test_puzzle_batch_sampler_packs_without_replacement(dataset_dir: Path) -> None:
    batches = PuzzleBatches(
        dataset_dir=dataset_dir,
        device="cpu",
        batch_size=4,
        rank=0,
        num_replicas=1,
        train=True,
        seed=7,
    )
    rng = np.random.Generator(np.random.Philox(seed=3))

    next_group, rows, puzzle_ids = batches._sample_batch(
        rng,
        np.arange(4, dtype=np.int64),
        0,
    )

    assert next_group == 2
    assert rows.shape == puzzle_ids.shape == (4,)
    assert len(np.unique(rows)) == 4
    assert len(np.unique(puzzle_ids)) == 2
    assert puzzle_ids.tolist() == [1, 1, 1, 2]
    assert sorted(cast(list[int], rows[:3].tolist())) == [3, 4, 5]
    assert rows[3] in {6, 7, 8}


def test_puzzle_data_load_state_updates_live_pass_count(dataset_dir: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    config.epochs_per_iter = 2
    data = config.make()
    loader = data.train_dataloader()
    next(iter(loader))

    data.load_state_dict({"train_iters": 4})

    assert loader.iters == 4
    state = data.state_dict()
    assert "train_iters" in state
    assert state["train_iters"] == 4


def test_puzzle_data_train_pass_state_round_trips(dataset_dir: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 2
    config.epochs_per_iter = 2
    source = config.make()
    source_loader = source.train_dataloader()
    list(source_loader)
    state = source.state_dict()

    restored = config.make()
    restored.load_state_dict(state)
    expected = list(source.train_dataloader())
    observed = list(restored.train_dataloader())

    assert len(expected) == len(observed)
    assert all(
        torch.equal(_tensor(left["media"]), _tensor(right["media"]))
        for left, right in zip(expected, observed, strict=True)
    )


def test_load_puzzle_dataset_keeps_one_puzzle_groups(dataset_dir: Path) -> None:
    np.save(
        dataset_dir / "train" / "all__group_indices.npy",
        np.arange(9, dtype=np.int32),
    )

    data = load_puzzle_dataset(dataset_dir, "train")

    assert data["group_indices"].tolist() == list(range(9))


def test_load_puzzle_dataset_caps_beyond_end_at_last_row(dataset_dir: Path) -> None:
    data = load_puzzle_dataset(dataset_dir, "train", max_samples=100)

    assert data["inputs"].shape == data["labels"].shape == (24, GRID)
    assert data["puzzle_indices"].tolist() == list(range(0, 25, 3))
    assert data["group_indices"].tolist() == [0, 2, 4, 6, 8]
    assert len(data["puzzle_identifiers"]) == len(data["spatial_tags"]) == 8


def test_load_puzzle_dataset_caps_all_puzzle_and_group_tables(
    dataset_dir: Path,
) -> None:
    data = load_puzzle_dataset(dataset_dir, "train", max_samples=5)

    assert data["inputs"].shape == data["labels"].shape == (5, GRID)
    assert data["puzzle_indices"].tolist() == [0, 3, 5]
    assert data["group_indices"].tolist() == [0, 2]
    assert data["puzzle_identifiers"].tolist() == [0, 1]
    assert data["spatial_tags"].tolist() == [[1, 0, 0], [1, 0, 0]]
    assert data["metadata"]["blank_identifier_id"] == 0


def test_load_puzzle_dataset_preserves_spatial_sidecar(
    dataset_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    expected = np.tile(
        np.array([2, 3, 4], dtype=np.int32),
        (TASKS * PUZZLES_PER_TASK, 1),
    )
    np.save(dataset_dir / "train" / "all__spatial_tags.npy", expected)

    data = load_puzzle_dataset(dataset_dir, "train")

    assert isinstance(data["inputs"], np.memmap)
    assert isinstance(data["labels"], np.memmap)
    assert data["spatial_tags"].dtype == np.int32
    assert np.array_equal(data["spatial_tags"], expected)
    assert [record.getMessage() for record in caplog.records] == [
        f"loading ARC dataset split 'train' from {dataset_dir / 'train'}",
    ]


def test_load_puzzle_dataset_rejects_empty_single_group(dataset_dir: Path) -> None:
    np.save(
        dataset_dir / "train" / "all__group_indices.npy",
        np.array([0, 0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match=r"empty group \(zero puzzles\)"):
        load_puzzle_dataset(dataset_dir, "train")


def test_sample_cap_preserves_empty_group_at_its_boundary(dataset_dir: Path) -> None:
    np.save(
        dataset_dir / "train" / "all__group_indices.npy",
        np.array([0, 2, 2, 4, 6], dtype=np.int32),
    )

    with pytest.raises(ValueError, match=r"empty group \(zero puzzles\)"):
        load_puzzle_dataset(dataset_dir, "train", max_samples=6)


def test_load_puzzle_dataset_rejects_bad_spatial_tags(dataset_dir: Path) -> None:
    tags_path = dataset_dir / "train" / "all__spatial_tags.npy"
    # A one-row sidecar intentionally models a partial dataset build.
    np.save(tags_path, np.zeros((2, 3), dtype=np.int32))

    with pytest.raises(ValueError, match=r"all__spatial_tags\.npy") as error:
        load_puzzle_dataset(dataset_dir, "train")

    assert str(error.value) == (
        f"all__spatial_tags.npy has 2 rows but the split has 8 puzzles at "
        f"{dataset_dir / 'train'}; the build is partial or corrupt -- re-run "
        "data ensure."
    )


def test_puzzle_dataset_missing_paths_report_exact_locations(tmp_path: Path) -> None:
    split = tmp_path / "train"

    with pytest.raises(FileNotFoundError) as missing_directory:
        load_puzzle_dataset(tmp_path, "train")
    assert str(missing_directory.value) == f"Dataset directory not found: {split}"

    split.mkdir()
    with pytest.raises(FileNotFoundError) as missing_metadata:
        load_puzzle_dataset(tmp_path, "train")
    assert str(missing_metadata.value) == (
        f"Dataset metadata not found: {split / 'dataset.json'}"
    )


def test_puzzle_dataset_metadata_must_be_an_object(dataset_dir: Path) -> None:
    (dataset_dir / "train" / "dataset.json").write_text("[1, 2]")
    with pytest.raises(ReadError):
        load_puzzle_dataset(dataset_dir, "train")


def test_puzzle_data_rank_past_eval_tail_emits_empty_batch(dataset_dir: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 5
    config.rank = 1
    config.num_replicas = 2
    config.max_samples = 12
    batches = list(config.make().eval_dataloader())

    assert [batch["valid_count"] for batch in batches] == [5, 0]
    assert _tensor(batches[-1]["puzzle_identifiers"]).tolist() == [0] * 5
    assert _tensor(batches[-1]["spatial_tags"]).tolist() == [[1, 0, 0]] * 5


def test_resolve_rank_uses_default_without_distributed_runtime() -> None:
    assert resolve_rank(-1, -1) == (0, 1)
    assert resolve_rank(0, 1) == (0, 1)
    assert resolve_rank(1, 3) == (1, 3)
    for rank, replicas in ((0, -1), (-1, 2), (1, 1)):
        with pytest.raises(ValueError, match="rank/num_replicas") as error:
            resolve_rank(rank, replicas)
        assert str(error.value) == (
            "rank/num_replicas must both be the -1 auto-sentinel or satisfy "
            f"0 <= rank < num_replicas; got rank={rank}, num_replicas={replicas}."
        )


def test_int32_loader_preserves_mmap_request(tmp_path: Path) -> None:
    path = tmp_path / "values.npy"
    np.save(path, np.array([[2, 3]], dtype=np.int32))

    assert type(_load_int32(path)) is np.ndarray
    assert isinstance(_load_int32(path, mmap=True), np.memmap)


def test_existing_load_puzzle_dataset_caps_at_group_boundary_and_preserves_memory_map(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path)

    full = load_puzzle_dataset(tmp_path, "train")
    capped = load_puzzle_dataset(tmp_path, "train", max_samples=9)

    assert isinstance(full["inputs"], np.memmap)
    assert isinstance(full["labels"], np.memmap)
    assert full["puzzle_indices"].dtype == np.int64
    assert full["group_indices"].dtype == np.int64
    assert full["puzzle_identifiers"].dtype == np.int32
    assert capped["inputs"].shape == capped["labels"].shape == (9, 4)
    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 9]
    assert capped["group_indices"].tolist() == [0, 2, 3]
    assert capped["puzzle_identifiers"].tolist() == [1, 2, 3]
    assert capped["metadata"] == {"ignore_label_id": 0, "blank_identifier_id": 9}


def test_existing_load_puzzle_dataset_partial_group_adds_boundary_without_mutating_file(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path)
    group_path = tmp_path / "train" / "all__group_indices.npy"
    groups_before = load_array(group_path, np.generic).copy()

    capped = load_puzzle_dataset(tmp_path, "train", max_samples=7)

    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 7]
    assert capped["group_indices"].tolist() == [0, 2, 3]
    assert np.array_equal(load_array(group_path, np.generic), groups_before)


def test_existing_load_puzzle_dataset_caps_beyond_rows_without_inventing_boundary(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path)

    capped = load_puzzle_dataset(tmp_path, "train", max_samples=99)

    assert capped["inputs"].shape == (12, 4)
    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 9, 12]
    assert capped["group_indices"].tolist() == [0, 2, 3, 4]


def test_existing_load_puzzle_dataset_rejects_adjacent_empty_group(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path)
    np.save(
        tmp_path / "train" / "all__group_indices.npy",
        np.array([0, 2, 2, 4]),
    )

    with pytest.raises(ValueError, match=r"empty group \(zero puzzles\)"):
        load_puzzle_dataset(tmp_path, "train")


def test_existing_puzzle_batches_len_uses_ceiling_for_eval_and_epoch_multiplier(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    _write_small_puzzle_split(tmp_path, "train")
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=5,
        rank=0,
        num_replicas=1,
        train=False,
        seed=4,
    )

    assert len(loader) == 3
    assert [batch["valid_count"] for batch in loader] == [5, 5, 2]

    training = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=5,
        rank=0,
        num_replicas=1,
        train=True,
        seed=4,
        epochs_per_iter=2,
    )
    assert len(training) == 4


def test_existing_puzzle_data_state_dict_tracks_live_pass_and_timer(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path)
    data = _small_puzzle_data(tmp_path, epochs_per_iter=2)
    loader = data.train_dataloader()
    next(iter(loader))

    state = data.state_dict()
    restored = _small_puzzle_data(tmp_path)
    restored.load_state_dict(state)

    assert "train_iters" in state
    assert state["train_iters"] == 1
    assert "timer_epoch" in state
    assert restored.train_iters == 1
    assert restored.state_dict() == state
    data.load_state_dict({"train_iters": 7})
    assert loader.iters == 7
    state = data.state_dict()
    assert "train_iters" in state
    assert state["train_iters"] == 7


def test_resolve_rank_accepts_only_complete_explicit_pairs() -> None:
    assert resolve_rank(-1, -1) == (0, 1)
    assert resolve_rank(2, 4) == (2, 4)
    for rank, replicas in ((-1, 1), (0, -1), (-2, 3), (3, 3)):
        with pytest.raises(ValueError, match="rank/num_replicas must both be"):
            resolve_rank(rank, replicas)


def _tensor(value: object) -> Tensor:
    assert isinstance(value, Tensor)
    return value


def _write_split(root: Path, name: str = "train") -> None:
    split = root / name
    split.mkdir(parents=True)
    (split / "dataset.json").write_text(
        json.dumps({"ignore_label_id": 0, "blank_identifier_id": 9}),
    )
    np.save(
        split / "all__inputs.npy",
        np.arange(48, dtype=np.int16).reshape(12, 4) + 2,
    )
    np.save(
        split / "all__labels.npy",
        np.arange(48, dtype=np.int16).reshape(12, 4) + 2,
    )
    np.save(split / "all__puzzle_indices.npy", np.array([0, 2, 5, 9, 12]))
    np.save(split / "all__group_indices.npy", np.array([0, 2, 3, 4]))
    np.save(split / "all__puzzle_identifiers.npy", np.array([1, 2, 3, 4]))


def _puzzle_data(root: Path, **options: object) -> PuzzleData:
    config = PuzzleData.Config(working_dir=root, device="cpu", batch_size=2)
    for name, value in options.items():
        setattr(config, name, value)
    return config.make()


def test_load_puzzle_dataset_caps_at_group_boundary_and_preserves_memory_map(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path)
    full = load_puzzle_dataset(tmp_path, "train")
    capped = load_puzzle_dataset(tmp_path, "train", max_samples=9)
    assert isinstance(full["inputs"], np.memmap)
    assert isinstance(full["labels"], np.memmap)
    assert full["puzzle_indices"].dtype == np.int64
    assert full["group_indices"].dtype == np.int64
    assert full["puzzle_identifiers"].dtype == np.int32
    assert capped["inputs"].shape == capped["labels"].shape == (9, 4)
    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 9]
    assert capped["group_indices"].tolist() == [0, 2, 3]
    assert capped["puzzle_identifiers"].tolist() == [1, 2, 3]
    assert capped["metadata"] == {"ignore_label_id": 0, "blank_identifier_id": 9}


def test_load_puzzle_dataset_partial_group_adds_boundary_without_mutating_file(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path)
    group_path = tmp_path / "train" / "all__group_indices.npy"
    groups_before = load_array(group_path, np.generic).copy()
    capped = load_puzzle_dataset(tmp_path, "train", max_samples=7)
    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 7]
    assert capped["group_indices"].tolist() == [0, 2, 3]
    assert np.array_equal(load_array(group_path, np.generic), groups_before)


def test_load_puzzle_dataset_caps_beyond_rows_without_inventing_boundary(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path)
    capped = load_puzzle_dataset(tmp_path, "train", max_samples=99)
    assert capped["inputs"].shape == (12, 4)
    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 9, 12]
    assert capped["group_indices"].tolist() == [0, 2, 3, 4]


def test_load_puzzle_dataset_rejects_adjacent_empty_group(tmp_path: Path) -> None:
    _write_split(tmp_path)
    np.save(
        tmp_path / "train" / "all__group_indices.npy",
        np.array([0, 2, 2, 4]),
    )
    with pytest.raises(ValueError, match=r"empty group \(zero puzzles\)"):
        load_puzzle_dataset(tmp_path, "train")


def test_puzzle_batches_len_uses_ceiling_for_eval_and_epoch_multiplier(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "test")
    _write_split(tmp_path, "train")
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=5,
        rank=0,
        num_replicas=1,
        train=False,
        seed=4,
    )
    assert len(loader) == 3
    assert [batch["valid_count"] for batch in loader] == [5, 5, 2]
    training = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=5,
        rank=0,
        num_replicas=1,
        train=True,
        seed=4,
        epochs_per_iter=2,
    )
    assert len(training) == 4


def test_puzzle_data_eval_dataloader_forwards_loader_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PuzzleData.Config(
        working_dir=tmp_path,
        device="cpu",
        batch_size=2,
        eval_batch_size=3,
        seed=31,
        rank=1,
        num_replicas=2,
        max_samples=7,
    )
    data = PuzzleData(config)
    result = object()
    constructor = Mock(return_value=result)
    monkeypatch.setattr(arc_data, "PuzzleBatches", constructor)

    assert data.eval_dataloader() is result
    constructor.assert_called_once_with(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=1,
        num_replicas=2,
        train=False,
        seed=31,
        max_samples=7,
    )


def test_puzzle_data_state_dict_tracks_live_pass_and_timer(tmp_path: Path) -> None:
    _write_split(tmp_path)
    data = _puzzle_data(tmp_path, epochs_per_iter=2)
    loader = data.train_dataloader()
    next(iter(loader))
    state = data.state_dict()
    restored = _puzzle_data(tmp_path)
    restored.load_state_dict(state)
    assert "train_iters" in state
    assert state["train_iters"] == 1
    assert "timer_epoch" in state
    assert restored.train_iters == 1
    assert restored.state_dict() == state
    data.load_state_dict({"train_iters": 7})
    assert loader.iters == 7
    state = data.state_dict()
    assert "train_iters" in state
    assert state["train_iters"] == 7


def test_g15_resolve_rank_accepts_only_complete_explicit_pairs() -> None:
    assert resolve_rank(-1, -1) == (0, 1)
    assert resolve_rank(2, 4) == (2, 4)
    for rank, replicas in ((-1, 1), (0, -1), (-2, 3), (3, 3)):
        with pytest.raises(ValueError, match="rank/num_replicas must both be"):
            resolve_rank(rank, replicas)


def test_load_puzzle_dataset_preserves_unmodified_array_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_split(tmp_path)
    np.save(
        tmp_path / "train" / "all__spatial_tags.npy",
        np.array([[1, 0, 0]] * 4, dtype=np.int32),
    )
    arrays = {
        "all__inputs.npy": np.arange(48, dtype=np.int32).reshape(12, 4),
        "all__labels.npy": np.arange(48, dtype=np.int32).reshape(12, 4),
        "all__puzzle_indices.npy": np.array([0, 2, 5, 9, 12], dtype=np.int64),
        "all__group_indices.npy": np.array([0, 2, 3, 4], dtype=np.int64),
        "all__puzzle_identifiers.npy": np.array([1, 2, 3, 4], dtype=np.int32),
        "all__spatial_tags.npy": np.array([[1, 0, 0]] * 4, dtype=np.int32),
    }

    def load_array(path: Path, *, mmap: bool = False) -> np.ndarray:
        del mmap
        return arrays[path.name]

    monkeypatch.setattr(arc_data, "_load_int32", load_array)
    loaded = load_puzzle_dataset(tmp_path, "train")
    assert np.shares_memory(
        loaded["puzzle_indices"],
        arrays["all__puzzle_indices.npy"],
    )
    assert np.shares_memory(
        loaded["group_indices"],
        arrays["all__group_indices.npy"],
    )
    assert np.shares_memory(
        loaded["puzzle_identifiers"],
        arrays["all__puzzle_identifiers.npy"],
    )
    assert np.shares_memory(
        loaded["spatial_tags"],
        arrays["all__spatial_tags.npy"],
    )


def test_load_puzzle_dataset_normalizes_index_and_sidecar_dtypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_split(tmp_path)
    arrays = {
        "all__inputs.npy": np.arange(48, dtype=np.int32).reshape(12, 4),
        "all__labels.npy": np.arange(48, dtype=np.int32).reshape(12, 4),
        "all__puzzle_indices.npy": np.array([0, 2, 5, 9, 12], dtype=np.int32),
        "all__group_indices.npy": np.array([0, 2, 3, 4], dtype=np.int32),
        "all__puzzle_identifiers.npy": np.array([1, 2, 3, 4], dtype=np.int64),
        "all__spatial_tags.npy": np.array([[1, 0, 0]] * 4, dtype=np.int64),
    }

    def load_array(path: Path, *, mmap: bool = False) -> np.ndarray:
        del mmap
        return arrays[path.name]

    monkeypatch.setattr(arc_data, "_load_int32", load_array)
    loaded = load_puzzle_dataset(tmp_path, "train")

    assert loaded["puzzle_indices"].dtype == np.int64
    assert loaded["group_indices"].dtype == np.int64
    assert loaded["puzzle_identifiers"].dtype == np.int32
    assert loaded["spatial_tags"].dtype == np.int32


def test_puzzle_batches_default_recipe_and_rank_sharded_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_split(tmp_path, "test")
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
    whole = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=4,
        rank=0,
        num_replicas=1,
        train=False,
        seed=5,
        max_samples=11,
    )
    first_rank = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=2,
        train=False,
        seed=5,
        max_samples=11,
    )
    second_rank = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=1,
        num_replicas=2,
        train=False,
        seed=5,
        max_samples=11,
    )
    assert first_rank.epochs_per_iter == 1
    assert first_rank.device == torch.device("cpu")
    assert first_rank.num_replicas == 2
    assert first_rank.global_batch_size == 4
    full_batches = list(whole)
    left_batches = list(first_rank)
    right_batches = list(second_rank)
    assert len(full_batches) == len(left_batches) == len(right_batches) == 3
    for full, left, right in zip(
        full_batches,
        left_batches,
        right_batches,
        strict=True,
    ):
        valid = full["valid_count"]
        assert valid == left["valid_count"] + right["valid_count"]
        assert torch.equal(
            full["media"][:valid],
            torch.cat(
                [
                    left["media"][: left["valid_count"]],
                    right["media"][: right["valid_count"]],
                ],
            ),
        )


def test_puzzle_batches_iter_train_counts_passes_and_never_reads_past_plan(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path)
    np.save(
        tmp_path / "train" / "all__inputs.npy",
        np.arange(20, dtype=np.int32).reshape(4, 5) + 2,
    )
    np.save(
        tmp_path / "train" / "all__labels.npy",
        np.arange(20, dtype=np.int32).reshape(4, 5) + 2,
    )
    np.save(tmp_path / "train" / "all__puzzle_indices.npy", np.array([0, 2, 4]))
    np.save(tmp_path / "train" / "all__group_indices.npy", np.array([0, 1, 2]))
    np.save(tmp_path / "train" / "all__puzzle_identifiers.npy", np.array([1, 2]))
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=4,
        rank=0,
        num_replicas=1,
        train=True,
        seed=12,
    )
    first = list(loader)
    second = list(loader)
    assert len(first) == len(second) == 1
    assert loader.iters == 2
    assert len({tuple(batch["media"][:, 0].tolist()) for batch in first + second}) == 2


def test_puzzle_batches_to_device_uses_resolved_device_and_nonblocking_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_split(tmp_path, "test")
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="meta",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=0,
    )
    original_to = torch.Tensor.to
    options: list[dict[str, object]] = []

    def tracked_to(
        tensor: Tensor,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> Tensor:
        options.append({"non_blocking": non_blocking})
        return original_to(tensor, device, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    result = loader._to_device(np.array([2, 3], dtype=np.int32))
    assert result.device == torch.device("meta")
    assert result.dtype == torch.int32
    assert options == [{"non_blocking": True}]


def test_puzzle_data_train_loader_applies_caps_epoch_and_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_split(tmp_path)
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
    data = _puzzle_data(
        tmp_path,
        batch_size=3,
        device="cpu",
        max_samples=7,
        seed=29,
        epochs_per_iter=3,
    )
    loader = data.train_dataloader()
    assert loader.train is True
    assert loader.device == torch.device("cpu")
    assert loader.seed == 29
    assert loader.epochs_per_iter == 3
    assert loader.n_examples == 7
    assert loader.iters == 0


def test_puzzle_data_timer_checkpoint_round_trips(tmp_path: Path) -> None:
    _write_split(tmp_path, "test")
    data = _puzzle_data(tmp_path)
    data.timer_epoch.global_count = 3
    data.timer_epoch.global_sec = 1.25
    state = data.state_dict()
    restored = _puzzle_data(tmp_path)
    restored.load_state_dict(state)
    assert state == {
        "train_iters": 0,
        "timer_epoch": {"global_count": 3, "global_sec": 1.25},
    }
    assert restored.state_dict() == state


def _write_helper_arc_split(
    root: Path,
    *,
    spatial_tags: np.ndarray | None = None,
) -> Path:
    split = root / "train"
    split.mkdir(parents=True)
    (split / "dataset.json").write_text(
        '{"ignore_label_id":255,"blank_identifier_id":0}',
    )
    np.save(split / "all__inputs.npy", np.arange(24, dtype=np.int32).reshape(6, 4))
    np.save(
        split / "all__labels.npy",
        np.arange(24, dtype=np.int32).reshape(6, 4) + 30,
    )
    np.save(split / "all__puzzle_indices.npy", np.array([0, 2, 5, 6], dtype=np.int64))
    np.save(split / "all__group_indices.npy", np.array([0, 2, 3], dtype=np.int64))
    np.save(
        split / "all__puzzle_identifiers.npy",
        np.array([11, 22, 33], dtype=np.int32),
    )
    if spatial_tags is not None:
        np.save(split / "all__spatial_tags.npy", spatial_tags)
    return root


def test_g11_load_puzzle_dataset_trims_rows_and_hierarchy(tmp_path: Path) -> None:
    root = _write_helper_arc_split(tmp_path)

    result = load_puzzle_dataset(root, "train", max_samples=5)

    assert result["inputs"].shape == (5, 4)
    assert result["labels"].shape == (5, 4)
    np.testing.assert_array_equal(result["puzzle_indices"], [0, 2, 5])
    np.testing.assert_array_equal(result["group_indices"], [0, 2])
    np.testing.assert_array_equal(result["puzzle_identifiers"], [11, 22])
    np.testing.assert_array_equal(result["spatial_tags"], [[1, 0, 0], [1, 0, 0]])
    assert result["metadata"] == {"ignore_label_id": 255, "blank_identifier_id": 0}
    assert result["inputs"].dtype == np.int32
    assert result["puzzle_indices"].dtype == np.int64


def test_puzzle_train_length_is_not_yield_count(tmp_path: Path) -> None:
    _write_small_puzzle_split(tmp_path)
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=0,
        num_replicas=1,
        train=True,
        seed=7,
        epochs_per_iter=2,
    )

    assert len(loader) == 8
    assert len(list(loader)) < len(loader)


def test_puzzle_to_device_copies_strided_int32_values(tmp_path: Path) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=0,
    )
    source = np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2]

    result = loader._to_device(source)

    assert result.tolist() == [[0, 2], [4, 6], [8, 10]]
    assert result.is_contiguous()
    assert result.dtype == torch.int32
    assert result.device == torch.device("cpu")


def test_eval_loader_honors_the_row_cap(tmp_path: Path) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    data = _small_puzzle_data(tmp_path, max_samples=4)

    assert sum(batch["valid_count"] for batch in data.eval_dataloader()) == 4


def test_puzzle_sample_batch_empty_plan_returns_int64_arrays(tmp_path: Path) -> None:
    loader = _g00_puzzle_batches(tmp_path, batch_size=2)
    rng = np.random.Generator(np.random.Philox(seed=3))

    next_group, rows, puzzle_ids = loader._sample_batch(
        rng,
        np.empty(0, dtype=np.int64),
        0,
    )

    assert next_group == 0
    assert rows.shape == puzzle_ids.shape == (0,)
    assert rows.dtype == puzzle_ids.dtype == np.int64


def test_puzzle_sample_batch_keeps_sampled_rows_unique_and_ids_int64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path, batch_size=4)
    rng = np.random.Generator(np.random.Philox(seed=3))
    full = Mock(wraps=np.full)
    monkeypatch.setattr(np, "full", full)

    next_group, rows, puzzle_ids = loader._sample_batch(
        rng,
        np.array([0, 1], dtype=np.int64),
        0,
    )

    assert next_group == 2
    assert rows.shape == puzzle_ids.shape == (4,)
    assert rows.dtype == puzzle_ids.dtype == np.int64
    assert np.unique(rows).size == 4
    assert puzzle_ids[0] == puzzle_ids[1]
    assert 0 <= puzzle_ids[0] < 2
    assert puzzle_ids[2:].tolist() == [2, 2]
    typed_calls = [call for call in full.call_args_list if len(call.args) == 2]
    assert typed_calls
    assert all(call.kwargs.get("dtype") is np.int64 for call in typed_calls)


def test_puzzle_sample_batch_requests_without_replacement(tmp_path: Path) -> None:
    loader = _g00_puzzle_batches(tmp_path, batch_size=4)

    class RecordingRng:
        def __init__(self) -> None:
            self.replacements: list[bool | None] = []

        def integers(self, low: int, high: int) -> int:
            assert low < high
            return low

        def choice(
            self,
            a: int,
            size: int,
            *,
            replace: bool | None = True,
        ) -> np.ndarray:
            assert a >= size
            self.replacements.append(replace)
            return np.arange(size, dtype=np.int64)

    rng = RecordingRng()
    next_group, rows, puzzle_ids = loader._sample_batch(
        rng,
        np.array([0, 1], dtype=np.int64),
        0,
    )

    assert next_group == 2
    assert rows.tolist() == [0, 1, 4, 5]
    assert puzzle_ids.tolist() == [0, 0, 2, 2]
    assert rng.replacements == [False, False]


def test_puzzle_collate_keeps_outputs_on_meta_device(tmp_path: Path) -> None:
    loader = _g00_puzzle_batches(tmp_path, device="meta")

    batch = loader._collate(
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )

    assert batch["media"].device == torch.device("meta")
    assert batch["label"].device == torch.device("meta")
    assert batch["puzzle_identifiers"].device == torch.device("meta")
    assert batch["spatial_tags"].device == torch.device("meta")
    assert batch["puzzle_identifiers"].dtype == torch.int64
    assert batch["spatial_tags"].dtype == torch.int64


def test_puzzle_collate_does_not_pad_full_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path, batch_size=2)
    cat = Mock(wraps=torch.cat)
    monkeypatch.setattr(torch, "cat", cat)

    loader._collate(
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )

    cat.assert_not_called()


def test_puzzle_collate_constructs_padding_tags_with_dtype_and_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path, batch_size=3, device="meta")
    tensor = Mock(wraps=torch.tensor)
    monkeypatch.setattr(torch, "tensor", tensor)

    loader._collate(
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )

    padding_calls = [
        call for call in tensor.call_args_list if call.args == ([1, 0, 0],)
    ]
    assert len(padding_calls) == 1
    assert padding_calls[0].kwargs == {
        "dtype": torch.int64,
        "device": loader.device,
    }


def test_puzzle_to_device_requests_int32_contiguous_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    source = np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2]
    contiguous = Mock(wraps=np.ascontiguousarray)
    monkeypatch.setattr(np, "ascontiguousarray", contiguous)

    result = loader._to_device(source)

    assert result.tolist() == [[0, 2], [4, 6], [8, 10]]
    assert result.is_contiguous()
    assert result.dtype == torch.int32
    assert result.device == torch.device("cpu")
    contiguous.assert_called_once_with(source, dtype=np.int32)


def test_puzzle_to_device_pins_cuda_before_nonblocking_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    device = torch.device("cuda")
    loader.device = device
    pin_calls: list[torch.dtype] = []
    move_calls: list[tuple[object, bool]] = []

    def pin_memory(tensor: Tensor) -> Tensor:
        pin_calls.append(tensor.dtype)
        return tensor

    def move(
        tensor: Tensor,
        target: object = None,
        *,
        non_blocking: bool = False,
    ) -> Tensor:
        move_calls.append((target, non_blocking))
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    monkeypatch.setattr(torch.Tensor, "to", move)

    result = loader._to_device(np.array([4, 8], dtype=np.int32))

    assert result.dtype == torch.int32
    assert pin_calls == [torch.int32]
    assert move_calls == [(device, True)]


def test_puzzle_eval_len_uses_ceiling_and_unit_batch_floor(tmp_path: Path) -> None:
    loader = _g00_puzzle_batches(tmp_path, batch_size=5)
    loader.n_examples = 6
    assert len(loader) == 2

    _write_small_puzzle_split(tmp_path, "train")
    training = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=1,
        rank=0,
        num_replicas=1,
        train=True,
        seed=1,
    )
    assert len(training) == 12


def test_puzzle_eval_iteration_requests_int64_identity_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    ranges = Mock(wraps=np.arange)
    monkeypatch.setattr(np, "arange", ranges)

    batches = list(loader)

    assert len(batches) == 3
    assert any(
        call.args == (6,) and call.kwargs == {"dtype": np.int64}
        for call in ranges.call_args_list
    )


def test_puzzle_train_iteration_stops_after_exact_full_plan(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "train")
    split = tmp_path / "train"
    np.save(split / "all__inputs.npy", np.arange(8, dtype=np.int32).reshape(2, 4))
    np.save(split / "all__labels.npy", np.arange(8, dtype=np.int32).reshape(2, 4))
    np.save(split / "all__puzzle_indices.npy", np.array([0, 2]))
    np.save(split / "all__group_indices.npy", np.array([0, 1]))
    np.save(split / "all__puzzle_identifiers.npy", np.array([1], dtype=np.int32))
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=True,
        seed=1,
    )

    batches = list(loader)

    assert len(batches) == 1
    assert batches[0]["media"].shape == (2, 4)


def test_puzzle_train_iteration_matches_previous_partial_tail_behavior(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "train")

    def make_loader() -> PuzzleBatches:
        return PuzzleBatches(
            dataset_dir=tmp_path,
            device="cpu",
            batch_size=5,
            rank=0,
            num_replicas=1,
            train=True,
            seed=17,
            epochs_per_iter=1,
        )

    previous = make_loader()
    current = make_loader()

    def previous_iteration(
        loader: PuzzleBatches,
    ) -> tuple[list[PuzzleData.Batch], bool]:
        loader.iters += 1
        rng = np.random.Generator(np.random.Philox(seed=loader.seed + loader.iters))
        group_order = np.concatenate(
            [rng.permutation(loader.n_groups) for _ in range(loader.epochs_per_iter)],
        )
        start = 0
        batches: list[PuzzleData.Batch] = []
        had_partial_tail = False
        while start < group_order.size:
            start, ex_idx, puz_idx = loader._sample_batch(rng, group_order, start)
            if ex_idx.size < loader.global_batch_size:
                had_partial_tail = True
                break
            local = slice(
                loader.rank * loader.batch_size,
                (loader.rank + 1) * loader.batch_size,
            )
            batches.append(
                loader._collate(ex_idx[local], puz_idx[local], valid=loader.batch_size),
            )
        return batches, had_partial_tail

    expected, had_partial_tail = previous_iteration(previous)
    observed = list(current)

    assert had_partial_tail
    assert len(observed) == len(expected)
    for observed_batch, expected_batch in zip(observed, expected, strict=True):
        assert observed_batch["valid_count"] == expected_batch["valid_count"]
        for field in ("media", "label", "puzzle_identifiers", "spatial_tags"):
            assert torch.equal(observed_batch[field], expected_batch[field])


def test_puzzle_train_iteration_uses_exact_rank_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_small_puzzle_split(tmp_path, "train")
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=2,
        train=True,
        seed=1,
    )

    def sample_batch(
        rng: np.random.Generator,
        group_order: np.ndarray,
        start_index: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        del rng
        assert start_index == 0
        return (
            group_order.size,
            np.array([0, 1, 2, 3], dtype=np.int64),
            np.array([0, 0, 1, 1], dtype=np.int64),
        )

    monkeypatch.setattr(loader, "_sample_batch", sample_batch)
    loader.rank = 0
    first_rank = list(loader)
    loader.rank = 1
    second_rank = list(loader)

    assert len(first_rank) == len(second_rank) == 1
    assert first_rank[0]["media"][:, 0].tolist() == [2, 6]
    assert first_rank[0]["puzzle_identifiers"].tolist() == [1, 1]
    assert second_rank[0]["media"][:, 0].tolist() == [10, 14]
    assert second_rank[0]["puzzle_identifiers"].tolist() == [2, 2]


def test_puzzle_data_init_ensures_and_checks_one_identifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "arc"
    dataset_dir.mkdir()
    (dataset_dir / "identifiers.json").write_text('["one"]')
    config = PuzzleData.Config(
        working_dir=dataset_dir,
        num_puzzle_identifiers=1,
        device="cpu",
    )
    config.augmentation.spec = config.spec
    ensure = Mock()
    monkeypatch.setattr(arc_data, "ensure_arc_dataset", ensure)

    data = PuzzleData(config)

    assert data.dataset_dir == dataset_dir
    ensure.assert_called_once()
    assert ensure.call_args.kwargs.keys() == {"target_dir", "augmentation"}
    assert ensure.call_args.kwargs["target_dir"] == dataset_dir
    assert ensure.call_args.kwargs["augmentation"] is not None


def test_puzzle_data_init_reports_exact_identifier_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "arc"
    dataset_dir.mkdir()
    (dataset_dir / "identifiers.json").write_text('["one"]')
    config = PuzzleData.Config(
        working_dir=dataset_dir,
        num_puzzle_identifiers=2,
        device="cpu",
    )
    config.augmentation.spec = config.spec
    ensure = Mock()
    monkeypatch.setattr(arc_data, "ensure_arc_dataset", ensure)

    with pytest.raises(ValueError, match="expected 2 puzzle identifiers") as error:
        PuzzleData(config)

    assert str(error.value) == (
        f"expected 2 puzzle identifiers (config) but found 1 at "
        f"{dataset_dir / 'identifiers.json'}; the dataset build differs or data "
        "is partial/corrupt -- re-run data ensure or update the config constant."
    )
    ensure.assert_called_once()


def _write_boundary_split(root: Path) -> None:
    split = root / "train"
    split.mkdir()
    (split / "dataset.json").write_text(
        '{"ignore_label_id":0,"blank_identifier_id":99}',
    )
    rows = np.arange(24, dtype=np.int32).reshape(6, 4)
    np.save(split / "all__inputs.npy", rows)
    np.save(split / "all__labels.npy", rows + 30)
    np.save(
        split / "all__puzzle_indices.npy",
        np.array([0, 2, 2, 5, 6], dtype=np.int32),
    )
    np.save(split / "all__group_indices.npy", np.array([0, 2, 4], dtype=np.int32))
    np.save(
        split / "all__puzzle_identifiers.npy",
        np.array([11, 12, 13, 14], dtype=np.int32),
    )


def test_load_puzzle_dataset_preserves_duplicate_boundary_at_cap(
    tmp_path: Path,
) -> None:
    _write_boundary_split(tmp_path)

    result = load_puzzle_dataset(tmp_path, "train", max_samples=2)

    assert result["puzzle_indices"].tolist() == [0, 2, 2]
    assert result["group_indices"].tolist() == [0, 2]
    assert result["puzzle_identifiers"].tolist() == [11, 12]
    assert result["spatial_tags"].tolist() == [[1, 0, 0], [1, 0, 0]]


def test_load_puzzle_dataset_requests_dtypes_for_appended_boundaries_and_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_boundary_split(tmp_path)
    array = Mock(wraps=np.array)
    asarray = Mock(wraps=np.asarray)
    monkeypatch.setattr(np, "array", array)
    monkeypatch.setattr(np, "asarray", asarray)

    result = load_puzzle_dataset(tmp_path, "train", max_samples=3)

    assert result["puzzle_indices"].tolist() == [0, 2, 2, 3]
    assert result["group_indices"].tolist() == [0, 2, 3]
    assert result["puzzle_identifiers"].tolist() == [11, 12, 13]
    assert result["puzzle_indices"].dtype == np.int64
    assert result["group_indices"].dtype == np.int64
    assert result["puzzle_identifiers"].dtype == np.int32
    assert result["spatial_tags"].dtype == np.int32
    boundary_calls = [
        call
        for call in array.call_args_list
        if call.args and isinstance(call.args[0], list) and call.args[0] == [3]
    ]
    assert len(boundary_calls) == 2
    assert all(call.kwargs == {"dtype": np.dtype(np.int32)} for call in boundary_calls)
    tag_calls = [
        call
        for call in array.call_args_list
        if call.args and isinstance(call.args[0], list) and call.args[0] == [1, 0, 0]
    ]
    assert len(tag_calls) == 1
    assert tag_calls[0].kwargs == {"dtype": np.int32}
    assert asarray.call_args_list[-1].kwargs == {"dtype": np.int32}


def _g00_puzzle_batches(
    root: Path,
    *,
    batch_size: int = 2,
    device: str = "cpu",
) -> PuzzleBatches:
    split = root / "test"
    split.mkdir(parents=True)
    (split / "dataset.json").write_text(
        '{"ignore_label_id":255,"blank_identifier_id":99}',
    )
    inputs = np.arange(24, dtype=np.int32).reshape(6, 4)
    labels = inputs + 30
    labels[0, 0] = 255
    np.save(split / "all__inputs.npy", inputs)
    np.save(split / "all__labels.npy", labels)
    np.save(split / "all__puzzle_indices.npy", np.array([0, 2, 4, 6]))
    np.save(split / "all__group_indices.npy", np.array([0, 2, 3]))
    np.save(
        split / "all__puzzle_identifiers.npy",
        np.array([1, 2, 3], dtype=np.int32),
    )
    np.save(
        split / "all__spatial_tags.npy",
        np.array([[2, 1, 0], [3, 0, 1], [4, 1, 1]], dtype=np.int32),
    )
    return PuzzleBatches(
        dataset_dir=root,
        device=device,
        batch_size=batch_size,
        rank=0,
        num_replicas=1,
        train=False,
        seed=0,
    )


def test_pair_view_collate_masks_labels_and_pads_every_field() -> None:
    inputs = [np.zeros((2, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.uint8)]
    samples: list[arc_data.PairViewSample] = [
        {
            "media": np.arange(1, 10, dtype=np.int32),
            "puzzle_identifiers": 5,
            "puzzle_idxs": 6,
            "dih_idxs": 7,
            "colorperm_slots": tuple(range(8, -1, -1)),
            "spatial_tags": (0, 2, 3),
            "canonical_puzzle_name": "first",
            "canonical_input": inputs[0],
            "canonical_output": None,
            "input_metadata": {"view": 0},
            "label": np.array([0, 5, 0, 7, 1, 0, 2, 3, 4], dtype=np.int32),
        },
        {
            "media": np.arange(10, 19, dtype=np.int32),
            "puzzle_identifiers": 9,
            "puzzle_idxs": 10,
            "dih_idxs": 11,
            "colorperm_slots": (1, 0, 2, 3, 4, 5, 6, 7, 8),
            "spatial_tags": (1, 4, 5),
            "canonical_puzzle_name": "second",
            "canonical_input": inputs[1],
            "canonical_output": inputs[0],
            "input_metadata": {"view": 1},
            "label": np.array([1, 0, 2, 0, 0, 3, 4, 5, 6], dtype=np.int32),
        },
    ]
    batch = arc_data._collate(samples, batch_size=4, canvas_size=3, include_label=True)
    assert set(batch) == {
        "media",
        "label",
        "puzzle_identifiers",
        "puzzle_idxs",
        "dih_idxs",
        "colorperm_slots",
        "spatial_tags",
        "valid_count",
        "canonical_puzzle_names",
        "canonical_inputs",
        "canonical_outputs",
        "input_metadata",
    }
    assert "label" in batch
    assert batch["media"].dtype == batch["label"].dtype == torch.int32
    assert batch["media"].tolist() == [
        list(range(1, 10)),
        list(range(10, 19)),
        [0] * 9,
        [0] * 9,
    ]
    assert batch["label"].tolist() == [
        [-100, 5, -100, 7, 1, -100, 2, 3, 4],
        [1, -100, 2, -100, -100, 3, 4, 5, 6],
        [-100] * 9,
        [-100] * 9,
    ]
    ids = ("puzzle_identifiers", "puzzle_idxs", "dih_idxs")
    assert all(batch[name].dtype == torch.int64 for name in ids)
    assert [batch[name].tolist() for name in ids] == [
        [5, 9, 0, 0],
        [6, 10, 0, 0],
        [7, 11, 0, 0],
    ]
    assert batch["colorperm_slots"].dtype == batch["spatial_tags"].dtype == torch.int64
    assert batch["colorperm_slots"].tolist() == [
        list(range(8, -1, -1)),
        [1, 0, 2, 3, 4, 5, 6, 7, 8],
        list(range(9)),
        list(range(9)),
    ]
    assert batch["spatial_tags"].tolist() == [
        [0, 2, 3],
        [1, 4, 5],
        [1, 0, 0],
        [1, 0, 0],
    ]
    assert batch["valid_count"] == 2
    assert batch["canonical_puzzle_names"] == ["first", "second"]
    assert batch["canonical_inputs"] == inputs
    assert batch["canonical_outputs"] == [None, inputs[0]]
    assert batch["input_metadata"] == [{"view": 0}, {"view": 1}]
    del samples[0]["label"]
    with pytest.raises(ValueError, match=r"^Training sample is missing its label\.$"):
        arc_data._collate(samples, batch_size=4, canvas_size=3, include_label=True)
    unlabelled = arc_data._collate(
        samples,
        batch_size=4,
        canvas_size=3,
        include_label=False,
    )
    assert "label" not in unlabelled


def test_iter_parquet_rows_streams_typed_v2_rows(tmp_path: Path) -> None:
    """Stream parquet batches into typed rows without changing source order."""
    path = tmp_path / "tiny.parquet"
    row: dict[str, object] = {
        "format_version": 2,
        "puzzle_id": "task-a",
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
        "offline_augs": ["", "dih|1||colorperm|0312456789"],
    }
    second_row: dict[str, object] = {**row, "puzzle_id": "task-b"}
    _write_parquet(path, rows=[row, second_row])

    got = list(iter_parquet_rows(path, batch_size=1))

    assert [item["puzzle_id"] for item in got] == ["task-a", "task-b"]
    assert got[0]["train"][0]["output"] == [[2, 1]]
    assert got[0]["offline_augs"] == ["", "dih|1||colorperm|0312456789"]


@pytest.mark.parametrize(
    ("aug", "dih", "slots"),
    [
        ("", 0, tuple(range(9))),
        ("dih|7", 7, tuple(range(9))),
        ("colorperm|0312456789", 0, (2, 0, 1, 3, 4, 5, 6, 7, 8)),
        (
            "dih|3||colorperm|0312456789",
            3,
            (2, 0, 1, 3, 4, 5, 6, 7, 8),
        ),
    ],
)
def test_parse_offline_aug(
    aug: str,
    dih: int,
    slots: tuple[int, ...],
) -> None:
    """Parse each supported offline view into its dihedral and color slots."""
    assert parse_offline_aug(aug) == (dih, slots)


@pytest.mark.parametrize(
    "aug",
    ["scale|2", "dih|8", "colorperm|0123456788", "dih|1||dih|2"],
)
def test_parse_offline_aug_rejects_unsupported_or_invalid_values(aug: str) -> None:
    """Reject unsupported transform names and malformed view encodings."""
    with pytest.raises(ValueError, match=r"Unsupported|Invalid|Duplicate"):
        parse_offline_aug(aug)


def test_iter_parquet_rows_rejects_wrong_version(tmp_path: Path) -> None:
    """Reject rows whose format version is not the released v2 schema."""
    path = tmp_path / "wrong-version.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 1,
                "puzzle_id": "task-a",
                "train": [],
                "test": [],
                "offline_augs": [""],
            },
        ],
    )

    with pytest.raises(ValueError, match="format_version=1"):
        list(iter_parquet_rows(path))


def test_reader_filters_large_grids_and_keeps_missing_eval_output(
    tmp_path: Path,
) -> None:
    """Filter oversized training grids and retain valid label-free eval rows."""
    oversized = [[1] * 31]
    path = tmp_path / "filtered.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "task",
                "train": [
                    {"input": oversized, "output": [[2]]},
                    {"input": [[3]], "output": None},
                ],
                "test": [
                    {"input": oversized, "output": [[2]]},
                    {"input": [[4]], "output": oversized},
                    {"input": [[5]], "output": None},
                ],
                "offline_augs": [""],
            },
        ],
    )

    source = read_parquet_source(path)

    assert source.row_train_counts.tolist() == [0]
    assert source.row_test_counts.tolist() == [2]
    assert source.row_test_train_counts.tolist() == [1]
    assert source.test_outputs.get(1) is None
    with pytest.raises(ValueError, match="pair kind"):
        source.get_pair(0, 3)


def test_reader_preserves_original_pair_indices_and_raw_aug_identity(
    tmp_path: Path,
) -> None:
    """Retain original task-local pair indexes and raw augmentation strings."""
    path = tmp_path / "original-indices.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "task",
                "train": [
                    {"input": [[1] * 31], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                ],
                "test": [
                    {"input": [[5] * 31], "output": [[6]]},
                    {"input": [[7]], "output": [[8]]},
                ],
                "offline_augs": ["", "dih|0"],
            },
        ],
    )

    source = read_parquet_source(path)

    assert source.row_train_counts.tolist() == [1]
    assert source.row_test_counts.tolist() == [1]
    assert source.train_original_indices.tolist() == [1]
    assert source.test_original_indices.tolist() == [1]
    assert source.offline_aug_strings == ("", "dih|0")
    assert source.offline_aug_ids.tolist() == [0, 1]


def test_reader_stores_shapes_without_canvas_dependent_narrowing(
    tmp_path: Path,
) -> None:
    """Store grid dimensions without canvas-dependent integer narrowing."""
    path = tmp_path / "wide-shape.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "wide",
                "train": [{"input": [[1] * 256], "output": [[2]]}],
                "test": [],
                "offline_augs": [""],
            },
        ],
    )

    source = read_parquet_source(path, max_grid_size=256)

    assert source.train_inputs.shapes.dtype == np.int64
    grid = source.train_inputs.get(0)
    assert grid is not None
    assert grid.shape == (1, 256)


def test_reader_preserves_evaluation_output_larger_than_training_canvas(
    tmp_path: Path,
) -> None:
    """Preserve full canonical evaluation outputs beyond model canvas bounds."""
    path = tmp_path / "wide-eval-target.parquet"
    wide_output = [[2] * 256]
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "wide-eval",
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": wide_output}],
                "offline_augs": [""],
            },
        ],
    )

    source = read_parquet_source(path, max_grid_size=30)

    assert source.row_train_counts.tolist() == [1]
    assert source.row_test_counts.tolist() == [1]
    output = source.test_outputs.get(0)
    assert output is not None
    assert output.shape == (1, 256)
    assert output.tolist() == wide_output


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ([], "positive height"),
        ([[1], [1, 2]], "rectangular"),
        ([[10]], "integer colors"),
    ],
)
def test_grid_validation_rejects_invalid_shapes_and_colors(
    tmp_path: Path,
    grid: list[list[int]],
    message: str,
) -> None:
    """Reject empty, ragged, or out-of-range ARC grids at the parquet boundary."""
    path = tmp_path / "invalid-grid.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "task",
                "train": [{"input": grid, "output": [[1]]}],
                "test": [],
                "offline_augs": [""],
            },
        ],
    )

    with pytest.raises(ValueError, match=message):
        list(iter_parquet_rows(path))


def test_parquet_schema_duplicate_rows_and_missing_identity(tmp_path: Path) -> None:
    """Reject missing schema fields, duplicate rows, and absent base views."""
    missing = tmp_path / "missing.parquet"
    _write_parquet(missing, rows=[{"format_version": 2, "puzzle_id": "task"}])
    with pytest.raises(ValueError, match="missing columns"):
        list(iter_parquet_rows(missing))

    duplicate = tmp_path / "duplicate.parquet"
    row: dict[str, object] = {
        "format_version": 2,
        "puzzle_id": "task",
        "train": [],
        "test": [],
        "offline_augs": [""],
    }
    _write_parquet(duplicate, rows=[row, row])
    with pytest.raises(ValueError, match="duplicate v2 puzzle row"):
        read_parquet_source(duplicate)

    invalid_aug = tmp_path / "missing-identity.parquet"
    row["offline_augs"] = ["dih|1"]
    _write_parquet(invalid_aug, rows=[row])
    with pytest.raises(ValueError, match="no-augmentation entry first"):
        list(iter_parquet_rows(invalid_aug))


def test_train_eval_ids_metadata_and_demo_separation(tmp_path: Path) -> None:
    """Check paired train/eval IDs and keep eval labels out of batches."""
    path = _write(
        tmp_path / "mixed.parquet",
        rows=[
            _row(
                "task-a",
                train=[
                    _pair([[1, 2], [3, 4]], out=[[5], [6], [7]]),
                    _pair([[8]], out=[[9, 0]]),
                ],
                test=[_pair([[2, 1]], out=[[4, 3]])],
                augs=["", "dih|1", "colorperm|0312456789"],
            ),
            _row(
                "task-b",
                train=[_pair([[1]], out=[[2]])],
                test=[_pair([[3]], out=[[4]])],
                augs=[""],
            ),
        ],
    )
    data = _configured_data(path, eval_path=path)

    train_batches = list(data.train_dataloader())
    train_names = [
        name for batch in train_batches for name in batch["canonical_puzzle_names"]
    ]
    assert len(train_names) == 28
    assert set(train_names) == {"task-a", "task-b"}
    assert {
        task_id
        for batch in train_batches
        for task_id in from_plain(batch["puzzle_idxs"].tolist(), list[int])
    } == {0, 1}
    assert "label" in train_batches[0]
    assert train_batches[0]["media"].dtype == torch.int32
    assert train_batches[0]["label"].dtype == torch.int32
    assert train_batches[0]["spatial_tags"].shape[1] == 3

    eval_batches = list(data.eval_dataloader())
    eval_names = [
        name for batch in eval_batches for name in batch["canonical_puzzle_names"]
    ]
    assert eval_names == ["task-a", "task-a", "task-a", "task-b"]
    assert all("label" not in batch for batch in eval_batches)
    canonical_output = eval_batches[0]["canonical_outputs"][0]
    assert canonical_output is not None
    assert canonical_output.tolist() == [[4, 3]]
    assert from_plain(eval_batches[0]["puzzle_idxs"].tolist(), list[int]) == [0, 0]
    assert eval_batches[0]["puzzle_identifiers"].tolist()[0] == 1
    assert eval_batches[0]["dih_idxs"].tolist() == [0, 1]
    assert data.instance_metadata[1] is not None
    assert data.instance_metadata[1].puzzle_name == "task-a"
    assert data.instance_metadata[1].task_id == 0
    assert data.instance_metadata[2] is not None
    assert data.instance_metadata[2].dih_idx == 1
    assert data.instance_metadata[3] is not None
    assert data.instance_metadata[3].colorperm_slots == (2, 0, 1, 3, 4, 5, 6, 7, 8)


def test_path_order_duplicate_task_identity_and_multiplicity(tmp_path: Path) -> None:
    """Preserve source order, semantic IDs, and appended repetitions."""
    first = _write(
        tmp_path / "first.parquet",
        rows=[
            _row(
                "shared",
                train=[_pair([[1]], out=[[2]])],
                test=[],
                augs=["", "dih|1"],
            ),
        ],
    )
    second = _write(
        tmp_path / "second.parquet",
        rows=[
            _row(
                "shared",
                train=[_pair([[3]], out=[[4]])],
                test=[],
                augs=["", "dih|1"],
            ),
            _row("later", train=[_pair([[5]], out=[[6]])], test=[], augs=[""]),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [
        PairSource.Config(path=first, pair_selection="train"),
        PairSource.Config(path=second, pair_selection="train", multiplicity=2),
    ]
    cfg.eval_source = first
    cfg.batch_size = 2
    cfg.p_translate = 0.0
    data = cfg.make()
    samples = [data.train_dataset[index] for index in range(len(data.train_dataset))]

    assert [sample["canonical_puzzle_name"] for sample in samples] == [
        "shared",
        "shared",
        "shared",
        "shared",
        "later",
        "shared",
        "shared",
        "later",
    ]
    assert [sample["puzzle_idxs"] for sample in samples[:5]] == [0, 0, 0, 0, 1]
    assert [sample["puzzle_identifiers"] for sample in samples] == [
        1,
        2,
        1,
        2,
        3,
        1,
        2,
        3,
    ]
    assert samples[0]["canonical_input"].tolist() == [[1]]
    assert samples[2]["canonical_input"].tolist() == [[3]]


def test_mixed_source_selection_keeps_test_only_rows_and_offsets(
    tmp_path: Path,
) -> None:
    """Append selected test pairs after demonstrations across source rows."""
    demonstrations = _write(
        tmp_path / "demonstrations.parquet",
        rows=[
            _row(
                "demo-task",
                train=[_pair([[1]], out=[[2]]), _pair([[3]], out=[[4]])],
                test=[],
                augs=[""],
            ),
        ],
    )
    mixed = _write(
        tmp_path / "mixed.parquet",
        rows=[
            _row(
                "prefix-task",
                train=[_pair([[5]], out=[[6]])],
                test=[_pair([[7]], out=[[8]])],
                augs=[""],
            ),
            _row(
                "test-only-task",
                train=[],
                test=[_pair([[8]], out=[[9]]), _pair([[9]], out=[[8]])],
                augs=[""],
            ),
            _row(
                "demo-only-in-both-task",
                train=[_pair([[4]], out=[[5]])],
                test=[],
                augs=[""],
            ),
        ],
    )
    config = ArcParquetData.Config()
    config.train_sources = [
        PairSource.Config(path=demonstrations, pair_selection="train"),
        PairSource.Config(path=mixed, pair_selection="both"),
    ]
    config.eval_source = mixed
    config.p_translate = 0.0
    data = config.make()

    samples = [data.train_dataset[index] for index in range(len(data.train_dataset))]

    assert [sample["canonical_puzzle_name"] for sample in samples] == [
        "demo-task",
        "demo-task",
        "prefix-task",
        "prefix-task",
        "test-only-task",
        "test-only-task",
        "demo-only-in-both-task",
    ]
    assert [sample["canonical_input"].tolist() for sample in samples] == [
        [[1]],
        [[3]],
        [[5]],
        [[7]],
        [[8]],
        [[9]],
        [[4]],
    ]
    assert [sample["input_metadata"]["pair_index"] for sample in samples] == [
        0,
        1,
        0,
        0,
        0,
        1,
        0,
    ]
    assert data.task_to_id == {
        "demo-task": 0,
        "prefix-task": 1,
        "test-only-task": 2,
        "demo-only-in-both-task": 3,
    }


def test_empty_eval_rows_do_not_stop_later_id_registration(tmp_path: Path) -> None:
    """Continue task and view registration after rows without test pairs."""
    train_path = _write(
        tmp_path / "train.parquet",
        rows=[
            _row("train-task", train=[_pair([[1]], out=[[2]])], test=[], augs=[""]),
        ],
    )
    eval_path = _write(
        tmp_path / "eval.parquet",
        rows=[
            _row("empty-first", train=[], test=[], augs=[""]),
            _row("eval-first", train=[], test=[_pair([[3]], out=[[4]])], augs=[""]),
            _row("empty-middle", train=[], test=[], augs=[""]),
            _row("eval-last", train=[], test=[_pair([[5]], out=[[6]])], augs=[""]),
        ],
    )
    config = ArcParquetData.Config()
    config.train_sources = [PairSource.Config(path=train_path)]
    config.eval_source = eval_path
    config.num_workers = 0
    data = config.make()

    samples = [data.eval_dataset[index] for index in range(len(data.eval_dataset))]

    assert [sample["canonical_puzzle_name"] for sample in samples] == [
        "eval-first",
        "eval-last",
    ]
    assert [sample["puzzle_idxs"] for sample in samples] == [1, 2]
    assert [sample["puzzle_identifiers"] for sample in samples] == [2, 3]


def test_sources_and_eval_share_one_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse each parsed parquet source across source selections and evaluation."""
    path = _write(
        tmp_path / "shared.parquet",
        rows=[_row("task", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    original_reader = read_parquet_source
    read_paths: list[Path] = []

    def counted_reader(
        source_path: Path,
        *,
        max_grid_size: int,
    ) -> ParquetSourceData:
        read_paths.append(source_path)
        return original_reader(source_path, max_grid_size=max_grid_size)

    monkeypatch.setattr(arc_data, "read_parquet_source", counted_reader)
    config = ArcParquetData.Config()
    config.train_sources = [
        PairSource.Config(path=path),
        PairSource.Config(path=path, multiplicity=2),
    ]
    config.eval_source = path

    config.make()

    assert read_paths == [path]


def test_distinct_eval_source_uses_configured_grid_filter(tmp_path: Path) -> None:
    """Apply the configured canvas limit when loading a distinct eval file."""
    train_path = _write(
        tmp_path / "train-small-canvas.parquet",
        rows=[
            _row("train", train=[_pair([[1]], out=[[2]])], test=[], augs=[""]),
        ],
    )
    eval_path = _write(
        tmp_path / "eval-small-canvas.parquet",
        rows=[
            _row("fits", train=[], test=[_pair([[3, 4, 5]], out=[[6]])], augs=[""]),
            _row(
                "too-wide",
                train=[],
                test=[_pair([[1, 2, 3, 4]], out=[[5]])],
                augs=[""],
            ),
        ],
    )
    config = ArcParquetData.Config()
    config.train_sources = [PairSource.Config(path=train_path)]
    config.eval_source = eval_path
    config.canvas_size = 3
    config.eval_batch_size = 2
    config.num_workers = 0
    data = config.make()

    batches = list(data.eval_dataloader())

    assert [batch["valid_count"] for batch in batches] == [1]
    assert batches[0]["canonical_puzzle_names"] == ["fits"]
    assert batches[0]["media"].shape == (2, 9)


def test_registration_skips_rows_with_no_selected_pairs(tmp_path: Path) -> None:
    """Register IDs only for rows that contribute selected training pairs."""
    path = _write(
        tmp_path / "registration.parquet",
        rows=[
            _row("empty", train=[], test=[_pair([[1]], out=[[2]])], augs=[""]),
            _row("live", train=[_pair([[3]], out=[[4]])], test=[], augs=[""]),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path

    data = cfg.make()

    assert data.task_to_id == {"live": 0, "empty": 1}
    assert data.instance_to_id == {("live", ""): 1, ("empty", ""): 2}


def test_raw_offline_aliases_keep_distinct_instance_ids(tmp_path: Path) -> None:
    """Keep literal offline augmentation aliases as distinct instances."""
    path = _write(
        tmp_path / "aliases.parquet",
        rows=[
            _row("task", train=[_pair([[1]], out=[[2]])], test=[], augs=["", "dih|0"]),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path

    data = cfg.make()
    samples = [data.train_dataset[index] for index in range(len(data.train_dataset))]

    assert [sample["puzzle_identifiers"] for sample in samples] == [1, 2]
    assert [sample["dih_idxs"] for sample in samples] == [0, 0]
    assert data.instance_metadata[1] == data.instance_metadata[2]
    assert data.instance_to_id == {("task", ""): 1, ("task", "dih|0"): 2}


def test_selected_targetless_test_fails_with_source_context(tmp_path: Path) -> None:
    """Reject selected targetless training pairs with source context."""
    path = _write(
        tmp_path / "targetless.parquet",
        rows=[
            _row(
                "targetless-task",
                train=[_pair([[1]], out=[[2]])],
                test=[{"input": [[3]], "output": None}],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="both")]
    cfg.eval_source = path

    with pytest.raises(
        ValueError,
        match=r"targetless\.parquet.*targetless-task.*pair 0",
    ):
        cfg.make()


def test_pair_metadata_retains_task_local_index_after_filtering(
    tmp_path: Path,
) -> None:
    """Retain original task-local pair indexes after grid filtering."""
    oversized = [[1] * 31]
    path = _write(
        tmp_path / "local-index.parquet",
        rows=[
            _row(
                "task",
                train=[
                    _pair(oversized, out=[[2]]),
                    _pair([[3]], out=[[4]]),
                ],
                test=[
                    _pair(oversized, out=[[5]]),
                    _pair([[6]], out=[[7]]),
                ],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="both")]
    cfg.eval_source = path

    data = cfg.make()

    assert data.train_dataset[0]["input_metadata"]["pair_index"] == 1
    assert data.train_dataset[1]["input_metadata"]["pair_index"] == 1
    assert data.eval_dataset[0]["input_metadata"]["pair_index"] == 1


def test_evaluation_retains_canonical_target_beyond_canvas_width(
    tmp_path: Path,
) -> None:
    """Keep canonical eval targets even when wider than the model canvas."""
    output = [[2] * 256]
    path = _write(
        tmp_path / "wide-eval-output.parquet",
        rows=[_row("task", train=[], test=[_pair([[1]], out=output)], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    cfg.canvas_size = 30
    cfg.num_workers = 0
    data = cfg.make()

    sample = data.eval_dataset[0]

    assert sample["canonical_output"] is not None
    assert sample["canonical_output"].shape == (1, 256)
    assert sample["canonical_output"].tolist() == output
    assert "label" not in sample


def test_eval_distributed_sampler_keeps_duplicate_prediction_votes(
    tmp_path: Path,
) -> None:
    """Retain DistributedSampler padding duplicates as evaluation votes."""
    path = _write(
        tmp_path / "eval-padding.parquet",
        rows=[
            _row(
                str(index),
                train=[],
                test=[_pair([[index]], out=[[index]])],
                augs=[""],
            )
            for index in range(5)
        ],
    )

    left = _eval_task_ids(path, rank=0)
    right = _eval_task_ids(path, rank=1)

    assert left == [0, 2, 4]
    assert right == [1, 3, 0]
    assert left.count(0) + right.count(0) == 2


def test_rank_shards_follow_distributed_sampler_epochs(tmp_path: Path) -> None:
    """Shard expanded training indexes using seeded sampler epochs."""
    path = _write(
        tmp_path / "rank.parquet",
        rows=[
            _row(
                str(index),
                train=[_pair([[index]], out=[[index]])],
                test=[],
                augs=[""],
            )
            for index in range(5)
        ],
    )

    left_data = _make_rank_data(path, rank=0)
    right_data = _make_rank_data(path, rank=1)
    left = left_data.train_dataloader()
    right = right_data.train_dataloader()
    left_data.train_dataset.set_epoch(4)
    right_data.train_dataset.set_epoch(4)
    left_ids = [
        task_id
        for batch in left
        for task_id in from_plain(batch["puzzle_idxs"].tolist(), list[int])
    ]
    right_ids = [
        task_id
        for batch in right
        for task_id in from_plain(batch["puzzle_idxs"].tolist(), list[int])
    ]
    expected_left = torch.utils.data.DistributedSampler[int](
        left_data.train_dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=17,
    )
    expected_right = torch.utils.data.DistributedSampler[int](
        right_data.train_dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=17,
    )
    expected_left.set_epoch(4)
    expected_right.set_epoch(4)
    assert left_ids == list(expected_left)
    assert right_ids == list(expected_right)
    assert len(left_ids) == len(right_ids) == 3
    assert sorted(set(left_ids + right_ids)) == [0, 1, 2, 3, 4]
    assert len(left_ids + right_ids) == 6


def test_expected_counts_and_resume_state(tmp_path: Path) -> None:
    """Verify configured counts and epoch-start checkpoint restart behavior."""
    path = _write(
        tmp_path / "state.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.num_tasks = 1
    cfg.num_instances = 1
    cfg.batch_size = 1
    cfg.num_workers = 0
    data = cfg.make()
    train_loader = data.train_dataloader()
    eval_loader = data.eval_dataloader()
    assert data.train_dataset.sampler is train_loader.sampler
    assert data.eval_dataset.sampler is eval_loader.sampler
    data.eval_dataset.set_epoch(4)
    assert data.eval_dataset.sampler is not None
    data.timer_epoch.global_count = 3
    state = data.state_dict()
    resumed = cfg.make()
    resumed.load_state_dict(state)

    assert resumed.timer_epoch.global_count == 3
    assert resumed.state_dict() == state
    first_pass = next(iter(data.train_dataloader()))
    restarted_pass = next(iter(resumed.train_dataloader()))
    assert (
        first_pass["canonical_puzzle_names"] == restarted_pass["canonical_puzzle_names"]
    )
    with pytest.raises(TypeError, match="timer_epoch"):
        resumed.load_state_dict({})

    wrong = ArcParquetData.Config()
    wrong.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    wrong.eval_source = path
    wrong.num_tasks = 2
    with pytest.raises(ValueError, match="num_tasks"):
        wrong.make()


def test_train_tail_eval_rank_padding_and_cpu_collation(tmp_path: Path) -> None:
    """Drop incomplete train tails and track padded eval tails on CPU."""
    path = _write(
        tmp_path / "tails.parquet",
        rows=[
            _row(
                str(index),
                train=[_pair([[index]], out=[[index]])],
                test=[_pair([[index]], out=[[index]])],
                augs=[""],
            )
            for index in range(5)
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.batch_size = 2
    cfg.eval_batch_size = 2
    cfg.num_workers = 0
    data = cfg.make()
    object.__setattr__(data, "device", torch.device("meta"))

    train_batches = list(data.train_dataloader())
    eval_batches = list(data.eval_dataloader())

    assert sum(batch["valid_count"] for batch in train_batches) == 4
    assert train_batches[0]["media"].device.type == "cpu"
    assert "label" in train_batches[0]
    assert train_batches[0]["label"].device.type == "cpu"
    assert [batch["valid_count"] for batch in eval_batches] == [2, 2, 1]
    assert eval_batches[-1]["canonical_puzzle_names"] == ["4"]


@pytest.mark.parametrize("pair_count", [0, 1])
def test_train_loader_rejects_no_complete_rank_local_batch(
    tmp_path: Path,
    pair_count: int,
) -> None:
    """Reject rank-local train indexes with no complete batch."""
    path = _write(
        tmp_path / f"short-{pair_count}.parquet",
        rows=[
            _row(
                "short",
                train=[_pair([[1]], out=[[1]])] if pair_count else [],
                test=[],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    cfg.batch_size = 2
    cfg.num_workers = 0
    data = cfg.make()

    with pytest.raises(ValueError, match="no complete rank-local training batch"):
        data.train_dataloader()
    assert list(data.eval_dataloader()) == []


def test_bind_step_passes_id_indexed_instance_metadata(tmp_path: Path) -> None:
    """Bind semantic metadata indexed by puzzle instance ID to the step."""
    path = _write(
        tmp_path / "bind.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    data = cfg.make()

    class Receiver:
        metadata: object | None = None

        def bind_instance_metadata(self, metadata: object) -> None:
            self.metadata = metadata

    receiver = Receiver()
    data.bind_step(cast(TrainStepProtocol, receiver))

    assert receiver.metadata is data.instance_metadata


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("canvas_size", 0, "Canvas and batch sizes"),
        ("num_workers", -1, "num_workers"),
        ("p_translate", 1.1, "p_translate"),
        ("p_translate", float("nan"), "p_translate"),
        ("rank", 0, "rank and num_replicas"),
    ],
)
def test_finalize_rejects_invalid_runtime_controls(
    field_name: str,
    value: object,
    message: str,
) -> None:
    """Keep scalar values printable at finalize and reject them before parquet I/O."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path="train.parquet")]
    cfg.eval_source = "eval.parquet"
    setattr(cfg, field_name, value)

    finalized = cfg.finalize()
    if field_name == "canvas_size":
        assert finalized.canvas_size == value
    elif field_name == "num_workers":
        assert finalized.num_workers == value
    elif field_name == "p_translate":
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(finalized.p_translate)
        else:
            assert finalized.p_translate == value
    else:
        assert finalized.rank == value

    with pytest.raises(ValueError, match=message):
        finalized.make()


def test_arc_parquet_rejects_canvas_larger_than_hash_shape_byte() -> None:
    """Reject an unrepresentable canvas before opening any parquet source."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path="missing-train.parquet")]
    cfg.eval_source = "missing-eval.parquet"
    cfg.canvas_size = 256

    with pytest.raises(ValueError, match="canvas_size must be at most 255"):
        cfg.finalize().make()


def test_finalize_limits_compact_source_slots_before_io() -> None:
    """Reject unrepresentable source slots in runtime construction before I/O."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [
        PairSource.Config(path=f"source-{index}.parquet") for index in range(257)
    ]
    cfg.eval_source = "eval.parquet"

    finalized = cfg.finalize()
    with pytest.raises(ValueError, match="At most 256 ordered train sources"):
        finalized.make()


def test_source_and_data_configuration_validation(tmp_path: Path) -> None:
    """Validate source policies and configured runtime counts."""
    invalid_selection = PairSource.Config()
    object.__setattr__(invalid_selection, "pair_selection", "eval")
    invalid_selection.finalize()
    invalid = ArcParquetData.Config()
    invalid.train_sources = [invalid_selection]
    invalid.eval_source = "missing.parquet"
    with pytest.raises(ValueError, match="pair_selection"):
        invalid.make()

    invalid_multiplicity = PairSource.Config()
    invalid_multiplicity.multiplicity = 0
    invalid_multiplicity.finalize()
    invalid = ArcParquetData.Config()
    invalid.train_sources = [invalid_multiplicity]
    invalid.eval_source = "missing.parquet"
    with pytest.raises(ValueError, match="multiplicity"):
        invalid.make()

    empty_config = ArcParquetData.Config().finalize()
    with pytest.raises(ValueError, match="At least one"):
        empty_config.make()

    path = _write(
        tmp_path / "count.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    wrong_instances = ArcParquetData.Config()
    wrong_instances.train_sources = [PairSource.Config(path=path)]
    wrong_instances.eval_source = path
    wrong_instances.num_instances = 2
    with pytest.raises(ValueError, match="num_instances"):
        wrong_instances.make()


def test_eval_only_tasks_join_shared_semantic_id_map(tmp_path: Path) -> None:
    """Assign eval-only tasks IDs after the ordered training task map."""
    train_path = _write(
        tmp_path / "train-only.parquet",
        rows=[_row("train-task", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    eval_path = _write(
        tmp_path / "eval-only.parquet",
        rows=[_row("eval-task", train=[], test=[_pair([[3]], out=[[4]])], augs=[""])],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=train_path)]
    cfg.eval_source = eval_path
    cfg.eval_batch_size = 1
    cfg.num_workers = 0
    data = cfg.make()

    batches = list(data.eval_dataloader())

    assert data.num_tasks == 2
    assert data.task_to_id == {"train-task": 0, "eval-task": 1}
    assert data.instance_metadata[2] == ArcInstanceMetadata(
        puzzle_name="eval-task",
        task_id=1,
        dih_idx=0,
        colorperm_slots=tuple(range(9)),
    )
    assert batches[0]["puzzle_idxs"].tolist() == [1]


def test_sample_rng_is_injected_per_access(tmp_path: Path) -> None:
    """Use the injected sample RNG factory on each dataset access."""
    path = _write(
        tmp_path / "rng.parquet",
        rows=[
            _row(
                "one",
                train=[_pair([[1]], out=[[2]])],
                test=[_pair([[3]], out=[[4]])],
                augs=[""],
            ),
        ],
    )
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    cfg.p_translate = 0.0
    rng_indices: list[int] = []

    def rng_for_sample(index: int) -> np.random.RandomState:
        rng_indices.append(index)
        return np.random.RandomState(index)

    data = ArcParquetData(
        cfg.finalize(),
        rng_for_sample=rng_for_sample,
    )
    data.train_dataset[0]
    data.train_dataset[0]
    data.eval_dataset[0]

    assert rng_indices == [0, 0, 0]


def test_distributed_defaults_and_invalid_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read initialized process-group coordinates and reject invalid ranks."""
    path = _write(
        tmp_path / "distributed.parquet",
        rows=[_row("one", train=[_pair([[1]], out=[[2]])], test=[], augs=[""])],
    )
    distributed = torch.distributed
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path)]
    cfg.eval_source = path
    data = cfg.make()
    assert (data.rank, data.num_replicas) == (1, 2)

    invalid = ArcParquetData.Config()
    invalid.train_sources = [PairSource.Config(path=path)]
    invalid.eval_source = path
    invalid.rank = 2
    invalid.num_replicas = 2
    with pytest.raises(ValueError, match="rank/num_replicas"):
        invalid.make()


def test_finalize_resolves_every_path_under_base_dir_without_io() -> None:
    """Resolve all source paths under base_dir during I/O-free finalize."""
    cfg = ArcParquetData.Config()
    cfg.base_dir = "/datasets/data-root"
    cfg.train_sources = [
        PairSource.Config(path="/datasets/train.parquet", pair_selection="both"),
    ]
    cfg.eval_source = "/datasets/eval.parquet"
    resolved = cfg.copy_tree().finalize()

    assert resolved.train_sources[0].path == Path(
        "/datasets/data-root/datasets/train.parquet",
    )
    assert resolved.eval_source == Path("/datasets/data-root/datasets/eval.parquet")


def _write_parquet(path: Path, *, rows: list[dict[str, object]]) -> None:
    parquet.write_table(pa.Table.from_pylist(rows), path)


def _write(path: Path, *, rows: list[dict[str, object]]) -> Path:
    parquet.write_table(pa.Table.from_pylist(rows), path)
    return path


def _row(
    name: str,
    *,
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    augs: list[str],
) -> dict[str, object]:
    return {
        "format_version": 2,
        "puzzle_id": name,
        "train": train,
        "test": test,
        "offline_augs": augs,
    }


def _pair(inp: list[list[int]], *, out: list[list[int]]) -> dict[str, object]:
    return {"input": inp, "output": out}


def _configured_data(train_path: Path, *, eval_path: Path) -> ArcParquetData:
    cfg = ArcParquetData.Config()
    cfg.train_sources = [
        PairSource.Config(path=train_path, pair_selection="train"),
        PairSource.Config(path=train_path, pair_selection="both", multiplicity=2),
    ]
    cfg.eval_source = eval_path
    cfg.batch_size = 2
    cfg.eval_batch_size = 2
    cfg.num_workers = 0
    cfg.p_translate = 0.0
    return cfg.make()


def _eval_task_ids(path: Path, *, rank: int) -> list[int]:
    """Return valid evaluation task ids from one DistributedSampler rank."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.rank = rank
    cfg.num_replicas = 2
    cfg.eval_batch_size = 4
    cfg.num_workers = 0
    data = cfg.make()
    return [
        task_id
        for batch in data.eval_dataloader()
        for task_id in from_plain(
            batch["puzzle_idxs"].tolist()[: batch["valid_count"]],
            list[int],
        )
    ]


def _make_rank_data(path: Path, *, rank: int) -> ArcParquetData:
    """Build one seeded training sampler shard for the requested rank."""
    cfg = ArcParquetData.Config()
    cfg.train_sources = [PairSource.Config(path=path, pair_selection="train")]
    cfg.eval_source = path
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.rank = rank
    cfg.num_replicas = 2
    cfg.seed = 17
    cfg.p_translate = 0.0
    return cfg.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
