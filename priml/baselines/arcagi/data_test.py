"""Tests for ARC data loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast
from unittest.mock import Mock

import json
import logging

from torch import Tensor
from treekle import ReadError

import numpy as np
import pytest
import torch

from priml.baselines.arcagi.data import (
    PuzzleBatches,
    PuzzleData,
    _load_int32,
    load_puzzle_dataset,
    resolve_rank,
)
from priml.data.arrays import load_array

import priml.baselines.arcagi.data as arc_data


if TYPE_CHECKING:
    from pathlib import Path


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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
