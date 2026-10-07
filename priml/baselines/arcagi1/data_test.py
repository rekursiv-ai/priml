"""Tests for ARC data loading."""

from __future__ import annotations

from importlib import util
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import Mock

import json
import logging

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.data import (
    ArcData,
    PuzzleBatches,
    PuzzleData,
    _ArcBatches,
    _identity_tags,
    _int_list,
    _last,
    _load_int32,
    _load_split,
    _rows_tensor,
    load_puzzle_dataset,
    resolve_rank,
)
from priml.lib.codec import from_plain

import priml.baselines.arcagi1.data as arc_data


if TYPE_CHECKING:
    from collections.abc import Iterator
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


def _data(dataset_dir: Path, **overrides: object) -> ArcData:
    config = ArcData.Config()
    config.base_dir = "/"
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


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


def _write_arc_split(root: Path, name: str = "train") -> None:
    split = root / name
    split.mkdir(parents=True)
    (split / "dataset.json").write_text(json.dumps({"ignore_label_id": 0}))
    rows = np.arange(24, dtype=np.int32).reshape(6, 4) + 2
    np.save(split / "all__inputs.npy", rows)
    labels = rows.copy()
    labels[0, 0] = 0
    np.save(split / "all__labels.npy", labels)
    np.save(split / "all__puzzle_indices.npy", np.array([0, 2, 4, 6]))
    np.save(split / "all__group_indices.npy", np.array([0, 1, 3]))
    np.save(split / "all__puzzle_identifiers.npy", np.array([7, 8, 9]))


def _arc_data(
    root: Path,
    *,
    batch_size: int = 2,
    eval_batch_size: int | None = None,
    num_tasks: int | None = None,
    num_eval_tasks: int | None = None,
    rank: int = -1,
    num_replicas: int = -1,
    seed: int = 0,
    epochs_per_iter: int = 1,
    device_resident: bool = True,
) -> ArcData:
    return ArcData.Config(
        working_dir=root,
        device="cpu",
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_tasks=num_tasks,
        num_eval_tasks=num_eval_tasks,
        rank=rank,
        num_replicas=num_replicas,
        seed=seed,
        epochs_per_iter=epochs_per_iter,
        device_resident=device_resident,
    ).make()


def test_batches_carry_the_puzzle_identity(dataset_dir: Path) -> None:
    """The per-task prefix and the metric both key on it."""
    data = _data(dataset_dir)
    train_loader = data.train_dataloader()
    eval_loader = data.eval_dataloader()
    assert isinstance(train_loader.inputs, Tensor)
    assert isinstance(eval_loader.inputs, Tensor)
    batch = next(iter(train_loader))
    assert _tensor(batch["puzzle_identifiers"]).shape == (4,)
    assert _tensor(batch["media"]).shape == (4, GRID)
    assert torch.equal(
        _tensor(batch["spatial_tags"]),
        torch.tensor([[1, 0, 0]] * 4, dtype=torch.int64),
    )


def test_arc_config_propagates_loader_options(dataset_dir: Path) -> None:
    data = _data(
        dataset_dir,
        batch_size=4,
        eval_batch_size=2,
        num_tasks=2,
        num_eval_tasks=1,
        epochs_per_iter=2,
        device_resident=False,
        seed=13,
    )
    train = data.train_dataloader()
    eval_loader = data.eval_dataloader()

    assert train.batch_size == 4
    assert train.global_batch_size == 4
    assert train.num_tasks == 2
    assert train.epochs_per_iter == 2
    assert train.seed == 13
    assert isinstance(train.inputs, np.memmap)
    assert train.device == torch.device("cpu")
    assert eval_loader.batch_size == 2
    assert eval_loader.sample_by_task is False
    assert eval_loader.num_tasks == 1
    assert eval_loader.seed == 13
    assert eval_loader.passes == 0
    assert eval_loader.rank == 0
    assert eval_loader.num_replicas == 1
    assert eval_loader.epochs_per_iter == 1
    assert isinstance(eval_loader.inputs, np.memmap)


def test_training_draws_whole_tasks(dataset_dir: Path) -> None:
    """Views of one puzzle arrive together, so a batch votes coherently.

    Sampling rows uniformly would over-weight tasks with more puzzles; the
    benchmark weights every task equally.
    """
    batch = next(iter(_data(dataset_dir).train_dataloader()))
    puzzle_identifiers_raw = _tensor(batch["puzzle_identifiers"])
    identifiers = puzzle_identifiers_raw.tolist()
    # Four slots at three views per puzzle: at most two puzzles can appear.
    assert len(set(identifiers)) <= 2
    row_ids = _tensor(batch["media"])[:, 0] - 2
    assert identifiers == [int(row_id) // VIEWS_PER_PUZZLE for row_id in row_ids]


def test_eval_walks_every_row_in_order(dataset_dir: Path) -> None:
    """pass@K votes across views, so evaluation must not sample."""
    rows = list(_data(dataset_dir).eval_dataloader())
    total = TASKS * PUZZLES_PER_TASK * VIEWS_PER_PUZZLE
    media = torch.cat(
        [_tensor(batch["media"])[: _int(batch["valid_count"])] for batch in rows],
    )
    labels = torch.cat(
        [_tensor(batch["label"])[: _int(batch["valid_count"])] for batch in rows],
    )
    identifiers = torch.cat(
        [
            _tensor(batch["puzzle_identifiers"])[: _int(batch["valid_count"])]
            for batch in rows
        ],
    )
    tags = torch.cat(
        [
            _tensor(batch["spatial_tags"])[: _int(batch["valid_count"])]
            for batch in rows
        ],
    )
    expected_media = (
        (torch.arange(total) + 2)
        .unsqueeze(1)
        .expand(
            total,
            GRID,
        )
    )
    assert torch.equal(media, expected_media)
    assert torch.equal(labels, expected_media)
    assert identifiers.tolist() == [puzzle for puzzle in range(8) for _ in range(3)]
    assert tags.tolist() == [[1, 0, 0]] * total
    # Twice through gives the same order.
    again = torch.cat(
        [
            _tensor(batch["media"])[: _int(batch["valid_count"])]
            for batch in _data(dataset_dir).eval_dataloader()
        ],
    )
    assert torch.equal(media, again)


def test_short_final_batch_is_padded(dataset_dir: Path) -> None:
    """Shapes stay constant, and the padding is reported not hidden."""
    batches = list(_data(dataset_dir, batch_size=7).eval_dataloader())
    assert all(_tensor(b["media"]).shape == (7, 12) for b in batches)
    assert _int(batches[-1]["valid_count"]) < 7
    tail = batches[-1]
    valid = _int(tail["valid_count"])
    assert _tensor(tail["media"])[valid:].eq(0).all()
    assert _tensor(tail["label"])[valid:].eq(-100).all()
    assert _tensor(tail["puzzle_identifiers"])[valid:].eq(0).all()
    assert _tensor(tail["spatial_tags"])[valid:].tolist() == [[1, 0, 0]] * (7 - valid)


def test_the_skipped_cell_marker_is_remapped(dataset_dir: Path) -> None:
    """The loss and the halt target both key on -100, so remap once here."""
    batch = next(iter(_data(dataset_dir).eval_dataloader()))
    assert int(_tensor(batch["label"]).min()) >= 2  # Nothing was 0 to remap here.
    # A padded row carries the marker.
    padded = list(_data(dataset_dir, batch_size=7).eval_dataloader())[-1]
    assert -100 in from_plain(_tensor(padded["label"])[-1].tolist(), list[int])


def test_sampling_is_reproducible_and_advances(dataset_dir: Path) -> None:
    """One seed replays; consecutive passes differ."""
    first = _tensor(next(iter(_data(dataset_dir, seed=5).train_dataloader()))["media"])
    again = _tensor(next(iter(_data(dataset_dir, seed=5).train_dataloader()))["media"])
    assert torch.equal(first, again)

    data = _data(dataset_dir, seed=5)
    loader = data.train_dataloader()
    pass_one = _tensor(next(iter(loader))["media"]).clone()
    pass_two = _tensor(next(iter(loader))["media"]).clone()
    assert not torch.equal(pass_one, pass_two)


def test_arc_data_load_state_updates_live_pass_count(dataset_dir: Path) -> None:
    data = _data(dataset_dir)
    loader = data.train_dataloader()
    next(iter(loader))

    data.load_state_dict({"passes": 4})

    state = data.state_dict()
    assert state["passes"] == 4
    assert state["loader"] == {
        "passes": 4,
        "active_pass": 0,
        "next_batch": 1,
    }


def test_arc_batch_loader_defaults_missing_checkpoint_fields(dataset_dir: Path) -> None:
    loader = _data(dataset_dir).train_dataloader()
    loader.load_state_dict({"passes": 5, "active_pass": 3, "next_batch": 7})

    loader.load_state_dict({})

    assert loader.state_dict() == {
        "passes": 5,
        "active_pass": None,
        "next_batch": 0,
    }


def test_arc_data_checkpoint_resumes_live_loader(dataset_dir: Path) -> None:
    source = _data(dataset_dir, seed=9)
    source_loader = source.train_dataloader()
    iterator = iter(source_loader)
    next(iterator)
    state = source.state_dict()
    expected = [_tensor(batch["media"]).clone() for batch in iterator]

    restored = _data(dataset_dir, seed=9)
    restored.load_state_dict(state)
    observed = [
        _tensor(batch["media"]).clone() for batch in restored.train_dataloader()
    ]

    assert state["passes"] == 1
    assert state["loader"] == {
        "passes": 1,
        "active_pass": 0,
        "next_batch": 1,
    }
    assert len(observed) == len(expected)
    assert all(
        torch.equal(left, right) for left, right in zip(observed, expected, strict=True)
    )


def test_pass_counter_round_trips(dataset_dir: Path) -> None:
    """Resume continues the sampling sequence rather than replaying it."""
    data = _data(dataset_dir, seed=1)
    loader = data.train_dataloader()
    list(loader)
    state = data.state_dict()
    assert state["passes"] == 1

    restored = _data(dataset_dir, seed=1)
    restored.load_state_dict(state)
    assert torch.equal(
        _tensor(next(iter(restored.train_dataloader()))["media"]),
        _tensor(next(iter(loader))["media"]),
    )


def test_checkpoint_resumes_the_unfinished_sampled_pass(dataset_dir: Path) -> None:
    data = _data(dataset_dir, seed=3)
    loader = data.train_dataloader()
    iterator = iter(loader)
    next(iterator)
    state = data.state_dict()
    expected = [_tensor(batch["media"]).clone() for batch in iterator]

    restored = _data(dataset_dir, seed=3)
    restored.load_state_dict(state)
    observed = [
        _tensor(batch["media"]).clone() for batch in restored.train_dataloader()
    ]

    assert len(observed) == len(expected)
    assert all(
        torch.equal(observed_batch, expected_batch)
        for observed_batch, expected_batch in zip(observed, expected, strict=True)
    )


def test_seed_and_pass_are_distinct_named_stream_inputs(dataset_dir: Path) -> None:
    later_pass = _data(dataset_dir, seed=4).train_dataloader()
    list(later_pass)
    later = [_tensor(batch["media"]).clone() for batch in later_pass]
    first = [
        _tensor(batch["media"]).clone()
        for batch in _data(dataset_dir, seed=5).train_dataloader()
    ]

    assert any(
        not torch.equal(later_batch, first_batch)
        for later_batch, first_batch in zip(later, first, strict=True)
    )


def test_task_cap_trims_whole_tasks(dataset_dir: Path) -> None:
    data = _data(dataset_dir, num_eval_tasks=2)
    rows = sum(_int(b["valid_count"]) for b in data.eval_dataloader())
    assert rows == 2 * PUZZLES_PER_TASK * VIEWS_PER_PUZZLE


def test_len_counts_the_batches_the_loader_actually_yields(
    dataset_dir: Path,
) -> None:
    """``len`` must match iteration on both paths, not just the ordered one."""
    data = _data(dataset_dir)
    for name, loader in (
        ("eval", data.eval_dataloader()),
        ("train", data.train_dataloader()),
    ):
        assert len(list(loader)) == len(loader), name


def test_sampled_len_handles_variable_puzzle_sizes(dataset_dir: Path) -> None:
    """The sampled pass plan, not one puzzle-size statistic, determines length."""
    train = dataset_dir / "train"
    np.save(train / "all__puzzle_indices.npy", np.array([0, 1, 5, 6, 10]))
    np.save(train / "all__group_indices.npy", np.arange(4 + 1))
    np.save(train / "all__puzzle_identifiers.npy", np.arange(4))

    loader = _data(dataset_dir).train_dataloader()

    assert len(loader) == len(list(loader))


def test_five_pass_global_plan_shards_to_memory_mapped_rank_batches(
    dataset_dir: Path,
) -> None:
    """Eight-GPU ARC uses one global task order, then slices each rank's rows."""
    global_loader = _data(
        dataset_dir,
        batch_size=4,
        epochs_per_iter=5,
        device_resident=False,
    ).train_dataloader()
    rank_loaders = [
        _data(
            dataset_dir,
            batch_size=2,
            rank=rank,
            num_replicas=2,
            epochs_per_iter=5,
            device_resident=False,
        ).train_dataloader()
        for rank in (0, 1)
    ]
    assert isinstance(global_loader.inputs, np.memmap)
    whole = list(global_loader)
    left, right = (list(loader) for loader in rank_loaders)
    assert len(whole) == len(left) == len(right)
    assert len(whole) > len(
        list(_data(dataset_dir, batch_size=4).train_dataloader()),
    )
    for full, first, second in zip(whole, left, right, strict=True):
        assert torch.equal(
            _tensor(full["media"]),
            torch.cat([_tensor(first["media"]), _tensor(second["media"])]),
        )


def test_eval_shards_keep_a_zero_valid_tail_rank(dataset_dir: Path) -> None:
    """Every rank yields the same number of eval batches for DDP collectives."""
    first = list(
        _data(dataset_dir, batch_size=5, rank=0, num_replicas=2).eval_dataloader(),
    )
    second = list(
        _data(dataset_dir, batch_size=5, rank=1, num_replicas=2).eval_dataloader(),
    )
    assert len(first) == len(second) == 3
    assert _int(first[-1]["valid_count"]) == 4
    assert _int(second[-1]["valid_count"]) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("eval_batch_size", 0),
        ("eval_batch_size", -1),
    ],
)
def test_nonpositive_batch_size_is_rejected(
    dataset_dir: Path,
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match=field):
        _data(dataset_dir, **{field: value})


def test_unit_batch_sizes_are_valid(dataset_dir: Path) -> None:
    # ArcData validates strictly positive sizes; one-example batches are valid.
    data = _data(dataset_dir, batch_size=1, eval_batch_size=1)
    assert _tensor(next(iter(data.train_dataloader()))["media"]).shape == (1, GRID)
    assert _tensor(next(iter(data.eval_dataloader()))["media"]).shape == (1, GRID)


def test_zero_epochs_per_iter_is_rejected(dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="epochs_per_iter must be positive"):
        _data(dataset_dir, epochs_per_iter=0)
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.epochs_per_iter = 0
    with pytest.raises(ValueError, match="epochs_per_iter must be positive"):
        config.make()


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        _data(tmp_path).train_dataloader()


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
    labels = cast(np.ndarray, np.load(train / "all__labels.npy"))
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


def test_load_split_returns_resident_or_memory_mapped_arrays(
    dataset_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    split_dir = dataset_dir / "test"
    for name in ("inputs", "labels"):
        path = split_dir / f"all__{name}.npy"
        np.save(path, cast(np.ndarray, np.load(path)).astype(np.int16))
    resident = _load_split(dataset_dir, split="test")
    mapped = _load_split(dataset_dir, split="test", mmap=True)

    assert isinstance(resident["inputs"], Tensor)
    assert isinstance(resident["labels"], Tensor)
    assert resident["inputs"].dtype == resident["labels"].dtype == torch.int32
    assert isinstance(mapped["inputs"], np.memmap)
    assert isinstance(mapped["labels"], np.memmap)
    assert mapped["inputs"].dtype == mapped["labels"].dtype == np.int16
    mapped_inputs = cast(
        "np.ndarray[tuple[int, ...], np.dtype[np.int16]]",
        mapped["inputs"],
    )
    assert np.array_equal(
        cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.int16]]",
            mapped_inputs[0],
        ),
        cast(
            "np.ndarray[tuple[int, ...], np.dtype[np.int16]]",
            np.full(GRID, 2, dtype=np.int16),
        ),
    )
    assert resident["ignore_label_id"] == 0
    assert [record.getMessage() for record in caplog.records] == [
        f"loading ARC split 'test' from {dataset_dir / 'test'}",
        "ARC 'test': 24 rows, 8 puzzles, 4 tasks",
        f"loading ARC split 'test' from {dataset_dir / 'test'}",
        "ARC 'test': 24 rows, 8 puzzles, 4 tasks",
    ]
    assert resident["spatial_tags"].dtype == np.int64
    assert np.array_equal(
        resident["spatial_tags"],
        np.tile(np.array([1, 0, 0], dtype=np.int64), (TASKS * PUZZLES_PER_TASK, 1)),
    )


def test_load_split_preserves_spatial_sidecar(dataset_dir: Path) -> None:
    expected = np.tile(
        np.array([4, 2, 3], dtype=np.int32),
        (TASKS * PUZZLES_PER_TASK, 1),
    )
    np.save(dataset_dir / "test" / "all__spatial_tags.npy", expected)

    result = _load_split(dataset_dir, split="test")

    assert result["spatial_tags"].dtype == np.int64
    assert result["spatial_tags"].tolist() == [[4, 2, 3]] * (TASKS * PUZZLES_PER_TASK)


def test_load_split_defaults_missing_ignore_label(dataset_dir: Path) -> None:
    metadata_path = dataset_dir / "test" / "dataset.json"
    metadata = from_plain(
        cast(object, json.loads(metadata_path.read_text())),
        dict[str, object],
    )
    del metadata["ignore_label_id"]
    metadata_path.write_text(json.dumps(metadata))

    assert _load_split(dataset_dir, split="test")["ignore_label_id"] == 0


def test_g11_load_split_requests_platform_independent_tag_dtype(
    dataset_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np_array = Mock(wraps=np.array)
    monkeypatch.setattr(np, "array", np_array)

    loaded = _load_split(dataset_dir, split="test")

    assert loaded["spatial_tags"].dtype == np.int64
    assert any(
        call.args == ([1, 0, 0],) and call.kwargs == {"dtype": np.int64}
        for call in np_array.call_args_list
    )


def test_load_split_preserves_nonzero_ignore_label(dataset_dir: Path) -> None:
    metadata_path = dataset_dir / "test" / "dataset.json"
    metadata = from_plain(
        cast(object, json.loads(metadata_path.read_text())),
        dict[str, object],
    )
    metadata["ignore_label_id"] = 7
    metadata_path.write_text(json.dumps(metadata))

    assert _load_split(dataset_dir, split="test")["ignore_label_id"] == 7


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


def test_load_split_missing_metadata_reports_exact_build_command(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train"
    split.mkdir()

    with pytest.raises(FileNotFoundError) as error:
        _load_split(tmp_path, split="train")

    assert str(error.value) == (
        f"no prepared ARC data at {split}; build it with `uv --quiet run --frozen "
        "python -m priml.baselines.arcagi1.scripts.prepare_data`."
    )
    # The command must name a module that imports, or the hint is a dead end.
    module = str(error.value).rpartition("python -m ")[2].rstrip("`.")
    assert util.find_spec(module) is not None


def test_puzzle_data_eval_subsets_and_rejects_competing_caps(dataset_dir: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    config.eval_max_augs_per_puzzle = 1
    data = config.make()

    batches = list(data.eval_dataloader())

    assert sum(batch["valid_count"] for batch in batches) == TASKS * PUZZLES_PER_TASK
    assert all(batch["valid_count"] == 4 for batch in batches)
    assert len(data.eval_dataloader()) == len(batches)
    selected = torch.cat(
        [_tensor(batch["media"])[: _int(batch["valid_count"]), 0] for batch in batches],
    )
    identifiers = torch.cat(
        [
            _tensor(batch["puzzle_identifiers"])[: _int(batch["valid_count"])]
            for batch in batches
        ],
    )
    assert identifiers.tolist() == list(range(8))
    assert all(
        3 * int(identifier) + 2 <= int(value) <= 3 * int(identifier) + 4
        for value, identifier in zip(selected, identifiers, strict=True)
    )
    data.config.eval_max_augs_per_puzzle = 0
    with pytest.raises(ValueError, match="max_augs_per_puzzle must be positive"):
        data.eval_dataloader()
    data.config.eval_max_augs_per_puzzle = 1
    data.config.eval_max_examples_per_group = 2
    with pytest.raises(ValueError, match="mutually exclusive"):
        data.eval_dataloader()


def test_puzzle_data_group_subset_and_full_eval_ignore_caps(dataset_dir: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    config.eval_max_examples_per_group = 2
    config.max_samples = 3
    data = config.make()

    with pytest.raises(ValueError, match="mutually exclusive"):
        data.eval_dataloader()
    data.config.max_samples = None
    proxy_batches = list(data.eval_dataloader())
    assert sum(batch["valid_count"] for batch in proxy_batches) == TASKS * 2
    assert torch.cat(
        [
            _tensor(batch["media"])[: _int(batch["valid_count"]), 0]
            for batch in proxy_batches
        ],
    ).tolist() == [2, 7, 8, 13, 14, 19, 20, 25]
    batches = list(data.full_eval_dataloader())
    assert sum(batch["valid_count"] for batch in batches) == (
        TASKS * PUZZLES_PER_TASK * VIEWS_PER_PUZZLE
    )


def test_puzzle_data_group_cap_boundary_and_zero_are_distinct(
    dataset_dir: Path,
) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.eval_max_examples_per_group = 6
    data = config.make()

    batches = list(data.eval_dataloader())

    assert sum(batch["valid_count"] for batch in batches) == 24
    assert torch.cat(
        [_tensor(batch["media"])[: _int(batch["valid_count"]), 0] for batch in batches],
    ).tolist() == [value + 2 for value in range(24)]
    data.config.eval_max_examples_per_group = 0
    with pytest.raises(ValueError, match="max_examples_per_group must be positive"):
        data.eval_dataloader()


def test_puzzle_data_rank_past_eval_tail_emits_empty_batch(dataset_dir: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 5
    config.rank = 1
    config.num_replicas = 2
    config.eval_max_samples = 12
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


def test_small_array_helpers_preserve_exact_values_and_dtype() -> None:
    values = np.array([2, 5, 9], dtype=np.int32)

    assert _last(values) == 9
    assert _int_list(values) == [2, 5, 9]
    identity = _identity_tags(2, device=torch.device("cpu"))
    assert identity.tolist() == [[1, 0, 0], [1, 0, 0]]
    meta = _identity_tags(2, device=torch.device("meta"))
    assert meta.shape == (2, 3)
    assert meta.dtype == torch.int64
    assert meta.device == torch.device("meta")


def test_rows_tensor_copies_selected_rows_as_int32() -> None:
    source = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)

    result = _rows_tensor(
        source,
        np.array([2, 0], dtype=np.int64),
        np.int32,
        torch.device("cpu"),
    )

    assert result.dtype == torch.int32
    assert result.device == torch.device("cpu")
    assert result.tolist() == [[5, 6], [1, 2]]
    source[2, 0] = 9
    assert result.tolist() == [[5, 6], [1, 2]]
    meta = _rows_tensor(
        source,
        np.array([0, 2], dtype=np.int64),
        np.int32,
        torch.device("meta"),
    )
    assert meta.device == torch.device("meta")
    assert meta.dtype == torch.int32


def test_int32_loader_preserves_mmap_request(tmp_path: Path) -> None:
    path = tmp_path / "values.npy"
    np.save(path, np.array([[2, 3]], dtype=np.int32))

    assert type(_load_int32(path)) is np.ndarray
    assert isinstance(_load_int32(path, mmap=True), np.memmap)


def test_puzzle_batch_identifier_remap_preserves_blank_and_offsets(
    dataset_dir: Path,
) -> None:
    batches = PuzzleBatches(
        dataset_dir=dataset_dir,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=0,
        puzzle_identifier_offset=20,
    )
    batches.puzzle_identifiers[:2] = [0, 1]
    batch = batches._collate(
        np.array([0, 3], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )
    assert batch["puzzle_identifiers"].tolist() == [0, 21]
    assert batch["puzzle_identifiers"].dtype == torch.int64

    remapped_batches = PuzzleBatches(
        dataset_dir=dataset_dir,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=0,
        puzzle_identifier_offset=20,
        puzzle_identifier_remap=np.arange(32, dtype=np.int64) + 50,
    )
    remapped_batches.puzzle_identifiers[:2] = [0, 1]
    remapped = remapped_batches._collate(
        np.array([0, 3], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )
    assert remapped["puzzle_identifiers"].tolist() == [50, 51]


def test_puzzle_subset_uses_seeded_one_view_per_puzzle(dataset_dir: Path) -> None:
    batches = PuzzleBatches(
        dataset_dir=dataset_dir,
        device="cpu",
        batch_size=4,
        rank=0,
        num_replicas=1,
        train=False,
        seed=0,
        max_augs_per_puzzle=1,
    )
    assert batches.eval_example_index is not None
    assert batches.eval_example_index.tolist() == [2, 3, 8, 9, 13, 15, 18, 23]


def test_group_subset_skips_empty_tasks_and_samples_one_row_per_task(
    dataset_dir: Path,
) -> None:
    test = dataset_dir / "test"
    np.save(
        test / "all__puzzle_indices.npy",
        np.array([0, 0, 0, 3, 6, 9, 10, 13, 15], dtype=np.int32),
    )
    batches = PuzzleBatches(
        dataset_dir=dataset_dir,
        device="cpu",
        batch_size=4,
        rank=0,
        num_replicas=1,
        train=False,
        seed=0,
        max_examples_per_group=1,
    )
    assert batches.eval_example_index is not None
    assert batches.eval_example_index.tolist() == [0, 6, 10]
    assert batches.n_examples == 3
    assert len(batches) == 1


def test_group_subset_linspace_matches_previous_selection(tmp_path: Path) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    loader.puzzle_indices = np.array([0, 0, 1, 3, 5, 8, 8, 10], dtype=np.int64)
    loader.group_indices = np.array([0, 2, 2, 4, 6, 7], dtype=np.int64)

    selected = loader._build_eval_group_subset(2)

    assert selected.tolist() == [0, 1, 4, 5, 7, 8, 9]


def test_puzzle_group_subset_reports_exact_nonpositive_cap_error(
    tmp_path: Path,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)

    with pytest.raises(
        ValueError,
        match="max_examples_per_group must be positive",
    ) as error:
        loader._build_eval_group_subset(0)

    assert str(error.value) == "max_examples_per_group must be positive, got 0."


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
    groups_before = cast(np.ndarray, np.load(group_path)).copy()

    capped = load_puzzle_dataset(tmp_path, "train", max_samples=7)

    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 7]
    assert capped["group_indices"].tolist() == [0, 2, 3]
    assert np.array_equal(cast(np.ndarray, np.load(group_path)), groups_before)


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


def test_existing_puzzle_batches_group_subset_uses_endpoints_and_identity_for_empty_groups(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    np.save(
        tmp_path / "test" / "all__group_indices.npy",
        np.array([0, 1, 4]),
    )

    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=11,
        max_examples_per_group=2,
    )

    assert loader.eval_example_index is not None
    assert loader.eval_example_index.tolist() == [0, 1, 2, 11]
    batches = list(loader)
    assert [batch["valid_count"] for batch in batches] == [2, 2]
    assert batches[0]["media"][:, 0].tolist() == [2, 6]
    assert batches[1]["media"][:, 0].tolist() == [10, 46]


def test_existing_puzzle_batches_one_per_puzzle_subset_is_seeded_and_sorted(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    first = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=0,
        num_replicas=1,
        train=False,
        seed=41,
        max_augs_per_puzzle=1,
    )
    repeated = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=0,
        num_replicas=1,
        train=False,
        seed=41,
        max_augs_per_puzzle=1,
    )

    assert first.eval_example_index is not None
    assert repeated.eval_example_index is not None
    assert first.eval_example_index.tolist() == repeated.eval_example_index.tolist()
    assert first.eval_example_index.tolist() == [0, 4, 8, 11]


def test_existing_puzzle_batches_collate_remaps_only_nonblank_ids_and_pads(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    batches = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=0,
        num_replicas=1,
        train=False,
        seed=1,
        puzzle_identifier_offset=20,
    )

    batch = batches._collate(
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )

    assert batch["puzzle_identifiers"].tolist() == [21, 22, 9]
    assert batch["spatial_tags"].tolist() == [[1, 0, 0]] * 3
    assert batch["label"][-1].tolist() == [-100] * 4
    assert batch["valid_count"] == 2


def test_existing_puzzle_data_eval_dataloader_rejects_both_proxy_cap_combinations(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    data = _small_puzzle_data(
        tmp_path,
        eval_max_augs_per_puzzle=1,
        eval_max_samples=8,
    )

    with pytest.raises(ValueError, match=r"prefix cap .* mutually exclusive"):
        data.eval_dataloader()
    data.config.eval_max_samples = None
    data.config.eval_max_examples_per_group = 2
    with pytest.raises(ValueError, match="mutually exclusive; set only one"):
        data.eval_dataloader()


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


def test_arc_data_eval_loader_uses_configured_eval_batch_and_num_tasks(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "train")
    _write_small_puzzle_split(tmp_path, "test")
    config = ArcData.Config(
        working_dir=tmp_path,
        device="cpu",
        batch_size=3,
        eval_batch_size=2,
        num_eval_tasks=2,
        seed=17,
        device_resident=False,
    )
    data = config.make()

    loader = data.eval_dataloader()

    assert loader.batch_size == 2
    assert loader.sample_by_task is False
    assert loader.num_tasks == 2
    assert loader.seed == 17
    assert loader.passes == 0
    assert isinstance(loader.inputs, np.memmap)
    batches = list(loader)
    assert [batch["valid_count"] for batch in batches] == [2, 2, 2, 2, 1]


def test_existing_arc_data_checkpoint_preserves_pending_loader_and_timer_state(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path)
    config = ArcData.Config(working_dir=tmp_path, device="cpu", batch_size=2)
    original = config.make()
    loader = original.train_dataloader()
    iterator = iter(loader)
    next(iterator)
    saved = original.state_dict()

    restored = config.make()
    restored.load_state_dict(saved)
    resumed = restored.train_dataloader()

    assert saved["passes"] == 1
    assert saved["loader"] == {"passes": 1, "active_pass": 0, "next_batch": 1}
    assert resumed.state_dict() == saved["loader"]
    assert restored.state_dict() == saved


def test_resolve_rank_accepts_only_complete_explicit_pairs() -> None:
    assert resolve_rank(-1, -1) == (0, 1)
    assert resolve_rank(2, 4) == (2, 4)
    for rank, replicas in ((-1, 1), (0, -1), (-2, 3), (3, 3)):
        with pytest.raises(ValueError, match="rank/num_replicas must both be"):
            resolve_rank(rank, replicas)


def test_arc_batches_ordered_shards_pad_and_remap_labels(tmp_path: Path) -> None:
    _write_arc_split(tmp_path, "test")
    data = _arc_data(tmp_path, eval_batch_size=4, num_eval_tasks=2)

    loader = data.eval_dataloader()
    batches = list(loader)

    assert loader.num_tasks == 2
    assert len(loader) == 2
    assert [batch["valid_count"] for batch in batches] == [4, 2]
    assert [_tensor(batch["puzzle_identifiers"]).tolist() for batch in batches] == [
        [7, 7, 8, 8],
        [9, 9, 0, 0],
    ]
    assert _tensor(batches[0]["label"]).tolist() == [
        [-100, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
    ]
    assert _tensor(batches[1]["media"]).tolist() == [
        [18, 19, 20, 21],
        [22, 23, 24, 25],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert _tensor(batches[1]["label"])[2:].tolist() == [[-100] * 4] * 2
    assert _tensor(batches[1]["spatial_tags"]).tolist() == [[1, 0, 0]] * 4


def test_arc_batches_ordered_rank_has_empty_aligned_tail(tmp_path: Path) -> None:
    _write_arc_split(tmp_path, "test")
    data = _arc_data(tmp_path, eval_batch_size=4, rank=3, num_replicas=4)

    batches = list(data.eval_dataloader())

    assert [batch["valid_count"] for batch in batches] == [0]
    assert _tensor(batches[-1]["media"]).shape == (4, 4)
    assert _tensor(batches[-1]["media"]).eq(0).all()
    assert _tensor(batches[-1]["label"]).tolist() == [[-100] * 4] * 4
    assert _tensor(batches[-1]["puzzle_identifiers"]).tolist() == [0] * 4


def test_arc_sampled_pass_checkpoint_resumes_exact_remaining_batches(
    tmp_path: Path,
) -> None:
    _write_arc_split(tmp_path)
    source = _arc_data(tmp_path, seed=19, epochs_per_iter=3)
    iterator = iter(source.train_dataloader())
    next(iterator)
    state = source.state_dict()
    expected = [_tensor(batch["media"]).clone() for batch in iterator]

    restored = _arc_data(tmp_path, seed=19, epochs_per_iter=3)
    restored.load_state_dict(state)
    observed = [
        _tensor(batch["media"]).clone() for batch in restored.train_dataloader()
    ]

    assert state["passes"] == 1
    assert state["loader"] == {"passes": 1, "active_pass": 0, "next_batch": 1}
    assert len(observed) == len(expected)
    assert all(
        torch.equal(left, right) for left, right in zip(observed, expected, strict=True)
    )


def test_arc_loader_state_defaults_and_live_pass_restore(tmp_path: Path) -> None:
    _write_arc_split(tmp_path)
    data = _arc_data(tmp_path)
    loader = data.train_dataloader()
    data.load_state_dict({"passes": 5, "loader": None})

    assert loader.passes == 5
    assert data.state_dict()["passes"] == 5
    loader.load_state_dict({})
    assert loader.state_dict() == {"passes": 5, "active_pass": None, "next_batch": 0}


def test_arc_training_task_cap_trims_rows_and_identifiers(tmp_path: Path) -> None:
    _write_arc_split(tmp_path)
    loader = _arc_data(tmp_path, num_tasks=1).train_dataloader()

    assert loader.num_tasks == 1
    assert loader.inputs.shape == loader.labels.shape == (2, 4)
    assert loader.identifiers.tolist() == [7]
    assert loader.groups.tolist() == [0, 1]
    assert loader.puzzles.tolist() == [0, 2]


def test_arc_config_rejects_nonpositive_sizes_and_epoch_count(tmp_path: Path) -> None:
    _write_arc_split(tmp_path)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _arc_data(tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="eval_batch_size must be positive"):
        _arc_data(tmp_path, eval_batch_size=0)
    with pytest.raises(ValueError, match="epochs_per_iter must be positive"):
        _arc_data(tmp_path, epochs_per_iter=0)


def _tensor(value: object) -> Tensor:
    assert isinstance(value, Tensor)
    return value


def _int(value: object) -> int:
    assert isinstance(value, int)
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
    groups_before = cast(np.ndarray, np.load(group_path)).copy()
    capped = load_puzzle_dataset(tmp_path, "train", max_samples=7)
    assert capped["puzzle_indices"].tolist() == [0, 2, 5, 7]
    assert capped["group_indices"].tolist() == [0, 2, 3]
    assert np.array_equal(cast(np.ndarray, np.load(group_path)), groups_before)


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


def test_puzzle_batches_group_subset_uses_endpoints_and_identity_for_empty_groups(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "test")
    np.save(tmp_path / "test" / "all__group_indices.npy", np.array([0, 1, 4]))
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=11,
        max_examples_per_group=2,
    )
    assert loader.eval_example_index is not None
    assert loader.eval_example_index.tolist() == [0, 1, 2, 11]
    batches = list(loader)
    assert [batch["valid_count"] for batch in batches] == [2, 2]
    assert batches[0]["media"][:, 0].tolist() == [2, 6]
    assert batches[1]["media"][:, 0].tolist() == [10, 46]


def test_puzzle_batches_one_per_puzzle_subset_is_seeded_and_sorted(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "test")
    first = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=0,
        num_replicas=1,
        train=False,
        seed=41,
        max_augs_per_puzzle=1,
    )
    repeated = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=0,
        num_replicas=1,
        train=False,
        seed=41,
        max_augs_per_puzzle=1,
    )
    assert first.eval_example_index is not None
    assert repeated.eval_example_index is not None
    assert first.eval_example_index.tolist() == repeated.eval_example_index.tolist()
    assert first.eval_example_index.tolist() == [0, 4, 8, 11]


def test_puzzle_batches_collate_remaps_only_nonblank_ids_and_pads(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "test")
    batches = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=3,
        rank=0,
        num_replicas=1,
        train=False,
        seed=1,
        puzzle_identifier_offset=20,
    )
    batch = batches._collate(
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )
    assert batch["puzzle_identifiers"].tolist() == [21, 22, 9]
    assert batch["spatial_tags"].tolist() == [[1, 0, 0]] * 3
    assert batch["label"][-1].tolist() == [-100] * 4
    assert batch["valid_count"] == 2


def test_puzzle_data_eval_dataloader_rejects_both_proxy_cap_combinations(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "test")
    data = _puzzle_data(
        tmp_path,
        eval_max_augs_per_puzzle=1,
        eval_max_samples=8,
    )
    with pytest.raises(ValueError, match=r"prefix cap .* mutually exclusive"):
        data.eval_dataloader()
    data.config.eval_max_samples = None
    data.config.eval_max_examples_per_group = 2
    with pytest.raises(ValueError, match="mutually exclusive; set only one"):
        data.eval_dataloader()


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
        max_augs_per_puzzle=None,
        max_examples_per_group=None,
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


def test_arc_data_eval_dataloader_uses_configured_eval_batch_and_num_tasks(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "train")
    _write_split(tmp_path, "test")
    config = ArcData.Config(
        working_dir=tmp_path,
        device="cpu",
        batch_size=3,
        eval_batch_size=2,
        num_eval_tasks=2,
        seed=17,
        device_resident=False,
    )
    data = config.make()
    loader = data.eval_dataloader()
    assert loader.batch_size == 2
    assert loader.sample_by_task is False
    assert loader.num_tasks == 2
    assert loader.seed == 17
    assert loader.passes == 0
    assert isinstance(loader.inputs, np.memmap)
    batches = list(loader)
    assert [batch["valid_count"] for batch in batches] == [2, 2, 2, 2, 1]


def test_arc_data_checkpoint_preserves_pending_loader_and_timer_state(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path)
    config = ArcData.Config(working_dir=tmp_path, device="cpu", batch_size=2)
    original = config.make()
    loader = original.train_dataloader()
    iterator = iter(loader)
    next(iterator)
    saved = original.state_dict()
    restored = config.make()
    restored.load_state_dict(saved)
    resumed = restored.train_dataloader()
    assert saved["passes"] == 1
    assert saved["loader"] == {"passes": 1, "active_pass": 0, "next_batch": 1}
    assert resumed.state_dict() == saved["loader"]
    assert restored.state_dict() == saved


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


def test_puzzle_batches_skips_zero_length_puzzle_in_both_eval_subsets(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "test")
    np.save(
        tmp_path / "test" / "all__puzzle_indices.npy",
        np.array([0, 0, 5, 9, 12]),
    )
    np.save(tmp_path / "test" / "all__group_indices.npy", np.array([0, 1, 2, 4]))
    per_puzzle = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=2,
        max_augs_per_puzzle=1,
    )
    per_group = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=2,
        max_examples_per_group=1,
    )
    assert per_puzzle.eval_example_index is not None
    assert per_puzzle.eval_example_index.tolist() == [2, 7, 11]
    assert per_group.eval_example_index is not None
    assert per_group.eval_example_index.tolist() == [0, 5]


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
        iters_offset=4,
    )
    loader = data.train_dataloader()
    assert loader.train is True
    assert loader.device == torch.device("cpu")
    assert loader.seed == 29
    assert loader.epochs_per_iter == 3
    assert loader.n_examples == 7
    assert loader.iters == 4


def test_puzzle_data_exact_proxy_error_and_timer_checkpoint(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path, "test")
    data = _puzzle_data(
        tmp_path,
        eval_max_augs_per_puzzle=1,
        eval_max_examples_per_group=1,
    )
    with pytest.raises(
        ValueError,
        match=r"eval_max_augs_per_puzzle and eval_max_examples_per_group are mutually exclusive; set only one\.",
    ) as error:
        data.eval_dataloader()
    assert str(error.value) == (
        "eval_max_augs_per_puzzle and eval_max_examples_per_group are "
        "mutually exclusive; set only one."
    )
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


def test_arc_data_keeps_live_and_pending_state_distinct(tmp_path: Path) -> None:
    _write_split(tmp_path)
    config = ArcData.Config(working_dir=tmp_path, device="cpu", batch_size=2)
    data = config.make()
    data.timer_epoch.global_count = 5
    data.timer_epoch.global_sec = 2.5
    assert data.state_dict() == {
        "passes": 0,
        "loader": None,
        "timer_epoch": {"global_count": 5, "global_sec": 2.5},
    }
    source_loader = data.train_dataloader()
    next(iter(source_loader))
    saved = data.state_dict()
    restored = config.make()
    restored.load_state_dict(saved)
    resumed = restored.train_dataloader()
    assert restored._pending_loader_state is None
    assert resumed.state_dict() == saved["loader"]
    assert restored.state_dict() == saved


def test_arc_data_loader_honors_explicit_device_over_torch_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_split(tmp_path, "train")
    _write_split(tmp_path, "test")
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
    config = ArcData.Config(
        working_dir=tmp_path,
        device="cpu",
        batch_size=2,
        eval_batch_size=2,
        device_resident=False,
    )
    data = config.make()
    assert data.train_dataloader().device == torch.device("cpu")
    assert data.eval_dataloader().device == torch.device("cpu")


def test_arc_resident_batches_keep_indices_and_outputs_on_explicit_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_arc_split(tmp_path, "test")
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
    loader = _arc_data(tmp_path, eval_batch_size=4).eval_dataloader()

    batches = list(loader)

    assert isinstance(loader.inputs, Tensor)
    assert loader.inputs.device == torch.device("cpu")
    assert _tensor(batches[0]["media"]).device == torch.device("cpu")
    assert _tensor(batches[0]["label"]).device == torch.device("cpu")
    assert _tensor(batches[0]["spatial_tags"]).device == torch.device("cpu")
    assert _tensor(batches[0]["puzzle_identifiers"]).device == torch.device("cpu")
    assert _tensor(batches[0]["media"])[:, 0].tolist() == [2, 6, 10, 14]


def test_arc_mmap_batches_keep_int32_on_explicit_device_and_ordered_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_arc_split(tmp_path, "test")
    split = tmp_path / "test"
    inputs = np.arange(24, dtype=np.int16).reshape(6, 4) + 2
    labels = inputs.copy()
    labels[0, 0] = 0
    np.save(split / "all__inputs.npy", inputs)
    np.save(split / "all__labels.npy", labels)
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))

    loader = _arc_data(
        tmp_path,
        eval_batch_size=4,
        num_eval_tasks=2,
        device_resident=False,
    ).eval_dataloader()
    batches = list(loader)

    assert isinstance(loader.inputs, np.memmap)
    assert [batch["valid_count"] for batch in batches] == [4, 2]
    assert all(_tensor(batch["media"]).dtype == torch.int32 for batch in batches)
    assert all(_tensor(batch["label"]).dtype == torch.int32 for batch in batches)
    assert all(
        _tensor(batch["media"]).device == torch.device("cpu") for batch in batches
    )
    assert _tensor(batches[0]["media"]).tolist() == [
        [2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
    ]
    assert _tensor(batches[1]["media"]).tolist() == [
        [18, 19, 20, 21],
        [22, 23, 24, 25],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert _tensor(batches[1]["puzzle_identifiers"]).tolist() == [9, 9, 0, 0]


def test_arc_data_restoring_live_loader_clears_pending_state(tmp_path: Path) -> None:
    _write_arc_split(tmp_path)
    data = _arc_data(tmp_path)
    loader = data.train_dataloader()
    next(iter(loader))

    data.load_state_dict(
        {
            "passes": 4,
            "loader": {"passes": 4, "active_pass": 2, "next_batch": 3},
        },
    )

    assert loader.state_dict() == {
        "passes": 4,
        "active_pass": 2,
        "next_batch": 3,
    }
    assert data._pending_loader_state is None
    assert data.state_dict()["loader"] == loader.state_dict()


def test_arc_train_loader_recreation_carries_completed_passes(tmp_path: Path) -> None:
    _write_arc_split(tmp_path)
    data = _arc_data(tmp_path, epochs_per_iter=2)
    first = data.train_dataloader()
    list(first)

    second = data.train_dataloader()

    assert first.passes == second.passes == 1
    assert second._active_pass is None
    assert second._next_batch == 0


def test_arc_sampled_loader_increments_nonzero_pass_count(tmp_path: Path) -> None:
    _write_arc_split(tmp_path)
    loader = _arc_data(tmp_path, epochs_per_iter=2).train_dataloader()
    loader.passes = 7

    batches = list(loader)

    assert batches
    assert loader.passes == 8


def test_arc_sample_plan_skips_empty_task_groups(tmp_path: Path) -> None:
    _write_arc_split(tmp_path)
    np.save(
        tmp_path / "train" / "all__group_indices.npy",
        np.array([0, 0, 1, 3]),
    )
    loader = _arc_data(tmp_path, epochs_per_iter=2).train_dataloader()

    batches = list(loader)

    assert batches
    assert all(batch["valid_count"] == 2 for batch in batches)


def test_arc_sampled_batch_plan_respects_rank_slice_and_full_batch_boundary(
    tmp_path: Path,
) -> None:
    _write_arc_split(tmp_path)
    whole = _arc_data(
        tmp_path,
        batch_size=4,
        seed=19,
        epochs_per_iter=3,
    ).train_dataloader()
    rank_loaders = [
        _arc_data(
            tmp_path,
            batch_size=2,
            rank=rank,
            num_replicas=2,
            seed=19,
            epochs_per_iter=3,
        ).train_dataloader()
        for rank in range(2)
    ]

    batches = list(whole)
    rank_batches = [list(loader) for loader in rank_loaders]

    assert [_tensor(batch["media"])[:, 0].tolist() for batch in batches] == [
        [14, 10, 6, 2],
        [10, 14, 2, 6],
        [6, 2, 22, 18],
    ]
    assert [_tensor(batch["puzzle_identifiers"]).tolist() for batch in batches] == [
        [8, 8, 7, 7],
        [8, 8, 7, 7],
        [7, 7, 9, 9],
    ]
    assert batches
    assert len(batches) == len(rank_batches[0]) == len(rank_batches[1])
    for batch, first_rank, second_rank in zip(
        batches,
        rank_batches[0],
        rank_batches[1],
        strict=True,
    ):
        assert batch["valid_count"] == 4
        assert torch.equal(
            _tensor(batch["media"]),
            torch.cat([_tensor(first_rank["media"]), _tensor(second_rank["media"])]),
        )
        assert torch.equal(
            _tensor(batch["puzzle_identifiers"]),
            torch.cat(
                [
                    _tensor(first_rank["puzzle_identifiers"]),
                    _tensor(second_rank["puzzle_identifiers"]),
                ],
            ),
        )
    assert whole.passes == 1
    assert whole.state_dict() == {
        "passes": 1,
        "active_pass": None,
        "next_batch": 0,
    }


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


def test_g11_int_list_returns_python_ints_in_order() -> None:
    values = np.array([4, 9, 12], dtype=np.int32)

    result = _int_list(values)

    assert result == [4, 9, 12]
    assert all(type(value) is int for value in result)


def test_g11_load_split_defaults_tags_and_mmap_representation(
    tmp_path: Path,
) -> None:
    root = _write_helper_arc_split(tmp_path)

    resident = _load_split(root, split="train")
    mapped = _load_split(root, split="train", mmap=True)

    assert isinstance(resident["inputs"], Tensor)
    assert isinstance(resident["labels"], Tensor)
    assert resident["inputs"].dtype == torch.int32
    assert resident["labels"].dtype == torch.int32
    assert isinstance(mapped["inputs"], np.memmap)
    assert isinstance(mapped["labels"], np.memmap)
    assert mapped["inputs"].dtype == np.int32
    assert mapped["labels"].dtype == np.int32
    expected_tags = np.tile(np.array([[1, 0, 0]], dtype=np.int64), (3, 1))
    np.testing.assert_array_equal(resident["spatial_tags"], expected_tags)
    np.testing.assert_array_equal(mapped["spatial_tags"], expected_tags)
    assert resident["ignore_label_id"] == mapped["ignore_label_id"] == 255


def test_g11_load_split_rejects_bad_tag_count(tmp_path: Path) -> None:
    tags = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.int32)
    root = _write_helper_arc_split(tmp_path, spatial_tags=tags)

    with pytest.raises(ValueError, match=r"spatial_tags\.npy has 2 rows.*3 puzzles"):
        _load_split(root, split="train")


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


def test_g11_arc_batches_init_caps_at_task_and_moves_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_helper_arc_split(tmp_path)

    def meta_device(device: torch.device | str) -> torch.device:
        del device
        return torch.device("meta")

    monkeypatch.setattr(arc_data, "get_device", meta_device)

    batches = _ArcBatches(
        dataset_dir=root,
        device="cpu",
        batch_size=2,
        split="train",
        sample_by_task=True,
        num_tasks=1,
        seed=7,
        passes=3,
        rank=0,
        num_replicas=1,
        epochs_per_iter=2,
        device_resident=True,
    )

    assert isinstance(batches.inputs, Tensor)
    assert isinstance(batches.labels, Tensor)
    assert batches.inputs.shape == (5, 4)
    assert batches.labels.shape == (5, 4)
    assert batches.inputs.device.type == "meta"
    assert batches.labels.device.type == "meta"
    assert batches.spatial_tags.device.type == "meta"
    np.testing.assert_array_equal(batches.groups, [0, 2])
    np.testing.assert_array_equal(batches.puzzles, [0, 2, 5])
    np.testing.assert_array_equal(batches.identifiers, [11, 22])
    assert batches.num_tasks == 1
    assert batches.global_batch_size == 2
    assert batches.passes == 3
    assert batches._active_pass is None
    assert batches._next_batch == 0


def test_g11_arc_batches_len_uses_rows_and_active_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_helper_arc_split(tmp_path)
    ordered = _ArcBatches(
        dataset_dir=root,
        device="cpu",
        batch_size=2,
        split="train",
        sample_by_task=False,
        num_tasks=None,
        seed=1,
        passes=0,
        rank=0,
        num_replicas=2,
        epochs_per_iter=2,
        device_resident=False,
    )
    assert len(ordered) == 2

    sampled = _ArcBatches(
        dataset_dir=root,
        device="cpu",
        batch_size=2,
        split="train",
        sample_by_task=True,
        num_tasks=None,
        seed=1,
        passes=4,
        rank=0,
        num_replicas=1,
        epochs_per_iter=2,
        device_resident=False,
    )
    plan_calls: list[int] = []

    def plan(pass_index: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        plan_calls.append(pass_index)
        return iter(
            (np.array([index]), np.array([index])) for index in range(pass_index)
        )

    monkeypatch.setattr(sampled, "_plan_sampled", plan)

    assert len(sampled) == 4
    assert plan_calls == [4]
    sampled._active_pass = 1
    assert len(sampled) == 1
    assert plan_calls == [4, 1]


def test_g11_arc_batches_iter_dispatches_by_sampling_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_helper_arc_split(tmp_path)
    batches = _ArcBatches(
        dataset_dir=root,
        device="cpu",
        batch_size=2,
        split="train",
        sample_by_task=True,
        num_tasks=None,
        seed=1,
        passes=0,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        device_resident=False,
    )
    monkeypatch.setattr(
        batches,
        "_iter_sampled",
        lambda: iter([{"which": "train"}]),
    )
    monkeypatch.setattr(
        batches,
        "_iter_ordered",
        lambda: iter([{"which": "eval"}]),
    )
    assert list(batches) == [{"which": "train"}]

    batches.sample_by_task = False
    assert list(batches) == [{"which": "eval"}]


def _write_sampled_plan_split(root: Path) -> None:
    split = root / "train"
    split.mkdir(parents=True)
    (split / "dataset.json").write_text('{"ignore_label_id":0}')
    np.save(split / "all__inputs.npy", np.arange(24, dtype=np.int32).reshape(12, 2))
    np.save(split / "all__labels.npy", np.arange(24, dtype=np.int32).reshape(12, 2))
    np.save(split / "all__puzzle_indices.npy", np.array([0, 2, 5, 8, 10, 12]))
    np.save(split / "all__group_indices.npy", np.array([0, 2, 3, 5]))
    np.save(
        split / "all__puzzle_identifiers.npy",
        np.array([11, 12, 21, 31, 32], dtype=np.int32),
    )


def _sampled_plan_loader(root: Path, *, seed: int, epochs_per_iter: int) -> _ArcBatches:
    return _ArcBatches(
        dataset_dir=root,
        device="cpu",
        batch_size=4,
        split="train",
        sample_by_task=True,
        num_tasks=None,
        seed=seed,
        passes=0,
        rank=0,
        num_replicas=1,
        epochs_per_iter=epochs_per_iter,
        device_resident=False,
    )


def test_arc_plan_sampled_returns_exact_seeded_rows_and_puzzles(
    tmp_path: Path,
) -> None:
    _write_sampled_plan_split(tmp_path)
    plan = list(
        _sampled_plan_loader(tmp_path, seed=29, epochs_per_iter=2)._plan_sampled(3),
    )

    assert [rows.tolist() for rows, _ in plan] == [
        [8, 9, 0, 1],
        [7, 6, 5, 5],
        [11, 10, 3, 2],
    ]
    assert [puzzles.tolist() for _, puzzles in plan] == [
        [3, 3, 0, 0],
        [2, 2, 2, 2],
        [4, 4, 1, 1],
    ]
    assert all(rows.dtype == puzzles.dtype == np.int64 for rows, puzzles in plan)


def test_arc_plan_sampled_replays_pass_and_changes_next_pass(tmp_path: Path) -> None:
    _write_sampled_plan_split(tmp_path)
    loader = _sampled_plan_loader(tmp_path, seed=41, epochs_per_iter=3)

    first = list(loader._plan_sampled(0))
    replay = list(loader._plan_sampled(0))
    next_pass = list(loader._plan_sampled(1))

    assert len(first) == 4
    assert [rows.tolist() for rows, _ in first] == [
        [10, 11, 6, 7],
        [1, 0, 4, 2],
        [9, 8, 7, 6],
        [3, 2, 4, 6],
    ]
    assert [puzzles.tolist() for _, puzzles in first] == [
        [4, 4, 2, 2],
        [0, 0, 1, 1],
        [3, 3, 2, 2],
        [1, 1, 1, 2],
    ]
    assert all(
        np.array_equal(first_rows, replay_rows)
        and np.array_equal(first_puzzles, replay_puzzles)
        for (first_rows, first_puzzles), (replay_rows, replay_puzzles) in zip(
            first,
            replay,
            strict=True,
        )
    )
    assert any(
        not np.array_equal(first_rows, next_rows)
        or not np.array_equal(first_puzzles, next_puzzles)
        for (first_rows, first_puzzles), (next_rows, next_puzzles) in zip(
            first,
            next_pass,
            strict=True,
        )
    )
    assert all(rows.shape == puzzles.shape == (4,) for rows, puzzles in first)
    assert all(np.unique(rows).size == 4 for rows, _ in first)


def test_arc_plan_sampled_skips_empty_task_before_filling_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sampled_plan_split(tmp_path)
    np.save(
        tmp_path / "train" / "all__group_indices.npy",
        np.array([0, 2, 2, 5], dtype=np.int64),
    )
    loader = _sampled_plan_loader(tmp_path, seed=29, epochs_per_iter=1)

    class FixedRng:
        def permutation(self, size: int) -> np.ndarray:
            assert size == 3
            return np.array([1, 0, 2])

        def integers(self, low: int, high: int) -> int:
            assert low < high
            return low

        def choice(self, size: int, count: int, *, replace: bool) -> np.ndarray:
            assert 0 <= count <= size
            assert replace is False
            return np.arange(count)

    def make_generator(bitgen: object) -> FixedRng:
        del bitgen
        return FixedRng()

    monkeypatch.setattr(np.random, "Generator", make_generator)

    plan = list(loader._plan_sampled(3))

    assert len(plan) == 1
    rows, puzzle_ids = plan[0]
    assert rows.tolist() == [0, 1, 5, 6]
    assert puzzle_ids.tolist() == [0, 0, 2, 2]


def test_arc_plan_sampled_puzzle_ids_use_explicit_int64_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_sampled_plan_split(tmp_path)
    loader = _sampled_plan_loader(tmp_path, seed=29, epochs_per_iter=1)
    full = Mock(wraps=np.full)
    monkeypatch.setattr(np, "full", full)

    plan = list(loader._plan_sampled(3))

    assert plan
    typed_calls = [call for call in full.call_args_list if len(call.args) == 2]
    assert typed_calls
    assert all(call.kwargs.get("dtype") is np.int64 for call in typed_calls)


def test_puzzle_subset_empty_and_singleton_outputs_are_int64(tmp_path: Path) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    loader = PuzzleBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=2,
        rank=0,
        num_replicas=1,
        train=False,
        seed=3,
    )
    loader.puzzle_indices = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    loader.group_indices = np.array([0, 2, 4], dtype=np.int64)

    empty_puzzles = loader._build_eval_subset(2)
    empty_groups = loader._build_eval_group_subset(2)
    assert empty_puzzles.shape == empty_groups.shape == (0,)
    assert empty_puzzles.dtype == empty_groups.dtype == np.int64

    loader.puzzle_indices = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    assert loader._build_eval_subset(1).tolist() == [0]
    assert loader._build_eval_group_subset(2).tolist() == [0]


def test_empty_eval_subsets_have_int64_outputs_without_puzzles_or_groups(
    tmp_path: Path,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    loader.puzzle_indices = np.array([0], dtype=np.int64)
    loader.group_indices = np.array([0], dtype=np.int64)

    per_puzzle = loader._build_eval_subset(1)
    per_group = loader._build_eval_group_subset(1)

    assert per_puzzle.shape == per_group.shape == (0,)
    assert per_puzzle.dtype == per_group.dtype == np.int64


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


def test_eval_loader_uses_max_samples_fallback_and_rejects_proxy_cap(
    tmp_path: Path,
) -> None:
    _write_small_puzzle_split(tmp_path, "test")
    data = _small_puzzle_data(tmp_path, max_samples=4)

    assert sum(batch["valid_count"] for batch in data.eval_dataloader()) == 4
    data.config.eval_max_augs_per_puzzle = 1
    with pytest.raises(ValueError, match="mutually exclusive"):
        data.eval_dataloader()


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
    assert _int_list(puzzle_ids[:2]) == [_int_list(puzzle_ids[:1])[0]] * 2
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


def test_puzzle_collate_preserves_large_ids_masks_and_pads(tmp_path: Path) -> None:
    loader = _g00_puzzle_batches(tmp_path, batch_size=3)
    loader.puzzle_identifiers[:] = [np.iinfo(np.int32).max, 2, 3]
    loader.puzzle_identifier_offset = 1

    batch = loader._collate(
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )

    assert batch["valid_count"] == 2
    assert batch["media"].tolist() == [[0, 1, 2, 3], [8, 9, 10, 11], [0] * 4]
    assert batch["label"].tolist() == [
        [-100, 31, 32, 33],
        [38, 39, 40, 41],
        [-100] * 4,
    ]
    assert batch["puzzle_identifiers"].dtype == torch.int64
    assert batch["puzzle_identifiers"].tolist() == [2**31, 3, 99]
    assert batch["spatial_tags"].dtype == torch.int64
    assert batch["spatial_tags"].tolist() == [
        [2, 1, 0],
        [3, 0, 1],
        [1, 0, 0],
    ]


def test_puzzle_collate_requests_int64_identifier_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    loader.puzzle_identifier_remap = np.arange(10, dtype=np.int64) + 100
    asarray = Mock(wraps=np.asarray)
    monkeypatch.setattr(np, "asarray", asarray)

    loader._collate(
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        valid=2,
    )

    assert (
        sum(call.kwargs.get("dtype") is np.int64 for call in asarray.call_args_list)
        == 2
    )


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


def test_int_list_requests_python_integer_coercion() -> None:
    result = _int_list(np.array([4, 9, 12], dtype=np.int32))

    assert result == [4, 9, 12]
    assert all(type(value) is int for value in result)


def test_puzzle_eval_subset_preserves_rng_for_exact_limit_puzzle(
    tmp_path: Path,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    loader.puzzle_indices = np.array([0, 2, 5], dtype=np.int64)
    loader.group_indices = np.array([0, 2], dtype=np.int64)
    rng = np.random.Generator(np.random.Philox(seed=loader.seed))
    second_puzzle_offsets = rng.permutation(3)[:2]
    second_puzzle_offsets.sort()

    selected = loader._build_eval_subset(2)

    assert selected.tolist() == [0, 1, *(2 + second_puzzle_offsets).tolist()]


def test_puzzle_eval_subset_requests_int64_range_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _g00_puzzle_batches(tmp_path)
    loader.puzzle_indices = np.array([0, 2, 5], dtype=np.int64)
    ranges = Mock(wraps=np.arange)
    monkeypatch.setattr(np, "arange", ranges)

    loader._build_eval_subset(2)

    assert any(
        call.args == (0, 2) and call.kwargs == {"dtype": np.int64}
        for call in ranges.call_args_list
    )


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


def test_puzzle_data_eval_dataloader_reports_exact_conflicting_caps() -> None:
    config = PuzzleData.Config(
        eval_max_samples=8,
        eval_max_augs_per_puzzle=1,
        device="cpu",
    )
    data = PuzzleData(config)

    with pytest.raises(ValueError, match=r"a prefix cap") as error:
        data.eval_dataloader()

    assert str(error.value) == (
        "a prefix cap (eval_max_samples, or the max_samples fallback) is "
        "mutually exclusive with the per-puzzle / per-group proxy caps; "
        "set only one."
    )


class _ArcSliceRecorder:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.slices: list[slice] = []

    def __getitem__(self, index: slice) -> np.ndarray:
        self.slices.append(index)
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def tolist(self) -> list[int]:
        return _int_list(self.values)


def _arc_slice_recorder(value: object) -> _ArcSliceRecorder:
    assert isinstance(value, _ArcSliceRecorder)
    return value


def _fake_arc_split(*, tracked: bool = False) -> dict[str, object]:
    inputs = np.arange(12, dtype=np.int32).reshape(3, 4)
    labels = inputs + 10
    puzzles = np.array([0, 1, 2, 3], dtype=np.int64)
    groups = np.array([0, 1, 3], dtype=np.int64)
    return {
        "inputs": _ArcSliceRecorder(inputs) if tracked else inputs,
        "labels": _ArcSliceRecorder(labels) if tracked else labels,
        "puzzle_indices": _ArcSliceRecorder(puzzles) if tracked else puzzles,
        "group_indices": _ArcSliceRecorder(groups) if tracked else groups,
        "puzzle_identifiers": np.array([11, 22, 33], dtype=np.int32),
        "spatial_tags": np.tile(np.array([1, 0, 0], dtype=np.int64), (3, 1)),
        "ignore_label_id": 0,
    }


def test_arc_batches_full_task_cap_uses_explicit_row_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _fake_arc_split(tracked=True)
    monkeypatch.setattr(arc_data, "_load_split", Mock(return_value=data))

    loader = _ArcBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=1,
        split="train",
        sample_by_task=True,
        num_tasks=2,
        seed=0,
        passes=0,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        device_resident=False,
    )

    assert loader.num_tasks == 2
    assert loader.inputs.shape == loader.labels.shape == (3, 4)
    assert _arc_slice_recorder(data["group_indices"]).slices == []
    assert _arc_slice_recorder(data["puzzle_indices"]).slices == []

    data = _fake_arc_split(tracked=True)
    monkeypatch.setattr(arc_data, "_load_split", Mock(return_value=data))
    _ArcBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=1,
        split="train",
        sample_by_task=True,
        num_tasks=None,
        seed=0,
        passes=0,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        device_resident=False,
    )

    expected = slice(None, 3)
    assert _arc_slice_recorder(data["inputs"]).slices == [expected]
    assert _arc_slice_recorder(data["labels"]).slices == [expected]


def test_arc_batches_normalize_and_bound_spatial_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _fake_arc_split()
    data["spatial_tags"] = np.arange(12, dtype=np.int32).reshape(4, 3)
    monkeypatch.setattr(arc_data, "_load_split", Mock(return_value=data))
    asarray = Mock(wraps=np.asarray)
    monkeypatch.setattr(np, "asarray", asarray)

    loader = _ArcBatches(
        dataset_dir=tmp_path,
        device="cpu",
        batch_size=1,
        split="train",
        sample_by_task=False,
        num_tasks=None,
        seed=0,
        passes=0,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        device_resident=False,
    )

    assert loader.spatial_tags.shape == (3, 3)
    assert loader.spatial_tags.dtype == torch.int64
    assert any(call.kwargs.get("dtype") is np.int64 for call in asarray.call_args_list)


def _arc_batches_for_batch_test(
    root: Path,
    *,
    device: str,
    batch_size: int,
    rank: int = 0,
    num_replicas: int = 1,
    device_resident: bool = False,
) -> _ArcBatches:
    _write_arc_split(root, "test")
    return _ArcBatches(
        dataset_dir=root,
        device=device,
        batch_size=batch_size,
        split="test",
        sample_by_task=False,
        num_tasks=None,
        seed=0,
        passes=0,
        rank=rank,
        num_replicas=num_replicas,
        epochs_per_iter=1,
        device_resident=device_resident,
    )


def test_arc_batch_mmap_outputs_and_padding_use_loader_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _arc_batches_for_batch_test(
        tmp_path,
        device="meta",
        batch_size=3,
    )
    original_to = Tensor.to
    devices: list[torch.device | str | None] = []

    def tracked_to(
        tensor: Tensor,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> Tensor:
        devices.append(device)
        return original_to(tensor, device, non_blocking=non_blocking)

    monkeypatch.setattr(Tensor, "to", tracked_to)
    cat = Mock(wraps=torch.cat)
    monkeypatch.setattr(torch, "cat", cat)

    full = loader._batch(
        np.array([0, 1, 2], dtype=np.int64),
        puzzle_ids=np.array([0, 0, 1], dtype=np.int64),
        valid=3,
    )

    assert cat.call_count == 0
    assert _tensor(full["media"]).device == torch.device("meta")
    assert _tensor(full["label"]).device == torch.device("meta")
    assert _tensor(full["puzzle_identifiers"]).device == torch.device("meta")
    assert _tensor(full["puzzle_identifiers"]).dtype == torch.int64
    assert _tensor(full["spatial_tags"]).device == torch.device("meta")

    partial = loader._batch(
        np.array([0, 1], dtype=np.int64),
        puzzle_ids=np.array([0, 0], dtype=np.int64),
        valid=2,
    )

    assert cat.call_count == 4
    assert _tensor(partial["media"]).device == torch.device("meta")
    assert _tensor(partial["label"]).device == torch.device("meta")
    assert _tensor(partial["puzzle_identifiers"]).device == torch.device("meta")
    assert _tensor(partial["spatial_tags"]).device == torch.device("meta")
    assert _tensor(partial["spatial_tags"]).shape == (3, 3)
    assert torch.device("meta") in devices
    assert None not in devices


def test_arc_batch_resident_indices_use_loader_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _arc_batches_for_batch_test(
        tmp_path,
        device="meta",
        batch_size=2,
        device_resident=True,
    )
    original_to = Tensor.to
    devices: list[torch.device | str | None] = []

    def tracked_to(
        tensor: Tensor,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> Tensor:
        devices.append(device)
        return original_to(tensor, device, non_blocking=non_blocking)

    monkeypatch.setattr(Tensor, "to", tracked_to)

    batch = loader._batch(
        np.array([0, 1], dtype=np.int64),
        puzzle_ids=np.array([0, 0], dtype=np.int64),
        valid=2,
    )

    assert _tensor(batch["media"]).device == torch.device("meta")
    assert _tensor(batch["label"]).device == torch.device("meta")
    assert torch.device("meta") in devices
    assert None not in devices


def test_arc_ordered_rank_slice_requests_explicit_int64_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _arc_batches_for_batch_test(
        tmp_path,
        device="cpu",
        batch_size=2,
        num_replicas=3,
    )
    arange = Mock(wraps=np.arange)
    monkeypatch.setattr(np, "arange", arange)

    batches = list(loader._iter_ordered())

    assert len(batches) == 1
    assert batches[0]["valid_count"] == 2
    assert _tensor(batches[0]["media"])[:, 0].tolist() == [2, 6]
    assert any(
        call.args[:2] == (0, 2) and call.kwargs.get("dtype") is np.int64
        for call in arange.call_args_list
    )


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
    from priml.lib.testing.main import test_main

    test_main(__file__)
