"""Tests for sudoku data loading and augmentation."""

from __future__ import annotations

from pathlib import Path

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.sudoku.data import SudokuData, augment_sudoku


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a tiny two-split dataset in the on-disk layout."""
    for split, puzzles in (("train", 3), ("test", 2)):
        directory = tmp_path / split
        directory.mkdir()
        copies = 2
        rows = puzzles * copies
        # Token 1 is the empty cell; 2..10 are digits, so a row of 2s is valid.
        inputs = np.full((rows, 81), 2, dtype=np.int32)
        inputs[:, 0] = np.arange(rows) % 9 + 2  # make rows distinguishable
        np.save(directory / "all__inputs.npy", inputs)
        np.save(directory / "all__labels.npy", inputs)
        np.save(
            directory / "all__group_indices.npy",
            np.arange(puzzles + 1, dtype=np.int32) * copies,
        )
        (directory / "dataset.json").write_text(
            json.dumps({"vocab_size": 11, "seq_len": 81}),
        )
    return tmp_path


def _data(dataset_dir: Path, **overrides: object) -> SudokuData:
    config = SudokuData.Config()
    config.base_dir = "/"
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_batches_are_always_full_width(dataset_dir: Path) -> None:
    """A short final batch is padded and reports how many rows are real."""
    data = _data(dataset_dir, augment=False)
    batches = list(data.train_dataloader())
    assert all(b["media"].shape == (4, 81) for b in batches)
    # 6 rows at batch 4: one full batch, one half batch padded up.
    assert [b["valid_count"] for b in batches] == [4, 2]
    tail = batches[-1]
    assert bool((tail["media"][2:] == 0).all())


def test_eval_is_neither_shuffled_nor_augmented(dataset_dir: Path) -> None:
    """A score must not depend on which transformation was drawn."""
    data = _data(dataset_dir, augment=True, seed=0)
    first = next(iter(data.eval_dataloader()))["media"]
    second = next(iter(data.eval_dataloader()))["media"]
    assert torch.equal(first, second)


def test_seeded_shuffle_is_reproducible_and_epoch_varying(
    dataset_dir: Path,
) -> None:
    """One seed fixes each epoch's order, and consecutive epochs differ."""
    orders: list[list[Tensor]] = []
    for _ in range(2):
        data = _data(dataset_dir, augment=False, seed=7)
        loader = data.train_dataloader()
        orders.append([b["media"].clone() for b in loader])
    assert torch.equal(orders[0][0], orders[1][0])

    data = _data(dataset_dir, augment=False, seed=7)
    loader = data.train_dataloader()
    epoch_one = next(iter(loader))["media"].clone()
    epoch_two = next(iter(loader))["media"].clone()
    assert not torch.equal(epoch_one, epoch_two)


def test_epoch_counter_round_trips(dataset_dir: Path) -> None:
    """Resume continues the shuffle sequence rather than replaying epoch 0."""
    data = _data(dataset_dir, augment=False, seed=1)
    loader = data.train_dataloader()
    list(loader)
    data.timer_epoch.global_count += 1
    state = data.state_dict()

    restored = _data(dataset_dir, augment=False, seed=1)
    restored.load_state_dict(state)
    assert restored.timer_epoch.global_count == 1
    assert torch.equal(
        next(iter(restored.train_dataloader()))["media"],
        next(iter(loader))["media"],
    )


def test_checkpoint_resumes_the_unfinished_epoch(dataset_dir: Path) -> None:
    data = _data(dataset_dir, augment=True, seed=3, augment_seed=7)
    loader = data.train_dataloader()
    iterator = iter(loader)
    next(iterator)
    state = data.state_dict()
    expected = [batch["media"].clone() for batch in iterator]

    restored = _data(dataset_dir, augment=True, seed=3, augment_seed=7)
    restored.load_state_dict(state)
    observed = [batch["media"].clone() for batch in restored.train_dataloader()]

    assert len(observed) == len(expected)
    assert all(
        torch.equal(observed_batch, expected_batch)
        for observed_batch, expected_batch in zip(observed, expected, strict=True)
    )


def test_seed_and_epoch_are_distinct_named_stream_inputs(dataset_dir: Path) -> None:
    later_epoch = _data(dataset_dir, augment=False, seed=4).train_dataloader()
    list(later_epoch)
    later = [batch["media"].clone() for batch in later_epoch]
    first = [
        batch["media"].clone()
        for batch in _data(dataset_dir, augment=False, seed=5).train_dataloader()
    ]

    assert any(
        not torch.equal(later_batch, first_batch)
        for later_batch, first_batch in zip(later, first, strict=True)
    )


def test_augmentation_moves_the_label_with_the_input() -> None:
    """A transformed puzzle must keep a correct solution, or it teaches noise."""
    torch.manual_seed(0)
    # A solved grid: input equals label, so the invariant is checkable directly.
    grid = torch.arange(81).reshape(1, 81) % 9 + 2
    inputs, labels = augment_sudoku(grid, grid.clone())
    assert torch.equal(inputs, labels)


def test_augmentation_preserves_empties_and_padding() -> None:
    """Tokens 0 and 1 are not digits and must survive relabeling."""
    torch.manual_seed(0)
    grid = torch.full((2, 81), 1, dtype=torch.long)  # every cell empty
    grid[:, :5] = 0  # padding
    inputs, _ = augment_sudoku(grid, grid.clone())
    assert set(inputs.unique().tolist()) <= {0, 1}


def test_augmentation_is_seedable() -> None:
    """A dedicated generator makes the stream independent of ambient draws."""
    grid = torch.arange(81).reshape(1, 81) % 9 + 2

    def once(disturb: bool) -> Tensor:
        generator = torch.Generator().manual_seed(3)
        if disturb:
            torch.rand(11)
        return augment_sudoku(grid, grid.clone(), generator=generator)[0]

    assert torch.equal(once(disturb=False), once(disturb=True))


def test_seeded_augmentation_resumes_at_the_next_epoch(dataset_dir: Path) -> None:
    data = _data(dataset_dir, seed=1, augment=True, augment_seed=7)
    loader = data.train_dataloader()
    list(loader)
    data.timer_epoch.global_count += 1
    state = data.state_dict()
    expected = [batch["media"].clone() for batch in loader]

    restored = _data(dataset_dir, seed=1, augment=True, augment_seed=7)
    restored.load_state_dict(state)
    observed = [batch["media"].clone() for batch in restored.train_dataloader()]

    assert len(observed) == len(expected)
    assert all(
        torch.equal(observed_batch, expected_batch)
        for observed_batch, expected_batch in zip(observed, expected, strict=True)
    )


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


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        data.train_dataloader()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
