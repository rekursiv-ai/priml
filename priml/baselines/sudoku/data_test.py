"""Tests for sudoku data loading and augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.sudoku.data import SudokuData, augment_sudoku
from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.lib.custom_json import ListCodec
from priml.testing.golden import assert_tensor_golden, stored


_CWD: Final = Path(__file__).resolve().parent


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
        inputs[:, 0] = np.arange(rows) % 9 + 2  # `make` rows distinguishable.
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


def test_augmentation_preserves_original_outputs_and_rng() -> None:
    """Preserve paired augmentation and the generator's subsequent draws."""
    grid = torch.arange(2 * 4).reshape(2, 4) % 11
    record: dict[str, Tensor] = {}
    for seed in (0, 3, 42):
        generator = torch.Generator().manual_seed(seed)
        inputs, labels = augment_sudoku(
            grid,
            grid.flip(0),
            spec=SudokuSpec(),
            generator=generator,
        )
        record[f"{seed}/inputs"] = stored(inputs)
        record[f"{seed}/labels"] = stored(labels)
        # The next draws pin the generator's position without its 5 KB state.
        record[f"{seed}/rng"] = torch.randint(
            0,
            2_147_483_647,
            (2,),
            generator=generator,
        )
    assert_tensor_golden(_CWD / "testdata" / "augmentation.pt", record)


def test_augmentation_uses_the_dataset_spec() -> None:
    spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=6)
    grid = torch.arange(32).reshape(2, 16) % 4 + 2
    actual, labels = augment_sudoku(
        grid,
        grid.clone(),
        spec=spec,
        generator=torch.Generator().manual_seed(0),
    )
    assert actual.shape == (2, 16)
    assert torch.equal(actual, labels)
    assert set(ListCodec.coerce(actual.flatten().tolist(), int)) == {2, 3, 4, 5}


def test_augmentation_moves_the_label_with_the_input() -> None:
    """A transformed puzzle must keep a correct solution, or it teaches noise."""
    torch.manual_seed(0)
    # A solved grid: input equals label, so the invariant is checkable directly.
    grid = torch.arange(162).reshape(2, 81) % 9 + 2
    inputs, labels = augment_sudoku(grid, grid.clone(), spec=SudokuSpec())
    assert torch.equal(inputs, labels)


def test_augmentation_preserves_empties_and_padding() -> None:
    """Tokens 0 and 1 are not digits and must survive relabeling."""
    torch.manual_seed(0)
    grid = torch.full((2, 81), 1, dtype=torch.long)  # Every cell empty.
    grid[:, :5] = 0  # Padding.
    inputs, _ = augment_sudoku(grid, grid.clone(), spec=SudokuSpec())
    assert set(ListCodec.coerce(inputs.flatten().tolist(), int)) <= {0, 1}


def test_augmentation_is_seedable() -> None:
    """A dedicated generator makes the stream independent of ambient draws."""
    grid = torch.arange(162).reshape(2, 81) % 9 + 2

    def once(disturb: bool) -> Tensor:
        generator = torch.Generator().manual_seed(3)
        if disturb:
            torch.rand(11)
        return augment_sudoku(
            grid,
            grid.clone(),
            spec=SudokuSpec(),
            generator=generator,
        )[0]

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


def test_num_puzzles_keeps_a_prefix_of_whole_puzzles(dataset_dir: Path) -> None:
    """Two copies per puzzle, so two puzzles are the first four rows."""
    data = _data(dataset_dir, augment=False, num_eval_puzzles=1, eval_batch_size=4)
    (only,) = list(data.eval_dataloader())
    assert only["valid_count"] == 2
    assert only["media"][:2, 0].tolist() == [2, 3]


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (SudokuSpec(vocab_size=12), "vocabulary"),
        (SudokuSpec(grid_shape=(4, 4)), "grid"),
    ],
)
def test_prepared_data_must_match_the_spec(
    dataset_dir: Path,
    spec: SudokuSpec,
    match: str,
) -> None:
    data = _data(dataset_dir, spec=spec)
    with pytest.raises(ValueError, match=match):
        data.train_dataloader()


def test_a_negative_epoch_is_rejected(dataset_dir: Path) -> None:
    data = _data(dataset_dir)
    live = data.train_dataloader()
    live.epoch = -1
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        data.train_dataloader()


def test_unseeded_augmentation_draws_from_the_ambient_stream(
    dataset_dir: Path,
) -> None:
    def first(seed: int) -> Tensor:
        torch.manual_seed(seed)
        return next(iter(_data(dataset_dir, seed=0).train_dataloader()))["media"]

    assert torch.equal(first(0), first(0))
    assert not torch.equal(first(0), first(1))


def test_a_new_loader_continues_the_live_loaders_epoch(dataset_dir: Path) -> None:
    data = _data(dataset_dir, augment=False, seed=2)
    list(data.train_dataloader())
    assert data.train_dataloader().epoch == 1


def test_restoring_state_into_a_live_loader(dataset_dir: Path) -> None:
    """A checkpoint loaded after the loader exists must still take effect."""
    source = _data(dataset_dir, augment=False, seed=6)
    iterator = iter(source.train_dataloader())
    next(iterator)
    mid_epoch = source.state_dict()
    expected = [batch["media"].clone() for batch in iterator]

    data = _data(dataset_dir, augment=False, seed=6)
    live = data.train_dataloader()
    data.load_state_dict(mid_epoch)
    observed = [batch["media"] for batch in live]

    assert len(observed) == len(expected)
    assert all(
        torch.equal(got, wanted) for got, wanted in zip(observed, expected, strict=True)
    )


def test_restoring_no_loader_rewinds_a_live_loader_to_the_epoch_count(
    dataset_dir: Path,
) -> None:
    data = _data(dataset_dir, augment=False, seed=6)
    live = data.train_dataloader()
    live.epoch = 9
    data.load_state_dict({"epoch": 0, "loader": None})
    assert live.epoch == 0


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        data.train_dataloader()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
