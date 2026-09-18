"""Tests for the sudoku data preparer.

Hermetic: every test feeds local CSV text, so nothing here touches the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import json

import numpy as np
import pytest

from priml.baselines.sudoku.scripts.prepare_data import (
    default_directory,
    prepare,
)


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _load_array(path: Path) -> NDArray[np.int64]:
    return cast("NDArray[np.int64]", np.load(path))


SOLUTION = (
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
)
"""One valid solved grid, used to synthesize puzzles."""


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    """Write train/test CSVs holding four puzzles apiece."""
    rows = ["source,question,answer,rating"]
    for i in range(4):
        cells = list(SOLUTION)
        for j in range(0, 81, 6):
            cells[(j + i) % 81] = "."
        rows.append(f"synthetic,{''.join(cells)},{SOLUTION},1.0")
    for split in ("train", "test"):
        (tmp_path / f"{split}.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_training_split_expands_and_test_split_does_not(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    """Copies raise the training rows; the test split stays verbatim.

    A transformed test puzzle would not be a held-out puzzle, so only the
    training side is expanded.
    """
    out = prepare(
        tmp_path / "data",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    train = _load_array(out / "train" / "all__inputs.npy")
    test = _load_array(out / "test" / "all__inputs.npy")
    assert train.shape == (3 * 3, 81)  # 3 puzzles x (1 original + 2 copies)
    assert test.shape == (4, 81)  # Every source puzzle, untouched.


def test_group_indices_bound_each_puzzles_copies(tmp_path: Path, csv_dir: Path) -> None:
    """The loader shuffles within a puzzle, so the boundaries must be right."""
    out = prepare(
        tmp_path / "data",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    bounds = _load_array(out / "train" / "all__group_indices.npy")
    assert bounds.tolist() == [0, 3, 6, 9]


def test_tokens_land_in_the_documented_vocabulary(
    tmp_path: Path,
    csv_dir: Path,
) -> None:
    """0 is pad, 1 is an empty cell, 2-10 are the digits."""
    out = prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
    )
    inputs = _load_array(out / "train" / "all__inputs.npy")
    labels = _load_array(out / "train" / "all__labels.npy")
    assert inputs.min() >= 1  # No padding in stored rows.
    assert inputs.max() <= 10
    unique_labels = cast(list[int], np.unique(labels).tolist())
    assert set(unique_labels) <= set(range(2, 11))  # Solved: no empties.
    assert json.loads((out / "train" / "dataset.json").read_text()) == {
        "vocab_size": 11,
        "seq_len": 81,
    }


def test_transformations_keep_the_solution_valid(tmp_path: Path, csv_dir: Path) -> None:
    """Every generated copy must still be a solvable puzzle.

    A transformation that broke sudoku's constraints would teach the model
    contradictions, so this checks the property rather than the mechanics:
    each label grid has all nine digits in every row, column, and box.
    """
    out = prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=4,
        csv_directory=csv_dir,
    )
    labels = _load_array(out / "train" / "all__labels.npy")
    label_rows = cast(list[list[int]], labels.tolist())
    for row in label_rows:
        grid = [row[index : index + 9] for index in range(0, 81, 9)]
        expected = set(range(2, 11))
        assert all(set(line) == expected for line in grid)
        assert all(set(column) == expected for column in zip(*grid, strict=True))
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box = [line[c : c + 3] for line in grid[r : r + 3]]
                assert {value for line in box for value in line} == expected


def test_the_clues_survive_transformation(tmp_path: Path, csv_dir: Path) -> None:
    """A transformed puzzle's clues must agree with its transformed solution."""
    out = prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=4,
        csv_directory=csv_dir,
    )
    inputs = _load_array(out / "train" / "all__inputs.npy")
    labels = _load_array(out / "train" / "all__labels.npy")
    given = inputs > 1  # Token 1 is an empty cell.
    assert np.array_equal(inputs[given], labels[given])


def test_the_build_is_deterministic(tmp_path: Path, csv_dir: Path) -> None:
    """One seed, one dataset -- otherwise a result cannot be reproduced."""
    first = prepare(
        tmp_path / "a",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    second = prepare(
        tmp_path / "b",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    assert np.array_equal(
        _load_array(first / "train" / "all__inputs.npy"),
        _load_array(second / "train" / "all__inputs.npy"),
    )


def test_a_different_seed_builds_different_data(tmp_path: Path, csv_dir: Path) -> None:
    first = prepare(
        tmp_path / "a",
        num_puzzles=3,
        copies_per_puzzle=2,
        csv_directory=csv_dir,
    )
    second = prepare(
        tmp_path / "b",
        num_puzzles=3,
        copies_per_puzzle=2,
        seed=1,
        csv_directory=csv_dir,
    )
    assert not np.array_equal(
        _load_array(first / "train" / "all__inputs.npy"),
        _load_array(second / "train" / "all__inputs.npy"),
    )


def test_rerunning_leaves_a_prepared_split_alone(tmp_path: Path, csv_dir: Path) -> None:
    """Idempotent, so re-running the preparer costs nothing."""
    out = prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
    )
    marker = out / "train" / "all__inputs.npy"
    stamp = marker.stat().st_mtime_ns
    prepare(
        tmp_path / "data",
        num_puzzles=2,
        copies_per_puzzle=1,
        csv_directory=csv_dir,
    )
    assert marker.stat().st_mtime_ns == stamp


def test_default_directory_matches_the_loaders(tmp_path: Path) -> None:
    """Preparer and training agree without either naming a path."""
    del tmp_path
    assert default_directory().name == "sudoku-extreme"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
