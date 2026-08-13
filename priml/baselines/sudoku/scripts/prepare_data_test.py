"""Tests for the sudoku data preparer.

Hermetic: every test feeds local CSV text, so nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest

from priml.baselines.sudoku.scripts.prepare_data import (
    default_directory,
    prepare,
)


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
    tmp_path: Path, csv_dir: Path
) -> None:
    """Copies raise the training rows; the test split stays verbatim.

    A transformed test puzzle would not be a held-out puzzle, so only the
    training side is expanded.
    """
    out = prepare(
        tmp_path / "data", num_puzzles=3, copies_per_puzzle=2, csv_directory=csv_dir
    )
    train = np.load(out / "train" / "all__inputs.npy")
    test = np.load(out / "test" / "all__inputs.npy")
    assert train.shape == (3 * 3, 81)  # 3 puzzles x (1 original + 2 copies)
    assert test.shape == (4, 81)  # every source puzzle, untouched


def test_group_indices_bound_each_puzzles_copies(tmp_path: Path, csv_dir: Path) -> None:
    """The loader shuffles within a puzzle, so the boundaries must be right."""
    out = prepare(
        tmp_path / "data", num_puzzles=3, copies_per_puzzle=2, csv_directory=csv_dir
    )
    bounds = np.load(out / "train" / "all__group_indices.npy")
    assert bounds.tolist() == [0, 3, 6, 9]


def test_tokens_land_in_the_documented_vocabulary(
    tmp_path: Path, csv_dir: Path
) -> None:
    """0 is pad, 1 is an empty cell, 2-10 are the digits."""
    out = prepare(
        tmp_path / "data", num_puzzles=2, copies_per_puzzle=1, csv_directory=csv_dir
    )
    inputs = np.load(out / "train" / "all__inputs.npy")
    labels = np.load(out / "train" / "all__labels.npy")
    assert inputs.min() >= 1  # no padding in stored rows
    assert inputs.max() <= 10
    assert set(np.unique(labels).tolist()) <= set(range(2, 11))  # solved: no empties
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
        tmp_path / "data", num_puzzles=2, copies_per_puzzle=4, csv_directory=csv_dir
    )
    labels = np.load(out / "train" / "all__labels.npy")
    for row in labels:
        grid = row.reshape(9, 9)
        expected = set(range(2, 11))
        assert all(set(line.tolist()) == expected for line in grid)
        assert all(set(line.tolist()) == expected for line in grid.T)
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                assert set(grid[r : r + 3, c : c + 3].flatten().tolist()) == expected


def test_the_clues_survive_transformation(tmp_path: Path, csv_dir: Path) -> None:
    """A transformed puzzle's clues must agree with its transformed solution."""
    out = prepare(
        tmp_path / "data", num_puzzles=2, copies_per_puzzle=4, csv_directory=csv_dir
    )
    inputs = np.load(out / "train" / "all__inputs.npy")
    labels = np.load(out / "train" / "all__labels.npy")
    given = inputs > 1  # token 1 is an empty cell
    assert np.array_equal(inputs[given], labels[given])


def test_the_build_is_deterministic(tmp_path: Path, csv_dir: Path) -> None:
    """One seed, one dataset -- otherwise a result cannot be reproduced."""
    first = prepare(
        tmp_path / "a", num_puzzles=3, copies_per_puzzle=2, csv_directory=csv_dir
    )
    second = prepare(
        tmp_path / "b", num_puzzles=3, copies_per_puzzle=2, csv_directory=csv_dir
    )
    assert np.array_equal(
        np.load(first / "train" / "all__inputs.npy"),
        np.load(second / "train" / "all__inputs.npy"),
    )


def test_a_different_seed_builds_different_data(tmp_path: Path, csv_dir: Path) -> None:
    first = prepare(
        tmp_path / "a", num_puzzles=3, copies_per_puzzle=2, csv_directory=csv_dir
    )
    second = prepare(
        tmp_path / "b",
        num_puzzles=3,
        copies_per_puzzle=2,
        seed=1,
        csv_directory=csv_dir,
    )
    assert not np.array_equal(
        np.load(first / "train" / "all__inputs.npy"),
        np.load(second / "train" / "all__inputs.npy"),
    )


def test_rerunning_leaves_a_prepared_split_alone(tmp_path: Path, csv_dir: Path) -> None:
    """Idempotent, so re-running the preparer costs nothing."""
    out = prepare(
        tmp_path / "data", num_puzzles=2, copies_per_puzzle=1, csv_directory=csv_dir
    )
    marker = out / "train" / "all__inputs.npy"
    stamp = marker.stat().st_mtime_ns
    prepare(
        tmp_path / "data", num_puzzles=2, copies_per_puzzle=1, csv_directory=csv_dir
    )
    assert marker.stat().st_mtime_ns == stamp


def test_default_directory_matches_the_loaders(tmp_path: Path) -> None:
    """Preparer and training agree without either naming a path."""
    del tmp_path
    assert default_directory().name == "sudoku-extreme"


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
