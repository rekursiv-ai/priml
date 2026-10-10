from __future__ import annotations

import torch

from priml.baselines.sudoku.puzzle_spec import SudokuSpec, sudoku_group_indices


def test_constraint_groups_follow_puzzle_spec() -> None:
    spec = SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=6)
    groups = sudoku_group_indices(spec)
    assert groups.shape == (12, 4)
    assert torch.equal(groups[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(groups[4], torch.tensor([0, 4, 8, 12]))
    assert torch.equal(groups[8], torch.tensor([0, 1, 4, 5]))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
