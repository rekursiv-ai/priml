from __future__ import annotations

import pytest
import torch

from priml.baselines.sudoku.views import (
    IDENTITY_DIGITS,
    NINE_VIEWS,
    View,
    validate_grid_permutation,
    validate_views,
)


def test_every_view_inverts_and_lines_must_map_bands_onto_bands() -> None:
    grid = torch.arange(2 * 81).reshape(2, 81) % 9 + 2
    for view in NINE_VIEWS:
        assert torch.equal(view.invert(view.apply(grid)), grid)
    with pytest.raises(ValueError, match="permutation"):
        validate_grid_permutation("rows", (0, 0, 1, 2, 3, 4, 5, 6, 7))
    with pytest.raises(ValueError, match="band"):
        validate_grid_permutation("rows", (0, 1, 3, 2, 4, 5, 6, 7, 8))


def test_view_validation_rejects_a_non_permutation() -> None:
    bad = View("bad", (1, 1, 2, 3, 4, 5, 6, 7, 8), False)
    with pytest.raises(ValueError, match="digit_permutation"):
        validate_views((bad,))


def test_view_validation_accepts_the_frozen_table_and_checks_both_line_axes() -> None:
    validate_views(NINE_VIEWS)
    bad_lines = (0, 1, 3, 2, 4, 5, 6, 7, 8)
    with pytest.raises(ValueError, match="row_permutation"):
        validate_views(
            (View("rows", IDENTITY_DIGITS, False, row_permutation=bad_lines),),
        )
    with pytest.raises(ValueError, match="col_permutation"):
        validate_views(
            (View("cols", IDENTITY_DIGITS, False, col_permutation=bad_lines),),
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
