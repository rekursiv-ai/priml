from __future__ import annotations

import pytest
import torch

from priml.baselines.sudoku.views import (
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
