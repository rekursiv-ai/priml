from __future__ import annotations

from torch import nn

import pytest
import torch

from priml.baselines.sudoku.losses import (
    csp_cardinality_per_sample,
    token_cross_entropy,
)
from priml.baselines.sudoku.puzzle_spec import SudokuSpec, sudoku_group_indices
from priml.baselines.sudoku.testing import sudoku_solution


def test_token_cross_entropy_is_per_token_and_zero_where_ignored() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 4, -100], [0, 2, 3]], dtype=torch.int32)
    loss = token_cross_entropy(logits, labels, label_smoothing=0.1, ignore_index=-100)
    expected = nn.functional.cross_entropy(
        logits.reshape(6, 5),
        labels.reshape(6).long(),
        label_smoothing=0.1,
        ignore_index=-100,
        reduction="none",
    ).reshape(2, 3)
    assert loss.shape == (2, 3)
    assert torch.equal(loss, expected)
    assert loss[0, 2] == 0


def test_cardinality_loss_matches_its_closed_form_on_uniform_logits() -> None:
    """Uniform logits give every token mass 9/11 in every group.

    A solved row needs mass 1 on digits 2-10 and 0 on the blank, so each group
    contributes (9 * (2/11)^2 + (9/11)^2) / 10 = 117/1210. An all-ignore row has
    an all-zero target: (9/11)^2 = 81/121.
    """
    labels = torch.stack([sudoku_solution(), torch.full((81,), -100)])
    per_sample = csp_cardinality_per_sample(
        torch.zeros(2, 81, 11),
        labels,
        sudoku_group_indices(SudokuSpec()),
    )
    torch.testing.assert_close(per_sample, torch.tensor([117 / 1210, 81 / 121]))
    with pytest.raises(ValueError, match="prefix-stripped"):
        csp_cardinality_per_sample(
            torch.zeros(2, 80, 11),
            labels[:, :80],
            sudoku_group_indices(SudokuSpec()),
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
