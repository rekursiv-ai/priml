from __future__ import annotations

from torch import nn

import pytest
import torch

from priml.baselines.sudoku.losses import (
    CardinalityLoss,
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


def test_cardinality_term_reduces_over_the_slots_that_trained() -> None:
    """Padding rows and inactive slots add nothing; the mean is over active slots."""
    config = CardinalityLoss.Config()
    config.spec = SudokuSpec()
    loss = config.make()
    loss.to(torch.device("cpu"))
    labels = torch.stack(
        [sudoku_solution(), sudoku_solution(), torch.full((81,), -100)],
    )
    logits = torch.zeros(3, 81, 11)
    active = torch.tensor([True, False, True])
    reduced = loss(logits, labels, active=active, ignore_index=-100)
    # Slot 0 contributes 117/1210 (the closed form above); slot 1 is inactive and
    # slot 2 is all padding, so the sum divides by the two active slots.
    torch.testing.assert_close(reduced, torch.tensor(117 / 1210 / 2))
    assert loss.weight == 1.0
    assert loss.warmup_steps == 0


def test_cardinality_term_needs_a_consistent_sudoku_spec() -> None:
    with pytest.raises(ValueError, match="filled by its loop"):
        CardinalityLoss.Config().make()
    for spec in (
        SudokuSpec(grid_shape=(4, 6), box_shape=(2, 2), vocab_size=6),
        SudokuSpec(grid_shape=(4, 4), box_shape=(3, 3), vocab_size=6),
        SudokuSpec(grid_shape=(4, 4), box_shape=(2, 2), vocab_size=7),
    ):
        config = CardinalityLoss.Config()
        config.spec = spec
        with pytest.raises(ValueError, match="grid, boxes, and vocabulary"):
            config.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
