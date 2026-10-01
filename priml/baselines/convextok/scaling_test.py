"""Scaling must reproduce cuOpt's Stable3 preconditioning of an LP.

cuOpt equilibrates the constraint matrix with ten Ruiz passes and one Pock-Chambolle
pass, then divides the bounds and the objective by one plus their norms. PDLP runs
entirely in that scaled space, so every factor enters the solver's trajectory. These
tests pin the arithmetic on programs small enough to work by hand; the solver's
checkpoints against cuOpt (step size, iterates) check it at scale.
"""

import math

import torch

from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.scaling import bound_norm, scale_program


def test_diagonal_matrix_is_equilibrated_by_ruiz() -> None:
    # Ruiz divides row 0 and column 0 by sqrt(2), row 1 and column 1 by sqrt(8);
    # later passes and Pock-Chambolle see unit entries.
    scaled = scale_program(_program([[2.0, 0.0], [0.0, 8.0]]))
    expected = 1 / torch.tensor([2.0, 8.0], dtype=torch.float64).sqrt()
    torch.testing.assert_close(scaled.row_scale, expected, rtol=1e-15, atol=0)
    torch.testing.assert_close(scaled.column_scale, expected, rtol=1e-15, atol=0)
    torch.testing.assert_close(
        scaled.program.values,
        torch.ones(2, dtype=torch.float64),
        rtol=1e-15,
        atol=0,
    )


def test_empty_rows_and_columns_keep_unit_scale() -> None:
    scaled = scale_program(_program([[0.0, 0.0], [0.0, 4.0]]))
    assert scaled.row_scale[0].item() == 1.0
    assert scaled.column_scale[0].item() == 1.0


def test_pock_chambolle_divides_by_root_of_absolute_sums() -> None:
    # Every row and column of [[1, 1], [1, -1]] has max 1, so Ruiz leaves it; each row
    # and column sums to 2 in absolute value, so Pock-Chambolle divides by sqrt(2).
    scaled = scale_program(_program([[1.0, 1.0], [1.0, -1.0]]))
    assert torch.equal(
        scaled.row_scale,
        1 / torch.full((2,), 2.0, dtype=torch.float64).sqrt(),
    )
    assert torch.equal(
        scaled.column_scale,
        1 / torch.full((2,), 2.0, dtype=torch.float64).sqrt(),
    )


def test_transpose_holds_the_same_entries_column_major() -> None:
    scaled = scale_program(_program([[1.0, 2.0], [3.0, 0.0]]))
    assert scaled.transpose.crow_indices.tolist() == [0, 2, 3]
    assert scaled.transpose.col_indices.tolist() == [0, 1, 0]
    assert scaled.transpose_order.tolist() == [0, 2, 1]
    torch.testing.assert_close(
        scaled.transpose.values,
        scaled.program.values[scaled.transpose_order],
        rtol=1e-15,
        atol=0,
    )


def test_bounds_and_objective_are_divided_by_one_plus_their_norms() -> None:
    program = _program(
        [[1.0, 0.0], [0.0, 1.0]],
        row_lower=[3.0, -math.inf],
        row_upper=[3.0, 4.0],
        objective=[0.0, 12.0],
        upper=[2.0, 5.0],
    )
    scaled = scale_program(program)
    # A unit diagonal is left unscaled. An equality row counts its bound once and an
    # infinite bound not at all: sqrt(3^2 + 4^2) = 5, and sqrt(12^2) = 12.
    assert scaled.bound_rescaling == 1 / 6
    assert scaled.objective_rescaling == 1 / 13
    assert scaled.program.row_lower.tolist() == [3.0 * (1 / 6), -math.inf]
    assert scaled.program.row_upper.tolist() == [3.0 * (1 / 6), 4.0 * (1 / 6)]
    assert scaled.program.upper.tolist() == [2.0 * (1 / 6), 5.0 * (1 / 6)]
    assert scaled.program.objective.tolist() == [0.0, 12.0 * (1 / 13)]


def test_bound_norm_counts_each_finite_side_of_each_row_kind() -> None:
    # Equality 2 counts once; range [-3, 5] counts both sides; [4, inf) and
    # (-inf, 6] count their finite side: sqrt(4 + 9 + 25 + 16 + 36) = sqrt(90).
    row_lower = torch.tensor([2.0, -3.0, 4.0, -math.inf], dtype=torch.float64)
    row_upper = torch.tensor([2.0, 5.0, math.inf, 6.0], dtype=torch.float64)
    expected = torch.tensor(90.0, dtype=torch.float64).sqrt()
    torch.testing.assert_close(
        bound_norm(row_lower, row_upper),
        expected,
        rtol=0,
        atol=0,
    )


def _program(
    dense: list[list[float]],
    *,
    row_lower: list[float] | None = None,
    row_upper: list[float] | None = None,
    objective: list[float] | None = None,
    upper: list[float] | None = None,
) -> LinearProgram:
    matrix = torch.tensor(dense, dtype=torch.float64)
    rows, columns = matrix.shape
    entry_rows, entry_columns = torch.nonzero(matrix, as_tuple=True)
    counts = torch.bincount(entry_rows, minlength=rows)
    return LinearProgram(
        crow_indices=torch.cat(
            [torch.zeros(1, dtype=torch.int64), counts.cumsum(0)],
        ).to(
            torch.int32,
        ),
        col_indices=entry_columns.to(torch.int32),
        values=matrix[entry_rows, entry_columns],
        num_columns=columns,
        row_lower=torch.tensor(row_lower or [0.0] * rows, dtype=torch.float64),
        row_upper=torch.tensor(row_upper or [0.0] * rows, dtype=torch.float64),
        objective=torch.tensor(objective or [0.0] * columns, dtype=torch.float64),
        lower=torch.zeros(columns, dtype=torch.float64),
        upper=torch.tensor(upper or [1.0] * columns, dtype=torch.float64),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
